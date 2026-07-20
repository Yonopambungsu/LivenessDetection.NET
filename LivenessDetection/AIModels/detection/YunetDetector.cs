using AIModels.Shared;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace AIModels.Detection;

/// <summary>
/// YuNet face detector (face_detection_yunet_2026may.onnx). Re-implements the reference decode from
/// OpenCV's FaceDetectorYNImpl (modules/objdetect/src/face_detect.cpp), verified against that source
/// directly: 3 FPN strides (8/16/32), 1 anchor per grid cell (FCOS-style). cls/obj are already
/// probabilities in the exported graph (clamped to [0,1], NOT re-passed through sigmoid); final
/// confidence is their geometric mean sqrt(cls*obj). bbox uses center + exponential-size encoding
/// (cx=(gx+dx)*stride, cy=(gy+dy)*stride, w=exp(dw)*stride, h=exp(dh)*stride) - this is NOT the same
/// left/top/right/bottom distance encoding SCRFD uses, despite the superficially similar per-stride
/// output layout.
/// Input: BGR channel order, mean subtraction [104, 117, 123], no std division (values 0-255).
/// Output keypoints follow YuNet order: right-eye, left-eye, nose, mouth-right, mouth-left
/// (as seen in image, matching the SCRFD bnkps order used by ArcFaceRecognizer's alignment template).
/// </summary>
public sealed class YunetDetector : IDisposable
{
    private static readonly int[] Strides = { 8, 16, 32 };
    // YuNet 2026may uses 1 anchor per grid cell (FCOS-style, unlike SCRFD which uses 2).
    private const int NumAnchors = 1;

    // BGR mean subtraction (Caffe/OpenCV VGG-face style), std = 1.0 (no division).
    private static readonly float[] BgrMean = { 104f, 117f, 123f };

    private readonly InferenceSession _session;
    private readonly string _inputName;
    private bool _diagnosticLogged;

    public YunetDetector(string modelPath)
    {
        _session = new InferenceSession(modelPath);
        _inputName = _session.InputMetadata.Keys.First();
    }

    public List<DetectedFace> Detect(
        Image<Rgb24> image,
        int inputSize = 320,
        float scoreThreshold = 0.5f,
        float nmsThreshold = 0.4f)
    {
        // Resize + pad to inputSize × inputSize (top-left aligned, black padding).
        var (padded, scale) = TensorUtils.ResizeAndPad(image, inputSize);
        try
        {
            var tensor = BuildInputTensor(padded, inputSize);
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(_inputName, tensor)
            };

            using var results = _session.Run(inputs);

            // Index by name — YuNet 2026may outputs: cls_8, obj_8, bbox_8, kps_8,
            // cls_16, obj_16, bbox_16, kps_16, cls_32, obj_32, bbox_32, kps_32.
            var outputMap = results.ToDictionary(
                r => r.Name,
                r => r.AsTensor<float>().ToArray());

            // Diagnostic (first call only): log actual output tensor names + lengths so
            // any future layout change is immediately visible in the console.
            if (!_diagnosticLogged)
            {
                _diagnosticLogged = true;
                Console.WriteLine("[YunetDetector] Output tensors:");
                foreach (var kv in outputMap)
                    Console.WriteLine($"  {kv.Key}: length={kv.Value.Length}");
            }

            var candidates = new List<(float[] Box, float Score, PointF[] Kps)>();

            foreach (int stride in Strides)
            {
                string suffix = stride.ToString();

                // Fall back gracefully: if named outputs are unavailable, this will throw,
                // making misconfiguration obvious at startup.
                var cls  = outputMap[$"cls_{suffix}"];
                var obj  = outputMap[$"obj_{suffix}"];
                var bbox = outputMap[$"bbox_{suffix}"];
                var kps  = outputMap[$"kps_{suffix}"];

                int gridH = inputSize / stride;
                int gridW = inputSize / stride;

                int anchorIdx = 0;
                for (int gy = 0; gy < gridH; gy++)
                {
                    for (int gx = 0; gx < gridW; gx++)
                    {
                        for (int a = 0; a < NumAnchors; a++, anchorIdx++)
                        {
                            // cls/obj are already probabilities in this export (the ONNX graph itself
                            // ends in Sigmoid) - OpenCV's reference decode (face_detect.cpp) only clamps
                            // them to [0,1] and never re-applies sigmoid. Final score is their geometric
                            // mean, not a plain product.
                            float clsScore = Math.Clamp(cls[anchorIdx], 0f, 1f);
                            float objScore = Math.Clamp(obj[anchorIdx], 0f, 1f);
                            float score    = MathF.Sqrt(clsScore * objScore);

                            if (score < scoreThreshold) continue;

                            // Center + exponential-size encoding (matches OpenCV's decode exactly -
                            // this is NOT SCRFD-style left/top/right/bottom distance encoding).
                            float cx = (gx + bbox[anchorIdx * 4 + 0]) * stride;
                            float cy = (gy + bbox[anchorIdx * 4 + 1]) * stride;
                            float w  = MathF.Exp(bbox[anchorIdx * 4 + 2]) * stride;
                            float h  = MathF.Exp(bbox[anchorIdx * 4 + 3]) * stride;

                            float x1 = cx - w / 2f;
                            float y1 = cy - h / 2f;
                            float x2 = x1 + w;
                            float y2 = y1 + h;

                            var kpPoints = new PointF[5];
                            for (int k = 0; k < 5; k++)
                            {
                                float kx = (gx + kps[anchorIdx * 10 + k * 2])     * stride;
                                float ky = (gy + kps[anchorIdx * 10 + k * 2 + 1]) * stride;
                                kpPoints[k] = new PointF(kx, ky);
                            }

                            candidates.Add((new[] { x1, y1, x2, y2 }, score, kpPoints));
                        }
                    }
                }
            }

            var kept = Nms(candidates, nmsThreshold);

            // Divide back by scale to get original-image coordinates.
            return kept.Select(c => new DetectedFace
            {
                X1 = c.Box[0] / scale,
                Y1 = c.Box[1] / scale,
                X2 = c.Box[2] / scale,
                Y2 = c.Box[3] / scale,
                Score = c.Score,
                Keypoints = c.Kps.Select(p => new PointF(p.X / scale, p.Y / scale)).ToArray(),
            }).ToList();
        }
        finally
        {
            padded.Dispose();
        }
    }

    // ----- helpers -----

    /// <summary>
    /// Builds a 1×3×H×W tensor in BGR channel order with mean [104,117,123] subtracted.
    /// ImageSharp stores pixels as R,G,B so we swap channels here.
    /// </summary>
    private static DenseTensor<float> BuildInputTensor(Image<Rgb24> image, int inputSize)
    {
        var tensor = new DenseTensor<float>(new[] { 1, 3, inputSize, inputSize });
        image.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < inputSize; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < inputSize; x++)
                {
                    var px = row[x];
                    // Channel 0 = B, Channel 1 = G, Channel 2 = R (BGR order).
                    tensor[0, 0, y, x] = px.B - BgrMean[0];
                    tensor[0, 1, y, x] = px.G - BgrMean[1];
                    tensor[0, 2, y, x] = px.R - BgrMean[2];
                }
            }
        });
        return tensor;
    }

    private static List<(float[] Box, float Score, PointF[] Kps)> Nms(
        List<(float[] Box, float Score, PointF[] Kps)> candidates,
        float iouThreshold)
    {
        var sorted = candidates.OrderByDescending(c => c.Score).ToList();
        var kept   = new List<(float[] Box, float Score, PointF[] Kps)>();

        while (sorted.Count > 0)
        {
            var best = sorted[0];
            kept.Add(best);
            sorted.RemoveAt(0);
            sorted.RemoveAll(c => Iou(best.Box, c.Box) > iouThreshold);
        }

        return kept;
    }

    private static float Iou(float[] a, float[] b)
    {
        float x1 = Math.Max(a[0], b[0]);
        float y1 = Math.Max(a[1], b[1]);
        float x2 = Math.Min(a[2], b[2]);
        float y2 = Math.Min(a[3], b[3]);

        float interW = Math.Max(0, x2 - x1);
        float interH = Math.Max(0, y2 - y1);
        float inter  = interW * interH;

        float areaA = (a[2] - a[0]) * (a[3] - a[1]);
        float areaB = (b[2] - b[0]) * (b[3] - b[1]);
        float union = areaA + areaB - inter;

        return union <= 0 ? 0 : inter / union;
    }

    public void Dispose() => _session.Dispose();
}
