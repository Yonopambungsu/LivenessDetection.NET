using AIModels.Shared;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace AIModels.Detection;

/// <summary>
/// SCRFD face detector (scrfd_10g_bnkps.onnx). Faithful re-implementation of insightface's
/// scrfd.py decode: 3 FPN strides (8/16/32), 2 anchors per location, distance-encoded
/// bbox/keypoint regression, greedy NMS. Input is BGR-trained but exported with swapRB=True
/// (i.e. fed as R,G,B), mean 127.5 / std 128.
/// </summary>
public sealed class ScrfdDetector : IDisposable
{
    private static readonly int[] Strides = { 8, 16, 32 };
    private const int NumAnchors = 2;
    private const float InputMean = 127.5f;
    private const float InputStd = 128f;

    private readonly InferenceSession _session;
    private readonly string _inputName;

    public ScrfdDetector(string modelPath)
    {
        _session = new InferenceSession(modelPath);
        _inputName = _session.InputMetadata.Keys.First();
    }

    public List<DetectedFace> Detect(Image<Rgb24> image, int inputSize = 320, float scoreThreshold = 0.5f, float nmsThreshold = 0.4f)
    {
        var (padded, scale) = TensorUtils.ResizeAndPad(image, inputSize);
        try
        {
            var tensor = TensorUtils.ToNchwTensor(padded, InputMean, InputStd);
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(_inputName, tensor) };

            using var results = _session.Run(inputs);
            // Flatten row-major regardless of whether the graph exports [N,c] or [1,N,c] shapes.
            var outputs = results.Select(r => r.AsTensor<float>().ToArray()).ToArray();

            var candidates = new List<(float[] Box, float Score, PointF[] Kps)>();

            for (int strideIdx = 0; strideIdx < Strides.Length; strideIdx++)
            {
                int stride = Strides[strideIdx];
                var scores = outputs[strideIdx];
                var bboxPreds = outputs[Strides.Length + strideIdx];
                var kpsPreds = outputs[Strides.Length * 2 + strideIdx];

                int gridH = inputSize / stride;
                int gridW = inputSize / stride;

                int anchorIdx = 0;
                for (int gy = 0; gy < gridH; gy++)
                {
                    for (int gx = 0; gx < gridW; gx++)
                    {
                        float cx = gx * stride;
                        float cy = gy * stride;

                        for (int a = 0; a < NumAnchors; a++)
                        {
                            float score = scores[anchorIdx];
                            if (score >= scoreThreshold)
                            {
                                float dx1 = bboxPreds[anchorIdx * 4 + 0] * stride;
                                float dy1 = bboxPreds[anchorIdx * 4 + 1] * stride;
                                float dx2 = bboxPreds[anchorIdx * 4 + 2] * stride;
                                float dy2 = bboxPreds[anchorIdx * 4 + 3] * stride;

                                float x1 = cx - dx1, y1 = cy - dy1, x2 = cx + dx2, y2 = cy + dy2;

                                var kps = new PointF[5];
                                for (int k = 0; k < 5; k++)
                                {
                                    float kx = cx + kpsPreds[anchorIdx * 10 + k * 2] * stride;
                                    float ky = cy + kpsPreds[anchorIdx * 10 + k * 2 + 1] * stride;
                                    kps[k] = new PointF(kx, ky);
                                }

                                candidates.Add((new[] { x1, y1, x2, y2 }, score, kps));
                            }

                            anchorIdx++;
                        }
                    }
                }
            }

            var kept = Nms(candidates, nmsThreshold);

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

    private static List<(float[] Box, float Score, PointF[] Kps)> Nms(
        List<(float[] Box, float Score, PointF[] Kps)> candidates, float iouThreshold)
    {
        var sorted = candidates.OrderByDescending(c => c.Score).ToList();
        var kept = new List<(float[] Box, float Score, PointF[] Kps)>();

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
        float inter = interW * interH;

        float areaA = (a[2] - a[0]) * (a[3] - a[1]);
        float areaB = (b[2] - b[0]) * (b[3] - b[1]);

        float union = areaA + areaB - inter;
        return union <= 0 ? 0 : inter / union;
    }

    public void Dispose() => _session.Dispose();
}
