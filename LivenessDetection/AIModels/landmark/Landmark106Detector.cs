using AIModels.Detection;
using AIModels.Shared;
using Microsoft.ML.OnnxRuntime;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace AIModels.Landmark;

/// <summary>
/// Insightface 2D-106 landmark model (2d106det.onnx). Re-implements the reference `Landmark.get()`
/// preprocessing: crop centered on the detected bbox at scale = inputSize / (max(w,h)*1.5), no
/// rotation, run the model, decode the [-1,1]-ish output back to pixel space, then invert the crop
/// transform to map points back to original image coordinates. Input is fed raw (mean 0 / std 1),
/// R,G,B order.
/// </summary>
public sealed class Landmark106Detector : IDisposable
{
    private const int InputSize = 192;
    private const float InputMean = 0f;
    private const float InputStd = 1f;

    private readonly InferenceSession _session;
    private readonly string _inputName;

    public Landmark106Detector(string modelPath)
    {
        _session = new InferenceSession(modelPath);
        _inputName = _session.InputMetadata.Keys.First();
    }

    public PointF[] GetLandmarks(Image<Rgb24> image, DetectedFace face)
    {
        float w = face.Width, h = face.Height;
        var center = face.Center;
        float scale = InputSize / (Math.Max(w, h) * 1.5f);

        // Maps original-image point -> crop-local point: ix = (ox-cx)*scale + size/2
        var cropTransform = new SimilarityTransform(scale, 0f, -center.X * scale + InputSize / 2f, -center.Y * scale + InputSize / 2f);

        using var crop = cropTransform.WarpTo(image, InputSize, InputSize);
        var tensor = TensorUtils.ToNchwTensor(crop, InputMean, InputStd);
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(_inputName, tensor) };

        using var results = _session.Run(inputs);
        var raw = results.First().AsTensor<float>().ToArray();

        int pointCount = raw.Length / 2;
        var points = new PointF[pointCount];
        for (int i = 0; i < pointCount; i++)
        {
            float px = (raw[i * 2] + 1f) * (InputSize / 2f);
            float py = (raw[i * 2 + 1] + 1f) * (InputSize / 2f);
            points[i] = cropTransform.ApplyInverse(new PointF(px, py));
        }

        return points;
    }

    public void Dispose() => _session.Dispose();
}

/// <summary>
/// Index groups for the insightface 106-point scheme actually produced by 2d106det.onnx, confirmed
/// empirically (not just from documentation) by running the model on a real photo, rendering every
/// index on top of it, and reading off which numbers land on which feature: contour 0-32, left eye
/// 33-42, left eyebrow 43-51, mouth 52-71, nose 72-86, right eye 87-96, right eyebrow 97-105 — note
/// mouth comes numerically before nose in this export, and "left"/"right" here mean image-left/right
/// (not anatomical), matching how the rest of this codebase names things.
/// </summary>
public static class Landmark106Indices
{
    public const int NoseTip = 80;
    public const int LeftEyeLeftCorner = 35;
    public const int LeftEyeRightCorner = 39;
    public const int LeftEyeTop = 40;
    public const int LeftEyeBottom = 33;
    public const int RightEyeLeftCorner = 89;
    public const int RightEyeRightCorner = 93;
    public const int RightEyeTop = 94;
    public const int RightEyeBottom = 87;
    public const int MouthLeftCorner = 52;
    public const int MouthRightCorner = 61;
    public const int MouthTop = 67;
    public const int MouthBottom = 53;
}
