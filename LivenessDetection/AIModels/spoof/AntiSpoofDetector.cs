using AIModels.Detection;
using AIModels.Shared;
using Microsoft.ML.OnnxRuntime;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace AIModels.Spoof;

public sealed record AntiSpoofResult(bool IsReal, float Confidence, float[] Probabilities);

/// <summary>
/// Silent-Face-Anti-Spoofing MiniFASNetV2. Re-implements the reference CropImage.crop (expand the
/// face bbox by a configurable scale around its center, clamped to image bounds) then resizes to
/// 80x80. Confirmed against yakhyo/face-anti-spoofing's onnx_inference.py: raw BGR pixel values are
/// fed as plain floats with NO /255 scaling and no mean subtraction at all (just `astype(np.float32)`)
/// — feeding normalized 0-1 values here previously (as this file used to) starved the network of
/// signal and made it saturate to one class regardless of input. Real class index is 1 (else fake).
/// </summary>
public sealed class AntiSpoofDetector : IDisposable
{
    private const int InputSize = 80;

    private readonly InferenceSession _session;
    private readonly string _inputName;

    public AntiSpoofDetector(string modelPath)
    {
        _session = new InferenceSession(modelPath);
        _inputName = _session.InputMetadata.Keys.First();
    }

    public string DescribeModel()
    {
        var inputs = _session.InputMetadata.Select(kv => $"{kv.Key}: [{string.Join(",", kv.Value.Dimensions)}] {kv.Value.ElementDataType}");
        var outputs = _session.OutputMetadata.Select(kv => $"{kv.Key}: [{string.Join(",", kv.Value.Dimensions)}] {kv.Value.ElementDataType}");
        return $"Inputs: {string.Join(" | ", inputs)}; Outputs: {string.Join(" | ", outputs)}";
    }

    public AntiSpoofResult Predict(Image<Rgb24> image, DetectedFace face, float cropScale = 2.7f, int realClassIndex = 1, bool swapToBgr = true)
    {
        var (x0, y0, x1, y1) = GetExpandedBox(image.Width, image.Height, face.X1, face.Y1, face.Width, face.Height, cropScale);

        using var crop = TensorUtils.CropWithPadding(image, x0, y0, x1, y1);
        using var resized = crop.Clone(ctx => ctx.Resize(InputSize, InputSize));

        var tensor = TensorUtils.ToNchwTensor(resized, mean: 0f, std: 1f, swapToBgr: swapToBgr);
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(_inputName, tensor) };

        using var results = _session.Run(inputs);
        var logits = results.First().AsTensor<float>().ToArray();
        var probs = Softmax(logits);

        int predicted = 0;
        for (int i = 1; i < probs.Length; i++)
        {
            if (probs[i] > probs[predicted]) predicted = i;
        }

        return new AntiSpoofResult(predicted == realClassIndex, probs[realClassIndex], probs);
    }

    /// <summary>Calibration-only variant: lets the debug endpoint try alternate normalization
    /// schemes (e.g. ImageNet mean/std) against the same crop, since the reference repo's "/255,
    /// no mean subtraction" convention is unconfirmed for this specific onnx export.</summary>
    public float[] PredictRawProbabilities(Image<Rgb24> image, DetectedFace face, float cropScale, bool swapToBgr, float[]? perChannelMean, float[]? perChannelStd)
    {
        var (x0, y0, x1, y1) = GetExpandedBox(image.Width, image.Height, face.X1, face.Y1, face.Width, face.Height, cropScale);
        using var crop = TensorUtils.CropWithPadding(image, x0, y0, x1, y1);
        using var resized = crop.Clone(ctx => ctx.Resize(InputSize, InputSize));

        var tensor = perChannelMean is not null && perChannelStd is not null
            ? TensorUtils.ToNchwTensorPerChannel(resized, perChannelMean, perChannelStd, swapToBgr)
            : TensorUtils.ToNchwTensor(resized, mean: 0f, std: 1f, swapToBgr: swapToBgr);

        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(_inputName, tensor) };
        using var results = _session.Run(inputs);
        return Softmax(results.First().AsTensor<float>().ToArray());
    }

    /// <summary>Port of Silent-Face-Anti-Spoofing's CropImage._get_new_box: expand bbox around its
    /// center by <paramref name="scale"/>, clamped so the crop stays inside the source image.</summary>
    private static (int x0, int y0, int x1, int y1) GetExpandedBox(int srcW, int srcH, float x, float y, float boxW, float boxH, float scale)
    {
        scale = Math.Min((srcH - 1) / boxH, Math.Min((srcW - 1) / boxW, scale));

        float newW = boxW * scale;
        float newH = boxH * scale;
        float centerX = boxW / 2f + x;
        float centerY = boxH / 2f + y;

        float left = centerX - newW / 2f;
        float top = centerY - newH / 2f;
        float right = centerX + newW / 2f;
        float bottom = centerY + newH / 2f;

        if (left < 0) { right -= left; left = 0; }
        if (top < 0) { bottom -= top; top = 0; }
        if (right > srcW - 1) { left -= right - (srcW - 1); right = srcW - 1; }
        if (bottom > srcH - 1) { top -= bottom - (srcH - 1); bottom = srcH - 1; }

        return ((int)left, (int)top, (int)right, (int)bottom);
    }

    private static float[] Softmax(float[] logits)
    {
        float max = logits.Max();
        var exps = logits.Select(l => MathF.Exp(l - max)).ToArray();
        float sum = exps.Sum();
        return exps.Select(e => e / sum).ToArray();
    }

    public void Dispose() => _session.Dispose();
}
