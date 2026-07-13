using AIModels.Detection;
using AIModels.Shared;
using Microsoft.ML.OnnxRuntime;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace AIModels.Recognition;

/// <summary>
/// ArcFace / GlintR100 face recognition (glintr100.onnx). Aligns the face to the canonical
/// insightface 112x112 5-point template using a least-squares similarity transform fit from the
/// SCRFD keypoints, then runs the model to get a 512-d embedding. Input mean/std 127.5, R,G,B order.
/// </summary>
public sealed class ArcFaceRecognizer : IDisposable
{
    private const int InputSize = 112;
    private const float InputMean = 127.5f;
    private const float InputStd = 127.5f;

    // Standard insightface arcface_dst template for 112x112 output, order: left eye, right eye,
    // nose, mouth-left, mouth-right (matches SCRFD's bnkps keypoint order).
    private static readonly PointF[] ArcfaceTemplate =
    {
        new(38.2946f, 51.6963f),
        new(73.5318f, 51.5014f),
        new(56.0252f, 71.7366f),
        new(41.5493f, 92.3655f),
        new(70.7299f, 92.2041f),
    };

    private readonly InferenceSession _session;
    private readonly string _inputName;

    public ArcFaceRecognizer(string modelPath)
    {
        _session = new InferenceSession(modelPath);
        _inputName = _session.InputMetadata.Keys.First();
    }

    public float[] GetEmbedding(Image<Rgb24> image, DetectedFace face)
    {
        var transform = SimilarityTransform.Estimate(face.Keypoints, ArcfaceTemplate);
        using var aligned = transform.WarpTo(image, InputSize, InputSize);

        var tensor = TensorUtils.ToNchwTensor(aligned, InputMean, InputStd);
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(_inputName, tensor) };

        using var results = _session.Run(inputs);
        var embedding = results.First().AsTensor<float>().ToArray();

        return L2Normalize(embedding);
    }

    /// <summary>Assumes both embeddings are already L2-normalized (as returned by <see cref="GetEmbedding"/>), so this is a plain dot product.</summary>
    public static float CosineSimilarity(float[] a, float[] b)
    {
        if (a.Length != b.Length) throw new ArgumentException("Embedding length mismatch.");

        float dot = 0;
        for (int i = 0; i < a.Length; i++) dot += a[i] * b[i];
        return dot;
    }

    private static float[] L2Normalize(float[] v)
    {
        double sumSq = 0;
        foreach (var x in v) sumSq += x * x;
        double norm = Math.Sqrt(sumSq);
        if (norm < 1e-12) return v;

        var result = new float[v.Length];
        for (int i = 0; i < v.Length; i++) result[i] = (float)(v[i] / norm);
        return result;
    }

    public void Dispose() => _session.Dispose();
}
