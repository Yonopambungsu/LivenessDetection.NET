using AIModels.Detection;
using AIModels.Recognition;
using LivenessAPI.Application.Abstractions;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace LivenessAPI.Infrastructure.FaceRecognition;

public sealed class FaceRecognitionServiceImpl(ArcFaceRecognizer recognizer) : IFaceRecognitionService
{
    public float[] GetEmbedding(Image<Rgb24> image, DetectedFace face) => recognizer.GetEmbedding(image, face);

    public float CosineSimilarity(float[] a, float[] b) => ArcFaceRecognizer.CosineSimilarity(a, b);
}
