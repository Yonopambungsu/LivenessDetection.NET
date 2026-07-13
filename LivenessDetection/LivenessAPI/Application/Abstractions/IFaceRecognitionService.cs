using AIModels.Detection;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace LivenessAPI.Application.Abstractions;

public interface IFaceRecognitionService
{
    float[] GetEmbedding(Image<Rgb24> image, DetectedFace face);
    float CosineSimilarity(float[] a, float[] b);
}
