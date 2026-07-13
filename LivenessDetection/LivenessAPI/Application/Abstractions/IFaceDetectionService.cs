using AIModels.Detection;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace LivenessAPI.Application.Abstractions;

public sealed record SingleFaceResult(DetectedFace? Face, string? Error);

public interface IFaceDetectionService
{
    SingleFaceResult DetectSingleFace(Image<Rgb24> image);
}
