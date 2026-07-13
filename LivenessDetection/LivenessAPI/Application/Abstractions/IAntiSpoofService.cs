using AIModels.Detection;
using AIModels.Spoof;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace LivenessAPI.Application.Abstractions;

public interface IAntiSpoofService
{
    AntiSpoofResult Predict(Image<Rgb24> image, DetectedFace face);
}
