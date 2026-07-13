using AIModels.Detection;
using AIModels.Spoof;
using LivenessAPI.Application;
using LivenessAPI.Application.Abstractions;
using Microsoft.Extensions.Options;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace LivenessAPI.Infrastructure.PassiveLiveness;

public sealed class AntiSpoofService(AntiSpoofDetector detector, IOptions<LivenessOptions> options) : IAntiSpoofService
{
    private readonly LivenessOptions _options = options.Value;

    public AntiSpoofResult Predict(Image<Rgb24> image, DetectedFace face)
    {
        var result = detector.Predict(image, face, _options.AntiSpoofCropScale, _options.AntiSpoofRealClassIndex);
        bool isReal = result.IsReal && result.Confidence >= _options.AntiSpoofThreshold;
        return result with { IsReal = isReal };
    }
}
