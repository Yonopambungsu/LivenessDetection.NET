using AIModels.Detection;
using LivenessAPI.Application;
using LivenessAPI.Application.Abstractions;
using Microsoft.Extensions.Options;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace LivenessAPI.Infrastructure.FaceDetection;

public sealed class FaceDetectionService(ScrfdDetector detector, IOptions<LivenessOptions> options) : IFaceDetectionService
{
    private readonly LivenessOptions _options = options.Value;

    public SingleFaceResult DetectSingleFace(Image<Rgb24> image)
    {
        var faces = detector.Detect(image, _options.DetectionInputSize, _options.DetectionScoreThreshold, _options.DetectionNmsThreshold);

        return faces.Count switch
        {
            0 => new SingleFaceResult(null, "No face detected. Make sure your face is clearly visible."),
            > 1 => new SingleFaceResult(null, "Multiple faces detected. Only one person should be in frame."),
            _ => new SingleFaceResult(faces[0], null),
        };
    }
}
