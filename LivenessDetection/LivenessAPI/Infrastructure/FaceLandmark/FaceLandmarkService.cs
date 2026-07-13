using AIModels.Detection;
using AIModels.Landmark;
using LivenessAPI.Application.Abstractions;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace LivenessAPI.Infrastructure.FaceLandmark;

public sealed class FaceLandmarkService(Landmark106Detector detector) : IFaceLandmarkService
{
    public PointF[] GetLandmarks(Image<Rgb24> image, DetectedFace face) => detector.GetLandmarks(image, face);
}
