using AIModels.Detection;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace LivenessAPI.Application.Abstractions;

public interface IFaceLandmarkService
{
    PointF[] GetLandmarks(Image<Rgb24> image, DetectedFace face);
}
