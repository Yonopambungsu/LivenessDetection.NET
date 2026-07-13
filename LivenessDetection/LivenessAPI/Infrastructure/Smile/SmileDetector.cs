using AIModels.Landmark;
using SixLabors.ImageSharp;

namespace LivenessAPI.Infrastructure.Smile;

/// <summary>
/// Mouth width-to-height ratio: widens noticeably during a smile. Normalized so it can be compared
/// against a fixed threshold regardless of face distance from the camera (uses eye distance as the
/// scale reference rather than an absolute pixel count).
/// </summary>
public static class SmileDetector
{
    public static float MouthRatio(PointF[] landmarks)
    {
        var leftCorner = landmarks[Landmark106Indices.MouthLeftCorner];
        var rightCorner = landmarks[Landmark106Indices.MouthRightCorner];
        var top = landmarks[Landmark106Indices.MouthTop];
        var bottom = landmarks[Landmark106Indices.MouthBottom];
        var leftEye = landmarks[Landmark106Indices.LeftEyeLeftCorner];
        var rightEye = landmarks[Landmark106Indices.RightEyeRightCorner];

        float mouthWidth = Distance(leftCorner, rightCorner);
        float mouthHeight = MathF.Max(Distance(top, bottom), 1e-3f);
        float eyeDistance = MathF.Max(Distance(leftEye, rightEye), 1e-3f);

        // Wider, flatter mouth (relative to face scale) => higher ratio while smiling.
        return (mouthWidth / eyeDistance) / (mouthHeight / eyeDistance + 0.15f);
    }

    private static float Distance(PointF a, PointF b) => MathF.Sqrt(MathF.Pow(a.X - b.X, 2) + MathF.Pow(a.Y - b.Y, 2));
}
