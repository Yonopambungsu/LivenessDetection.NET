using AIModels.Landmark;
using SixLabors.ImageSharp;

namespace LivenessAPI.Infrastructure.Blink;

/// <summary>
/// Eye-aspect-ratio (EAR) style openness metric: vertical eyelid gap over horizontal eye width,
/// averaged across both eyes. Drops sharply while the eyes are closed. Uses the 4 named corner/
/// top/bottom indices from <see cref="Landmark68Indices"/> rather than the full per-eye point set.
/// </summary>
public static class BlinkDetector
{
    public static float EyeAspectRatio(PointF[] landmarks)
    {
        float leftEar = SingleEyeRatio(
            landmarks[Landmark68Indices.LeftEyeLeftCorner],
            landmarks[Landmark68Indices.LeftEyeRightCorner],
            landmarks[Landmark68Indices.LeftEyeTop],
            landmarks[Landmark68Indices.LeftEyeBottom]);

        float rightEar = SingleEyeRatio(
            landmarks[Landmark68Indices.RightEyeLeftCorner],
            landmarks[Landmark68Indices.RightEyeRightCorner],
            landmarks[Landmark68Indices.RightEyeTop],
            landmarks[Landmark68Indices.RightEyeBottom]);

        return (leftEar + rightEar) / 2f;
    }

    private static float SingleEyeRatio(PointF left, PointF right, PointF top, PointF bottom)
    {
        float horizontal = Distance(left, right);
        if (horizontal < 1e-3f) return 0f;
        float vertical = Distance(top, bottom);
        return vertical / horizontal;
    }

    private static float Distance(PointF a, PointF b) => MathF.Sqrt(MathF.Pow(a.X - b.X, 2) + MathF.Pow(a.Y - b.Y, 2));
}
