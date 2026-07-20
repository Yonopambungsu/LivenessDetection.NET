using AIModels.Landmark;
using SixLabors.ImageSharp;

namespace LivenessAPI.Infrastructure.HeadPose;

/// <summary>
/// Cheap yaw proxy: horizontal offset of the nose tip from the midpoint between the outer eye
/// corners, normalized by inter-eye distance. Landmark "left"/"right" follow image coordinates
/// (see Landmark68Indices). Pass invertSign=true when the client mirrors frames before upload so
/// "turn left" on a selfie preview produces a positive delta from baseline.
/// </summary>
public static class HeadPoseEstimator
{
    public static float YawOffset(PointF[] landmarks, bool invertSign = false)
    {
        var leftEye  = landmarks[Landmark68Indices.LeftEyeLeftCorner];
        var rightEye = landmarks[Landmark68Indices.RightEyeRightCorner];
        var nose     = landmarks[Landmark68Indices.NoseTip];

        float eyeMidX = (leftEye.X + rightEye.X) / 2f;
        float eyeDistance = MathF.Max(MathF.Abs(rightEye.X - leftEye.X), 1e-3f);

        float yaw = (nose.X - eyeMidX) / eyeDistance;
        return invertSign ? -yaw : yaw;
    }

    public static bool IsNearNeutral(float yaw, float neutralBaseline, float tolerance) =>
        MathF.Abs(yaw - neutralBaseline) < tolerance;
}
