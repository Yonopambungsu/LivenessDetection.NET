using AIModels.Landmark;
using SixLabors.ImageSharp;

namespace LivenessAPI.Infrastructure.HeadPose;

/// <summary>
/// Cheap yaw proxy (no dedicated head-pose model available): horizontal offset of the nose tip from
/// the midpoint between the outer eye corners, normalized by inter-eye distance so it's roughly
/// distance-invariant. Sign convention (which side is "positive") depends on camera mirroring, which
/// varies by client (a mirrored front-camera preview vs. the raw un-mirrored frame sent to the API
/// can flip it) — TODO: verify against a real turn-left/turn-right capture and adjust
/// LivenessSessionService's LookLeft/LookRight sign mapping if it comes out backwards.
/// </summary>
public static class HeadPoseEstimator
{
    public static float YawOffset(PointF[] landmarks)
    {
        var leftEye = landmarks[Landmark106Indices.LeftEyeLeftCorner];
        var rightEye = landmarks[Landmark106Indices.RightEyeRightCorner];
        var nose = landmarks[Landmark106Indices.NoseTip];

        float eyeMidX = (leftEye.X + rightEye.X) / 2f;
        float eyeDistance = MathF.Max(MathF.Abs(rightEye.X - leftEye.X), 1e-3f);

        return (nose.X - eyeMidX) / eyeDistance;
    }
}
