namespace LivenessAPI.Application;

/// <summary>Bound from the "Liveness" section of appsettings.json. Every threshold called out as
/// needing empirical tuning in the implementation plan lives here so it can be adjusted without
/// recompiling.</summary>
public sealed class LivenessOptions
{
    public const string SectionName = "Liveness";

    public string YunetModelPath { get; set; } = "../detection/face_detection_yunet.onnx";
    public string LandmarkModelPath { get; set; } = "../landmark/pipnet_r18_300w_celeba_68.onnx";
    public string RecognitionModelPath { get; set; } = "../recognition/glintr100-aura-face.onnx";
    public string AntiSpoofModelPath { get; set; } = "../spoof/MiniFASNetV2.onnx";

    public int DetectionInputSize { get; set; } = 320;
    public float DetectionScoreThreshold { get; set; } = 0.5f;
    public float DetectionNmsThreshold { get; set; } = 0.4f;

    /// <summary>Set to false to skip the anti-spoofing step entirely (session goes straight from
    /// ChallengesPassed to AntiSpoofPassed). Escape hatch for when the anti-spoof model itself is
    /// unusable — e.g. this project's stock MiniFASNetV2.onnx was found to return ~99% on the same
    /// class for every input (noise, solid black, real photos alike) across every crop scale, channel
    /// order, and normalization scheme tried, indicating a broken/degenerate export rather than a
    /// preprocessing mismatch. Re-enable once a verified-working anti-spoof model is in place.</summary>
    public bool AntiSpoofEnabled { get; set; } = true;
    public float AntiSpoofCropScale { get; set; } = 2.7f;
    public int AntiSpoofRealClassIndex { get; set; } = 1;
    public float AntiSpoofThreshold { get; set; } = 0.6f;

    public float RecognitionThreshold { get; set; } = 0.35f;

    /// <summary>When true, challenges run in FixedChallengeOrder (blink → look left → look right → smile).
    /// When false, a random subset is chosen (legacy behaviour).</summary>
    public bool UseFixedChallengeOrder { get; set; } = true;

    /// <summary>Ordered challenge types used when UseFixedChallengeOrder is true. Only the first
    /// ChallengeCount entries are used.</summary>
    public string[] FixedChallengeOrder { get; set; } = ["Blink", "LookLeft", "LookRight", "Smile"];

    /// <summary>How many retries the user gets per challenge after a wrong gesture or timeout.</summary>
    public int MaxRetriesPerChallenge { get; set; } = 1;

    /// <summary>Seconds the user must hold a good centered pose during initial calibration.</summary>
    public double CalibrationSeconds { get; set; } = 1.0;

    /// <summary>Minimum seconds to show the between-challenge return-to-neutral prompt.</summary>
    public double TransitionMinSeconds { get; set; } = 0.4;

    /// <summary>Maximum seconds to wait for return-to-neutral before advancing anyway.</summary>
    public double TransitionMaxSeconds { get; set; } = 3.0;

    /// <summary>Face bounding-box area as a fraction of the frame; below this the user is too far.</summary>
    public float MinFaceAreaRatio { get; set; } = 0.06f;

    /// <summary>Face bounding-box area as a fraction of the frame; above this the user is too close.</summary>
    public float MaxFaceAreaRatio { get; set; } = 0.45f;

    /// <summary>Horizontal center tolerance as a fraction of frame width during calibration.</summary>
    public float FaceCenterToleranceRatio { get; set; } = 0.22f;

    /// <summary>Include raw metric values in API messages (useful for tuning; off in production).</summary>
    public bool IncludeDebugMetrics { get; set; } = false;

    /// <summary>Eyes count as "closed" once EAR drops below this fraction of the session's observed-open baseline.</summary>
    public float BlinkClosedRatio { get; set; } = 0.75f;
    /// <summary>After a dip, eyes count as "reopened" once EAR climbs back above this fraction of the baseline.</summary>
    public float BlinkRecoverRatio { get; set; } = 0.85f;

    /// <summary>Consecutive reopened-eye frames required to confirm a blink (lower than smile because
    /// a blink is transient).</summary>
    public int BlinkRequiredConsecutiveFrames { get; set; } = 2;

    /// <summary>Minimum mouth-ratio increase above the user's personal resting baseline to count as smiling.</summary>
    public float SmileDeltaThreshold { get; set; } = 0.06f;

    /// <summary>Frames spent observing the resting mouth before the smile challenge starts counting.</summary>
    public int SmileCalibrationFrames { get; set; } = 3;

    /// <summary>Consecutive frames required to confirm a smile hold.</summary>
    public int SmileRequiredConsecutiveFrames { get; set; } = 2;

    /// <summary>Legacy absolute threshold kept for reference; smile now uses resting baseline + SmileDeltaThreshold.</summary>
    public float SmileRatioThreshold { get; set; } = 0.38f;
    public float YawOffsetThreshold { get; set; } = 0.10f;
    /// <summary>Negate raw yaw so "look left" on a mirrored selfie feed maps to positive delta.
    /// Must be true when the client horizontally flips frames before upload (standard selfie UX).</summary>
    public bool InvertYawSign { get; set; } = true;

    /// <summary>Consecutive frames required to confirm a head turn (lower than blink/smile because
    /// users rarely hold a turn perfectly still).</summary>
    public int YawRequiredConsecutiveFrames { get; set; } = 2;

    /// <summary>A LookLeft/LookRight challenge only starts counting hold-frames once the yaw has been seen
    /// within this distance of center first. Without this, a head turn left over from the previous challenge
    /// (e.g. a wrong-direction turn that just got auto-advanced past) could instantly satisfy the next
    /// challenge in a couple of frames, which feels like the check barely ran at all. Kept meaningfully below
    /// YawOffsetThreshold (so a genuinely turned leftover pose still can't pass as neutral) but generous
    /// enough that everyday resting-face variance (camera angle, facial asymmetry) reaches it quickly -
    /// too tight a value here reads as the challenge never starting.</summary>
    public float YawNeutralThreshold { get; set; } = 0.10f;

    public int SessionTtlSeconds { get; set; } = 120;

    public int ChallengeCount { get; set; } = 3;

    public int RequiredConsecutiveFrames { get; set; } = 3;

    /// <summary>Max wall-clock seconds a single challenge gets before it's given up on and the session moves
    /// to the next challenge in the queue (same outcome as a confirmed wrong gesture). Without this, Blink
    /// and Smile - which have no "wrong gesture" concept, unlike LookLeft/LookRight - would wait indefinitely
    /// (up to SessionTtlSeconds) if the action is never detected, which is especially noticeable when one of
    /// them lands as the last challenge in the queue. Deliberately measured in seconds rather than a frame
    /// count so this budget doesn't silently shrink or grow whenever a client changes how often it submits
    /// frames - the actual pass/fail decision itself is never time-based, only this give-up fallback is.</summary>
    public double MaxSecondsPerChallenge { get; set; } = 15;
}
