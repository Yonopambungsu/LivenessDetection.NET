namespace LivenessAPI.Application;

/// <summary>Bound from the "Liveness" section of appsettings.json. Every threshold called out as
/// needing empirical tuning in the implementation plan lives here so it can be adjusted without
/// recompiling.</summary>
public sealed class LivenessOptions
{
    public const string SectionName = "Liveness";

    public string ScrfdModelPath { get; set; } = "../detection/scrfd_10g_bnkps.onnx";
    public string LandmarkModelPath { get; set; } = "../landmark/2d106det.onnx";
    public string RecognitionModelPath { get; set; } = "../recognition/glintr100.onnx";
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

    /// <summary>Eyes count as "closed" once EAR drops below this fraction of the session's observed-open baseline.</summary>
    public float BlinkClosedRatio { get; set; } = 0.75f;
    /// <summary>After a dip, eyes count as "reopened" once EAR climbs back above this fraction of the baseline.</summary>
    public float BlinkRecoverRatio { get; set; } = 0.85f;
    public float SmileRatioThreshold { get; set; } = 0.38f;
    public float YawOffsetThreshold { get; set; } = 0.18f;

    public int SessionTtlSeconds { get; set; } = 120;

    public int ChallengeCount { get; set; } = 2;

    public int RequiredConsecutiveFrames { get; set; } = 2;
}
