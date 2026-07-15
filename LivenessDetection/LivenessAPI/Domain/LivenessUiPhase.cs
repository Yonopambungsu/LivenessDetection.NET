namespace LivenessAPI.Domain;

/// <summary>Client-facing phase labels for overlay text and progress UI. Decoupled from the internal
/// status machine so the demo can show friendly instructions without parsing server messages.</summary>
public static class LivenessUiPhase
{
    public const string Calibrating = "Calibrating";
    public const string Challenge = "Challenge";
    public const string ChallengeRetry = "ChallengeRetry";
    public const string ReturnToNeutral = "ReturnToNeutral";
    public const string Verifying = "Verifying";
    public const string Complete = "Complete";
    public const string Failed = "Failed";
}
