namespace LivenessAPI.Domain;

public enum LivenessStatus
{
    /// <summary>Initial phase: user positions face in the guide and holds a neutral pose.</summary>
    Calibrating,
    AwaitingChallenge,
    /// <summary>Brief pause between challenges while the user returns to a centered pose.</summary>
    ReturnToNeutral,
    ChallengesPassed,
    AntiSpoofPassed,
    Success,
    Failed,
    Expired,
}
