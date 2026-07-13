namespace LivenessAPI.Domain;

public enum LivenessStatus
{
    AwaitingChallenge,
    ChallengesPassed,
    AntiSpoofPassed,
    Success,
    Failed,
    Expired,
}
