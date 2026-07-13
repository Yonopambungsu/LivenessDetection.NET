namespace LivenessAPI.Domain;

/// <summary>
/// Server-side state for one liveness attempt. Held in-memory (see MemoryLivenessSessionStore) and
/// mutated in place as frames are submitted; the challenge order is randomized at creation to
/// resist replayed/pre-recorded video attacks.
/// </summary>
public sealed class LivenessSession
{
    public required string SessionId { get; init; }
    public required float[] ReferenceEmbedding { get; init; }
    public required List<ChallengeType> ChallengeQueue { get; init; }
    public int CurrentChallengeIndex { get; set; }
    public LivenessStatus Status { get; set; } = LivenessStatus.AwaitingChallenge;
    public string? FailureReason { get; set; }
    public required DateTimeOffset CreatedAt { get; init; }
    public required DateTimeOffset ExpiresAt { get; init; }

    /// <summary>Rolling metric history (EAR / mouth-ratio / yaw-offset) for the currently active challenge, used to detect a transition (e.g. open -> closed -> open for a blink) across frames.</summary>
    public List<float> MetricHistory { get; } = new();

    /// <summary>True once the "baseline" state (eyes open / neutral yaw) has been observed, so a subsequent frame crossing the action threshold can count as a genuine transition rather than a static pose.</summary>
    public bool ChallengeArmed { get; set; }

    public float? Similarity { get; set; }

    public ChallengeType? CurrentChallenge =>
        CurrentChallengeIndex < ChallengeQueue.Count ? ChallengeQueue[CurrentChallengeIndex] : null;
}
