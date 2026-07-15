namespace LivenessAPI.Domain;

/// <summary>
/// Server-side state for one liveness attempt. Held in-memory (see MemoryLivenessSessionStore) and
/// mutated in place as frames are submitted.
/// </summary>
public sealed class LivenessSession
{
    public required string SessionId { get; init; }
    public required float[] ReferenceEmbedding { get; init; }
    public required List<ChallengeType> ChallengeQueue { get; init; }
    public int CurrentChallengeIndex { get; set; }
    public LivenessStatus Status { get; set; } = LivenessStatus.Calibrating;
    public string? FailureReason { get; set; }
    public required DateTimeOffset CreatedAt { get; init; }
    public required DateTimeOffset ExpiresAt { get; init; }

    /// <summary>Rolling metric history (EAR / mouth-ratio / yaw-offset) for the currently active challenge.</summary>
    public List<float> MetricHistory { get; } = new();

    /// <summary>True once the baseline state (eyes open / neutral yaw / resting mouth) has been observed.</summary>
    public bool ChallengeArmed { get; set; }

    /// <summary>Consecutive-frame counter for a clearly-wrong gesture on the active challenge.</summary>
    public int WrongHoldCount { get; set; }

    /// <summary>Challenges that failed after all retries were exhausted.</summary>
    public List<ChallengeType> FailedChallenges { get; } = new();

    /// <summary>When the currently active challenge began; reset on retry and queue advance.</summary>
    public required DateTimeOffset ChallengeStartedAt { get; set; }

    /// <summary>Retries already consumed on the current challenge (0 = first attempt).</summary>
    public int CurrentChallengeRetries { get; set; }

    /// <summary>When a continuous good calibration pose was first seen; null until face is well positioned.</summary>
    public DateTimeOffset? CalibrationGoodSince { get; set; }

    /// <summary>When the between-challenge return-to-neutral phase started.</summary>
    public DateTimeOffset? TransitionStartedAt { get; set; }

    /// <summary>Consecutive neutral frames observed during ReturnToNeutral.</summary>
    public int NeutralHoldCount { get; set; }

    /// <summary>Personal resting yaw captured at calibration; neutral checks compare against this.</summary>
    public float NeutralYawBaseline { get; set; }

    /// <summary>Yaw at the moment a head-turn challenge arms; pass/fail uses delta from this value.</summary>
    public float? YawChallengeBaseline { get; set; }

    public float? Similarity { get; set; }

    public ChallengeType? CurrentChallenge =>
        CurrentChallengeIndex < ChallengeQueue.Count ? ChallengeQueue[CurrentChallengeIndex] : null;
}
