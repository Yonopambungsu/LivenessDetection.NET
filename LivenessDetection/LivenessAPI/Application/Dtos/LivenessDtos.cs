namespace LivenessAPI.Application.Dtos;

public sealed record StartSessionRequest(string ReferenceImageBase64);

public sealed record StartSessionResponse(
    string SessionId,
    string Status,
    string? CurrentChallenge,
    DateTimeOffset ExpiresAt,
    int ChallengeTotal,
    int ChallengeIndex,
    string UiPhase,
    string Instruction);

public sealed record SubmitFrameRequest(string ImageBase64);

public sealed record SubmitFrameResponse(
    string SessionId,
    string Status,
    string? CurrentChallenge,
    string Message,
    float? Similarity = null,
    bool? Matched = null,
    IReadOnlyList<string>? FailedChallenges = null,
    int ChallengeIndex = 0,
    int ChallengeTotal = 0,
    string UiPhase = "",
    string Instruction = "",
    int? RetriesRemaining = null,
    bool ChallengeJustPassed = false,
    string? FacePositionHint = null);

public sealed record SessionStatusResponse(
    string SessionId,
    string Status,
    string? CurrentChallenge,
    IReadOnlyList<string> RemainingChallenges,
    DateTimeOffset ExpiresAt,
    string? FailureReason,
    int ChallengeTotal,
    int ChallengeIndex,
    string UiPhase,
    string Instruction);
