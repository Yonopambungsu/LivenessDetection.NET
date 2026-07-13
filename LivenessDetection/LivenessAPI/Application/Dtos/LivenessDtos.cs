namespace LivenessAPI.Application.Dtos;

public sealed record StartSessionRequest(string ReferenceImageBase64);

public sealed record StartSessionResponse(string SessionId, string Status, string? CurrentChallenge, DateTimeOffset ExpiresAt);

public sealed record SubmitFrameRequest(string ImageBase64);

public sealed record SubmitFrameResponse(
    string SessionId,
    string Status,
    string? CurrentChallenge,
    string Message,
    float? Similarity = null,
    bool? Matched = null);

public sealed record SessionStatusResponse(
    string SessionId,
    string Status,
    string? CurrentChallenge,
    IReadOnlyList<string> RemainingChallenges,
    DateTimeOffset ExpiresAt,
    string? FailureReason);
