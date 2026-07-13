using LivenessAPI.Application.Abstractions;
using LivenessAPI.Application.Dtos;
using LivenessAPI.Domain;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace LivenessAPI.Application;

/// <summary>
/// Orchestrates one liveness attempt end to end: reference embedding at session start, then per
/// submitted frame dispatches to the current challenge check, anti-spoofing, or final recognition
/// depending on session status. See Infra.txt for the intended pipeline shape.
/// </summary>
public sealed class LivenessSessionService(
    IFaceDetectionService faceDetection,
    IFaceLandmarkService landmarkService,
    IChallengeEngine challengeEngine,
    IAntiSpoofService antiSpoofService,
    IFaceRecognitionService recognitionService,
    ILivenessSessionStore store,
    IOptions<LivenessOptions> options,
    ILogger<LivenessSessionService> logger) : ILivenessSessionService
{
    private readonly LivenessOptions _options = options.Value;

    public StartSessionResponse StartSession(StartSessionRequest request)
    {
        using var image = DecodeBase64Image(request.ReferenceImageBase64);

        var faceResult = faceDetection.DetectSingleFace(image);
        if (faceResult.Face is null)
        {
            throw new LivenessValidationException(faceResult.Error ?? "No face detected in reference image.");
        }

        var embedding = recognitionService.GetEmbedding(image, faceResult.Face);

        var challengeQueue = Enum.GetValues<ChallengeType>()
            .OrderBy(_ => Random.Shared.Next())
            .Take(Math.Clamp(_options.ChallengeCount, 1, 4))
            .ToList();

        var now = DateTimeOffset.UtcNow;
        var session = new LivenessSession
        {
            SessionId = Guid.NewGuid().ToString("N"),
            ReferenceEmbedding = embedding,
            ChallengeQueue = challengeQueue,
            CreatedAt = now,
            ExpiresAt = now.AddSeconds(_options.SessionTtlSeconds),
        };
        store.Save(session);

        return new StartSessionResponse(session.SessionId, session.Status.ToString(), session.CurrentChallenge?.ToString(), session.ExpiresAt);
    }

    public SubmitFrameResponse SubmitFrame(string sessionId, SubmitFrameRequest request)
    {
        var session = store.Get(sessionId) ?? throw new LivenessSessionNotFoundException(sessionId);

        if (IsFinished(session.Status))
        {
            return Respond(session, $"Session already finished with status {session.Status}.");
        }

        if (DateTimeOffset.UtcNow > session.ExpiresAt)
        {
            session.Status = LivenessStatus.Expired;
            session.FailureReason = "Session expired.";
            store.Save(session);
            return Respond(session, session.FailureReason);
        }

        using var image = DecodeBase64Image(request.ImageBase64);

        var faceResult = faceDetection.DetectSingleFace(image);
        if (faceResult.Face is null)
        {
            return Respond(session, faceResult.Error ?? "No face detected.");
        }

        var face = faceResult.Face;

        switch (session.Status)
        {
            case LivenessStatus.AwaitingChallenge:
            {
                var landmarks = landmarkService.GetLandmarks(image, face);
                var challenge = session.CurrentChallenge!.Value;
                var evaluation = challengeEngine.Evaluate(challenge, landmarks, session);

                if (evaluation.Passed)
                {
                    session.CurrentChallengeIndex++;
                    session.MetricHistory.Clear();
                    session.ChallengeArmed = false;

                    if (session.CurrentChallenge is null)
                    {
                        session.Status = LivenessStatus.ChallengesPassed;
                    }
                }

                store.Save(session);
                var baseMessage = evaluation.Passed ? $"Challenge {challenge} passed." : $"Keep performing: {challenge}.";
                return Respond(session, evaluation.Detail is null ? baseMessage : $"{baseMessage} [{evaluation.Detail}]");
            }

            case LivenessStatus.ChallengesPassed:
            {
                if (!_options.AntiSpoofEnabled)
                {
                    session.Status = LivenessStatus.AntiSpoofPassed;
                    store.Save(session);
                    return Respond(session, "Anti-spoofing skipped (disabled in config).");
                }

                var spoof = antiSpoofService.Predict(image, face);
                var probs = string.Join(", ", spoof.Probabilities.Select((p, i) => $"class{i}={p:F3}"));

                if (!spoof.IsReal)
                {
                    session.Status = LivenessStatus.Failed;
                    session.FailureReason = $"Anti-spoofing check failed. [{probs}]";
                    store.Save(session);
                    return Respond(session, session.FailureReason);
                }

                session.Status = LivenessStatus.AntiSpoofPassed;
                store.Save(session);
                return Respond(session, $"Anti-spoofing passed. [{probs}]");
            }

            case LivenessStatus.AntiSpoofPassed:
            {
                var embedding = recognitionService.GetEmbedding(image, face);
                var similarity = recognitionService.CosineSimilarity(embedding, session.ReferenceEmbedding);
                bool matched = similarity >= _options.RecognitionThreshold;

                session.Similarity = similarity;
                session.Status = matched ? LivenessStatus.Success : LivenessStatus.Failed;
                session.FailureReason = matched ? null : "Face does not match reference image.";
                store.Save(session);

                return new SubmitFrameResponse(
                    session.SessionId,
                    session.Status.ToString(),
                    null,
                    matched ? "Liveness and recognition succeeded." : session.FailureReason!,
                    similarity,
                    matched);
            }

            default:
                return Respond(session, "Unexpected session state.");
        }
    }

    public SessionStatusResponse GetStatus(string sessionId)
    {
        var session = store.Get(sessionId) ?? throw new LivenessSessionNotFoundException(sessionId);
        var remaining = session.ChallengeQueue.Skip(session.CurrentChallengeIndex).Select(c => c.ToString()).ToList();
        return new SessionStatusResponse(session.SessionId, session.Status.ToString(), session.CurrentChallenge?.ToString(), remaining, session.ExpiresAt, session.FailureReason);
    }

    private static bool IsFinished(LivenessStatus status) =>
        status is LivenessStatus.Success or LivenessStatus.Failed or LivenessStatus.Expired;

    private static SubmitFrameResponse Respond(LivenessSession session, string message) =>
        new(session.SessionId, session.Status.ToString(), session.CurrentChallenge?.ToString(), message, session.Similarity);

    private Image<Rgb24> DecodeBase64Image(string base64)
    {
        var commaIndex = base64.IndexOf(',');
        var payload = base64.StartsWith("data:", StringComparison.OrdinalIgnoreCase) && commaIndex >= 0
            ? base64[(commaIndex + 1)..]
            : base64;

        byte[] bytes;
        try
        {
            bytes = Convert.FromBase64String(payload);
        }
        catch (FormatException ex)
        {
            logger.LogWarning(ex, "Rejected frame: not valid base64.");
            throw new LivenessValidationException("Image is not valid base64.");
        }

        try
        {
            return Image.Load<Rgb24>(bytes);
        }
        catch (Exception ex) when (ex is not LivenessValidationException)
        {
            logger.LogWarning(ex, "Rejected frame: could not decode image data ({ByteCount} bytes).", bytes.Length);
            throw new LivenessValidationException("Could not decode image data.");
        }
    }
}
