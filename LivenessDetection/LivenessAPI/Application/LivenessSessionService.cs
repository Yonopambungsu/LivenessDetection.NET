using System.Collections.Concurrent;
using AIModels.Detection;
using LivenessAPI.Application.Abstractions;
using LivenessAPI.Application.Dtos;
using LivenessAPI.Domain;
using LivenessAPI.Infrastructure.HeadPose;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace LivenessAPI.Application;

/// <summary>
/// Orchestrates one liveness attempt end to end: calibration, challenge loop with retries,
/// return-to-neutral transitions, anti-spoofing, and face recognition.
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
    private static readonly ConcurrentDictionary<string, SemaphoreSlim> SessionLocks = new();
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
        var challengeQueue = BuildChallengeQueue();
        var now = DateTimeOffset.UtcNow;

        var session = new LivenessSession
        {
            SessionId = Guid.NewGuid().ToString("N"),
            ReferenceEmbedding = embedding,
            ChallengeQueue = challengeQueue,
            Status = LivenessStatus.Calibrating,
            CreatedAt = now,
            ExpiresAt = now.AddSeconds(_options.SessionTtlSeconds),
            ChallengeStartedAt = now,
        };
        store.Save(session);

        return BuildStartResponse(session);
    }

    public SubmitFrameResponse SubmitFrame(string sessionId, SubmitFrameRequest request)
    {
        var gate = SessionLocks.GetOrAdd(sessionId, _ => new SemaphoreSlim(1, 1));
        gate.Wait();
        try
        {
            return SubmitFrameCore(sessionId, request);
        }
        finally
        {
            gate.Release();
        }
    }

    public SessionStatusResponse GetStatus(string sessionId)
    {
        var session = store.Get(sessionId) ?? throw new LivenessSessionNotFoundException(sessionId);
        var remaining = session.ChallengeQueue.Skip(session.CurrentChallengeIndex).Select(c => c.ToString()).ToList();
        var (uiPhase, instruction) = GetUiContext(session);

        return new SessionStatusResponse(
            session.SessionId,
            session.Status.ToString(),
            session.CurrentChallenge?.ToString(),
            remaining,
            session.ExpiresAt,
            session.FailureReason,
            session.ChallengeQueue.Count,
            session.CurrentChallengeIndex,
            uiPhase,
            instruction);
    }

    private SubmitFrameResponse SubmitFrameCore(string sessionId, SubmitFrameRequest request)
    {
        var session = store.Get(sessionId) ?? throw new LivenessSessionNotFoundException(sessionId);

        // Sliding expiration: extend the deadline on every frame so slow calibration, ONNX cold
        // start, and user hesitation do not expire an otherwise active session. Persisted right away
        // (not just mutated on the in-memory object) so it takes effect even on the code paths below
        // that return early without their own store.Save() call - e.g. no face detected in this
        // particular frame. Without this, a run of no-face-detected frames (flaky lighting, a stale
        // camera frame, momentary occlusion) would silently stop refreshing the cached TTL until it
        // finally lapsed, surfacing as "session not found or expired" instead of a normal timeout.
        session.ExpiresAt = DateTimeOffset.UtcNow.AddSeconds(_options.SessionTtlSeconds);
        store.Save(session);

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
            return Respond(session, faceResult.Error ?? "No face detected.", facePositionHint: "Pastikan wajah terlihat jelas di kamera.");
        }

        return session.Status switch
        {
            LivenessStatus.Calibrating => HandleCalibrating(session, image, faceResult.Face),
            LivenessStatus.ReturnToNeutral => HandleReturnToNeutral(session, image, faceResult.Face),
            LivenessStatus.AwaitingChallenge => HandleChallenge(session, image, faceResult.Face),
            LivenessStatus.ChallengesPassed => HandleAntiSpoof(session, image, faceResult.Face),
            LivenessStatus.AntiSpoofPassed => HandleRecognition(session, image, faceResult.Face),
            _ => Respond(session, "Unexpected session state."),
        };
    }

    private SubmitFrameResponse HandleCalibrating(LivenessSession session, Image<Rgb24> image, DetectedFace face)
    {
        var hint = EvaluateFacePosition(image, face);
        if (hint is not null)
        {
            session.CalibrationGoodSince = null;
            store.Save(session);
            return Respond(session, hint, facePositionHint: hint);
        }

        var now = DateTimeOffset.UtcNow;
        session.CalibrationGoodSince ??= now;

        var held = now - session.CalibrationGoodSince.Value;
        if (held < TimeSpan.FromSeconds(_options.CalibrationSeconds))
        {
            store.Save(session);
            return Respond(session, "Tahan posisi wajah...");
        }

        var landmarks = landmarkService.GetLandmarks(image, face);
        session.NeutralYawBaseline = MeasureYaw(landmarks);
        session.Status = LivenessStatus.AwaitingChallenge;
        session.ChallengeStartedAt = now;
        ResetChallengeState(session);
        store.Save(session);

        return Respond(session, AppendDebug("Kalibrasi selesai. Mulai verifikasi.",
            $"neutralYaw={session.NeutralYawBaseline:F3}"));
    }

    private SubmitFrameResponse HandleReturnToNeutral(LivenessSession session, Image<Rgb24> image, DetectedFace face)
    {
        session.TransitionStartedAt ??= DateTimeOffset.UtcNow;
        var elapsed = DateTimeOffset.UtcNow - session.TransitionStartedAt.Value;

        var landmarks = landmarkService.GetLandmarks(image, face);
        var yaw = MeasureYaw(landmarks);
        if (HeadPoseEstimator.IsNearNeutral(yaw, session.NeutralYawBaseline, _options.YawNeutralThreshold))
        {
            session.NeutralHoldCount++;
        }
        else
        {
            session.NeutralHoldCount = 0;
        }

        bool minTimeReached = elapsed >= TimeSpan.FromSeconds(_options.TransitionMinSeconds);
        bool neutralConfirmed = session.NeutralHoldCount >= _options.YawRequiredConsecutiveFrames;
        bool maxTimeReached = elapsed >= TimeSpan.FromSeconds(_options.TransitionMaxSeconds);

        if (!minTimeReached || (!neutralConfirmed && !maxTimeReached))
        {
            store.Save(session);
            return Respond(session, AppendDebug("Kembali hadap lurus ke kamera.",
                $"yaw={yaw:F3} neutral={session.NeutralYawBaseline:F3}"));
        }

        session.Status = LivenessStatus.AwaitingChallenge;
        session.TransitionStartedAt = null;
        session.NeutralHoldCount = 0;
        session.ChallengeStartedAt = DateTimeOffset.UtcNow;
        ResetChallengeState(session);
        store.Save(session);

        return Respond(session, "Siap. Lanjut ke langkah berikutnya.");
    }

    private SubmitFrameResponse HandleChallenge(LivenessSession session, Image<Rgb24> image, DetectedFace face)
    {
        var landmarks = landmarkService.GetLandmarks(image, face);
        var challenge = session.CurrentChallenge!.Value;
        var evaluation = challengeEngine.Evaluate(challenge, landmarks, session);

        bool timedOut = false;
        if (!evaluation.Passed && !evaluation.WrongGesture)
        {
            var elapsed = DateTimeOffset.UtcNow - session.ChallengeStartedAt;
            if (elapsed < TimeSpan.FromSeconds(_options.MaxSecondsPerChallenge))
            {
                store.Save(session);
                var waitMessage = GetChallengeInstruction(challenge);
                return Respond(session, AppendDebug(waitMessage, evaluation.Detail));
            }

            timedOut = true;
        }

        if (evaluation.Passed)
        {
            return ResolveChallengePassed(session, challenge, evaluation.Detail);
        }

        if (evaluation.WrongGesture || timedOut)
        {
            return ResolveChallengeFailed(session, challenge, timedOut, evaluation.Detail);
        }

        store.Save(session);
        return Respond(session, AppendDebug(GetChallengeInstruction(challenge), evaluation.Detail));
    }

    private SubmitFrameResponse ResolveChallengePassed(LivenessSession session, ChallengeType challenge, string? detail)
    {
        session.CurrentChallengeRetries = 0;
        ResetChallengeState(session);

        bool hasMore = session.CurrentChallengeIndex + 1 < session.ChallengeQueue.Count;
        if (hasMore)
        {
            session.CurrentChallengeIndex++;
            session.Status = LivenessStatus.ReturnToNeutral;
            session.TransitionStartedAt = DateTimeOffset.UtcNow;
            session.NeutralHoldCount = 0;
            store.Save(session);

            return Respond(
                session,
                AppendDebug($"{GetChallengeSuccessLabel(challenge)} Lanjut ke langkah berikutnya.", detail),
                challengeJustPassed: true);
        }

        session.CurrentChallengeIndex++;
        if (session.FailedChallenges.Count == 0)
        {
            session.Status = LivenessStatus.ChallengesPassed;
            logger.LogInformation("Liveness session {SessionId}: all challenges passed.", session.SessionId);
            store.Save(session);
            return Respond(
                session,
                AppendDebug("Semua langkah selesai. Memverifikasi identitas...", detail),
                challengeJustPassed: true);
        }

        session.Status = LivenessStatus.Failed;
        session.FailureReason =
            $"Liveness verification failed: could not confirm {string.Join(", ", session.FailedChallenges)}.";
        store.Save(session);
        return Respond(session, session.FailureReason, challengeJustPassed: true);
    }

    private SubmitFrameResponse ResolveChallengeFailed(
        LivenessSession session,
        ChallengeType challenge,
        bool timedOut,
        string? detail)
    {
        if (session.CurrentChallengeRetries < _options.MaxRetriesPerChallenge)
        {
            session.CurrentChallengeRetries++;
            session.ChallengeStartedAt = DateTimeOffset.UtcNow;
            ResetChallengeState(session);
            store.Save(session);

            var reason = timedOut ? "Waktu habis." : "Gerakan tidak sesuai.";
            var retriesLeft = _options.MaxRetriesPerChallenge - session.CurrentChallengeRetries;
            return Respond(
                session,
                AppendDebug($"{reason} Coba lagi ({retriesLeft} percobaan tersisa).", detail),
                retriesRemaining: retriesLeft);
        }

        session.FailedChallenges.Add(challenge);
        session.CurrentChallengeRetries = 0;
        ResetChallengeState(session);

        bool hasMore = session.CurrentChallengeIndex + 1 < session.ChallengeQueue.Count;
        if (hasMore)
        {
            session.CurrentChallengeIndex++;
            session.Status = LivenessStatus.ReturnToNeutral;
            session.TransitionStartedAt = DateTimeOffset.UtcNow;
            session.NeutralHoldCount = 0;
            store.Save(session);

            var failMsg = timedOut
                ? $"Langkah {GetChallengeLabel(challenge)} waktu habis, lanjut ke berikutnya."
                : $"Langkah {GetChallengeLabel(challenge)} gagal, lanjut ke berikutnya.";
            return Respond(session, AppendDebug(failMsg, detail));
        }

        session.CurrentChallengeIndex++;
        session.Status = LivenessStatus.Failed;
        session.FailureReason =
            $"Liveness verification failed: could not confirm {string.Join(", ", session.FailedChallenges)}.";
        logger.LogWarning(
            "Liveness session {SessionId}: verification failed. Failed: {FailedChallenges}.",
            session.SessionId, string.Join(", ", session.FailedChallenges));
        store.Save(session);
        return Respond(session, session.FailureReason);
    }

    private SubmitFrameResponse HandleAntiSpoof(LivenessSession session, Image<Rgb24> image, DetectedFace face)
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
            session.FailureReason = "Verifikasi keaslian wajah gagal.";
            store.Save(session);
            return Respond(session, AppendDebug(session.FailureReason, _options.IncludeDebugMetrics ? probs : null));
        }

        session.Status = LivenessStatus.AntiSpoofPassed;
        store.Save(session);
        return Respond(session, AppendDebug("Memverifikasi kemiripan wajah...", _options.IncludeDebugMetrics ? probs : null));
    }

    private SubmitFrameResponse HandleRecognition(LivenessSession session, Image<Rgb24> image, DetectedFace face)
    {
        var embedding = recognitionService.GetEmbedding(image, face);
        var similarity = recognitionService.CosineSimilarity(embedding, session.ReferenceEmbedding);
        bool matched = similarity >= _options.RecognitionThreshold;

        session.Similarity = similarity;
        session.Status = matched ? LivenessStatus.Success : LivenessStatus.Failed;
        session.FailureReason = matched ? null : "Wajah tidak cocok dengan foto referensi.";
        store.Save(session);

        var (uiPhase, instruction) = GetUiContext(session);
        return new SubmitFrameResponse(
            session.SessionId,
            session.Status.ToString(),
            null,
            matched ? "Verifikasi berhasil." : session.FailureReason!,
            similarity,
            matched,
            session.FailedChallenges.Count > 0
                ? session.FailedChallenges.Select(c => c.ToString()).ToList()
                : null,
            session.CurrentChallengeIndex,
            session.ChallengeQueue.Count,
            uiPhase,
            instruction);
    }

    private List<ChallengeType> BuildChallengeQueue()
    {
        var count = Math.Clamp(_options.ChallengeCount, 1, 4);

        if (_options.UseFixedChallengeOrder)
        {
            var queue = new List<ChallengeType>();
            foreach (var name in _options.FixedChallengeOrder)
            {
                if (Enum.TryParse<ChallengeType>(name, ignoreCase: true, out var type))
                {
                    queue.Add(type);
                }
            }

            if (queue.Count == 0)
            {
                queue.AddRange(Enum.GetValues<ChallengeType>());
            }

            return queue.Take(count).ToList();
        }

        return Enum.GetValues<ChallengeType>()
            .OrderBy(_ => Random.Shared.Next())
            .Take(count)
            .ToList();
    }

    private string? EvaluateFacePosition(Image<Rgb24> image, DetectedFace face)
    {
        float frameArea = image.Width * image.Height;
        float faceAreaRatio = face.Width * face.Height / frameArea;

        if (faceAreaRatio < _options.MinFaceAreaRatio)
        {
            return "Dekatkan wajah ke kamera.";
        }

        if (faceAreaRatio > _options.MaxFaceAreaRatio)
        {
            return "Jauhkan sedikit dari kamera.";
        }

        float centerX = image.Width / 2f;
        float tolerance = image.Width * _options.FaceCenterToleranceRatio;
        if (MathF.Abs(face.Center.X - centerX) > tolerance)
        {
            return "Posisikan wajah di tengah lingkaran.";
        }

        return null;
    }

    private static void ResetChallengeState(LivenessSession session)
    {
        session.MetricHistory.Clear();
        session.ChallengeArmed = false;
        session.WrongHoldCount = 0;
        session.YawChallengeBaseline = null;
    }

    private float MeasureYaw(PointF[] landmarks) =>
        HeadPoseEstimator.YawOffset(landmarks, _options.InvertYawSign);

    private (string UiPhase, string Instruction) GetUiContext(LivenessSession session)
    {
        if (session.Status is LivenessStatus.Success)
        {
            return (LivenessUiPhase.Complete, "Verifikasi berhasil.");
        }

        if (session.Status is LivenessStatus.Failed or LivenessStatus.Expired)
        {
            return (LivenessUiPhase.Failed, session.FailureReason ?? "Verifikasi gagal.");
        }

        if (session.Status is LivenessStatus.Calibrating)
        {
            return (LivenessUiPhase.Calibrating, "Posisikan wajah di dalam lingkaran dan hadap lurus.");
        }

        if (session.Status is LivenessStatus.ReturnToNeutral)
        {
            return (LivenessUiPhase.ReturnToNeutral, "Kembali hadap lurus ke kamera.");
        }

        if (session.Status is LivenessStatus.ChallengesPassed or LivenessStatus.AntiSpoofPassed)
        {
            return (LivenessUiPhase.Verifying, "Sedang memverifikasi identitas Anda...");
        }

        if (session.CurrentChallengeRetries > 0)
        {
            var challenge = session.CurrentChallenge!.Value;
            return (LivenessUiPhase.ChallengeRetry, $"Coba lagi: {GetChallengeInstruction(challenge)}");
        }

        if (session.CurrentChallenge is { } active)
        {
            return (LivenessUiPhase.Challenge, GetChallengeInstruction(active));
        }

        return (LivenessUiPhase.Verifying, "Memverifikasi...");
    }

    private static string GetChallengeInstruction(ChallengeType challenge) => challenge switch
    {
        ChallengeType.Blink => "Kedipkan mata Anda secara normal",
        ChallengeType.Smile => "Berikan senyuman lebar",
        ChallengeType.LookLeft => "Tolehkan kepala perlahan ke kiri",
        ChallengeType.LookRight => "Tolehkan kepala perlahan ke kanan",
        _ => challenge.ToString(),
    };

    private static string GetChallengeLabel(ChallengeType challenge) => challenge switch
    {
        ChallengeType.Blink => "kedip",
        ChallengeType.Smile => "senyum",
        ChallengeType.LookLeft => "toleh kiri",
        ChallengeType.LookRight => "toleh kanan",
        _ => challenge.ToString(),
    };

    private static string GetChallengeSuccessLabel(ChallengeType challenge) => challenge switch
    {
        ChallengeType.Blink => "Kedip terdeteksi.",
        ChallengeType.Smile => "Senyum terdeteksi.",
        ChallengeType.LookLeft => "Toleh kiri terdeteksi.",
        ChallengeType.LookRight => "Toleh kanan terdeteksi.",
        _ => "Langkah selesai.",
    };

    private StartSessionResponse BuildStartResponse(LivenessSession session)
    {
        var (uiPhase, instruction) = GetUiContext(session);
        return new StartSessionResponse(
            session.SessionId,
            session.Status.ToString(),
            session.CurrentChallenge?.ToString(),
            session.ExpiresAt,
            session.ChallengeQueue.Count,
            session.CurrentChallengeIndex,
            uiPhase,
            instruction);
    }

    private SubmitFrameResponse Respond(
        LivenessSession session,
        string message,
        bool challengeJustPassed = false,
        int? retriesRemaining = null,
        string? facePositionHint = null)
    {
        var (uiPhase, instruction) = GetUiContext(session);
        if (facePositionHint is not null && session.Status == LivenessStatus.Calibrating)
        {
            instruction = facePositionHint;
        }

        return new SubmitFrameResponse(
            session.SessionId,
            session.Status.ToString(),
            session.CurrentChallenge?.ToString(),
            message,
            session.Similarity,
            FailedChallenges: session.FailedChallenges.Count > 0
                ? session.FailedChallenges.Select(c => c.ToString()).ToList()
                : null,
            ChallengeIndex: session.CurrentChallengeIndex,
            ChallengeTotal: session.ChallengeQueue.Count,
            UiPhase: uiPhase,
            Instruction: instruction,
            RetriesRemaining: retriesRemaining,
            ChallengeJustPassed: challengeJustPassed,
            FacePositionHint: facePositionHint);
    }

    private string AppendDebug(string message, string? detail) =>
        _options.IncludeDebugMetrics && detail is not null ? $"{message} [{detail}]" : message;

    private static bool IsFinished(LivenessStatus status) =>
        status is LivenessStatus.Success or LivenessStatus.Failed or LivenessStatus.Expired;

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
