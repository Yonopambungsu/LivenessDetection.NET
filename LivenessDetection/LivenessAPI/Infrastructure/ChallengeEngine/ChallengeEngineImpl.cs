using LivenessAPI.Application;
using LivenessAPI.Application.Abstractions;
using LivenessAPI.Domain;
using LivenessAPI.Infrastructure.HeadPose;
using Microsoft.Extensions.Options;
using SixLabors.ImageSharp;
using BlinkDetector = LivenessAPI.Infrastructure.Blink.BlinkDetector;
using SmileDetector = LivenessAPI.Infrastructure.Smile.SmileDetector;

namespace LivenessAPI.Infrastructure.ChallengeEngine;

/// <summary>
/// Dispatches the currently active challenge to its geometric check and drives the small per-session
/// state machine needed to confirm an *action* happened across frames (not just a static pose).
/// Reuses LivenessSession.MetricHistory/ChallengeArmed as scratch state; both are cleared by the
/// caller (LivenessSessionService) whenever a challenge is passed and the next one begins.
/// </summary>
public sealed class ChallengeEngineImpl(IOptions<LivenessOptions> options) : IChallengeEngine
{
    private readonly LivenessOptions _options = options.Value;

    public ChallengeEvaluation Evaluate(ChallengeType type, PointF[] landmarks, LivenessSession session)
    {
        return type switch
        {
            ChallengeType.Blink => EvaluateBlink(landmarks, session),
            ChallengeType.Smile => EvaluateHold("smile", SmileDetector.MouthRatio(landmarks), _options.SmileRatioThreshold, above: true, session),
            ChallengeType.LookLeft => EvaluateHold("yaw", HeadPoseEstimator.YawOffset(landmarks), _options.YawOffsetThreshold, above: true, session),
            ChallengeType.LookRight => EvaluateHold("yaw", HeadPoseEstimator.YawOffset(landmarks), -_options.YawOffsetThreshold, above: false, session),
            _ => new ChallengeEvaluation(false),
        };
    }

    /// <summary>
    /// Blink = a relative dip-then-recover in eye-openness, not a fixed absolute EAR value. A fixed
    /// threshold depends on the eye landmark points being exactly right, which we can't guarantee
    /// (see the TODO on Landmark106Indices) — a wrong index mapping means the "eyes open" baseline
    /// might never cross a hardcoded threshold, getting the challenge permanently stuck. Instead we
    /// track the highest EAR seen so far this attempt as a rolling baseline (session.MetricHistory[0]),
    /// and call it a blink as soon as the EAR drops well below that baseline and then recovers close to
    /// it again — works regardless of what the absolute EAR numbers happen to be for this model.
    /// </summary>
    private ChallengeEvaluation EvaluateBlink(PointF[] landmarks, LivenessSession session)
    {
        float ear = BlinkDetector.EyeAspectRatio(landmarks);

        if (session.MetricHistory.Count == 0)
        {
            // First frame of the attempt: seed the baseline from whatever we see (assumed roughly open).
            session.MetricHistory.Add(ear);
            session.ChallengeArmed = false;
            return new ChallengeEvaluation(false, ($"ear={ear:F3} (calibrating)"));
        }

        float baseline = session.MetricHistory[0];

        if (!session.ChallengeArmed)
        {
            // Still tracking the "eyes open" baseline upward; also catches a dip if the user blinks
            // before the baseline has settled, so an early blink isn't missed.
            if (ear > baseline)
            {
                session.MetricHistory[0] = ear;
                baseline = ear;
            }
            else if (ear < baseline * _options.BlinkClosedRatio)
            {
                session.ChallengeArmed = true; // dip observed, now waiting for recovery
            }

            return new ChallengeEvaluation(false, ($"ear={ear:F3} baseline={baseline:F3}"));
        }

        // Dip already observed; pass as soon as EAR recovers back toward the open baseline.
        if (ear > baseline * _options.BlinkRecoverRatio)
        {
            return new ChallengeEvaluation(true, ($"ear={ear:F3} baseline={baseline:F3} (blink confirmed)"));
        }

        return new ChallengeEvaluation(false, ($"ear={ear:F3} baseline={baseline:F3} (waiting for eyes to reopen)"));
    }

    /// <summary>Generic "metric crosses threshold and stays there for a few frames" check, used for
    /// smile and head-turn challenges where there's no natural open/closed cycle to look for.</summary>
    private ChallengeEvaluation EvaluateHold(string label, float metric, float threshold, bool above, LivenessSession session)
    {
        bool satisfied = above ? metric > threshold : metric < threshold;

        if (satisfied)
        {
            session.MetricHistory.Add(metric);
        }
        else
        {
            session.MetricHistory.Clear();
        }

        bool passed = session.MetricHistory.Count >= _options.RequiredConsecutiveFrames;
        return new ChallengeEvaluation(passed, ($"{label}={metric:F3} threshold={threshold:F3} hold={session.MetricHistory.Count}/{_options.RequiredConsecutiveFrames}"));
    }
}
