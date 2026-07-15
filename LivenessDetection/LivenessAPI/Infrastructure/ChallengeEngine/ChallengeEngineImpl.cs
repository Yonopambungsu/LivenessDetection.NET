using LivenessAPI.Application;
using LivenessAPI.Application.Abstractions;
using LivenessAPI.Domain;
using LivenessAPI.Infrastructure.HeadPose;
using Microsoft.Extensions.Options;
using SixLabors.ImageSharp;
using BlinkDetector = LivenessAPI.Infrastructure.Blink.BlinkDetector;
using SmileDetector = LivenessAPI.Infrastructure.Smile.SmileDetector;

namespace LivenessAPI.Infrastructure.ChallengeEngine;

public sealed class ChallengeEngineImpl(IOptions<LivenessOptions> options) : IChallengeEngine
{
    private readonly LivenessOptions _options = options.Value;

    public ChallengeEvaluation Evaluate(ChallengeType type, PointF[] landmarks, LivenessSession session)
    {
        return type switch
        {
            ChallengeType.Blink => EvaluateBlink(landmarks, session),
            ChallengeType.Smile => EvaluateSmile(SmileDetector.MouthRatio(landmarks), session),
            ChallengeType.LookLeft => EvaluateYaw(landmarks, turnLeft: true, session),
            ChallengeType.LookRight => EvaluateYaw(landmarks, turnLeft: false, session),
            _ => new ChallengeEvaluation(false),
        };
    }

    private float MeasureYaw(PointF[] landmarks) =>
        HeadPoseEstimator.YawOffset(landmarks, _options.InvertYawSign);

    private ChallengeEvaluation EvaluateBlink(PointF[] landmarks, LivenessSession session)
    {
        float ear = BlinkDetector.EyeAspectRatio(landmarks);

        if (session.MetricHistory.Count == 0)
        {
            session.MetricHistory.Add(ear);
            session.ChallengeArmed = false;
            return new ChallengeEvaluation(false, $"ear={ear:F3} (calibrating)");
        }

        float baseline = session.MetricHistory[0];

        if (!session.ChallengeArmed)
        {
            if (ear > baseline)
            {
                session.MetricHistory[0] = ear;
                baseline = ear;
            }
            else if (ear < baseline * _options.BlinkClosedRatio)
            {
                session.ChallengeArmed = true;
            }

            return new ChallengeEvaluation(false, $"ear={ear:F3} baseline={baseline:F3}");
        }

        if (ear > baseline * _options.BlinkRecoverRatio)
        {
            session.MetricHistory.Add(ear);
        }
        else if (session.MetricHistory.Count > 1)
        {
            session.MetricHistory.RemoveRange(1, session.MetricHistory.Count - 1);
        }

        int recoveryHold = Math.Max(0, session.MetricHistory.Count - 1);
        if (recoveryHold >= _options.BlinkRequiredConsecutiveFrames)
        {
            return new ChallengeEvaluation(true, $"ear={ear:F3} baseline={baseline:F3} (blink confirmed)");
        }

        return new ChallengeEvaluation(false, $"ear={ear:F3} baseline={baseline:F3} reopenHold={recoveryHold}/{_options.BlinkRequiredConsecutiveFrames}");
    }

    /// <summary>Smile is measured as an increase in mouth ratio from a personal resting baseline captured
    /// over the first few frames, not a fixed absolute threshold — avoids failing users whose resting
    /// mouth is naturally wider than average.</summary>
    private ChallengeEvaluation EvaluateSmile(float mouthRatio, LivenessSession session)
    {
        if (!session.ChallengeArmed)
        {
            if (session.MetricHistory.Count == 0)
            {
                session.MetricHistory.Add(mouthRatio);
                session.WrongHoldCount = 1;
                return new ChallengeEvaluation(false, $"smile={mouthRatio:F3} (calibrating resting mouth)");
            }

            float baseline = session.MetricHistory[0];
            if (mouthRatio < baseline)
            {
                session.MetricHistory[0] = mouthRatio;
                baseline = mouthRatio;
            }

            session.WrongHoldCount++;
            if (session.WrongHoldCount >= _options.SmileCalibrationFrames)
            {
                session.ChallengeArmed = true;
                session.WrongHoldCount = 0;
            }

            return new ChallengeEvaluation(false,
                $"smile={mouthRatio:F3} baseline={baseline:F3} calFrames={session.WrongHoldCount}/{_options.SmileCalibrationFrames}");
        }

        float restingBaseline = session.MetricHistory[0];
        float smileThreshold = restingBaseline + _options.SmileDeltaThreshold;
        bool satisfied = mouthRatio > smileThreshold;

        if (satisfied)
        {
            session.MetricHistory.Add(mouthRatio);
        }
        else if (session.MetricHistory.Count > 1)
        {
            session.MetricHistory.RemoveRange(1, session.MetricHistory.Count - 1);
        }

        int hold = Math.Max(0, session.MetricHistory.Count - 1);
        bool passed = hold >= _options.SmileRequiredConsecutiveFrames;
        return new ChallengeEvaluation(passed,
            $"smile={mouthRatio:F3} baseline={restingBaseline:F3} threshold={smileThreshold:F3} hold={hold}/{_options.SmileRequiredConsecutiveFrames}");
    }

    private ChallengeEvaluation EvaluateYaw(PointF[] landmarks, bool turnLeft, LivenessSession session)
    {
        float yaw = MeasureYaw(landmarks);
        float threshold = _options.YawOffsetThreshold;
        int holdRequired = _options.YawRequiredConsecutiveFrames;

        if (!session.ChallengeArmed)
        {
            if (HeadPoseEstimator.IsNearNeutral(yaw, session.NeutralYawBaseline, _options.YawNeutralThreshold))
            {
                session.ChallengeArmed = true;
                session.YawChallengeBaseline = yaw;
            }

            return new ChallengeEvaluation(false,
                $"yaw={yaw:F3} neutral={session.NeutralYawBaseline:F3} (waiting for neutral pose before starting)");
        }

        float baseline = session.YawChallengeBaseline ?? session.NeutralYawBaseline;
        float delta = yaw - baseline;

        bool satisfied = turnLeft ? delta > threshold : delta < -threshold;
        bool wrongDirection = turnLeft ? delta < -threshold : delta > threshold;

        if (satisfied)
        {
            session.MetricHistory.Add(delta);
            session.WrongHoldCount = 0;
        }
        else if (wrongDirection)
        {
            session.WrongHoldCount++;
            session.MetricHistory.Clear();
        }
        else
        {
            session.MetricHistory.Clear();
            session.WrongHoldCount = 0;
        }

        bool passed = session.MetricHistory.Count >= holdRequired;
        bool wrongConfirmed = session.WrongHoldCount >= holdRequired;
        return new ChallengeEvaluation(
            passed,
            $"yaw={yaw:F3} delta={delta:F3} threshold=±{threshold:F3} hold={session.MetricHistory.Count}/{holdRequired} wrongHold={session.WrongHoldCount}/{holdRequired}",
            WrongGesture: wrongConfirmed);
    }
}
