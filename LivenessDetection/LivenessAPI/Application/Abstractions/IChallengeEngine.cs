using LivenessAPI.Domain;
using SixLabors.ImageSharp;

namespace LivenessAPI.Application.Abstractions;

/// <summary>Detail carries the raw metric value (e.g. "ear=0.19 baseline=0.31") so callers can surface
/// it for troubleshooting/threshold-tuning without needing a separate debug endpoint.</summary>
public sealed record ChallengeEvaluation(bool Passed, string? Detail = null);

/// <summary>
/// Evaluates the currently active challenge (blink/smile/look-left/look-right) for one session
/// against one frame's landmarks. Stateful across calls via the session's MetricHistory/ChallengeArmed
/// fields, since a single frame can't tell "blinking" from "eyes momentarily occluded" — it needs to
/// see the open -> closed -> open (or neutral -> turned) transition across a few frames.
/// </summary>
public interface IChallengeEngine
{
    ChallengeEvaluation Evaluate(ChallengeType type, PointF[] landmarks, LivenessSession session);
}
