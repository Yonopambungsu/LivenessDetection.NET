using System.Collections.Concurrent;
using LivenessAPI.Application.Abstractions;
using LivenessAPI.Domain;

namespace LivenessAPI.Infrastructure.Session;

/// <summary>
/// In-process session store backed by a concurrent dictionary. Unlike IMemoryCache, entries are
/// never evicted under memory pressure — eviction was a common cause of "session not found" errors
/// while the user was still actively submitting frames.
/// </summary>
public sealed class MemoryLivenessSessionStore : ILivenessSessionStore
{
    private readonly ConcurrentDictionary<string, LivenessSession> _sessions = new(StringComparer.Ordinal);

    public void Save(LivenessSession session)
    {
        _sessions[session.SessionId] = session;
        CleanupExpiredSessions();
    }

    public LivenessSession? Get(string sessionId) =>
        _sessions.TryGetValue(sessionId, out var session) ? session : null;

    public void Remove(string sessionId) => _sessions.TryRemove(sessionId, out _);

    private void CleanupExpiredSessions()
    {
        var cutoff = DateTimeOffset.UtcNow.AddMinutes(-5);
        foreach (var (id, session) in _sessions)
        {
            if (session.ExpiresAt < cutoff)
            {
                _sessions.TryRemove(id, out _);
            }
        }
    }
}
