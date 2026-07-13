using LivenessAPI.Application.Abstractions;
using LivenessAPI.Domain;
using Microsoft.Extensions.Caching.Memory;

namespace LivenessAPI.Infrastructure.Session;

public sealed class MemoryLivenessSessionStore(IMemoryCache cache) : ILivenessSessionStore
{
    private static string Key(string sessionId) => $"liveness-session:{sessionId}";

    public void Save(LivenessSession session)
    {
        var ttl = session.ExpiresAt - DateTimeOffset.UtcNow;
        cache.Set(Key(session.SessionId), session, ttl > TimeSpan.Zero ? ttl : TimeSpan.FromSeconds(1));
    }

    public LivenessSession? Get(string sessionId) => cache.Get<LivenessSession>(Key(sessionId));

    public void Remove(string sessionId) => cache.Remove(Key(sessionId));
}
