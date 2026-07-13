using LivenessAPI.Domain;

namespace LivenessAPI.Application.Abstractions;

public interface ILivenessSessionStore
{
    void Save(LivenessSession session);
    LivenessSession? Get(string sessionId);
    void Remove(string sessionId);
}
