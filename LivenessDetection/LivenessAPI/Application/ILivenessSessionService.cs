using LivenessAPI.Application.Dtos;

namespace LivenessAPI.Application;

public interface ILivenessSessionService
{
    StartSessionResponse StartSession(StartSessionRequest request);
    SubmitFrameResponse SubmitFrame(string sessionId, SubmitFrameRequest request);
    SessionStatusResponse GetStatus(string sessionId);
}
