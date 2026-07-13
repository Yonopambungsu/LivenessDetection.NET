namespace LivenessAPI.Application;

public sealed class LivenessValidationException(string message) : Exception(message);

public sealed class LivenessSessionNotFoundException(string sessionId)
    : Exception($"Liveness session '{sessionId}' was not found or has expired.");
