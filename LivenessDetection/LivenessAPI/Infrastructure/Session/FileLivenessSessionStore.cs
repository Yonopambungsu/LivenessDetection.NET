using System.Text.Json;
using LivenessAPI.Application.Abstractions;
using LivenessAPI.Domain;

namespace LivenessAPI.Infrastructure.Session;

/// <summary>
/// Session store backed by one JSON file per session on local disk, rather than any in-process memory
/// structure (IMemoryCache, ConcurrentDictionary, etc.). On shared/budget hosting - this app is deployed
/// to MonsterASP.NET - the IIS app pool worker process can be recycled at any time: on an idle timeout,
/// a fixed schedule, or a private-memory limit trip. Any of those instantly wipes in-memory state
/// regardless of how careful that store's own eviction logic is, which surfaced as "session not found
/// or expired" moments into an otherwise-active verification. Sessions are short-lived (a few minutes)
/// verification attempts, not something that needs a real database - a small JSON file per session,
/// cleaned up on expiry, is the simplest fix that survives a worker-process restart.
/// </summary>
public sealed class FileLivenessSessionStore : ILivenessSessionStore
{
    private static readonly JsonSerializerOptions JsonOptions = new();
    private readonly string _directory;

    public FileLivenessSessionStore()
    {
        _directory = Path.Combine(AppContext.BaseDirectory, "App_Data", "liveness-sessions");
        Directory.CreateDirectory(_directory);
        CleanupExpiredFiles();
    }

    public void Save(LivenessSession session)
    {
        if (!IsValidSessionId(session.SessionId)) return;

        var path = PathFor(session.SessionId);
        var tempPath = path + ".tmp";
        File.WriteAllText(tempPath, JsonSerializer.Serialize(session, JsonOptions));
        File.Move(tempPath, path, overwrite: true);
    }

    public LivenessSession? Get(string sessionId)
    {
        // sessionId comes straight from the URL route value (client-supplied), so it must be validated
        // before ever touching the filesystem with it - otherwise a crafted value like "../../whatever"
        // could read or influence files outside the session-store directory (path traversal).
        if (!IsValidSessionId(sessionId)) return null;

        var path = PathFor(sessionId);
        if (!File.Exists(path)) return null;

        try
        {
            var session = JsonSerializer.Deserialize<LivenessSession>(File.ReadAllText(path), JsonOptions);
            if (session is null) return null;

            if (session.ExpiresAt < DateTimeOffset.UtcNow)
            {
                TryDelete(path);
                return null;
            }

            return session;
        }
        catch (IOException)
        {
            // Another request is mid-write to the same file (the per-session semaphore in
            // LivenessSessionService makes this rare, but not impossible under a web-garden config) -
            // treat as transiently unavailable rather than surfacing a 500 for one unlucky frame.
            return null;
        }
        catch (JsonException)
        {
            // Corrupted/partial file - e.g. a process kill mid-write. Discard rather than fail forever.
            TryDelete(path);
            return null;
        }
    }

    public void Remove(string sessionId)
    {
        if (!IsValidSessionId(sessionId)) return;
        TryDelete(PathFor(sessionId));
    }

    private string PathFor(string sessionId) => Path.Combine(_directory, $"{sessionId}.json");

    /// <summary>Valid session IDs are always Guid.ToString("N") - 32 lowercase hex characters - since
    /// that's the only thing StartSession ever generates. Anything else is rejected before it can reach
    /// a file-path operation.</summary>
    private static bool IsValidSessionId(string sessionId)
    {
        if (sessionId.Length != 32) return false;
        foreach (var c in sessionId)
        {
            bool isHex = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
            if (!isHex) return false;
        }
        return true;
    }

    private void CleanupExpiredFiles()
    {
        foreach (var file in Directory.EnumerateFiles(_directory, "*.json"))
        {
            try
            {
                var session = JsonSerializer.Deserialize<LivenessSession>(File.ReadAllText(file), JsonOptions);
                if (session is null || session.ExpiresAt < DateTimeOffset.UtcNow)
                {
                    TryDelete(file);
                }
            }
            catch
            {
                // Unreadable/corrupt leftover from a previous run - safe to discard on startup.
                TryDelete(file);
            }
        }
    }

    private static void TryDelete(string path)
    {
        try { File.Delete(path); } catch (IOException) { /* best-effort cleanup */ }
    }
}
