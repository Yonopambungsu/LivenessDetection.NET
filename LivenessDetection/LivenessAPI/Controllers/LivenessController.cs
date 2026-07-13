using LivenessAPI.Application;
using LivenessAPI.Application.Dtos;
using Microsoft.AspNetCore.Mvc;

namespace LivenessAPI.Controllers;

[ApiController]
[Route("api/liveness")]
public sealed class LivenessController(ILivenessSessionService sessionService) : ControllerBase
{
    /// <summary>Starts a liveness session from a reference photo (e.g. an ID photo) and returns the
    /// first randomized challenge instruction the client should ask the user to perform.</summary>
    [HttpPost("session/start")]
    public IActionResult StartSession([FromBody] StartSessionRequest request)
    {
        if (string.IsNullOrWhiteSpace(request.ReferenceImageBase64))
        {
            return BadRequest(new { error = "referenceImageBase64 is required." });
        }

        try
        {
            return Ok(sessionService.StartSession(request));
        }
        catch (LivenessValidationException ex)
        {
            return BadRequest(new { error = ex.Message });
        }
    }

    /// <summary>Submits one camera frame for the current step of the flow. Call repeatedly (e.g. a
    /// few times per second) while the client shows the user the current challenge instruction.</summary>
    [HttpPost("session/{sessionId}/frame")]
    public IActionResult SubmitFrame(string sessionId, [FromBody] SubmitFrameRequest request)
    {
        if (string.IsNullOrWhiteSpace(request.ImageBase64))
        {
            return BadRequest(new { error = "imageBase64 is required." });
        }

        try
        {
            return Ok(sessionService.SubmitFrame(sessionId, request));
        }
        catch (LivenessSessionNotFoundException ex)
        {
            return NotFound(new { error = ex.Message });
        }
        catch (LivenessValidationException ex)
        {
            return BadRequest(new { error = ex.Message });
        }
    }

    /// <summary>Read-only snapshot of a session's progress, useful for polling/debugging.</summary>
    [HttpGet("session/{sessionId}/status")]
    public IActionResult GetStatus(string sessionId)
    {
        try
        {
            return Ok(sessionService.GetStatus(sessionId));
        }
        catch (LivenessSessionNotFoundException ex)
        {
            return NotFound(new { error = ex.Message });
        }
    }
}
