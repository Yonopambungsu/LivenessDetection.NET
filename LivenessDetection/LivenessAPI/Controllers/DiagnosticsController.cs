using AIModels.Detection;
using AIModels.Landmark;
using AIModels.Shared;
using AIModels.Spoof;
using Microsoft.AspNetCore.Mvc;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace LivenessAPI.Controllers;

public sealed record LandmarkPoint(int Index, float X, float Y);
public sealed record LandmarkDebugResponse(float X1, float Y1, float X2, float Y2, IReadOnlyList<LandmarkPoint> Points);
public sealed record AntiSpoofDebugResponse(string CropPreviewBase64, IReadOnlyList<AntiSpoofVariant> Variants);
public sealed record AntiSpoofVariant(string Normalization, bool SwapToBgr, float[] Probabilities);

/// <summary>
/// Not part of the liveness flow itself — a calibration aid for verifying/correcting
/// Landmark68Indices and the AntiSpoofDetector's crop-scale/real-class-index assumptions (see the
/// TODOs there) against real photos, since neither could be confirmed from documentation alone.
/// </summary>
[ApiController]
[Route("api/liveness/debug")]
public sealed class DiagnosticsController(YunetDetector detector, Pipnet68Detector landmarkDetector, AntiSpoofDetector antiSpoofDetector) : ControllerBase
{
    public sealed record DebugRequest(string ImageBase64);

    private static Image<Rgb24> DecodeImage(string imageBase64)
    {
        var payload = imageBase64;
        var commaIndex = payload.IndexOf(',');
        if (payload.StartsWith("data:", StringComparison.OrdinalIgnoreCase) && commaIndex >= 0)
        {
            payload = payload[(commaIndex + 1)..];
        }

        return Image.Load<Rgb24>(Convert.FromBase64String(payload));
    }

    [HttpGet("antispoof-model-info")]
    public IActionResult GetAntiSpoofModelInfo() => Ok(new { description = antiSpoofDetector.DescribeModel() });

    [HttpPost("landmarks")]
    public IActionResult GetLandmarks([FromBody] DebugRequest request)
    {
        using var image = DecodeImage(request.ImageBase64);
        var faces = detector.Detect(image);
        if (faces.Count == 0)
        {
            return BadRequest(new { error = "No face detected." });
        }

        var face = faces[0];
        var points = landmarkDetector.GetLandmarks(image, face);

        return Ok(new LandmarkDebugResponse(
            face.X1, face.Y1, face.X2, face.Y2,
            points.Select((p, i) => new LandmarkPoint(i, p.X, p.Y)).ToList()));
    }

    private static readonly float[] ImageNetMean = { 0.485f, 0.456f, 0.406f };
    private static readonly float[] ImageNetStd = { 0.229f, 0.224f, 0.225f };

    private static List<AntiSpoofVariant> RunNormalizationVariants(AntiSpoofDetector antiSpoofDetector, Image<Rgb24> image, DetectedFace face, float cropScale)
    {
        var variants = new List<AntiSpoofVariant>();
        foreach (var swapToBgr in new[] { true, false })
        {
            variants.Add(new AntiSpoofVariant("raw/255", swapToBgr,
                antiSpoofDetector.PredictRawProbabilities(image, face, cropScale, swapToBgr, null, null)));
            variants.Add(new AntiSpoofVariant("imagenet", swapToBgr,
                antiSpoofDetector.PredictRawProbabilities(image, face, cropScale, swapToBgr, ImageNetMean, ImageNetStd)));
        }
        return variants;
    }

    /// <summary>Runs anti-spoofing across normalization-scheme x channel-order combinations and
    /// reports the raw per-class probabilities for each, plus a preview of the actual crop fed to the
    /// model — so a degenerate crop (wrong region, all-background) is visible instead of guessed at.</summary>
    [HttpPost("antispoof")]
    public IActionResult GetAntiSpoof([FromBody] DebugRequest request)
    {
        using var image = DecodeImage(request.ImageBase64);
        var faces = detector.Detect(image);
        if (faces.Count == 0)
        {
            return BadRequest(new { error = "No face detected." });
        }

        var face = faces[0];
        var variants = RunNormalizationVariants(antiSpoofDetector, image, face, 2.7f);

        using var previewCrop = BuildCropPreview(image, face, 2.7f);
        using var ms = new MemoryStream();
        previewCrop.SaveAsPng(ms);
        var previewBase64 = Convert.ToBase64String(ms.ToArray());

        return Ok(new AntiSpoofDebugResponse(previewBase64, variants));
    }

    /// <summary>Runs anti-spoofing on the WHOLE image (no face detection) so arbitrary test images
    /// (noise, blank, non-face) can be probed — isolates whether the model responds to input content
    /// at all, or saturates to one class regardless of what's fed in.</summary>
    [HttpPost("antispoof-raw")]
    public IActionResult GetAntiSpoofRaw([FromBody] DebugRequest request)
    {
        using var image = DecodeImage(request.ImageBase64);
        var wholeImageFace = new DetectedFace
        {
            X1 = 0, Y1 = 0, X2 = image.Width, Y2 = image.Height, Score = 1f,
            Keypoints = new SixLabors.ImageSharp.PointF[5],
        };

        var variants = RunNormalizationVariants(antiSpoofDetector, image, wholeImageFace, 1f);
        return Ok(new { variants });
    }

    private static Image<Rgb24> BuildCropPreview(Image<Rgb24> image, DetectedFace face, float scale)
    {
        float boxW = face.Width, boxH = face.Height;
        float x = face.X1, y = face.Y1;
        int srcW = image.Width, srcH = image.Height;

        scale = Math.Min((srcH - 1) / boxH, Math.Min((srcW - 1) / boxW, scale));
        float newW = boxW * scale, newH = boxH * scale;
        float centerX = boxW / 2f + x, centerY = boxH / 2f + y;
        float left = centerX - newW / 2f, top = centerY - newH / 2f;
        float right = centerX + newW / 2f, bottom = centerY + newH / 2f;

        if (left < 0) { right -= left; left = 0; }
        if (top < 0) { bottom -= top; top = 0; }
        if (right > srcW - 1) { left -= right - (srcW - 1); right = srcW - 1; }
        if (bottom > srcH - 1) { top -= bottom - (srcH - 1); bottom = srcH - 1; }

        using var crop = TensorUtils.CropWithPadding(image, (int)left, (int)top, (int)right, (int)bottom);
        return crop.Clone(ctx => ctx.Resize(160, 160));
    }
}
