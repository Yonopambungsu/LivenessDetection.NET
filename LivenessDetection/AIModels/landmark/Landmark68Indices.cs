namespace AIModels.Landmark;

/// <summary>
/// Named indices into the standard IBUG 68-point facial landmark scheme, as produced by
/// PIPNet (pipnet_r18_300w_celeba_68.onnx).
///
/// Convention: "Left"/"Right" refers to the IMAGE side (camera perspective), NOT the
/// subject's anatomical side — consistent with the former Landmark106Indices naming:
///   • "LeftEye"  = the eye on the LEFT side of the image  (subject's anatomical RIGHT eye, IBUG 36–41)
///   • "RightEye" = the eye on the RIGHT side of the image (subject's anatomical LEFT eye,  IBUG 42–47)
///
/// IBUG 68-point layout reference:
///   0–16   Jawline / contour
///   17–21  Right eyebrow (image-right)
///   22–26  Left eyebrow (image-left)
///   27–30  Nose bridge
///   31–35  Nose base / nostrils
///   36–41  Left eye in image (subject's anatomical right)
///   42–47  Right eye in image (subject's anatomical left)
///   48–59  Outer lip
///   60–67  Inner lip
/// </summary>
public static class Landmark68Indices
{
    // ── Nose ──────────────────────────────────────────────────────────────────
    /// <summary>Tip of the nose (IBUG point 30).</summary>
    public const int NoseTip = 30;

    // ── Left eye (image-left = subject's anatomical RIGHT eye, IBUG 36–41) ────
    /// <summary>Outer corner of the left eye – leftmost pixel (IBUG 36).</summary>
    public const int LeftEyeLeftCorner = 36;

    /// <summary>Inner corner of the left eye – rightmost pixel (IBUG 39).</summary>
    public const int LeftEyeRightCorner = 39;

    /// <summary>Upper eyelid center of the left eye (IBUG 38).</summary>
    public const int LeftEyeTop = 38;

    /// <summary>Lower eyelid center of the left eye (IBUG 40).</summary>
    public const int LeftEyeBottom = 40;

    // ── Right eye (image-right = subject's anatomical LEFT eye, IBUG 42–47) ──
    /// <summary>Inner corner of the right eye – leftmost pixel (IBUG 42).</summary>
    public const int RightEyeLeftCorner = 42;

    /// <summary>Outer corner of the right eye – rightmost pixel (IBUG 45).</summary>
    public const int RightEyeRightCorner = 45;

    /// <summary>Upper eyelid center of the right eye (IBUG 44).</summary>
    public const int RightEyeTop = 44;

    /// <summary>Lower eyelid center of the right eye (IBUG 46).</summary>
    public const int RightEyeBottom = 46;

    // ── Mouth (IBUG outer-lip 48–59) ──────────────────────────────────────────
    /// <summary>Left corner of the mouth in the image (IBUG 48).</summary>
    public const int MouthLeftCorner = 48;

    /// <summary>Right corner of the mouth in the image (IBUG 54).</summary>
    public const int MouthRightCorner = 54;

    /// <summary>Top center of the upper lip (IBUG 51).</summary>
    public const int MouthTop = 51;

    /// <summary>Bottom center of the lower lip (IBUG 57).</summary>
    public const int MouthBottom = 57;
}
