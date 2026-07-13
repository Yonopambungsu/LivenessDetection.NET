using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Advanced;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace AIModels.Shared;

/// <summary>
/// Closed-form 2D similarity transform (uniform scale + rotation + translation, no shear/reflection),
/// fit by least squares. Equivalent to the non-reflective case of Umeyama's algorithm for 2D point sets,
/// used to align a detected face (via its keypoints) to a canonical template before recognition,
/// and to warp/crop an image region without relying on ImageSharp's higher-level affine APIs.
/// </summary>
public readonly struct SimilarityTransform
{
    public float A { get; }
    public float B { get; }
    public float Tx { get; }
    public float Ty { get; }

    public SimilarityTransform(float a, float b, float tx, float ty)
    {
        A = a;
        B = b;
        Tx = tx;
        Ty = ty;
    }

    /// <summary>
    /// Fits [x'; y'] = [[a,-b],[b,a]] * [x; y] + [tx; ty] minimizing squared error over the point pairs.
    /// </summary>
    public static SimilarityTransform Estimate(PointF[] src, PointF[] dst)
    {
        if (src.Length != dst.Length || src.Length < 2)
        {
            throw new ArgumentException("Need at least 2 matching point pairs of equal length.");
        }

        int n = src.Length;
        float sx = 0, sy = 0, sxp = 0, syp = 0;
        for (int i = 0; i < n; i++)
        {
            sx += src[i].X;
            sy += src[i].Y;
            sxp += dst[i].X;
            syp += dst[i].Y;
        }

        float meanX = sx / n, meanY = sy / n, meanXp = sxp / n, meanYp = syp / n;

        float num1 = 0, num2 = 0, den = 0;
        for (int i = 0; i < n; i++)
        {
            float xc = src[i].X - meanX;
            float yc = src[i].Y - meanY;
            float xpc = dst[i].X - meanXp;
            float ypc = dst[i].Y - meanYp;

            num1 += xc * xpc + yc * ypc;
            num2 += xc * ypc - yc * xpc;
            den += xc * xc + yc * yc;
        }

        float a = den > 1e-8f ? num1 / den : 1f;
        float b = den > 1e-8f ? num2 / den : 0f;
        float tx = meanXp - a * meanX + b * meanY;
        float ty = meanYp - b * meanX - a * meanY;

        return new SimilarityTransform(a, b, tx, ty);
    }

    public PointF Apply(PointF p)
    {
        return new PointF(A * p.X - B * p.Y + Tx, B * p.X + A * p.Y + Ty);
    }

    public PointF ApplyInverse(PointF p)
    {
        float det = A * A + B * B;
        if (det < 1e-12f) return p;
        float dx = p.X - Tx;
        float dy = p.Y - Ty;
        return new PointF((A * dx + B * dy) / det, (-B * dx + A * dy) / det);
    }

    /// <summary>
    /// Warps <paramref name="src"/> so that points mapped by this transform land at their destination
    /// positions, producing an <paramref name="outW"/>x<paramref name="outH"/> output image. Implemented
    /// as inverse-mapping + bilinear sampling (equivalent to cv2.warpAffine with this forward matrix).
    /// </summary>
    public Image<Rgb24> WarpTo(Image<Rgb24> src, int outW, int outH)
    {
        var result = new Image<Rgb24>(outW, outH);
        var self = this;
        src.ProcessPixelRows(accessor =>
        {
            int srcW = src.Width, srcH = src.Height;
            for (int oy = 0; oy < outH; oy++)
            {
                var row = result.DangerousGetPixelRowMemory(oy).Span;
                for (int ox = 0; ox < outW; ox++)
                {
                    var sp = self.ApplyInverse(new PointF(ox, oy));
                    row[ox] = SampleBilinear(accessor, srcW, srcH, sp.X, sp.Y);
                }
            }
        });
        return result;
    }

    private static Rgb24 SampleBilinear(PixelAccessor<Rgb24> accessor, int srcW, int srcH, float x, float y)
    {
        x = Math.Clamp(x, 0, srcW - 1.001f);
        y = Math.Clamp(y, 0, srcH - 1.001f);

        int x0 = (int)x, y0 = (int)y;
        int x1 = Math.Min(x0 + 1, srcW - 1);
        int y1 = Math.Min(y0 + 1, srcH - 1);
        float fx = x - x0, fy = y - y0;

        var p00 = accessor.GetRowSpan(y0)[x0];
        var p10 = accessor.GetRowSpan(y0)[x1];
        var p01 = accessor.GetRowSpan(y1)[x0];
        var p11 = accessor.GetRowSpan(y1)[x1];

        byte Lerp(byte a, byte b, byte c, byte d)
        {
            float top = a + (b - a) * fx;
            float bottom = c + (d - c) * fx;
            return (byte)Math.Round(top + (bottom - top) * fy);
        }

        return new Rgb24(
            Lerp(p00.R, p10.R, p01.R, p11.R),
            Lerp(p00.G, p10.G, p01.G, p11.G),
            Lerp(p00.B, p10.B, p01.B, p11.B));
    }
}
