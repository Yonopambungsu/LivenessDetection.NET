using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace AIModels.Shared;

/// <summary>
/// Shared image -> NCHW tensor conversion for the ONNX models in this project. ImageSharp decodes
/// pixels in R,G,B order; most of these models were exported from OpenCV pipelines using
/// cv2.dnn.blobFromImage(..., swapRB=True), which also produces R,G,B channel order, so no channel
/// swap is needed for them. MiniFASNetV2 (anti-spoof) is the exception: its reference implementation
/// feeds OpenCV's native B,G,R order, so <see cref="ToNchwTensor"/> exposes a swapToBgr flag for it.
/// </summary>
public static class TensorUtils
{
    public static DenseTensor<float> ToNchwTensor(Image<Rgb24> image, float mean, float std, bool swapToBgr = false)
    {
        int w = image.Width, h = image.Height;
        var tensor = new DenseTensor<float>(new[] { 1, 3, h, w });

        image.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < h; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < w; x++)
                {
                    var px = row[x];
                    float r = (px.R - mean) / std;
                    float g = (px.G - mean) / std;
                    float b = (px.B - mean) / std;

                    if (swapToBgr)
                    {
                        tensor[0, 0, y, x] = b;
                        tensor[0, 1, y, x] = g;
                        tensor[0, 2, y, x] = r;
                    }
                    else
                    {
                        tensor[0, 0, y, x] = r;
                        tensor[0, 1, y, x] = g;
                        tensor[0, 2, y, x] = b;
                    }
                }
            }
        });

        return tensor;
    }

    /// <summary>Per-channel mean/std variant (e.g. ImageNet normalization), for calibration testing
    /// against models whose training normalization isn't documented.</summary>
    public static DenseTensor<float> ToNchwTensorPerChannel(Image<Rgb24> image, float[] mean, float[] std, bool swapToBgr = false)
    {
        int w = image.Width, h = image.Height;
        var tensor = new DenseTensor<float>(new[] { 1, 3, h, w });

        image.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < h; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < w; x++)
                {
                    var px = row[x];
                    float r = (px.R / 255f - mean[0]) / std[0];
                    float g = (px.G / 255f - mean[1]) / std[1];
                    float b = (px.B / 255f - mean[2]) / std[2];

                    if (swapToBgr)
                    {
                        tensor[0, 0, y, x] = b;
                        tensor[0, 1, y, x] = g;
                        tensor[0, 2, y, x] = r;
                    }
                    else
                    {
                        tensor[0, 0, y, x] = r;
                        tensor[0, 1, y, x] = g;
                        tensor[0, 2, y, x] = b;
                    }
                }
            }
        });

        return tensor;
    }

    /// <summary>
    /// Resizes <paramref name="image"/> to fit within a square canvas of <paramref name="size"/>
    /// preserving aspect ratio, then pads (top-left aligned) with black. Returns the padded image
    /// and the scale factor applied, needed to map detections back to original coordinates.
    /// </summary>
    public static (Image<Rgb24> Padded, float Scale) ResizeAndPad(Image<Rgb24> image, int size)
    {
        float ratio = (float)image.Height / image.Width;
        int newW, newH;
        if (ratio > 1f)
        {
            newH = size;
            newW = (int)Math.Round(size / ratio);
        }
        else
        {
            newW = size;
            newH = (int)Math.Round(size * ratio);
        }

        float scale = (float)newH / image.Height;

        var resized = image.Clone(ctx => ctx.Resize(newW, newH));
        var canvas = new Image<Rgb24>(size, size);
        canvas.Mutate(ctx => ctx.DrawImage(resized, Point.Empty, 1f));
        resized.Dispose();

        return (canvas, scale);
    }

    public static Image<Rgb24> CropWithPadding(Image<Rgb24> image, int x0, int y0, int x1, int y1)
    {
        int srcW = image.Width, srcH = image.Height;
        int cropW = x1 - x0, cropH = y1 - y0;
        var result = new Image<Rgb24>(cropW, cropH);

        int clampedX0 = Math.Max(0, x0);
        int clampedY0 = Math.Max(0, y0);
        int clampedX1 = Math.Min(srcW, x1);
        int clampedY1 = Math.Min(srcH, y1);

        if (clampedX1 <= clampedX0 || clampedY1 <= clampedY0)
        {
            return result;
        }

        var region = new Rectangle(clampedX0, clampedY0, clampedX1 - clampedX0, clampedY1 - clampedY0);
        using var sub = image.Clone(ctx => ctx.Crop(region));
        result.Mutate(ctx => ctx.DrawImage(sub, new Point(clampedX0 - x0, clampedY0 - y0), 1f));

        return result;
    }
}
