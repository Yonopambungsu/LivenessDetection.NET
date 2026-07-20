using AIModels.Detection;
using AIModels.Shared;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace AIModels.Landmark;

/// <summary>
/// PIPNet (Pixel-in-Pixel Net) 68-point landmark detector (pipnet_r18_300w_celeba_68.onnx).
///
/// Architecture: ResNet-18 backbone with PIP head. The model is trained on 300W + CelebA datasets
/// with semi-supervised learning and outputs 68 IBUG-compatible facial landmark points.
///
/// Preprocessing: crop face with asymmetric 10% padding (matches the reference yakhyo/pipnet-onnx
/// decode exactly: left/right expand by 10% of bbox width, top shrinks inward by 10% of bbox height,
/// bottom expands by 10% of bbox height - NOT a symmetric square crop), resize to 256x256, ImageNet
/// per-channel normalization (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]).
///
/// Output - 5 tensors:
///   cls_map  [1,68,8,8]  - per-cell classification heatmap (which grid cell contains each landmark)
///   offset_x [1,68,8,8]  - sub-cell x offset in [0,1] range, read at that landmark's own peak cell
///   offset_y [1,68,8,8]  - sub-cell y offset in [0,1] range
///   nb_x     [1,680,8,8] - 680 = 68 landmarks x 10 neighbors. Channel (i*10+j) holds landmark i's
///                          prediction of where its j-th nearest mean-face neighbor sits, read at
///                          landmark i's OWN peak cell (not the neighbor's).
///   nb_y     [1,680,8,8] - same layout as nb_x for the y axis.
///
/// Decoding (pixel-in-pixel algorithm, ported from PIPNetONNX._decode in yakhyo/pipnet-onnx):
///   1. For each landmark i: argmax(cls_map[i]) -> (mx, my) grid cell, then
///      predX[i] = (mx + offset_x[i,my,mx]) / GridSize, predY[i] likewise - its own direct prediction.
///   2. For each landmark i and neighbor slot j in [0,10): nbPredX[i,j] = (mx + nb_x[i*10+j,my,mx]) /
///      GridSize (still landmark i's own (mx,my) - the neighbor prediction is made FROM there).
///   3. Every landmark k is voted on by whichever (i,j) pairs have k as their j-th nearest mean-face
///      neighbor (a fixed geometric relationship, independent of any single frame). The final position
///      for k is the mean of predX[k]/predY[k] plus every nbPredX[i,j]/nbPredY[i,j] that voted for k.
///      Which (i,j) pairs vote for which k is precomputed once from the mean-face point cloud (a
///      k-nearest-neighbors query, num_nb=10) - see <see cref="BuildReverseIndex"/> - since it depends
///      only on the fixed mean-face geometry, not on any particular frame.
///   4. The merged [0,1]-normalized (x,y) is scaled by the crop's pixel width/height and shifted by
///      the crop's top-left corner to land in original-image coordinates.
///
/// Landmark indices follow the standard IBUG 68-point scheme (see <see cref="Landmark68Indices"/>).
/// </summary>
public sealed class Pipnet68Detector : IDisposable
{
    private const int InputSize = 256;
    private const int GridSize  = 8;   // InputSize / 32 (backbone stride)
    private const int NumPoints = 68;
    private const int NumNeighbors = 10;

    // ImageNet per-channel mean and std (values normalized to [0,1] first).
    private static readonly float[] Mean = { 0.485f, 0.456f, 0.406f };
    private static readonly float[] Std  = { 0.229f, 0.224f, 0.225f };

    // Mean-face 68-point (300W) layout, normalized to [0,1]. Vendored from upstream PIPNet (MIT) via
    // yakhyo/pipnet-onnx's model/meanface.py - this is fixed reference geometry (not learned per-model
    // weights), used purely to precompute which landmarks are each other's nearest neighbors.
    private static readonly float[] MeanFace68 =
    {
        0.05558998895410058f, 0.23848280098218655f, 0.05894856684324656f, 0.3590187767402909f,
        0.0736574254414371f, 0.4792196439871159f, 0.09980016420365162f, 0.5959029676167197f,
        0.14678670154995865f, 0.7035615597409001f, 0.21847188218752928f, 0.7971705893013413f,
        0.30554692814599393f, 0.8750572978073209f, 0.4018434142644611f, 0.9365018059444535f,
        0.5100536090382116f, 0.9521295666029498f, 0.6162039414413925f, 0.9309467340899419f,
        0.7094522484942942f, 0.8669275031738761f, 0.7940993502957612f, 0.7879369615524398f,
        0.8627063649669019f, 0.6933756633633967f, 0.9072386130534111f, 0.5836975017700834f,
        0.9298874997796132f, 0.4657004930314701f, 0.9405202670724796f, 0.346063993805527f,
        0.9425419553088846f, 0.22558131891345742f, 0.13304298285530403f, 0.14853071838028062f,
        0.18873587368440375f, 0.09596491613770254f, 0.2673231915839219f, 0.08084218279128136f,
        0.34878638553224905f, 0.09253591849498964f, 0.4226713753717798f, 0.12466063383809506f,
        0.5618513152452376f, 0.11839668911898667f, 0.6394952560845826f, 0.08480191391770678f,
        0.7204375851516752f, 0.07249669092117161f, 0.7988615904537885f, 0.08766933146893043f,
        0.8534884939460948f, 0.1380096813348583f, 0.49610677423740546f, 0.21516740699375395f,
        0.49709661403980665f, 0.2928875699060973f, 0.4982292618461611f, 0.3699985379939941f,
        0.49982965173254235f, 0.4494119144493957f, 0.406772397599095f, 0.5032397294041786f,
        0.45231994786363067f, 0.5197953144002292f, 0.49969685987914064f, 0.5332489262413073f,
        0.5470074224053442f, 0.518413595827126f, 0.5892261151542287f, 0.5023530079850803f,
        0.22414578747180394f, 0.22835847349949062f, 0.27262947128194215f, 0.19915251892241678f,
        0.3306759252861797f, 0.20026034220607236f, 0.38044435864341913f, 0.23839196034290633f,
        0.32884072789429913f, 0.24902443794896897f, 0.2707409300714473f, 0.24950886025380967f,
        0.6086826011068529f, 0.23465048639345917f, 0.660397116846103f, 0.1937087938594717f,
        0.7177815187666494f, 0.19317079039835858f, 0.7652328176062365f, 0.22088822845258235f,
        0.722727677909097f, 0.24195514178450958f, 0.6658378927310327f, 0.2441554205021945f,
        0.32894370935769124f, 0.6496589505331646f, 0.39347179739100613f, 0.6216899667490776f,
        0.4571976492475472f, 0.60794251109236f, 0.4990484623797022f, 0.6190124015360254f,
        0.5465555522325872f, 0.6071477960565326f, 0.6116127327356168f, 0.6205387097430033f,
        0.6742318496058836f, 0.6437466364395467f, 0.6144773141699744f, 0.7077526646009754f,
        0.5526442055374252f, 0.7363350735898412f, 0.5018120662554302f, 0.7424476622366345f,
        0.4554458875556401f, 0.7382303858617719f, 0.3923750731597415f, 0.7118887028663435f,
        0.35530766372404593f, 0.6524479416354049f, 0.457111071610868f, 0.6467108367268608f,
        0.49974082228815025f, 0.6508406774477011f, 0.5477027224368399f, 0.6451242819422733f,
        0.6478392760505715f, 0.647852382880368f, 0.5488474760115958f, 0.6779061893042735f,
        0.5001073351044452f, 0.6845280260362221f, 0.4564831746654594f, 0.6799300301441035f,
    };

    // Precomputed once (fixed mean-face geometry, independent of any frame/model weights): for target
    // landmark k, ReverseIndex1[k*MaxLen+m]/ReverseIndex2[k*MaxLen+m] give the (sourceLandmark,
    // neighborSlot) pair of the m-th vote for k, for m in [0, MaxLen).
    private static readonly int[] ReverseIndex1;
    private static readonly int[] ReverseIndex2;
    private static readonly int MaxLen;

    static Pipnet68Detector()
    {
        (ReverseIndex1, ReverseIndex2, MaxLen) = BuildReverseIndex(MeanFace68, NumPoints, NumNeighbors);
    }

    private readonly InferenceSession _session;
    private readonly string _inputName;

    public Pipnet68Detector(string modelPath)
    {
        _session = new InferenceSession(modelPath);
        _inputName = _session.InputMetadata.Keys.First();
    }

    public PointF[] GetLandmarks(Image<Rgb24> image, DetectedFace face)
    {
        // ── 1. Crop with asymmetric 10% padding (matches the reference _crop_face exactly) ──────────
        float detW = face.Width;
        float detH = face.Height;
        const float pad = 0.1f;

        int cropX1 = (int)(face.X1 - detW * pad);
        int cropY1 = (int)(face.Y1 + detH * pad); // shrinks inward from the top, not a symmetric expand
        int cropX2 = (int)(face.X2 + detW * pad);
        int cropY2 = (int)(face.Y2 + detH * pad);

        int cropW = cropX2 - cropX1;
        int cropH = cropY2 - cropY1;

        using var cropped = TensorUtils.CropWithPadding(image, cropX1, cropY1, cropX2, cropY2);
        using var resized = cropped.Clone(ctx => ctx.Resize(InputSize, InputSize));

        // ── 2. Build NCHW input tensor with ImageNet normalization ─────────────────
        var tensor = BuildInputTensor(resized);

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_inputName, tensor)
        };

        // ── 3. Inference ───────────────────────────────────────────────────────────
        using var results = _session.Run(inputs);

        var outputMap = results.ToDictionary(
            r => r.Name,
            r => r.AsTensor<float>().ToArray());

        var clsMap  = outputMap["cls_map"];
        var offsetX = outputMap["offset_x"];
        var offsetY = outputMap["offset_y"];
        var nbX     = outputMap["nb_x"];
        var nbY     = outputMap["nb_y"];

        int cellCount = GridSize * GridSize;

        // ── 4a. Each landmark's own direct prediction + its 10 neighbor-slot predictions ───────────
        var predX = new float[NumPoints];
        var predY = new float[NumPoints];
        var nbPredX = new float[NumPoints, NumNeighbors];
        var nbPredY = new float[NumPoints, NumNeighbors];

        for (int i = 0; i < NumPoints; i++)
        {
            int baseOffset = i * cellCount;
            int maxCell = 0;
            float maxVal = float.MinValue;
            for (int cell = 0; cell < cellCount; cell++)
            {
                float v = clsMap[baseOffset + cell];
                if (v > maxVal) { maxVal = v; maxCell = cell; }
            }

            int mx = maxCell % GridSize;
            int my = maxCell / GridSize;
            int cellIdx = baseOffset + maxCell;

            predX[i] = (mx + offsetX[cellIdx]) / GridSize;
            predY[i] = (my + offsetY[cellIdx]) / GridSize;

            for (int j = 0; j < NumNeighbors; j++)
            {
                int nbCellIdx = (i * NumNeighbors + j) * cellCount + maxCell;
                nbPredX[i, j] = (mx + nbX[nbCellIdx]) / GridSize;
                nbPredY[i, j] = (my + nbY[nbCellIdx]) / GridSize;
            }
        }

        // ── 4b. Fuse: each landmark's final position = mean of its own prediction plus every
        // neighbor-slot prediction that was trained to vote for it ─────────────────────────────────
        var points = new PointF[NumPoints];
        for (int k = 0; k < NumPoints; k++)
        {
            float sumX = predX[k];
            float sumY = predY[k];
            int count = 1;

            for (int m = 0; m < MaxLen; m++)
            {
                int src  = ReverseIndex1[k * MaxLen + m];
                int slot = ReverseIndex2[k * MaxLen + m];
                sumX += nbPredX[src, slot];
                sumY += nbPredY[src, slot];
                count++;
            }

            float lx = sumX / count;
            float ly = sumY / count;

            // ── 5. Normalized [0,1] crop-local -> original-image pixel coordinates ─────────────────
            points[k] = new PointF(lx * cropW + cropX1, ly * cropH + cropY1);
        }

        return points;
    }

    /// <summary>
    /// Converts the 256x256 crop to a 1x3x256x256 NCHW tensor with ImageNet normalization:
    /// (pixel / 255 - mean) / std in R,G,B channel order.
    /// </summary>
    private static DenseTensor<float> BuildInputTensor(Image<Rgb24> image)
    {
        var tensor = new DenseTensor<float>(new[] { 1, 3, InputSize, InputSize });
        image.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < InputSize; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < InputSize; x++)
                {
                    var px = row[x];
                    tensor[0, 0, y, x] = (px.R / 255f - Mean[0]) / Std[0];
                    tensor[0, 1, y, x] = (px.G / 255f - Mean[1]) / Std[1];
                    tensor[0, 2, y, x] = (px.B / 255f - Mean[2]) / Std[2];
                }
            }
        });
        return tensor;
    }

    /// <summary>
    /// Ports PIPNetONNX's _build_neighbor_indices: for each landmark, find its num_nb nearest other
    /// landmarks by Euclidean distance in the mean-face point cloud, then invert that mapping so each
    /// landmark knows which (sourceLandmark, neighborSlot) pairs voted for it. Padded so every landmark
    /// has exactly MaxLen entries (short lists cycle from the start, matching the reference's
    /// list-repeat-then-slice padding).
    /// </summary>
    private static (int[] Index1, int[] Index2, int MaxLen) BuildReverseIndex(float[] meanFaceXY, int numLandmarks, int numNeighbors)
    {
        var neighborsPerLandmark = new int[numLandmarks][];
        for (int i = 0; i < numLandmarks; i++)
        {
            float px = meanFaceXY[i * 2], py = meanFaceXY[i * 2 + 1];
            var order = Enumerable.Range(0, numLandmarks)
                .OrderBy(j =>
                {
                    float dx = px - meanFaceXY[j * 2];
                    float dy = py - meanFaceXY[j * 2 + 1];
                    return dx * dx + dy * dy;
                })
                .ToArray();
            // order[0] is always i itself (distance 0); skip it.
            neighborsPerLandmark[i] = order.Skip(1).Take(numNeighbors).ToArray();
        }

        var reverseLists = new List<(int SrcLandmark, int NeighborSlot)>[numLandmarks];
        for (int i = 0; i < numLandmarks; i++) reverseLists[i] = new List<(int, int)>();

        for (int i = 0; i < numLandmarks; i++)
        {
            for (int j = 0; j < numNeighbors; j++)
            {
                int target = neighborsPerLandmark[i][j];
                reverseLists[target].Add((i, j));
            }
        }

        int maxLen = reverseLists.Max(l => l.Count);
        var index1 = new int[numLandmarks * maxLen];
        var index2 = new int[numLandmarks * maxLen];

        for (int i = 0; i < numLandmarks; i++)
        {
            var list = reverseLists[i];
            for (int m = 0; m < maxLen; m++)
            {
                var (src, slot) = list[m % list.Count];
                index1[i * maxLen + m] = src;
                index2[i * maxLen + m] = slot;
            }
        }

        return (index1, index2, maxLen);
    }

    public void Dispose() => _session.Dispose();
}
