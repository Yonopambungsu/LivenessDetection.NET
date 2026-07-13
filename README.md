#Demo liveness detection using .Net
"http://yoono.tryasp.net/demo.html"

# Required AI Models

This project requires several pre-trained ONNX models for face detection, facial landmark detection, face recognition, and anti-spoofing.

> **Important**
>
> Due to the licensing terms of the original model providers, the ONNX model files are **not included** in this repository.
>
> Please download the models from their respective official sources and place them in the folders shown below.

## Folder Structure

Place the models as follows:

```
AIModels/
├── detection/
│   └── scrfd_10g_bnkps.onnx
│
├── landmark/
│   └── 2d106det.onnx
│
├── recognition/
│   └── glintr100.onnx
│
└── spoof/
    └── MiniFASNetV2.onnx
```

## Required Models

| Model | Purpose | Destination Folder |
|--------|---------|--------------------|
| `scrfd_10g_bnkps.onnx` | Face Detection | `AIModels/detection/` |
| `2d106det.onnx` | 106 Facial Landmark Detection | `AIModels/landmark/` |
| `glintr100.onnx` | Face Recognition (ArcFace) | `AIModels/recognition/` |
| `MiniFASNetV2.onnx` | Face Anti-Spoofing | `AIModels/spoof/` |

## Example

After downloading the models, your project structure should look like this:

```
AIModels/
├── detection/
│   ├── ScrfdDetector.cs
│   ├── DetectedFace.cs
│   └── scrfd_10g_bnkps.onnx
│
├── landmark/
│   ├── Landmark106Detector.cs
│   └── 2d106det.onnx
│
├── recognition/
│   ├── ArcFaceRecognizer.cs
│   └── glintr100.onnx
│
├── spoof/
│   ├── AntiSpoofDetector.cs
│   └── MiniFASNetV2.onnx
│
└── Shared/
```

## Notes

- Keep the original model filenames unchanged.
- Do not rename or move the model files.
- The application loads the models using these relative paths.
- If a model cannot be found, the application will fail to initialize the corresponding detector.

## License

The source code in this repository is licensed under the repository's license.

The ONNX models are distributed under their respective licenses by their original authors. Please refer to the original model repositories for their licensing terms.
