# Bundled model licenses and attribution

SpiderVision bundles three **third-party model artifacts**. They are not relicensed by SpiderVision's own source-code license. Keep the applicable license text and attribution when redistributing the model files.

| Bundled artifact | Upstream project and model | License text included here |
| --- | --- | --- |
| `face_detection_yunet_2023mar.onnx` | [OpenCV Zoo — YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) | [YuNet MIT license](licenses/YUNET-MIT.txt), copied verbatim from its [upstream LICENSE](https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/LICENSE). Its original Shiqi Yu copyright notice is preserved. |
| `face_recognition_sface_2021dec.onnx` | [OpenCV Zoo — SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) | [Apache License 2.0](licenses/APACHE-2.0.txt), copied verbatim from its [upstream LICENSE](https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/LICENSE). |
| `vosk-model-small-en-us-0.15/` | [Alpha Cephei Vosk model catalog](https://alphacephei.com/vosk/models) and [model download](https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip) | The catalog lists this particular model as **Apache 2.0**. The [Apache License 2.0](licenses/APACHE-2.0.txt) is the same complete text as the upstream SFace license and [the Apache Software Foundation's copy](https://www.apache.org/licenses/LICENSE-2.0.txt). Keep the model's own `README`, which credits Copyright 2020 Alpha Cephei Inc. |

These are **license copies and notices for the models**, not a claim that SpiderVision authored or owns the models. The educational anatomy artwork has different terms; see [`../anatomy/ATTRIBUTION.md`](../anatomy/ATTRIBUTION.md). A face enrolled through the app produces a separate, private `.face_identity.npz` file in the application directory; it is **not** one of these public models and must not be committed or redistributed.
