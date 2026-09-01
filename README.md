# Security Detection System

Real-time face recognition and face-occlusion detection running entirely on a **Raspberry Pi 5** (CPU-only inference with OpenVINO).

The system watches a live camera stream and, for every face it finds, answers two independent questions:

1. **Who is it?** — is this the enrolled *target* person, or someone else?
2. **Is the face visible?** — is the face uncovered (*safe*) or hidden behind a mask, hand, scarf or other occlusion (*unsafe*)?

The two answers are combined into a single label drawn on the video feed: `TARGET SAFE`, `TARGET UNSAFE`, `OTHER SAFE`, `OTHER UNSAFE`.

Course project for *Intelligent Consumer Technologies*, Università degli Studi di Milano-Bicocca.

---

## Pipeline

```
                        ┌─────────────────────────┐
  camera frame          │  YOLOv5n-face           │
  (rpicam-vid, MJPEG) ──►  face detection         │
                        │  letterbox + NMS        │
                        └───────────┬─────────────┘
                                    │ face crop
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
        ┌───────────────────────┐       ┌───────────────────────┐
        │  ArcFace              │       │  MobileNetV3          │
        │  512-D embedding      │       │  occlusion classifier │
        │  L2 normalization     │       │  3 classes            │
        └───────────┬───────────┘       └───────────┬───────────┘
                    │                               │
      cosine similarity vs. target       P(partial) + P(occluded)
                    │                               │
              > 0.30 ?                          > 30% ?
                    │                               │
              TARGET / OTHER                   UNSAFE / SAFE
                    └───────────────┬───────────────┘
                                    ▼
                            label + coloured box
```

### 1. Face detection — YOLOv5n-face

`YOLOv5n-face` is a YOLOv5-nano backbone fine-tuned exclusively on facial datasets. It was chosen over classical detectors (Haar cascades, HOG) because it is far more robust to profile views and to varying lighting, while still being small enough to run on the Pi.

Post-processing is implemented by hand in `detector.py`:

- **Letterbox** resize to 640×640 preserving aspect ratio (grey padding), so faces are not distorted before inference.
- Objectness thresholding (`conf_thres = 0.4`).
- Coordinate rescaling back to the original frame (padding removed, then divided by the letterbox ratio).
- **Non-Maximum Suppression** via `cv2.dnn.NMSBoxes` (`iou_thres = 0.45`) to collapse duplicate boxes.

### 2. Face recognition — ArcFace

ArcFace maps a 112×112 face crop to a **512-dimensional embedding**. It is trained with an *Additive Angular Margin* softmax loss, which pushes classes apart by an angular margin on the hypersphere; the result is an embedding space where cosine similarity is a meaningful identity metric and where the same person stays consistent across pose and lighting changes.

**Enrollment** (`Embedding_extraction.py`, run once, offline):

1. Collect 15–20+ photos of the target in `target_images/`.
2. Detect the largest face in each photo with YOLOv5n-face.
3. Extract one ArcFace embedding per photo and L2-normalize it.
4. Average the embeddings and re-normalize → a single **template** saved as `target_emb.npy`.

Averaging several views makes the template much less sensitive to any single unlucky photo.

**Verification** (at runtime): the live embedding is L2-normalized and compared to the template with **cosine similarity**. Since both vectors are unit-norm, this is just a dot product. `similarity ≥ 0.30` → `TARGET`, otherwise `OTHER`.

### 3. Occlusion detection — MobileNetV3

A single MobileNetV3 classifier decides whether the face is covered, with three classes:

| index | class             |
|-------|-------------------|
| 0     | no occlusion      |
| 1     | partial occlusion |
| 2     | occlusion         |

The two "covered" probabilities are summed:

```
covered % = ( P(partial occlusion) + P(occlusion) ) × 100
unsafe    = covered % > 30
```

Summing the two classes instead of taking the `argmax` matters in practice: a face that is genuinely half-covered often splits its probability mass between *partial* and *full* occlusion, so the argmax can still land on *no occlusion* even when the total covered mass is large.

> **Note.** An earlier version of the project used two separate networks — a MobileNetV2 mask classifier (*no mask / mask / incorrectly worn mask*) plus the MobileNetV3 occlusion classifier — and OR-ed their verdicts. This was simplified to the single occlusion model: a mask *is* an occlusion, so the second network mostly duplicated the first one while adding a full extra forward pass per face on a CPU-bound device.

### 4. Deployment on Raspberry Pi 5

**Model conversion** (`Models_quantization.py`). Every model is exported to ONNX and then converted to the OpenVINO **IR** format:

- `.xml` → network topology
- `.bin` → weights

Conversion uses `compress_to_fp16=True`, i.e. weights are stored in **FP16 instead of FP32**. This halves the size of the `.bin` files and reduces memory bandwidth, which is the real bottleneck on the Pi. Strictly speaking this is FP16 weight compression, not INT8 quantization — there is no calibration dataset and no activation quantization involved.

**Multithreading** (`main.py`, `camera.py`). Inference on four models is far slower than the camera frame rate, so a naive single loop makes the preview stutter. The work is split across threads:

- **Capture thread** — reads the MJPEG byte stream from `rpicam-vid`, splits it on the JPEG `FFD8`/`FFD9` markers, decodes each frame and keeps only the most recent one under a lock. Old frames are dropped rather than queued, so latency does not build up.
- **AI thread** — grabs the latest frame, runs detection + recognition + occlusion, and publishes the resulting list of boxes/labels under a lock.
- **Main thread** — reads the latest frame and the latest results and renders, staying at the camera's 30 FPS regardless of how long inference takes.

The trade-off is that boxes can lag the video by one inference cycle; for a monitoring application this is a much better deal than a smooth-but-slow feed.

---

## Repository structure

```
.
├── main.py                    # entry point: threads, rendering, label logic
├── camera.py                  # rpicam-vid MJPEG capture thread
├── detector.py                # OpenVINOModel wrapper + YoloDetector (letterbox, NMS)
├── arcface_recognition.py     # embedding extraction, L2 norm, cosine matching
├── Cover.py                   # SafetyAnalyzer: occlusion classification → safe/unsafe
├── Embedding_extraction.py    # offline enrollment → target_emb.npy
├── Models_quantization.py     # ONNX → OpenVINO IR (FP16)
├── config.py                  # model paths and thresholds
├── models/
│   ├── yolov5n_face.onnx
│   ├── arcface.onnx
│   ├── occlusion_mobilenetv3.onnx
│   └── openvino/              # generated IR files (.xml / .bin)
├── target_images/             # enrollment photos of the target
└── target_emb.npy             # generated target template
```

> **The ONNX weights are not included in this repository.** Place your own `yolov5n_face.onnx`, `arcface.onnx` and `occlusion_mobilenetv3.onnx` in `models/` before running the conversion step.

---

## Requirements

- Raspberry Pi 5 with a CSI camera and `rpicam-vid` available (`rpicam-apps`), or any Linux machine if you replace the capture backend in `camera.py`
- Python 3.9+

```bash
pip install openvino opencv-python numpy
```

No PyTorch or TensorFlow is needed at runtime — only OpenVINO, OpenCV and NumPy.

---

## Usage

**1. Convert the models to OpenVINO IR**

```bash
python Models_quantization.py
```

Produces `models/openvino/*.xml` and `*.bin`.

**2. Enroll the target**

Put 15–20+ photos of the target person in `target_images/` (different angles, lighting and expressions), then:

```bash
python Embedding_extraction.py
```

Produces `target_emb.npy`.

**3. Run the system**

```bash
python main.py
```

Press `ESC` to quit.

---

## Configuration

All tunables live in `config.py`:

| Parameter                  | Default | Meaning                                                                 |
|----------------------------|---------|-------------------------------------------------------------------------|
| `THRESHOLD_TARGET`         | `0.30`  | Minimum cosine similarity to classify a face as the target             |
| `THRESHOLD_UNSAFE_PERCENT` | `30.0`  | Minimum combined partial+full occlusion probability to flag *unsafe*   |

Detection thresholds (`conf_thres = 0.4`, `iou_thres = 0.45`) are arguments of `YoloDetector`.

Raising `THRESHOLD_TARGET` reduces false acceptances at the cost of more missed detections of the target; lowering `THRESHOLD_UNSAFE_PERCENT` makes the system more suspicious of partially covered faces.

---

## Output

| Label           | Box colour |
|-----------------|------------|
| `TARGET SAFE`   | green      |
| `TARGET UNSAFE` | yellow     |
| `OTHER SAFE`    | blue       |
| `OTHER UNSAFE`  | red        |

The cosine similarity score is printed next to each label.

---

## Authors

- Marco Baruffi

