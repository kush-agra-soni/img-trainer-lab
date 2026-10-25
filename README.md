# Image AI Platform

Four image-AI tools in one local web app. Everything runs on your own machine, and a GPU is optional:

- **Transfer Learning:** train an image classifier on your own photos in seconds, then publish it to the Hugging Face Hub.
- **Object Detection:** detect 80 kinds of objects in images, videos or your live webcam.
- **Image Generation:** create images from text with SD-Turbo, in 1–4 steps.
- **Image Enhancement:** upscale images to 480p, 720p, 1080p, 2K or 4K with Real-ESRGAN, and compare the result with classical resizing.

A one-click launcher sets everything up on Windows.

![Generation](docs/screenshots/generation.png)

## Contents

- [Quick start](#quick-start)
- [The tools](#the-tools)
- [Results](#results)
- [How it works](#how-it-works)
- [API](#api)
- [Development](#development)
- [Models and licences](#models-and-licences)

## Quick start

**Requirements:** Windows 10/11 with Python 3.10 or newer ([python.org](https://www.python.org/downloads/), with "Add python.exe to PATH" ticked). About 6 GB of free disk space covers PyTorch and every model. A webcam is optional.

1. Download or clone this repository.
2. Double-click `launcher.exe`.

The first run takes about 5 minutes: the launcher installs PyTorch (CPU build) and the other dependencies into `.venv/`, then opens `http://localhost:8506`. Models download the first time each tool is used:

- detection and enhancement models: 5–70 MB each
- the demo dataset: 94 MB
- SD-Turbo: 2.6 GB

Closing the launcher window stops everything.

Optional settings in `.env`:

```
HF_TOKEN=hf_...        # Hugging Face write token, to publish trained classifiers
IMGLAB_DEVICE=auto     # auto | cpu | cuda
```

For an NVIDIA GPU, install the CUDA build of PyTorch into `.venv` (see pytorch.org). Generation automatically offloads model parts, so it fits GPUs with as little as 2 GB of memory.

## The tools

### Transfer Learning

- **Data:** upload a zip with one folder per class (at least 3 images each), or build a demo dataset from [Imagenette](https://github.com/fastai/imagenette), a subset of ImageNet.
- **Backbones:** MobileNetV3-Large, EfficientNet-B0 or ResNet-18, all pretrained on ImageNet.
- **Modes:**
  - *Feature extraction:* the backbone stays frozen and its features are cached, so each epoch takes about a second on CPU.
  - *Fine-tuning:* the top N blocks are unfrozen and trained with augmentation and a lower learning rate.
- **Training:** adjustable epochs, learning rate, batch size and validation split. A stratified 80/20 split, a live training curve, and early stopping on validation accuracy, with ties broken by validation loss.
- **Results:** accuracy, macro F1, per-class precision and recall, and a confusion matrix.
- **Testing:** classify a test image. A low-confidence warning flags images that don't look like any trained class.
- **Publishing:** to the Hugging Face Hub in `safetensors` format with a generated model card. Repositories are private by default.

### Object Detection

- **Models:** SSDlite MobileNetV3 (fast) or Faster R-CNN MobileNetV3 FPN (more accurate), trained on COCO's 80 classes.
- **Controls:** confidence threshold and an optional class filter.
- **Sources:** images, video files (annotated MP4 output, analysing every Nth frame) and **live webcam** (frames are processed locally and never stored).
- **Export:** annotated image and COCO-format JSON.

### Image Generation

- **Model:** SD-Turbo, a version of Stable Diffusion trained to produce an image in 1–4 denoising steps, which makes CPU generation practical.
- **Controls:** prompt, steps, size (256 / 384 / 512), 1–4 images, and a seed for reproducibility.
- **Gallery:** every result is saved with its prompt and seed.
- **Prompt filter:** a basic blocklist, because SD-Turbo ships without a safety checker.

### Image Enhancement

- **Methods:**
  - *Real-ESRGAN general v3* (fast)
  - *Real-ESRGAN x4plus* (most detail)
  - *Real-ESRNet x4plus* (highest fidelity)
  - Lanczos and bicubic baselines
- **Output size:** a target resolution (480p–4K) or a factor (2x, 4x, 8x). The input is resized first so the last 4x neural pass lands exactly on the target, instead of overshooting and shrinking.
- **Memory:** large images are processed in overlapping tiles, so memory use stays bounded.
- **Comparison:** a before-and-after crop, and a benchmark that downscales your image 4x, restores it with every method, and reports PSNR, SSIM and sharpness.

### Screenshots

| Transfer Learning | Object Detection |
|---|---|
| ![Transfer](docs/screenshots/transfer.png) | ![Detection](docs/screenshots/detection.png) |

![Enhancement](docs/screenshots/enhancement.png)

## Results

Measured with `scripts/run_eval.py` on a laptop CPU (Intel i7-1195G7, 4 cores, 8 GB RAM; no GPU used).

**Transfer learning.** Imagenette, 10 classes, 60 images per class (48 for training and 12 for validation per class):

| Backbone | Mode | Validation accuracy | Macro F1 | Training time |
|---|---|---|---|---|
| EfficientNet-B0 | feature extraction | **98.2%** | 0.982 | 14 s |
| MobileNetV3-Large | feature extraction | 97.2% | 0.973 | 12 s |
| ResNet-18 | feature extraction | 97.2% | 0.973 | 11 s |
| MobileNetV3-Large | fine-tune top 2 blocks, 4 epochs | 95.4% | 0.953 | 48 s |

With only 48 images per class, frozen ImageNet features generalise better than fine-tuning. Fine-tuning pays off with larger datasets or images unlike ImageNet. Early stopping originally kept an under-trained first-epoch head, which gave only 41% confidence on a clear image. Breaking accuracy ties by validation loss raised that to 96%.

**Object detection.** Time per image, after warm-up:

| Model | Astronaut | Cat | Coffee | Rocket |
|---|---|---|---|---|
| SSDlite | person, 64 ms | cat, 53 ms | cup and dining table, 44 ms | nothing (not a COCO class), 62 ms |
| Faster R-CNN | person, about 360 ms | cat (plus a false "carrot" at 61%) | cup and dining table | nothing |

COCO has no "fox" class, so a fox is reported as the nearest class, "dog". Use transfer learning for custom classes.

**Image enhancement.** Each photo was downscaled 4x, restored, and compared with the original:

| Method | PSNR (astronaut / cat / coffee) | Sharpness (astronaut) | Time (astronaut) |
|---|---|---|---|
| Bicubic | 25.4 / 30.5 / 25.8 | 21 | < 0.01 s |
| Lanczos | 25.7 / 30.7 / 26.0 | 21 | < 0.01 s |
| Real-ESRNet x4plus | **26.4** / 30.2 / **26.5** | 230 | 3.4 s |
| Real-ESRGAN general v3 | 23.4 / 28.6 / 25.3 | 439 | 0.3 s |
| Real-ESRGAN x4plus | 23.8 / 27.5 / 25.8 | **733** | 3.4 s |

The original astronaut image has a sharpness of 861. The trade-off between the two kinds of model:

- **Fidelity:** Real-ESRNet gives the best fidelity (PSNR) on 2 of 3 images.
- **Detail:** the GAN models restore 20–35x more fine detail than Lanczos, at the cost of PSNR, because they synthesise plausible texture instead of averaging.

Upscaling 144p (256x144) to 4K takes 8.3 s on CPU.

**Generation.** SD-Turbo at 512 px on CPU:

| Steps | Time |
|---|---|
| 1 | 27 s |
| 2 | 34 s |
| 4 | 61 s |

Peak memory is about 1.8 GB. One step already produces a coherent image; more steps sharpen textures.

## How it works

```
Streamlit UI (:8506) -> FastAPI (:8006) -> background jobs (training, generation, video) with live logs
   Transfer:   zip (validated) -> torchvision backbone -> cached features or partial fine-tune -> safetensors + config
               -> classify / publish (huggingface_hub)
   Detection:  torchvision SSDlite / Faster R-CNN -> boxes -> annotate / COCO JSON; OpenCV webcam thread; video writer
   Generation: diffusers SD-Turbo (fp16 weights, fp32 on CPU; model offload on small GPUs) -> gallery
   Enhance:    spandrel-loaded Real-ESRGAN family -> tiled 4x passes -> resize to target; PSNR / SSIM / sharpness
```

| Module | Responsibility |
|---|---|
| `transfer.py` | Datasets (zip safety, Imagenette demo), training loops, metrics, saving, classification, Hugging Face publishing |
| `detect.py` | Detection, annotation, COCO export, video processing, webcam streamer |
| `generate.py` | SD-Turbo pipeline, prompt filter |
| `enhance.py` | Model download and loading, tiled super-resolution, target sizing, quality metrics |
| `api.py` | REST API and background jobs |

## API

Interactive documentation is available at `http://localhost:8006/docs`.

| Method | Path | Purpose |
|---|---|---|
| POST | `/transfer/datasets/upload`, `/transfer/datasets/demo` | Add a zip dataset, or build the Imagenette demo |
| POST | `/transfer/train` | Train (background job; poll `/jobs/{id}`, stop with `/jobs/{id}/stop`) |
| GET | `/transfer/models`, `/transfer/models/{name}` | Trained classifiers and their metrics |
| POST | `/transfer/models/{name}/classify`, `/transfer/models/{name}/publish` | Classify an image; publish to the Hugging Face Hub |
| POST | `/detect/image` | Detections as JSON, annotated JPEG or COCO |
| POST | `/detect/video` | Annotate a video (background job) |
| POST | `/detect/webcam/start`, `/detect/webcam/stop` | Live camera |
| GET | `/detect/webcam/frame`, `/detect/webcam/status` | Latest annotated frame; camera status |
| POST | `/generate` | Text-to-image (background job) |
| GET | `/gallery` | Generated images |
| POST | `/enhance`, `/enhance/benchmark` | Upscale an image; compare all methods |

## Development

```
.venv/Scripts/python -m pytest              # 29 tests (small real models; generation and the Hugging Face API are faked)
.venv/Scripts/python scripts/run_eval.py    # live benchmark of all four tools
powershell -File scripts/build_launcher.ps1
```

## Models and licences

| Model | Source | Licence |
|---|---|---|
| MobileNetV3, EfficientNet-B0, ResNet-18, SSDlite, Faster R-CNN | torchvision | BSD-3-Clause (code); ImageNet / COCO-trained weights |
| Real-ESRGAN / Real-ESRNet | xinntao/Real-ESRGAN | BSD-3-Clause |
| SD-Turbo | stabilityai/sd-turbo | Stability AI Non-Commercial Research Community License: non-commercial use only |
| Imagenette (demo data) | fast.ai | Subset of ImageNet; research use |

Ultralytics YOLO was deliberately not used, because its AGPL licence would apply to the whole application.
