# ALPRFast

A [CodeProject.AI](https://www.codeproject.com/AI/index.aspx) module for automatic
license-plate recognition (ALPR), built for **Linux + NVIDIA CUDA (GPU)** and aimed at
real CCTV — small, angled, distant, night-time plates on wide overview cameras.

It packages [fast-alpr](https://github.com/ankandrew/fast-alpr) (a YOLOv9 plate detector
plus a compact-convolutional-transformer OCR model, all ONNX / `onnxruntime-gpu`) as a
drop-in CodeProject.AI module and adds a CCTV-specific accuracy stack on top. It serves
CodeProject.AI's standard `/v1/vision/alpr` route, so **Blue Iris** — or anything that
already talks to the CPAI ALPR endpoint — uses it with **no client changes**.

## Why this exists

On Linux with a GPU, the ALPR options for CodeProject.AI are thin. MikeLud's well-regarded
YOLO11-based ALPR modules
([MikeLud-CodeProject.AI-Modules](https://github.com/MikeLud/MikeLud-CodeProject.AI-Modules))
are **Windows-only**, so on a Linux/Docker server they don't run at all — which leaves only
the cross-platform PaddleOCR "License Plate Reader", and that struggles on real CCTV plates.
ALPRFast brings a modern YOLO-based ALPR pipeline (`fast-alpr`'s YOLOv9 plate detector plus a
transformer OCR model) to **Linux + CUDA**, packaged as a CodeProject.AI module.

On a wide 4K scene the fundamental problem is size: the plate is often only a few dozen
pixels, so a full-frame pass simply doesn't see it. ALPRFast wraps `fast-alpr` with the
extra steps that make a wide overview camera usable as an opportunistic plate reader.

> **On Windows?** MikeLud's
> [ALPR (YOLO11)](https://github.com/MikeLud/MikeLud-CodeProject.AI-Modules) modules
> ([IPCamTalk thread](https://ipcamtalk.com/threads/new-codeproject-ai-license-plate-recognition-yolo11-module.83478/))
> are the natural choice there — ALPRFast is aimed at the Linux/GPU case they don't cover.
> I've also opened a [Linux/CUDA PR](https://github.com/MikeLud/CodeProject.AI-ALPR-YOLO11/pull/2)
> against his YOLO11 ALPR to help close that gap upstream.

## Which CodeProject.AI ALPR module should I use?

ALPRFast is one of several ALPR options for CodeProject.AI — and **not always the right one.**
Pick by platform and GPU:

| Your setup | Use | Notes |
|---|---|---|
| **Windows** | MikeLud's [**ALPR (YOLO11)**](https://github.com/MikeLud/MikeLud-CodeProject.AI-Modules) | The strong choice on Windows (DirectML GPU). |
| **Any platform, simplest** | stock **License Plate Reader** (PaddleOCR) | One-click from the dashboard; CPU-only on Linux. |
| **Linux + NVIDIA, want YOLO11** | the [**Linux/CUDA YOLO11 fork**](https://github.com/MikeLud/CodeProject.AI-ALPR-YOLO11/pull/2) | Adds `onnxruntime-gpu` to MikeLud's Windows-only module. |
| **Linux + NVIDIA, wide 4K CCTV** | **ALPRFast** (this repo) | GPU-native; the crop-first + voting stack below. |

Step-by-step install for every option is in [`docs/INSTALL.md`](docs/INSTALL.md).

## Features — the accuracy stack

Beyond vanilla `fast-alpr`, ALPRFast adds:

1. **Crop-first.** If the full-frame pass finds no plate, it calls your object-detection
   module (over HTTP), crops the detected vehicles out of the full-resolution frame, and
   re-runs the plate detector on each crop. A plate that was ~40 px in the 4K frame
   becomes a few hundred pixels in the vehicle crop.
2. **Super-resolution upscale.** The tight plate crop is upscaled with **Lanczos**
   interpolation before OCR. This is deliberately *not* neural super-resolution — neural
   SR hallucinates characters and scored ~0% exact-match on sub-100 px plates in testing.
3. **Format + confidence gating.** Reads are checked against a US/CA plate shape
   (5–8 alphanumerics) and a per-character confidence floor, so junk reads are dropped
   instead of emitted.
4. **Live multi-frame voting.** A time-windowed, positional-majority vote across the
   frames in a pass resolves OCR ambiguity. A car read as `9RDL852` / `9ROL852` /
   `9RQL852` across frames votes down to a single plate.

Two optional pieces, both **off by default**, are covered under
[Optional: capture harness & night mode](#optional-capture-harness--night-mode).

## Requirements

- **Linux** (x64 or arm64).
- **NVIDIA GPU with CUDA 12.** The module pins `onnxruntime-gpu==1.23.0`, whose
  `[cuda,cudnn]` extras pull the matching CUDA 12 runtime + cuDNN 9 into the module's
  virtual environment, so you don't need a system-wide CUDA install — but you do need a
  working NVIDIA driver. CPU-only works but is slow and not the target.
- **CodeProject.AI Server 2.9+.**
- **An object-detection module** (e.g. `ObjectDetectionYOLOv5` — any version, v5/v8/v11,
  works) running on the same server — the crop-first step calls it. Without one, ALPRFast
  still runs full-frame only.

## Install

ALPRFast installs like any other CodeProject.AI module:

1. Place this folder in your CodeProject.AI Server's `modules/` directory as `ALPRFast`.
2. Run the server's module setup for it (the dashboard's install button, or
   `bash <server>/src/setup.sh` from the module directory).

Setup installs `fast-alpr[onnx-gpu]` and `onnxruntime-gpu[cuda,cudnn]` (plus OpenCV's
Linux system libs on bare metal). The **ONNX model weights are not bundled** — `fast-alpr`
downloads the YOLOv9 detector and the CCT OCR model from HuggingFace on **first use** and
caches them under `~/.cache`. The first request after a fresh install is therefore slower
while the models download.

### onnxruntime `preload_dlls()` — important

`onnxruntime-gpu` **≥ 1.19** ships its CUDA and cuDNN libraries as pip extras that the
Linux loader does **not** find automatically. If they aren't preloaded, ORT silently
falls back to `CPUExecutionProvider` — you get CPU-speed inference with no error, just
slow reads. ALPRFast calls `ort.preload_dlls()` at startup to make `CUDAExecutionProvider`
initialise correctly. You don't have to do anything; this is documented so that if you
fork or debug the module you know why the call is there. If CUDA still isn't used, check
the module log for the provider list and confirm your NVIDIA driver is healthy.

## Configuration

All settings are environment variables (defined in `modulesettings.json`; override them
in the CodeProject.AI dashboard under the module's settings).

| Variable | Default | Purpose |
|---|---|---|
| `USE_CUDA` | `True` | Use the GPU (`CUDAExecutionProvider`). Set `False` to force CPU. |
| `DETECTOR_MODEL` | `yolo-v9-t-640-license-plate-end2end` | `open-image-models` YOLOv9 plate detector. |
| `OCR_MODEL` | `cct-s-v2-global-model` | `fast-plate-ocr` CCT recognizer. |
| `PLATE_DETECTOR_CONFIDENCE` | `0.35` | Minimum detector confidence to accept a plate box. |
| `MIN_CHAR_CONFIDENCE` | `0.55` | Per-character OCR confidence floor; reads below this are rejected by gating. |
| `PLATE_PAD` | `0.12` | Fractional padding added around each detected plate box before OCR. |
| `SR_TARGET_HEIGHT` | `64` | Lanczos-upscale plate crops shorter than this many pixels up to this height. |
| `ENABLE_VEHICLE_CROP_FALLBACK` | `True` | Enable the crop-first step on a full-frame miss. |
| `OBJECT_DETECTION_URL` | `http://localhost:32168/v1/vision/detection` | Object-detection endpoint used by crop-first. |
| `VEHICLE_LABELS` | `car,truck,bus,motorcycle,motorbike` | Labels treated as vehicles for cropping. |
| `VEHICLE_CROP_CONFIDENCE` | `0.25` | Minimum object-detection confidence for a vehicle crop. |
| `MAX_VEHICLE_CROPS` | `3` | Cap on vehicle crops re-run per frame (largest first). |
| `ENABLE_VOTING` | `True` | Enable time-windowed multi-frame voting. |
| `VOTE_WINDOW_SECS` | `8.0` | Voting window, in seconds, for collapsing near-duplicate reads. |
| `SAVE_CAPTURES` | `False` | Capture harness: log every processed frame + read to CSV (see below). |
| `CAPTURE_DIR` | `/app/modules/ALPRFast/captures` | Where the capture harness writes frames and `log.csv`. |
| `CAPTURE_MAX` | `8000` | Stop capturing after this many frames. |
| `NIGHT_ENHANCE` | `False` | Low-light branch: CLAHE local-contrast on dark crops (see below). |
| `NIGHT_LUMA_THRESH` | `80` | Mean-luma threshold below which a crop is treated as "night". |
| `NIGHT_DENOISE` | `True` | Apply denoising in the night branch (only relevant when `NIGHT_ENHANCE=True`). |

> **Note:** the `modulesettings.json` shipped in this repo has `SAVE_CAPTURES` set to
> `True` for corpus-building during development. If you don't want captures written to
> disk, set it to `False` before running in production.

The `/v1/vision/alpr` response returns, per plate: `label` (the plate text), `plate`,
`confidence`, a bounding box (`x_min`/`y_min`/`x_max`/`y_max`), and `votes` (how many
frames backed the voted read).

## Using it from Blue Iris

Blue Iris already knows how to call CodeProject.AI's ALPR endpoint, so there's nothing
special to configure on the ALPR side:

1. In Blue Iris, point the CodeProject.AI / AI server at your CPAI host and port (default
   `http://<host>:32168`).
2. Enable ALPR on the camera(s) you want. Blue Iris posts to `/v1/vision/alpr`, which
   ALPRFast now serves.

No plugin, no custom URL, no client changes — ALPRFast is a drop-in alternative to the
stock ALPR module on that route (same endpoint, different reader).

## Optional: capture harness & night mode

Both are **off by default**. Turn them on deliberately.

- **Capture harness (`SAVE_CAPTURES`).** Writes every processed frame as a JPEG under
  `CAPTURE_DIR/frames/` and appends a row per frame to `CAPTURE_DIR/log.csv` (timestamp,
  frame size, mean luma, read counts, the best plate/confidence/votes, and timing). It's
  for building a labelled tuning corpus — especially for night work — not for normal
  operation. It stops after `CAPTURE_MAX` frames. Expect real disk usage; leave it off
  unless you're actively tuning.

- **Night low-light branch (`NIGHT_ENHANCE`).** Gated by mean luma: only crops darker
  than `NIGHT_LUMA_THRESH` get CLAHE local-contrast (and optional denoising via
  `NIGHT_DENOISE`); already-bright daytime crops pass through untouched. It's off by
  default because blanket enhancement can *hurt* already-legible plates, and the right
  thresholds are deployment-specific. Treat it as something to tune per site, not a
  free win.

## Tuning for a wide overview camera

ALPRFast runs in production feeding Blue Iris on an NVIDIA vGPU, reading real plates all
day off a wide 4K Hikvision ColorVu carport/street overview camera, at roughly
250–350 ms per read once warm. It's genuinely useful there. Set expectations honestly:

- **A wide overview camera is not a dedicated LPR camera.** Many plates are under 100 px
  and face the camera only briefly, so angled, distant, fast, and night plates get
  missed. The dominant failure mode here is **geometry, not the reader** — no amount of
  OCR tuning fixes a plate that's 30 px and turned away. If you need guaranteed capture,
  use a dedicated, zoomed LPR camera at a choke-point (a driveway or gate) pointed to
  catch plates head-on. ALPRFast shines as an *opportunistic* reader on cameras you
  already have.
- **Blue Iris bounds the voting.** BI forwards only its "best" plate-detected frames to
  the ALPR endpoint — often ~3 per pass — not every frame, which caps how much the
  multi-frame vote has to work with. You can widen it in BI's **Alert Confirmation**
  settings: enable pre/post-trigger images and "analyze one image each" so more frames
  reach ALPRFast. More frames in the window generally means better voting.
- **Night/IR is hard, and tuning is deployment-specific.** The night branch and the
  capture harness exist to help you tune for *your* scene; there's no universal setting.

## Related upstream contributions

Fixes I've contributed to the *other* CodeProject.AI ALPR modules while getting ALPR
working on Linux/GPU — all open PRs to their upstreams. Use those modules directly if they
suit your setup better than this one:

- **[Linux/CUDA support for the YOLO11 ALPR](https://github.com/MikeLud/CodeProject.AI-ALPR-YOLO11/pull/2)**
  — so MikeLud's (Windows-only) YOLO11 module can run on Linux/NVIDIA.
- **[PaddleOCR GPU on Linux / CUDA 12](https://github.com/codeproject/CodeProject.AI-ALPR/pull/25)**
  — a GPU path for the stock PaddleOCR "License Plate Reader" on Linux.
- **[Vehicle-crop fallback](https://github.com/codeproject/CodeProject.AI-ALPR/pull/26)**
  — read plates in wide / high-resolution scenes; ALPRFast's crop-first step shares this idea.

## Credits

The detection and OCR core is [**fast-alpr**](https://github.com/ankandrew/fast-alpr) by
[ankandrew](https://github.com/ankandrew) (MIT), which combines:

- [**open-image-models**](https://github.com/ankandrew/open-image-models) — the YOLOv9
  license-plate detector.
- [**fast-plate-ocr**](https://github.com/ankandrew/fast-plate-ocr) — the CCT
  (compact-convolutional-transformer) OCR recognizer.

ALPRFast packages and extends that work for CodeProject.AI + CCTV — the module adapter,
crop-first vehicle detection, Lanczos super-resolution, format/confidence gating, and
multi-frame voting. Thanks to the `fast-alpr` author for the underlying models and library.

Thanks also to [**MikeLud**](https://github.com/MikeLud/MikeLud-CodeProject.AI-Modules),
whose ALPR modules are the reference for license-plate recognition on CodeProject.AI. His
YOLO11 ALPR is Windows-only, which is the exact niche ALPRFast fills; the crop-first idea
here also shares lineage with the vehicle-crop fallback contributed to the PaddleOCR ALPR
module. If you're on Windows, use his modules — this one is for Linux/GPU.

## License

MIT. See [`LICENSE`](LICENSE). The bundled/downloaded models and dependencies carry their
own licenses (`fast-alpr`, `open-image-models`, and `fast-plate-ocr` are MIT).
