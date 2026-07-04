# ALPRFast — a CodeProject.AI ALPR module for wide 4K CCTV

Automatic License Plate Recognition tuned for **wide-angle 4K security cameras**, where
a plate is only a few dozen pixels in the frame. Built on
[fast-alpr](https://github.com/ankandrew/fast-alpr) (a YOLOv9 plate detector +
`fast-plate-ocr` CCT recognizer), wrapped as a CodeProject.AI module so **Blue Iris**
(or any CPAI client) can call it at `/v1/vision/alpr`.

Pure ONNX / onnxruntime, **CUDA on Linux**.

## Why not stock ALPR modules?
On a wide 4K scene the plate shrinks below the detector's input size and is missed, and
the common YOLO char-OCR over-detects (reading state script / signage as plate chars).
ALPRFast adds the accuracy stack that actually works here:

1. **Crop-first** — on a full-frame miss, ask the object-detection module for vehicles,
   crop each from the 4K source, and re-run plate detection on the crop.
2. **Bigger detector** — `yolo-v9-t-640` input (vs 384) for small/distant plates.
3. **Super-resolution** — Lanczos-upscale the tight plate crop before OCR.
4. **Format + confidence gating** — US/CA plate pattern + per-character confidence
   drop garbage reads instead of emitting them.
5. **Live multi-frame voting** — collapse near-duplicate reads seen within a short
   window into one confident plate (resolves the `O`/`D`/`Q` flicker across frames).

## Install
Drop the folder in your CodeProject.AI `modules/` dir as `ALPRFast` and run module setup
(`requirements.linux.txt` installs `fast-alpr[onnx-gpu]` + `onnxruntime-gpu[cuda,cudnn]`;
models auto-download from HuggingFace on first use). Requires an object-detection module
(e.g. `ObjectDetectionYOLOv5`) running for the crop-first step. Then point Blue Iris'
AI at `http://<host>:32168`.

## Key settings (env, in `modulesettings.json`)
| Var | Default | Purpose |
|---|---|---|
| `USE_CUDA` | `True` | GPU (CUDA) inference |
| `DETECTOR_MODEL` | `yolo-v9-t-640-license-plate-end2end` | plate detector (bigger = better small plates) |
| `OCR_MODEL` | `cct-s-v2-global-model` | fast-plate-ocr model |
| `MIN_CHAR_CONFIDENCE` | `0.55` | gating threshold |
| `ENABLE_VEHICLE_CROP_FALLBACK` | `True` | crop-first for wide 4K |
| `ENABLE_VOTING` / `VOTE_WINDOW_SECS` | `True` / `8.0` | live multi-frame voting |

## Credits
Detection/OCR by [fast-alpr](https://github.com/ankandrew/fast-alpr) /
[fast-plate-ocr](https://github.com/ankandrew/fast-plate-ocr) /
[open-image-models](https://github.com/ankandrew/open-image-models) (MIT). This module
(adapter, crop-first, super-res, gating, voting) is MIT.
