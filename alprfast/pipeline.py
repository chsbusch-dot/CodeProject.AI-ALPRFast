"""
ALPRFast pipeline: fast-alpr (open-image-models YOLOv9 detector + fast-plate-ocr CCT)
with the accuracy stack that makes it work on wide 4K CCTV:

  full frame -> (crop-first) vehicle detect -> crop from 4K -> big plate detector
             -> tight plate crop -> Lanczos super-resolution -> OCR (+per-char conf)
             -> US/CA format + confidence gating -> live multi-frame voting.

Pure ONNX/onnxruntime, CUDA-capable. See README.
"""
import os
import re
import time
import cv2
import numpy as np
from collections import Counter, deque

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from open_image_models import LicensePlateDetector
from fast_plate_ocr import LicensePlateRecognizer

# Loose US / North-American plate shape: 5-8 alphanumerics.
PLATE_RE = re.compile(r'^[A-Z0-9]{5,8}$')


def _env(name, default):
    return os.environ.get(name, default)


class VotingBuffer:
    """Time-windowed per-plate vote. Collapses near-duplicate reads (<=1 char diff)
    seen within `window_secs` into one canonical plate by positional majority."""
    def __init__(self, window_secs=8.0, max_items=64):
        self.window = window_secs
        self.buf = deque(maxlen=max_items)   # (timestamp, text)

    def add_and_vote(self, text, now):
        self.buf.append((now, text))
        recent = [t for (ts, t) in self.buf if now - ts <= self.window and len(t) == len(text)
                  and sum(a != b for a, b in zip(t, text)) <= 1]
        if len(recent) <= 1:
            return text, 1
        L = len(text)
        voted = ''.join(Counter(r[i] for r in recent).most_common(1)[0][0] for i in range(L))
        return voted, len(recent)


class ALPRFast:
    def __init__(self):
        use_cuda = _env('USE_CUDA', 'True').lower() == 'true'
        if use_cuda and ort is not None and hasattr(ort, 'preload_dlls'):
            # onnxruntime-gpu >=1.19 ships CUDA/cuDNN as pip packages the Linux loader
            # doesn't auto-find; preload so CUDAExecutionProvider initialises.
            try:
                ort.preload_dlls()
            except Exception as e:
                print(f"onnxruntime preload_dlls skipped: {e}")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_cuda else ['CPUExecutionProvider']

        self.detector_model = _env('DETECTOR_MODEL', 'yolo-v9-t-640-license-plate-end2end')
        self.ocr_model = _env('OCR_MODEL', 'cct-s-v2-global-model')
        self.det_conf = float(_env('PLATE_DETECTOR_CONFIDENCE', '0.35'))
        self.min_char_conf = float(_env('MIN_CHAR_CONFIDENCE', '0.55'))
        self.plate_pad = float(_env('PLATE_PAD', '0.12'))
        self.sr_target_h = int(_env('SR_TARGET_HEIGHT', '64'))

        self.enable_crop_first = _env('ENABLE_VEHICLE_CROP_FALLBACK', 'True').lower() == 'true'
        self.object_detection_url = _env('OBJECT_DETECTION_URL', 'http://localhost:32168/v1/vision/detection')
        self.vehicle_labels = set(_env('VEHICLE_LABELS', 'car,truck,bus,motorcycle,motorbike').split(','))
        self.vehicle_conf = float(_env('VEHICLE_CROP_CONFIDENCE', '0.25'))
        self.max_vehicle_crops = int(_env('MAX_VEHICLE_CROPS', '3'))

        self.det = LicensePlateDetector(detection_model=self.detector_model,
                                        conf_thresh=self.det_conf, providers=providers)
        self.ocr = LicensePlateRecognizer(self.ocr_model, device='cuda' if use_cuda else 'cpu')

        self.enable_voting = _env('ENABLE_VOTING', 'True').lower() == 'true'
        self.voter = VotingBuffer(window_secs=float(_env('VOTE_WINDOW_SECS', '8.0')))
        self.inference_device = 'GPU' if use_cuda else 'CPU'

    # ---- OCR + gating -----------------------------------------------------
    def _upscale(self, img):
        h, w = img.shape[:2]
        if h < 1 or w < 1:
            return img
        if h < self.sr_target_h:
            s = self.sr_target_h / h
            img = cv2.resize(img, (max(1, int(w * s)), self.sr_target_h), interpolation=cv2.INTER_LANCZOS4)
        return img

    def _read(self, plate_img):
        preds = self.ocr.run(self._upscale(plate_img), return_confidence=True)
        if not preds:
            return None
        p = preds[0]
        text = (p.plate or '').upper().replace(' ', '')
        cps = np.asarray(p.char_probs).ravel() if p.char_probs is not None else np.array([])
        cps = cps[:len(text)] if len(cps) >= len(text) and text else cps
        return {'text': text,
                'min_char_conf': float(cps.min()) if cps.size else 0.0,
                'mean_conf': float(cps.mean()) if cps.size else 0.0}

    def _valid(self, r):
        return bool(r and 5 <= len(r['text']) <= 8 and PLATE_RE.match(r['text'])
                    and r['min_char_conf'] >= self.min_char_conf)

    def read_region(self, image, offset=(0, 0)):
        """Detect + read plates within `image`; bboxes returned in +offset coords."""
        H, W = image.shape[:2]
        ox, oy = offset
        out = []
        for d in self.det.predict(image):
            b = d.bounding_box
            pw, ph = b.x2 - b.x1, b.y2 - b.y1
            px, py = int(pw * self.plate_pad), int(ph * self.plate_pad)
            x1, y1 = max(0, b.x1 - px), max(0, b.y1 - py)
            x2, y2 = min(W, b.x2 + px), min(H, b.y2 + py)
            r = self._read(image[y1:y2, x1:x2])
            if r:
                r['det_conf'] = float(d.confidence)
                r['valid'] = self._valid(r)
                r['box'] = [x1 + ox, y1 + oy, x2 + ox, y2 + oy]
                out.append(r)
        return out

    # ---- crop-first vehicle detection ------------------------------------
    def _detect_vehicles(self, image):
        import requests
        ok, buf = cv2.imencode('.jpg', image)
        if not ok:
            return []
        try:
            resp = requests.post(self.object_detection_url,
                                 files={'image': ('frame.jpg', buf.tobytes(), 'image/jpeg')}, timeout=30)
            preds = resp.json().get('predictions', []) or []
        except Exception as e:
            print(f"crop-first: object detection failed: {e}")
            return []
        boxes = [(int(p['x_min']), int(p['y_min']), int(p['x_max']), int(p['y_max']))
                 for p in preds if p.get('label') in self.vehicle_labels
                 and float(p.get('confidence', 0)) >= self.vehicle_conf]
        boxes.sort(key=lambda b: -((b[2] - b[0]) * (b[3] - b[1])))
        return boxes

    # ---- top-level --------------------------------------------------------
    def read_image(self, img_bgr):
        """Full pipeline. Returns (predictions, timings)."""
        t0 = time.perf_counter()
        H, W = img_bgr.shape[:2]
        reads = self.read_region(img_bgr)
        if not any(r['valid'] for r in reads) and self.enable_crop_first:
            for (x1, y1, x2, y2) in self._detect_vehicles(img_bgr)[:self.max_vehicle_crops]:
                dw, dh = int((x2 - x1) * 0.12), int((y2 - y1) * 0.12)
                cx1, cy1 = max(0, x1 - dw), max(0, y1 - dh)
                cx2, cy2 = min(W, x2 + dw), min(H, y2 + dh)
                crop = img_bgr[cy1:cy2, cx1:cx2]
                if crop.size:
                    reads += self.read_region(crop, offset=(cx1, cy1))
                if any(r['valid'] for r in reads):
                    break
        valid = sorted([r for r in reads if r['valid']], key=lambda r: r['mean_conf'], reverse=True)
        now = time.time()
        preds = []
        for r in valid:
            text = r['text']; nvotes = 1
            if self.enable_voting:
                text, nvotes = self.voter.add_and_vote(text, now)
            b = r['box']
            preds.append({'label': text, 'plate': text, 'confidence': round(r['mean_conf'], 4),
                          'x_min': b[0], 'y_min': b[1], 'x_max': b[2], 'y_max': b[3],
                          'votes': nvotes})
        ms = int((time.perf_counter() - t0) * 1000)
        return preds, ms
