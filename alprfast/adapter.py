"""CodeProject.AI SDK adapter for the ALPRFast module (exposes /v1/vision/alpr)."""
import os
import sys
import threading
import numpy as np
import cv2

from typing import Dict, Any
from codeproject_ai_sdk import RequestData, ModuleRunner, LogMethod, JSON

from .pipeline import ALPRFast


class ALPRFastAdapter(ModuleRunner):
    def __init__(self):
        super().__init__()
        self.alpr = None
        self._lock = threading.Lock()
        self._plates_detected = 0
        self._histogram: Dict[str, int] = {}

    def initialise(self):
        try:
            self.alpr = ALPRFast()
            self.inference_device = self.alpr.inference_device
            self.inference_library = "CUDA" if self.alpr.inference_device == "GPU" else "CPU"
            self.can_use_GPU = self.alpr.inference_device == "GPU"
            self.log(LogMethod.Info | LogMethod.Server, {
                "filename": __file__, "loglevel": "information",
                "method": sys._getframe().f_code.co_name,
                "message": f"ALPRFast initialised on {self.alpr.inference_device} "
                           f"(detector={self.alpr.detector_model}, ocr={self.alpr.ocr_model})"})
        except Exception as ex:
            self.report_error(ex, __file__, f"Error initialising ALPRFast: {ex}")
            self.alpr = None

    def process(self, data: RequestData) -> JSON:
        with self._lock:
            try:
                if self.alpr is None:
                    return {"success": False, "error": "ALPRFast not initialised"}
                img = data.get_image(0)
                arr = np.array(img)
                if arr.ndim == 3 and arr.shape[2] == 3:
                    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                preds, ms = self.alpr.read_image(arr)
                self._plates_detected += len(preds)
                for p in preds:
                    self._histogram[p["label"]] = self._histogram.get(p["label"], 0) + 1
                return {"success": True, "predictions": preds, "count": len(preds),
                        "message": (preds[0]["label"] if preds else "No plates found"),
                        "inferenceMs": ms, "processMs": ms}
            except Exception as ex:
                self.report_error(ex, __file__, f"Error processing request: {ex}")
                return {"success": False, "error": f"Error processing request: {ex}"}

    def status(self) -> JSON:
        s = super().status()
        with self._lock:
            s["platesDetected"] = self._plates_detected
            s["histogram"] = dict(self._histogram)
        return s

    def selftest(self) -> JSON:
        if self.alpr is None:
            return {"success": False, "message": "ALPRFast failed to initialise"}
        test_file = os.path.join("test", "license_plate_test.jpg")
        if not os.path.exists(test_file):
            return {"success": True, "message": "ALPRFast initialised (no test image)"}
        rd = RequestData(); rd.queue = self.queue_name; rd.command = "detect"
        rd.add_file(test_file); rd.add_value("min_confidence", 0.4)
        res = self.process(rd)
        return {"success": res.get("success", False),
                "message": f"ALPRFast self-test: {res.get('message')}"}


if __name__ == "__main__":
    ALPRFastAdapter().start_loop()
