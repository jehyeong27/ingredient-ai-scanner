import os
from pathlib import Path
from typing import List, Optional
import numpy as np
from core.schemas import DetectionResult, BoundingBox
from models.base_detector import BaseLayoutDetector
from core.exceptions import LayoutDetectionError

class YOLOLayoutDetector(BaseLayoutDetector):
    """
    Ultralytics YOLO 기반 실제 레이아웃 검출기
    """

    def __init__(self, model_path: str | Path, conf_threshold: float = 0.25):
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.model = None
        self._load_model()

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO weight file not found at: {self.model_path}")
        
        try:
            from ultralytics import YOLO
            self.model = YOLO(str(self.model_path))
        except ImportError:
            raise LayoutDetectionError("ultralytics package is not installed. Please install it with `pip install ultralytics`.")
        except Exception as e:
            raise LayoutDetectionError(f"Failed to load YOLO model: {e}")

    def predict(self, image: np.ndarray) -> List[DetectionResult]:
        if self.model is None:
            raise LayoutDetectionError("YOLO model is not initialized.")
        
        if image is None or image.size == 0:
            return []

        try:
            results = self.model.predict(source=image, conf=self.conf_threshold, verbose=False)
            detections: List[DetectionResult] = []

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    label = self.model.names.get(cls_id, f"CLASS_{cls_id}")
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].tolist()
                    
                    bbox = BoundingBox(
                        x1=int(xyxy[0]),
                        y1=int(xyxy[1]),
                        x2=int(xyxy[2]),
                        y2=int(xyxy[3])
                    )
                    detections.append(DetectionResult(label=label, confidence=conf, bbox=bbox))

            return detections
        except Exception as e:
            raise LayoutDetectionError(f"YOLO prediction error: {e}")
