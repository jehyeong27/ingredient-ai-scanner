from typing import List
import numpy as np
from core.schemas import DetectionResult, BoundingBox
from models.base_detector import BaseLayoutDetector

class MockYOLOModel(BaseLayoutDetector):
    """
    개발 및 테스트용 Mock Layout Detector
    실제 YOLO 모델이 없어도 전체 파이프라인이 정상 작동하도록 지원합니다.
    """

    def __init__(self, default_label: str = "INGREDIENT", confidence: float = 0.99):
        self.default_label = default_label
        self.confidence = confidence

    def predict(self, image: np.ndarray) -> List[DetectionResult]:
        if image is None or image.size == 0:
            return []

        height, width = image.shape[:2]

        # 전체 영역을 INGREDIENT 영역으로 반환
        return [
            DetectionResult(
                label=self.default_label,
                confidence=self.confidence,
                bbox=BoundingBox(x1=0, y1=0, x2=width, y2=height)
            )
        ]
