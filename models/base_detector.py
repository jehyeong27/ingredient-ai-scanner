from abc import ABC, abstractmethod
from typing import List
import numpy as np
from core.schemas import DetectionResult

class BaseLayoutDetector(ABC):
    """Layout Detection 모델을 위한 추상 베이스 클래스"""

    @abstractmethod
    def predict(self, image: np.ndarray) -> List[DetectionResult]:
        """
        입력 이미지에서 레이아웃 영역(성분, 주의사항 등)을 탐지합니다.
        
        Args:
            image (np.ndarray): BGR 또는 RGB 이미지 배열
            
        Returns:
            List[DetectionResult]: 탐지된 BoundingBox 및 라벨 목록
        """
        pass
