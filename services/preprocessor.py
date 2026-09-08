from typing import Optional, Tuple
import cv2
import numpy as np
from core.schemas import BoundingBox
from core.exceptions import ImagePreprocessingError

class ImagePreprocessor:
    """
    OpenCV 기반 이미지 전처리 및 영역 Crop 서비스
    """

    def __init__(self, max_dimension: int = 1920):
        self.max_dimension = max_dimension

    def resize_if_needed(self, image: np.ndarray) -> np.ndarray:
        """이미지가 너무 큰 경우 비율을 유지하며 축소"""
        h, w = image.shape[:2]
        if max(h, w) > self.max_dimension:
            scale = self.max_dimension / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return image

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """CLAHE(Contrast Limited Adaptive Histogram Equalization)를 사용한 대비 개선"""
        if len(image.shape) == 3:
            # LAB 색공간의 L 채널에 CLAHE 적용 후 복원
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)

    def sharpen(self, image: np.ndarray) -> np.ndarray:
        """텍스트 외곽선을 돋보이게 하는 Unsharp Masking 적용"""
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
        sharpened = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        return sharpened

    def preprocess(self, image: np.ndarray, apply_enhancement: bool = True) -> np.ndarray:
        """
        기본 전처리 파이프라인 (Resize -> Contrast -> Sharpen)
        """
        if image is None or image.size == 0:
            raise ImagePreprocessingError("Empty or invalid image provided.")

        try:
            processed = self.resize_if_needed(image)
            if apply_enhancement:
                processed = self.enhance_contrast(processed)
                processed = self.sharpen(processed)
            return processed
        except Exception as e:
            raise ImagePreprocessingError(f"Image preprocessing failed: {e}")

    def crop_region(self, image: np.ndarray, bbox: BoundingBox, margin_ratio: float = 0.02) -> np.ndarray:
        """
        BoundingBox 영역을 기준으로 약간의 마진을 포함하여 Crop합니다.
        """
        if image is None or image.size == 0:
            raise ImagePreprocessingError("Empty image for cropping.")

        h, w = image.shape[:2]

        # 마진 계산
        margin_x = int((bbox.x2 - bbox.x1) * margin_ratio)
        margin_y = int((bbox.y2 - bbox.y1) * margin_ratio)

        x1 = max(0, bbox.x1 - margin_x)
        y1 = max(0, bbox.y1 - margin_y)
        x2 = min(w, bbox.x2 + margin_x)
        y2 = min(h, bbox.y2 + margin_y)

        if x2 <= x1 or y2 <= y1:
            return image  # Fallback to full image if bbox is invalid

        return image[y1:y2, x1:x2]
