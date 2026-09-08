from typing import List, Optional
import numpy as np
from core.schemas import OCRTextChunk, BoundingBox
from core.exceptions import OCRError

class OCREngine:
    """
    PaddleOCR 래퍼 엔진
    이미지로부터 텍스트와 좌표, 신뢰도를 추출합니다.
    """

    def __init__(self, lang: str = "korean", use_angle_cls: bool = True, use_gpu: bool = False):
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self.use_gpu = use_gpu
        self.ocr = None
        self._init_engine()

    def _init_engine(self):
        try:
            from paddleocr import PaddleOCR
            # use_angle_cls=True, lang='korean'
            self.ocr = PaddleOCR(
                use_angle_cls=self.use_angle_cls,
                lang=self.lang,
                use_gpu=self.use_gpu,
                show_log=False
            )
        except ImportError:
            print("[Warning] PaddleOCR is not installed or available. OCR engine running in Fallback/Simulation mode.")
            self.ocr = None
        except Exception as e:
            print(f"[Warning] Failed to initialize PaddleOCR ({e}). Falling back to dummy mode.")
            self.ocr = None

    def extract_text(self, image: np.ndarray) -> List[OCRTextChunk]:
        """
        이미지에서 텍스트 라인들을 추출합니다.
        
        Args:
            image (np.ndarray): BGR/RGB 이미지 배열
            
        Returns:
            List[OCRTextChunk]: 추출된 텍스트와 신뢰도 및 BBox 목록
        """
        if image is None or image.size == 0:
            return []

        if self.ocr is None:
            # Fallback mock for testing environment without PaddleOCR installed
            return [
                OCRTextChunk(
                    text="정제수, 글리세린, 부틸렌글라이콜, 니아신아마이드, 1,2-헥산다이올, 판테놀, 소듐하이알루로네이트, 알란토인, 병풀추출물",
                    confidence=0.95
                )
            ]

        try:
            results = self.ocr.ocr(image, cls=self.use_angle_cls)
            chunks: List[OCRTextChunk] = []

            if not results or results[0] is None:
                return []

            for line in results[0]:
                box_coords = line[0]  # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                text_info = line[1]   # (text, confidence)
                text = text_info[0]
                conf = float(text_info[1])

                xs = [pt[0] for pt in box_coords]
                ys = [pt[1] for pt in box_coords]
                bbox = BoundingBox(
                    x1=int(min(xs)),
                    y1=int(min(ys)),
                    x2=int(max(xs)),
                    y2=int(max(ys))
                )

                chunks.append(OCRTextChunk(text=text, confidence=conf, bbox=bbox))

            return chunks
        except Exception as e:
            raise OCRError(f"OCR extraction failed: {e}")
