import time
from typing import Dict, Optional
import numpy as np

import config
from core.schemas import (
    PipelineResult,
    DetectionResult,
    MatchedIngredient,
    IngredientDetail,
    LLMAnalysisResult,
    BoundingBox
)
from services.preprocessor import ImagePreprocessor
from services.ingredient_parser import IngredientParser
from services.fuzzy_matcher import FuzzyMatcher
from services.ingredient_service import IngredientService
from services.llm_service import LLMService
from models.base_detector import BaseLayoutDetector
from models.mock_yolo import MockYOLOModel
from models.ocr_engine import OCREngine

class AnalysisPipeline:
    """이미지 분석의 전체 순서를 관리하는 클래스입니다.

    이 클래스는 직접 OCR이나 LLM을 수행하지 않습니다. 각 작업을 담당하는
    작은 객체들을 순서대로 호출하고, 마지막에 ``PipelineResult``로 묶습니다.

    전체 흐름:
    이미지 -> 전처리 -> 영역 탐지 -> 성분 영역 자르기 -> OCR -> 파싱
    -> 오타 보정 -> DB 조회 -> LLM 요약
    """

    def __init__(
        self,
        layout_detector: Optional[BaseLayoutDetector] = None,
        ocr_engine: Optional[OCREngine] = None,
        preprocessor: Optional[ImagePreprocessor] = None,
        parser: Optional[IngredientParser] = None,
        fuzzy_matcher: Optional[FuzzyMatcher] = None,
        ingredient_service: Optional[IngredientService] = None,
        llm_service: Optional[LLMService] = None
    ):
        # 테스트에서 가짜 객체를 전달하면 실제 모델을 만들지 않습니다.
        # 이 패턴 덕분에 각 부품을 따로 연습하거나 테스트할 수 있습니다.
        self.detector = layout_detector or self._create_detector()

        # 2. OCR Engine
        self.ocr_engine = ocr_engine or OCREngine(
            lang=config.OCR_LANG,
            use_angle_cls=config.OCR_USE_ANGLE_CLS,
            use_gpu=config.OCR_USE_GPU
        )

        # 3. Services
        self.preprocessor = preprocessor or ImagePreprocessor()
        self.parser = parser or IngredientParser()
        self.fuzzy_matcher = fuzzy_matcher or FuzzyMatcher(threshold=config.FUZZY_MATCH_THRESHOLD)
        self.ingredient_service = ingredient_service or IngredientService(db_path=config.INGREDIENT_DB_PATH)
        self.llm_service = llm_service or LLMService(
            provider=config.LLM_PROVIDER,
            openai_api_key=config.OPENAI_API_KEY,
            openai_model=config.OPENAI_MODEL,
            gemini_api_key=config.GEMINI_API_KEY,
            gemini_model=config.GEMINI_MODEL
        )

    @staticmethod
    def _create_detector() -> BaseLayoutDetector:
        """설정에 따라 실제 YOLO 또는 테스트용 Mock을 만듭니다."""
        should_use_real_yolo = (
            not config.USE_MOCK_YOLO and config.YOLO_MODEL_PATH.exists()
        )
        if not should_use_real_yolo:
            return MockYOLOModel()

        try:
            from models.yolo_detector import YOLOLayoutDetector

            return YOLOLayoutDetector(
                model_path=config.YOLO_MODEL_PATH,
                conf_threshold=config.YOLO_CONF_THRESHOLD,
            )
        except Exception as error:
            print(f"[Warning] Failed to load real YOLO ({error}). Using MockYOLO.")
            return MockYOLOModel()

    def _select_ingredient_box(
        self,
        detections: list[DetectionResult],
        image_height: int,
        image_width: int,
    ) -> BoundingBox:
        """OCR에 사용할 영역을 고릅니다.

        INGREDIENT 영역을 우선하고, 없으면 첫 번째 탐지 영역을 사용합니다.
        탐지 결과가 전혀 없으면 이미지 전체를 사용합니다.
        """
        for detection in detections:
            if "INGREDIENT" in detection.label.upper():
                return detection.bbox

        if detections:
            return detections[0].bbox
        return BoundingBox(x1=0, y1=0, x2=image_width, y2=image_height)

    def analyze(self, image: np.ndarray) -> PipelineResult:
        """
        이미지 1장을 입력받아 End-to-End 전체 분석을 실행합니다.
        """
        latencies: Dict[str, float] = {}
        image_height, image_width = image.shape[:2]

        # 각 단계 전후의 시간을 재서 화면에 보여줍니다.
        step_start_time = time.time()
        processed_img = self.preprocessor.preprocess(image)
        latencies["preprocessing"] = round(time.time() - step_start_time, 3)

        step_start_time = time.time()
        detections = self.detector.predict(processed_img)
        latencies["layout_detection"] = round(time.time() - step_start_time, 3)

        step_start_time = time.time()
        ingredient_box = self._select_ingredient_box(
            detections, 
            image_height, 
            image_width
        )
        cropped_img = self.preprocessor.crop_region(
            processed_img,
            ingredient_box
        )
        latencies["crop"] = round(time.time() - step_start_time, 3)

        step_start_time = time.time()
        ocr_chunks = self.ocr_engine.extract_text(cropped_img)
        raw_ocr_text = "\n".join(chunk.text for chunk in ocr_chunks)
        latencies["ocr"] = round(time.time() - step_start_time, 3)

        step_start_time = time.time()
        parsed_names = self.parser.parse(raw_ocr_text)
        latencies["parsing"] = round(time.time() - step_start_time, 3)

        step_start_time = time.time()
        db_candidates = self.ingredient_service.get_candidate_names()
        matched_results = self.fuzzy_matcher.match_all(
            parsed_names, 
            db_candidates
        )
        latencies["fuzzy_matching"] = round(time.time() - step_start_time, 3)

        step_start_time = time.time()
        ingredient_details = self.ingredient_service.get_details(matched_results)
        unmatched_names = [
            match.raw_ocr_name
            for match in matched_results
            if not match.is_matched
        ]
        latencies["db_lookup"] = round(time.time() - step_start_time, 3)

        step_start_time = time.time()
        llm_analysis = self.llm_service.analyze(ingredient_details)
        latencies["llm_analysis"] = round(time.time() - step_start_time, 3)

        latencies["total"] = round(sum(latencies.values()), 3)

        return PipelineResult(
            original_image_shape=(image_height, image_width),
            detected_regions=detections,
            ocr_raw_text=raw_ocr_text,
            parsed_ingredients=parsed_names,
            matched_ingredients=matched_results,
            ingredient_details=ingredient_details,
            unmatched_ingredients=unmatched_names,
            llm_analysis=llm_analysis,
            latency_breakdown=latencies
        )
