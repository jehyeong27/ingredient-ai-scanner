from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int

    def to_dict(self) -> Dict[str, int]:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}

    def to_list(self) -> List[int]:
        return [self.x1, self.y1, self.x2, self.y2]

@dataclass
class DetectionResult:
    label: str  # e.g., "INGREDIENT", "CAUTION", "OTHER"
    confidence: float
    bbox: BoundingBox

@dataclass
class OCRTextChunk:
    text: str
    confidence: float
    bbox: Optional[BoundingBox] = None

@dataclass
class MatchedIngredient:
    raw_ocr_name: str
    matched_name: str
    similarity: float
    is_matched: bool

@dataclass
class IngredientDetail:
    name: str
    function: str = ""
    efficacy: str = ""
    caution: str = ""
    raw_ocr: Optional[str] = None
    similarity: Optional[float] = None
    is_matched: bool = True

@dataclass
class LLMAnalysisResult:
    summary: str
    key_ingredients: List[Dict[str, str]] = field(default_factory=list)
    caution_summary: str = ""
    full_text: str = ""

@dataclass
class PipelineResult:
    original_image_shape: tuple
    detected_regions: List[DetectionResult]
    ocr_raw_text: str
    parsed_ingredients: List[str]
    matched_ingredients: List[MatchedIngredient]
    ingredient_details: List[IngredientDetail]
    unmatched_ingredients: List[str]
    llm_analysis: Optional[LLMAnalysisResult] = None
    latency_breakdown: Dict[str, float] = field(default_factory=dict)
