import json
from pathlib import Path
from typing import Dict, List, Optional
from core.schemas import IngredientDetail, MatchedIngredient
from core.exceptions import IngredientDBError

class IngredientService:
    """
    JSON 기반 성분 지식 데이터베이스 조회 및 관리 서비스
    """

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db: Dict[str, Dict[str, str]] = {}
        self.ingredient_names: List[str] = []
        self._load_database()

    def _load_database(self):
        """JSON 데이터베이스 로드 및 인덱싱"""
        if not self.db_path.exists():
            print(f"[Warning] Ingredient DB not found at {self.db_path}. Initializing empty DB.")
            self.db = {}
            self.ingredient_names = []
            return

        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                self.db = json.load(f)
            self.ingredient_names = list(self.db.keys())
            print(f"Loaded {len(self.ingredient_names)} ingredients from database.")
        except Exception as e:
            raise IngredientDBError(f"Failed to load ingredient database: {e}")

    def get_candidate_names(self) -> List[str]:
        """Fuzzy Matching에 사용할 성분명 후보 목록 반환"""
        return self.ingredient_names

    def get_ingredient_info(self, name: str) -> Optional[Dict[str, str]]:
        """단일 성분의 상세 정보 조회"""
        return self.db.get(name)

    def get_details(self, matched_items: List[MatchedIngredient]) -> List[IngredientDetail]:
        """
        Fuzzy Matching 결과 목록을 받아 상세 정보 리스트로 변환합니다.
        """
        details: List[IngredientDetail] = []
        for item in matched_items:
            if item.is_matched and item.matched_name in self.db:
                info = self.db[item.matched_name]
                details.append(
                    IngredientDetail(
                        name=item.matched_name,
                        function=info.get("function", ""),
                        efficacy=info.get("efficacy", ""),
                        caution=info.get("caution", ""),
                        raw_ocr=item.raw_ocr_name,
                        similarity=item.similarity,
                        is_matched=True
                    )
                )
            else:
                # 미확인 성분 (DB에 없음)
                details.append(
                    IngredientDetail(
                        name=item.raw_ocr_name,
                        function="미분류 성분",
                        efficacy="데이터베이스에 상세 정보가 등록되지 않은 성분입니다.",
                        caution="-",
                        raw_ocr=item.raw_ocr_name,
                        similarity=item.similarity,
                        is_matched=False
                    )
                )
        return details
