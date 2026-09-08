from typing import List
from core.schemas import MatchedIngredient

class FuzzyMatcher:
    """
    OCR 성분명과 데이터베이스 표준 성분명을 비교해 오타를 보정합니다.

    예를 들어 OCR 결과가 ``니아신아마이드``이고 DB에
    ``나이아신아마이드``가 있으면 두 문자열이 충분히 비슷할 때 DB 이름을
    사용합니다.
    """

    def __init__(self, threshold: int = 85):
        self.threshold = threshold
        self._has_rapidfuzz = False
        try:
            from rapidfuzz import fuzz, process
            self.fuzz = fuzz
            self.process = process
            self._has_rapidfuzz = True
        except ImportError:
            import difflib
            self.difflib = difflib
            self._has_rapidfuzz = False

    def _calc_similarity_difflib(self, first_name: str, second_name: str) -> float:
        """외부 라이브러리가 없을 때 사용하는 유사도 계산입니다."""
        first_clean = first_name.replace(" ", "")
        second_clean = second_name.replace(" ", "")
        seq = self.difflib.SequenceMatcher(None, first_clean, second_clean)
        return seq.ratio() * 100.0

    def match_single(self, raw_name: str, db_candidates: List[str]) -> MatchedIngredient:
        """
        단일 OCR 성분명에 대해 DB 후보군 중 가장 유사한 성분명을 탐색합니다.
        """
        cleaned_name = raw_name.strip()
        if not cleaned_name or not db_candidates:
            return MatchedIngredient(
                raw_ocr_name=cleaned_name,
                matched_name=cleaned_name,
                similarity=0.0,
                is_matched=False
            )

        # 가장 먼저 정확히 일치하는 이름을 처리합니다.
        if cleaned_name in db_candidates:
            return MatchedIngredient(
                raw_ocr_name=cleaned_name,
                matched_name=cleaned_name,
                similarity=100.0,
                is_matched=True
            )

        if self._has_rapidfuzz:
            best_match = self.process.extractOne(
                cleaned_name,
                db_candidates,
                # partial_ratio는 한국어 한 글자 오타처럼 짧은 이름의
                # 작은 차이를 더 잘 찾아냅니다.
                scorer=self.fuzz.partial_ratio,
            )
            if best_match:
                candidate_name, score, _ = best_match
                return self._make_result(cleaned_name, candidate_name, score)
        else:
            best_score = 0.0
            best_name = cleaned_name
            for candidate_name in db_candidates:
                score = self._calc_similarity_difflib(cleaned_name, candidate_name)
                if score > best_score:
                    best_score = score
                    best_name = candidate_name
            return self._make_result(cleaned_name, best_name, best_score)

        return MatchedIngredient(
            raw_ocr_name=cleaned_name,
            matched_name=cleaned_name,
            similarity=0.0,
            is_matched=False
        )

    def _make_result(
        self, raw_name: str, candidate_name: str, score: float
    ) -> MatchedIngredient:
        """유사도 점수를 최종 매칭 결과 객체로 변환합니다."""
        is_matched = score >= self.threshold
        return MatchedIngredient(
            raw_ocr_name=raw_name,
            matched_name=candidate_name if is_matched else raw_name,
            similarity=round(float(score), 1),
            is_matched=is_matched,
        )

    def match_all(self, raw_names: List[str], db_candidates: List[str]) -> List[MatchedIngredient]:
        """성분 목록 전체에 대해 Fuzzy Matching 수행"""
        return [self.match_single(name, db_candidates) for name in raw_names]
