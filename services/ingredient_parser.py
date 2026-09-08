import re
from typing import List

class IngredientParser:
    """
    OCR 텍스트를 성분 이름 목록으로 바꾸는 규칙 기반 파서입니다.

    파싱은 항상 다음 세 단계로 진행됩니다.
    1. clean_text: 제목과 불필요한 문장을 제거합니다.
    2. split_ingredients: 쉼표 등의 구분자로 성분을 나눕니다.
    3. remove_duplicates: 같은 성분을 한 번만 남깁니다.
    """

    PREFIX_PATTERNS = [
        r"^\[?\s*전성분\s*\]?\s*[:\-]?\s*",
        r"^\[?\s*주요성분\s*\]?\s*[:\-]?\s*",
        r"^\[?\s*성분\s*\]?\s*[:\-]?\s*",
        r"^\[?\s*Ingredients\s*\]?\s*[:\-]?\s*",
        r"^\[?\s*All\s+Ingredients\s*\]?\s*[:\-]?\s*"
    ]

    NOISE_PATTERNS = [
        r"사용\s*시\s*주의사항.*",
        r"화장품책임판매업자.*",
        r"제조업자.*",
        r"용량\s*:\s*\d+.*",
        r"www\..*",
        r"MADE\s+IN\s+.*",
        r"고객상담실.*"
    ]

    def __init__(self):
        self.prefix_regexes = [re.compile(p, re.IGNORECASE) for p in self.PREFIX_PATTERNS]
        self.noise_regexes = [re.compile(p, re.IGNORECASE) for p in self.NOISE_PATTERNS]

    def clean_text(self, text: str) -> str:
        """텍스트 정제 (접두어 제거, 줄바꿈 통일, 노이즈 제거)"""
        if not text:
            return ""

        # OCR은 줄바꿈 형식과 불필요한 공백이 일정하지 않을 수 있습니다.
        cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
        
        # 불필요한 노이즈 라인 필터링
        useful_lines = []
        for line in cleaned.split("\n"):
            stripped_line = line.strip()
            if not stripped_line:
                continue
            contains_noise = False
            for noise_regex in self.noise_regexes:
                if noise_regex.search(stripped_line):
                    contains_noise = True
                    break
            if not contains_noise:
                useful_lines.append(stripped_line)
        
        joined = " ".join(useful_lines)

        # 접두어 제거
        for p_reg in self.prefix_regexes:
            joined = p_reg.sub("", joined).strip()

        # 다중 공백 정리
        joined = re.sub(r"\s+", " ", joined).strip()
        return joined

    def split_ingredients(self, text: str) -> List[str]:
        """구분자(쉼표, 슬래시, 가운데점 등)를 기준으로 성분 분리"""
        if not text:
            return []

        # 쉼표(,), 가운데점(·), 슬래시(/), 세미콜론(;) 등으로 분리
        # 단, '1,2-헥산다이올'과 같은 숫자 사이의 쉼표는 분리되지 않도록 처리
        # 정규식: (?<!\d),(?!\d) -> 앞뒤가 숫자가 아닌 쉼표
        # 임시 플레이스홀더 치환 방식 활용
        temp_text = re.sub(r"(\d+),(\d+)", r"\1__COMMA__\2", text)
        
        # 구분자로 분리: , / · • ; |
        raw_items = re.split(r"[,/·•;|]", temp_text)
        
        ingredient_names: List[str] = []
        for raw_item in raw_items:
            restored_item = raw_item.replace("__COMMA__", ",")
            cleaned_item = re.sub(r"[^\w\s\-\(\),]", "", restored_item).strip()
            cleaned_item = re.sub(r"\s+", " ", cleaned_item).strip()
            if len(cleaned_item) >= 2:
                ingredient_names.append(cleaned_item)

        return ingredient_names

    def remove_duplicates(self, ingredients: List[str]) -> List[str]:
        """순서를 유지하면서 중복 성분 제거"""
        seen_names = set()
        unique_names = []
        for ingredient_name in ingredients:
            normalized_name = ingredient_name.replace(" ", "")
            if normalized_name not in seen_names:
                seen_names.add(normalized_name)
                unique_names.append(ingredient_name)
        return unique_names

    def parse(self, raw_text: str) -> List[str]:
        """
        전체 파싱 파이프라인 (Clean -> Split -> Deduplicate)
        """
        cleaned = self.clean_text(raw_text)
        splits = self.split_ingredients(cleaned)
        return self.remove_duplicates(splits)
