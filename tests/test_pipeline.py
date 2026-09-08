import unittest
import numpy as np
from pathlib import Path

from services.ingredient_parser import IngredientParser
from services.fuzzy_matcher import FuzzyMatcher
from services.ingredient_service import IngredientService
from services.pipeline import AnalysisPipeline
import config

class TestCosmeticAIPipeline(unittest.TestCase):

    def setUp(self):
        self.parser = IngredientParser()
        self.matcher = FuzzyMatcher(threshold=85)
        self.db_service = IngredientService(db_path=config.INGREDIENT_DB_PATH)
        self.pipeline = AnalysisPipeline()

    def test_parser_cleaning_and_splitting(self):
        sample_ocr = "[전성분] 정제수, 글리세린, 1,2-헥산다이올, 니아신아마이드, 판테놀"
        parsed = self.parser.parse(sample_ocr)
        
        self.assertIn("정제수", parsed)
        self.assertIn("글리세린", parsed)
        self.assertIn("1,2-헥산다이올", parsed)  # 1,2 속 쉼표가 보존되는지 확인
        self.assertIn("니아신아마이드", parsed)
        self.assertIn("판테놀", parsed)
        self.assertNotIn("[전성분]", parsed)

    def test_fuzzy_matcher_typo_correction(self):
        candidates = self.db_service.get_candidate_names()
        
        # 오타 입력: "니아신아마이드" -> DB: "나이아신아마이드"
        match = self.matcher.match_single("니아신아마이드", candidates)
        self.assertTrue(match.is_matched)
        self.assertEqual(match.matched_name, "나이아신아마이드")
        self.assertGreaterEqual(match.similarity, 85.0)

    def test_database_lookup(self):
        candidates = self.db_service.get_candidate_names()
        matched = self.matcher.match_all(["정제수", "글리세린", "판테놀"], candidates)
        details = self.db_service.get_details(matched)
        
        self.assertEqual(len(details), 3)
        self.assertTrue(all(d.is_matched for d in details))
        self.assertTrue(any("보습" in d.function for d in details))

    def test_full_pipeline_mock_run(self):
        # 100x100 가상 이미지로 파이프라인 호출
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = self.pipeline.analyze(dummy_img)

        self.assertIsNotNone(result)
        self.assertGreater(len(result.parsed_ingredients), 0)
        self.assertIsNotNone(result.llm_analysis)
        self.assertIn("total", result.latency_breakdown)
        print("\n[Pipeline Latency Breakdown]:", result.latency_breakdown)

if __name__ == "__main__":
    unittest.main()
