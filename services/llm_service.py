import os
import json
from typing import List, Dict, Optional
from core.schemas import IngredientDetail, LLMAnalysisResult
from core.exceptions import LLMServiceError

class LLMService:
    """
    구조화된 성분 정보를 바탕으로 요약 보고서 생성 및 Q&A를 수행하는 LLM 서비스
    """

    SYSTEM_PROMPT = """당신은 전문 화장품 성분 분석 AI 어시스턴트 'CosmeticAI'입니다.
제공된 [화장품 성분 데이터베이스 목록]을 바탕으로 사용자가 이해하기 쉽게 분석하여 설명해주세요.

[엄격한 준수 원칙]
1. 제공된 성분 목록에 존재하지 않는 성분을 임의로 생성하거나 지어내지 마십시오.
2. 성분의 기능, 효능, 주의사항은 제공된 데이터베이스 내용을 바탕으로 객관적으로 서술하십시오.
3. 특정 질병 치료나 의학적 효능으로 단정 짓지 마십시오.
4. 사용자가 한눈에 파악하기 좋게 마크다운 서식을 활용해 정중하고 친절한 한국어로 답변하십시오.
"""

    def __init__(
        self,
        provider: str = "fallback",
        openai_api_key: str = "",
        openai_model: str = "gpt-4o-mini",
        gemini_api_key: str = "",
        gemini_model: str = "gemini-2.5-flash"
    ):
        self.provider = provider
        self.openai_api_key = openai_api_key
        self.openai_model = openai_model
        self.gemini_api_key = gemini_api_key
        self.gemini_model = gemini_model

    def _build_context(self, ingredient_details: List[IngredientDetail]) -> str:
        """성분 목록을 LLM 입력용 텍스트 컨텍스트로 변환"""
        lines = []
        for idx, item in enumerate(ingredient_details, 1):
            status = "확인됨" if item.is_matched else "미확인"
            line = f"{idx}. {item.name} ({status})\n   - 기능: {item.function}\n   - 효능/설명: {item.efficacy}\n   - 주의사항: {item.caution}"
            lines.append(line)
        return "\n".join(lines)

    def _fallback_analyze(self, ingredient_details: List[IngredientDetail]) -> LLMAnalysisResult:
        """API 키가 없거나 실패 시 동작하는 템플릿 기반 Fallback 요약기"""
        matched_items = [item for item in ingredient_details if item.is_matched]
        unmatched_items = [item for item in ingredient_details if not item.is_matched]

        # 주요 기능 추출
        functions = set()
        for item in matched_items:
            for f in item.function.split("/"):
                f_clean = f.strip()
                if f_clean:
                    functions.add(f_clean)

        func_summary = ", ".join(list(functions)[:4]) if functions else "기본 보습 및 제형 안정화"

        summary_text = (
            f"이 제품은 총 **{len(ingredient_details)}가지 성분**(확인 성분 {len(matched_items)}종, 미확인 {len(unmatched_items)}종)으로 구성되어 있습니다.\n"
            f"주요 특징으로는 **{func_summary}** 기능을 중심으로 균형 있게 배합되어 있습니다."
        )

        key_ingredients = []
        for item in matched_items[:5]:  # 상위 5개 성분
            key_ingredients.append({
                "name": item.name,
                "point": f"[{item.function}] {item.efficacy}"
            })

        cautions = [f"• **{item.name}**: {item.caution}" for item in matched_items if item.caution and item.caution != "-"]
        caution_text = "\n".join(cautions) if cautions else "특별히 주의해야 할 위험 성분이 발견되지 않았습니다."

        full_text = f"### 🌿 AI 성분 종합 분석 요약\n\n{summary_text}\n\n"
        full_text += "### ✨ 주요 핵심 성분\n"
        for k in key_ingredients:
            full_text += f"- **{k['name']}**: {k['point']}\n"
        full_text += f"\n### ⚠️ 주의사항 및 참고사항\n{caution_text}"

        return LLMAnalysisResult(
            summary=summary_text,
            key_ingredients=key_ingredients,
            caution_summary=caution_text,
            full_text=full_text
        )

    def analyze(self, ingredient_details: List[IngredientDetail]) -> LLMAnalysisResult:
        """성분 목록을 LLM에 전달하여 종합 분석 보고서 생성"""
        if not ingredient_details:
            return LLMAnalysisResult(
                summary="검출된 성분이 없습니다.",
                key_ingredients=[],
                caution_summary="",
                full_text="검출된 성분이 없어 분석을 진행할 수 없습니다."
            )

        context = self._build_context(ingredient_details)
        prompt = (
            f"다음은 화장품에서 추출 및 확인된 성분 목록입니다:\n\n{context}\n\n"
            f"위 성분 정보를 바탕으로 다음 세 가지 항목을 작성해주세요:\n"
            f"1. 전체 제품의 성분 특징 요약 (2~3문장)\n"
            f"2. 주요 핵심 성분의 기능 및 효능 설명\n"
            f"3. 주의해야 할 성분 및 팁\n"
        )

        if self.provider == "openai" and self.openai_api_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=self.openai_api_key)
                response = client.chat.completions.create(
                    model=self.openai_model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3
                )
                full_text = response.choices[0].message.content
                return LLMAnalysisResult(
                    summary="OpenAI 기반 성분 종합 분석 완료",
                    full_text=full_text
                )
            except Exception as e:
                print(f"[Warning] OpenAI API call failed: {e}. Using fallback generator.")

        elif self.provider == "gemini" and self.gemini_api_key:
            try:
                from google import genai
                client = genai.Client(api_key=self.gemini_api_key)
                response = client.models.generate_content(
                    model=self.gemini_model,
                    contents=f"{self.SYSTEM_PROMPT}\n\n{prompt}"
                )
                full_text = response.text
                return LLMAnalysisResult(
                    summary="Gemini 기반 성분 종합 분석 완료",
                    full_text=full_text
                )
            except Exception as e:
                print(f"[Warning] Gemini API call failed: {e}. Using fallback generator.")

        # Fallback Generator
        return self._fallback_analyze(ingredient_details)

    def answer_question(self, question: str, ingredient_details: List[IngredientDetail]) -> str:
        """
        사용자의 후속 질문에 대해 현재 분석된 성분 DB를 바탕으로 답변
        """
        if not ingredient_details:
            return "현재 분석된 성분 정보가 없습니다. 먼저 화장품 성분 이미지를 분석해주세요."

        context = self._build_context(ingredient_details)
        prompt = (
            f"[현재 분석된 화장품 성분 목록]\n{context}\n\n"
            f"[사용자 질문]\n{question}\n\n"
            f"위 성분 목록에 근거하여 정확하고 친절하게 답변해주세요."
        )

        if self.provider == "openai" and self.openai_api_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=self.openai_api_key)
                response = client.chat.completions.create(
                    model=self.openai_model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"[Warning] OpenAI Q&A failed: {e}. Using rule fallback.")

        # Fallback Question Answering Logic
        q_lower = question.lower()
        matched_hits = []
        for item in ingredient_details:
            if any(k in q_lower for k in [item.name.lower(), item.function.lower(), item.efficacy.lower()]):
                matched_hits.append(item)

        if "보습" in q_lower:
            moisturizers = [item.name for item in ingredient_details if "보습" in item.function or "수분" in item.efficacy]
            if moisturizers:
                return f"현재 제품에서 확인된 대표적인 보습 성분은 **{', '.join(moisturizers)}** 입니다."
            return "현재 제품에서 뚜렷한 보습 전용 성분은 확인되지 않았습니다."

        if "주의" in q_lower or "부작용" in q_lower or "위험" in q_lower:
            cautions = [f"• **{item.name}**: {item.caution}" for item in ingredient_details if item.caution and item.caution != "-"]
            if cautions:
                return "확인된 성분 중 참고해야 할 주의사항입니다:\n" + "\n".join(cautions)
            return "현재 확인된 성분 중 특별한 고위험 주의 성분은 발견되지 않았습니다."

        if matched_hits:
            res = f"질문과 관련된 성분 정보입니다:\n"
            for h in matched_hits[:3]:
                res += f"\n• **{h.name}**\n  - 기능: {h.function}\n  - 효능: {h.efficacy}\n  - 주의사항: {h.caution}"
            return res

        return f"현재 분석된 제품의 성분({len(ingredient_details)}종)을 바탕으로 확인한 결과, 질문하신 내용에 대한 추가 정보입니다:\n- 분석 성분: {', '.join([i.name for i in ingredient_details[:6]])} 등"
