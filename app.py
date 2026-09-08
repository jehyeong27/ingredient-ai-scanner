import cv2
import numpy as np
import pandas as pd
import gradio as gr
from PIL import Image

import config
from services.pipeline import AnalysisPipeline
from core.schemas import PipelineResult

# 앱이 시작될 때 분석에 필요한 부품들을 한 번 준비합니다.
# 이후 사용자가 이미지를 업로드할 때 같은 파이프라인을 재사용합니다.
pipeline = AnalysisPipeline()

# 세션 상태 저장을 위한 변수
current_pipeline_result: dict = {"result": None}

def draw_bboxes(image: np.ndarray, detections) -> np.ndarray:
    """이미지 위에 YOLO BoundingBox 및 라벨을 그립니다."""
    vis_img = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2
        label = f"{det.label} ({det.confidence:.2f})"
        
        color = (0, 200, 0) if "INGREDIENT" in det.label.upper() else (0, 165, 255)
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 3)
        
        # 라벨 텍스트 배경
        cv2.putText(vis_img, label, (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return vis_img

def process_cosmetic_image(uploaded_image):
    """
    사용자가 업로드한 이미지를 받아 전체 파이프라인 분석을 수행하고 UI 결과를 생성합니다.
    """
    if uploaded_image is None:
        return (
            None,
            "⚠️ 이미지를 업로드해주세요.",
            pd.DataFrame(),
            "⚠️ 이미지가 업로드되지 않았습니다.",
            ""
        )

    # PIL Image -> NumPy BGR
    image_rgb = np.array(uploaded_image)
    if len(image_rgb.shape) == 3 and image_rgb.shape[2] == 3:
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image_rgb

    # 1. 이미지 분석: 전처리부터 LLM 요약까지 pipeline이 순서대로 처리합니다.
    result: PipelineResult = pipeline.analyze(image_bgr)
    current_pipeline_result["result"] = result

    # 2. 탐지 영역을 원본 이미지 위에 그립니다.
    visualized_bgr = draw_bboxes(image_bgr, result.detected_regions)
    visualized_rgb = cv2.cvtColor(visualized_bgr, cv2.COLOR_BGR2RGB)

    # 3. 화면의 표에 넣을 행을 만듭니다.
    table_rows = []
    for item in result.ingredient_details:
        status_badge = f"✅ 매칭 ({item.similarity}%)" if item.is_matched else "⚠️ 미확인"
        table_rows.append({
            "상태": status_badge,
            "성분명 (표준)": item.name,
            "OCR 원문": item.raw_ocr or "-",
            "주요 기능": item.function,
            "효능 및 특징": item.efficacy,
            "주의사항": item.caution
        })
    df_ingredients = pd.DataFrame(table_rows)

    # 4. LLM 분석 결과를 화면에 표시할 문자열로 꺼냅니다.
    llm_report = result.llm_analysis.full_text if result.llm_analysis else "분석 결과가 없습니다."

    # 5. 각 단계에 걸린 시간을 읽기 쉬운 문장으로 만듭니다.
    latencies = result.latency_breakdown
    metrics_text = (
        f"⏱️ **처리 시간 요약**: 총 {latencies.get('total', 0)}초 "
        f"(전처리: {latencies.get('preprocessing', 0)}s | "
        f"YOLO: {latencies.get('layout_detection', 0)}s | "
        f"OCR: {latencies.get('ocr', 0)}s | "
        f"매칭: {latencies.get('fuzzy_matching', 0)}s | "
        f"LLM: {latencies.get('llm_analysis', 0)}s)"
    )

    # 6. OCR이 처음 읽은 원문도 별도로 보여줍니다.
    ocr_preview = f"**[추출된 원문 텍스트]**\n```\n{result.ocr_raw_text}\n```"

    return (
        visualized_rgb,
        llm_report,
        df_ingredients,
        metrics_text,
        ocr_preview
    )

def handle_chat_qa(user_message, chat_history):
    """
    현재 분석된 성분 DB를 바탕으로 사용자의 질문에 답변합니다.
    """
    if not user_message.strip():
        return "", chat_history

    result: PipelineResult = current_pipeline_result.get("result")
    if result is None or not result.ingredient_details:
        bot_response = "먼저 상단에서 화장품 성분표 이미지를 업로드하고 [성분표 분석 시작] 버튼을 눌러주세요."
    else:
        bot_response = pipeline.llm_service.answer_question(
            user_message,
            result.ingredient_details
        )

    chat_history.append((user_message, bot_response))
    return "", chat_history

# ==================== Gradio UI Layout ====================
custom_css = """
#main-title { text-align: center; margin-bottom: 5px; }
#sub-title { text-align: center; color: #555; margin-bottom: 20px; }
.metric-box { background-color: #f8f9fa; padding: 10px; border-radius: 8px; margin-bottom: 10px; }
"""

with gr.Blocks(title="CosmeticAI - AI 화장품 성분 분석 서비스", css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧴 CosmeticAI", elem_id="main-title")
    gr.Markdown(
        "**YOLO 영역 탐지 → PaddleOCR 텍스트 추출 → Rule-based Parser → Fuzzy Matching 오타 보정 → 성분 DB 조회 → LLM 분석**",
        elem_id="sub-title"
    )

    with gr.Row():
        # 왼쪽 컬럼: 이미지 입력 & 영역 탐지 시각화
        with gr.Column(scale=1):
            gr.Markdown("### 📸 1. 성분표 이미지 업로드")
            input_image = gr.Image(
                label="화장품 사진 업로드 (JPG, PNG)",
                type="pil"
            )
            analyze_btn = gr.Button("🔍 성분표 분석 시작", variant="primary", size="lg")
            
            gr.Markdown("### 🎯 2. 영역 탐지 결과 (YOLO/Mock)")
            output_vis_image = gr.Image(label="탐지 영역 시각화 (INGREDIENT BBox)", interactive=False)
            metrics_display = gr.Markdown("⏱️ 분석을 시작하면 처리 시간이 표시됩니다.", elem_classes=["metric-box"])

        # 오른쪽 컬럼: LLM 분석 요약 & 성분 데이터 테이블
        with gr.Column(scale=2):
            gr.Markdown("### 🧠 3. AI 성분 종합 분석 보고서 (LLM)")
            llm_summary_output = gr.Markdown("이미지를 업로드하고 분석을 시작하면 AI 요약 보고서가 생성됩니다.")
            
            gr.Markdown("### 📋 4. 검출 및 매칭된 성분 목록")
            ingredients_table = gr.DataFrame(
                headers=["상태", "성분명 (표준)", "OCR 원문", "주요 기능", "효능 및 특징", "주의사항"],
                datatype=["str", "str", "str", "str", "str", "str"],
                interactive=False,
                wrap=True
            )

            with gr.Accordion("🔍 OCR 인식 원문 보기", open=False):
                ocr_raw_output = gr.Markdown("추출된 원문 텍스트가 여기에 표시됩니다.")

    gr.Markdown("---")
    
    # 하단: Q&A 챗봇
    gr.Markdown("### 💬 5. 성분 Q&A 챗봇 (현재 제품 전용)")
    gr.Markdown("분석된 성분 정보에 대해 무엇이든 물어보세요. (예: *'보습 성분만 알려줘'*, *'나이아신아마이드는 무슨 역할이야?'*, *'주의할 성분 있어?'*)")
    
    chatbot = gr.Chatbot()
    with gr.Row():
        chat_input = gr.Textbox(
            placeholder="성분에 대해 궁금한 점을 입력하세요...",
            scale=8,
            show_label=False
        )
        chat_submit = gr.Button("질문하기", scale=1, variant="secondary")

    # 이벤트 연결
    analyze_btn.click(
        fn=process_cosmetic_image,
        inputs=[input_image],
        outputs=[
            output_vis_image,
            llm_summary_output,
            ingredients_table,
            metrics_display,
            ocr_raw_output
        ]
    )

    chat_submit.click(
        fn=handle_chat_qa,
        inputs=[chat_input, chatbot],
        outputs=[chat_input, chatbot]
    )
    chat_input.submit(
        fn=handle_chat_qa,
        inputs=[chat_input, chatbot],
        outputs=[chat_input, chatbot]
    )

if __name__ == "__main__":
    demo.launch(
        server_name=config.GRADIO_SERVER_NAME,
        server_port=config.GRADIO_SERVER_PORT,
        share=False
    )
