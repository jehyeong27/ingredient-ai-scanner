# CosmeticAI — AI 기반 화장품 성분표 분석

화장품 전성분표 이미지를 분석해 OCR 텍스트, 표준 성분명, 성분별 설명과 요약 보고서를 제공하는 Gradio 웹 애플리케이션입니다. 분석 결과에 대해 질문할 수 있는 제품별 Q&A도 포함합니다.

## 주요 기능

- 성분표 영역 탐지: 기본 Mock 탐지기 또는 선택적 YOLO 모델
- 한글 성분 텍스트 추출: PaddleOCR
- OCR 결과 파싱 및 RapidFuzz 기반 성분명 유사도 매칭
- `data/processed/ingredients.json` 성분 데이터 조회
- 템플릿 기반 분석 또는 OpenAI/Gemini를 이용한 분석 및 Q&A
- 탐지 결과 시각화, OCR 원문, 단계별 처리 시간 표시

## 처리 흐름

```text
이미지 입력
  → 이미지 전처리
  → 성분표 영역 탐지 및 자르기
  → OCR 텍스트 추출
  → 성분명 파싱 및 오타 보정
  → 성분 데이터 조회
  → 분석 보고서 및 Q&A
```

## 시작하기

Python 3.10 이상을 권장합니다.

1. 저장소를 내려받고 프로젝트 폴더로 이동합니다.

   ```bash
   git clone https://github.com/jehyeong27/ingredient-ai-scanner.git
   cd ingredient-ai-scanner
   ```

2. 가상 환경을 만들고 활성화합니다.

   ```bash
   python -m venv .venv
   ```

   Windows PowerShell:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   macOS/Linux:

   ```bash
   source .venv/bin/activate
   ```

3. 의존성을 설치합니다.

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

   PaddlePaddle 설치는 OS와 하드웨어에 따라 추가 설정이 필요할 수 있습니다. 설치 방법은 [PaddleOCR 설치 안내](https://github.com/PaddlePaddle/PaddleOCR)를 참고하세요.

4. 예제 환경 변수 파일을 `.env`로 복사합니다.

   Windows PowerShell:

   ```powershell
   Copy-Item .env.example .env
   ```

   macOS/Linux:

   ```bash
   cp .env.example .env
   ```

   `.env`를 열어 필요한 설정과 API 키를 입력하세요. 기본 설정은 Mock YOLO 및 템플릿 기반 분석이므로 API 키 없이도 실행할 수 있습니다.

5. 애플리케이션을 실행합니다.

   ```bash
   python app.py
   ```

6. 브라우저에서 <http://127.0.0.1:7860>을 열고 화장품 성분표 이미지를 업로드합니다.

## 실행 모드 및 환경 변수

별도 설정 없이 실행하면 Mock YOLO 탐지기와 템플릿 기반 LLM 분석을 사용합니다. PaddleOCR가 설치되지 않았거나 초기화에 실패하면 코드에 포함된 fallback 텍스트가 사용됩니다. 실제 이미지 OCR 결과를 확인하려면 코드와 호환되는 PaddleOCR 및 PaddlePaddle을 설치해야 합니다.

프로젝트 루트의 `.env` 파일을 앱 시작 시 읽습니다. 이미 운영체제 환경 변수에 같은 이름의 값이 설정되어 있으면 운영체제 값이 우선합니다.

| 환경 변수 | 기본값 | 설명 |
| --- | --- | --- |
| `USE_MOCK_YOLO` | `true` | `false`로 설정하고 `weights/best.pt` 모델 파일이 있으면 실제 YOLO 탐지기를 사용합니다. |
| `YOLO_CONF_THRESHOLD` | `0.25` | YOLO 탐지 신뢰도 임계값입니다. |
| `OCR_LANG` | `korean` | PaddleOCR 언어 설정입니다. |
| `OCR_USE_GPU` | `false` | PaddleOCR에서 GPU를 사용할지 설정합니다. |
| `FUZZY_MATCH_THRESHOLD` | `85` | 성분명 유사도 매칭 임계값입니다. |
| `LLM_PROVIDER` | `fallback` | `fallback`, `openai`, `gemini` 중 분석 제공자를 선택합니다. |
| `OPENAI_API_KEY` | 비어 있음 | OpenAI 사용 시 API 키입니다. |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI 모델 이름입니다. |
| `GEMINI_API_KEY` | 비어 있음 | Gemini 사용 시 API 키입니다. |
| `GEMINI_MODEL` | `gemini-3.6-flash` | Gemini 모델 이름입니다. |
| `GRADIO_SERVER_NAME` | `127.0.0.1` | Gradio 서버 주소입니다. |
| `GRADIO_SERVER_PORT` | `7860` | Gradio 서버 포트입니다. |

LLM API를 사용하려면 `.env`에서 제공자와 해당 API 키를 설정합니다.

```dotenv
LLM_PROVIDER=openai
OPENAI_API_KEY=여기에_API_키_설정
```

Gemini를 사용하려면 `LLM_PROVIDER=gemini`와 `GEMINI_API_KEY`를 설정합니다. `.env`에는 비밀 키를 입력할 수 있으므로 Git에 포함되지 않도록 `.gitignore`에 제외되어 있습니다. `.env` 파일이나 API 키를 GitHub에 올리지 마세요. `.env.example`에는 비밀값을 넣지 말고 설정 예시만 유지하세요.

실제 YOLO 모델을 사용하려면 `weights/best.pt`를 준비하고 `USE_MOCK_YOLO=false`로 설정하세요. 가중치 파일은 저장소에 포함되어 있지 않습니다.

## 테스트

```bash
python -m unittest discover -s tests -v
```

## 프로젝트 구조

```text
.
├── app.py                     # Gradio 웹 UI 및 입력/출력 처리
├── config.py                  # 경로와 환경 변수 설정
├── core/                      # 공통 데이터 스키마 및 예외
├── data/
│   └── processed/
│       └── ingredients.json  # 성분 설명 데이터
├── models/                    # YOLO 및 OCR 구현
├── services/                  # 분석 파이프라인과 처리 서비스
├── tests/                     # 단위 및 파이프라인 테스트
├── requirements.txt
└── BEGINNER_GUIDE.md          # 초보자용 코드 읽기 안내
```

## 참고

이 프로젝트의 성분 정보와 분석 결과는 참고용이며, 의학적 진단이나 치료 조언을 대신하지 않습니다. OCR 및 성분명 매칭 결과는 이미지 품질과 데이터베이스 포함 여부에 따라 달라질 수 있으므로 원본 성분표와 함께 확인하세요.
