# CosmeticAI 코드 읽기 안내서

이 프로젝트는 화장품 성분표 이미지를 분석하는 작은 웹 애플리케이션입니다.
처음부터 모든 파일을 이해하려고 하지 말고, 아래 순서로 읽어 보세요.

## 1. 가장 먼저 볼 전체 흐름

`app.py`의 `process_cosmetic_image` 함수가 사용자가 버튼을 눌렀을 때 실행됩니다.
이 함수는 입력 이미지를 준비한 뒤 파이프라인을 호출합니다.

```python
result = pipeline.analyze(img_bgr)
```

파이프라인의 흐름은 다음과 같습니다.

1. 이미지 크기와 대비를 조정합니다.
2. YOLO 또는 Mock YOLO가 성분표 영역을 찾습니다.
3. 찾은 영역을 잘라냅니다.    
4. OCR이 이미지에서 글자를 읽습니다.
5. 파서가 긴 글자를 성분 목록으로 나눕니다.
6. FuzzyMatcher가 OCR 오타를 DB 이름과 비교합니다.
7. IngredientService가 성분 설명을 JSON에서 읽습니다.
8. LLMService가 분석 보고서를 만듭니다.
9. 결과를 Gradio 화면에 표시합니다.

## 2. `pipeline = AnalysisPipeline()`의 뜻

```python
pipeline = AnalysisPipeline()
```

이 한 줄은 `AnalysisPipeline`이라는 클래스로 객체를 하나 만드는 코드입니다.

- `AnalysisPipeline`: 분석 작업 순서를 적어 둔 설계도(클래스)
- `AnalysisPipeline()`: 설계도로 실제 객체를 만드는 작업
- `pipeline`: 만들어진 객체를 담은 변수

`AnalysisPipeline()`을 만드는 순간 내부에서 전처리기, OCR, 파서, DB 서비스,
LLM 서비스를 준비합니다. 아직 이미지를 분석하는 것은 아닙니다.
실제 분석은 나중에 `pipeline.analyze(image)`를 호출할 때 시작됩니다.

## 3. 파일을 읽는 추천 순서

1. `core/schemas.py`: 여러 단계가 주고받는 데이터 모양
2. `services/pipeline.py`: 전체 순서
3. `services/preprocessor.py`: 이미지 준비와 자르기
4. `models/mock_yolo.py`: 테스트용 영역 탐지
5. `models/ocr_engine.py`: 글자 읽기
6. `services/ingredient_parser.py`: 글자를 성분 목록으로 나누기
7. `services/fuzzy_matcher.py`: 오타 보정
8. `services/ingredient_service.py`: JSON DB 조회
9. `services/llm_service.py`: 보고서와 질문 답변
10. `app.py`: 위 기능을 Gradio 화면에 연결

## 4. 자주 나오는 Python 문법

### 점(`.`)으로 기능 사용하기

```python
pipeline.analyze(image)
```

`pipeline` 객체 안에 있는 `analyze` 함수를 호출한다는 뜻입니다.

### `or`로 기본 객체 선택하기

```python
self.parser = parser or IngredientParser()
```

`parser`가 전달되었으면 그것을 사용하고, 전달되지 않았으면 새로 만듭니다.
테스트에서 가짜 객체를 넣을 수 있도록 만든 코드입니다.

### 리스트 컴프리헨션

```python
names = [match.raw_ocr_name for match in matches if not match.is_matched]
```

`matches`를 하나씩 살펴보면서, 매칭되지 않은 이름만 새 리스트에 넣는 코드입니다.
처음에는 아래처럼 풀어 쓴 코드와 비교해서 읽으면 됩니다.

```python
names = []
for match in matches:
    if not match.is_matched:
        names.append(match.raw_ocr_name)
```

### 타입 힌트

```python
def parse(self, raw_text: str) -> List[str]:
```

`raw_text`는 문자열이고, 함수 결과는 문자열 목록이라는 설명입니다.
실행에 반드시 필요한 문법은 아니지만 코드 이해와 자동 완성에 도움을 줍니다.

## 5. 실행과 테스트

프로젝트 폴더에서 다음 명령을 실행합니다.

```text
python -m unittest discover -s tests -v
```

웹 화면은 다음 명령으로 실행합니다.

```text
python app.py
```

기본 주소는 `http://127.0.0.1:7860`입니다.
처음에는 실제 YOLO와 LLM 없이도 실행되는 Mock/Fallback 경로부터 확인하면 됩니다.

## 6. 공부할 때의 작은 단위

한 번에 파일 전체를 읽지 말고, 다음 질문 하나만 확인하세요.

- 이 함수의 입력은 무엇인가?
- 이 함수가 반환하는 값은 무엇인가?
- 다음으로 그 값을 받는 함수는 어디인가?
- 문제가 생기면 어느 단계의 값부터 출력해 볼 것인가?

예를 들어 파싱을 공부할 때는 `clean_text` 결과를 먼저 보고,
그 다음 `split_ingredients` 결과를 확인하면 됩니다.
