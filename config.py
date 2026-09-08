import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_DIR = DATA_DIR / "raw"
WEIGHTS_DIR = BASE_DIR / "weights"

# Ensure directories exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

# Database
INGREDIENT_DB_PATH = PROCESSED_DATA_DIR / "ingredients.json"
RAW_EXCEL_PATH = RAW_DATA_DIR / "source.xlsx"

# Model Paths & Settings
YOLO_MODEL_PATH = WEIGHTS_DIR / "best.pt"
USE_MOCK_YOLO = os.getenv("USE_MOCK_YOLO", "true").lower() == "true"
YOLO_CONF_THRESHOLD = float(os.getenv("YOLO_CONF_THRESHOLD", "0.25"))

# OCR Settings
OCR_LANG = os.getenv("OCR_LANG", "korean")  # paddleocr language
OCR_USE_ANGLE_CLS = True
OCR_USE_GPU = os.getenv("OCR_USE_GPU", "false").lower() == "true"

# Fuzzy Matching Settings
FUZZY_MATCH_THRESHOLD = int(os.getenv("FUZZY_MATCH_THRESHOLD", "85"))

# LLM Settings
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "fallback")  # "openai", "gemini", "fallback"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

# Server / UI Settings
GRADIO_SERVER_NAME = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
GRADIO_SERVER_PORT = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
