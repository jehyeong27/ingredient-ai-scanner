import json
import os
import sys
from pathlib import Path
import pandas as pd

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from config import RAW_EXCEL_PATH, INGREDIENT_DB_PATH

def build_database_from_excel(excel_path: Path, output_json_path: Path):
    """
    Excel 파일(source.xlsx)을 읽어서 빠른 조회가 가능한 JSON 파일(ingredients.json)로 변환합니다.
    """
    if not excel_path.exists():
        print(f"[Error] Excel file not found: {excel_path}")
        print("Please place your ingredient Excel file at data/raw/source.xlsx")
        return False

    print(f"Reading Excel from: {excel_path}")
    try:
        df = pd.read_excel(excel_path)
    except Exception as e:
        print(f"[Error] Failed to read Excel file: {e}")
        return False

    # 컬럼 매핑 자동 추론 (성분명, 기능, 효능, 주의사항 등)
    col_mapping = {}
    for col in df.columns:
        c_clean = str(col).strip().lower()
        if "성분" in c_clean or "name" in c_clean or "ingredient" in c_clean:
            col_mapping["name"] = col
        elif "기능" in c_clean or "function" in c_clean or "역할" in c_clean:
            col_mapping["function"] = col
        elif "효능" in c_clean or "efficacy" in c_clean or "특징" in c_clean or "설명" in c_clean:
            col_mapping["efficacy"] = col
        elif "주의" in c_clean or "caution" in c_clean or "부작용" in c_clean:
            col_mapping["caution"] = col

    name_col = col_mapping.get("name", df.columns[0])
    func_col = col_mapping.get("function")
    eff_col = col_mapping.get("efficacy")
    caut_col = col_mapping.get("caution")

    print(f"Detected columns -> Name: {name_col}, Function: {func_col}, Efficacy: {eff_col}, Caution: {caut_col}")

    db = {}
    for _, row in df.iterrows():
        name = str(row[name_col]).strip() if pd.notna(row[name_col]) else ""
        if not name or name.lower() == "nan":
            continue

        func = str(row[func_col]).strip() if func_col and pd.notna(row[func_col]) else ""
        eff = str(row[eff_col]).strip() if eff_col and pd.notna(row[eff_col]) else ""
        caut = str(row[caut_col]).strip() if caut_col and pd.notna(row[caut_col]) else ""

        db[name] = {
            "function": func,
            "efficacy": eff,
            "caution": caut
        }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

    print(f"Successfully converted {len(db)} ingredients to: {output_json_path}")
    return True

if __name__ == "__main__":
    build_database_from_excel(RAW_EXCEL_PATH, INGREDIENT_DB_PATH)
