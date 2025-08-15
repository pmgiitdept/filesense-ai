import os
import json
import pandas as pd
import pdfplumber

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))

def read_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    elif ext == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    elif ext == ".csv":
        return pd.read_csv(file_path)
    elif ext == ".pdf":
        return read_pdf(file_path)
    else:
        return f"[Unsupported file type: {ext}]"

def read_pdf(file_path):
    text_content = []
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                text_content.append(f"--- Page {page_num} ---\n{text}")
    return "\n".join(text_content) if text_content else "[No text extracted]"
