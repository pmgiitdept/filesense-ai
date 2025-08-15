# scripts/file_handlers.py
import os
import json
import pandas as pd
import pdfplumber
from reportlab.pdfgen import canvas

from scripts.config import DATA_DIR

# ---------- Text File ----------
def handle_text_file():
    file_path = os.path.join(DATA_DIR, "sample.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("Hello from FileSense AI!\nThis is a test text file.")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    print("\n[TXT] File Content:\n", content)

# ---------- JSON File ----------
def handle_json_file():
    file_path = os.path.join(DATA_DIR, "sample.json")
    data = {"project": "FileSense AI", "version": 1.0, "status": "active"}
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    with open(file_path, "r", encoding="utf-8") as f:
        loaded_data = json.load(f)
    print("\n[JSON] Loaded Data:\n", loaded_data)

# ---------- CSV File ----------
def handle_csv_file():
    file_path = os.path.join(DATA_DIR, "sample.csv")
    df = pd.DataFrame({"Name": ["Alice", "Bob"], "Score": [90, 85]})
    df.to_csv(file_path, index=False)
    loaded_df = pd.read_csv(file_path)
    print("\n[CSV] Loaded DataFrame:\n", loaded_df)

# ---------- PDF File ----------
def handle_pdf_file():
    file_path = os.path.join(DATA_DIR, "sample.pdf")
    if not os.path.exists(file_path):
        c = canvas.Canvas(file_path)
        c.drawString(100, 750, "Hello from FileSense AI PDF!")
        c.drawString(100, 730, "This is a test PDF file.")
        c.save()

    # Read PDF text
    text_content = []
    import pdfplumber
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                text_content.append(f"--- Page {page_num} ---\n{text}")
    print("\n[PDF] Extracted Text:\n", "\n".join(text_content))

# ---------- List Files ----------
def list_files():
    print("\n📂 Files in data folder:")
    files = os.listdir(DATA_DIR)
    for f in files:
        print(" -", f)
    return files
