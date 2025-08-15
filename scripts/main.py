import os
import json
import pandas as pd
import numpy as np
import pdfplumber

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
DATA_DIR = os.path.abspath(DATA_DIR) 

def handle_text_file():
    file_path = os.path.join(DATA_DIR, "sample.txt")

    # Write to TXT
    with open(file_path, "w") as f:
        f.write("Hello from FileSense AI!\nThis is a test text file.")

    # Read from TXT
    with open(file_path, "r") as f:
        content = f.read()
    print("\n[TXT] File Content:\n", content)

def handle_json_file():
    file_path = os.path.join(DATA_DIR, "sample.json")
    data = {"project": "FileSense AI", "version": 1.0, "status": "active"}

    # Write JSON
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

    # Read JSON
    with open(file_path, "r") as f:
        loaded_data = json.load(f)
    print("\n[JSON] Loaded Data:\n", loaded_data)

def handle_csv_file():
    file_path = os.path.join(DATA_DIR, "sample.csv")
    df = pd.DataFrame({"Name": ["Alice", "Bob"], "Score": [90, 85]})

    # Write CSV
    df.to_csv(file_path, index=False)

    # Read CSV
    loaded_df = pd.read_csv(file_path)
    print("\n[CSV] Loaded DataFrame:\n", loaded_df)

def list_files():
    print("\n📂 Files in data folder:")
    files = os.listdir(DATA_DIR)
    for f in files:
        print(" -", f)
    return files

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

def main():
    print("🚀 FileSense AI Project Initialized!")
    print("DEBUG: DATA_DIR =", DATA_DIR)

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # Create sample files
    handle_text_file()
    handle_json_file()
    handle_csv_file()

    files = list_files()
    if not files:
        print("⚠ No files found in data folder!")
        return

    print("\n📖 Reading files:")
    for file in files:
        path = os.path.join(DATA_DIR, file)
        content = read_file(path)

        print(f"\nFile: {file}")
        print("Content Preview:")

        if isinstance(content, pd.DataFrame):
            print(content.head())
        elif isinstance(content, dict):
            print(json.dumps(content, indent=4))
        elif isinstance(content, str):
            preview = content[:500] + ("..." if len(content) > 500 else "")
            print(preview)
        else:
            print("[Unsupported content type for preview]")
