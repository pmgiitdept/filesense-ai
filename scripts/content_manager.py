import os
import json
import pandas as pd
from scripts.utils import read_file, DATA_DIR
from PyPDF2 import PdfReader
import psycopg2

DATA_FOLDER = "data"

# PostgreSQL connection (update credentials if needed)
conn = psycopg2.connect(
    host="localhost",
    dbname="filesense_ai",
    user="postgres",
    password="25547"
)
cur = conn.cursor()

def gather_all_content(user_id=None):
    """
    Retrieve all project files for a specific user.
    Returns a list of dicts: { filename, type, content }.
    """
    aggregated = []

    if user_id is not None:
        # If user_id is provided, fetch files stored in DB for that user
        cur.execute("""
            SELECT filename, file_type, chunk_text
            FROM file_chunks
            WHERE session_id LIKE %s
            GROUP BY filename, file_type, chunk_text
        """, (f"{user_id}_%",))
        rows = cur.fetchall()
        for row in rows:
            aggregated.append({
                "filename": row[0],
                "type": row[1],
                "content": row[2]
            })
    else:
        # Fallback: load all files from DATA_FOLDER
        for filename in os.listdir(DATA_FOLDER):
            file_path = os.path.join(DATA_FOLDER, filename)
            if os.path.isfile(file_path):
                ext = filename.split('.')[-1].lower()
                content = None
                try:
                    if ext == "txt":
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                    elif ext == "json":
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = json.load(f)
                    elif ext == "csv":
                        content = pd.read_csv(file_path).to_dict(orient="records")
                    elif ext == "pdf":
                        reader = PdfReader(file_path)
                        content = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
                    else:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                except Exception as e:
                    content = ""
                aggregated.append({
                    "filename": filename,
                    "type": ext,
                    "content": content
                })

    return aggregated

def print_content_summary(aggregated):
    print("\n📦 Aggregated Content Summary:")
    for item in aggregated:
        print(f"\nFile: {item['filename']} ({item['type']})")
        if isinstance(item["content"], pd.DataFrame):
            print(item["content"].head())
        elif isinstance(item["content"], dict):
            print(json.dumps(item["content"], indent=4))
        elif isinstance(item["content"], str):
            preview = item["content"][:200] + ("..." if len(item["content"]) > 200 else "")
            print(preview)
        else:
            print("[Unsupported content type]")
