import os
import json
import pandas as pd
from scripts.utils import read_file, DATA_DIR

def gather_all_content():
    aggregated = []
    files = os.listdir(DATA_DIR)

    for file in files:
        file_path = os.path.join(DATA_DIR, file)
        ext = os.path.splitext(file)[1].lower()

        try:
            content = read_file(file_path)
            aggregated.append({
                "filename": file,
                "type": ext.replace(".", ""),
                "content": content
            })
        except Exception as e:
            aggregated.append({
                "filename": file,
                "type": ext.replace(".", ""),
                "content": f"[Error reading file: {e}]"
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
