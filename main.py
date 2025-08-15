import os
import json
import pandas as pd
from reportlab.pdfgen import canvas
from scripts.content_manager import gather_all_content, print_content_summary
from scripts.utils import DATA_DIR, read_pdf
from scripts.ai_engine import ask_ai, summarize_files
from scripts.file_handlers import handle_text_file, handle_json_file, handle_csv_file, handle_pdf_file, list_files
from scripts.config import HISTORY_FILE

# ---------- Initialize conversation history ----------
conversation_history = []

def handle_text_file():
    file_path = os.path.join(DATA_DIR, "sample.txt")
    with open(file_path, "w") as f:
        f.write("Hello from FileSense AI!\nThis is a test text file.")
    with open(file_path, "r") as f:
        print("\n[TXT] File Content:\n", f.read())

def handle_json_file():
    file_path = os.path.join(DATA_DIR, "sample.json")
    data = {"project": "FileSense AI", "version": 1.0, "status": "active"}
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
    with open(file_path, "r") as f:
        print("\n[JSON] Loaded Data:\n", json.load(f))

def handle_csv_file():
    file_path = os.path.join(DATA_DIR, "sample.csv")
    df = pd.DataFrame({"Name": ["Alice", "Bob"], "Score": [90, 85]})
    df.to_csv(file_path, index=False)
    print("\n[CSV] Loaded DataFrame:\n", pd.read_csv(file_path))

def handle_pdf_file():
    file_path = os.path.join(DATA_DIR, "sample.pdf")
    if not os.path.exists(file_path):
        c = canvas.Canvas(file_path)
        c.drawString(100, 750, "Hello from FileSense AI PDF!")
        c.drawString(100, 730, "This is a test PDF file.")
        c.save()
    print("\n[PDF] Extracted Text:\n", read_pdf(file_path))

def list_files():
    print("\n📂 Files in data folder:")
    for f in os.listdir(DATA_DIR):
        print(" -", f)

def main():
    print("🚀 FileSense AI Project Initialized!")
    print("DEBUG: DATA_DIR =", DATA_DIR)

    os.makedirs(DATA_DIR, exist_ok=True)

    # Load previous conversation history if available
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                conversation_history.extend(json.load(f))
            print(f"\n📖 Loaded previous conversation history ({len(conversation_history)} entries).")
        except Exception as e:
            print(f"[Error loading conversation history: {e}]")

    # ---------- Prepare sample files ----------
    handle_text_file()
    handle_json_file()
    handle_csv_file()
    handle_pdf_file()

    # ---------- List files ----------
    list_files()

    # ---------- Aggregate file content ----------
    aggregated = gather_all_content()
    print_content_summary(aggregated)

    # ---------- AI Summary of all files ----------
    print("\n🤖 AI Summary of All Files:")
    ai_summary = summarize_files(
        aggregated,
        conversation_history=conversation_history,
        show_progress=True
    )
    conversation_history.append({"role": "assistant", "content": ai_summary})
    print("\n", ai_summary)

    # ---------- Interactive Q&A Mode ----------
    print("\n💬 Enter Q&A mode. Type 'exit' to quit.")
    while True:
        user_q = input("\nAsk about your files or general questions: ")
        if user_q.lower() in ["exit", "quit"]:
            break

        ai_answer = ask_ai(
            user_q,
            aggregated,
            conversation_history=conversation_history,
            show_progress=True
        )

        # Store the conversation
        conversation_history.append({"role": "user", "content": user_q})
        conversation_history.append({"role": "assistant", "content": ai_answer})

        # Save to JSON
        try:
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(conversation_history, f, indent=4)
        except Exception as e:
            print(f"[Error saving conversation history: {e}]")

        # Print AI answer
        print("\n🤖", ai_answer)


if __name__ == "__main__":
    main()