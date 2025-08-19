from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.testclient import TestClient
from pydantic import BaseModel
from typing import Optional
import io, base64, json, os, re
from fastapi.middleware.cors import CORSMiddleware
import matplotlib.pyplot as plt
import pandas as pd

from scripts.ai_engine import (
    ask_ai_with_memory,
    ask_ollama_stream,
    handle_uploaded_file,
    get_top_chunks,
    create_or_update_faiss_db,
    get_recent_conversation,
    store_conversation
)
from app import try_generate_chart

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Request Models --------------------
class Query(BaseModel):
    user_input: str
    session_id: str
    user_id: str

UPLOAD_DIR = "./uploads" 

def extract_json_block(ai_output: str) -> dict:
    """
    Extract JSON block from AI output safely.
    Handles cases where the model wraps JSON with ```json ... ```
    or extra text.
    """
    try:
        match = re.search(r"\{.*\}", ai_output, re.S)
        if match:
            return json.loads(match.group(0))
        return json.loads(ai_output)
    except Exception as e:
        print(f"[ERROR] Could not parse AI JSON: {e}")
        return {}

def chart_from_ai_spec(ai_json_str, user_id=None):
    try:
        spec = extract_json_block(ai_json_str)
        if not spec:
            print("[ERROR] Empty spec")
            return None

        dataset = spec.get("dataset")
        x_col = spec.get("x")
        y_col = spec.get("y")
        aggregate = spec.get("aggregate", "none")

        dataset_path = os.path.join(UPLOAD_DIR, str(user_id), dataset) \
            if user_id else os.path.join(UPLOAD_DIR, dataset)

        if not os.path.exists(dataset_path):
            print(f"[ERROR] Dataset not found: {dataset_path}")
            return None

        df = pd.read_csv(dataset_path)

        col_map = {c.lower().strip(): c for c in df.columns}
        if x_col.lower() not in col_map or y_col.lower() not in col_map:
            print(f"[ERROR] Columns not found in dataset. Available: {df.columns}")
            return None

        x_col_real = col_map[x_col.lower()]
        y_col_real = col_map[y_col.lower()]

        if aggregate == "sum":
            df = df.groupby(x_col_real)[y_col_real].sum().reset_index()

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(df[x_col_real], df[y_col_real])
        ax.set_xlabel(x_col_real)
        ax.set_ylabel(y_col_real)
        ax.set_title(spec.get("title", "Chart"))
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig

    except Exception as e:
        print(f"[ERROR] Chart generation failed: {e}")
        return None

# -------------------- Streaming AI Endpoint --------------------
@app.get("/ask_stream")
def ask_ai_stream(session_id: str, user_id: str, question: str):
    """
    Streaming AI response using top-K context + conversation memory.
    Supports both text streaming and chart generation.
    """
    import io, base64
    from matplotlib import pyplot as plt
    import re

    def clean_table_rows(md_text):
        """
        Remove only table rows without numbers. Keep normal text intact.
        """
        lines = md_text.split("\n")
        cleaned_lines = []
        for line in lines:
            if "|" in line:
                # keep table row only if it has numbers
                if re.search(r"\d", line):
                    cleaned_lines.append(line)
            else:
                # normal text, keep
                cleaned_lines.append(line)
        return "\n".join(cleaned_lines)

    def event_generator():
        chart_keywords = ["chart", "graph", "bar graph", "plot"]
        is_chart_request = any(word in question.lower() for word in chart_keywords)

        # Retrieve top-K chunks for context
        top_chunks = get_top_chunks(question, top_k=5, user_id=user_id)
        context_text = "\n\n".join(
            [f"[{idx+1}] {c['content'][:500]}..." for idx, c in enumerate(top_chunks)]
        ) if top_chunks else "No relevant context found."

        # Get recent conversation
        recent_conv = get_recent_conversation(session_id, user_id)
        if len(recent_conv) > 2000:
            recent_conv = recent_conv[-2000:]

        # Build the AI prompt
        prompt = f"""
You are a precise and helpful assistant.
Use the retrieved context and recent conversation to answer the question.
When possible, cite sources inline using the numbers in brackets [1], [2], etc.

Context:
{context_text}

Conversation:
{recent_conv}

Question: {question}
Answer:
"""
        store_conversation(session_id, user_id, "user", question)

        # --- Chart Request ---
        if is_chart_request:
            yield f"data: {json.dumps({'text': '📊 Generating chart, please wait...'})}\n\n"

            chart_prompt = f"""
You are a precise assistant.
Generate a JSON object for a chart with these fields: dataset, x, y, aggregate (default 'none'), title.
Use the context below to determine the correct values.

Context:
{context_text}

Question: {question}
"""
            ai_json_str = ask_ollama_stream(chart_prompt, stream_placeholder=None)
            chart_spec = extract_json_block(ai_json_str)

            fig = chart_from_ai_spec(json.dumps(chart_spec), user_id=user_id)
            if fig:
                buf = io.BytesIO()
                fig.savefig(buf, format="png")
                buf.seek(0)
                img_base64 = base64.b64encode(buf.read()).decode("utf-8")
                plt.close(fig)
                yield f"data: {json.dumps({'text': '', 'chart': img_base64, 'done': True})}\n\n"
            else:
                yield f"data: {json.dumps({'text': '⚠ Could not generate chart'})}\n\n"
            return

        # --- Normal AI Streaming ---
        full_ai_text = ""
        try:
            for chunk in ask_ollama_stream(prompt, stream_placeholder=None):
                chunk_clean = clean_table_rows(chunk)
                full_ai_text += chunk_clean
                yield f"data: {json.dumps({'text': chunk_clean})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'text': f'⚠ Error: {str(e)}'})}\n\n"

        store_conversation(session_id, user_id, "assistant", full_ai_text)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# -------------------- Background AI Task --------------------
def process_ai_query_background(question: str, session_id: str, user_id: str, context: Optional[str] = None):
    """
    Background AI query processor.
    Saves AI response to conversation memory.
    """
    try:
        answer = ask_ai_with_memory(
            question=question,
            session_id=session_id,
            user_id=user_id,
            stream_placeholder=None,
            top_k=5,
            max_chars=5000,
            context=context
        )
        print(f"[INFO] AI response saved for session {session_id}, user {user_id}")
    except Exception as e:
        print(f"[ERROR] AI processing failed: {e}")

# -------------------- File Upload Endpoint --------------------
@app.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    session_id: str = Form(...),
    user_id: str = Form(...),
):
    """
    Optimized file upload: returns immediately and processes in background.
    """
    try:
        contents = await file.read()
        background_tasks.add_task(process_and_index_file, session_id, user_id, file.filename, contents)
        return {
            "success": True,
            "message": "File uploaded successfully. AI processing will happen in background."
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# -------------------- Non-streaming AI Chat Endpoint --------------------
@app.post("/ask")
async def ask_ai_endpoint(query: Query):
    """
    Returns AI response immediately for smalltalk or simple queries.
    For heavy queries, consider streaming or background.
    """
    try:
        answer = ask_ai_with_memory(
            question=query.user_input,
            session_id=query.session_id,
            user_id=query.user_id,
            stream_placeholder=None,
            top_k=5,
            max_chars=5000
        )

        if not answer:
            answer = "Hello! 😊 How can I help you?"

        return {"success": True, "reply": answer}

    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)

# -------------------- Background File Processor --------------------
async def process_and_index_file(session_id: str, user_id: str, filename: str, contents: bytes):
    """
    Convert uploaded file to text, process chunks, and update FAISS DB.
    """
    try:
        if filename.endswith(".csv"):
            import pandas as pd
            df = pd.read_csv(io.BytesIO(contents))
            text_content = df.to_string(index=False)
        elif filename.endswith(".xlsx"):
            import openpyxl
            wb = openpyxl.load_workbook(io.BytesIO(contents))
            sheet_texts = []
            for sheet in wb.worksheets:
                rows = [[str(c) if c is not None else "" for c in row] for row in sheet.iter_rows(values_only=True)]
                sheet_texts.append("\n".join(["\t".join(r) for r in rows]))
            text_content = "\n\n".join(sheet_texts)
        else:
            text_content = contents.decode("utf-8", errors="ignore")

        aggregated_content = [{
            "filename": filename,
            "type": filename.split('.')[-1],
            "content": text_content
        }]


        handle_uploaded_file(session_id, user_id, filename, contents)

        create_or_update_faiss_db(aggregated_content, session_id, user_id)

        print(f"[INFO] File '{filename}' processed and indexed for user {user_id}.")
    except Exception as e:
        print(f"[ERROR] Failed to process '{filename}': {e}")

# -------------------- Chart Generation Endpoint --------------------
@app.post("/chart")
async def generate_chart(query: Query, background_tasks: BackgroundTasks):
    """
    Generate chart + optional AI insights from file context.
    AI processing is backgrounded to avoid timeout.
    """
    aggregated = [] 
    context_text = "\n\n".join([item["content"] for item in aggregated]) if aggregated else None

    fig = try_generate_chart(
        query.user_input,
        aggregated=aggregated,
        session_id=query.session_id,
        user_id=query.user_id
    )

    if fig is None:
        return {"error": "Could not generate chart"}

    background_tasks.add_task(
        process_ai_query_background,
        question=f"Provide insights on this chart: {query.user_input}",
        session_id=query.session_id,
        user_id=query.user_id,
        context=context_text
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")

    return {"image_base64": img_base64, "message": "AI insight is being generated in background."}