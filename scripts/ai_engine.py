# scripts/ai_engine.py
import ollama
import json
import sys
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import psycopg2
from scripts.ollama_wrapper import ask_ollama_stream
import re
from datetime import datetime
import nltk
from nltk.tokenize import sent_tokenize
import pandas as pd
from pathlib import Path
import io
from typing import Optional
import openpyxl
import threading

# ----------------- NLTK Setup ----------------
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# -------------------- Config --------------------
MAX_CHARS = 1500 
EMBED_DIM = 384
TOP_K = 5
FAISS_FILE = "faiss_index.bin"
MAPPING_FILE = "chunk_mapping.pkl"

# -------------------- Globals --------------------
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
user_faiss = {} 
user_chunk_mapping = {} 
user_chunk_texts = {}  
user_faiss_locks = {}

current_session_id = None 

conn = psycopg2.connect(
    host="localhost",
    dbname="filesense_ai",
    user="postgres",
    password="25547"
)
cur = conn.cursor()

SMALLTALK_FILE = os.path.join(os.path.dirname(__file__), "small_talk.json")
if os.path.exists(SMALLTALK_FILE):
    with open(SMALLTALK_FILE, "r", encoding="utf-8") as f:
        SMALLTALK_CONFIG = json.load(f)
else:
    SMALLTALK_CONFIG = {}

SMALLTALK_RESPONSES = {
    "hello": "Hello! 👋 How can I help you today?",
    "thanks": "You're welcome! 😊 Happy to help.",
    "bye": "Goodbye! 👋 Have a great day ahead.",
    "how_are_you": "I’m doing great, thanks for asking! How about you?"
}

def start_new_session(session_id):
    """Resets FAISS and mappings for a new session."""
    global current_session_id
    current_session_id = session_id
    print(f"[Info] Started new session: {session_id}")

# -------------------- Helpers --------------------
def semantic_chunk_text(text, max_chars=MAX_CHARS):
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) + 1 <= max_chars:
            current_chunk += (" " + para if current_chunk else para)
        else:
            if current_chunk:
                chunks.append(current_chunk)
            if len(para) > max_chars:
                sentences = sent_tokenize(para)
                sentence_chunk = ""
                for sent in sentences:
                    if len(sentence_chunk) + len(sent) + 1 <= max_chars:
                        sentence_chunk += (" " + sent if sentence_chunk else sent)
                    else:
                        if sentence_chunk:
                            chunks.append(sentence_chunk)
                        sentence_chunk = sent
                if sentence_chunk:
                    chunks.append(sentence_chunk)
            else:
                chunks.append(para)
            current_chunk = ""
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def chunk_file_content(filename, content, chunk_size=500):
    """Universal chunking function for any content type"""
    if isinstance(content, (dict, list)):
        content = json.dumps(content, indent=2)
    elif isinstance(content, pd.DataFrame):
        content = content.to_csv(index=False)
    elif not isinstance(content, str):
        content = str(content)

    sentences = sent_tokenize(content)
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += " " + sentence
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return chunks

# -------------------- FAISS Helpers --------------------
def create_or_update_faiss_db_threadsafe(aggregated_content, session_id, user_id, reset=False):
    """Thread-safe wrapper for per-user FAISS updates"""
    if user_id not in user_faiss_locks:
        user_faiss_locks[user_id] = threading.Lock()
    with user_faiss_locks[user_id]:
        create_or_update_faiss_db(aggregated_content, session_id, user_id, reset)

def create_or_update_faiss_db(aggregated_content, session_id, user_id, reset=False):
    """Store semantic chunks in DB and update per-user FAISS"""
    global user_faiss, user_chunk_mapping, user_chunk_texts

    if user_id not in user_faiss:
        user_faiss[user_id] = faiss.IndexFlatIP(EMBED_DIM)
        user_chunk_mapping[user_id] = []
        user_chunk_texts[user_id] = []

    faiss_index = user_faiss[user_id]
    chunk_mapping = user_chunk_mapping[user_id]
    chunk_texts = user_chunk_texts[user_id]

    if reset:
        faiss_index.reset()
        chunk_mapping.clear()
        chunk_texts.clear()

    new_chunks, new_mapping = [], []

    for item in aggregated_content:
        chunks = chunk_file_content(item['filename'], item['content'])
        for chunk in chunks:
            emb = embed_model.encode([chunk], convert_to_numpy=True, normalize_embeddings=True)[0].astype('float32')
            cur.execute("""
                INSERT INTO file_chunks (session_id, user_id, filename, file_type, chunk_text, embedding, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, NOW())
            """, (session_id, user_id, item['filename'], item['type'], chunk, emb.tolist()))
            conn.commit()

            new_chunks.append(chunk)
            new_mapping.append({"filename": item['filename'], "type": item['type'], "content": chunk})

    if new_chunks:
        embeddings = embed_model.encode(new_chunks, convert_to_numpy=True, normalize_embeddings=True).astype('float32')
        faiss_index.add(embeddings)
        chunk_mapping.extend(new_mapping)
        chunk_texts.extend(new_chunks)

# -------------------- File Handling --------------------
def handle_uploaded_file(session_id, user_id, filename, content_bytes, file_type=None):
    """Synchronous ingestion of files"""
    if filename.endswith(".txt"):
        text = content_bytes.decode("utf-8")
    elif filename.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(content_bytes))
        text = df.to_string(index=False)
    elif filename.endswith(".xlsx"):
        wb = openpyxl.load_workbook(io.BytesIO(content_bytes))
        sheet_texts = []
        for sheet in wb.worksheets:
            rows = []
            for row in sheet.iter_rows(values_only=True):
                rows.append([str(c) if c is not None else "" for c in row])
            sheet_texts.append("\n".join(["\t".join(r) for r in rows]))
        text = "\n\n".join(sheet_texts)
    else:
        raise ValueError("Unsupported file type")

    file_type = file_type or filename.split(".")[-1]
    chunks = chunk_file_content(filename, text)
    aggregated_content = [{"filename": filename, "type": file_type, "content": chunk} for chunk in chunks]
    create_or_update_faiss_db_threadsafe(aggregated_content, session_id, user_id)

def handle_uploaded_file_async(session_id, user_id, filename, content_bytes, file_type=None):
    """Asynchronous ingestion of files"""
    thread = threading.Thread(
        target=handle_uploaded_file,
        args=(session_id, user_id, filename, content_bytes, file_type),
        daemon=True
    )
    thread.start()
    print(f"[Info] Async upload started for user {user_id}, file {filename}")

# -------------------- FAISS Retrieval --------------------
def get_top_chunks(question, top_k=TOP_K, user_id=None):
    if user_id not in user_faiss:
        return []

    faiss_index = user_faiss[user_id]
    chunk_mapping = user_chunk_mapping[user_id]

    if faiss_index.ntotal == 0 or not chunk_mapping:
        return []

    filename_boost = None
    for chunk in chunk_mapping:
        if chunk["filename"].lower() in question.lower():
            filename_boost = chunk["filename"].lower()
            break

    q_emb = embed_model.encode([question], convert_to_numpy=True, normalize_embeddings=True).astype('float32')
    D, I = faiss_index.search(q_emb, min(top_k * 3, len(chunk_mapping)))

    candidates = []
    for idx, score in zip(I[0], D[0]):
        if idx < 0 or idx >= len(chunk_mapping):
            continue
        chunk_info = chunk_mapping[idx]
        filename = chunk_info["filename"]

        cur.execute("""
            SELECT created_at
            FROM file_chunks
            WHERE user_id=%s AND filename=%s
            LIMIT 1
        """, (user_id, filename))
        row = cur.fetchone()
        recency_score = 0
        if row and row[0]:
            days_old = (datetime.utcnow() - row[0]).days
            recency_score = max(0, 1 - (days_old / 30))
        boost_score = 0.3 if filename_boost and filename_boost in filename.lower() else 0
        final_score = (score * 0.6) + (recency_score * 0.3) + boost_score
        candidates.append({**chunk_info, "final_score": final_score})

    candidates.sort(key=lambda x: x["final_score"], reverse=True)
    return candidates[:top_k]

# -------------------- Conversation Memory --------------------
def ensure_user_exists(cur, user_id, username="unknown"):
    cur.execute("SELECT user_id FROM users WHERE user_id=%s", (user_id,))
    if not cur.fetchone():
        cur.execute(
            "INSERT INTO users (user_id, username, password_hash, created_at) VALUES (%s, %s, %s, NOW())",
            (user_id, username, "")
        )

def store_conversation(session_id, user_id, role, content):
    ensure_user_exists(cur, user_id)
    emb = embed_model.encode([content], convert_to_numpy=True, normalize_embeddings=True)[0].astype('float32')
    cur.execute("""
        INSERT INTO conversation_history (session_id, user_id, role, content, embedding, created_at)
        VALUES (%s, %s, %s, %s, %s, NOW())
    """, (session_id, user_id, role, content, emb.tolist()))
    conn.commit()

def get_recent_conversation(session_id, user_id, limit=5):
    cur.execute("""
        SELECT content
        FROM conversation_history
        WHERE session_id = %s AND user_id = %s
        ORDER BY created_at DESC
        LIMIT %s
    """, (session_id, user_id, limit))
    rows = cur.fetchall()
    return "\n".join([r[0] for r in rows[::-1]]) if rows else ""

# -------------------- RAG + Memory AI --------------------
def ask_ai_with_memory(
    question: str,
    session_id: str,
    user_id: str,
    stream_placeholder=None,
    top_k: int = TOP_K,
    max_chars: int = 5000,
    context: Optional[str] = None
):
    q_lower = question.strip().lower()
    for category, triggers in SMALLTALK_CONFIG.items():
        if any(phrase in q_lower for phrase in triggers):
            reply = SMALLTALK_RESPONSES.get(category, "I'm here! 😊 How can I help?")
            store_conversation(session_id, user_id, "user", question)
            store_conversation(session_id, user_id, "assistant", reply)
            return reply

    if context:
        combined_context = context[:max_chars] if len(context) <= max_chars else context[:context.rfind(" ", 0, max_chars)] + "\n...[truncated]"
    else:
        top_chunks = get_top_chunks(question, top_k=top_k, user_id=user_id)
        if top_chunks:
            truncated_chunks = [f"[{i+1}] File: {c['filename']} ({c['type']}):\n{c['content'][:500]}..." for i, c in enumerate(top_chunks)]
            combined_context = "\n\n".join(truncated_chunks)
            if len(combined_context) > max_chars:
                combined_context = combined_context[:combined_context.rfind(" ", 0, max_chars)] + "\n...[truncated]"
        else:
            combined_context = "No relevant context found from uploaded files."

    recent_conv = get_recent_conversation(session_id, user_id)
    if len(recent_conv) > 2000:
        recent_conv = "\n".join(recent_conv.split("\n")[-50:])

    prompt = f"""
You are a precise and helpful assistant.
Use the retrieved context and recent conversation to answer the question.
When possible, cite sources inline using the numbers in brackets [1], [2], etc.

Context:
{combined_context}

Conversation:
{recent_conv}

Question: {question}
Answer:
"""

    store_conversation(session_id, user_id, "user", question)
    final_answer = ask_ollama_stream(prompt, stream_placeholder=stream_placeholder).strip()
    store_conversation(session_id, user_id, "assistant", final_answer)
    return final_answer

# -------------------- Load FAISS --------------------
def build_faiss_from_db(user_id):
    """Load all uploaded files for a user into FAISS"""
    if not user_id:
        return
    if user_id not in user_faiss:
        user_faiss[user_id] = faiss.IndexFlatIP(EMBED_DIM)
        user_chunk_mapping[user_id] = []
        user_chunk_texts[user_id] = []

    faiss_index = user_faiss[user_id]
    chunk_mapping = user_chunk_mapping[user_id]
    chunk_texts = user_chunk_texts[user_id]

    cur.execute("""
        SELECT filename, file_type, chunk_text, embedding
        FROM file_chunks
        WHERE user_id=%s
        ORDER BY created_at ASC
    """, (user_id,))
    rows = cur.fetchall()
    if not rows:
        return

    embeddings = np.array([row[3] for row in rows]).astype('float32')
    faiss_index.reset()
    chunk_texts.clear()
    chunk_mapping.clear()
    chunk_texts[:] = [row[2] for row in rows]
    chunk_mapping[:] = [{"filename": row[0], "type": row[1], "content": row[2]} for row in rows]
    if len(embeddings) > 0:
        faiss_index.add(embeddings)

# Preload all users
cur.execute("SELECT DISTINCT user_id FROM file_chunks")
for row in cur.fetchall():
    build_faiss_from_db(row[0])