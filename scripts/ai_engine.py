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

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

# -------------------- Config --------------------
MAX_CHARS = 1500  # Max chunk size for semantic chunking
EMBED_DIM = 384
TOP_K = 5
FAISS_FILE = "faiss_index.bin"
MAPPING_FILE = "chunk_mapping.pkl"

# -------------------- Globals --------------------
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
faiss_index = None
chunk_mapping = []
chunk_texts = []
chunk_ids = []

# PostgreSQL connection (update credentials)
conn = psycopg2.connect(
    host="localhost",
    dbname="filesense_ai",
    user="postgres",
    password="25547"
)
cur = conn.cursor()

current_session_id = None  # Track active session

def start_new_session(session_id):
    """
    Resets FAISS and mappings for a new session.
    """
    global current_session_id, faiss_index, chunk_mapping, chunk_texts, chunk_ids
    current_session_id = session_id
    faiss_index = faiss.IndexFlatIP(EMBED_DIM)
    chunk_mapping = []
    chunk_texts = []
    chunk_ids = []
    print(f"[Info] Started new session: {session_id}")

# -------------------- Helpers --------------------
def semantic_chunk_text(text, max_chars=MAX_CHARS):
    """
    Split text semantically by paragraphs and sentences.
    Each chunk <= max_chars.
    """
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) + 1 <= max_chars:
            current_chunk += (" " + para if current_chunk else para)
        else:
            if current_chunk:
                chunks.append(current_chunk)
            # Split large paragraphs by sentence
            if len(para) > max_chars:
                sentences = re.split(r'(?<=[.!?]) +', para)
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
    """
    Convert file content into text chunks for FAISS.
    Handles text, JSON, CSV (DataFrame/dict), PDF.
    """

    # --- Normalize content into a string ---
    if isinstance(content, dict):  # JSON
        content = json.dumps(content, indent=2)
    elif isinstance(content, list):  # list of dicts (CSV rows)
        content = json.dumps(content, indent=2)
    elif isinstance(content, pd.DataFrame):  # CSV DataFrame
        content = content.to_csv(index=False)
    elif not isinstance(content, str):  # any other type
        content = str(content)

    # --- Sentence tokenization ---
    sentences = sent_tokenize(content)

    # --- Chunk sentences ---
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# -------------------- FAISS + PostgreSQL --------------------
def create_or_update_faiss_db(aggregated_content, session_id):
    """
    Store semantic chunks in PostgreSQL and FAISS index for a given session.
    This clears previous in-memory FAISS index for a fresh session.
    """
    global faiss_index, chunk_mapping, chunk_texts, current_session_id

    if session_id != current_session_id:
        start_new_session(session_id)

    new_chunks, new_mapping = [], []

    for item in aggregated_content:
        chunks = chunk_file_content(item['filename'], item['content'])
        for chunk in chunks:
            emb = embed_model.encode(
                [chunk],
                convert_to_numpy=True,
                normalize_embeddings=True
            )[0].tolist()

            # Store in PostgreSQL with session_id and created_at
            cur.execute("""
                INSERT INTO file_chunks (session_id, filename, file_type, chunk_text, embedding, created_at)
                VALUES (%s, %s, %s, %s, %s, NOW())
            """, (session_id, item['filename'], item['type'], chunk, emb))
            conn.commit()

            # Prepare for FAISS
            new_chunks.append(chunk)
            new_mapping.append({
                "filename": item['filename'],
                "type": item['type'],
                "content": chunk
            })

    if new_chunks:
        embeddings = embed_model.encode(new_chunks, convert_to_numpy=True, normalize_embeddings=True)
        faiss_index.add(embeddings)
        chunk_mapping.extend(new_mapping)
        chunk_texts.extend(new_chunks)

    print(f"[Info] Session {session_id} FAISS + DB updated. Total chunks: {len(chunk_mapping)}")

# -------------------- Save/Load FAISS --------------------
def save_faiss():
    global faiss_index, chunk_mapping
    if faiss_index and chunk_mapping:
        faiss.write_index(faiss_index, FAISS_FILE)
        with open(MAPPING_FILE, 'wb') as f:
            pickle.dump(chunk_mapping, f)

def load_faiss():
    global faiss_index, chunk_mapping
    if os.path.exists(FAISS_FILE) and os.path.exists(MAPPING_FILE):
        faiss_index = faiss.read_index(FAISS_FILE)
        with open(MAPPING_FILE, 'rb') as f:
            chunk_mapping = pickle.load(f)
        print(f"[Info] Loaded FAISS index with {len(chunk_mapping)} chunks.")

def get_embedding(text: str):
    """
    Generates an embedding for the given text using SentenceTransformers.
    """
    if not text or not text.strip():
        return []
    return embed_model.encode(text).tolist()

# -------------------- Retrieval --------------------
def get_top_chunks(question, top_k=TOP_K, session_id=None):
    """
    Retrieves top_k chunks for the given session.
    Only searches chunks from the current session unless overridden.
    """
    global faiss_index, chunk_mapping, current_session_id

    if not faiss_index or not chunk_mapping:
        return []

    if session_id is None:
        session_id = current_session_id

    # Detect filename boost
    filename_boost = None
    for chunk in chunk_mapping:
        if chunk["filename"].lower() in question.lower():
            filename_boost = chunk["filename"].lower()
            break

    # FAISS search
    q_emb = embed_model.encode([question], convert_to_numpy=True, normalize_embeddings=True).astype('float32')
    D, I = faiss_index.search(q_emb, min(top_k * 3, len(chunk_mapping)))

    candidates = []
    for idx, score in zip(I[0], D[0]):
        chunk_info = chunk_mapping[idx]
        filename = chunk_info["filename"]

        # Ensure chunk belongs to this session in DB
        cur.execute("""
            SELECT filename, file_type, chunk_text, embedding, created_at
            FROM file_chunks
            WHERE session_id = %s AND filename = %s
            LIMIT 1
        """, (session_id, filename))

        row = cur.fetchone()
        if not row:
            continue

        created_at = row[4]

        # Safe recency scoring
        if created_at:
            days_old = (datetime.utcnow() - created_at).days
            recency_score = max(0, 1 - (days_old / 30))
        else:
            recency_score = 0  # fallback for old rows without created_at

        boost_score = 0.3 if filename_boost and filename_boost in filename.lower() else 0
        final_score = (score * 0.6) + (recency_score * 0.3) + boost_score

        candidates.append({
            **chunk_info,
            "similarity": float(score),
            "recency": recency_score,
            "boost": boost_score,
            "final_score": final_score
        })

    candidates.sort(key=lambda x: x["final_score"], reverse=True)
    return candidates[:top_k]

# -------------------- Conversation --------------------
def store_conversation(session_id, role, content):
    emb = embed_model.encode([content], convert_to_numpy=True, normalize_embeddings=True)[0].tolist()
    cur.execute("""
        INSERT INTO conversation_history (session_id, role, content, embedding)
        VALUES (%s, %s, %s, %s)
    """, (session_id, role, content, emb))
    conn.commit()

def get_recent_conversation(session_id, limit=10):
    cur.execute("""
        SELECT content FROM conversation_history
        WHERE session_id=%s
        ORDER BY created_at DESC
        LIMIT %s
    """, (session_id, limit))
    rows = cur.fetchall()
    return "\n".join([r[0] for r in reversed(rows)])

# -------------------- RAG + Memory AI --------------------
def ask_ai_with_memory(question, session_id, stream_placeholder=None, top_k=5, max_chars=5000):
    # Retrieve top chunks
    top_chunks = get_top_chunks(question, top_k=top_k, session_id=session_id)

    print("\n[DEBUG] Retrieved chunks for question:")
    for idx, c in enumerate(top_chunks, 1):
        print(
            f"[{idx}] From {c['filename']} ({c['type']}):\n"
            f"Similarity: {c['similarity']:.4f}, Recency: {c['recency']:.4f}, Boost: {c['boost']:.2f}, Final Score: {c['final_score']:.4f}\n"
            f"Content Preview: {c['content'][:200]}...\n"
        )

    # Build context with numbered references for inline citations
    combined_context = "\n\n".join(
        [f"[{idx}] File: {c['filename']} ({c['type']}):\n{c['content']}"
         for idx, c in enumerate(top_chunks, 1)]
    )

    if max_chars:
        combined_context = combined_context[:max_chars]

    # Get conversation memory
    recent_conv = get_recent_conversation(session_id)

    # Prompt with inline citation instructions
    prompt = f"""You are a precise and helpful assistant.
Use the retrieved context and recent conversation to answer the question.
When possible, cite sources inline using the numbers in brackets [1], [2], etc.
If the answer is not in the context, clearly say so.

Context:
{combined_context}

Conversation:
{recent_conv}

Question: {question}
Answer:"""

    store_conversation(session_id, "user", question)

    # Ask Ollama
    final_answer = ask_ollama_stream(prompt, stream_placeholder=stream_placeholder)

    # Build Sources section with numbered references
    #sources_str = "\n\nSources:\n" + "\n".join(
    #    f"[{idx}] {c['filename']} ({c['type']})"
    #    for idx, c in enumerate(top_chunks, 1)
    #)

    final_answer = final_answer.strip()

    store_conversation(session_id, "assistant", final_answer)

    return final_answer

# -------------------- Load FAISS --------------------
load_faiss()

def build_faiss_from_db():
    global faiss_index, chunk_texts, chunk_ids, chunk_mapping
    cur.execute("SELECT id, filename, file_type, chunk_text, embedding FROM file_chunks")
    rows = cur.fetchall()
    if not rows:
        faiss_index = faiss.IndexFlatIP(EMBED_DIM)
        chunk_texts, chunk_ids, chunk_mapping = [], [], []
        return

    embeddings = np.array([row[4] for row in rows]).astype('float32')
    chunk_ids = [row[0] for row in rows]
    chunk_texts = [row[3] for row in rows]
    chunk_mapping = [{"filename": row[1], "type": row[2], "content": row[3]} for row in rows]

    faiss_index = faiss.IndexFlatIP(EMBED_DIM)
    faiss_index.add(embeddings)
    print(f"[Info] FAISS loaded from DB with {len(chunk_ids)} chunks.")
