# scripts/ai_engine.py
import ollama
from ollama import chat
import json
import sys
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

MAX_CHARS = 3000  # Safe chunk size
EMBED_DIM = 384   # Embedding dimension for all-MiniLM-L6-v2
TOP_K = 3         # Number of relevant chunks to retrieve
FAISS_FILE = "faiss_index.bin"
MAPPING_FILE = "chunk_mapping.pkl"

# Initialize embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

faiss_index = None
chunk_mapping = []

# -------------------- Helpers --------------------
def chunk_text(text, max_chars=MAX_CHARS):
    """Split text into manageable chunks."""
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

def save_faiss():
    """Persist FAISS index and mapping to disk."""
    global faiss_index, chunk_mapping
    if faiss_index and chunk_mapping:
        faiss.write_index(faiss_index, FAISS_FILE)
        with open(MAPPING_FILE, 'wb') as f:
            pickle.dump(chunk_mapping, f)

def load_faiss():
    """Load FAISS index and mapping from disk."""
    global faiss_index, chunk_mapping
    if os.path.exists(FAISS_FILE) and os.path.exists(MAPPING_FILE):
        faiss_index = faiss.read_index(FAISS_FILE)
        with open(MAPPING_FILE, 'rb') as f:
            chunk_mapping = pickle.load(f)
        print(f"[Info] Loaded FAISS index with {len(chunk_mapping)} chunks.")

# -------------------- Ollama Interaction --------------------
def ask_ollama_stream(question, conversation_history, model="llama3", stream_placeholder=None):
    """Send prompt to Ollama and optionally stream to Streamlit."""
    conversation_history.append({"role": "user", "content": question})
    full_response = ""
    try:
        stream = ollama.chat(model=model, messages=conversation_history, stream=True)
        for chunk in stream:
            if "message" in chunk and "content" in chunk["message"]:
                text_part = chunk["message"]["content"]
                full_response += text_part
                if stream_placeholder:
                    stream_placeholder.text(full_response)
                else:
                    sys.stdout.write(text_part)
                    sys.stdout.flush()
        if not stream_placeholder:
            print()
        conversation_history.append({"role": "assistant", "content": full_response})
        return full_response
    except Exception as e:
        err_msg = f"[Error communicating with Ollama: {e}]"
        if stream_placeholder:
            stream_placeholder.text(err_msg)
        return err_msg

# -------------------- FAISS Memory --------------------
def create_or_update_faiss(aggregated_content):
    """
    Convert all files to embeddings and store in FAISS.
    Updates existing FAISS index if it exists.
    """
    global faiss_index, chunk_mapping
    new_chunks = []
    new_mapping = []

    for item in aggregated_content:
        content = item['content']
        content_str = json.dumps(content) if isinstance(content, (dict, list)) else str(content)
        for chunk in chunk_text(content_str):
            new_chunks.append(chunk)
            new_mapping.append({
                'filename': item['filename'],
                'type': item['type'],
                'content': chunk
            })

    if new_chunks:
        embeddings = embed_model.encode(new_chunks, convert_to_numpy=True, normalize_embeddings=True)
        if faiss_index:
            faiss_index.add(embeddings)
            chunk_mapping.extend(new_mapping)
        else:
            faiss_index = faiss.IndexFlatIP(EMBED_DIM)
            faiss_index.add(embeddings)
            chunk_mapping = new_mapping

    save_faiss()

# -------------------- AI Q&A --------------------
def ask_ai(question, conversation_history, top_k=TOP_K, stream_placeholder=None):
    """
    Handles Q&A using FAISS for retrieval.
    Only top-k relevant chunks are used.
    """
    global faiss_index, chunk_mapping
    if not faiss_index:
        raise ValueError("FAISS index not initialized. Call create_or_update_faiss first.")

    q_emb = embed_model.encode([question], convert_to_numpy=True, normalize_embeddings=True)
    D, I = faiss_index.search(q_emb, top_k)
    relevant_chunks = [chunk_mapping[i] for i in I[0]]

    responses = []
    for chunk in relevant_chunks:
        prompt = (
            f"You are a helpful assistant. Use the content below to answer the question.\n\n"
            f"File: {chunk['filename']} ({chunk['type']}):\n{chunk['content']}\n\n"
            f"Question: {question}\nAnswer:"
        )
        response = chat(model="llama3", messages=conversation_history + [{"role": "user", "content": prompt}])
        content = response.get('content', '')
        responses.append(content)
        if stream_placeholder:
            stream_placeholder.text("\n\n".join(responses))

    final_answer = "\n\n".join(responses)
    conversation_history.append({"role": "assistant", "content": final_answer})
    return final_answer

# -------------------- Summarization --------------------
def summarize_files(aggregated_content, conversation_history, stream_placeholder=None):
    """Summarize all files at once using Ollama."""
    combined_text = "\n\n".join(
        f"{item['filename']}:\n{json.dumps(item['content'], indent=4) if isinstance(item['content'], (dict, list)) else str(item['content'])}"
        for item in aggregated_content
    )
    prompt = f"Summarize the following project files:\n\n{combined_text}"
    return ask_ollama_stream(prompt, conversation_history, stream_placeholder=stream_placeholder)

# -------------------- Load persistent memory on startup --------------------
load_faiss()
