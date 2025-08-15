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

# -------------------- Config --------------------
MAX_CHARS = 3000  # Safe chunk size for embedding
EMBED_DIM = 384   # all-MiniLM-L6-v2 embedding dimension
TOP_K = 5         # Number of chunks to retrieve
FAISS_FILE = "faiss_index.bin"
MAPPING_FILE = "chunk_mapping.pkl"

# -------------------- Globals --------------------
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
faiss_index = None
chunk_mapping = []

# -------------------- Helpers --------------------
def chunk_text(text, max_chars=MAX_CHARS):
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

def chunk_file_content(filename, content):
    """Chunk content intelligently based on file type."""
    chunks = []
    if filename.endswith('.json') and isinstance(content, (dict, list)):
        if isinstance(content, dict):
            for k, v in content.items():
                chunks.append(json.dumps({k: v}))
        else:  # list
            for elem in content:
                chunks.append(json.dumps(elem))
    else:
        text = json.dumps(content) if isinstance(content, (dict, list)) else str(content)
        chunks = chunk_text(text)
    return chunks

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

# -------------------- Ollama --------------------
def ask_ollama_stream(question, conversation_history, model="llama3", stream_placeholder=None):
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

# -------------------- FAISS --------------------
def create_or_update_faiss(aggregated_content):
    """Create or update FAISS index with new content."""
    global faiss_index, chunk_mapping
    new_chunks, new_mapping = [], []

    for item in aggregated_content:
        chunks = chunk_file_content(item['filename'], item['content'])
        for chunk in chunks:
            new_chunks.append(chunk)
            new_mapping.append({
                "filename": item['filename'],
                "type": item['type'],
                "content": chunk
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
    print(f"[Info] FAISS index updated. Total chunks: {len(chunk_mapping)}")

# -------------------- AI Q&A --------------------
def ask_ai(question, conversation_history, top_k=TOP_K, stream_placeholder=None, max_chars=1000):
    """Retrieve top_k chunks from FAISS and ask Ollama using full content."""
    global faiss_index, chunk_mapping
    if not faiss_index:
        raise ValueError("FAISS index not initialized. Call create_or_update_faiss first.")

    # Embed the question
    q_emb = embed_model.encode([question], convert_to_numpy=True, normalize_embeddings=True)

    # Retrieve top-k relevant chunks
    D, I = faiss_index.search(q_emb, top_k)
    relevant_chunks = [chunk_mapping[i] for i in I[0]]

    # Combine all chunks into a single prompt
    combined_content = ""
    for chunk in relevant_chunks:
        chunk_text = chunk['content']  # no per-chunk truncation
        combined_content += f"File: {chunk['filename']} ({chunk['type']}):\n{chunk_text}\n\n"

    # Build the final prompt
    prompt = (
        f"You are a helpful assistant. Use the following content to answer the question.\n\n"
        f"{combined_content}"
        f"Question: {question}\nAnswer:"
    )

    # Ask Ollama
    conversation_history.append({"role": "user", "content": prompt})
    final_answer = ""
    try:
        stream = ollama.chat(model="llama3", messages=conversation_history, stream=True)
        for chunk in stream:
            if "message" in chunk and "content" in chunk["message"]:
                text_part = chunk["message"]["content"]
                final_answer += text_part
                if stream_placeholder:
                    stream_placeholder.text(final_answer)
                else:
                    sys.stdout.write(text_part)
                    sys.stdout.flush()
        if not stream_placeholder:
            print()
    except Exception as e:
        final_answer = f"[Error communicating with Ollama: {e}]"
        if stream_placeholder:
            stream_placeholder.text(final_answer)

    conversation_history.append({"role": "assistant", "content": final_answer})
    return final_answer

# -------------------- Summarization --------------------
def summarize_files(aggregated_content, conversation_history, stream_placeholder=None):
    combined_text = "\n\n".join(
        f"{item['filename']}:\n{json.dumps(item['content'], indent=4) if isinstance(item['content'], (dict, list)) else str(item['content'])}"
        for item in aggregated_content
    )
    prompt = f"Summarize the following project files:\n\n{combined_text}"
    return ask_ollama_stream(prompt, conversation_history, stream_placeholder=stream_placeholder)

# -------------------- Load persistent FAISS --------------------
load_faiss()