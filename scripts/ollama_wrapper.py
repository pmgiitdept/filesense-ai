# scripts/ollama_wrapper.py
import ollama
import sys

def ask_ollama_stream(prompt, conversation_history=None, stream_placeholder=None, model="llama3"):
    """
    Ask Ollama AI using a streaming approach.

    Args:
        prompt (str): The user question or prompt.
        conversation_history (list): List of previous messages [{"role": "user/assistant", "content": "..."}].
        stream_placeholder: Streamlit placeholder for live updates (optional).
        model (str): Ollama model name.
    
    Returns:
        str: Full response from Ollama.
    """
    if conversation_history is None:
        conversation_history = []

    # Add user prompt to conversation
    conversation_history.append({"role": "user", "content": prompt})
    full_response = ""

    try:
        # Start streaming from Ollama
        stream = ollama.chat(model=model, messages=conversation_history, stream=True)
        for chunk in stream:
            if "message" in chunk and "content" in chunk["message"]:
                text_part = chunk["message"]["content"]
                full_response += text_part

                # Update Streamlit placeholder if provided
                if stream_placeholder:
                    stream_placeholder.text(full_response)
                else:
                    sys.stdout.write(text_part)
                    sys.stdout.flush()

        if not stream_placeholder:
            print()

        # Append assistant response to conversation history
        conversation_history.append({"role": "assistant", "content": full_response})

    except Exception as e:
        full_response = f"[Error communicating with Ollama: {e}]"
        if stream_placeholder:
            stream_placeholder.text(full_response)

    return full_response
