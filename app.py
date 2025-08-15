import streamlit as st
from scripts.content_manager import gather_all_content
from scripts.file_handlers import handle_text_file, handle_json_file, handle_csv_file, handle_pdf_file
from scripts.ai_engine import ask_ai, create_or_update_faiss

# ---------- Initialize session ----------
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# ---------- Page setup ----------
st.set_page_config(page_title="FileSense AI", layout="centered")
st.title("📦 FileSense AI")
st.write("Ask questions about your project files!")

# ---------- Load files ----------
handle_text_file()
handle_json_file()
handle_csv_file()
handle_pdf_file()
aggregated = gather_all_content()

# Build or update FAISS index
create_or_update_faiss(aggregated)

# ---------- Slider for max response length ----------
max_chars = st.slider("Max characters per AI response", 200, 2000, 1000, 100)

# ---------- Chat display ----------
def render_chat():
    chat_html = "<div id='chat-box' style='height:400px; overflow-y:auto; padding:5px;'>"
    for msg in st.session_state.conversation_history:
        bg = "#DCF8C6" if msg["role"] == "user" else "#EAEAEA"
        chat_html += f"<div style='background-color:{bg}; padding:10px; border-radius:10px; margin:5px 0; width: fit-content;'>{msg['content']}</div>"
    chat_html += "</div>"
    st.markdown(chat_html, unsafe_allow_html=True)

render_chat()

# ---------- Input form ----------
st.markdown("<div style='position: fixed; bottom: 10px; width: 100%; max-width: 600px;'>", unsafe_allow_html=True)
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Type your message here:", value="", key="chat_input_field")
    submit_button = st.form_submit_button("Send")

    if submit_button and user_input.strip():
        # Store user message
        st.session_state.conversation_history.append({"role": "user", "content": user_input})

        # AI placeholder for streaming
        ai_placeholder = st.empty()
        ai_placeholder.text("🤖 Thinking...")

        # Ask AI using improved engine
        answer = ask_ai(
            user_input,
            st.session_state.conversation_history,
            top_k=5,  # Increased to cover more chunks
            stream_placeholder=ai_placeholder,
            max_chars=max_chars
        )

        # Store AI response
        st.session_state.conversation_history.append({"role": "assistant", "content": answer})
        ai_placeholder.text(answer)

st.markdown("</div>", unsafe_allow_html=True)
render_chat()

# ---------- Auto-scroll ----------
st.write(
    """
    <script>
    var chatContainer = document.getElementById('chat-box');
    if(chatContainer){
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }
    </script>
    """,
    unsafe_allow_html=True
)
