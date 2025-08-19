import streamlit as st
import uuid
import json
import pandas as pd
from PyPDF2 import PdfReader
import threading
import matplotlib.pyplot as plt
import os
import psycopg2
import re
from werkzeug.security import generate_password_hash, check_password_hash

from scripts.content_manager import gather_all_content
from scripts.file_handlers import handle_text_file, handle_json_file, handle_csv_file, handle_pdf_file
from scripts.ai_engine import create_or_update_faiss_db, ask_ai_with_memory, build_faiss_from_db
from scripts.file_watcher import start_watcher

# ---------- Page setup ----------
st.set_page_config(page_title="FileSense AI", layout="wide")
st.title("📦 FileSense AI")
st.caption("Chat with your project files — now with memory + citations.")

DATA_FOLDER = "data"
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

# ---------- Database connection ----------
conn = psycopg2.connect(
    host="localhost",
    dbname="filesense_ai",
    user="postgres",
    password="25547"
)
cur = conn.cursor()

# ---------- User login/registration ----------
def register(username, password):
    hashed_pw = generate_password_hash(password)
    try:
        cur.execute("INSERT INTO users (username, password_hash) VALUES (%s, %s) RETURNING user_id", (username, hashed_pw))
        conn.commit()
        user_id = cur.fetchone()[0]
        return user_id
    except psycopg2.IntegrityError:
        conn.rollback()
        return None

def login(username, password):
    cur.execute("SELECT user_id, password_hash FROM users WHERE username=%s", (username,))
    row = cur.fetchone()
    if row and check_password_hash(row[1], password):
        return row[0]
    return None

# ---------- Streamlit session ----------
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())
if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = []

if "user_id" not in st.session_state:
    st.session_state["user_id"] = None

# ---------- Login Page ----------
if st.session_state.user_id is None:
    st.title("🔐 Login to FileSense AI")
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            user_id = login(username, password)
            if user_id:
                st.session_state.user_id = user_id
                st.success(f"Logged in as {username}")
            else:
                st.error("Invalid credentials")

    with tab2:
        new_user = st.text_input("New Username", key="reg_user")
        new_pass = st.text_input("New Password", type="password", key="reg_pass")
        if st.button("Register"):
            user_id = register(new_user, new_pass)
            if user_id:
                st.session_state.user_id = user_id
                st.success(f"Registered and logged in as {new_user}")
            else:
                st.error("Username already exists")

    st.stop()

# ---------- Main App ----------
session_id = f"{st.session_state.user_id}_{str(uuid.uuid4())}"

def extract_json_from_text(text):
    """Extract the first JSON object found in a string."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None
    return None

def try_generate_chart(user_input, aggregated, session_id, user_id):
    chart_spec = extract_json_from_text(user_input)
    if chart_spec:
        pass  
    else:
        keywords = ["plot", "graph", "chart", "visualize", "draw"]
        if not any(k in user_input.lower() for k in keywords):
            return None 

        chart_spec_str = ask_ai_with_memory(
            question=f"Create a JSON spec for a chart based on this query: {user_input}. "
                     f"Available datasets: {aggregated}. "
                     "Format: { 'dataset': 'filename', 'chart_type': 'bar/line/scatter/pie', "
                     "'x': 'column', 'y': 'column', 'title': 'optional', 'aggregate': 'sum/avg/none' }. "
                     "Output JSON only.",
            session_id=session_id,
            stream_placeholder=None,
            user_id=user_id,
            top_k=5,
            max_chars=500
        )
        chart_spec = extract_json_from_text(chart_spec_str)

    if not chart_spec:
        return None

    dataset_name = chart_spec.get("dataset")
    dataset_item = next((item for item in aggregated if item["filename"] == dataset_name), None)
    if not dataset_item and aggregated:
        dataset_item = aggregated[0]
    if not dataset_item:
        return None

    if isinstance(dataset_item["content"], pd.DataFrame):
        df = dataset_item["content"]
    elif isinstance(dataset_item["content"], list) and isinstance(dataset_item["content"][0], dict):
        df = pd.DataFrame(dataset_item["content"])
    else:
        return None

    agg_func = chart_spec.get("aggregate", "none")
    x_col = chart_spec.get("x")
    y_col = chart_spec.get("y")
    if agg_func in ["sum", "avg"]:
        df = df.groupby(x_col, as_index=False)[y_col].agg("sum" if agg_func=="sum" else "mean")

    fig, ax = plt.subplots()
    chart_type = chart_spec.get("chart_type", "bar").lower()

    if chart_type == "bar":
        ax.bar(df[x_col], df[y_col])
    elif chart_type == "line":
        ax.plot(df[x_col], df[y_col], marker='o')
    elif chart_type == "scatter":
        ax.scatter(df[x_col], df[y_col])
    elif chart_type == "pie":
        if y_col:
            ax.pie(df[y_col], labels=df[x_col], autopct="%1.1f%%")
        else:
            return None  #

    ax.set_title(chart_spec.get("title", "Generated Chart"))
    ax.set_xlabel(x_col if chart_type != "pie" else "")
    ax.set_ylabel(y_col if chart_type not in ["pie"] else "")

    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

# ---------- Database Models for Chat History ----------
def create_conversation(user_id, title="New Chat"):
    session_id = str(uuid.uuid4()) 
    cur.execute(
        "INSERT INTO conversations (user_id, session_id, title) VALUES (%s, %s, %s) RETURNING conversation_id",
        (user_id, session_id, title)
    )
    conv_id = cur.fetchone()[0]
    conn.commit()
    return conv_id, session_id

def save_message(conversation_id, role, content):
    cur.execute(
        "INSERT INTO messages (conversation_id, role, content) VALUES (%s, %s, %s)",
        (conversation_id, role, content)
    )
    conn.commit()

def load_conversations(user_id):
    cur.execute("SELECT conversation_id, title, created_at FROM conversations WHERE user_id=%s ORDER BY created_at DESC", (user_id,))
    return cur.fetchall()

def load_messages(conversation_id):
    cur.execute("SELECT role, content FROM messages WHERE conversation_id=%s ORDER BY created_at ASC", (conversation_id,))
    return cur.fetchall()

def get_active_session():
    """
    Return the active chat session dict from chat_sessions using active_chat_id.
    """
    conv_id = st.session_state["active_chat"]
    for chat in st.session_state["chat_sessions"]:
        if chat["id"] == conv_id:
            return chat
    new_conv_id, new_session_id = create_conversation(st.session_state["user_id"], "New Chat")
    new_chat = {"id": new_conv_id, "session_id": new_session_id, "title": "New Chat", "history": []}
    st.session_state["chat_sessions"].append(new_chat)
    st.session_state["active_chat"] = (new_conv_id, new_session_id) 
    return new_chat

if "chat_sessions" not in st.session_state:
    st.session_state["chat_sessions"] = []

if st.session_state.user_id and not st.session_state["chat_sessions"]:
    conversations = load_conversations(st.session_state.user_id)
    for conv_id, title, created_at in conversations:
        messages = load_messages(conv_id)
        st.session_state["chat_sessions"].append({
            "id": conv_id,
            "session_id": str(uuid.uuid4()),
            "title": title if title else "New Chat",
            "history": [{"role": r, "content": c} for r, c in messages]
        })

if "active_chat" not in st.session_state or isinstance(st.session_state["active_chat"], int):
    if st.session_state["chat_sessions"]:
        last_chat = st.session_state["chat_sessions"][-1]
        st.session_state["active_chat"] = (last_chat["id"], last_chat.get("session_id", str(uuid.uuid4())))
    else:
        new_conv_id, new_session_id = create_conversation(st.session_state["user_id"], "New Chat")
        st.session_state["chat_sessions"].append({
            "id": new_conv_id,
            "session_id": new_session_id,
            "title": "New Chat",
            "history": []
        })
        st.session_state["active_chat"] = (new_conv_id, new_session_id)

def get_active_session():
    conv_id, sess_id = st.session_state["active_chat"]
    for chat in st.session_state["chat_sessions"]:
        if chat["id"] == conv_id:
            return chat
    new_conv_id, new_session_id = create_conversation(st.session_state["user_id"], "New Chat")
    new_chat = {"id": new_conv_id, "session_id": new_session_id, "title": "New Chat", "history": []}
    st.session_state["chat_sessions"].append(new_chat)
    st.session_state["active_chat"] = (new_conv_id, new_session_id)
    return new_chat

active_session = get_active_session()
conv_id, sess_id = active_session["id"], active_session["session_id"]
active_session["history"] = [{"role": r, "content": c} for r, c in load_messages(conv_id)]

# ---------- Sidebar with chat history ----------
with st.sidebar:
    st.subheader("💾 Chat History")

    chat_titles = [chat["title"] for chat in st.session_state["chat_sessions"]]
    chat_ids = [chat["id"] for chat in st.session_state["chat_sessions"]]
    options = chat_titles + ["➕ New Chat"]

    conv_id, sess_id = st.session_state["active_chat"]
    current_index = chat_ids.index(conv_id) if conv_id in chat_ids else len(options) - 1
    selected = st.selectbox("Select a conversation", options, index=current_index)

    if selected == "➕ New Chat":
        new_conv_id, new_session_id = create_conversation(st.session_state["user_id"], "New Chat")
        new_chat = {
            "id": new_conv_id,
            "session_id": new_session_id,
            "title": "New Chat",
            "history": []
        }
        st.session_state["chat_sessions"].append(new_chat)
        st.session_state["active_chat"] = (new_conv_id, new_session_id)
        st.stop() 
    else:
        selected_id = chat_ids[chat_titles.index(selected)]
        if conv_id != selected_id:
            selected_chat = next((c for c in st.session_state["chat_sessions"] if c["id"] == selected_id), None)
            if selected_chat:
                selected_chat["history"] = [{"role": r, "content": c} for r, c in load_messages(selected_id)]
                st.session_state["active_chat"] = (selected_id, selected_chat["session_id"])
                st.stop()

    st.header("📂 Your Project Files")
    
    handle_text_file()
    handle_json_file()
    handle_csv_file()
    handle_pdf_file()

    aggregated = gather_all_content(user_id=st.session_state.user_id)
    create_or_update_faiss_db(aggregated, session_id, st.session_state.user_id)
    build_faiss_from_db(st.session_state.user_id)

    uploaded_files = st.file_uploader(
        "Upload files",
        type=["txt", "json", "csv", "pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = os.path.join(DATA_FOLDER, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            uploaded_file.seek(0)

            content = None
            try:
                if uploaded_file.name.endswith(".txt"):
                    content = uploaded_file.read().decode("utf-8")
                elif uploaded_file.name.endswith(".json"):
                    content = json.load(uploaded_file)
                elif uploaded_file.name.endswith(".csv"):
                    content = pd.read_csv(uploaded_file).to_dict(orient="records")
                elif uploaded_file.name.endswith(".pdf"):
                    try:
                        reader = PdfReader(uploaded_file)
                        content = "\n".join(
                            page.extract_text() for page in reader.pages if page.extract_text()
                        )
                    except Exception as e:
                        st.warning(f"Could not read PDF {uploaded_file.name}: {e}")
                        content = ""
            except Exception as e:
                st.warning(f"Failed to process {uploaded_file.name}: {e}")
                content = ""

            aggregated_content = [{
                "filename": uploaded_file.name,
                "type": uploaded_file.name.split('.')[-1],
                "content": content
            }]

            create_or_update_faiss_db(aggregated_content, session_id, st.session_state.user_id)
            build_faiss_from_db(st.session_state.user_id)

        st.success(f"Uploaded and saved {len(uploaded_files)} file(s)!")

    # ---------- Interactive Chart Options ----------
    st.header("📊 Chart Options (Interactive)")
    if aggregated:
        dataset_names = [item["filename"] for item in aggregated]
        selected_dataset_name = st.selectbox("Select dataset", dataset_names)

        dataset_item = next((item for item in aggregated if item["filename"] == selected_dataset_name), None)
        if dataset_item:
            if isinstance(dataset_item["content"], pd.DataFrame):
                df = dataset_item["content"]
            elif isinstance(dataset_item["content"], list) and isinstance(dataset_item["content"][0], dict):
                df = pd.DataFrame(dataset_item["content"])
            else:
                df = None

            if df is not None:
                x_col = st.selectbox("X-axis column", df.columns, key=f"x_{selected_dataset_name}")
                y_col = st.selectbox("Y-axis column", df.columns, key=f"y_{selected_dataset_name}")
                chart_type = st.selectbox("Chart type", ["bar", "line", "scatter", "pie"], key=f"type_{selected_dataset_name}")
                aggregate = st.selectbox("Aggregate function", ["none", "sum", "avg"], key=f"agg_{selected_dataset_name}")

                if st.button("Generate Chart", key=f"btn_{selected_dataset_name}"):
                    fig, ax = plt.subplots()
                    df_plot = df.copy()
                    if aggregate in ["sum", "avg"]:
                        df_plot = df_plot.groupby(x_col, as_index=False)[y_col].agg("sum" if aggregate=="sum" else "mean")

                    if chart_type == "bar":
                        ax.bar(df_plot[x_col], df_plot[y_col])
                    elif chart_type == "line":
                        ax.plot(df_plot[x_col], df_plot[y_col], marker='o')
                    elif chart_type == "scatter":
                        ax.scatter(df_plot[x_col], df_plot[y_col])
                    elif chart_type == "pie":
                        if y_col:
                            ax.pie(df_plot[y_col], labels=df_plot[x_col], autopct="%1.1f%%")
                        else:
                            st.warning("Pie chart requires a Y column.")

                    ax.set_title(f"{chart_type.capitalize()} Chart: {selected_dataset_name}")
                    ax.set_xlabel(x_col if chart_type != "pie" else "")
                    ax.set_ylabel(y_col if chart_type not in ["pie"] else "")
                    plt.xticks(rotation=45)
                    plt.tight_layout()

                    st.pyplot(fig)
                    st.success("📊 Chart generated!")

# ---------- Start file watcher ----------
threading.Thread(target=start_watcher, daemon=True).start()

# ---------- Main Chat ----------
active_session = next(
    (chat for chat in st.session_state["chat_sessions"] if chat["id"] == st.session_state["active_chat"][0]),
    None
)

if active_session:
    conv_id, sess_id = active_session["id"], active_session["session_id"]
    active_session["history"] = [{"role": r, "content": c} for r, c in load_messages(conv_id)]

st.subheader("💬 Chat")
for msg in active_session["history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask about your files..."):
    save_message(conv_id, "user", user_input)
    active_session["history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if active_session["title"] == "New Chat":
        active_session["title"] = user_input[:30] + ("..." if len(user_input) > 30 else "")

    with st.chat_message("assistant"):
        ai_placeholder = st.empty()
        ai_placeholder.markdown("🤖 Thinking...")

        chart = try_generate_chart(user_input, aggregated, sess_id, st.session_state["user_id"])
        if chart:
            st.pyplot(chart)
            save_message(conv_id, "assistant", "📊 Generated a chart based on your request.")
            active_session["history"].append({
                "role": "assistant",
                "content": "📊 Generated a chart based on your request."
            })
            ai_placeholder.empty()
        else:
            answer = ask_ai_with_memory(
                question=user_input,
                session_id=sess_id,
                user_id=st.session_state["user_id"],
                stream_placeholder=ai_placeholder,
                top_k=5,
                max_chars=1000
            )
            save_message(conv_id, "assistant", answer)
            active_session["history"].append({
                "role": "assistant",
                "content": answer
            })
            ai_placeholder.markdown(answer)

if st.button("➕ New Chat"):
    new_conv_id, new_session_id = create_conversation(st.session_state["user_id"], "New Chat")
    new_chat = {"id": new_conv_id, "session_id": new_session_id, "title": "New Chat", "history": []}
    st.session_state["chat_sessions"].append(new_chat)
    st.session_state["active_chat"] = (new_conv_id, new_session_id)
    st.rerun()