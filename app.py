import streamlit as st
import uuid
import json
import pandas as pd
from PyPDF2 import PdfReader
import threading
import matplotlib.pyplot as plt

from scripts.content_manager import gather_all_content
from scripts.file_handlers import handle_text_file, handle_json_file, handle_csv_file, handle_pdf_file
from scripts.ai_engine import create_or_update_faiss_db, ask_ai_with_memory, build_faiss_from_db
from scripts.file_watcher import start_watcher

# ---------- Session ----------
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

session_id = st.session_state.session_id

# ---------- Page setup ----------
st.set_page_config(page_title="FileSense AI", layout="wide")
st.title("📦 FileSense AI")
st.caption("Chat with your project files — now with memory + citations.")

import re
import json
import pandas as pd
import matplotlib.pyplot as plt

def extract_json_from_text(text):
    """Extract the first JSON object found in a string."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None
    return None

def try_generate_chart(user_input, aggregated):
    import matplotlib.pyplot as plt
    import pandas as pd

    # Step 1: Check if it's a graph request
    keywords = ["plot", "graph", "chart", "visualize", "draw"]
    if not any(k in user_input.lower() for k in keywords):
        return None  # Not a chart request

    # Step 2: Gather dataset info for AI
    dataset_info = []
    for item in aggregated:
        if isinstance(item["content"], pd.DataFrame):
            dataset_info.append({"filename": item["filename"], "columns": list(item["content"].columns)})
        elif isinstance(item["content"], list) and isinstance(item["content"][0], dict):
            dataset_info.append({"filename": item["filename"], "columns": list(item["content"][0].keys())})

    # Ask AI for chart spec
    chart_spec_str = ask_ai_with_memory(
        question=f"Create a JSON spec for a chart based on this query: {user_input}. "
                 f"Available datasets: {dataset_info}. "
                 "Format: { 'dataset': 'filename', 'chart_type': 'bar/line/scatter/pie', "
                 "'x': 'column', 'y': 'column', 'title': 'optional', 'aggregate': 'sum/avg/none' }. "
                 "Output JSON only.",
        session_id=session_id,
        stream_placeholder=None,
        top_k=5,
        max_chars=500
    )

    try:
        chart_spec = json.loads(chart_spec_str)
    except:
        return None

    # Step 3: Pick dataset
    dataset_name = chart_spec.get("dataset")
    dataset_item = next((item for item in aggregated if item["filename"] == dataset_name), None)
    if not dataset_item and aggregated:
        dataset_item = aggregated[0]
    if not dataset_item:
        return None

    # Convert dataset into DataFrame
    if isinstance(dataset_item["content"], pd.DataFrame):
        df = dataset_item["content"]
    elif isinstance(dataset_item["content"], list) and isinstance(dataset_item["content"][0], dict):
        df = pd.DataFrame(dataset_item["content"])
    else:
        return None

    # Step 4: Apply aggregation if specified
    agg_func = chart_spec.get("aggregate", "none")
    x_col = chart_spec.get("x")
    y_col = chart_spec.get("y")
    if agg_func in ["sum", "avg"]:
        df = df.groupby(x_col, as_index=False)[y_col].agg("sum" if agg_func=="sum" else "mean")

    # Step 5: Render chart
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
            return None  # Pie chart needs y values

    ax.set_title(chart_spec.get("title", "Generated Chart"))
    ax.set_xlabel(x_col if chart_type != "pie" else "")
    ax.set_ylabel(y_col if chart_type not in ["pie"] else "")

    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

# ---------- Sidebar for interactive chart options ----------
# ---------- Sidebar ----------
with st.sidebar:
    st.header("📂 Project Files")
    handle_text_file()
    handle_json_file()
    handle_csv_file()
    handle_pdf_file()

    # Gather aggregated content first
    aggregated = gather_all_content()
    create_or_update_faiss_db(aggregated, session_id)
    build_faiss_from_db()

    uploaded_files = st.file_uploader(
        "Upload files",
        type=["txt", "json", "csv", "pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name.endswith(".txt"):
                content = uploaded_file.read().decode("utf-8")
            elif uploaded_file.name.endswith(".json"):
                content = json.load(uploaded_file)
            elif uploaded_file.name.endswith(".csv"):
                content = pd.read_csv(uploaded_file).to_dict(orient="records")
            elif uploaded_file.name.endswith(".pdf"):
                reader = PdfReader(uploaded_file)
                content = "\n".join(
                    page.extract_text() for page in reader.pages if page.extract_text()
                )
            else:
                content = uploaded_file.read().decode("utf-8")

            aggregated_content = [{
                "filename": uploaded_file.name,
                "type": uploaded_file.name.split('.')[-1],
                "content": content
            }]

            create_or_update_faiss_db(aggregated_content, session_id)
            build_faiss_from_db()

        st.success(f"Uploaded {len(uploaded_files)} file(s)!")

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
                x_col = st.selectbox("X-axis column", df.columns)
                y_col = st.selectbox("Y-axis column", df.columns)
                chart_type = st.selectbox("Chart type", ["bar", "line", "scatter", "pie"])
                aggregate = st.selectbox("Aggregate function", ["none", "sum", "avg"])

                if st.button("Generate Chart"):
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
st.subheader("💬 Chat")

# Render existing conversation
for msg in st.session_state.conversation_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if user_input := st.chat_input("Ask about your files..."):
    # Log user message
    st.session_state.conversation_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Placeholder for assistant response
    with st.chat_message("assistant"):
        ai_placeholder = st.empty()
        ai_placeholder.markdown("🤖 Thinking...")

        # 1️⃣ Attempt to generate chart first
        chart = try_generate_chart(user_input, aggregated)

        if chart:
            # Render the chart
            st.pyplot(chart)

            # Log chart generation in conversation history
            st.session_state.conversation_history.append({
                "role": "assistant",
                "content": "📊 Generated a chart based on your request."
            })

            ai_placeholder.empty()  # remove "thinking..." placeholder
        else:
            # 2️⃣ Fallback to normal AI response
            answer = ask_ai_with_memory(
                question=user_input,
                session_id=session_id,
                stream_placeholder=ai_placeholder,
                top_k=5,
                max_chars=1000
            )

            # Log AI text response
            st.session_state.conversation_history.append({
                "role": "assistant",
                "content": answer
            })

            ai_placeholder.markdown(answer)