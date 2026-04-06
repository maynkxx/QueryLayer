# app.py

import streamlit as st
import tempfile
import os
from rag_engine import build_vectorstore, get_answer

st.set_page_config(page_title="Doc Chatbot", page_icon="📄")
st.title("📄 Chat with your PDF")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# Sidebar
with st.sidebar:
    st.header("Upload Document")
    uploaded = st.file_uploader("Choose a PDF", type="pdf")

    if uploaded:
        pdf_bytes = uploaded.read()

        if uploaded.name != st.session_state.pdf_name:
            # Save to persistent temp path (avoids Mac temp file cleanup issue)
            tmp_path = os.path.join(tempfile.gettempdir(), f"rag_{uploaded.name}")
            with open(tmp_path, "wb") as f:
                f.write(pdf_bytes)

            print(f"✅ Saved PDF: {tmp_path} ({os.path.getsize(tmp_path)} bytes)")

            with st.spinner("Processing PDF..."):
                vs = build_vectorstore(tmp_path)

            if vs is not None:
                st.session_state.vectorstore = vs
                st.session_state.pdf_name = uploaded.name
                st.session_state.chat_history = []
                st.success("✅ Ready! Ask anything.")
            else:
                st.error("❌ Failed to process PDF. Check terminal for errors.")

        if st.session_state.pdf_name:
            st.info(f"📄 **{st.session_state.pdf_name}**")

        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

# Render full chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])
    if message["role"] == "assistant" and message.get("sources"):
        with st.expander("📄 Sources used"):
            for i, chunk in enumerate(message["sources"], 1):
                page = chunk.metadata.get("page", "?")
                st.markdown(f"**Chunk {i} — Page {page}:**")
                st.caption(chunk.page_content[:250] + "...")

# Chat input
if st.session_state.vectorstore is None:
    st.info("👈 Upload a PDF from the sidebar to get started.")
else:
    question = st.chat_input("Ask a question about your document...")

    if question:
        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        with st.spinner("Thinking..."):
            answer, sources = get_answer(question, st.session_state.vectorstore)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })

        st.rerun()