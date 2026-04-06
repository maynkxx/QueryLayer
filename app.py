import streamlit as st
import tempfile
import os
from rag_engine import build_vectorstore, get_answer

st.set_page_config(page_title="Doc Chatbot", page_icon="📄")
st.title("📄 Chat with your PDF")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# Sidebar for upload + info
with st.sidebar:
    st.header("Upload Document")
    uploaded = st.file_uploader("Choose a PDF", type="pdf")

    if uploaded:
        # Only reprocess if a NEW pdf is uploaded
        if uploaded.name != st.session_state.pdf_name:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                f.write(uploaded.read())
                tmp_path = f.name

            with st.spinner("Processing PDF..."):
                st.session_state.vectorstore = build_vectorstore(tmp_path)
                st.session_state.pdf_name = uploaded.name
                st.session_state.chat_history = []  # clear history on new PDF

            st.success(f"✅ Ready!")

        st.info(f"📄 **{st.session_state.pdf_name}**")

        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

# Main chat area — render full history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])
    if message["role"] == "assistant" and "sources" in message:
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
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        # Get answer
        with st.spinner("Thinking..."):
            answer, sources = get_answer(question, st.session_state.vectorstore)

        # Add assistant message to history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })

        st.rerun()