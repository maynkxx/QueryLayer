import streamlit as st
import tempfile
import os

from rag_engine import build_vectorstore, get_answer

st.set_page_config(page_title="Doc Chatbot", page_icon="📄")

st.title("📄 Chat with your PDF")

# Upload PDF
uploaded = st.file_uploader("Upload a PDF", type="pdf")

if uploaded:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(uploaded.read())
        tmp_path = f.name

    # Build vectorstore only once
    if "vectorstore" not in st.session_state:
        with st.spinner("Processing your PDF..."):
            st.session_state.vectorstore = build_vectorstore(tmp_path)
        st.success("✅ Ready! Ask anything about the document.")

    # Chat input
    question = st.chat_input("Ask a question...")

    if question:
        # Show user message
        st.chat_message("user").write(question)

        # Get answer
        with st.spinner("Thinking..."):
            answer, sources = get_answer(
                question, st.session_state.vectorstore
            )

        # Show AI response
        st.chat_message("assistant").write(answer)

        # Show sources
        with st.expander("📄 Sources used"):
            for i, doc in enumerate(sources, 1):
                st.markdown(f"**Chunk {i}:** {doc.page_content[:200]}...")