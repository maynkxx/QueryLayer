# app.py

import streamlit as st
import tempfile
import os
from rag_engine import (
    build_combined_vectorstore,
    get_answer,
    generate_summary,
    generate_suggested_questions
)

st.set_page_config(page_title="DocChat AI", page_icon="", layout="wide")

# ── Session State Init ─────────────────────────────────────
for key, default in {
    "chat_history": [],
    "vectorstore": None,
    "pdf_names": [],
    "summary": None,
    "suggested_questions": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.title("DocChat AI")
    st.caption("Chat with your documents in any language")
    st.divider()

    # Multi-file uploader
    uploaded_files = st.file_uploader(
        "Upload PDFs (one or more)",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:
        current_names = sorted([f.name for f in uploaded_files])

        # Only reprocess if files changed
        if current_names != sorted(st.session_state.pdf_names):
            pdf_paths = []

            for uploaded in uploaded_files:
                tmp_path = os.path.join(
                    tempfile.gettempdir(),
                    f"rag_{uploaded.name}"
                )
                with open(tmp_path, "wb") as f:
                    f.write(uploaded.read())
                pdf_paths.append(tmp_path)
                print(f"Saved: {tmp_path}")

            with st.spinner(f"Processing {len(pdf_paths)} PDF(s)..."):
                vs = build_combined_vectorstore(pdf_paths)

            if vs is not None:
                st.session_state.vectorstore = vs
                st.session_state.pdf_names = current_names
                st.session_state.chat_history = []

                # Auto-generate summary + suggested questions
                with st.spinner("Generating summary..."):
                    st.session_state.summary = generate_summary(vs)

                with st.spinner("Generating suggested questions..."):
                    st.session_state.suggested_questions = generate_suggested_questions(vs)

                st.success(f"{len(pdf_paths)} PDF(s) ready!")
            else:
                st.error("Failed to process PDFs. Check terminal.")

    # Show uploaded files list
    if st.session_state.pdf_names:
        st.divider()
        st.markdown("**Loaded documents:**")
        for name in st.session_state.pdf_names:
            st.markdown(f"- {name}")

    # Language info
    st.divider()
    st.markdown("** Multilingual support**")
    st.caption("Ask questions in any language — Hindi, Marathi, French, Spanish, and 50+ more.")

    # Clear button
    st.divider()
    if st.button("Clear Everything", use_container_width=True):
        for key in ["chat_history", "vectorstore", "pdf_names", "summary", "suggested_questions"]:
            st.session_state[key] = [] if key != "vectorstore" and key != "summary" else None
        st.rerun()

# ── Main Area ──────────────────────────────────────────────
if st.session_state.vectorstore is None:
    # Empty state
    st.title("DocChat AI")
    st.markdown("### Upload PDFs from the sidebar to get started")
    st.info("You can upload multiple PDFs and ask questions across all of them. Ask in any language!")

else:
    # Header
    st.title("DocChat AI")

    # Show auto summary
    if st.session_state.summary:
        with st.expander("Document Summary", expanded=True):
            st.write(st.session_state.summary)

    # Show suggested questions as clickable buttons
    if st.session_state.suggested_questions:
        st.markdown("** Suggested questions — click to ask:**")
        cols = st.columns(len(st.session_state.suggested_questions))
        for i, q in enumerate(st.session_state.suggested_questions):
            if cols[i].button(q, use_container_width=True):
                # Treat button click as a question
                st.session_state.chat_history.append({
                    "role": "user", "content": q
                })
                with st.spinner("Thinking..."):
                    answer, sources = get_answer(
                        q,
                        st.session_state.vectorstore,
                        st.session_state.chat_history
                    )
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
                st.rerun()

    st.divider()

    # Render full chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Sources used"):
                for i, chunk in enumerate(message["sources"], 1):
                    page = chunk.metadata.get("page", "?")
                    source_file = os.path.basename(
                        chunk.metadata.get("source", "unknown")
                    )
                    st.markdown(f"**Chunk {i} — {source_file} — Page {page}:**")
                    st.caption(chunk.page_content[:250] + "...")

    # Chat input
    question = st.chat_input("Ask anything about your documents (any language)...")

    if question:
        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        with st.spinner("Thinking..."):
            answer, sources = get_answer(
                question,
                st.session_state.vectorstore,
                st.session_state.chat_history
            )

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })

        st.rerun()

    # Chat export button
    if st.session_state.chat_history:
        st.divider()
        export_text = ""
        for msg in st.session_state.chat_history:
            role = "You" if msg["role"] == "user" else "Assistant"
            export_text += f"{role}:\n{msg['content']}\n\n"

        st.download_button(
            label="Download Chat",
            data=export_text,
            file_name="chat_export.txt",
            mime="text/plain"
        )