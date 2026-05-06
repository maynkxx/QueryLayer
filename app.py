import streamlit as st
import tempfile
import os
from rag_engine import (
    build_combined_vectorstore,
    get_answer,
    generate_summary,
    generate_suggested_questions
)

st.set_page_config(
    page_title="QueryLayer",
    page_icon="Q",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS: Light Theme ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #F7F5F2;
    color: #1A1A1A;
}

/* ── Hide Streamlit default header and footer bars ── */
[data-testid="stHeader"] {
    background-color: #F7F5F2 !important;
    border-bottom: none !important;
}

header[data-testid="stHeader"] {
    background: #F7F5F2 !important;
}

footer {
    display: none !important;
}

/* Hide the three-dot deploy menu bar at top */
#MainMenu {
    visibility: hidden;
}

[data-testid="stToolbar"] {
    display: none !important;
}

[data-testid="stDecoration"] {
    display: none !important;
}

/* ── Fix bottom black bar around chat input ── */
[data-testid="stBottom"] {
    background-color: #F7F5F2 !important;
    border-top: 1px solid #E5E0D8 !important;
}

[data-testid="stBottomBlockContainer"] {
    background-color: #F7F5F2 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E5E0D8;
}

[data-testid="stSidebar"] * {
    color: #1A1A1A !important;
}

[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li {
    color: #6B6560 !important;
    font-size: 0.85rem;
}

[data-testid="stSidebar"] h1 {
    color: #C0392B !important;
    font-size: 1.4rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background-color: #FDF9F6 !important;
    border: 1.5px dashed #C0392B !important;
    border-radius: 10px !important;
    padding: 8px !important;
}

[data-testid="stFileUploaderDropzone"] {
    background-color: #FDF9F6 !important;
}

[data-testid="stFileUploaderDropzoneInstructions"] * {
    color: #8C8078 !important;
}

/* Fix the uploaded file pill — dark background with unreadable text */
[data-testid="stFileUploaderFile"] {
    background-color: #F2EDE8 !important;
    border: 1px solid #E0D8D0 !important;
    border-radius: 6px !important;
}

[data-testid="stFileUploaderFile"] * {
    color: #1A1A1A !important;
}

[data-testid="stFileUploaderFileName"] {
    color: #1A1A1A !important;
}

[data-testid="stFileUploaderFileData"] {
    color: #6B6560 !important;
}

/* ── Buttons ── */
.stButton > button {
    background-color: #C0392B !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.45rem 1rem !important;
    transition: background-color 0.2s ease, transform 0.1s ease !important;
    width: 100%;
}

.stButton > button:hover {
    background-color: #A93226 !important;
    transform: translateY(-1px) !important;
}

.stButton > button:active {
    transform: translateY(0px) !important;
}

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
    background-color: #FFFFFF !important;
    color: #C0392B !important;
    border: 1.5px solid #C0392B !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    width: 100%;
}

[data-testid="stDownloadButton"] > button:hover {
    background-color: #FDF0EE !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background-color: #FFFFFF !important;
    border: 1.5px solid #C0392B !important;
    border-radius: 12px !important;
}

[data-testid="stChatInput"] textarea {
    background-color: #FFFFFF !important;
    color: #1A1A1A !important;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #B0A8A0 !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    border-radius: 12px !important;
    padding: 12px 16px !important;
    margin-bottom: 8px !important;
    border: 1px solid #E8E2DA !important;
}

/* User message */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background-color: #FDF0EE !important;
    border-color: #F0D5D0 !important;
}

/* Assistant message */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background-color: #FFFFFF !important;
    border-color: #E8E2DA !important;
}

/* Force all text inside chat messages to be dark */
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] strong,
[data-testid="stChatMessage"] em,
[data-testid="stChatMessage"] h1,
[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3,
[data-testid="stChatMessage"] code {
    color: #1A1A1A !important;
    line-height: 1.65 !important;
}

/* Fix markdown rendered inside chat — bold headers etc */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] strong,
[data-testid="stMarkdownContainer"] em,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 {
    color: #1A1A1A !important;
}

/* Inline code inside chat */
[data-testid="stChatMessage"] code {
    background-color: #F2EDE8 !important;
    color: #C0392B !important;
    padding: 1px 4px !important;
    border-radius: 4px !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E5E0D8 !important;
    border-radius: 10px !important;
}

[data-testid="stExpander"] summary {
    color: #C0392B !important;
    font-weight: 600 !important;
}

[data-testid="stExpander"] p,
[data-testid="stExpander"] li,
[data-testid="stExpander"] span {
    color: #4A4540 !important;
    font-size: 0.875rem !important;
}

/* ── Alerts ── */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    font-size: 0.85rem !important;
}

/* ── Divider ── */
hr {
    border-color: #E5E0D8 !important;
    margin: 12px 0 !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] * {
    color: #C0392B !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar {
    width: 6px;
}
::-webkit-scrollbar-track {
    background: #F7F5F2;
}
::-webkit-scrollbar-thumb {
    background: #D5CEC6;
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: #C0392B;
}

/* ── Custom components ── */
.ql-card {
    background-color: #FFFFFF;
    border: 1px solid #E5E0D8;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
}

.ql-badge {
    display: inline-block;
    background-color: #FDF0EE;
    color: #C0392B;
    border: 1px solid #F0C8C0;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-bottom: 8px;
}

.ql-title {
    font-size: 2rem;
    font-weight: 700;
    color: #1A1A1A;
    letter-spacing: -1px;
    margin-bottom: 0;
}

.ql-title span {
    color: #C0392B;
}

.ql-subtitle {
    color: #8C8078;
    font-size: 0.9rem;
    margin-top: 4px;
}

.ql-empty-state {
    text-align: center;
    padding: 80px 40px;
}

.ql-empty-title {
    font-size: 1.4rem;
    font-weight: 600;
    color: #4A4540;
    margin-bottom: 8px;
}

.ql-empty-sub {
    font-size: 0.875rem;
    color: #8C8078;
    line-height: 1.6;
}

.ql-doc-pill {
    display: inline-block;
    background-color: #F2EDE8;
    color: #4A4540;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 0.78rem;
    margin: 2px 3px 2px 0;
    border: 1px solid #E0D8D0;
}

.ql-section-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    color: #A09890;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.ql-card-title {
    font-weight: 600;
    color: #1A1A1A;
    margin-bottom: 4px;
    font-size: 0.95rem;
}

.ql-card-body {
    font-size: 0.8rem;
    color: #6B6560;
    line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)


# ── Session State Init ─────────────────────────────────────────────────────────
for key, default in {
    "chat_history": [],
    "vectorstore": None,
    "pdf_names": [],
    "summary": None,
    "suggested_questions": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h1>QueryLayer</h1>", unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#8C8078;font-size:0.82rem;margin-top:-8px;">Chat with your documents</p>',
        unsafe_allow_html=True
    )
    st.divider()

    # ── Upload section ─────────────────────────────────────────────────────────
    st.markdown('<div class="ql-section-label">Upload Documents</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        label="Drop PDFs here",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        current_names = sorted([f.name for f in uploaded_files])

        if current_names != sorted(st.session_state.pdf_names):
            pdf_paths = []
            for uploaded in uploaded_files:
                tmp_path = os.path.join(
                    tempfile.gettempdir(),
                    f"ql_{uploaded.name}"
                )
                with open(tmp_path, "wb") as f:
                    f.write(uploaded.read())
                pdf_paths.append(tmp_path)

            with st.spinner(f"Processing {len(pdf_paths)} PDF(s)..."):
                vs = build_combined_vectorstore(pdf_paths)

            if vs is not None:
                st.session_state.vectorstore = vs
                st.session_state.pdf_names = current_names
                st.session_state.chat_history = []

                with st.spinner("Generating summary..."):
                    st.session_state.summary = generate_summary(vs)

                with st.spinner("Generating questions..."):
                    st.session_state.suggested_questions = generate_suggested_questions(vs)

                st.success(f"{len(pdf_paths)} PDF(s) ready!")
            else:
                st.error("Failed to process PDFs. Check your files and try again.")

    # ── Loaded documents ───────────────────────────────────────────────────────
    if st.session_state.pdf_names:
        st.divider()
        st.markdown('<div class="ql-section-label">Loaded Documents</div>', unsafe_allow_html=True)
        pills_html = "".join(
            f'<span class="ql-doc-pill">{name}</span>'
            for name in st.session_state.pdf_names
        )
        st.markdown(pills_html, unsafe_allow_html=True)

    # ── Multilingual note ──────────────────────────────────────────────────────
    st.divider()
    st.markdown(
        '<div class="ql-section-label">Languages</div>'
        '<p style="color:#8C8078;font-size:0.8rem;line-height:1.5;">'
        'Ask in Hindi, Marathi, French, Spanish, and 50+ more languages.</p>',
        unsafe_allow_html=True
    )

    # ── Clear button ───────────────────────────────────────────────────────────
    st.divider()
    if st.button("Clear Everything", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.vectorstore = None
        st.session_state.pdf_names = []
        st.session_state.summary = None
        st.session_state.suggested_questions = []
        st.rerun()


# ── Main Area ──────────────────────────────────────────────────────────────────

if st.session_state.vectorstore is None:
    # ── Empty state ────────────────────────────────────────────────────────────
    st.markdown("""
        <div style="margin-top: 20px;">
            <div class="ql-title">Query<span>Layer</span></div>
            <div class="ql-subtitle">RAG-powered document intelligence</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
        <div class="ql-empty-state">
            <div class="ql-empty-title">No documents loaded</div>
            <div class="ql-empty-sub">
                Upload one or more PDFs from the sidebar to get started.<br>
                Ask questions in any language — QueryLayer will answer from your documents.
            </div>
        </div>
    """, unsafe_allow_html=True)

    # ── Feature cards ──────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
            <div class="ql-card">
                <div class="ql-card-title">Semantic Search</div>
                <div class="ql-card-body">Retrieves the most relevant chunks using
                FAISS vector search and CrossEncoder reranking.</div>
            </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
            <div class="ql-card">
                <div class="ql-card-title">Multi-query Retrieval</div>
                <div class="ql-card-body">Complex questions are decomposed into
                sub-queries to collect evidence from all relevant sections.</div>
            </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
            <div class="ql-card">
                <div class="ql-card-title">Multilingual</div>
                <div class="ql-card-body">Ask in 50+ languages. QueryLayer
                auto-detects and responds in your language.</div>
            </div>
        """, unsafe_allow_html=True)

else:
    # ── Header ─────────────────────────────────────────────────────────────────
    st.markdown("""
        <div style="margin-bottom:16px;">
            <div class="ql-title">Query<span>Layer</span></div>
            <div class="ql-subtitle">RAG-powered document intelligence</div>
        </div>
    """, unsafe_allow_html=True)

    # ── Summary ────────────────────────────────────────────────────────────────
    if st.session_state.summary:
        with st.expander("Document Summary", expanded=True):
            st.write(st.session_state.summary)

    # ── Suggested questions ────────────────────────────────────────────────────
    if st.session_state.suggested_questions:
        st.markdown(
            '<div class="ql-section-label" style="margin-top:8px;">Suggested Questions</div>',
            unsafe_allow_html=True
        )
        cols = st.columns(len(st.session_state.suggested_questions))
        for i, q in enumerate(st.session_state.suggested_questions):
            if cols[i].button(q, use_container_width=True, key=f"sq_{i}"):
                st.session_state.chat_history.append({"role": "user", "content": q})
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

    # ── Chat history ───────────────────────────────────────────────────────────
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
                    st.markdown(
                        f'<span class="ql-badge">Chunk {i} · {source_file} · Page {page}</span>',
                        unsafe_allow_html=True
                    )
                    st.caption(chunk.page_content[:300] + "...")
                    if i < len(message["sources"]):
                        st.divider()

    # ── Chat input ─────────────────────────────────────────────────────────────
    question = st.chat_input("Ask anything about your documents...")

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})

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

    # ── Export ─────────────────────────────────────────────────────────────────
    if st.session_state.chat_history:
        st.divider()
        export_text = ""
        for msg in st.session_state.chat_history:
            role = "You" if msg["role"] == "user" else "QueryLayer"
            export_text += f"{role}:\n{msg['content']}\n\n"

        st.download_button(
            label="Export Chat",
            data=export_text,
            file_name="querylayer_chat.txt",
            mime="text/plain",
            use_container_width=True
        )