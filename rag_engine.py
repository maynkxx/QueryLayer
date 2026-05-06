from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder
from groq import Groq
import os
import re
from dotenv import load_dotenv

load_dotenv()

# ── Lazy-loaded singletons ─────────────────────────────────────────────────────
# Models are heavy — load once, reuse forever.

_embeddings = None
_client = None
_reranker = None


def get_api_key():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("GROQ_API_KEY")
        except Exception:
            pass
    if not key:
        raise ValueError("GROQ_API_KEY not found in environment or Streamlit secrets.")
    return key


def get_client():
    global _client
    if _client is None:
        _client = Groq(api_key=get_api_key())
    return _client


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        # BGE-base chosen for its strong English retrieval performance on BEIR benchmarks
        # while remaining lightweight enough for CPU inference.
        _embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5"
        )
    return _embeddings


def get_reranker():
    global _reranker
    if _reranker is None:
        # CrossEncoder reranker: scores each (query, chunk) pair jointly,
        # giving much higher precision than bi-encoder cosine similarity alone.
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


# ── Retrieval & Reranking ──────────────────────────────────────────────────────

def rerank_docs(question, docs):
    """
    Rerank retrieved chunks using a CrossEncoder model.

    FAISS retrieval via bi-encoder embeddings is fast but imprecise —
    it finds semantically similar chunks, not necessarily the most relevant ones.
    The CrossEncoder jointly encodes (question, chunk) and produces a relevance
    score, improving precision significantly. We retrieve k=5 with FAISS,
    then keep the top 3 after reranking.
    """
    reranker = get_reranker()
    pairs = [[question, d.page_content] for d in docs]
    scores = reranker.predict(pairs)
    scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored_docs[:3]]


def generate_sub_queries(question):
    """
    Decompose a complex question into 2-3 focused sub-queries.

    Single-query retrieval fails on questions that span multiple sections of a
    document — for example, "How does the Why Culture connect to Alexa+?" requires
    chunks about the philosophy AND chunks about the product. By decomposing the
    question into targeted sub-queries and retrieving separately for each, we
    collect evidence from all relevant parts of the document before generating
    an answer.

    Uses the LLM itself to do the decomposition, which keeps sub-queries
    semantically aligned with the document's language.
    """
    prompt = f"""You are a search query assistant. Break the following question into 2-3 short, 
focused search queries that together would help retrieve all the information needed to answer it.

Rules:
- Each sub-query must be independently searchable
- Sub-queries should cover different aspects of the original question
- Keep each sub-query under 10 words
- Output ONLY the sub-queries, one per line, no numbering or bullets

Question: {question}

Sub-queries:"""

    try:
        response = get_client().chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100,
        )
        raw = response.choices[0].message.content.strip()
        sub_queries = [
            line.strip()
            for line in raw.split("\n")
            if line.strip() and len(line.strip()) > 5
        ]
        return sub_queries[:3] if sub_queries else [question]
    except Exception:
        return [question]


def is_complex_question(question):
    """
    Determine whether a question is complex enough to warrant multi-query retrieval.

    Simple factual questions (single entity, single fact) are answered well by
    standard single-query retrieval. Multi-query retrieval is reserved for questions
    that require connecting ideas across sections, comparing entities, or reasoning
    over aggregated evidence — which is where single-query retrieval fails.

    Heuristics used:
    - Connective words: "and", "how does...connect", "compare", "relate"
    - Reasoning words: "why", "explain", "analyze", "implications"
    - Synthesis words: "overall", "across", "both", "despite"
    - Question length above threshold (longer questions tend to be multi-part)
    """
    question_lower = question.lower()

    complexity_signals = [
        " and ",
        "connect",
        "relate",
        "compare",
        "both",
        "despite",
        "across",
        "overall",
        "how does",
        "why does",
        "explain",
        "analyze",
        "implications",
        "justify",
        "what specific",
    ]

    signal_count = sum(1 for signal in complexity_signals if signal in question_lower)

    # Complex if multiple signals or question is long
    return signal_count >= 2 or len(question.split()) > 15


def multi_query_retrieve(question, vectorstore):
    """
    Run multi-query retrieval for complex questions.

    Pipeline:
    1. Decompose question into 2-3 focused sub-queries via LLM
    2. Run FAISS similarity search for each sub-query independently
    3. Deduplicate chunks by page_content to avoid repeating the same evidence
    4. Rerank the merged pool against the original question using CrossEncoder
    5. Return top-5 most relevant chunks from the combined pool

    This solves the core failure mode where a question spans multiple sections —
    each sub-query targets a different part of the document, and the merged pool
    contains evidence from all relevant sections before the LLM sees it.
    """
    sub_queries = generate_sub_queries(question)

    seen_contents = set()
    all_docs = []

    for sub_query in sub_queries:
        results = vectorstore.similarity_search(sub_query, k=4)
        for doc in results:
            content_key = doc.page_content.strip()[:200]
            if content_key not in seen_contents:
                seen_contents.add(content_key)
                all_docs.append(doc)

    if not all_docs:
        return []

    # Rerank the full merged pool against the original question
    reranker = get_reranker()
    pairs = [[question, d.page_content] for d in all_docs]
    scores = reranker.predict(pairs)
    scored_docs = sorted(zip(all_docs, scores), key=lambda x: x[1], reverse=True)

    # Return top 5 — more context for complex questions
    return [doc for doc, _ in scored_docs[:5]]


# ── Document Processing ────────────────────────────────────────────────────────

def build_vectorstore_from_file(pdf_path):
    """
    Load a single PDF, chunk it, embed chunks, and store in a FAISS index.

    Chunk size 1000 / overlap 150 chosen to:
    - Keep chunks within LLM context limits
    - Preserve enough context per chunk for meaningful retrieval
    - Overlap prevents answers from being split across chunk boundaries
    """
    if not pdf_path or not os.path.exists(pdf_path):
        raise ValueError(f"PDF path is invalid or file does not exist: {pdf_path}")

    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()

    if not docs:
        raise ValueError("No content could be extracted from the PDF.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(docs)

    if not chunks:
        raise ValueError("Document was loaded but produced no chunks after splitting.")

    return FAISS.from_documents(chunks, get_embeddings())


def build_combined_vectorstore(pdf_paths):
    """
    Build a single merged FAISS index from multiple PDFs.
    Allows cross-document retrieval in a single similarity search call.
    """
    if not pdf_paths:
        raise ValueError("No PDF paths provided.")

    combined = None
    for path in pdf_paths:
        vs = build_vectorstore_from_file(path)
        if combined is None:
            combined = vs
        else:
            combined.merge_from(vs)
    return combined


# ── Language Detection ─────────────────────────────────────────────────────────

def detect_language(text):
    try:
        from langdetect import detect
        return detect(text)
    except Exception:
        return "en"


# ── Core Q&A ──────────────────────────────────────────────────────────────────

def get_answer(question, vectorstore, chat_history=None):
    """
    Full RAG pipeline: retrieve → rerank → generate.

    For simple questions:
      1. Embed question, retrieve top-5 chunks via FAISS
      2. Rerank with CrossEncoder, keep top-3

    For complex questions (multi-part, comparative, synthesis):
      1. Decompose into 2-3 sub-queries via LLM
      2. Retrieve top-4 chunks per sub-query independently
      3. Deduplicate and merge all chunks into one pool
      4. Rerank entire pool against original question, keep top-5
      This ensures evidence from all relevant sections is collected
      before the LLM generates an answer.

    Final step for both paths:
      5. Build structured prompt with conversation history + context
      6. Generate answer via LLaMA 3.1 on Groq, grounded strictly in context
    """
    # ── Input validation ───────────────────────────────────────────────────────
    if not question or not question.strip():
        return "Please enter a valid question.", []

    if vectorstore is None:
        return "No document loaded. Please upload a PDF first.", []

    if chat_history is None:
        chat_history = []

    user_lang = detect_language(question)

    # ── Retrieval: simple vs complex path ─────────────────────────────────────
    if is_complex_question(question):
        # Multi-query path: decompose → retrieve per sub-query → merge → rerank
        docs = multi_query_retrieve(question, vectorstore)
    else:
        # Standard path: single query → retrieve → rerank
        docs = vectorstore.similarity_search(question, k=5)
        if len(docs) > 2:
            docs = rerank_docs(question, docs)

    if not docs:
        return "Could not find relevant content in the document. Try rephrasing your question.", []

    context = "\n\n".join(
        f"[Source: Page {d.metadata.get('page', '?')}]\n{d.page_content.strip()}"
        for d in docs
    )

    if len(context) < 120:
        return "The document does not contain enough information to answer this question.", []

    # ── Conversation history ───────────────────────────────────────────────────
    history_text = ""
    if chat_history:
        for msg in chat_history[-6:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"

    # ── Language instruction ───────────────────────────────────────────────────
    lang_instruction = ""
    if user_lang != "en":
        lang_instruction = f"\nIMPORTANT: The user is writing in language code '{user_lang}'. You MUST respond in that same language."

    # ── Prompt ────────────────────────────────────────────────────────────────
    prompt = f"""You are a precise and helpful document assistant.

INSTRUCTIONS:
- Answer ONLY using information present in the Document Context below.
- Do NOT use any external knowledge or assumptions.
- If the answer is not found in the context, respond with exactly: "The document does not contain enough information to answer this question."
- Structure your answer clearly. Use bullet points or numbered lists when listing multiple items.
- For questions that ask you to connect, compare, or synthesize — reason explicitly across the evidence provided. Do not just list facts; explain the relationship.
- Include exact figures, names, or dates from the document when available.
- Be concise — avoid unnecessary filler or repetition.{lang_instruction}

--- CONVERSATION HISTORY ---
{history_text if history_text else "No previous conversation."}

--- DOCUMENT CONTEXT ---
{context}

--- QUESTION ---
{question}

Answer:"""

    # ── LLM call ──────────────────────────────────────────────────────────────
    try:
        response = get_client().chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are a document Q&A assistant. Answer strictly from the provided context. "
                        f"When asked to connect or synthesize ideas, reason explicitly across all evidence provided. "
                        f"Respond in the same language as the user (detected: {user_lang})."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_tokens=1000,
        )
        answer = response.choices[0].message.content.strip()

    except Exception as e:
        answer = f"An error occurred while generating the answer: {str(e)}"
        docs = []

    return answer, docs


# ── Summary Generation ────────────────────────────────────────────────────────

def generate_summary(vectorstore):
    """
    Generate a structured executive summary of the uploaded document.

    Uses a broad query ('overview introduction purpose') to retrieve the most
    representative chunks from the document, then prompts the LLM to produce
    a concise, structured summary rather than a raw extraction.
    """
    if vectorstore is None:
        return ""

    try:
        docs = vectorstore.similarity_search("overview introduction purpose main topic", k=4)
        context = "\n\n".join(d.page_content.strip() for d in docs)

        prompt = f"""You are a document analyst. Based ONLY on the content below, write a concise executive summary.

Your summary must:
- Start with one sentence describing what this document is about
- Cover the key topics, findings, or main points in 3-5 bullet points
- End with one sentence about who this document is relevant for or what action it implies
- Be no longer than 150 words total
- Use only information present in the content — do not add external knowledge

DOCUMENT CONTENT:
{context}

Executive Summary:"""

        response = get_client().chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=250,
        )
        return response.choices[0].message.content.strip()

    except Exception:
        return ""


# ── Suggested Questions Generation ────────────────────────────────────────────

def generate_suggested_questions(vectorstore):
    """
    Generate 3 insightful suggested questions based on document content.

    Questions are designed to be:
    - Specific to the actual content of the document (not generic)
    - Genuinely answerable from the document
    - Varied in type: one factual, one analytical, one practical

    The parser handles multiple LLM output formats:
    numbered lists (1. ...), bullet lists (- ...), or plain lines.
    """
    if vectorstore is None:
        return []

    try:
        docs = vectorstore.similarity_search("main topics key facts important details", k=3)
        context = "\n\n".join(d.page_content.strip() for d in docs)

        prompt = f"""You are a document analyst helping users explore a document through questions.

Based ONLY on the document content below, generate exactly 3 questions that:
1. Are specific to the actual content — avoid vague or generic questions
2. Are clearly answerable from the document
3. Vary in type: one should ask about a specific fact, one about a process or explanation, one about implications or decisions

FORMAT RULES:
- Output exactly 3 questions, one per line
- Do NOT number them or add bullet points — plain text only, one question per line
- Each question must end with a question mark
- Each question should be under 15 words

DOCUMENT CONTENT:
{context}

Questions:"""

        response = get_client().chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=150,
        )
        raw = response.choices[0].message.content.strip()

        # ── Robust parser ──────────────────────────────────────────────────────
        # Handles: numbered (1. Q), bulleted (- Q), or plain lines
        questions = []
        for line in raw.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Strip leading numbering like "1.", "1)", "- ", "* "
            line = re.sub(r"^[\d]+[.)]\s*", "", line)
            line = re.sub(r"^[-*•]\s*", "", line)
            line = line.strip()
            if line and "?" in line:
                questions.append(line)
            if len(questions) == 3:
                break

        return questions

    except Exception:
        return []