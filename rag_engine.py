# rag_engine.py

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

# ── API Key ────────────────────────────────────────────────
def get_api_key():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("GROQ_API_KEY")
        except Exception:
            pass
    if not key:
        raise ValueError("GROQ_API_KEY not found.")
    return key

client = Groq(api_key=get_api_key())

# ── Embedding Model (cached globally) ─────────────────────
_embeddings = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        print("Loading embedding model...")
        # multilingual model — supports 50+ languages
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        print("Multilingual embedding model loaded")
    return _embeddings


# ── Build vectorstore from ONE pdf ────────────────────────
def build_vectorstore_from_file(pdf_path):
    try:
        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()
        print(f"Loaded {len(docs)} pages from {pdf_path}")

        if not docs:
            print("No pages loaded!")
            return None

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )
        chunks = splitter.split_documents(docs)
        print(f"Created {len(chunks)} chunks")

        vectorstore = FAISS.from_documents(chunks, get_embeddings())
        return vectorstore

    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return None


# ── Build MERGED vectorstore from MULTIPLE pdfs ───────────
def build_combined_vectorstore(pdf_paths):
    """
    Accepts a list of PDF file paths.
    Merges all into one FAISS vectorstore.
    """
    combined = None

    for path in pdf_paths:
        vs = build_vectorstore_from_file(path)
        if vs is None:
            continue
        if combined is None:
            combined = vs
        else:
            # FAISS merge — this is the key multi-PDF technique
            combined.merge_from(vs)
            print(f"Merged vectorstore — total index size: {combined.index.ntotal}")

    if combined is None:
        print("No vectorstores were built.")
    return combined


# ── Language Detection ─────────────────────────────────────
def detect_language(text):
    """
    Simple language detection using langdetect.
    Falls back to English if detection fails.
    """
    try:
        from langdetect import detect
        lang = detect(text)
        print(f"Detected language: {lang}")
        return lang
    except Exception:
        return "en"


# ── Get Answer (with memory + multilingual) ───────────────
def get_answer(question, vectorstore, chat_history=None):
    """
    Args:
        question: user's question (any language)
        vectorstore: FAISS vectorstore
        chat_history: list of {"role": "user"/"assistant", "content": "..."}
    """
    if chat_history is None:
        chat_history = []

    # Step 1 — Detect user's language
    user_lang = detect_language(question)

    # Step 2 — Retrieve relevant chunks
    docs = vectorstore.similarity_search(question, k=5)

    print(f"\n{'='*50}")
    print(f"Question: {question}")
    print(f"Language detected: {user_lang}")
    print(f"Docs retrieved: {len(docs)}")
    for i, doc in enumerate(docs):
        print(f"Chunk {i+1} (page {doc.metadata.get('page','?')}): {doc.page_content[:150]}")
    print(f"{'='*50}\n")

    if not docs:
        return "Could not find relevant content. Try rephrasing.", []

    # Step 3 — Build context from retrieved chunks
    context = "\n\n".join([d.page_content.strip() for d in docs])

    # Step 4 — Build conversation history string for memory
    history_text = ""
    if chat_history:
        # Only use last 6 messages to avoid token overflow
        recent = chat_history[-6:]
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"

    # Step 5 — Build multilingual prompt
    lang_instruction = ""
    if user_lang != "en":
        lang_instruction = f"\nIMPORTANT: The user asked in language code '{user_lang}'. You MUST respond in that same language."

    prompt = f"""You are a helpful multilingual document assistant.

Instructions:
- Answer ONLY using the provided document context below.
- If the question refers to previous conversation, use the conversation history.
- If the answer is not in the context, say: "The document does not contain enough information."
- Be clear, structured, and concise.{lang_instruction}

--- Conversation History ---
{history_text if history_text else "No previous conversation."}

--- Document Context ---
{context}

--- Current Question ---
{question}

Answer:"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": f"You are a multilingual document assistant. Always answer in the same language the user used. Language detected: {user_lang}."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_tokens=1024
        )
        answer = response.choices[0].message.content.strip()

    except Exception as e:
        answer = f"Error: {str(e)}"
        docs = []

    return answer, docs


# ── Auto Summary on Upload ─────────────────────────────────
def generate_summary(vectorstore):
    """
    Generates a short summary of the uploaded document(s).
    Called automatically after upload.
    """
    try:
        # Grab a broad sample of chunks
        docs = vectorstore.similarity_search("overview summary introduction", k=6)
        context = "\n\n".join([d.page_content.strip() for d in docs])

        prompt = f"""Based on the following document content, write a short 3-5 sentence summary 
of what this document is about. Be concise and clear.

Content:
{context}

Summary:"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Could not generate summary: {str(e)}"


# ── Suggested Questions ────────────────────────────────────
def generate_suggested_questions(vectorstore):
    """
    Generates 3 relevant questions the user could ask about the document.
    """
    try:
        docs = vectorstore.similarity_search("main topics key points", k=4)
        context = "\n\n".join([d.page_content.strip() for d in docs])

        prompt = f"""Based on the following document content, generate exactly 3 interesting 
questions a user might want to ask. Return ONLY the 3 questions, one per line, 
no numbering, no extra text.

Content:
{context}

Questions:"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=200
        )
        raw = response.choices[0].message.content.strip()
        questions = [q.strip() for q in raw.split("\n") if q.strip()]
        return questions[:3]

    except Exception as e:
        print(f"Error generating questions: {e}")
        return []