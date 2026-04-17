from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

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
        raise ValueError("GROQ_API_KEY not found.")
    return key

def get_client():
    global _client
    if _client is None:
        _client = Groq(api_key=get_api_key())
    return _client

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5"
        )
    return _embeddings

def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker

def rerank_docs(question, docs):
    reranker = get_reranker()
    pairs = [[question, d.page_content] for d in docs]
    scores = reranker.predict(pairs)
    scored_docs = list(zip(docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in scored_docs[:3]]

def build_vectorstore_from_file(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    if not docs:
        raise ValueError("No documents loaded")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(docs)
    return FAISS.from_documents(chunks, get_embeddings())

def build_combined_vectorstore(pdf_paths):
    combined = None
    for path in pdf_paths:
        vs = build_vectorstore_from_file(path)
        if combined is None:
            combined = vs
        else:
            combined.merge_from(vs)
    return combined

def detect_language(text):
    try:
        from langdetect import detect
        return detect(text)
    except Exception:
        return "en"

def get_answer(question, vectorstore, chat_history=None):
    if chat_history is None:
        chat_history = []
    
    user_lang = detect_language(question)
    docs = vectorstore.similarity_search(question, k=5)

    if len(docs) > 2:
        docs = rerank_docs(question, docs)

    if not docs:
        return "Could not find relevant content. Try rephrasing.", []

    context = "\n\n".join(
        [
            f"[Source: Page {d.metadata.get('page', '?')}]\n{d.page_content.strip()}"
            for d in docs
        ]
    )

    if len(context) < 120:
        return "The document does not contain enough information.", []

    history_text = ""
    if chat_history:
        for msg in chat_history[-6:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content']}\n"

    lang_instruction = ""
    if user_lang != "en":
        lang_instruction = f"\nIMPORTANT: Respond in language '{user_lang}'."

    prompt = f"""You are a helpful document assistant.
Instructions:

Answer using ONLY the provided context.
Do NOT use external knowledge.
If the answer is not present, say: "The document does not contain enough information."
Use only relevant information.
Keep answers clear and structured.
Include exact numbers if available.{lang_instruction}
--- Conversation History ---
{history_text if history_text else "No previous conversation."}
--- Document Context ---
{context}
--- Question ---
{question}
Answer:"""

    try:
        response = get_client().chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": f"Answer in the same language as user. Language: {user_lang}"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_tokens=800,
        )
        answer = response.choices[0].message.content.strip()

    except Exception as e:
        answer = f"Error: {str(e)}"
        docs = []

    return answer, docs

def generate_summary(vectorstore):
    try:
        docs = vectorstore.similarity_search("overview summary", k=4)
        context = "\n\n".join([d.page_content.strip() for d in docs])
        prompt = f"""Write a short summary:\n{context}"""

        response = get_client().chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return ""

def generate_suggested_questions(vectorstore):
    try:
        docs = vectorstore.similarity_search("main topics", k=3)
        context = "\n\n".join([d.page_content.strip() for d in docs])
        prompt = f"""Generate 3 questions:\n{context}"""

        response = get_client().chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=150,
        )
        raw = response.choices[0].message.content.strip()
        return [q.strip() for q in raw.split("\n") if q.strip()][:3]
    except Exception:
        return []