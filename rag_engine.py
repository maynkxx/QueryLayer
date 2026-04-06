# rag_engine.py

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from groq import Groq
import os
from dotenv import load_dotenv
import streamlit as st

# 🔹 Load environment variables
load_dotenv()

# 🔹 API key handling (local + deployed)
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    api_key = st.secrets.get("GROQ_API_KEY")

if not api_key:
    raise ValueError("❌ GROQ_API_KEY not found in .env or Streamlit secrets.")

# 🔹 Initialize Groq client
client = Groq(api_key=api_key)

# 🔹 Cache embeddings
_embeddings = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        print("🔄 Loading embedding model...")
        _embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        print("✅ Embedding model loaded")
    return _embeddings


# 🔹 Build vector store
def build_vectorstore(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    print(f"✅ Loaded {len(docs)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = splitter.split_documents(docs)
    print(f"✅ Created {len(chunks)} chunks")

    vectorstore = FAISS.from_documents(chunks, get_embeddings())
    print("✅ Vector store created successfully")

    return vectorstore


# 🔹 Get answer
def get_answer(question, vectorstore):

    # 🔍 Retrieve relevant chunks (with filtering)
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 4, "score_threshold": 0.5}
    )

    docs = retriever.invoke(question)

    # 🚨 Guard: no docs
    if not docs:
        return "⚠️ Could not find relevant content in the document. Try rephrasing your question.", docs

    # 🧠 Handle summary-type questions
    if "what is the pdf about" in question.lower() or "summary" in question.lower():
        docs = docs[:6]

    # 🔍 Debug logs
    print(f"\n{'='*50}")
    print(f"Question: {question}")
    print(f"Docs retrieved: {len(docs)}")
    for i, doc in enumerate(docs):
        print(f"\nChunk {i+1} (page {doc.metadata.get('page', '?')}):")
        print(doc.page_content[:200])
    print(f"{'='*50}\n")

    # 🧾 Build context
    context = "\n\n".join(
        [d.page_content.strip() for d in docs if d.page_content]
    )

    # 🚨 Guard: empty context
    if not context.strip():
        return "⚠️ The retrieved content appears to be empty. Please try a different question.", docs

    # 🧠 Strict prompt
    prompt = f"""
You are a strict document-based assistant.

Rules:
- Answer ONLY using the provided context.
- If the answer is not clearly present in the context, say exactly:
  "The document does not contain enough information."
- Do NOT guess.
- Do NOT use external knowledge.
- If the question is general (like summary), summarize ONLY from the context.

Context:
\"\"\"
{context}
\"\"\"

Question: {question}

Answer:
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # 🔥 faster + more stable
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict document assistant. Only answer from context."
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
        answer = f"❌ Error generating response: {str(e)}"
        docs = []

    return answer, docs