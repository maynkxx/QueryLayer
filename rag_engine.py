# rag_engine.py

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

# API key — works for both local and Streamlit Cloud
def get_api_key():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("GROQ_API_KEY")
        except Exception:
            pass
    if not key:
        raise ValueError("❌ GROQ_API_KEY not found in .env or Streamlit secrets.")
    return key

client = Groq(api_key=get_api_key())

# Cache embedding model globally
_embeddings = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        print("🔄 Loading embedding model...")
        _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        print("✅ Embedding model loaded")
    return _embeddings


def build_vectorstore(pdf_path):
    try:
        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()
        print(f"✅ Loaded {len(docs)} pages")

        if not docs:
            print("❌ No pages loaded from PDF!")
            return None

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )
        chunks = splitter.split_documents(docs)
        print(f"✅ Created {len(chunks)} chunks")
        print(f"📄 Sample chunk:\n{chunks[0].page_content[:300]}")

        vectorstore = FAISS.from_documents(chunks, get_embeddings())
        print("✅ Vector store created")

        # Sanity check — confirm retrieval works right after building
        test_results = vectorstore.similarity_search("document", k=1)
        print(f"✅ Sanity check passed — got {len(test_results)} result(s)")

        return vectorstore

    except Exception as e:
        print(f"❌ Error in build_vectorstore: {str(e)}")
        return None


def get_answer(question, vectorstore):
    # Simple similarity search — NO score threshold (that was the bug)
    docs = vectorstore.similarity_search(question, k=4)

    print(f"\n{'='*50}")
    print(f"Question: {question}")
    print(f"Docs retrieved: {len(docs)}")
    for i, doc in enumerate(docs):
        print(f"\nChunk {i+1} (page {doc.metadata.get('page', '?')}):")
        print(doc.page_content[:200])
    print(f"{'='*50}\n")

    if not docs:
        return "⚠️ Could not retrieve any content from the document.", []

    context = "\n\n".join([d.page_content.strip() for d in docs])

    # Detect summary-type questions and fetch more chunks
    summary_keywords = ["summary", "about", "what is this", "overview", "describe"]
    is_summary = any(kw in question.lower() for kw in summary_keywords)

    if is_summary:
        extra_docs = vectorstore.similarity_search(question, k=8)
        context = "\n\n".join([d.page_content.strip() for d in extra_docs])

    prompt = f"""You are a helpful document assistant.

Instructions:
- Answer using ONLY the context provided below.
- If the question asks for a summary or overview, summarize the key points from the context.
- If the answer is not in the context, say: "The document does not contain enough information to answer this."
- Be clear, structured, and concise.

Context:
\"\"\"
{context}
\"\"\"

Question: {question}

Answer:"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions strictly from the provided document context."
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