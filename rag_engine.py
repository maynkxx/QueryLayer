# rag_engine.py

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("❌ GROQ_API_KEY not found. Please check your .env file.")

client = Groq(api_key=api_key)

# Cache the embedding model so it's not reloaded every time
_embeddings = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        print("🔄 Loading embedding model...")
        _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        print("✅ Embedding model loaded")
    return _embeddings


def build_vectorstore(pdf_path):
    # Load PDF
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    print(f"✅ Loaded {len(docs)} pages")

    # Chunk the text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)
    print(f"✅ Created {len(chunks)} chunks")

    # Build vector store
    vectorstore = FAISS.from_documents(chunks, get_embeddings())
    print("✅ Vector store created successfully")

    return vectorstore


def get_answer(question, vectorstore):
    # Retrieve relevant chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(question)

    # ── DEBUG (check your terminal) ──────────────────────────
    print(f"\n{'='*50}")
    print(f"Question: {question}")
    print(f"Docs retrieved: {len(docs)}")
    for i, doc in enumerate(docs):
        print(f"\nChunk {i+1} (page {doc.metadata.get('page', '?')}):")
        print(doc.page_content[:200])
    print(f"{'='*50}\n")
    # ─────────────────────────────────────────────────────────

    # Guard: if nothing retrieved
    if not docs:
        return "⚠️ Could not find relevant content in the document. Try rephrasing your question.", docs

    context = "\n\n".join([d.page_content for d in docs])

    # Guard: if context is empty
    if not context.strip():
        return "⚠️ The retrieved content appears to be empty. Please try a different question.", docs

    prompt = f"""You are a helpful assistant that answers questions based on the provided document context.

Instructions:
- Answer based ONLY on the context provided below.
- If the context contains relevant information, use it to give a detailed answer.
- If the context does not contain enough information, say "The document does not contain enough information to answer this question."
- Do NOT say "I don't know" if there is relevant information in the context.
- Be clear, helpful, and concise.

Context from document:
\"\"\"
{context}
\"\"\"

Question: {question}

Answer:"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful document assistant. Always try to answer from the provided context."
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