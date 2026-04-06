
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()


# 🔹 Function 1: Build vector DB
def build_vectorstore(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    print(f"✅ Loaded {len(docs)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)
    print(f"✅ Created {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    print("✅ Embedding model loaded")

    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("✅ Vector store created successfully")

    return vectorstore


# 🔹 Function 2: Get answer using Groq
def get_answer(question, vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(question)

    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""Use only the context below to answer.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}
Answer:"""

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    answer = response.choices[0].message.content

    return answer, docs