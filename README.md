# RAG Document Chatbot

Chat with any PDF using AI. Upload a document and ask questions — 
the chatbot answers strictly from your document content.

## Live Demo
https://querylayer-f6oysgzwxrbybz8lezcrmr.streamlit.app/

## How it works
1. **Ingestion** — PDF is parsed, split into chunks, embedded using 
   sentence-transformers and stored in a FAISS vector index
2. **Retrieval** — User question is embedded and top-4 similar chunks 
   are retrieved via cosine similarity  
3. **Generation** — Retrieved chunks are passed as context to LLaMA 3.1 
   (via Groq API) which generates a grounded answer

## Tech Stack
- Python, Streamlit
- LangChain, FAISS, HuggingFace sentence-transformers
- Groq API (LLaMA 3.1 8B)

## Run locally
git clone https://github.com/maynkxx/rag-chatbot
cd rag-chatbot
pip install -r requirements.txt
echo "GROQ_API_KEY=your_key" > .env
streamlit run app.py
