"""
QueryLayer — FastAPI Backend
Exposes the RAG pipeline as a production-ready REST API.

Run with:
    uvicorn fastapi_app:app --reload

Docs available at:
    http://localhost:8000/docs      (Swagger UI)
    http://localhost:8000/redoc     (ReDoc)
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import tempfile
import shutil
import uuid
import os

from rag_engine import (
    build_vectorstore_from_file,
    get_answer,
    generate_summary,
    generate_suggested_questions,
)

# ── App Init ───

app = FastAPI(
    title="QueryLayer API",
    description=(
        "RAG-powered document Q&A API.\n\n"
        "**Workflow:**\n"
        "1. `POST /upload` — Upload a PDF, receive a `session_id`\n"
        "2. `POST /chat` — Ask questions using your `session_id`\n"
        "3. `GET /summary/{session_id}` — Fetch the document summary\n"
        "4. `DELETE /session/{session_id}` — Clear the session when done"
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-Memory Session Store ────
# Stores vectorstore + chat history per session.
# Resets on server restart — intentional for a demo-scale project.
# Structure: { session_id: { "vectorstore": ..., "chat_history": [...], "filename": ... } }

sessions: dict = {}


# ── Request / Response Models ───

class ChatRequest(BaseModel):
    session_id: str
    question: str

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "a1b2c3d4-...",
                "question": "What is the main topic of this document?"
            }
        }


class ChatResponse(BaseModel):
    session_id: str
    question: str
    answer: str
    sources: list[dict]


class UploadResponse(BaseModel):
    session_id: str
    filename: str
    summary: str
    suggested_questions: list[str]
    message: str


class SummaryResponse(BaseModel):
    session_id: str
    filename: str
    summary: str


class SessionDeleteResponse(BaseModel):
    session_id: str
    message: str


class HealthResponse(BaseModel):
    status: str
    message: str
    active_sessions: int


# ── Helper ───

def get_session_or_404(session_id: str) -> dict:
    """Fetch a session by ID or raise a 404 with a clear message."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found. Please upload a PDF first via POST /upload."
        )
    return session


def serialize_sources(docs: list) -> list[dict]:
    """
    Convert LangChain Document objects to JSON-serializable dicts.
    Extracts page number, source filename, and a content preview.
    """
    result = []
    for doc in docs:
        result.append({
            "page": doc.metadata.get("page", "?"),
            "source": os.path.basename(doc.metadata.get("source", "unknown")),
            "preview": doc.page_content[:300].strip() + "..."
        })
    return result


# ── Routes ───

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Check API status"
)
def health_check():
    """
    Returns API status and number of active sessions.
    Use this to verify the server is running before making other calls.
    """
    return HealthResponse(
        status="ok",
        message="QueryLayer API is running.",
        active_sessions=len(sessions)
    )


@app.post(
    "/upload",
    response_model=UploadResponse,
    tags=["Document"],
    summary="Upload a PDF and create a session"
)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF document to start a Q&A session.

    **What happens internally:**
    - PDF is parsed using PyMuPDF
    - Text is chunked (size=1000, overlap=150)
    - Chunks are embedded using `BAAI/bge-base-en-v1.5`
    - Embeddings are stored in an in-memory FAISS index
    - A summary and 3 suggested questions are auto-generated

    **Returns:** A `session_id` you must pass to `/chat` and `/summary`.
    """
    # ── Validate file type ────
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only PDF files are accepted."
        )

    # ── Save to temp file ───
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        # ── Build vectorstore ──────────────────────────────────────────────────
        vectorstore = build_vectorstore_from_file(tmp_path)

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process PDF: {str(e)}"
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # ── Create session ─────────────────────────────────────────────────────────
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "vectorstore": vectorstore,
        "chat_history": [],
        "filename": file.filename,
    }

    # ── Generate summary + suggested questions 
    summary = generate_summary(vectorstore)
    suggested_questions = generate_suggested_questions(vectorstore)

    return UploadResponse(
        session_id=session_id,
        filename=file.filename,
        summary=summary,
        suggested_questions=suggested_questions,
        message=f"'{file.filename}' processed successfully. Use the session_id to start chatting."
    )


@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["Chat"],
    summary="Ask a question about your uploaded document"
)
def chat(request: ChatRequest):
    """
    Ask a question grounded in your uploaded document.

    **What happens internally:**
    - Question is embedded and top-5 chunks retrieved via FAISS
    - Chunks are reranked using CrossEncoder (`ms-marco-MiniLM-L-6-v2`)
    - Top-3 reranked chunks + last 6 conversation turns sent to LLaMA 3.1 (Groq)
    - Answer is strictly grounded in document context — no hallucination

    **Multilingual:** Ask in any language — the API auto-detects and responds in kind.

    **Requires:** A valid `session_id` from `POST /upload`.
    """
    # ── Validate input 
    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    session = get_session_or_404(request.session_id)

    # ── Get answer 
    try:
        answer, docs = get_answer(
            question=request.question.strip(),
            vectorstore=session["vectorstore"],
            chat_history=session["chat_history"],
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating answer: {str(e)}"
        )

    # ── Update chat history in session ─────────────────────────────────────────
    session["chat_history"].append({"role": "user", "content": request.question.strip()})
    session["chat_history"].append({"role": "assistant", "content": answer})

    return ChatResponse(
        session_id=request.session_id,
        question=request.question.strip(),
        answer=answer,
        sources=serialize_sources(docs)
    )


@app.get(
    "/summary/{session_id}",
    response_model=SummaryResponse,
    tags=["Document"],
    summary="Get the auto-generated document summary"
)
def get_summary(session_id: str):
    """
    Fetch the auto-generated summary for an uploaded document.

    The summary is generated at upload time — this endpoint simply
    regenerates it on demand from the session's vectorstore.

    **Requires:** A valid `session_id` from `POST /upload`.
    """
    session = get_session_or_404(session_id)

    try:
        summary = generate_summary(session["vectorstore"])
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating summary: {str(e)}"
        )

    return SummaryResponse(
        session_id=session_id,
        filename=session["filename"],
        summary=summary
    )


@app.delete(
    "/session/{session_id}",
    response_model=SessionDeleteResponse,
    tags=["Session"],
    summary="Delete a session and free memory"
)
def delete_session(session_id: str):
    """
    Delete a session and release its vectorstore from memory.

    Call this when the user is done with a document to free up
    server memory. Sessions also reset automatically on server restart.

    **Requires:** A valid `session_id` from `POST /upload`.
    """
    get_session_or_404(session_id)
    del sessions[session_id]

    return SessionDeleteResponse(
        session_id=session_id,
        message="Session deleted successfully."
    )