import os
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from RagAgent import process_and_store_pdf, run_query, PERSIST_DIR

app = FastAPI(
    title="LangGraph RAG Agent API",
    description="API server powering PDF processing and RAG query agent",
    version="1.0.0"
)

# Enable CORS for Next.js on Vercel or local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to specific Vercel URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_UPLOAD_DIR = "/tmp/temp_uploads" if os.getenv("VERCEL") else "./temp_uploads"

try:
    os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
    os.makedirs(PERSIST_DIR, exist_ok=True)
except Exception as e:
    print(f"Warning creating directories: {e}")

class QueryRequest(BaseModel):
    session_id: str
    query: str

class QueryResponse(BaseModel):
    session_id: str
    answer: str

class UploadResponse(BaseModel):
    session_id: str
    filename: str
    total_chunks: int

@app.get("/")
def root():
    return {"status": "ok", "service": "LangGraph RAG Agent API"}

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "LangGraph RAG Agent API"}

@app.post("/api/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    session_id = str(uuid.uuid4())
    temp_file_path = os.path.join(TEMP_UPLOAD_DIR, f"{session_id}_{file.filename}")

    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        chunks_count = process_and_store_pdf(temp_file_path, session_id)

        return UploadResponse(
            session_id=session_id,
            filename=file.filename,
            total_chunks=chunks_count
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/api/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    if not request.session_id or not request.query:
        raise HTTPException(status_code=400, detail="session_id and query are required.")

    try:
        answer = run_query(request.session_id, request.query)
        return QueryResponse(
            session_id=request.session_id,
            answer=answer
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing agent query: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
