from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.rag_service import RAGService
import os
import time


API_KEY = os.getenv("API_KEY")
RAG_SERVICE = None

# Initialize fast API app
app = FastAPI(
    title="RAG Service API",
    description="API for the Retrieval-Augmented Generation (RAG) service using LangChain and Llama 3.",
    version="1.0.0"
)


# Model - query request body
class QueryRequest(BaseModel):
    question: str


# Model - response body
class QueryResponse(BaseModel):
    answer: str
    source_model: str = os.getenv("LLM_MODEL", "llama3")


# startup events
@app.on_event("startup")
async def startup_event():

    global RAG_SERVICE
    RAG_SERVICE = RAGService()

    MAX_RETRIES = 5
    RETRY_DELAY = 10

    # Check API key
    if not API_KEY:
        print("LLM API Key not set.")
        return

    # Load embedding model
    try:
        RAG_SERVICE.load_embedding_model()
        print("Embedding model loaded successfully.")
    except Exception as e:
        print(f"Failed to load embedding model. Error: {str(e)}")
        return

    # Load vector store
    try:
        load_status = RAG_SERVICE.init_vectorstore()
        print(f"Vector Store Load Status: {load_status}")
    except Exception as e:
        print(f"Failed to initialize vector store. Error: {str(e)}")
        load_status = "failed"

    # Load llm chain with retries
    if "successfully" in load_status.lower():
        for i in range(MAX_RETRIES):
            try:
                RAG_SERVICE.init_qa_chain()
                print("QA Chain initialized successfully.")
                break
            except Exception as e:
                # Check if this was the last attempt
                if i < MAX_RETRIES - 1:
                    print(f"LLM connection failed (Attempt {i+1}/{MAX_RETRIES}). Retrying in {RETRY_DELAY}s.")
                    print(f"Error detail: {str(e)}")
                    time.sleep(RETRY_DELAY)
                else:
                    # Last attempt failed: log a FATAL error and break
                    print(f"FATAL: Failed to initialize QA Chain after {MAX_RETRIES} attempts.")
                    print(f"The service will start, but the /query endpoint will return a 503 error until the LLM is ready. Error: {str(e)}")
                    # Do not re-raise the exception; let FastAPI start gracefully


# API ENDPOINTS 

# 1. Health check endpoint
@app.get("/health", summary="Health Check")
def health_check():
    if RAG_SERVICE.vectorstore is None:
        raise HTTPException(status_code=503,
                            detail="Vector store not initialized."
                            )
    print(str(RAG_SERVICE.vectorstore))
    return {"status": "We are live and on air!."}


# 2. RAG Query Endpoint
@app.post("/query", response_model=QueryResponse, summary="Query RAG Service")
async def query_rag_endpoint(request: QueryRequest):
    if RAG_SERVICE.vectorstore is None:
        raise HTTPException(
            status_code=503,
            detail="Vector store not initialized."
        )
    
    answer = RAG_SERVICE.query_rag(request.question)
    return QueryResponse(answer=answer)
