# retriver/service.py

import os
import asyncio
import time
from typing import Dict, Any, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware 

# CORRECTED IMPORT: Use a relative import to find the logic file and the correct class name.
from .logic import RetrieverService

# --- Configuration & Initialization ---
load_dotenv()

VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH")
if not VECTOR_DB_PATH:
    raise ValueError("VECTOR_DB_PATH not set in the .env file.")

MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "16"))
BATCH_TIMEOUT_S = float(os.getenv("BATCH_TIMEOUT_S", "0.02"))

# --- Global Objects ---
retriever_service = RetrieverService(vector_db_path=VECTOR_DB_PATH)
app = FastAPI(
    title="Data Retriever API",
    description="An API for retrieving documents using JinaV3 embeddings with in-flight batching.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
) 

request_queue = asyncio.Queue()

# --- Pydantic Models for API Contract ---
class RetrievalRequest(BaseModel):
    query: str = Field(..., description="User's question.", example="জন্ম নিবন্ধন সনদের জন্য আবেদন করার নিয়ম কি?")
    k: int = Field(default=3, gt=0, le=20, description="Number of documents to retrieve.")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters.", example={"category": "ভূমি সেবা"})

class RetrievedPassage(BaseModel):
    text: str
    url: str
    score: float
    metadata: Dict[str, Any]

class RetrievalResponse(BaseModel):
    query: str
    retrieved_passages: List[RetrievedPassage]

# --- In-Flight Batching Worker ---
async def batch_retrieval_worker():
    while True:
        first_request_item = await request_queue.get()
        batch = [first_request_item]
        start_time = time.monotonic()
        
        while (len(batch) < MAX_BATCH_SIZE and (time.monotonic() - start_time) < BATCH_TIMEOUT_S and not request_queue.empty()):
            try:
                batch.append(request_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        
        batch_start_time = time.time()
        requests_data = [item["data"] for item in batch]
        futures = [item["future"] for item in batch]
        
        queries = [req.query for req in requests_data]
        k_values = [req.k for req in requests_data]
        filters_list = [req.filters for req in requests_data]

        try:
            embeddings = retriever_service.embedding_model.embed_queries(queries)
            results_batch = retriever_service.retrieve_batch(queries, embeddings, k_values, filters_list)

            for i, future in enumerate(futures):
                response_data = RetrievalResponse(query=queries[i], retrieved_passages=results_batch[i])
                future.set_result(response_data)

            elapsed_ms = (time.time() - batch_start_time) * 1000
            print(f"✅ Processed batch of {len(batch)} items in {elapsed_ms:.2f} ms.")

        except Exception as e:
            print(f"❌ Error processing batch: {e}")
            for future in futures:
                future.set_exception(e)

@app.on_event("startup")
async def startup_event():
    print("Starting background batch retrieval worker...")
    asyncio.create_task(batch_retrieval_worker())

@app.post("/retrieve", response_model=RetrievalResponse)
async def retrieve_documents(request: RetrievalRequest = Body(...)):
    future = asyncio.get_running_loop().create_future()
    await request_queue.put({"data": request, "future": future})
    try:
        return await future
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

# Add this to allow running the file directly for testing
if __name__ == "__main__":
    print("--- Starting Data Retriever API Server ---")
    uvicorn.run(
        "service:app",  # When running directly, uvicorn looks in the current file
        host="0.0.0.0",
        port=7000,
        reload=True
    )