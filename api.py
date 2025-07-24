# RAG-LLama-JinaV3/api_service.py

import asyncio
import json
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict

# Corrected the import to use 'final_chatservice' as per previous steps
# If your file is named 'chatservice.py', use that instead.
from chatservice import ProactiveChatService
from fastapi.middleware.cors import CORSMiddleware

# --- API Setup ---
app = FastAPI(
    title="Multi-User RAG Chat API",
    description="A production-grade API for the Proactive Chat Service with multi-user session management.",
    version="1.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
) 

# --- Session Management ---
# This dictionary will store active chat sessions, mapping a user_id to their own ProactiveChatService instance.
# WARNING: This is an in-memory store. For large-scale production, consider using a distributed cache like Redis
# to store session objects or conversation histories, which would allow the service to be stateless and scale horizontally.
chat_sessions: Dict[str, ProactiveChatService] = {}
sessions_lock = asyncio.Lock()  # A lock to prevent race conditions when creating/deleting sessions.


# --- Pydantic Models for Request Bodies ---
class ChatRequest(BaseModel):
    """Defines the structure for a chat request."""
    user_id: str
    query: str

class ClearSessionRequest(BaseModel):
    """Defines the structure for a session clearing request."""
    user_id: str


# --- Core Session Logic ---
async def get_or_create_session(user_id: str) -> ProactiveChatService:
    """
    Retrieves an existing chat session for a user or creates a new one if it doesn't exist.
    This function is thread-safe.
    """
    async with sessions_lock:
        if user_id not in chat_sessions:
            print(f"-> Creating new chat session for user_id: {user_id}")
            # Each user gets their own instance with a history length of 10 turns.
            chat_sessions[user_id] = ProactiveChatService(history_length=10)
        else:
            print(f"-> Found existing session for user_id: {user_id}")
        return chat_sessions[user_id]


# --- API Endpoints ---
@app.get("/health", tags=["Monitoring"])
async def health_check():
    """A simple endpoint to confirm the service is running and check session count."""
    return {"status": "ok", "active_sessions": len(chat_sessions)}


@app.post("/chat/clear_session", tags=["Session Management"])
async def clear_session(request: ClearSessionRequest):
    """
    Clears the conversation history for a specific user_id, effectively resetting their session.
    """
    user_id = request.user_id
    message = ""
    
    async with sessions_lock:
        if user_id in chat_sessions:
            # The session exists, so we delete it from our in-memory dictionary.
            del chat_sessions[user_id]
            message = f"Session for user_id '{user_id}' has been cleared."
            print(f"-> Cleared session for user_id: {user_id}")
        else:
            # The session didn't exist, which is not an error.
            message = f"No active session found for user_id '{user_id}'. Nothing to clear."
            print(f"-> Attempted to clear non-existent session for user_id: {user_id}")
            
    return {"status": "success", "message": message}


@app.post("/chat/stream", tags=["Chat"])
async def stream_chat(chat_request: ChatRequest):
    """
    The main chat endpoint. It receives a user query, manages the user's session,
    and streams the response back in real-time as newline-delimited JSON objects.
    """
    user_id = chat_request.user_id
    query = chat_request.query

    session = await get_or_create_session(user_id)

    async def response_generator():
        """
        An async generator that yields events from the chat service, formatted for streaming.
        """
        try:
            for event in session.chat(query):
                json_event = json.dumps(event, ensure_ascii=False)
                yield f"{json_event}\n"
                await asyncio.sleep(0.01)
        except Exception as e:
            print(f"An error occurred during generation for user {user_id}: {e}")
            error_event = {
                "type": "error",
                "content": "An internal error occurred. Please try again later."
            }
            yield f"{json.dumps(error_event)}\n"

    return StreamingResponse(response_generator(), media_type="application/x-ndjson")


if __name__ == "__main__":
    # To run this service:
    # 1. Ensure uvicorn and fastapi are installed: pip install uvicorn fastapi
    # 2. In your terminal, run the command:
    #    uvicorn api_service:app --host 0.0.0.0 --port 9000 --reload
    print("Starting Uvicorn server on http://0.0.0.0:9000")
    uvicorn.run("api_service:app", host="0.0.0.0", port=9000, reload=True)