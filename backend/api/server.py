"""
FastAPI server for the Voice Agent application.
Provides REST endpoints for the frontend to interact with the voice agent.
"""

import logging
import os
import secrets
from typing import Dict, Optional
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import jwt

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Voice Agent API",
    description="API for managing voice agent sessions and LiveKit integration",
    version="1.0.0"
)

# Configure CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# LiveKit configuration
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "your_livekit_api_key")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "your_livekit_api_secret")

# In-memory storage for active sessions (use Redis in production)
active_sessions: Dict[str, dict] = {}


class SessionRequest(BaseModel):
    """Request model for creating a new voice agent session."""
    participant_name: Optional[str] = None
    room_name: Optional[str] = None
    voice_instructions: Optional[str] = None


class SessionResponse(BaseModel):
    """Response model for session creation."""
    session_id: str
    room_name: str
    token: str
    livekit_url: str
    participant_identity: str


class SessionStatus(BaseModel):
    """Model for session status information."""
    session_id: str
    room_name: str
    participant_identity: str
    status: str
    created_at: datetime
    is_active: bool


def generate_session_id() -> str:
    """Generate a unique session ID."""
    return secrets.token_urlsafe(16)


def generate_room_name() -> str:
    """Generate a unique room name."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = secrets.token_hex(4)
    return f"voice_agent_{timestamp}_{random_suffix}"


def create_livekit_token(room_name: str, participant_identity: str) -> str:
    """
    Create a LiveKit JWT token for the participant.
    
    Args:
        room_name: Name of the LiveKit room
        participant_identity: Unique identifier for the participant
        
    Returns:
        JWT token string
    """
    try:
        # Token payload
        payload = {
            "iss": LIVEKIT_API_KEY,
            "sub": participant_identity,
            "aud": "livekit",
            "exp": int((datetime.utcnow() + timedelta(hours=24)).timestamp()),
            "room": room_name,
            "permissions": {
                "canPublish": True,
                "canSubscribe": True,
                "canPublishData": True,
            }
        }
        
        # Generate token
        token = jwt.encode(payload, LIVEKIT_API_SECRET, algorithm="HS256")
        return token
        
    except Exception as e:
        logger.error(f"Failed to create LiveKit token: {e}")
        raise HTTPException(status_code=500, detail="Failed to create session token")


@app.get("/")
async def root():
    """Root endpoint with basic information."""
    return {
        "message": "Voice Agent API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "livekit_url": LIVEKIT_URL
    }


@app.post("/sessions", response_model=SessionResponse)
async def create_session(request: SessionRequest):
    """
    Create a new voice agent session.
    
    Args:
        request: Session creation request
        
    Returns:
        Session information including LiveKit token
    """
    try:
        # Generate session details
        session_id = generate_session_id()
        room_name = request.room_name or generate_room_name()
        participant_identity = request.participant_name or f"user_{secrets.token_hex(4)}"
        
        # Create LiveKit token
        token = create_livekit_token(room_name, participant_identity)
        
        # Store session information
        session_info = {
            "session_id": session_id,
            "room_name": room_name,
            "participant_identity": participant_identity,
            "voice_instructions": request.voice_instructions,
            "created_at": datetime.utcnow(),
            "status": "active",
            "is_active": True
        }
        
        active_sessions[session_id] = session_info
        
        logger.info(f"Created session {session_id} for room {room_name}")
        
        return SessionResponse(
            session_id=session_id,
            room_name=room_name,
            token=token,
            livekit_url=LIVEKIT_URL,
            participant_identity=participant_identity
        )
        
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        raise HTTPException(status_code=500, detail="Failed to create session")


@app.get("/sessions/{session_id}", response_model=SessionStatus)
async def get_session(session_id: str):
    """
    Get session status information.
    
    Args:
        session_id: Unique session identifier
        
    Returns:
        Session status information
    """
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = active_sessions[session_id]
    
    return SessionStatus(
        session_id=session_id,
        room_name=session["room_name"],
        participant_identity=session["participant_identity"],
        status=session["status"],
        created_at=session["created_at"],
        is_active=session["is_active"]
    )


@app.delete("/sessions/{session_id}")
async def end_session(session_id: str):
    """
    End a voice agent session.
    
    Args:
        session_id: Unique session identifier
    """
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Mark session as inactive
    active_sessions[session_id]["status"] = "ended"
    active_sessions[session_id]["is_active"] = False
    
    logger.info(f"Ended session {session_id}")
    
    return {"message": "Session ended successfully"}


@app.get("/sessions")
async def list_sessions():
    """List all active sessions."""
    return {
        "sessions": [
            {
                "session_id": session_id,
                "room_name": session["room_name"],
                "participant_identity": session["participant_identity"],
                "status": session["status"],
                "created_at": session["created_at"],
                "is_active": session["is_active"]
            }
            for session_id, session in active_sessions.items()
        ]
    }


@app.get("/config")
async def get_config():
    """Get frontend configuration."""
    return {
        "livekit_url": LIVEKIT_URL,
        "available_voices": {
            "rachel": "21m00Tcm4TlvDq8ikWAM",
            "roger": "CwhRBWXzGAHq8TQ4Fs17",
            "sarah": "EXAVITQu4vr4xnSDxMaL",
            "laura": "FGY2WhTYpPnrIDTdsKH5",
            "george": "JBFqnCBsd6RMkjVDRZzb"
        },
        "available_models": [
            "gpt-4o",
            "gpt-4o-mini"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    logger.info(f"Starting FastAPI server on {host}:{port}")
    
    uvicorn.run(
        "api.server:app",
        host=host,
        port=port,
        reload=True,
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )