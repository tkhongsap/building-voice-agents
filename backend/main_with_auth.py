"""
Enhanced Voice Agent Backend with Enterprise Authentication

This example shows how to integrate the new authentication system with the existing
voice agent platform for production enterprise deployments.
"""

import logging
import os
from typing import Dict, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from dotenv import load_dotenv

# Import the new authentication system
from src.auth import (
    initialize_auth_system,
    cleanup_auth_system,
    get_auth_manager,
    get_rbac_manager,
    get_tenant_manager,
    get_admin_interface
)
from src.auth.fastapi_integration import (
    AuthenticationMiddleware,
    require_authentication,
    require_permission,
    get_current_user,
    get_tenant_context,
    setup_auth_routes,
    setup_admin_routes
)
from src.auth.models import User, PermissionType
from src.auth.exceptions import AuthenticationError, PermissionDeniedError

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Voice Agent Platform - Enterprise Edition",
    description="Enterprise Voice Agent Platform with comprehensive authentication and authorization",
    version="2.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# LiveKit configuration
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")

# Pydantic models
class VoiceAgentRequest(BaseModel):
    """Request model for creating a voice agent session."""
    participant_name: Optional[str] = None
    room_name: Optional[str] = None
    voice_instructions: Optional[str] = None
    agent_config: Optional[Dict] = None

class VoiceAgentResponse(BaseModel):
    """Response model for voice agent session."""
    session_id: str
    room_name: str
    token: str
    livekit_url: str
    participant_identity: str
    agent_id: Optional[str] = None

class AgentConfig(BaseModel):
    """Configuration for voice agents."""
    name: str
    description: str
    model: str = "gpt-4o"
    voice_id: str = "rachel"
    instructions: str
    tools: Optional[list] = None

# In-memory storage (use database in production)
voice_agents: Dict[str, dict] = {}
active_sessions: Dict[str, dict] = {}

@app.on_event("startup")
async def startup():
    """Initialize the application."""
    logger.info("Starting Voice Agent Platform with Enterprise Authentication")
    
    # Initialize authentication system
    await initialize_auth_system()
    
    # Add authentication middleware
    auth_manager = get_auth_manager()
    app.add_middleware(AuthenticationMiddleware, auth_manager=auth_manager)
    
    # Setup authentication routes
    setup_auth_routes(app)
    setup_admin_routes(app)
    
    # Create default admin user if not exists
    await create_default_admin()
    
    logger.info("Voice Agent Platform started successfully")

@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    logger.info("Shutting down Voice Agent Platform")
    await cleanup_auth_system()

async def create_default_admin():
    """Create default admin user if none exists."""
    try:
        auth_manager = get_auth_manager()
        
        # Check if admin already exists
        admin_user = await auth_manager.get_user_by_email("admin@voiceagent.local")
        if not admin_user:
            # Create default admin
            admin_user = await auth_manager.create_user(
                email="admin@voiceagent.local",
                username="admin",
                full_name="System Administrator",
                password=os.getenv("DEFAULT_ADMIN_PASSWORD", "VoiceAgent2024!"),
                roles=["Super Administrator"]
            )
            logger.info("Created default admin user: admin@voiceagent.local")
    
    except Exception as e:
        logger.error(f"Failed to create default admin user: {e}")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with system information."""
    return {
        "message": "Voice Agent Platform - Enterprise Edition",
        "version": "2.0.0",
        "status": "running",
        "features": [
            "Enterprise Authentication",
            "OAuth/SAML SSO",
            "Role-Based Access Control",
            "Multi-Tenant Support",
            "API Key Management",
            "Audit Logging"
        ]
    }

# Health check
@app.get("/health")
async def health_check(user: User = Depends(get_current_user)):
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "user": user.email if user else "anonymous",
        "livekit_url": LIVEKIT_URL
    }

# Voice Agent Management Endpoints

@app.post("/agents", response_model=dict)
async def create_voice_agent(
    agent_config: AgentConfig,
    user: User = Depends(require_permission(PermissionType.CREATE_AGENT))
):
    """Create a new voice agent configuration."""
    try:
        tenant_manager = get_tenant_manager()
        
        # Check tenant quota
        await tenant_manager.consume_quota(user.tenant_id, 'agents')
        
        # Create agent
        agent_id = f"agent_{len(voice_agents) + 1}"
        agent_data = {
            "id": agent_id,
            "name": agent_config.name,
            "description": agent_config.description,
            "model": agent_config.model,
            "voice_id": agent_config.voice_id,
            "instructions": agent_config.instructions,
            "tools": agent_config.tools or [],
            "owner_id": user.id,
            "tenant_id": user.tenant_id,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        voice_agents[agent_id] = agent_data
        
        # Log creation
        auth_manager = get_auth_manager()
        await auth_manager.audit_logger.log_event({
            "event_type": "agent.created",
            "user_id": user.id,
            "tenant_id": user.tenant_id,
            "resource_id": agent_id,
            "details": {"agent_name": agent_config.name}
        })
        
        logger.info(f"Created voice agent {agent_id} for user {user.email}")
        
        return {
            "message": "Voice agent created successfully",
            "agent": agent_data
        }
        
    except Exception as e:
        # Release quota on error
        await tenant_manager.release_quota(user.tenant_id, 'agents')
        logger.error(f"Failed to create voice agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents")
async def list_voice_agents(
    user: User = Depends(require_permission(PermissionType.READ_AGENT))
):
    """List voice agents accessible to the user."""
    tenant_manager = get_tenant_manager()
    
    # Filter agents by tenant
    user_agents = []
    for agent_id, agent in voice_agents.items():
        if tenant_manager.validate_tenant_access(user, agent.get('tenant_id')):
            user_agents.append(agent)
    
    return {
        "agents": user_agents,
        "total": len(user_agents)
    }

@app.get("/agents/{agent_id}")
async def get_voice_agent(
    agent_id: str,
    user: User = Depends(require_permission(PermissionType.READ_AGENT))
):
    """Get specific voice agent."""
    agent = voice_agents.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    # Check tenant access
    tenant_manager = get_tenant_manager()
    if not tenant_manager.validate_tenant_access(user, agent.get('tenant_id')):
        raise HTTPException(status_code=403, detail="Access denied")
    
    return {"agent": agent}

@app.delete("/agents/{agent_id}")
async def delete_voice_agent(
    agent_id: str,
    user: User = Depends(require_permission(PermissionType.DELETE_AGENT))
):
    """Delete voice agent."""
    agent = voice_agents.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    # Check tenant access
    tenant_manager = get_tenant_manager()
    if not tenant_manager.validate_tenant_access(user, agent.get('tenant_id')):
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Delete agent
    del voice_agents[agent_id]
    
    # Release quota
    await tenant_manager.release_quota(user.tenant_id, 'agents')
    
    logger.info(f"Deleted voice agent {agent_id} by user {user.email}")
    
    return {"message": "Voice agent deleted successfully"}

# Session Management Endpoints

@app.post("/sessions", response_model=VoiceAgentResponse)
async def create_voice_session(
    request: VoiceAgentRequest,
    user: User = Depends(require_permission(PermissionType.CREATE_SESSION))
):
    """Create a new voice agent session with authentication."""
    try:
        from .api.server import generate_session_id, generate_room_name, create_livekit_token
        
        tenant_manager = get_tenant_manager()
        
        # Check tenant quota
        await tenant_manager.consume_quota(user.tenant_id, 'sessions')
        
        # Generate session details
        session_id = generate_session_id()
        room_name = request.room_name or generate_room_name()
        participant_identity = request.participant_name or f"{user.username}_{session_id}"
        
        # Create LiveKit token
        token = create_livekit_token(room_name, participant_identity)
        
        # Store session information
        session_info = {
            "session_id": session_id,
            "room_name": room_name,
            "participant_identity": participant_identity,
            "voice_instructions": request.voice_instructions,
            "agent_config": request.agent_config,
            "user_id": user.id,
            "tenant_id": user.tenant_id,
            "created_at": datetime.utcnow(),
            "status": "active",
            "is_active": True
        }
        
        active_sessions[session_id] = session_info
        
        # Log session creation
        auth_manager = get_auth_manager()
        await auth_manager.audit_logger.log_event({
            "event_type": "session.created",
            "user_id": user.id,
            "tenant_id": user.tenant_id,
            "resource_id": session_id,
            "details": {"room_name": room_name}
        })
        
        logger.info(f"Created voice session {session_id} for user {user.email}")
        
        return VoiceAgentResponse(
            session_id=session_id,
            room_name=room_name,
            token=token,
            livekit_url=LIVEKIT_URL,
            participant_identity=participant_identity,
            agent_id=request.agent_config.get('agent_id') if request.agent_config else None
        )
        
    except Exception as e:
        logger.error(f"Failed to create voice session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}")
async def get_voice_session(
    session_id: str,
    user: User = Depends(require_permission(PermissionType.READ_SESSION))
):
    """Get voice session information."""
    session = active_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Check if user can access this session
    if session['user_id'] != user.id and not await get_rbac_manager().check_permission(
        user, PermissionType.MANAGE_ALL_SESSIONS
    ):
        raise HTTPException(status_code=403, detail="Access denied")
    
    return {"session": session}

@app.get("/sessions")
async def list_voice_sessions(
    user: User = Depends(require_permission(PermissionType.READ_SESSION))
):
    """List voice sessions accessible to the user."""
    rbac_manager = get_rbac_manager()
    
    # Check if user can see all sessions
    can_see_all = await rbac_manager.check_permission(user, PermissionType.MANAGE_ALL_SESSIONS)
    
    user_sessions = []
    for session_id, session in active_sessions.items():
        if can_see_all or session['user_id'] == user.id:
            user_sessions.append(session)
    
    return {
        "sessions": user_sessions,
        "total": len(user_sessions)
    }

@app.delete("/sessions/{session_id}")
async def end_voice_session(
    session_id: str,
    user: User = Depends(require_permission(PermissionType.UPDATE_SESSION))
):
    """End a voice session."""
    session = active_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Check if user can manage this session
    if session['user_id'] != user.id and not await get_rbac_manager().check_permission(
        user, PermissionType.MANAGE_ALL_SESSIONS
    ):
        raise HTTPException(status_code=403, detail="Access denied")
    
    # End session
    session['status'] = "ended"
    session['is_active'] = False
    session['ended_at'] = datetime.utcnow()
    
    # Release quota
    tenant_manager = get_tenant_manager()
    await tenant_manager.release_quota(session['tenant_id'], 'sessions')
    
    logger.info(f"Ended voice session {session_id}")
    
    return {"message": "Session ended successfully"}

# Configuration endpoint
@app.get("/config")
async def get_config(
    user: User = Depends(require_authentication),
    tenant_context = Depends(get_tenant_context)
):
    """Get user and tenant-specific configuration."""
    config = {
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
        ],
        "user": {
            "id": user.id,
            "email": user.email,
            "tenant_id": user.tenant_id,
            "roles": [role.name for role in user.roles]
        }
    }
    
    # Add tenant-specific configuration
    if tenant_context:
        config["tenant"] = {
            "id": tenant_context.tenant_id,
            "name": tenant_context.tenant.name,
            "features": list(tenant_context.tenant.features),
            "settings": tenant_context.tenant.settings
        }
    
    return config

# Error handlers
@app.exception_handler(AuthenticationError)
async def authentication_error_handler(request: Request, exc: AuthenticationError):
    return JSONResponse(
        status_code=401,
        content={"detail": str(exc), "type": "authentication_error"}
    )

@app.exception_handler(PermissionDeniedError)
async def permission_denied_handler(request: Request, exc: PermissionDeniedError):
    return JSONResponse(
        status_code=403,
        content={"detail": str(exc), "type": "permission_denied"}
    )

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    logger.info(f"Starting Voice Agent Platform with Enterprise Authentication on {host}:{port}")
    
    uvicorn.run(
        "main_with_auth:app",
        host=host,
        port=port,
        reload=True,
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )