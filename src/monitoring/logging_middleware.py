"""
Logging Middleware for FastAPI and LiveKit Integration

This module provides middleware components for automatic correlation ID tracking
and structured logging in web requests and LiveKit agent sessions.
"""

import time
import uuid
from typing import Callable, Optional, Dict, Any, Awaitable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging

from .structured_logging import (
    CorrelationIdManager,
    StructuredLogger,
    correlation_id_var,
    request_id_var,
    session_id_var,
    user_id_var,
    component_var
)
from .logging_config import get_logger


class CorrelationMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for automatic correlation ID management and request logging.
    """
    
    def __init__(
        self,
        app,
        correlation_header: str = "X-Correlation-ID",
        request_id_header: str = "X-Request-ID",
        user_id_header: str = "X-User-ID",
        generate_request_id: bool = True,
        log_requests: bool = True,
        log_responses: bool = True,
        log_body: bool = False,
        exclude_paths: Optional[list] = None
    ):
        super().__init__(app)
        self.correlation_header = correlation_header
        self.request_id_header = request_id_header
        self.user_id_header = user_id_header
        self.generate_request_id = generate_request_id
        self.log_requests = log_requests
        self.log_responses = log_responses
        self.log_body = log_body
        self.exclude_paths = exclude_paths or ["/health", "/metrics", "/docs", "/openapi.json"]
        self.logger = StructuredLogger(__name__, "api")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with correlation tracking."""
        # Skip excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)
        
        start_time = time.time()
        
        # Extract or generate correlation ID
        correlation_id = request.headers.get(self.correlation_header)
        if not correlation_id:
            correlation_id = CorrelationIdManager.generate_correlation_id()
        
        # Extract or generate request ID
        request_id = request.headers.get(self.request_id_header)
        if not request_id and self.generate_request_id:
            request_id = CorrelationIdManager.generate_request_id()
        
        # Extract user ID if available
        user_id = request.headers.get(self.user_id_header)
        
        # Set context
        CorrelationIdManager.set_correlation_id(correlation_id)
        if request_id:
            CorrelationIdManager.set_request_id(request_id)
        if user_id:
            CorrelationIdManager.set_user_id(user_id)
        CorrelationIdManager.set_component_context("api")
        
        # Log request
        if self.log_requests:
            await self._log_request(request, correlation_id, request_id, user_id)
        
        try:
            # Process request
            response = await call_next(request)
            
            # Add correlation headers to response
            response.headers[self.correlation_header] = correlation_id
            if request_id:
                response.headers[self.request_id_header] = request_id
            
            # Log response
            if self.log_responses:
                await self._log_response(
                    request, response, start_time, correlation_id, request_id
                )
            
            return response
            
        except Exception as e:
            # Log error
            duration = time.time() - start_time
            self.logger.error(
                f"Request failed: {request.method} {request.url.path}",
                extra_data={
                    "method": request.method,
                    "url": str(request.url),
                    "path": request.url.path,
                    "duration_ms": duration * 1000,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "status": "error"
                }
            )
            raise
    
    async def _log_request(
        self, 
        request: Request, 
        correlation_id: str,
        request_id: Optional[str],
        user_id: Optional[str]
    ):
        """Log incoming request."""
        extra_data = {
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "headers": dict(request.headers),
            "client_ip": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
            "content_type": request.headers.get("content-type"),
            "content_length": request.headers.get("content-length"),
            "correlation_id": correlation_id,
            "request_id": request_id,
            "user_id": user_id,
            "phase": "request"
        }
        
        # Log request body if enabled and appropriate
        if (self.log_body and 
            request.headers.get("content-type", "").startswith("application/json") and
            request.method in ["POST", "PUT", "PATCH"]):
            try:
                body = await request.body()
                if body:
                    extra_data["body_size"] = len(body)
                    # Don't log the actual body content for security reasons
            except Exception:
                pass
        
        self.logger.info(
            f"Incoming request: {request.method} {request.url.path}",
            extra_data=extra_data
        )
    
    async def _log_response(
        self,
        request: Request,
        response: Response,
        start_time: float,
        correlation_id: str,
        request_id: Optional[str]
    ):
        """Log outgoing response."""
        duration = time.time() - start_time
        
        extra_data = {
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration * 1000,
            "response_headers": dict(response.headers),
            "correlation_id": correlation_id,
            "request_id": request_id,
            "phase": "response"
        }
        
        # Determine log level based on status code
        if response.status_code >= 500:
            level = "error"
        elif response.status_code >= 400:
            level = "warning"
        else:
            level = "info"
        
        message = f"Response: {request.method} {request.url.path} -> {response.status_code}"
        
        getattr(self.logger, level)(message, extra_data=extra_data)


class LiveKitSessionMiddleware:
    """
    Middleware for LiveKit agent sessions to track correlation across voice pipeline.
    """
    
    def __init__(self):
        self.logger = StructuredLogger(__name__, "livekit")
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
    
    async def session_start(
        self,
        room_name: str,
        participant_identity: Optional[str] = None,
        session_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Start a new session with correlation tracking."""
        # Generate session ID and correlation ID
        session_id = CorrelationIdManager.generate_session_id()
        correlation_id = CorrelationIdManager.generate_correlation_id()
        
        # Set context
        CorrelationIdManager.set_session_id(session_id)
        CorrelationIdManager.set_correlation_id(correlation_id)
        if participant_identity:
            CorrelationIdManager.set_user_id(participant_identity)
        CorrelationIdManager.set_component_context("livekit")
        
        # Store session info
        session_info = {
            "session_id": session_id,
            "correlation_id": correlation_id,
            "room_name": room_name,
            "participant_identity": participant_identity,
            "start_time": time.time(),
            "metadata": session_metadata or {},
            "pipeline_stages": []
        }
        
        self.active_sessions[session_id] = session_info
        
        # Log session start
        self.logger.info(
            f"LiveKit session started: {room_name}",
            extra_data={
                "session_id": session_id,
                "room_name": room_name,
                "participant_identity": participant_identity,
                "phase": "session_start",
                "metadata": session_metadata
            }
        )
        
        return session_id
    
    async def session_end(self, session_id: str, reason: Optional[str] = None):
        """End session and log summary."""
        if session_id not in self.active_sessions:
            return
        
        session_info = self.active_sessions[session_id]
        duration = time.time() - session_info["start_time"]
        
        # Set session context
        CorrelationIdManager.set_session_id(session_id)
        CorrelationIdManager.set_correlation_id(session_info["correlation_id"])
        CorrelationIdManager.set_component_context("livekit")
        
        # Log session end
        self.logger.info(
            f"LiveKit session ended: {session_info['room_name']}",
            extra_data={
                "session_id": session_id,
                "room_name": session_info["room_name"],
                "participant_identity": session_info["participant_identity"],
                "duration_ms": duration * 1000,
                "pipeline_stages": session_info["pipeline_stages"],
                "reason": reason,
                "phase": "session_end"
            }
        )
        
        # Clean up
        del self.active_sessions[session_id]
    
    async def log_pipeline_stage(
        self,
        session_id: str,
        stage: str,
        component: str,
        duration: float,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log pipeline stage completion."""
        if session_id not in self.active_sessions:
            return
        
        session_info = self.active_sessions[session_id]
        
        # Set session context
        CorrelationIdManager.set_session_id(session_id)
        CorrelationIdManager.set_correlation_id(session_info["correlation_id"])
        CorrelationIdManager.set_component_context(component)
        
        stage_info = {
            "stage": stage,
            "component": component,
            "duration_ms": duration * 1000,
            "success": success,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        
        # Add to session pipeline stages
        session_info["pipeline_stages"].append(stage_info)
        
        # Log stage
        level = "info" if success else "error"
        message = f"Pipeline stage {stage} {'completed' if success else 'failed'}"
        
        getattr(self.logger, level)(
            message,
            extra_data={
                "session_id": session_id,
                "stage": stage,
                "component": component,
                "duration_ms": duration * 1000,
                "success": success,
                "phase": "pipeline_stage",
                "metadata": metadata
            }
        )
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information."""
        return self.active_sessions.get(session_id)
    
    def get_active_sessions(self) -> Dict[str, Dict[str, Any]]:
        """Get all active sessions."""
        return self.active_sessions.copy()


@asynccontextmanager
async def correlation_context(
    correlation_id: Optional[str] = None,
    request_id: Optional[str] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    component: Optional[str] = None
):
    """Context manager for correlation tracking."""
    # Save current context
    old_context = CorrelationIdManager.get_all_context()
    
    try:
        # Set new context
        if correlation_id:
            CorrelationIdManager.set_correlation_id(correlation_id)
        if request_id:
            CorrelationIdManager.set_request_id(request_id)
        if session_id:
            CorrelationIdManager.set_session_id(session_id)
        if user_id:
            CorrelationIdManager.set_user_id(user_id)
        if component:
            CorrelationIdManager.set_component_context(component)
        
        yield CorrelationIdManager.get_all_context()
        
    finally:
        # Restore old context
        correlation_id_var.set(old_context.get("correlation_id"))
        request_id_var.set(old_context.get("request_id"))
        session_id_var.set(old_context.get("session_id"))
        user_id_var.set(old_context.get("user_id"))
        component_var.set(old_context.get("component"))


class PipelineCorrelationTracker:
    """
    Tracks correlation across the entire voice pipeline (STT -> LLM -> TTS).
    """
    
    def __init__(self):
        self.logger = StructuredLogger(__name__, "pipeline")
        self.active_pipelines: Dict[str, Dict[str, Any]] = {}
    
    async def start_pipeline(
        self,
        session_id: str,
        user_input: str,
        pipeline_id: Optional[str] = None
    ) -> str:
        """Start a new pipeline execution."""
        if not pipeline_id:
            pipeline_id = f"pipe_{uuid.uuid4().hex[:8]}"
        
        correlation_id = CorrelationIdManager.get_correlation_id()
        if not correlation_id:
            correlation_id = CorrelationIdManager.generate_correlation_id()
            CorrelationIdManager.set_correlation_id(correlation_id)
        
        pipeline_info = {
            "pipeline_id": pipeline_id,
            "session_id": session_id,
            "correlation_id": correlation_id,
            "user_input": user_input,
            "start_time": time.time(),
            "stages": {},
            "total_duration": 0,
            "success": True
        }
        
        self.active_pipelines[pipeline_id] = pipeline_info
        
        # Log pipeline start
        self.logger.info(
            f"Voice pipeline started: {pipeline_id}",
            extra_data={
                "pipeline_id": pipeline_id,
                "session_id": session_id,
                "user_input_length": len(user_input),
                "phase": "pipeline_start"
            }
        )
        
        return pipeline_id
    
    async def log_stage_start(
        self,
        pipeline_id: str,
        stage: str,
        component: str,
        input_data: Optional[Dict[str, Any]] = None
    ):
        """Log the start of a pipeline stage."""
        if pipeline_id not in self.active_pipelines:
            return
        
        pipeline_info = self.active_pipelines[pipeline_id]
        stage_info = {
            "stage": stage,
            "component": component,
            "start_time": time.time(),
            "input_data": input_data or {},
            "success": None,
            "duration": 0
        }
        
        pipeline_info["stages"][stage] = stage_info
        
        # Set context
        CorrelationIdManager.set_correlation_id(pipeline_info["correlation_id"])
        CorrelationIdManager.set_session_id(pipeline_info["session_id"])
        CorrelationIdManager.set_component_context(component)
        
        self.logger.info(
            f"Pipeline stage started: {stage}",
            extra_data={
                "pipeline_id": pipeline_id,
                "stage": stage,
                "component": component,
                "phase": "stage_start",
                "input_data": input_data
            }
        )
    
    async def log_stage_end(
        self,
        pipeline_id: str,
        stage: str,
        success: bool = True,
        output_data: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ):
        """Log the end of a pipeline stage."""
        if pipeline_id not in self.active_pipelines:
            return
        
        pipeline_info = self.active_pipelines[pipeline_id]
        if stage not in pipeline_info["stages"]:
            return
        
        stage_info = pipeline_info["stages"][stage]
        stage_info["success"] = success
        stage_info["duration"] = time.time() - stage_info["start_time"]
        stage_info["output_data"] = output_data or {}
        
        if not success:
            pipeline_info["success"] = False
            stage_info["error"] = error
        
        # Set context
        CorrelationIdManager.set_correlation_id(pipeline_info["correlation_id"])
        CorrelationIdManager.set_session_id(pipeline_info["session_id"])
        CorrelationIdManager.set_component_context(stage_info["component"])
        
        level = "info" if success else "error"
        message = f"Pipeline stage {'completed' if success else 'failed'}: {stage}"
        
        extra_data = {
            "pipeline_id": pipeline_id,
            "stage": stage,
            "component": stage_info["component"],
            "duration_ms": stage_info["duration"] * 1000,
            "success": success,
            "phase": "stage_end",
            "output_data": output_data
        }
        
        if error:
            extra_data["error"] = error
        
        getattr(self.logger, level)(message, extra_data=extra_data)
    
    async def end_pipeline(
        self,
        pipeline_id: str,
        final_output: Optional[str] = None
    ):
        """End pipeline execution and log summary."""
        if pipeline_id not in self.active_pipelines:
            return
        
        pipeline_info = self.active_pipelines[pipeline_id]
        pipeline_info["total_duration"] = time.time() - pipeline_info["start_time"]
        pipeline_info["final_output"] = final_output
        
        # Set context
        CorrelationIdManager.set_correlation_id(pipeline_info["correlation_id"])
        CorrelationIdManager.set_session_id(pipeline_info["session_id"])
        CorrelationIdManager.set_component_context("pipeline")
        
        # Calculate stage durations
        stage_durations = {
            stage: info["duration"] * 1000 
            for stage, info in pipeline_info["stages"].items()
        }
        
        level = "info" if pipeline_info["success"] else "error"
        message = f"Voice pipeline {'completed' if pipeline_info['success'] else 'failed'}: {pipeline_id}"
        
        self.logger.log(
            getattr(logging, level.upper()),
            message,
            extra_data={
                "pipeline_id": pipeline_id,
                "total_duration_ms": pipeline_info["total_duration"] * 1000,
                "stage_durations": stage_durations,
                "success": pipeline_info["success"],
                "stages_completed": len(pipeline_info["stages"]),
                "final_output_length": len(final_output) if final_output else 0,
                "phase": "pipeline_end"
            }
        )
        
        # Clean up
        del self.active_pipelines[pipeline_id]
    
    def get_pipeline_info(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """Get pipeline information."""
        return self.active_pipelines.get(pipeline_id)


# Global instances
livekit_middleware = LiveKitSessionMiddleware()
pipeline_tracker = PipelineCorrelationTracker()


def add_correlation_middleware(
    app: FastAPI,
    **kwargs
) -> FastAPI:
    """Add correlation middleware to FastAPI app."""
    app.add_middleware(CorrelationMiddleware, **kwargs)
    return app