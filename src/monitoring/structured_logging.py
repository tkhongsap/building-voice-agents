"""
Structured Logging System with Correlation IDs

This module provides a comprehensive structured logging system with JSON formatting,
correlation ID tracking, and integration with the voice agent pipeline components.
"""

import logging
import json
import uuid
import time
import traceback
import contextvars
import functools
from typing import Any, Dict, Optional, Union, Callable
from datetime import datetime, timezone
from enum import Enum
from dataclasses import dataclass, asdict, field
from pathlib import Path
import sys
import os

# Correlation ID context variable
correlation_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    'correlation_id', default=None
)

# Request ID context variable (for web requests)
request_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    'request_id', default=None
)

# Session ID context variable (for voice sessions)
session_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    'session_id', default=None
)

# User ID context variable
user_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    'user_id', default=None
)

# Component context variable (which pipeline component is logging)
component_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    'component', default=None
)


class LogLevel(Enum):
    """Enhanced log levels for structured logging."""
    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ComponentType(Enum):
    """Types of voice pipeline components."""
    STT = "stt"
    LLM = "llm"
    TTS = "tts"
    VAD = "vad"
    PIPELINE = "pipeline"
    API = "api"
    AGENT = "agent"
    SYSTEM = "system"
    MONITORING = "monitoring"


@dataclass
class StructuredLogRecord:
    """Structured log record with all required fields."""
    timestamp: str
    level: str
    message: str
    correlation_id: Optional[str] = None
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    component: Optional[str] = None
    component_type: Optional[str] = None
    module: Optional[str] = None
    function: Optional[str] = None
    line_number: Optional[int] = None
    file_path: Optional[str] = None
    process_id: Optional[int] = None
    thread_id: Optional[int] = None
    extra_data: Dict[str, Any] = field(default_factory=dict)
    error_details: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert log record to dictionary, excluding None values."""
        data = asdict(self)
        return {k: v for k, v in data.items() if v is not None}
    
    def to_json(self) -> str:
        """Convert log record to JSON string."""
        return json.dumps(self.to_dict(), default=str, ensure_ascii=False)


class StructuredJSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def __init__(
        self,
        include_extra: bool = True,
        include_traceback: bool = True,
        sort_keys: bool = True,
        indent: Optional[int] = None
    ):
        super().__init__()
        self.include_extra = include_extra
        self.include_traceback = include_traceback
        self.sort_keys = sort_keys
        self.indent = indent
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        # Create timestamp in ISO format with timezone
        timestamp = datetime.fromtimestamp(
            record.created, tz=timezone.utc
        ).isoformat()
        
        # Get context variables
        correlation_id = correlation_id_var.get()
        request_id = request_id_var.get()
        session_id = session_id_var.get()
        user_id = user_id_var.get()
        component = component_var.get()
        
        # Extract component type from component name if available
        component_type = None
        if component:
            for comp_type in ComponentType:
                if comp_type.value.lower() in component.lower():
                    component_type = comp_type.value
                    break
        
        # Build structured log record
        log_record = StructuredLogRecord(
            timestamp=timestamp,
            level=record.levelname,
            message=record.getMessage(),
            correlation_id=correlation_id,
            request_id=request_id,
            session_id=session_id,
            user_id=user_id,
            component=component,
            component_type=component_type,
            module=record.module,
            function=record.funcName,
            line_number=record.lineno,
            file_path=record.pathname,
            process_id=os.getpid(),
            thread_id=record.thread
        )
        
        # Add extra data if available
        if self.include_extra and hasattr(record, 'extra_data'):
            log_record.extra_data.update(record.extra_data)
        
        # Add performance metrics if available
        if hasattr(record, 'performance_metrics'):
            log_record.performance_metrics = record.performance_metrics
        
        # Handle exceptions
        if record.exc_info and self.include_traceback:
            log_record.error_details = {
                "exception_type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "exception_message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info) if record.exc_info else None
            }
        
        # Convert to JSON
        return json.dumps(
            log_record.to_dict(),
            default=str,
            ensure_ascii=False,
            sort_keys=self.sort_keys,
            indent=self.indent
        )


class CorrelationIdManager:
    """Manager for correlation ID generation and propagation."""
    
    @staticmethod
    def generate_correlation_id() -> str:
        """Generate a new correlation ID."""
        return str(uuid.uuid4())
    
    @staticmethod
    def get_correlation_id() -> Optional[str]:
        """Get the current correlation ID from context."""
        return correlation_id_var.get()
    
    @staticmethod
    def set_correlation_id(correlation_id: Optional[str] = None) -> str:
        """Set correlation ID in context. Generates new one if not provided."""
        if correlation_id is None:
            correlation_id = CorrelationIdManager.generate_correlation_id()
        correlation_id_var.set(correlation_id)
        return correlation_id
    
    @staticmethod
    def clear_correlation_id():
        """Clear correlation ID from context."""
        correlation_id_var.set(None)
    
    @staticmethod
    def generate_request_id() -> str:
        """Generate a new request ID."""
        return f"req_{uuid.uuid4().hex[:8]}"
    
    @staticmethod
    def set_request_id(request_id: Optional[str] = None) -> str:
        """Set request ID in context."""
        if request_id is None:
            request_id = CorrelationIdManager.generate_request_id()
        request_id_var.set(request_id)
        return request_id
    
    @staticmethod
    def generate_session_id() -> str:
        """Generate a new session ID."""
        return f"sess_{uuid.uuid4().hex[:12]}"
    
    @staticmethod
    def set_session_id(session_id: Optional[str] = None) -> str:
        """Set session ID in context."""
        if session_id is None:
            session_id = CorrelationIdManager.generate_session_id()
        session_id_var.set(session_id)
        return session_id
    
    @staticmethod
    def set_user_id(user_id: str):
        """Set user ID in context."""
        user_id_var.set(user_id)
    
    @staticmethod
    def set_component_context(component: str):
        """Set component context."""
        component_var.set(component)
    
    @staticmethod
    def get_all_context() -> Dict[str, Optional[str]]:
        """Get all context variables."""
        return {
            "correlation_id": correlation_id_var.get(),
            "request_id": request_id_var.get(),
            "session_id": session_id_var.get(),
            "user_id": user_id_var.get(),
            "component": component_var.get()
        }


class StructuredLogger:
    """Enhanced logger with structured logging capabilities."""
    
    def __init__(self, name: str, component: Optional[str] = None):
        self.logger = logging.getLogger(name)
        self.component = component
        if component:
            component_var.set(component)
    
    def _log_with_context(
        self,
        level: int,
        message: str,
        extra_data: Optional[Dict[str, Any]] = None,
        performance_metrics: Optional[Dict[str, Any]] = None,
        exc_info: bool = False
    ):
        """Log with structured context."""
        # Set component context if specified
        if self.component:
            component_var.set(self.component)
        
        # Create extra data for the log record
        extra = {}
        if extra_data:
            extra['extra_data'] = extra_data
        if performance_metrics:
            extra['performance_metrics'] = performance_metrics
        
        self.logger.log(level, message, extra=extra, exc_info=exc_info)
    
    def trace(self, message: str, **kwargs):
        """Log trace level message."""
        self._log_with_context(5, message, **kwargs)  # TRACE level
    
    def debug(self, message: str, **kwargs):
        """Log debug level message."""
        self._log_with_context(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info level message."""
        self._log_with_context(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning level message."""
        self._log_with_context(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error level message."""
        self._log_with_context(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical level message."""
        self._log_with_context(logging.CRITICAL, message, **kwargs)
    
    def exception(self, message: str, **kwargs):
        """Log exception with traceback."""
        kwargs['exc_info'] = True
        self._log_with_context(logging.ERROR, message, **kwargs)
    
    def log_operation_start(
        self, 
        operation: str, 
        extra_data: Optional[Dict[str, Any]] = None
    ):
        """Log the start of an operation."""
        data = {"operation": operation, "phase": "start"}
        if extra_data:
            data.update(extra_data)
        self.info(f"Starting operation: {operation}", extra_data=data)
    
    def log_operation_end(
        self, 
        operation: str, 
        duration: Optional[float] = None,
        success: bool = True,
        extra_data: Optional[Dict[str, Any]] = None
    ):
        """Log the end of an operation."""
        data = {
            "operation": operation, 
            "phase": "end", 
            "success": success
        }
        if duration is not None:
            data["duration_ms"] = duration * 1000
        if extra_data:
            data.update(extra_data)
        
        level_msg = "Completed" if success else "Failed"
        level = logging.INFO if success else logging.ERROR
        self._log_with_context(
            level, 
            f"{level_msg} operation: {operation}", 
            extra_data=data
        )
    
    def log_performance_metrics(
        self, 
        operation: str, 
        metrics: Dict[str, Any]
    ):
        """Log performance metrics for an operation."""
        self.info(
            f"Performance metrics for {operation}",
            performance_metrics=metrics
        )


def log_with_correlation(
    correlation_id: Optional[str] = None,
    request_id: Optional[str] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    component: Optional[str] = None
):
    """Decorator to set logging context for a function."""
    def decorator(func: Callable) -> Callable:
        if hasattr(func, '__call__'):
            if asyncio and asyncio.iscoroutinefunction(func):
                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs):
                    # Set context
                    old_context = CorrelationIdManager.get_all_context()
                    
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
                    
                    try:
                        return await func(*args, **kwargs)
                    finally:
                        # Restore old context
                        for key, value in old_context.items():
                            if key == 'correlation_id':
                                correlation_id_var.set(value)
                            elif key == 'request_id':
                                request_id_var.set(value)
                            elif key == 'session_id':
                                session_id_var.set(value)
                            elif key == 'user_id':
                                user_id_var.set(value)
                            elif key == 'component':
                                component_var.set(value)
                
                return async_wrapper
            else:
                @functools.wraps(func)
                def sync_wrapper(*args, **kwargs):
                    # Set context
                    old_context = CorrelationIdManager.get_all_context()
                    
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
                    
                    try:
                        return func(*args, **kwargs)
                    finally:
                        # Restore old context
                        for key, value in old_context.items():
                            if key == 'correlation_id':
                                correlation_id_var.set(value)
                            elif key == 'request_id':
                                request_id_var.set(value)
                            elif key == 'session_id':
                                session_id_var.set(value)
                            elif key == 'user_id':
                                user_id_var.set(value)
                            elif key == 'component':
                                component_var.set(value)
                
                return sync_wrapper
        
        return func
    return decorator


def timed_operation(
    operation_name: Optional[str] = None,
    component: Optional[str] = None,
    log_args: bool = False,
    log_result: bool = False
):
    """Decorator to automatically log operation timing and context."""
    def decorator(func: Callable) -> Callable:
        op_name = operation_name or func.__name__
        comp_name = component or func.__module__.split('.')[-1]
        
        if hasattr(func, '__call__'):
            if asyncio and asyncio.iscoroutinefunction(func):
                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs):
                    logger = StructuredLogger(func.__module__, comp_name)
                    start_time = time.time()
                    
                    # Log operation start
                    extra_data = {"operation": op_name}
                    if log_args:
                        extra_data["args"] = args
                        extra_data["kwargs"] = kwargs
                    
                    logger.log_operation_start(op_name, extra_data)
                    
                    try:
                        result = await func(*args, **kwargs)
                        end_time = time.time()
                        duration = end_time - start_time
                        
                        # Log successful completion
                        end_data = {"operation": op_name}
                        if log_result:
                            end_data["result"] = result
                        
                        logger.log_operation_end(op_name, duration, True, end_data)
                        
                        # Log performance metrics
                        metrics = {
                            "operation": op_name,
                            "duration_ms": duration * 1000,
                            "success": True
                        }
                        logger.log_performance_metrics(op_name, metrics)
                        
                        return result
                        
                    except Exception as e:
                        end_time = time.time()
                        duration = end_time - start_time
                        
                        # Log failure
                        error_data = {
                            "operation": op_name,
                            "error": str(e),
                            "error_type": type(e).__name__
                        }
                        logger.log_operation_end(op_name, duration, False, error_data)
                        logger.exception(f"Operation {op_name} failed: {e}")
                        
                        raise
                
                return async_wrapper
            else:
                @functools.wraps(func)
                def sync_wrapper(*args, **kwargs):
                    logger = StructuredLogger(func.__module__, comp_name)
                    start_time = time.time()
                    
                    # Log operation start
                    extra_data = {"operation": op_name}
                    if log_args:
                        extra_data["args"] = args
                        extra_data["kwargs"] = kwargs
                    
                    logger.log_operation_start(op_name, extra_data)
                    
                    try:
                        result = func(*args, **kwargs)
                        end_time = time.time()
                        duration = end_time - start_time
                        
                        # Log successful completion
                        end_data = {"operation": op_name}
                        if log_result:
                            end_data["result"] = result
                        
                        logger.log_operation_end(op_name, duration, True, end_data)
                        
                        # Log performance metrics
                        metrics = {
                            "operation": op_name,
                            "duration_ms": duration * 1000,
                            "success": True
                        }
                        logger.log_performance_metrics(op_name, metrics)
                        
                        return result
                        
                    except Exception as e:
                        end_time = time.time()
                        duration = end_time - start_time
                        
                        # Log failure
                        error_data = {
                            "operation": op_name,
                            "error": str(e),
                            "error_type": type(e).__name__
                        }
                        logger.log_operation_end(op_name, duration, False, error_data)
                        logger.exception(f"Operation {op_name} failed: {e}")
                        
                        raise
                
                return sync_wrapper
        
        return func
    return decorator


# Import asyncio at module level to avoid issues
try:
    import asyncio
except ImportError:
    asyncio = None