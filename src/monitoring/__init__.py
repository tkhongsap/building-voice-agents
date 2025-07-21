"""
Monitoring package for voice agents.
Includes performance monitoring, conversation inspection, debugging tools,
and comprehensive structured logging with correlation ID tracking.
"""

from .performance_monitor import PerformanceMonitor, global_performance_monitor
from .conversation_inspector import ConversationInspector
from .debug_dashboard import DebugDashboard
from .performance_profiler import PerformanceProfiler

# Structured Logging System
from .structured_logging import (
    StructuredLogger,
    CorrelationIdManager,
    log_with_correlation,
    timed_operation,
    ComponentType,
    LogLevel
)
from .logging_config import (
    setup_logging,
    get_logger,
    LoggingConfig,
    LoggingManager,
    LogFormat,
    LogOutput,
    set_correlation_context,
    clear_correlation_context
)
from .logging_middleware import (
    CorrelationMiddleware,
    LiveKitSessionMiddleware,
    PipelineCorrelationTracker,
    correlation_context,
    add_correlation_middleware,
    livekit_middleware,
    pipeline_tracker
)

__all__ = [
    # Performance and debugging
    "PerformanceMonitor",
    "global_performance_monitor",
    "ConversationInspector", 
    "DebugDashboard",
    "PerformanceProfiler",
    
    # Structured logging core
    "StructuredLogger",
    "CorrelationIdManager",
    "log_with_correlation",
    "timed_operation",
    "ComponentType",
    "LogLevel",
    
    # Logging configuration
    "setup_logging",
    "get_logger",
    "LoggingConfig",
    "LoggingManager",
    "LogFormat",
    "LogOutput",
    "set_correlation_context",
    "clear_correlation_context",
    
    # Middleware and tracking
    "CorrelationMiddleware",
    "LiveKitSessionMiddleware", 
    "PipelineCorrelationTracker",
    "correlation_context",
    "add_correlation_middleware",
    "livekit_middleware",
    "pipeline_tracker"
]