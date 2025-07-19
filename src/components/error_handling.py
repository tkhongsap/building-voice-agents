"""
Comprehensive Error Handling and Graceful Degradation System

This module provides error handling, fallback mechanisms, and graceful degradation
for the voice processing pipeline to ensure robust operation.
"""

import asyncio
import logging
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List, Callable, Union, Type, AsyncIterator
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"           # Minor issues, system continues normally
    MEDIUM = "medium"     # Noticeable issues, degraded performance
    HIGH = "high"         # Significant issues, major functionality affected
    CRITICAL = "critical" # System-threatening, immediate attention required


class ErrorCategory(Enum):
    """Error category types."""
    NETWORK = "network"                 # Network connectivity issues
    API_LIMIT = "api_limit"            # Rate limiting, quota exceeded
    AUTHENTICATION = "authentication"  # API key, authentication failures
    PROVIDER_ERROR = "provider_error"  # AI provider service errors
    AUDIO_PROCESSING = "audio_processing"  # Audio format, processing errors
    CONFIGURATION = "configuration"    # Invalid configuration, missing settings
    RESOURCE = "resource"              # Memory, disk, CPU issues
    TIMEOUT = "timeout"                # Operation timeouts
    UNKNOWN = "unknown"                # Unclassified errors


@dataclass
class ErrorInfo:
    """Comprehensive error information."""
    error_type: str
    message: str
    severity: ErrorSeverity
    category: ErrorCategory
    timestamp: float = field(default_factory=time.time)
    component: Optional[str] = None
    provider: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    backoff_delay: float = 1.0
    context: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    
    def __post_init__(self):
        """Capture stack trace if not provided."""
        if self.stack_trace is None:
            self.stack_trace = traceback.format_exc()
    
    def is_retryable(self) -> bool:
        """Check if this error should be retried."""
        retryable_categories = {
            ErrorCategory.NETWORK,
            ErrorCategory.API_LIMIT,
            ErrorCategory.TIMEOUT,
            ErrorCategory.PROVIDER_ERROR
        }
        return (
            self.category in retryable_categories and
            self.retry_count < self.max_retries and
            self.severity != ErrorSeverity.CRITICAL
        )
    
    def get_next_retry_delay(self) -> float:
        """Calculate next retry delay with exponential backoff."""
        return self.backoff_delay * (2 ** self.retry_count)


class FallbackStrategy(Enum):
    """Fallback strategy types."""
    RETRY = "retry"                    # Retry the same operation
    FALLBACK_PROVIDER = "fallback_provider"  # Switch to backup provider
    DEGRADE_QUALITY = "degrade_quality"      # Reduce quality/features
    SKIP_COMPONENT = "skip_component"        # Skip optional component
    CACHE_RESPONSE = "cache_response"        # Use cached response
    DEFAULT_RESPONSE = "default_response"    # Use predefined default


@dataclass
class FallbackConfig:
    """Configuration for fallback behavior."""
    strategy: FallbackStrategy
    max_retries: int = 3
    retry_delay: float = 1.0
    exponential_backoff: bool = True
    fallback_providers: List[str] = field(default_factory=list)
    quality_degradation_steps: List[Dict[str, Any]] = field(default_factory=list)
    default_response: Any = None
    cache_duration: float = 300.0  # 5 minutes
    enabled: bool = True


class ErrorHandler:
    """Centralized error handling and recovery system."""
    
    def __init__(self):
        self.error_history: List[ErrorInfo] = []
        self.fallback_configs: Dict[str, FallbackConfig] = {}
        self.provider_health: Dict[str, Dict[str, Any]] = {}
        self.circuit_breakers: Dict[str, 'CircuitBreaker'] = {}
        self.error_callbacks: List[Callable[[ErrorInfo], None]] = []
        self.max_history_size = 1000
        
        # Default fallback configurations
        self._setup_default_fallbacks()
    
    def _setup_default_fallbacks(self):
        """Setup default fallback configurations."""
        # STT fallback: try alternative providers
        self.fallback_configs["stt"] = FallbackConfig(
            strategy=FallbackStrategy.FALLBACK_PROVIDER,
            fallback_providers=["openai_whisper", "azure_speech", "google_speech"],
            max_retries=2
        )
        
        # LLM fallback: try alternative providers with quality degradation
        self.fallback_configs["llm"] = FallbackConfig(
            strategy=FallbackStrategy.FALLBACK_PROVIDER,
            fallback_providers=["openai_gpt", "anthropic_claude", "local_llm"],
            quality_degradation_steps=[
                {"max_tokens": 1000, "temperature": 0.7},
                {"max_tokens": 500, "temperature": 0.5},
                {"max_tokens": 100, "temperature": 0.3}
            ],
            max_retries=3
        )
        
        # TTS fallback: ElevenLabs primary, OpenAI fallback
        self.fallback_configs["tts"] = FallbackConfig(
            strategy=FallbackStrategy.FALLBACK_PROVIDER,
            fallback_providers=["elevenlabs", "openai"],
            max_retries=2
        )
        
        # VAD fallback: switch to simpler algorithm
        self.fallback_configs["vad"] = FallbackConfig(
            strategy=FallbackStrategy.FALLBACK_PROVIDER,
            fallback_providers=["silero", "webrtc"],
            max_retries=1
        )
    
    def register_error_callback(self, callback: Callable[[ErrorInfo], None]):
        """Register callback for error notifications."""
        self.error_callbacks.append(callback)
    
    def classify_error(self, exception: Exception, component: str = None) -> ErrorInfo:
        """Classify an exception into structured error information."""
        error_type = type(exception).__name__
        message = str(exception)
        
        # Determine category and severity
        category = self._categorize_error(exception, message)
        severity = self._assess_severity(exception, category, component)
        
        # Extract additional context
        context = self._extract_error_context(exception)
        
        return ErrorInfo(
            error_type=error_type,
            message=message,
            severity=severity,
            category=category,
            component=component,
            context=context
        )
    
    def _categorize_error(self, exception: Exception, message: str) -> ErrorCategory:
        """Categorize error based on exception type and message."""
        error_type = type(exception).__name__.lower()
        message_lower = message.lower()
        
        # Network-related errors
        if any(term in error_type for term in ['connection', 'network', 'timeout']):
            return ErrorCategory.NETWORK
        if any(term in message_lower for term in ['connection', 'network', 'unreachable']):
            return ErrorCategory.NETWORK
        
        # Authentication errors
        if any(term in error_type for term in ['auth', 'permission', 'forbidden']):
            return ErrorCategory.AUTHENTICATION
        if any(term in message_lower for term in ['unauthorized', 'api key', 'authentication']):
            return ErrorCategory.AUTHENTICATION
        
        # API rate limiting
        if any(term in message_lower for term in ['rate limit', 'quota', 'too many requests']):
            return ErrorCategory.API_LIMIT
        
        # Timeout errors
        if 'timeout' in error_type or 'timeout' in message_lower:
            return ErrorCategory.TIMEOUT
        
        # Audio processing errors
        if any(term in message_lower for term in ['audio', 'wav', 'mp3', 'format', 'sample rate']):
            return ErrorCategory.AUDIO_PROCESSING
        
        # Configuration errors
        if any(term in error_type for term in ['value', 'type', 'attribute']):
            return ErrorCategory.CONFIGURATION
        if any(term in message_lower for term in ['config', 'setting', 'parameter']):
            return ErrorCategory.CONFIGURATION
        
        # Resource errors
        if any(term in message_lower for term in ['memory', 'disk', 'space', 'resource']):
            return ErrorCategory.RESOURCE
        
        return ErrorCategory.UNKNOWN
    
    def _assess_severity(self, exception: Exception, category: ErrorCategory, component: str) -> ErrorSeverity:
        """Assess error severity based on various factors."""
        # Critical errors
        if isinstance(exception, (MemoryError, SystemError)):
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        if category in [ErrorCategory.AUTHENTICATION, ErrorCategory.CONFIGURATION]:
            return ErrorSeverity.HIGH
        
        # Component-specific severity
        if component == "llm":
            # LLM errors are high impact
            return ErrorSeverity.HIGH
        elif component in ["stt", "tts"]:
            # Audio processing errors are medium impact
            return ErrorSeverity.MEDIUM
        elif component == "vad":
            # VAD errors are lower impact
            return ErrorSeverity.LOW
        
        # Category-based severity
        severity_map = {
            ErrorCategory.NETWORK: ErrorSeverity.MEDIUM,
            ErrorCategory.API_LIMIT: ErrorSeverity.MEDIUM,
            ErrorCategory.PROVIDER_ERROR: ErrorSeverity.MEDIUM,
            ErrorCategory.AUDIO_PROCESSING: ErrorSeverity.LOW,
            ErrorCategory.TIMEOUT: ErrorSeverity.MEDIUM,
            ErrorCategory.RESOURCE: ErrorSeverity.HIGH,
        }
        
        return severity_map.get(category, ErrorSeverity.LOW)
    
    def _extract_error_context(self, exception: Exception) -> Dict[str, Any]:
        """Extract additional context from exception."""
        context = {}
        
        # HTTP-related context
        if hasattr(exception, 'response'):
            response = exception.response
            context.update({
                "status_code": getattr(response, 'status_code', None),
                "headers": dict(getattr(response, 'headers', {})),
                "url": str(getattr(response, 'url', ''))
            })
        
        # Request context
        if hasattr(exception, 'request'):
            request = exception.request
            context.update({
                "method": getattr(request, 'method', None),
                "url": str(getattr(request, 'url', ''))
            })
        
        return context
    
    async def handle_error(
        self, 
        exception: Exception, 
        component: str = None,
        operation: str = None,
        **kwargs
    ) -> Optional[Any]:
        """
        Handle an error with appropriate fallback strategy.
        
        Args:
            exception: The exception that occurred
            component: Component where error occurred
            operation: Operation that failed
            **kwargs: Additional context
            
        Returns:
            Fallback result if available, None otherwise
        """
        # Classify the error
        error_info = self.classify_error(exception, component)
        error_info.context.update(kwargs)
        if operation:
            error_info.context["operation"] = operation
        
        # Add to error history
        self._add_to_history(error_info)
        
        # Notify callbacks
        for callback in self.error_callbacks:
            try:
                callback(error_info)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")
        
        # Log the error
        self._log_error(error_info)
        
        # Update provider health if applicable
        if error_info.provider:
            self._update_provider_health(error_info.provider, error_info)
        
        # Apply fallback strategy
        return await self._apply_fallback(error_info, component, operation, **kwargs)
    
    async def _apply_fallback(
        self, 
        error_info: ErrorInfo, 
        component: str, 
        operation: str,
        **kwargs
    ) -> Optional[Any]:
        """Apply appropriate fallback strategy."""
        if not component or component not in self.fallback_configs:
            return None
        
        fallback_config = self.fallback_configs[component]
        if not fallback_config.enabled:
            return None
        
        strategy = fallback_config.strategy
        
        try:
            if strategy == FallbackStrategy.RETRY:
                return await self._retry_operation(error_info, operation, **kwargs)
            
            elif strategy == FallbackStrategy.FALLBACK_PROVIDER:
                return await self._try_fallback_providers(error_info, component, operation, **kwargs)
            
            elif strategy == FallbackStrategy.DEGRADE_QUALITY:
                return await self._degrade_quality(error_info, component, operation, **kwargs)
            
            elif strategy == FallbackStrategy.DEFAULT_RESPONSE:
                return fallback_config.default_response
            
            elif strategy == FallbackStrategy.SKIP_COMPONENT:
                logger.warning(f"Skipping component {component} due to error")
                return None
            
        except Exception as e:
            logger.error(f"Fallback strategy failed: {e}")
        
        return None
    
    async def _retry_operation(self, error_info: ErrorInfo, operation: str, **kwargs) -> Optional[Any]:
        """Retry the failed operation with backoff."""
        if not error_info.is_retryable():
            return None
        
        retry_delay = error_info.get_next_retry_delay()
        error_info.retry_count += 1
        
        logger.info(f"Retrying operation {operation} (attempt {error_info.retry_count}) after {retry_delay}s")
        
        await asyncio.sleep(retry_delay)
        
        # The actual retry would be handled by the calling code
        # This method just handles the retry logic
        return None
    
    async def _try_fallback_providers(
        self, 
        error_info: ErrorInfo, 
        component: str, 
        operation: str, 
        **kwargs
    ) -> Optional[Any]:
        """Try fallback providers in order."""
        fallback_config = self.fallback_configs[component]
        
        for provider in fallback_config.fallback_providers:
            if self._is_provider_healthy(provider):
                logger.info(f"Trying fallback provider {provider} for {component}")
                # The calling code would need to implement the actual provider switching
                # This method provides the fallback provider recommendation
                return {"fallback_provider": provider}
        
        return None
    
    async def _degrade_quality(
        self, 
        error_info: ErrorInfo, 
        component: str, 
        operation: str, 
        **kwargs
    ) -> Optional[Any]:
        """Apply quality degradation steps."""
        fallback_config = self.fallback_configs[component]
        
        if fallback_config.quality_degradation_steps:
            step_index = min(error_info.retry_count, len(fallback_config.quality_degradation_steps) - 1)
            degradation_params = fallback_config.quality_degradation_steps[step_index]
            
            logger.info(f"Degrading quality for {component}: {degradation_params}")
            return {"quality_degradation": degradation_params}
        
        return None
    
    def _is_provider_healthy(self, provider: str) -> bool:
        """Check if a provider is currently healthy."""
        if provider in self.provider_health:
            health_info = self.provider_health[provider]
            last_error_time = health_info.get("last_error_time", 0)
            error_count = health_info.get("error_count", 0)
            
            # Consider unhealthy if too many recent errors
            if error_count > 5 and (time.time() - last_error_time) < 300:  # 5 minutes
                return False
        
        return True
    
    def _update_provider_health(self, provider: str, error_info: ErrorInfo):
        """Update provider health tracking."""
        if provider not in self.provider_health:
            self.provider_health[provider] = {
                "error_count": 0,
                "last_error_time": 0,
                "last_success_time": 0
            }
        
        health = self.provider_health[provider]
        health["error_count"] += 1
        health["last_error_time"] = error_info.timestamp
        
        # Reset error count if last success was recent
        if (error_info.timestamp - health.get("last_success_time", 0)) > 3600:  # 1 hour
            health["error_count"] = 1
    
    def record_success(self, provider: str):
        """Record successful operation for provider health."""
        if provider not in self.provider_health:
            self.provider_health[provider] = {
                "error_count": 0,
                "last_error_time": 0,
                "last_success_time": 0
            }
        
        self.provider_health[provider]["last_success_time"] = time.time()
        # Reduce error count on success
        if self.provider_health[provider]["error_count"] > 0:
            self.provider_health[provider]["error_count"] -= 1
    
    def _add_to_history(self, error_info: ErrorInfo):
        """Add error to history with size management."""
        self.error_history.append(error_info)
        
        # Trim history if too large
        if len(self.error_history) > self.max_history_size:
            self.error_history = self.error_history[-self.max_history_size//2:]
    
    def _log_error(self, error_info: ErrorInfo):
        """Log error with appropriate level."""
        log_message = (
            f"Error in {error_info.component or 'unknown'}: "
            f"{error_info.error_type} - {error_info.message}"
        )
        
        if error_info.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        elif error_info.severity == ErrorSeverity.HIGH:
            logger.error(log_message)
        elif error_info.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics and health metrics."""
        now = time.time()
        recent_errors = [e for e in self.error_history if now - e.timestamp < 3600]  # Last hour
        
        stats = {
            "total_errors": len(self.error_history),
            "recent_errors": len(recent_errors),
            "errors_by_severity": {},
            "errors_by_category": {},
            "errors_by_component": {},
            "provider_health": self.provider_health.copy(),
            "most_frequent_errors": {}
        }
        
        # Count by severity
        for severity in ErrorSeverity:
            stats["errors_by_severity"][severity.value] = len([
                e for e in recent_errors if e.severity == severity
            ])
        
        # Count by category
        for category in ErrorCategory:
            stats["errors_by_category"][category.value] = len([
                e for e in recent_errors if e.category == category
            ])
        
        # Count by component
        for error in recent_errors:
            component = error.component or "unknown"
            stats["errors_by_component"][component] = stats["errors_by_component"].get(component, 0) + 1
        
        # Most frequent error types
        error_types = {}
        for error in recent_errors:
            error_types[error.error_type] = error_types.get(error.error_type, 0) + 1
        
        stats["most_frequent_errors"] = dict(sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:10])
        
        return stats


class CircuitBreaker:
    """Circuit breaker for failing operations."""
    
    def __init__(
        self, 
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    @asynccontextmanager
    async def call(self):
        """Context manager for circuit breaker protected calls."""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            yield
            # Success
            self._on_success()
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return False
        return time.time() - self.last_failure_time >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful operation."""
        self.failure_count = 0
        self.state = "closed"
    
    def _on_failure(self):
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"


# Global error handler instance
global_error_handler = ErrorHandler()


def handle_errors(component: str = None, operation: str = None):
    """Decorator for automatic error handling."""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                try:
                    result = await func(*args, **kwargs)
                    # Record success if component/provider is specified
                    if hasattr(args[0], 'provider_name'):
                        global_error_handler.record_success(args[0].provider_name)
                    return result
                except Exception as e:
                    return await global_error_handler.handle_error(
                        e, component=component, operation=operation or func.__name__, 
                        args=args, kwargs=kwargs
                    )
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                try:
                    result = func(*args, **kwargs)
                    # Record success if component/provider is specified
                    if hasattr(args[0], 'provider_name'):
                        global_error_handler.record_success(args[0].provider_name)
                    return result
                except Exception as e:
                    # For sync functions, we can't await the handler
                    error_info = global_error_handler.classify_error(e, component)
                    global_error_handler._add_to_history(error_info)
                    global_error_handler._log_error(error_info)
                    raise
            return sync_wrapper
    return decorator


async def with_fallback(
    primary_func: Callable,
    fallback_funcs: List[Callable],
    component: str = None,
    operation: str = None,
    **kwargs
) -> Any:
    """Execute function with fallback options."""
    funcs_to_try = [primary_func] + fallback_funcs
    
    for i, func in enumerate(funcs_to_try):
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(**kwargs)
            else:
                result = func(**kwargs)
            
            # Record success
            if hasattr(func, 'provider_name'):
                global_error_handler.record_success(func.provider_name)
            
            return result
            
        except Exception as e:
            is_last_attempt = i == len(funcs_to_try) - 1
            
            if not is_last_attempt:
                # Log and continue to next fallback
                await global_error_handler.handle_error(
                    e, component=component, operation=operation,
                    attempt=i+1, total_attempts=len(funcs_to_try)
                )
                continue
            else:
                # Last attempt failed, re-raise
                await global_error_handler.handle_error(
                    e, component=component, operation=operation,
                    final_failure=True
                )
                raise