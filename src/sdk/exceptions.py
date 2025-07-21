"""
SDK-specific exceptions for the LiveKit Voice Agents Platform.

This module defines custom exceptions used throughout the SDK to provide
clear and actionable error messages to developers.
"""

from typing import Optional, Dict, Any


class VoiceAgentError(Exception):
    """Base exception for all voice agent SDK errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class ConfigurationError(VoiceAgentError):
    """Raised when there's an error in SDK configuration."""
    pass


class ComponentNotFoundError(VoiceAgentError):
    """Raised when a required component is not found or not registered."""
    
    def __init__(self, component_type: str, component_name: Optional[str] = None):
        message = f"Component of type '{component_type}' not found"
        if component_name:
            message += f": '{component_name}'"
        super().__init__(message, {"component_type": component_type, "component_name": component_name})


class ProviderError(VoiceAgentError):
    """Raised when there's an error with an AI provider."""
    
    def __init__(self, provider_name: str, message: str, original_error: Optional[Exception] = None):
        super().__init__(
            f"Provider '{provider_name}' error: {message}",
            {
                "provider": provider_name,
                "original_error": str(original_error) if original_error else None
            }
        )
        self.original_error = original_error


class ConnectionError(VoiceAgentError):
    """Raised when there's a connection error."""
    
    def __init__(self, service: str, message: str, retry_after: Optional[float] = None):
        super().__init__(
            f"Connection to '{service}' failed: {message}",
            {
                "service": service,
                "retry_after": retry_after
            }
        )


class ValidationError(VoiceAgentError):
    """Raised when validation fails."""
    
    def __init__(self, field: str, value: Any, reason: str):
        super().__init__(
            f"Validation failed for '{field}': {reason}",
            {
                "field": field,
                "value": value,
                "reason": reason
            }
        )


class StateError(VoiceAgentError):
    """Raised when an operation is performed in an invalid state."""
    
    def __init__(self, current_state: str, expected_state: str, operation: str):
        super().__init__(
            f"Invalid state for operation '{operation}'. Current: {current_state}, Expected: {expected_state}",
            {
                "current_state": current_state,
                "expected_state": expected_state,
                "operation": operation
            }
        )


class ResourceError(VoiceAgentError):
    """Raised when there's an issue with resource allocation or limits."""
    
    def __init__(self, resource_type: str, message: str, limit: Optional[int] = None, current: Optional[int] = None):
        super().__init__(
            f"Resource error for '{resource_type}': {message}",
            {
                "resource_type": resource_type,
                "limit": limit,
                "current": current
            }
        )


class TimeoutError(VoiceAgentError):
    """Raised when an operation times out."""
    
    def __init__(self, operation: str, timeout_seconds: float):
        super().__init__(
            f"Operation '{operation}' timed out after {timeout_seconds} seconds",
            {
                "operation": operation,
                "timeout_seconds": timeout_seconds
            }
        )


class AuthenticationError(VoiceAgentError):
    """Raised when authentication fails."""
    
    def __init__(self, service: str, reason: Optional[str] = None):
        message = f"Authentication failed for '{service}'"
        if reason:
            message += f": {reason}"
        super().__init__(message, {"service": service, "reason": reason})


class PermissionError(VoiceAgentError):
    """Raised when permission is denied for an operation."""
    
    def __init__(self, operation: str, resource: str, required_permission: Optional[str] = None):
        message = f"Permission denied for operation '{operation}' on resource '{resource}'"
        if required_permission:
            message += f". Required permission: {required_permission}"
        super().__init__(
            message,
            {
                "operation": operation,
                "resource": resource,
                "required_permission": required_permission
            }
        )


class SDKNotInitializedError(VoiceAgentError):
    """Raised when SDK is used before initialization."""
    
    def __init__(self):
        super().__init__(
            "SDK not initialized. Call VoiceAgentSDK.initialize() first.",
            {"hint": "Ensure you call await sdk.initialize() before using the SDK"}
        )


class DuplicateComponentError(VoiceAgentError):
    """Raised when trying to register a component that already exists."""
    
    def __init__(self, component_type: str, component_name: str):
        super().__init__(
            f"Component '{component_name}' of type '{component_type}' is already registered",
            {
                "component_type": component_type,
                "component_name": component_name
            }
        )