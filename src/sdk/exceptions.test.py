"""
Unit tests for SDK exceptions module.

Tests all custom exception classes and their behavior.
"""

import pytest
from typing import Dict, Any

from exceptions import (
    VoiceAgentError,
    ConfigurationError,
    ComponentNotFoundError,
    ProviderError,
    ConnectionError,
    ValidationError,
    StateError,
    ResourceError,
    TimeoutError,
    AuthenticationError,
    PermissionError,
    SDKNotInitializedError,
    DuplicateComponentError
)


class TestVoiceAgentError:
    """Test the base VoiceAgentError exception."""
    
    def test_basic_error(self):
        """Test creating a basic error with message only."""
        error = VoiceAgentError("Test error message")
        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert error.details == {}
    
    def test_error_with_details(self):
        """Test creating an error with details."""
        details = {"code": 123, "context": "test"}
        error = VoiceAgentError("Test error", details)
        assert error.message == "Test error"
        assert error.details == details
        assert "Details: {'code': 123, 'context': 'test'}" in str(error)
    
    def test_inheritance(self):
        """Test that VoiceAgentError inherits from Exception."""
        error = VoiceAgentError("Test")
        assert isinstance(error, Exception)
        assert isinstance(error, VoiceAgentError)


class TestConfigurationError:
    """Test ConfigurationError exception."""
    
    def test_configuration_error(self):
        """Test basic configuration error."""
        error = ConfigurationError("Invalid configuration")
        assert isinstance(error, VoiceAgentError)
        assert error.message == "Invalid configuration"
    
    def test_configuration_error_with_details(self):
        """Test configuration error with details."""
        error = ConfigurationError(
            "Missing API key",
            {"provider": "openai", "field": "api_key"}
        )
        assert error.details["provider"] == "openai"
        assert error.details["field"] == "api_key"


class TestComponentNotFoundError:
    """Test ComponentNotFoundError exception."""
    
    def test_component_not_found_basic(self):
        """Test basic component not found error."""
        error = ComponentNotFoundError("stt")
        assert error.message == "Component of type 'stt' not found"
        assert error.details["component_type"] == "stt"
        assert error.details["component_name"] is None
    
    def test_component_not_found_with_name(self):
        """Test component not found error with specific name."""
        error = ComponentNotFoundError("llm", "gpt-4")
        assert error.message == "Component of type 'llm' not found: 'gpt-4'"
        assert error.details["component_type"] == "llm"
        assert error.details["component_name"] == "gpt-4"


class TestProviderError:
    """Test ProviderError exception."""
    
    def test_provider_error_basic(self):
        """Test basic provider error."""
        error = ProviderError("openai", "API request failed")
        assert error.message == "Provider 'openai' error: API request failed"
        assert error.details["provider"] == "openai"
        assert error.details["original_error"] is None
        assert error.original_error is None
    
    def test_provider_error_with_original(self):
        """Test provider error with original exception."""
        original = ValueError("Invalid response")
        error = ProviderError("anthropic", "Processing failed", original)
        assert "Provider 'anthropic' error: Processing failed" in str(error)
        assert error.details["original_error"] == "Invalid response"
        assert error.original_error == original


class TestConnectionError:
    """Test ConnectionError exception."""
    
    def test_connection_error_basic(self):
        """Test basic connection error."""
        error = ConnectionError("LiveKit", "Connection refused")
        assert error.message == "Connection to 'LiveKit' failed: Connection refused"
        assert error.details["service"] == "LiveKit"
        assert error.details["retry_after"] is None
    
    def test_connection_error_with_retry(self):
        """Test connection error with retry information."""
        error = ConnectionError("OpenAI API", "Rate limited", retry_after=60.0)
        assert "Connection to 'OpenAI API' failed" in str(error)
        assert error.details["retry_after"] == 60.0


class TestValidationError:
    """Test ValidationError exception."""
    
    def test_validation_error(self):
        """Test validation error with all parameters."""
        error = ValidationError("temperature", 2.5, "Must be between 0 and 1")
        assert "Validation failed for 'temperature'" in str(error)
        assert error.details["field"] == "temperature"
        assert error.details["value"] == 2.5
        assert error.details["reason"] == "Must be between 0 and 1"


class TestStateError:
    """Test StateError exception."""
    
    def test_state_error(self):
        """Test state error with all parameters."""
        error = StateError("stopped", "running", "start_recording")
        expected_msg = "Invalid state for operation 'start_recording'. Current: stopped, Expected: running"
        assert error.message == expected_msg
        assert error.details["current_state"] == "stopped"
        assert error.details["expected_state"] == "running"
        assert error.details["operation"] == "start_recording"


class TestResourceError:
    """Test ResourceError exception."""
    
    def test_resource_error_basic(self):
        """Test basic resource error."""
        error = ResourceError("memory", "Out of memory")
        assert "Resource error for 'memory': Out of memory" in str(error)
        assert error.details["resource_type"] == "memory"
        assert error.details["limit"] is None
        assert error.details["current"] is None
    
    def test_resource_error_with_limits(self):
        """Test resource error with limit information."""
        error = ResourceError(
            "concurrent_connections",
            "Connection limit exceeded",
            limit=100,
            current=105
        )
        assert error.details["limit"] == 100
        assert error.details["current"] == 105


class TestTimeoutError:
    """Test TimeoutError exception."""
    
    def test_timeout_error(self):
        """Test timeout error."""
        error = TimeoutError("transcription", 30.0)
        assert error.message == "Operation 'transcription' timed out after 30.0 seconds"
        assert error.details["operation"] == "transcription"
        assert error.details["timeout_seconds"] == 30.0


class TestAuthenticationError:
    """Test AuthenticationError exception."""
    
    def test_authentication_error_basic(self):
        """Test basic authentication error."""
        error = AuthenticationError("OpenAI")
        assert error.message == "Authentication failed for 'OpenAI'"
        assert error.details["service"] == "OpenAI"
        assert error.details["reason"] is None
    
    def test_authentication_error_with_reason(self):
        """Test authentication error with reason."""
        error = AuthenticationError("Azure", "Invalid subscription key")
        assert "Authentication failed for 'Azure': Invalid subscription key" in str(error)
        assert error.details["reason"] == "Invalid subscription key"


class TestPermissionError:
    """Test PermissionError exception."""
    
    def test_permission_error_basic(self):
        """Test basic permission error."""
        error = PermissionError("write", "/system/config")
        expected = "Permission denied for operation 'write' on resource '/system/config'"
        assert error.message == expected
        assert error.details["operation"] == "write"
        assert error.details["resource"] == "/system/config"
        assert error.details["required_permission"] is None
    
    def test_permission_error_with_required_permission(self):
        """Test permission error with required permission."""
        error = PermissionError("delete", "user_data", "admin")
        assert "Required permission: admin" in str(error)
        assert error.details["required_permission"] == "admin"


class TestSDKNotInitializedError:
    """Test SDKNotInitializedError exception."""
    
    def test_sdk_not_initialized_error(self):
        """Test SDK not initialized error."""
        error = SDKNotInitializedError()
        assert "SDK not initialized" in str(error)
        assert "Call VoiceAgentSDK.initialize() first" in str(error)
        assert error.details["hint"] == "Ensure you call await sdk.initialize() before using the SDK"


class TestDuplicateComponentError:
    """Test DuplicateComponentError exception."""
    
    def test_duplicate_component_error(self):
        """Test duplicate component error."""
        error = DuplicateComponentError("stt", "openai")
        expected = "Component 'openai' of type 'stt' is already registered"
        assert error.message == expected
        assert error.details["component_type"] == "stt"
        assert error.details["component_name"] == "openai"


class TestExceptionHierarchy:
    """Test the exception hierarchy and relationships."""
    
    def test_all_exceptions_inherit_from_base(self):
        """Test that all custom exceptions inherit from VoiceAgentError."""
        exceptions = [
            ConfigurationError("test"),
            ComponentNotFoundError("test"),
            ProviderError("test", "test"),
            ConnectionError("test", "test"),
            ValidationError("test", "test", "test"),
            StateError("test", "test", "test"),
            ResourceError("test", "test"),
            TimeoutError("test", 1.0),
            AuthenticationError("test"),
            PermissionError("test", "test"),
            SDKNotInitializedError(),
            DuplicateComponentError("test", "test")
        ]
        
        for exc in exceptions:
            assert isinstance(exc, VoiceAgentError)
            assert isinstance(exc, Exception)
    
    def test_exception_catching(self):
        """Test that exceptions can be caught at different levels."""
        # Catch specific exception
        with pytest.raises(ConfigurationError):
            raise ConfigurationError("Test")
        
        # Catch base SDK exception
        with pytest.raises(VoiceAgentError):
            raise ComponentNotFoundError("test")
        
        # Catch general exception
        with pytest.raises(Exception):
            raise ProviderError("test", "test")


class TestExceptionUsagePatterns:
    """Test common usage patterns for exceptions."""
    
    def test_chaining_exceptions(self):
        """Test exception chaining pattern."""
        try:
            # Simulate an operation that fails
            raise ValueError("Original error")
        except ValueError as e:
            # Wrap in SDK exception
            provider_error = ProviderError("test_provider", "Operation failed", e)
            assert provider_error.original_error == e
            assert "Original error" in str(provider_error.details["original_error"])
    
    def test_error_context_building(self):
        """Test building rich error context."""
        # Simulate a complex error scenario
        details = {
            "endpoint": "https://api.example.com",
            "method": "POST",
            "status_code": 401,
            "headers": {"Authorization": "Bearer ***"}
        }
        
        error = AuthenticationError("API Service")
        error.details.update(details)
        
        assert error.details["endpoint"] == "https://api.example.com"
        assert error.details["status_code"] == 401
    
    def test_conditional_error_creation(self):
        """Test conditional error creation based on context."""
        def validate_config(config: Dict[str, Any]):
            errors = []
            
            if "api_key" not in config:
                errors.append(ConfigurationError("Missing API key"))
            
            if "temperature" in config:
                temp = config["temperature"]
                if not 0 <= temp <= 1:
                    errors.append(ValidationError(
                        "temperature",
                        temp,
                        "Must be between 0 and 1"
                    ))
            
            return errors
        
        # Test with invalid config
        errors = validate_config({"temperature": 2.0})
        assert len(errors) == 2
        assert isinstance(errors[0], ConfigurationError)
        assert isinstance(errors[1], ValidationError)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])