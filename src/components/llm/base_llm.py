"""
Base Large Language Model (LLM) Provider Abstraction

This module defines the abstract base class that all LLM providers must implement,
ensuring a consistent interface across different language model services.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, AsyncIterator, Union, List
from enum import Enum
from dataclasses import dataclass
import asyncio
import logging
import json

logger = logging.getLogger(__name__)


class LLMRole(Enum):
    """Message roles in conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"


class LLMModelType(Enum):
    """Types of LLM models."""
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_4 = "gpt-4"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"
    CLAUDE_3_HAIKU = "claude-3-haiku"
    CLAUDE_3_SONNET = "claude-3-sonnet"
    CLAUDE_3_OPUS = "claude-3-opus"
    CLAUDE_3_5_SONNET = "claude-3.5-sonnet"
    LLAMA_3_8B = "llama-3-8b"
    LLAMA_3_70B = "llama-3-70b"
    CUSTOM = "custom"


@dataclass
class LLMMessage:
    """Represents a message in the conversation."""
    role: LLMRole
    content: str
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format."""
        msg = {
            "role": self.role.value,
            "content": self.content
        }
        if self.name:
            msg["name"] = self.name
        if self.function_call:
            msg["function_call"] = self.function_call
        return msg


@dataclass
class LLMFunction:
    """Represents a function that can be called by the LLM."""
    name: str
    description: str
    parameters: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert function to dictionary format."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }


@dataclass
class LLMResponse:
    """Represents the response from an LLM."""
    content: str
    role: LLMRole = LLMRole.ASSISTANT
    finish_reason: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    usage: Optional[Dict[str, int]] = None
    model: Optional[str] = None
    is_streaming: bool = False
    metadata: Optional[Dict[str, Any]] = None
    
    def __str__(self) -> str:
        return f"LLMResponse(content='{self.content[:100]}...', model={self.model})"


class LLMConfig:
    """Configuration for LLM providers."""
    
    def __init__(
        self,
        model: LLMModelType = LLMModelType.GPT_4O_MINI,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop_sequences: Optional[List[str]] = None,
        system_message: Optional[str] = None,
        enable_streaming: bool = True,
        enable_function_calling: bool = True,
        functions: Optional[List[LLMFunction]] = None,
        max_context_length: int = 4096,
        conversation_memory: int = 10,  # Number of messages to keep in memory
        **kwargs
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop_sequences = stop_sequences or []
        self.system_message = system_message
        self.enable_streaming = enable_streaming
        self.enable_function_calling = enable_function_calling
        self.functions = functions or []
        self.max_context_length = max_context_length
        self.conversation_memory = conversation_memory
        
        # Store any additional provider-specific config
        for key, value in kwargs.items():
            setattr(self, key, value)


class BaseLLMProvider(ABC):
    """
    Abstract base class for all Large Language Model providers.
    
    All LLM implementations must inherit from this class and implement
    the required methods to ensure consistent behavior across providers.
    """
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self._conversation_history: List[LLMMessage] = []
        self._is_streaming = False
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Add system message if provided
        if config.system_message:
            self._conversation_history.append(
                LLMMessage(role=LLMRole.SYSTEM, content=config.system_message)
            )
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of this LLM provider."""
        pass
    
    @property
    @abstractmethod
    def supported_models(self) -> List[LLMModelType]:
        """Return list of models supported by this provider."""
        pass
    
    @property
    @abstractmethod
    def supports_function_calling(self) -> bool:
        """Return whether this provider supports function calling."""
        pass
    
    @property
    @abstractmethod
    def supports_streaming(self) -> bool:
        """Return whether this provider supports streaming responses."""
        pass
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the LLM provider (authenticate, setup connections, etc.)."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources and close connections."""
        pass
    
    @abstractmethod
    async def generate_response(
        self,
        messages: List[LLMMessage],
        functions: Optional[List[LLMFunction]] = None
    ) -> LLMResponse:
        """
        Generate a response to the given messages.
        
        Args:
            messages: List of conversation messages
            functions: Optional list of functions for function calling
            
        Returns:
            LLMResponse with generated content
        """
        pass
    
    @abstractmethod
    async def generate_streaming_response(
        self,
        messages: List[LLMMessage],
        functions: Optional[List[LLMFunction]] = None
    ) -> AsyncIterator[LLMResponse]:
        """
        Generate a streaming response to the given messages.
        
        Args:
            messages: List of conversation messages
            functions: Optional list of functions for function calling
            
        Yields:
            LLMResponse chunks as they become available
        """
        pass
    
    async def chat(self, user_message: str) -> LLMResponse:
        """
        Send a user message and get a response (maintains conversation history).
        
        Args:
            user_message: The user's message
            
        Returns:
            LLMResponse with assistant's reply
        """
        # Add user message to conversation
        self.add_message(LLMRole.USER, user_message)
        
        # Get response
        if self.config.enable_streaming:
            response_content = ""
            async for chunk in self.generate_streaming_response(
                self._get_context_messages(),
                self.config.functions if self.config.enable_function_calling else None
            ):
                response_content += chunk.content
            
            response = LLMResponse(
                content=response_content,
                model=self.config.model.value,
                is_streaming=True
            )
        else:
            response = await self.generate_response(
                self._get_context_messages(),
                self.config.functions if self.config.enable_function_calling else None
            )
        
        # Add assistant response to conversation
        self.add_message(LLMRole.ASSISTANT, response.content)
        
        return response
    
    async def chat_streaming(self, user_message: str) -> AsyncIterator[LLMResponse]:
        """
        Send a user message and get a streaming response.
        
        Args:
            user_message: The user's message
            
        Yields:
            LLMResponse chunks as they become available
        """
        # Add user message to conversation
        self.add_message(LLMRole.USER, user_message)
        
        response_content = ""
        async for chunk in self.generate_streaming_response(
            self._get_context_messages(),
            self.config.functions if self.config.enable_function_calling else None
        ):
            response_content += chunk.content
            yield chunk
        
        # Add complete assistant response to conversation
        self.add_message(LLMRole.ASSISTANT, response_content)
    
    def add_message(self, role: LLMRole, content: str, **kwargs) -> None:
        """Add a message to the conversation history."""
        message = LLMMessage(role=role, content=content, **kwargs)
        self._conversation_history.append(message)
        
        # Trim conversation history if needed
        self._trim_conversation_history()
    
    def clear_conversation(self) -> None:
        """Clear conversation history except system message."""
        system_messages = [
            msg for msg in self._conversation_history 
            if msg.role == LLMRole.SYSTEM
        ]
        self._conversation_history = system_messages
    
    def get_conversation_history(self) -> List[LLMMessage]:
        """Get the current conversation history."""
        return self._conversation_history.copy()
    
    def _get_context_messages(self) -> List[LLMMessage]:
        """Get messages for current context (respecting context length limits)."""
        if not self._conversation_history:
            return []
        
        # Always include system messages
        system_messages = [
            msg for msg in self._conversation_history 
            if msg.role == LLMRole.SYSTEM
        ]
        
        # Get recent non-system messages
        non_system_messages = [
            msg for msg in self._conversation_history 
            if msg.role != LLMRole.SYSTEM
        ]
        
        # Take the most recent messages up to conversation_memory limit
        recent_messages = non_system_messages[-self.config.conversation_memory:]
        
        return system_messages + recent_messages
    
    def _trim_conversation_history(self) -> None:
        """Trim conversation history to stay within limits."""
        if len(self._conversation_history) <= self.config.conversation_memory + 1:  # +1 for system
            return
        
        # Keep system messages and recent messages
        system_messages = [
            msg for msg in self._conversation_history 
            if msg.role == LLMRole.SYSTEM
        ]
        
        non_system_messages = [
            msg for msg in self._conversation_history 
            if msg.role != LLMRole.SYSTEM
        ]
        
        # Keep only the most recent messages
        recent_messages = non_system_messages[-self.config.conversation_memory:]
        
        self._conversation_history = system_messages + recent_messages
    
    def validate_config(self) -> bool:
        """
        Validate the current configuration.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        if self.config.model not in self.supported_models:
            raise ValueError(f"Model {self.config.model} not supported by {self.provider_name}")
        
        if not 0 <= self.config.temperature <= 2:
            raise ValueError("Temperature must be between 0 and 2")
        
        if not 0 <= self.config.top_p <= 1:
            raise ValueError("Top_p must be between 0 and 1")
        
        if self.config.enable_function_calling and not self.supports_function_calling:
            raise ValueError(f"Function calling not supported by {self.provider_name}")
        
        if self.config.enable_streaming and not self.supports_streaming:
            raise ValueError(f"Streaming not supported by {self.provider_name}")
        
        return True
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of the LLM provider.
        
        Returns:
            Dictionary with health status information
        """
        try:
            await self.initialize()
            test_response = await self.generate_response([
                LLMMessage(role=LLMRole.USER, content="Hello, this is a health check.")
            ])
            
            return {
                "status": "healthy",
                "provider": self.provider_name,
                "model": self.config.model.value,
                "response_received": bool(test_response.content)
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": self.provider_name,
                "error": str(e)
            }
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get provider-specific metrics.
        
        Returns:
            Dictionary with metrics data
        """
        return {
            "provider": self.provider_name,
            "model": self.config.model.value,
            "conversation_length": len(self._conversation_history),
            "is_streaming": self._is_streaming,
            "config": {
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "function_calling_enabled": self.config.enable_function_calling,
                "streaming_enabled": self.config.enable_streaming
            }
        }
    
    async def __aenter__(self):
        """Context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.cleanup()


class LLMProviderFactory:
    """Factory for creating LLM provider instances."""
    
    _providers: Dict[str, type] = {}
    
    @classmethod
    def register_provider(cls, name: str, provider_class: type):
        """Register a new LLM provider."""
        cls._providers[name] = provider_class
    
    @classmethod
    def create_provider(cls, name: str, config: LLMConfig) -> BaseLLMProvider:
        """Create an instance of the specified LLM provider."""
        if name not in cls._providers:
            raise ValueError(f"Unknown LLM provider: {name}")
        
        provider_class = cls._providers[name]
        return provider_class(config)
    
    @classmethod
    def list_providers(cls) -> List[str]:
        """List all registered LLM providers."""
        return list(cls._providers.keys())