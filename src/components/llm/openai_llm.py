"""
OpenAI GPT Large Language Model Provider Implementation

This module implements the OpenAI GPT LLM provider with function calling support
and streaming response capabilities.
"""

import asyncio
import logging
import json
import time
from typing import Optional, Dict, Any, AsyncIterator, List
import openai
from openai import AsyncOpenAI

from .base_llm import (
    BaseLLMProvider, LLMResponse, LLMMessage, LLMFunction, LLMConfig,
    LLMRole, LLMModelType, LLMProviderFactory
)

logger = logging.getLogger(__name__)


class OpenAILLMConfig(LLMConfig):
    """Configuration specific to OpenAI GPT models."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 30.0,
        stream_chunk_size: int = 1024,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.organization = organization
        self.base_url = base_url
        self.max_retries = max_retries
        self.timeout = timeout
        self.stream_chunk_size = stream_chunk_size


class OpenAILLMProvider(BaseLLMProvider):
    """
    OpenAI GPT LLM provider implementation.
    
    Provides language model functionality using OpenAI's GPT models
    with support for function calling and streaming responses.
    """
    
    def __init__(self, config: OpenAILLMConfig):
        super().__init__(config)
        self.config: OpenAILLMConfig = config
        self.client: Optional[AsyncOpenAI] = None
        self._request_count = 0
        self._total_tokens_used = 0
        self._last_request_time = 0
    
    @property
    def provider_name(self) -> str:
        """Return the name of this LLM provider."""
        return "openai_gpt"
    
    @property
    def supported_models(self) -> List[LLMModelType]:
        """Return list of models supported by OpenAI."""
        return [
            LLMModelType.GPT_3_5_TURBO,
            LLMModelType.GPT_4,
            LLMModelType.GPT_4_TURBO,
            LLMModelType.GPT_4O,
            LLMModelType.GPT_4O_MINI,
            LLMModelType.GPT_4_1_MINI  # Latest model - better performance and cost
        ]
    
    @property
    def supports_function_calling(self) -> bool:
        """Return whether this provider supports function calling."""
        return True
    
    @property
    def supports_streaming(self) -> bool:
        """Return whether this provider supports streaming responses."""
        return True
    
    async def initialize(self) -> None:
        """Initialize the OpenAI LLM provider."""
        try:
            api_key = self.config.api_key
            if not api_key:
                import os
                api_key = os.getenv("OPENAI_API_KEY")
            
            if not api_key:
                raise ValueError("OpenAI API key is required")
            
            # Initialize client with configuration
            client_kwargs = {
                "api_key": api_key,
                "max_retries": self.config.max_retries,
                "timeout": self.config.timeout
            }
            
            if self.config.organization:
                client_kwargs["organization"] = self.config.organization
            
            if self.config.base_url:
                client_kwargs["base_url"] = self.config.base_url
            
            self.client = AsyncOpenAI(**client_kwargs)
            
            # Test the connection
            await self._test_connection()
            
            self.logger.info(f"OpenAI LLM provider initialized successfully with model {self.config.model.value}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI LLM: {e}")
            raise
    
    async def cleanup(self) -> None:
        """Clean up resources and close connections."""
        if self.client:
            await self.client.close()
            self.client = None
        
        self.logger.info("OpenAI LLM provider cleanup completed")
    
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
        if not self.client:
            raise RuntimeError("Provider not initialized")
        
        try:
            start_time = time.time()
            
            # Prepare messages for OpenAI format
            openai_messages = [msg.to_dict() for msg in messages]
            
            # Prepare request parameters
            request_params = {
                "model": self.config.model.value,
                "messages": openai_messages,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "frequency_penalty": self.config.frequency_penalty,
                "presence_penalty": self.config.presence_penalty,
                "stream": False
            }
            
            # Add max_tokens if specified
            if self.config.max_tokens:
                request_params["max_tokens"] = self.config.max_tokens
            
            # Add stop sequences if specified
            if self.config.stop_sequences:
                request_params["stop"] = self.config.stop_sequences
            
            # Add functions if provided and supported
            if functions and self.config.enable_function_calling:
                openai_functions = [func.to_dict() for func in functions]
                request_params["tools"] = [
                    {"type": "function", "function": func} for func in openai_functions
                ]
                request_params["tool_choice"] = "auto"
            
            # Make API call
            response = await self.client.chat.completions.create(**request_params)
            
            # Extract response data
            choice = response.choices[0]
            message = choice.message
            
            # Handle function calls
            function_call = None
            if hasattr(message, 'tool_calls') and message.tool_calls:
                tool_call = message.tool_calls[0]
                if tool_call.type == "function":
                    function_call = {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
            
            # Extract usage information
            usage = None
            if hasattr(response, 'usage') and response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
                self._total_tokens_used += usage["total_tokens"]
            
            end_time = time.time()
            processing_time = end_time - start_time
            self._request_count += 1
            self._last_request_time = end_time
            
            # Create response
            result = LLMResponse(
                content=message.content or "",
                role=LLMRole.ASSISTANT,
                finish_reason=choice.finish_reason,
                function_call=function_call,
                usage=usage,
                model=self.config.model.value,
                is_streaming=False,
                metadata={
                    "provider": self.provider_name,
                    "processing_time": processing_time,
                    "request_id": getattr(response, 'id', None)
                }
            )
            
            self.logger.debug(f"Generated response in {processing_time:.3f}s: '{result.content[:50]}...'")
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return LLMResponse(
                content="",
                metadata={"error": str(e), "provider": self.provider_name}
            )
    
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
        if not self.client:
            raise RuntimeError("Provider not initialized")
        
        try:
            start_time = time.time()
            
            # Prepare messages for OpenAI format
            openai_messages = [msg.to_dict() for msg in messages]
            
            # Prepare request parameters
            request_params = {
                "model": self.config.model.value,
                "messages": openai_messages,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "frequency_penalty": self.config.frequency_penalty,
                "presence_penalty": self.config.presence_penalty,
                "stream": True
            }
            
            # Add max_tokens if specified
            if self.config.max_tokens:
                request_params["max_tokens"] = self.config.max_tokens
            
            # Add stop sequences if specified
            if self.config.stop_sequences:
                request_params["stop"] = self.config.stop_sequences
            
            # Add functions if provided and supported
            if functions and self.config.enable_function_calling:
                openai_functions = [func.to_dict() for func in functions]
                request_params["tools"] = [
                    {"type": "function", "function": func} for func in openai_functions
                ]
                request_params["tool_choice"] = "auto"
            
            # Make streaming API call
            stream = await self.client.chat.completions.create(**request_params)
            
            # Process streaming response
            accumulated_content = ""
            function_call_data = {}
            
            async for chunk in stream:
                if not chunk.choices:
                    continue
                
                choice = chunk.choices[0]
                delta = choice.delta
                
                # Handle content
                if hasattr(delta, 'content') and delta.content:
                    accumulated_content += delta.content
                    
                    yield LLMResponse(
                        content=delta.content,
                        role=LLMRole.ASSISTANT,
                        finish_reason=choice.finish_reason,
                        model=self.config.model.value,
                        is_streaming=True,
                        metadata={
                            "provider": self.provider_name,
                            "chunk_type": "content",
                            "accumulated_length": len(accumulated_content)
                        }
                    )
                
                # Handle function calls
                if hasattr(delta, 'tool_calls') and delta.tool_calls:
                    tool_call = delta.tool_calls[0]
                    if tool_call.type == "function":
                        # Accumulate function call data
                        if tool_call.function.name:
                            function_call_data["name"] = tool_call.function.name
                        if tool_call.function.arguments:
                            if "arguments" not in function_call_data:
                                function_call_data["arguments"] = ""
                            function_call_data["arguments"] += tool_call.function.arguments
                        
                        yield LLMResponse(
                            content="",
                            role=LLMRole.ASSISTANT,
                            function_call=function_call_data.copy(),
                            model=self.config.model.value,
                            is_streaming=True,
                            metadata={
                                "provider": self.provider_name,
                                "chunk_type": "function_call"
                            }
                        )
                
                # Handle completion
                if choice.finish_reason:
                    end_time = time.time()
                    processing_time = end_time - start_time
                    self._request_count += 1
                    self._last_request_time = end_time
                    
                    yield LLMResponse(
                        content="",
                        role=LLMRole.ASSISTANT,
                        finish_reason=choice.finish_reason,
                        function_call=function_call_data if function_call_data else None,
                        model=self.config.model.value,
                        is_streaming=True,
                        metadata={
                            "provider": self.provider_name,
                            "chunk_type": "completion",
                            "processing_time": processing_time,
                            "total_content_length": len(accumulated_content)
                        }
                    )
                    break
            
        except Exception as e:
            self.logger.error(f"Error in streaming response: {e}")
            yield LLMResponse(
                content="",
                metadata={"error": str(e), "provider": self.provider_name, "chunk_type": "error"}
            )
    
    async def _test_connection(self) -> None:
        """Test the OpenAI API connection."""
        try:
            # Simple test with minimal request
            response = await self.client.chat.completions.create(
                model=self.config.model.value,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=1,
                temperature=0
            )
            
            if not response.choices:
                raise Exception("No response received from OpenAI API")
                
        except Exception as e:
            raise Exception(f"OpenAI API connection test failed: {e}")
    
    def _convert_messages_to_openai_format(self, messages: List[LLMMessage]) -> List[Dict[str, Any]]:
        """Convert internal message format to OpenAI format."""
        openai_messages = []
        
        for message in messages:
            openai_msg = {
                "role": message.role.value,
                "content": message.content
            }
            
            if message.name:
                openai_msg["name"] = message.name
            
            if message.function_call:
                openai_msg["function_call"] = message.function_call
            
            openai_messages.append(openai_msg)
        
        return openai_messages
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of the OpenAI LLM provider.
        
        Returns:
            Dictionary with health status information
        """
        try:
            if not self.client:
                return {
                    "status": "unhealthy",
                    "provider": self.provider_name,
                    "error": "Client not initialized"
                }
            
            # Test API connectivity
            start_time = time.time()
            await self._test_connection()
            response_time = time.time() - start_time
            
            return {
                "status": "healthy",
                "provider": self.provider_name,
                "model": self.config.model.value,
                "response_time": response_time,
                "request_count": self._request_count,
                "total_tokens_used": self._total_tokens_used,
                "config": {
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                    "function_calling_enabled": self.config.enable_function_calling,
                    "streaming_enabled": self.config.enable_streaming
                }
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
        base_metrics = super().get_metrics()
        base_metrics.update({
            "request_count": self._request_count,
            "total_tokens_used": self._total_tokens_used,
            "last_request_time": self._last_request_time,
            "average_tokens_per_request": (
                self._total_tokens_used / self._request_count 
                if self._request_count > 0 else 0
            ),
            "api_config": {
                "max_retries": self.config.max_retries,
                "timeout": self.config.timeout,
                "organization": self.config.organization is not None,
                "base_url": self.config.base_url
            }
        })
        return base_metrics


# Register the provider
LLMProviderFactory.register_provider("openai", OpenAILLMProvider)