"""
Local Large Language Model Provider Implementation

This module implements support for local LLM models including Llama models
and custom endpoints using OpenAI-compatible APIs.
"""

import asyncio
import logging
import json
import time
from typing import Optional, Dict, Any, AsyncIterator, List, Union
import httpx

from .base_llm import (
    BaseLLMProvider, LLMResponse, LLMMessage, LLMFunction, LLMConfig,
    LLMRole, LLMModelType, LLMProviderFactory
)

logger = logging.getLogger(__name__)


class LocalLLMConfig(LLMConfig):
    """Configuration specific to local LLM models."""
    
    def __init__(
        self,
        endpoint_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,  # Optional for local models
        model_name: str = "llama-2-7b-chat",
        model_type: str = "llama",  # llama, custom, openai-compatible
        custom_headers: Optional[Dict[str, str]] = None,
        trust_remote_code: bool = False,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        max_retries: int = 3,
        timeout: float = 120.0,  # Longer timeout for local models
        **kwargs
    ):
        super().__init__(**kwargs)
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.model_name = model_name
        self.model_type = model_type
        self.custom_headers = custom_headers or {}
        self.trust_remote_code = trust_remote_code
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.max_retries = max_retries
        self.timeout = timeout


class LocalLLMProvider(BaseLLMProvider):
    """
    Local LLM provider implementation.
    
    Supports local LLM models including Llama models and custom endpoints
    with OpenAI-compatible APIs.
    """
    
    def __init__(self, config: LocalLLMConfig):
        super().__init__(config)
        self.config: LocalLLMConfig = config
        self.client: Optional[httpx.AsyncClient] = None
        self._request_count = 0
        self._total_tokens_used = 0
        self._last_request_time = 0
        self._model_info: Optional[Dict[str, Any]] = None
    
    @property
    def provider_name(self) -> str:
        """Return the name of this LLM provider."""
        return f"local_{self.config.model_type}"
    
    @property
    def supported_models(self) -> List[LLMModelType]:
        """Return list of models supported by local provider."""
        return [
            LLMModelType.LLAMA_3_8B,
            LLMModelType.LLAMA_3_70B,
            LLMModelType.CUSTOM
        ]
    
    @property
    def supports_function_calling(self) -> bool:
        """Return whether this provider supports function calling."""
        # Function calling support depends on the specific model and endpoint
        return self.config.enable_function_calling and self._supports_tools()
    
    @property
    def supports_streaming(self) -> bool:
        """Return whether this provider supports streaming responses."""
        return True
    
    async def initialize(self) -> None:
        """Initialize the local LLM provider."""
        try:
            # Build headers
            headers = {"Content-Type": "application/json"}
            
            # Add API key if provided
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            
            # Add custom headers
            headers.update(self.config.custom_headers)
            
            # Initialize HTTP client
            self.client = httpx.AsyncClient(
                base_url=self.config.endpoint_url,
                headers=headers,
                timeout=self.config.timeout
            )
            
            # Test connection and get model info
            await self._test_connection()
            await self._get_model_info()
            
            self.logger.info(f"Local LLM provider initialized successfully with model {self.config.model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize local LLM: {e}")
            raise
    
    async def cleanup(self) -> None:
        """Clean up resources and close connections."""
        if self.client:
            await self.client.aclose()
            self.client = None
        
        self.logger.info("Local LLM provider cleanup completed")
    
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
            
            # Convert messages based on model type
            formatted_messages = self._format_messages(messages)
            
            # Prepare request parameters
            if self.config.model_type == "llama":
                request_params = self._build_llama_request(formatted_messages, functions)
            else:
                request_params = self._build_openai_compatible_request(formatted_messages, functions)
            
            # Make API call
            response = await self.client.post("/v1/chat/completions", json=request_params)
            response.raise_for_status()
            
            # Parse response
            result_data = response.json()
            
            # Extract response content (OpenAI-compatible format)
            if "choices" in result_data and result_data["choices"]:
                choice = result_data["choices"][0]
                message = choice.get("message", {})
                content = message.get("content", "")
                finish_reason = choice.get("finish_reason")
                
                # Handle function calls
                function_call = None
                if "function_call" in message:
                    function_call = message["function_call"]
                elif "tool_calls" in message and message["tool_calls"]:
                    # Convert tool calls to function call format
                    tool_call = message["tool_calls"][0]
                    if tool_call.get("type") == "function":
                        function_call = {
                            "name": tool_call["function"]["name"],
                            "arguments": tool_call["function"]["arguments"]
                        }
            else:
                content = ""
                finish_reason = "error"
                function_call = None
            
            # Extract usage information
            usage = None
            if "usage" in result_data:
                usage_data = result_data["usage"]
                usage = {
                    "prompt_tokens": usage_data.get("prompt_tokens", 0),
                    "completion_tokens": usage_data.get("completion_tokens", 0),
                    "total_tokens": usage_data.get("total_tokens", 0)
                }
                self._total_tokens_used += usage["total_tokens"]
            
            end_time = time.time()
            processing_time = end_time - start_time
            self._request_count += 1
            self._last_request_time = end_time
            
            # Create response
            result = LLMResponse(
                content=content,
                role=LLMRole.ASSISTANT,
                finish_reason=finish_reason,
                function_call=function_call,
                usage=usage,
                model=self.config.model_name,
                is_streaming=False,
                metadata={
                    "provider": self.provider_name,
                    "processing_time": processing_time,
                    "model_type": self.config.model_type,
                    "endpoint": self.config.endpoint_url
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
            
            # Convert messages based on model type
            formatted_messages = self._format_messages(messages)
            
            # Prepare request parameters with streaming
            if self.config.model_type == "llama":
                request_params = self._build_llama_request(formatted_messages, functions, stream=True)
            else:
                request_params = self._build_openai_compatible_request(formatted_messages, functions, stream=True)
            
            # Make streaming API call
            async with self.client.stream("POST", "/v1/chat/completions", json=request_params) as response:
                response.raise_for_status()
                
                accumulated_content = ""
                function_call_data = {}
                
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix
                        
                        if data_str == "[DONE]":
                            break
                        
                        try:
                            chunk_data = json.loads(data_str)
                            
                            if "choices" in chunk_data and chunk_data["choices"]:
                                choice = chunk_data["choices"][0]
                                delta = choice.get("delta", {})
                                
                                # Handle content
                                if "content" in delta and delta["content"]:
                                    content_chunk = delta["content"]
                                    accumulated_content += content_chunk
                                    
                                    yield LLMResponse(
                                        content=content_chunk,
                                        role=LLMRole.ASSISTANT,
                                        model=self.config.model_name,
                                        is_streaming=True,
                                        metadata={
                                            "provider": self.provider_name,
                                            "chunk_type": "content",
                                            "accumulated_length": len(accumulated_content)
                                        }
                                    )
                                
                                # Handle function calls
                                if "function_call" in delta:
                                    fc = delta["function_call"]
                                    if "name" in fc:
                                        function_call_data["name"] = fc["name"]
                                    if "arguments" in fc:
                                        if "arguments" not in function_call_data:
                                            function_call_data["arguments"] = ""
                                        function_call_data["arguments"] += fc["arguments"]
                                
                                # Handle completion
                                if choice.get("finish_reason"):
                                    end_time = time.time()
                                    processing_time = end_time - start_time
                                    self._request_count += 1
                                    self._last_request_time = end_time
                                    
                                    yield LLMResponse(
                                        content="",
                                        role=LLMRole.ASSISTANT,
                                        finish_reason=choice["finish_reason"],
                                        function_call=function_call_data if function_call_data else None,
                                        model=self.config.model_name,
                                        is_streaming=True,
                                        metadata={
                                            "provider": self.provider_name,
                                            "chunk_type": "completion",
                                            "processing_time": processing_time,
                                            "total_content_length": len(accumulated_content)
                                        }
                                    )
                                    break
                        
                        except json.JSONDecodeError:
                            continue
            
        except Exception as e:
            self.logger.error(f"Error in streaming response: {e}")
            yield LLMResponse(
                content="",
                metadata={"error": str(e), "provider": self.provider_name, "chunk_type": "error"}
            )
    
    def _format_messages(self, messages: List[LLMMessage]) -> List[Dict[str, Any]]:
        """Format messages based on model type."""
        if self.config.model_type == "llama":
            return self._format_llama_messages(messages)
        else:
            return self._format_openai_compatible_messages(messages)
    
    def _format_llama_messages(self, messages: List[LLMMessage]) -> List[Dict[str, Any]]:
        """Format messages for Llama models."""
        formatted_messages = []
        
        for message in messages:
            # Llama models typically use specific formatting
            if message.role == LLMRole.SYSTEM:
                role = "system"
            elif message.role == LLMRole.USER:
                role = "user"
            elif message.role == LLMRole.ASSISTANT:
                role = "assistant"
            else:
                role = "user"  # Default fallback
            
            formatted_msg = {
                "role": role,
                "content": message.content
            }
            
            # Handle function calls if supported
            if message.function_call:
                formatted_msg["function_call"] = message.function_call
            
            formatted_messages.append(formatted_msg)
        
        return formatted_messages
    
    def _format_openai_compatible_messages(self, messages: List[LLMMessage]) -> List[Dict[str, Any]]:
        """Format messages for OpenAI-compatible endpoints."""
        formatted_messages = []
        
        for message in messages:
            formatted_msg = {
                "role": message.role.value,
                "content": message.content
            }
            
            if message.name:
                formatted_msg["name"] = message.name
            
            if message.function_call:
                formatted_msg["function_call"] = message.function_call
            
            formatted_messages.append(formatted_msg)
        
        return formatted_messages
    
    def _build_llama_request(
        self, 
        messages: List[Dict[str, Any]], 
        functions: Optional[List[LLMFunction]] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """Build request parameters for Llama models."""
        request_params = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens or 2048,
            "stream": stream
        }
        
        # Add Llama-specific parameters
        if self.config.top_p < 1.0:
            request_params["top_p"] = self.config.top_p
        
        if self.config.stop_sequences:
            request_params["stop"] = self.config.stop_sequences
        
        # Add functions if supported
        if functions and self.supports_function_calling:
            request_params["functions"] = [func.to_dict() for func in functions]
            request_params["function_call"] = "auto"
        
        return request_params
    
    def _build_openai_compatible_request(
        self, 
        messages: List[Dict[str, Any]], 
        functions: Optional[List[LLMFunction]] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """Build request parameters for OpenAI-compatible endpoints."""
        request_params = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": self.config.temperature,
            "stream": stream
        }
        
        # Add optional parameters
        if self.config.max_tokens:
            request_params["max_tokens"] = self.config.max_tokens
        
        if self.config.top_p < 1.0:
            request_params["top_p"] = self.config.top_p
        
        if self.config.frequency_penalty != 0:
            request_params["frequency_penalty"] = self.config.frequency_penalty
        
        if self.config.presence_penalty != 0:
            request_params["presence_penalty"] = self.config.presence_penalty
        
        if self.config.stop_sequences:
            request_params["stop"] = self.config.stop_sequences
        
        # Add functions if supported
        if functions and self.supports_function_calling:
            if self._supports_tools():
                # Use tools format for newer endpoints
                request_params["tools"] = [
                    {"type": "function", "function": func.to_dict()} 
                    for func in functions
                ]
                request_params["tool_choice"] = "auto"
            else:
                # Use functions format for older endpoints
                request_params["functions"] = [func.to_dict() for func in functions]
                request_params["function_call"] = "auto"
        
        return request_params
    
    def _supports_tools(self) -> bool:
        """Check if the endpoint supports tools format."""
        if self._model_info:
            return self._model_info.get("supports_tools", False)
        
        # Default assumption based on model type
        return self.config.model_type in ["openai-compatible"]
    
    async def _test_connection(self) -> None:
        """Test the local LLM endpoint connection."""
        try:
            # Try to get models list or health check
            response = await self.client.get("/v1/models")
            
            if response.status_code == 200:
                return
            elif response.status_code == 404:
                # Try alternative health check endpoint
                response = await self.client.get("/health")
                if response.status_code == 200:
                    return
            
            # If we get here, try a minimal completion request
            test_params = {
                "model": self.config.model_name,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 5
            }
            
            response = await self.client.post("/v1/chat/completions", json=test_params)
            if response.status_code in [200, 400]:  # 400 might be due to minimal request
                return
            
            response.raise_for_status()
            
        except Exception as e:
            raise Exception(f"Local LLM endpoint connection test failed: {e}")
    
    async def _get_model_info(self) -> None:
        """Get information about the model and endpoint capabilities."""
        try:
            # Try to get model information
            response = await self.client.get("/v1/models")
            
            if response.status_code == 200:
                models_data = response.json()
                
                # Find our specific model
                if "data" in models_data:
                    for model in models_data["data"]:
                        if model.get("id") == self.config.model_name:
                            self._model_info = {
                                "id": model.get("id"),
                                "object": model.get("object"),
                                "created": model.get("created"),
                                "owned_by": model.get("owned_by"),
                                "supports_tools": "tool" in str(model).lower(),
                                "supports_functions": "function" in str(model).lower()
                            }
                            break
                
                if not self._model_info:
                    # If specific model not found, use default info
                    self._model_info = {
                        "id": self.config.model_name,
                        "supports_tools": False,
                        "supports_functions": False
                    }
            
        except Exception as e:
            self.logger.warning(f"Could not get model info: {e}")
            self._model_info = {
                "id": self.config.model_name,
                "supports_tools": False,
                "supports_functions": False
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of the local LLM provider.
        
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
            
            # Test endpoint connectivity
            start_time = time.time()
            await self._test_connection()
            response_time = time.time() - start_time
            
            return {
                "status": "healthy",
                "provider": self.provider_name,
                "model": self.config.model_name,
                "model_type": self.config.model_type,
                "endpoint": self.config.endpoint_url,
                "response_time": response_time,
                "request_count": self._request_count,
                "total_tokens_used": self._total_tokens_used,
                "model_info": self._model_info,
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
            "model_name": self.config.model_name,
            "model_type": self.config.model_type,
            "endpoint": self.config.endpoint_url,
            "model_info": self._model_info,
            "local_config": {
                "trust_remote_code": self.config.trust_remote_code,
                "load_in_8bit": self.config.load_in_8bit,
                "load_in_4bit": self.config.load_in_4bit,
                "device_map": self.config.device_map,
                "torch_dtype": self.config.torch_dtype
            }
        })
        return base_metrics


# Register the provider
LLMProviderFactory.register_provider("local", LocalLLMProvider)