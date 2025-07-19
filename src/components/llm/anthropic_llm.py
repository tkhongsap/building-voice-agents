"""
Anthropic Claude Large Language Model Provider Implementation

This module implements the Anthropic Claude LLM provider with conversation
capabilities and tool use (function calling) support.
"""

import asyncio
import logging
import json
import time
from typing import Optional, Dict, Any, AsyncIterator, List
import httpx

from .base_llm import (
    BaseLLMProvider, LLMResponse, LLMMessage, LLMFunction, LLMConfig,
    LLMRole, LLMModelType, LLMProviderFactory
)

logger = logging.getLogger(__name__)


class AnthropicLLMConfig(LLMConfig):
    """Configuration specific to Anthropic Claude models."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.anthropic.com",
        anthropic_version: str = "2023-06-01",
        max_retries: int = 3,
        timeout: float = 60.0,
        stream_chunk_size: int = 1024,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.base_url = base_url
        self.anthropic_version = anthropic_version
        self.max_retries = max_retries
        self.timeout = timeout
        self.stream_chunk_size = stream_chunk_size


class AnthropicLLMProvider(BaseLLMProvider):
    """
    Anthropic Claude LLM provider implementation.
    
    Provides language model functionality using Anthropic's Claude models
    with support for conversation and tool use capabilities.
    """
    
    def __init__(self, config: AnthropicLLMConfig):
        super().__init__(config)
        self.config: AnthropicLLMConfig = config
        self.client: Optional[httpx.AsyncClient] = None
        self._request_count = 0
        self._total_tokens_used = 0
        self._last_request_time = 0
    
    @property
    def provider_name(self) -> str:
        """Return the name of this LLM provider."""
        return "anthropic_claude"
    
    @property
    def supported_models(self) -> List[LLMModelType]:
        """Return list of models supported by Anthropic."""
        return [
            LLMModelType.CLAUDE_3_HAIKU,
            LLMModelType.CLAUDE_3_SONNET,
            LLMModelType.CLAUDE_3_OPUS,
            LLMModelType.CLAUDE_3_5_SONNET
        ]
    
    @property
    def supports_function_calling(self) -> bool:
        """Return whether this provider supports function calling."""
        return True  # Claude supports tool use
    
    @property
    def supports_streaming(self) -> bool:
        """Return whether this provider supports streaming responses."""
        return True
    
    async def initialize(self) -> None:
        """Initialize the Anthropic LLM provider."""
        try:
            api_key = self.config.api_key
            if not api_key:
                import os
                api_key = os.getenv("ANTHROPIC_API_KEY")
            
            if not api_key:
                raise ValueError("Anthropic API key is required")
            
            # Initialize client with configuration
            headers = {
                "x-api-key": api_key,
                "anthropic-version": self.config.anthropic_version,
                "content-type": "application/json"
            }
            
            self.client = httpx.AsyncClient(
                base_url=self.config.base_url,
                headers=headers,
                timeout=self.config.timeout
            )
            
            # Test the connection
            await self._test_connection()
            
            self.logger.info(f"Anthropic LLM provider initialized successfully with model {self.config.model.value}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Anthropic LLM: {e}")
            raise
    
    async def cleanup(self) -> None:
        """Clean up resources and close connections."""
        if self.client:
            await self.client.aclose()
            self.client = None
        
        self.logger.info("Anthropic LLM provider cleanup completed")
    
    async def generate_response(
        self,
        messages: List[LLMMessage],
        functions: Optional[List[LLMFunction]] = None
    ) -> LLMResponse:
        """
        Generate a response to the given messages.
        
        Args:
            messages: List of conversation messages
            functions: Optional list of functions for tool use
            
        Returns:
            LLMResponse with generated content
        """
        if not self.client:
            raise RuntimeError("Provider not initialized")
        
        try:
            start_time = time.time()
            
            # Convert messages to Anthropic format
            anthropic_messages = self._convert_messages_to_anthropic_format(messages)
            
            # Extract system message if present
            system_message = ""
            if anthropic_messages and anthropic_messages[0]["role"] == "system":
                system_message = anthropic_messages[0]["content"]
                anthropic_messages = anthropic_messages[1:]
            
            # Prepare request parameters
            request_params = {
                "model": self._get_anthropic_model_name(),
                "messages": anthropic_messages,
                "temperature": self.config.temperature,
                "stream": False
            }
            
            # Add system message if present
            if system_message:
                request_params["system"] = system_message
            
            # Add max_tokens (required for Anthropic)
            request_params["max_tokens"] = self.config.max_tokens or 4000
            
            # Add top_p if specified
            if self.config.top_p < 1.0:
                request_params["top_p"] = self.config.top_p
            
            # Add stop sequences if specified
            if self.config.stop_sequences:
                request_params["stop_sequences"] = self.config.stop_sequences
            
            # Add tools if provided and supported
            if functions and self.config.enable_function_calling:
                anthropic_tools = self._convert_functions_to_anthropic_tools(functions)
                request_params["tools"] = anthropic_tools
            
            # Make API call
            response = await self.client.post("/v1/messages", json=request_params)
            response.raise_for_status()
            
            # Parse response
            result_data = response.json()
            
            # Extract response content
            content = ""
            tool_calls = []
            
            for content_block in result_data.get("content", []):
                if content_block.get("type") == "text":
                    content += content_block.get("text", "")
                elif content_block.get("type") == "tool_use":
                    tool_calls.append({
                        "name": content_block.get("name"),
                        "arguments": content_block.get("input", {}),
                        "id": content_block.get("id")
                    })
            
            # Handle tool calls
            function_call = None
            if tool_calls:
                # For compatibility, use the first tool call as function_call
                function_call = {
                    "name": tool_calls[0]["name"],
                    "arguments": json.dumps(tool_calls[0]["arguments"])
                }
            
            # Extract usage information
            usage = None
            if "usage" in result_data:
                usage_data = result_data["usage"]
                usage = {
                    "prompt_tokens": usage_data.get("input_tokens", 0),
                    "completion_tokens": usage_data.get("output_tokens", 0),
                    "total_tokens": usage_data.get("input_tokens", 0) + usage_data.get("output_tokens", 0)
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
                finish_reason=result_data.get("stop_reason"),
                function_call=function_call,
                usage=usage,
                model=self.config.model.value,
                is_streaming=False,
                metadata={
                    "provider": self.provider_name,
                    "processing_time": processing_time,
                    "request_id": result_data.get("id"),
                    "tool_calls": tool_calls
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
            functions: Optional list of functions for tool use
            
        Yields:
            LLMResponse chunks as they become available
        """
        if not self.client:
            raise RuntimeError("Provider not initialized")
        
        try:
            start_time = time.time()
            
            # Convert messages to Anthropic format
            anthropic_messages = self._convert_messages_to_anthropic_format(messages)
            
            # Extract system message if present
            system_message = ""
            if anthropic_messages and anthropic_messages[0]["role"] == "system":
                system_message = anthropic_messages[0]["content"]
                anthropic_messages = anthropic_messages[1:]
            
            # Prepare request parameters
            request_params = {
                "model": self._get_anthropic_model_name(),
                "messages": anthropic_messages,
                "temperature": self.config.temperature,
                "stream": True
            }
            
            # Add system message if present
            if system_message:
                request_params["system"] = system_message
            
            # Add max_tokens (required for Anthropic)
            request_params["max_tokens"] = self.config.max_tokens or 4000
            
            # Add top_p if specified
            if self.config.top_p < 1.0:
                request_params["top_p"] = self.config.top_p
            
            # Add stop sequences if specified
            if self.config.stop_sequences:
                request_params["stop_sequences"] = self.config.stop_sequences
            
            # Add tools if provided and supported
            if functions and self.config.enable_function_calling:
                anthropic_tools = self._convert_functions_to_anthropic_tools(functions)
                request_params["tools"] = anthropic_tools
            
            # Make streaming API call
            async with self.client.stream("POST", "/v1/messages", json=request_params) as response:
                response.raise_for_status()
                
                accumulated_content = ""
                tool_calls = []
                
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix
                        
                        if data_str == "[DONE]":
                            break
                        
                        try:
                            chunk_data = json.loads(data_str)
                            
                            if chunk_data.get("type") == "content_block_delta":
                                delta = chunk_data.get("delta", {})
                                
                                if delta.get("type") == "text_delta":
                                    text_chunk = delta.get("text", "")
                                    accumulated_content += text_chunk
                                    
                                    yield LLMResponse(
                                        content=text_chunk,
                                        role=LLMRole.ASSISTANT,
                                        model=self.config.model.value,
                                        is_streaming=True,
                                        metadata={
                                            "provider": self.provider_name,
                                            "chunk_type": "content",
                                            "accumulated_length": len(accumulated_content)
                                        }
                                    )
                            
                            elif chunk_data.get("type") == "content_block_start":
                                content_block = chunk_data.get("content_block", {})
                                if content_block.get("type") == "tool_use":
                                    tool_calls.append({
                                        "name": content_block.get("name"),
                                        "arguments": {},
                                        "id": content_block.get("id")
                                    })
                            
                            elif chunk_data.get("type") == "message_stop":
                                end_time = time.time()
                                processing_time = end_time - start_time
                                self._request_count += 1
                                self._last_request_time = end_time
                                
                                # Handle tool calls in final response
                                function_call = None
                                if tool_calls:
                                    function_call = {
                                        "name": tool_calls[0]["name"],
                                        "arguments": json.dumps(tool_calls[0]["arguments"])
                                    }
                                
                                yield LLMResponse(
                                    content="",
                                    role=LLMRole.ASSISTANT,
                                    finish_reason=chunk_data.get("stop_reason"),
                                    function_call=function_call,
                                    model=self.config.model.value,
                                    is_streaming=True,
                                    metadata={
                                        "provider": self.provider_name,
                                        "chunk_type": "completion",
                                        "processing_time": processing_time,
                                        "total_content_length": len(accumulated_content),
                                        "tool_calls": tool_calls
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
    
    def _convert_messages_to_anthropic_format(self, messages: List[LLMMessage]) -> List[Dict[str, Any]]:
        """Convert internal message format to Anthropic format."""
        anthropic_messages = []
        
        for message in messages:
            anthropic_msg = {
                "role": "user" if message.role == LLMRole.USER else 
                       "assistant" if message.role == LLMRole.ASSISTANT else 
                       "system",
                "content": message.content
            }
            
            # Handle function calls
            if message.function_call:
                # Convert function call to tool use format
                tool_use = {
                    "type": "tool_use",
                    "id": f"tool_{int(time.time() * 1000)}",
                    "name": message.function_call["name"],
                    "input": json.loads(message.function_call.get("arguments", "{}"))
                }
                
                # For function call messages, content should be array format
                if isinstance(anthropic_msg["content"], str):
                    anthropic_msg["content"] = [
                        {"type": "text", "text": anthropic_msg["content"]},
                        tool_use
                    ]
                else:
                    anthropic_msg["content"].append(tool_use)
            
            anthropic_messages.append(anthropic_msg)
        
        return anthropic_messages
    
    def _convert_functions_to_anthropic_tools(self, functions: List[LLMFunction]) -> List[Dict[str, Any]]:
        """Convert function definitions to Anthropic tools format."""
        tools = []
        
        for function in functions:
            tool = {
                "name": function.name,
                "description": function.description,
                "input_schema": function.parameters
            }
            tools.append(tool)
        
        return tools
    
    def _get_anthropic_model_name(self) -> str:
        """Get the Anthropic model name from the config."""
        model_mapping = {
            LLMModelType.CLAUDE_3_HAIKU: "claude-3-haiku-20240307",
            LLMModelType.CLAUDE_3_SONNET: "claude-3-sonnet-20240229",
            LLMModelType.CLAUDE_3_OPUS: "claude-3-opus-20240229",
            LLMModelType.CLAUDE_3_5_SONNET: "claude-3-5-sonnet-20241022"
        }
        
        return model_mapping.get(self.config.model, "claude-3-sonnet-20240229")
    
    async def _test_connection(self) -> None:
        """Test the Anthropic API connection."""
        try:
            # Simple test with minimal request
            test_params = {
                "model": self._get_anthropic_model_name(),
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 10
            }
            
            response = await self.client.post("/v1/messages", json=test_params)
            
            if response.status_code in [200, 400]:  # 400 might be due to minimal request
                return
            else:
                response.raise_for_status()
                
        except Exception as e:
            raise Exception(f"Anthropic API connection test failed: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of the Anthropic LLM provider.
        
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
                    "streaming_enabled": self.config.enable_streaming,
                    "anthropic_version": self.config.anthropic_version
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
            "anthropic_model": self._get_anthropic_model_name(),
            "anthropic_version": self.config.anthropic_version,
            "api_config": {
                "timeout": self.config.timeout,
                "max_retries": self.config.max_retries,
                "base_url": self.config.base_url
            }
        })
        return base_metrics


# Register the provider
LLMProviderFactory.register_provider("anthropic", AnthropicLLMProvider)