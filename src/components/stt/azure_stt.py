"""
Azure Speech Services STT Provider Implementation

This module implements the Azure Speech Services STT provider with real-time transcription
and continuous recognition capabilities.
"""

import asyncio
import logging
import json
import time
import uuid
from typing import Optional, Dict, Any, AsyncIterator, List
import websockets
import httpx

from .base_stt import (
    BaseSTTProvider, STTResult, STTConfig, STTLanguage, STTQuality,
    STTProviderFactory
)

logger = logging.getLogger(__name__)


class AzureSTTConfig(STTConfig):
    """Configuration specific to Azure Speech Services STT."""
    
    def __init__(
        self,
        subscription_key: Optional[str] = None,
        region: str = "eastus",
        endpoint: Optional[str] = None,
        recognition_mode: str = "conversation",  # conversation, dictation, interactive
        output_format: str = "detailed",  # simple, detailed
        profanity_option: str = "masked",  # masked, removed, raw
        word_level_timestamps: bool = True,
        enable_dictation: bool = False,
        phrase_list: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.subscription_key = subscription_key
        self.region = region
        self.endpoint = endpoint or f"https://{region}.stt.speech.microsoft.com"
        self.recognition_mode = recognition_mode
        self.output_format = output_format
        self.profanity_option = profanity_option
        self.word_level_timestamps = word_level_timestamps
        self.enable_dictation = enable_dictation
        self.phrase_list = phrase_list or []


class AzureSTTProvider(BaseSTTProvider):
    """
    Azure Speech Services STT provider implementation.
    
    Provides speech-to-text functionality using Azure Speech Services
    with support for real-time transcription and continuous recognition.
    """
    
    def __init__(self, config: AzureSTTConfig):
        super().__init__(config)
        self.config: AzureSTTConfig = config
        self.client: Optional[httpx.AsyncClient] = None
        self._websocket: Optional[websockets.WebSocketServerProtocol] = None
        self._streaming_active = False
        self._connection_id = None
        self._request_id = None
        self._recognition_results = asyncio.Queue()
        self._websocket_task: Optional[asyncio.Task] = None
    
    @property
    def provider_name(self) -> str:
        """Return the name of this STT provider."""
        return "azure_speech"
    
    @property
    def supported_languages(self) -> List[STTLanguage]:
        """Return list of languages supported by Azure Speech."""
        return [
            STTLanguage.AUTO,
            STTLanguage.ENGLISH,
            STTLanguage.SPANISH,
            STTLanguage.FRENCH,
            STTLanguage.GERMAN,
            STTLanguage.ITALIAN,
            STTLanguage.PORTUGUESE,
            STTLanguage.RUSSIAN,
            STTLanguage.JAPANESE,
            STTLanguage.KOREAN,
            STTLanguage.CHINESE
        ]
    
    @property
    def supported_sample_rates(self) -> List[int]:
        """Return list of supported audio sample rates."""
        return [8000, 16000, 24000, 48000]
    
    async def initialize(self) -> None:
        """Initialize the Azure STT provider."""
        try:
            subscription_key = self.config.subscription_key
            if not subscription_key:
                import os
                subscription_key = os.getenv("AZURE_SPEECH_KEY")
            
            if not subscription_key:
                raise ValueError("Azure Speech subscription key is required")
            
            # Initialize HTTP client for REST API calls
            headers = {
                "Ocp-Apim-Subscription-Key": subscription_key,
                "Content-Type": "application/json"
            }
            
            self.client = httpx.AsyncClient(
                base_url=self.config.endpoint,
                headers=headers,
                timeout=30.0
            )
            
            # Test the connection
            await self._test_connection()
            
            self.logger.info("Azure STT provider initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Azure STT: {e}")
            raise
    
    async def cleanup(self) -> None:
        """Clean up resources and close connections."""
        await self.stop_streaming()
        
        if self.client:
            await self.client.aclose()
            self.client = None
        
        self.logger.info("Azure STT provider cleanup completed")
    
    async def transcribe_audio(self, audio_data: bytes) -> STTResult:
        """
        Transcribe audio data using Azure Speech Services.
        
        Args:
            audio_data: Raw audio bytes
            
        Returns:
            STTResult with transcription
        """
        if not self.client:
            raise RuntimeError("Provider not initialized")
        
        try:
            start_time = time.time()
            
            # Prepare headers for audio upload
            headers = {
                "Content-Type": f"audio/wav; codecs=audio/pcm; samplerate={self.config.sample_rate}",
                "Accept": "application/json"
            }
            
            # Prepare URL parameters
            params = {
                "language": self._get_azure_language_code(),
                "format": self.config.output_format,
                "profanity": self.config.profanity_option
            }
            
            if self.config.phrase_list:
                params["phrases"] = ",".join(self.config.phrase_list)
            
            # Make API request
            response = await self.client.post(
                "/speech/recognition/conversation/cognitiveservices/v1",
                content=audio_data,
                headers=headers,
                params=params
            )
            response.raise_for_status()
            
            # Parse response
            result_data = response.json()
            
            # Extract transcription results
            if result_data.get("RecognitionStatus") == "Success":
                text = result_data.get("DisplayText", "")
                confidence = result_data.get("Confidence", 0.0)
                
                # Extract detailed results if available
                alternatives = []
                if "NBest" in result_data:
                    for alternative in result_data["NBest"][:self.config.max_alternatives]:
                        alt_data = {
                            "text": alternative.get("Display", ""),
                            "confidence": alternative.get("Confidence", 0.0),
                            "lexical": alternative.get("Lexical", ""),
                            "itn": alternative.get("ITN", ""),
                            "masked_itn": alternative.get("MaskedITN", "")
                        }
                        
                        # Add word-level timestamps if available
                        if "Words" in alternative:
                            alt_data["words"] = alternative["Words"]
                        
                        alternatives.append(alt_data)
                
                # Extract duration information
                duration = result_data.get("Duration", 0) / 10000000.0  # Convert from ticks
                offset = result_data.get("Offset", 0) / 10000000.0
                
            else:
                # Handle recognition errors
                text = ""
                confidence = 0.0
                alternatives = []
                duration = 0.0
                offset = 0.0
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            result = STTResult(
                text=text,
                confidence=confidence,
                is_final=True,
                language=self._get_azure_language_code(),
                start_time=offset,
                end_time=offset + duration,
                alternatives=alternatives,
                metadata={
                    "provider": self.provider_name,
                    "recognition_status": result_data.get("RecognitionStatus"),
                    "processing_time": processing_time,
                    "audio_length": len(audio_data),
                    "region": self.config.region,
                    "recognition_mode": self.config.recognition_mode
                }
            )
            
            self.logger.debug(f"Transcribed audio in {processing_time:.3f}s: '{text[:50]}...'")
            return result
            
        except Exception as e:
            self.logger.error(f"Error transcribing audio: {e}")
            return STTResult(
                text="",
                confidence=0.0,
                is_final=True,
                metadata={"error": str(e), "provider": self.provider_name}
            )
    
    async def start_streaming(self) -> None:
        """Start streaming transcription mode."""
        if self._streaming_active:
            self.logger.warning("Streaming already active")
            return
        
        try:
            # Generate connection identifiers
            self._connection_id = str(uuid.uuid4()).replace('-', '')
            self._request_id = str(uuid.uuid4()).replace('-', '')
            
            # Build WebSocket URL
            ws_url = self._build_websocket_url()
            
            # Connect to Azure Speech WebSocket
            self._websocket = await websockets.connect(
                ws_url,
                extra_headers={"Ocp-Apim-Subscription-Key": self.config.subscription_key}
            )
            
            # Send speech configuration
            await self._send_speech_config()
            
            # Start WebSocket message handler
            self._websocket_task = asyncio.create_task(self._handle_websocket_messages())
            
            self._streaming_active = True
            self.logger.info("Started Azure STT streaming mode")
            
        except Exception as e:
            self.logger.error(f"Failed to start Azure STT streaming: {e}")
            await self.stop_streaming()
            raise
    
    async def stop_streaming(self) -> None:
        """Stop streaming transcription mode."""
        if not self._streaming_active:
            return
        
        self._streaming_active = False
        
        try:
            # Send end of stream message
            if self._websocket and not self._websocket.closed:
                await self._send_audio_end()
                await asyncio.sleep(0.1)  # Brief delay for final results
        except Exception as e:
            self.logger.warning(f"Error sending end of stream: {e}")
        
        # Cancel WebSocket task
        if self._websocket_task and not self._websocket_task.done():
            self._websocket_task.cancel()
            try:
                await self._websocket_task
            except asyncio.CancelledError:
                pass
        
        # Close WebSocket connection
        if self._websocket and not self._websocket.closed:
            await self._websocket.close()
        
        self._websocket = None
        self._websocket_task = None
        
        self.logger.info("Stopped Azure STT streaming mode")
    
    async def stream_audio(self, audio_chunk: bytes) -> None:
        """
        Send audio chunk for streaming transcription.
        
        Args:
            audio_chunk: Audio data chunk
        """
        if not self._streaming_active or not self._websocket:
            await self.start_streaming()
        
        try:
            await self._send_audio_chunk(audio_chunk)
        except Exception as e:
            self.logger.error(f"Error streaming audio chunk: {e}")
            # Try to reconnect
            await self.stop_streaming()
            await self.start_streaming()
    
    async def get_streaming_results(self) -> AsyncIterator[STTResult]:
        """
        Get streaming transcription results.
        
        Yields:
            STTResult objects as they become available
        """
        while self._streaming_active:
            try:
                # Wait for recognition results
                result = await asyncio.wait_for(
                    self._recognition_results.get(),
                    timeout=0.1
                )
                yield result
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error getting streaming results: {e}")
                yield STTResult(
                    text="",
                    confidence=0.0,
                    is_final=False,
                    metadata={"error": str(e), "provider": self.provider_name}
                )
                await asyncio.sleep(0.5)
    
    def _build_websocket_url(self) -> str:
        """Build WebSocket URL for Azure Speech Services."""
        base_url = self.config.endpoint.replace("https://", "wss://")
        
        params = {
            "language": self._get_azure_language_code(),
            "format": self.config.output_format,
            "Ocp-Apim-Subscription-Key": self.config.subscription_key,
            "X-ConnectionId": self._connection_id
        }
        
        if self.config.profanity_option:
            params["profanity"] = self.config.profanity_option
        
        param_string = "&".join([f"{k}={v}" for k, v in params.items()])
        
        return f"{base_url}/speech/recognition/{self.config.recognition_mode}/cognitiveservices/v1?{param_string}"
    
    async def _send_speech_config(self) -> None:
        """Send speech configuration message."""
        config_message = {
            "context": {
                "synthesis": {
                    "audio": {
                        "metadataoptions": {
                            "sentenceBoundaryEnabled": "false",
                            "wordBoundaryEnabled": "true"
                        },
                        "outputFormat": "audio-16khz-32kbitrate-mono-mp3"
                    }
                }
            }
        }
        
        message = self._create_websocket_message(
            "speech.config",
            json.dumps(config_message)
        )
        
        await self._websocket.send(message)
    
    async def _send_audio_chunk(self, audio_data: bytes) -> None:
        """Send audio chunk via WebSocket."""
        message = self._create_websocket_message("audio", audio_data, is_binary=True)
        await self._websocket.send(message)
    
    async def _send_audio_end(self) -> None:
        """Send end of audio stream message."""
        message = self._create_websocket_message("audio", b"", is_binary=True)
        await self._websocket.send(message)
    
    def _create_websocket_message(self, message_type: str, content, is_binary: bool = False) -> bytes:
        """Create properly formatted WebSocket message for Azure Speech."""
        timestamp = int(time.time() * 10000000)  # Convert to ticks
        
        headers = [
            f"Path: {message_type}",
            f"Request-Id: {self._request_id}",
            f"X-RequestId: {self._request_id}",
            f"X-Timestamp: {timestamp}"
        ]
        
        if is_binary:
            headers.append("Content-Type: audio/x-wav")
        else:
            headers.append("Content-Type: application/json")
        
        header_string = "\r\n".join(headers) + "\r\n\r\n"
        header_bytes = header_string.encode('utf-8')
        
        if is_binary:
            return header_bytes + content
        else:
            return header_bytes + content.encode('utf-8')
    
    async def _handle_websocket_messages(self) -> None:
        """Handle incoming WebSocket messages from Azure Speech."""
        try:
            async for message in self._websocket:
                if isinstance(message, str):
                    await self._process_text_message(message)
                elif isinstance(message, bytes):
                    await self._process_binary_message(message)
                    
        except websockets.exceptions.ConnectionClosed:
            self.logger.info("Azure Speech WebSocket connection closed")
        except Exception as e:
            self.logger.error(f"Error handling WebSocket messages: {e}")
    
    async def _process_text_message(self, message: str) -> None:
        """Process text message from Azure Speech WebSocket."""
        try:
            # Parse message headers and body
            if '\r\n\r\n' in message:
                headers_part, body_part = message.split('\r\n\r\n', 1)
                
                # Parse headers
                headers = {}
                for line in headers_part.split('\r\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        headers[key.strip()] = value.strip()
                
                # Process based on message path
                path = headers.get('Path', '')
                
                if path == 'speech.phrase':
                    # Parse recognition result
                    result_data = json.loads(body_part)
                    await self._process_recognition_result(result_data)
                
                elif path == 'speech.hypothesis':
                    # Parse interim result
                    result_data = json.loads(body_part)
                    await self._process_interim_result(result_data)
                
                elif path == 'speech.endDetected':
                    # End of speech detected
                    self.logger.debug("Speech end detected")
                
                elif path == 'speech.startDetected':
                    # Start of speech detected
                    self.logger.debug("Speech start detected")
                    
        except Exception as e:
            self.logger.error(f"Error processing text message: {e}")
    
    async def _process_binary_message(self, message: bytes) -> None:
        """Process binary message from Azure Speech WebSocket."""
        # Binary messages typically contain audio data for synthesis
        # Not used in STT scenarios
        pass
    
    async def _process_recognition_result(self, result_data: Dict[str, Any]) -> None:
        """Process final recognition result."""
        recognition_status = result_data.get('RecognitionStatus', '')
        
        if recognition_status == 'Success':
            text = result_data.get('DisplayText', '')
            confidence = result_data.get('Confidence', 0.0)
            
            # Extract timing information
            duration = result_data.get('Duration', 0) / 10000000.0
            offset = result_data.get('Offset', 0) / 10000000.0
            
            # Extract alternatives
            alternatives = []
            if 'NBest' in result_data:
                alternatives = result_data['NBest'][:self.config.max_alternatives]
            
            result = STTResult(
                text=text,
                confidence=confidence,
                is_final=True,
                start_time=offset,
                end_time=offset + duration,
                alternatives=alternatives,
                metadata={
                    "provider": self.provider_name,
                    "recognition_status": recognition_status,
                    "result_id": result_data.get('Id', ''),
                    "streaming": True
                }
            )
            
            await self._recognition_results.put(result)
    
    async def _process_interim_result(self, result_data: Dict[str, Any]) -> None:
        """Process interim recognition result."""
        text = result_data.get('Text', '')
        
        if text:
            result = STTResult(
                text=text,
                confidence=0.5,  # Interim results have lower confidence
                is_final=False,
                metadata={
                    "provider": self.provider_name,
                    "result_type": "interim",
                    "result_id": result_data.get('Id', ''),
                    "streaming": True
                }
            )
            
            await self._recognition_results.put(result)
    
    def _get_azure_language_code(self) -> str:
        """Convert internal language enum to Azure language code."""
        language_mapping = {
            STTLanguage.AUTO: "auto-detect",
            STTLanguage.ENGLISH: "en-US",
            STTLanguage.SPANISH: "es-ES",
            STTLanguage.FRENCH: "fr-FR",
            STTLanguage.GERMAN: "de-DE",
            STTLanguage.ITALIAN: "it-IT",
            STTLanguage.PORTUGUESE: "pt-BR",
            STTLanguage.RUSSIAN: "ru-RU",
            STTLanguage.JAPANESE: "ja-JP",
            STTLanguage.KOREAN: "ko-KR",
            STTLanguage.CHINESE: "zh-CN"
        }
        
        return language_mapping.get(self.config.language, "en-US")
    
    async def _test_connection(self) -> None:
        """Test the Azure Speech Services connection."""
        try:
            # Test with a simple REST API call
            response = await self.client.get("/speechtotext/v3.0/datasets")
            # 401 is expected without proper authentication setup, but confirms connectivity
            if response.status_code in [200, 401, 403]:
                return
            else:
                raise Exception(f"Unexpected status code: {response.status_code}")
                
        except Exception as e:
            raise Exception(f"Azure Speech API connection test failed: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of the Azure STT provider.
        
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
                "response_time": response_time,
                "streaming_active": self._streaming_active,
                "config": {
                    "region": self.config.region,
                    "language": self.config.language.value,
                    "recognition_mode": self.config.recognition_mode,
                    "output_format": self.config.output_format,
                    "sample_rate": self.config.sample_rate
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
            "region": self.config.region,
            "recognition_mode": self.config.recognition_mode,
            "output_format": self.config.output_format,
            "streaming_active": self._streaming_active,
            "connection_id": self._connection_id,
            "websocket_connected": self._websocket is not None and not self._websocket.closed if self._websocket else False,
            "pending_results": self._recognition_results.qsize()
        })
        return base_metrics


# Register the provider
STTProviderFactory.register_provider("azure", AzureSTTProvider)