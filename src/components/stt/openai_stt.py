"""
OpenAI Whisper Speech-to-Text Provider Implementation

This module implements the OpenAI Whisper STT provider with streaming support
and real-time transcription capabilities.
"""

import asyncio
import logging
import io
import time
from typing import Optional, Dict, Any, AsyncIterator, List
import openai
from openai import AsyncOpenAI

from .base_stt import (
    BaseSTTProvider, STTResult, STTConfig, STTLanguage, STTQuality,
    STTProviderFactory
)

logger = logging.getLogger(__name__)


class OpenAISTTConfig(STTConfig):
    """Configuration specific to OpenAI Whisper STT."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "whisper-1",
        response_format: str = "json",
        temperature: float = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.model = model
        self.response_format = response_format
        self.temperature = temperature


class OpenAISTTProvider(BaseSTTProvider):
    """
    OpenAI Whisper STT provider implementation.
    
    Provides speech-to-text functionality using OpenAI's Whisper model
    with support for streaming and batch processing.
    """
    
    def __init__(self, config: OpenAISTTConfig):
        super().__init__(config)
        self.config: OpenAISTTConfig = config
        self.client: Optional[AsyncOpenAI] = None
        self._streaming_active = False
        self._audio_buffer = io.BytesIO()
        self._buffer_lock = asyncio.Lock()
        self._last_transcription_time = 0
        self._min_transcription_interval = 0.5  # Minimum time between transcriptions
    
    @property
    def provider_name(self) -> str:
        """Return the name of this STT provider."""
        return "openai_whisper"
    
    @property
    def supported_languages(self) -> List[STTLanguage]:
        """Return list of languages supported by OpenAI Whisper."""
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
        return [8000, 16000, 22050, 44100, 48000]
    
    async def initialize(self) -> None:
        """Initialize the OpenAI STT provider."""
        try:
            api_key = self.config.api_key
            if not api_key:
                import os
                api_key = os.getenv("OPENAI_API_KEY")
            
            if not api_key:
                raise ValueError("OpenAI API key is required")
            
            self.client = AsyncOpenAI(api_key=api_key)
            
            # Test the connection
            await self._test_connection()
            
            self.logger.info("OpenAI STT provider initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI STT: {e}")
            raise
    
    async def cleanup(self) -> None:
        """Clean up resources and close connections."""
        self._streaming_active = False
        
        if self.client:
            await self.client.close()
            self.client = None
        
        self.logger.info("OpenAI STT provider cleanup completed")
    
    async def transcribe_audio(self, audio_data: bytes) -> STTResult:
        """
        Transcribe audio data using OpenAI Whisper.
        
        Args:
            audio_data: Raw audio bytes
            
        Returns:
            STTResult with transcription
        """
        if not self.client:
            raise RuntimeError("Provider not initialized")
        
        try:
            start_time = time.time()
            
            # Create audio file-like object
            audio_file = io.BytesIO(audio_data)
            audio_file.name = "audio.wav"  # OpenAI requires a filename
            
            # Prepare transcription parameters
            params = {
                "file": audio_file,
                "model": self.config.model,
                "response_format": self.config.response_format,
                "temperature": self.config.temperature
            }
            
            # Add language if specified (not auto)
            if self.config.language != STTLanguage.AUTO:
                params["language"] = self.config.language.value
            
            # Add prompt if using custom vocabulary
            if self.config.custom_vocabulary:
                params["prompt"] = " ".join(self.config.custom_vocabulary)
            
            # Make API call
            transcript = await self.client.audio.transcriptions.create(**params)
            
            # Parse response based on format
            if self.config.response_format == "verbose_json":
                text = transcript.text
                confidence = getattr(transcript, 'confidence', 0.9)
                language = getattr(transcript, 'language', None)
                
                # Extract word-level timestamps if available
                words = getattr(transcript, 'words', [])
                alternatives = []
                
                if words:
                    # Create alternatives from word segments
                    for word in words[:self.config.max_alternatives]:
                        alternatives.append({
                            "text": word.get('word', ''),
                            "confidence": word.get('confidence', 0.9),
                            "start_time": word.get('start', 0),
                            "end_time": word.get('end', 0)
                        })
            else:
                # Simple text response
                text = transcript.text if hasattr(transcript, 'text') else str(transcript)
                confidence = 0.9  # OpenAI doesn't provide confidence in simple format
                language = None
                alternatives = []
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            result = STTResult(
                text=text,
                confidence=confidence,
                is_final=True,
                language=language,
                start_time=start_time,
                end_time=end_time,
                alternatives=alternatives,
                metadata={
                    "provider": self.provider_name,
                    "model": self.config.model,
                    "processing_time": processing_time,
                    "audio_length": len(audio_data)
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
        
        self._streaming_active = True
        self._audio_buffer = io.BytesIO()
        self._last_transcription_time = time.time()
        
        self.logger.info("Started OpenAI STT streaming mode")
    
    async def stop_streaming(self) -> None:
        """Stop streaming transcription mode."""
        self._streaming_active = False
        
        # Process any remaining audio in buffer
        async with self._buffer_lock:
            if self._audio_buffer.tell() > 0:
                await self._process_buffer_content()
        
        self.logger.info("Stopped OpenAI STT streaming mode")
    
    async def stream_audio(self, audio_chunk: bytes) -> None:
        """
        Send audio chunk for streaming transcription.
        
        Args:
            audio_chunk: Audio data chunk
        """
        if not self._streaming_active:
            await self.start_streaming()
        
        async with self._buffer_lock:
            # Add chunk to buffer
            self._audio_buffer.write(audio_chunk)
            
            # Check if we should process the buffer
            current_time = time.time()
            buffer_size = self._audio_buffer.tell()
            time_since_last = current_time - self._last_transcription_time
            
            # Process buffer if:
            # 1. Enough time has passed (avoid too frequent API calls)
            # 2. Buffer has sufficient audio data
            # 3. We detect silence (could implement VAD-based triggering)
            should_process = (
                time_since_last >= self._min_transcription_interval and
                buffer_size >= self.config.chunk_size * 4  # Minimum 4 chunks
            )
            
            if should_process:
                await self._process_buffer_content()
                self._last_transcription_time = current_time
    
    async def get_streaming_results(self) -> AsyncIterator[STTResult]:
        """
        Get streaming transcription results.
        
        Yields:
            STTResult objects as they become available
        """
        # Note: OpenAI Whisper doesn't support true real-time streaming
        # This implementation uses buffered processing with periodic transcription
        
        while self._streaming_active:
            try:
                # Check buffer periodically
                await asyncio.sleep(0.1)
                
                async with self._buffer_lock:
                    if self._audio_buffer.tell() > 0:
                        current_time = time.time()
                        time_since_last = current_time - self._last_transcription_time
                        
                        # Process buffer if enough time has passed
                        if time_since_last >= self._min_transcription_interval:
                            result = await self._process_buffer_content()
                            if result and result.text.strip():
                                self._last_transcription_time = current_time
                                yield result
                
            except Exception as e:
                self.logger.error(f"Error in streaming results: {e}")
                yield STTResult(
                    text="",
                    confidence=0.0,
                    is_final=False,
                    metadata={"error": str(e), "provider": self.provider_name}
                )
                await asyncio.sleep(0.5)  # Brief pause on error
    
    async def _process_buffer_content(self) -> Optional[STTResult]:
        """Process current buffer content and reset buffer."""
        if self._audio_buffer.tell() == 0:
            return None
        
        # Get buffer content
        self._audio_buffer.seek(0)
        audio_data = self._audio_buffer.read()
        
        # Reset buffer
        self._audio_buffer = io.BytesIO()
        
        # Skip very short audio segments
        if len(audio_data) < self.config.chunk_size:
            return None
        
        # Transcribe the audio
        result = await self.transcribe_audio(audio_data)
        
        # Mark as interim result for streaming
        result.is_final = False
        
        return result
    
    async def _test_connection(self) -> None:
        """Test the OpenAI API connection."""
        try:
            # Create a minimal test audio file (silence)
            test_audio = b'\x00' * 1024  # 1KB of silence
            audio_file = io.BytesIO(test_audio)
            audio_file.name = "test.wav"
            
            # Test API call with minimal audio
            await self.client.audio.transcriptions.create(
                file=audio_file,
                model=self.config.model,
                response_format="text"
            )
            
        except Exception as e:
            if "audio" in str(e).lower() and "too short" in str(e).lower():
                # This is expected for silent test audio
                pass
            else:
                raise e
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of the OpenAI STT provider.
        
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
            await self._test_connection()
            
            return {
                "status": "healthy",
                "provider": self.provider_name,
                "model": self.config.model,
                "streaming_active": self._streaming_active,
                "config": {
                    "language": self.config.language.value,
                    "quality": self.config.quality.value,
                    "sample_rate": self.config.sample_rate,
                    "response_format": self.config.response_format
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
            "model": self.config.model,
            "response_format": self.config.response_format,
            "temperature": self.config.temperature,
            "streaming_active": self._streaming_active,
            "buffer_size": self._audio_buffer.tell() if self._audio_buffer else 0,
            "last_transcription_time": self._last_transcription_time
        })
        return base_metrics


# Register the provider
STTProviderFactory.register_provider("openai", OpenAISTTProvider)