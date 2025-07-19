"""
OpenAI Text-to-Speech Provider Implementation

This module implements the OpenAI TTS provider with support for multiple voices
and audio formats as a cost-effective fallback for ElevenLabs.
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, AsyncIterator, List, Union
import openai
from openai import AsyncOpenAI
import io

from .base_tts import (
    BaseTTSProvider, TTSResult, TTSConfig, Voice, TTSLanguage, TTSVoice, 
    AudioFormat, TTSQuality, TTSProviderFactory
)

logger = logging.getLogger(__name__)


class OpenAITTSConfig(TTSConfig):
    """Configuration specific to OpenAI TTS."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "tts-1",  # tts-1 for standard, tts-1-hd for high quality
        voice: str = "alloy",  # alloy, echo, fable, onyx, nova, shimmer
        response_format: str = "mp3",  # mp3, opus, aac, flac
        speed: float = 1.0,  # 0.25 to 4.0
        organization: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 30.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.response_format = response_format
        self.speed = speed
        self.organization = organization
        self.max_retries = max_retries
        self.timeout = timeout


class OpenAITTSProvider(BaseTTSProvider):
    """
    OpenAI TTS provider implementation.
    
    Provides text-to-speech functionality using OpenAI's TTS models
    as a cost-effective fallback option.
    """
    
    def __init__(self, config: OpenAITTSConfig):
        super().__init__(config)
        self.config: OpenAITTSConfig = config
        self.client: Optional[AsyncOpenAI] = None
        self._request_count = 0
        self._total_characters_synthesized = 0
        self._last_request_time = 0
        
        # OpenAI TTS available voices
        self._available_voices = {
            "alloy": Voice(
                id="alloy",
                name="Alloy",
                language=TTSLanguage.ENGLISH,
                gender=TTSVoice.NEUTRAL,
                description="Balanced, natural voice"
            ),
            "echo": Voice(
                id="echo", 
                name="Echo",
                language=TTSLanguage.ENGLISH,
                gender=TTSVoice.MALE,
                description="Clear, articulate male voice"
            ),
            "fable": Voice(
                id="fable",
                name="Fable", 
                language=TTSLanguage.ENGLISH,
                gender=TTSVoice.MALE,
                description="Warm, engaging male voice"
            ),
            "onyx": Voice(
                id="onyx",
                name="Onyx",
                language=TTSLanguage.ENGLISH, 
                gender=TTSVoice.MALE,
                description="Deep, resonant male voice"
            ),
            "nova": Voice(
                id="nova",
                name="Nova",
                language=TTSLanguage.ENGLISH,
                gender=TTSVoice.FEMALE,
                description="Bright, energetic female voice"
            ),
            "shimmer": Voice(
                id="shimmer",
                name="Shimmer",
                language=TTSLanguage.ENGLISH,
                gender=TTSVoice.FEMALE,
                description="Soft, gentle female voice"
            )
        }
    
    @property
    def provider_name(self) -> str:
        """Return the name of this TTS provider."""
        return "openai_tts"
    
    @property
    def supported_languages(self) -> List[TTSLanguage]:
        """Return list of supported languages."""
        # OpenAI TTS supports many languages, focusing on main ones
        return [
            TTSLanguage.ENGLISH,
            TTSLanguage.SPANISH,
            TTSLanguage.FRENCH,
            TTSLanguage.GERMAN,
            TTSLanguage.ITALIAN,
            TTSLanguage.PORTUGUESE,
            TTSLanguage.RUSSIAN,
            TTSLanguage.JAPANESE,
            TTSLanguage.KOREAN,
            TTSLanguage.CHINESE
        ]
    
    @property 
    def supported_formats(self) -> List[AudioFormat]:
        """Return list of supported audio formats."""
        return [
            AudioFormat.MP3,
            AudioFormat.WAV,  # We'll convert from supported formats
            AudioFormat.OGG   # OPUS can be mapped to OGG
        ]
    
    @property
    def supports_streaming(self) -> bool:
        """Return whether this provider supports streaming synthesis."""
        return False  # OpenAI TTS doesn't support true streaming yet
    
    async def initialize(self) -> None:
        """Initialize the OpenAI TTS provider."""
        try:
            # Create OpenAI client
            self.client = AsyncOpenAI(
                api_key=self.config.api_key,
                organization=self.config.organization,
                max_retries=self.config.max_retries,
                timeout=self.config.timeout
            )
            
            # Validate configuration
            if self.config.voice not in self._available_voices:
                self.logger.warning(
                    f"Voice '{self.config.voice}' not recognized. "
                    f"Available voices: {list(self._available_voices.keys())}"
                )
            
            if not (0.25 <= self.config.speed <= 4.0):
                raise ValueError("Speed must be between 0.25 and 4.0")
            
            self.logger.info(
                f"OpenAI TTS provider initialized with model {self.config.model}, "
                f"voice {self.config.voice}"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI TTS: {e}")
            raise
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        if self.client:
            await self.client.close()
            self.client = None
        
        self.logger.info("OpenAI TTS provider cleanup completed")
    
    async def synthesize_speech(self, text: str, voice_id: Optional[str] = None) -> TTSResult:
        """
        Synthesize speech from text using OpenAI TTS.
        
        Args:
            text: Text to synthesize
            voice_id: Voice ID to use (optional, uses config default)
            
        Returns:
            TTSResult with synthesized audio
        """
        if not self.client:
            raise RuntimeError("Provider not initialized")
        
        try:
            start_time = time.time()
            
            # Use provided voice or config default
            voice = voice_id or self.config.voice
            
            # Map audio format for OpenAI API
            openai_format = self._map_audio_format(self.config.response_format)
            
            # Make API request
            response = await self.client.audio.speech.create(
                model=self.config.model,
                voice=voice,
                input=text,
                response_format=openai_format,
                speed=self.config.speed
            )
            
            # Get audio data
            audio_data = response.content
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Update metrics
            self._request_count += 1
            self._total_characters_synthesized += len(text)
            self._last_request_time = end_time
            
            # Create result
            result = TTSResult(
                audio_data=audio_data,
                format=self._openai_to_audio_format(openai_format),
                sample_rate=self.config.sample_rate,
                duration=self._estimate_duration(len(text)),
                text=text,
                voice_id=voice,
                processing_time=processing_time,
                metadata={
                    "provider": self.provider_name,
                    "model": self.config.model,
                    "voice": voice,
                    "speed": self.config.speed,
                    "character_count": len(text),
                    "format": openai_format
                }
            )
            
            self.logger.debug(
                f"Synthesized {len(text)} characters in {processing_time:.3f}s "
                f"using voice '{voice}'"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error synthesizing speech: {e}")
            raise
    
    async def synthesize_speech_streaming(
        self, 
        text: str, 
        voice_id: Optional[str] = None
    ) -> AsyncIterator[TTSResult]:
        """
        Stream speech synthesis (not supported by OpenAI TTS yet).
        
        Falls back to batch synthesis.
        """
        self.logger.warning("OpenAI TTS doesn't support streaming, using batch synthesis")
        
        result = await self.synthesize_speech(text, voice_id)
        yield result
    
    async def get_available_voices(self) -> List[Voice]:
        """
        Get list of available voices.
        
        Returns:
            List of available Voice objects
        """
        return list(self._available_voices.values())
    
    async def get_voice_by_id(self, voice_id: str) -> Optional[Voice]:
        """
        Get voice information by ID.
        
        Args:
            voice_id: Voice identifier
            
        Returns:
            Voice object if found, None otherwise
        """
        return self._available_voices.get(voice_id)
    
    def _map_audio_format(self, format_str: str) -> str:
        """Map our audio format to OpenAI API format."""
        format_mapping = {
            "mp3": "mp3",
            "wav": "wav",  # OpenAI doesn't support WAV, we'll use mp3 and convert if needed
            "ogg": "opus"  # Map OGG to OPUS
        }
        return format_mapping.get(format_str.lower(), "mp3")
    
    def _openai_to_audio_format(self, openai_format: str) -> AudioFormat:
        """Convert OpenAI format string to AudioFormat enum."""
        format_mapping = {
            "mp3": AudioFormat.MP3,
            "opus": AudioFormat.OGG,
            "aac": AudioFormat.MP3,  # Closest match
            "flac": AudioFormat.WAV  # Closest match
        }
        return format_mapping.get(openai_format.lower(), AudioFormat.MP3)
    
    def _estimate_duration(self, text_length: int) -> float:
        """Estimate audio duration based on text length."""
        # Rough estimate: ~150 words per minute, ~5 characters per word
        words_per_minute = 150
        characters_per_word = 5
        words = text_length / characters_per_word
        duration_minutes = words / words_per_minute
        return duration_minutes * 60
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of the OpenAI TTS provider.
        
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
            
            # Test with a small synthesis request
            start_time = time.time()
            test_text = "Health check test."
            
            response = await self.client.audio.speech.create(
                model=self.config.model,
                voice=self.config.voice,
                input=test_text,
                response_format="mp3"
            )
            
            response_time = time.time() - start_time
            
            return {
                "status": "healthy",
                "provider": self.provider_name,
                "response_time": response_time,
                "model": self.config.model,
                "voice": self.config.voice,
                "request_count": self._request_count,
                "total_characters": self._total_characters_synthesized,
                "available_voices": len(self._available_voices),
                "supports_streaming": self.supports_streaming
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
            "total_characters_synthesized": self._total_characters_synthesized,
            "last_request_time": self._last_request_time,
            "average_characters_per_request": (
                self._total_characters_synthesized / self._request_count 
                if self._request_count > 0 else 0
            ),
            "available_voices": len(self._available_voices),
            "current_voice": self.config.voice,
            "current_model": self.config.model,
            "openai_config": {
                "model": self.config.model,
                "voice": self.config.voice,
                "speed": self.config.speed,
                "response_format": self.config.response_format,
                "max_retries": self.config.max_retries,
                "timeout": self.config.timeout
            }
        })
        return base_metrics


# Register the provider
TTSProviderFactory.register_provider("openai", OpenAITTSProvider)