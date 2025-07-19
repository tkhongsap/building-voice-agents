"""
ElevenLabs Text-to-Speech Provider Implementation

This module implements the ElevenLabs TTS provider with voice cloning support
and streaming synthesis capabilities.
"""

import asyncio
import logging
import httpx
import json
import time
from typing import Optional, Dict, Any, AsyncIterator, List
import io

from .base_tts import (
    BaseTTSProvider, TTSResult, TTSConfig, Voice, TTSLanguage, 
    TTSQuality, AudioFormat, TTSProviderFactory
)

logger = logging.getLogger(__name__)


class ElevenLabsTTSConfig(TTSConfig):
    """Configuration specific to ElevenLabs TTS."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_id: str = "eleven_monolingual_v1",
        voice_settings: Optional[Dict[str, float]] = None,
        pronunciation_dictionary_locators: Optional[List[str]] = None,
        seed: Optional[int] = None,
        previous_text: Optional[str] = None,
        next_text: Optional[str] = None,
        previous_request_ids: Optional[List[str]] = None,
        next_request_ids: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.model_id = model_id
        self.voice_settings = voice_settings or {
            "stability": 0.5,
            "similarity_boost": 0.5,
            "style": 0.0,
            "use_speaker_boost": True
        }
        self.pronunciation_dictionary_locators = pronunciation_dictionary_locators or []
        self.seed = seed
        self.previous_text = previous_text
        self.next_text = next_text
        self.previous_request_ids = previous_request_ids or []
        self.next_request_ids = next_request_ids or []


class ElevenLabsTTSProvider(BaseTTSProvider):
    """
    ElevenLabs TTS provider implementation.
    
    Provides text-to-speech functionality using ElevenLabs' API
    with support for voice cloning and streaming synthesis.
    """
    
    def __init__(self, config: ElevenLabsTTSConfig):
        super().__init__(config)
        self.config: ElevenLabsTTSConfig = config
        self.client: Optional[httpx.AsyncClient] = None
        self.base_url = "https://api.elevenlabs.io/v1"
        self._available_voices: List[Voice] = []
        self._voices_cache_time = 0
        self._voices_cache_duration = 300  # 5 minutes
        self._request_count = 0
        self._total_characters_synthesized = 0
    
    @property
    def provider_name(self) -> str:
        """Return the name of this TTS provider."""
        return "elevenlabs"
    
    @property
    def supported_languages(self) -> List[TTSLanguage]:
        """Return list of languages supported by ElevenLabs."""
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
        """Return list of audio formats supported by ElevenLabs."""
        return [AudioFormat.MP3, AudioFormat.WAV, AudioFormat.PCM]
    
    @property
    def supported_sample_rates(self) -> List[int]:
        """Return list of supported audio sample rates."""
        return [22050, 44100]
    
    async def initialize(self) -> None:
        """Initialize the ElevenLabs TTS provider."""
        try:
            api_key = self.config.api_key
            if not api_key:
                import os
                api_key = os.getenv("ELEVEN_LABS_API_KEY")
            
            if not api_key:
                raise ValueError("ElevenLabs API key is required")
            
            # Initialize HTTP client
            headers = {
                "xi-api-key": api_key,
                "Content-Type": "application/json",
                "Accept": "audio/mpeg"
            }
            
            self.client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=30.0
            )
            
            # Test the connection and load voices
            await self._test_connection()
            await self.get_voices()  # Cache voices
            
            self.logger.info("ElevenLabs TTS provider initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ElevenLabs TTS: {e}")
            raise
    
    async def cleanup(self) -> None:
        """Clean up resources and close connections."""
        if self.client:
            await self.client.aclose()
            self.client = None
        
        self.logger.info("ElevenLabs TTS provider cleanup completed")
    
    async def get_voices(self, language: Optional[TTSLanguage] = None) -> List[Voice]:
        """
        Get available voices for the specified language.
        
        Args:
            language: Optional language filter
            
        Returns:
            List of available voices
        """
        if not self.client:
            raise RuntimeError("Provider not initialized")
        
        # Check cache
        current_time = time.time()
        if (self._available_voices and 
            current_time - self._voices_cache_time < self._voices_cache_duration):
            voices = self._available_voices
        else:
            # Fetch voices from API
            try:
                response = await self.client.get("/voices")
                response.raise_for_status()
                
                voices_data = response.json()
                voices = []
                
                for voice_data in voices_data.get("voices", []):
                    # Determine language from voice metadata
                    voice_language = TTSLanguage.ENGLISH  # Default
                    labels = voice_data.get("labels", {})
                    if "language" in labels:
                        lang_code = labels["language"].lower()
                        for tts_lang in TTSLanguage:
                            if tts_lang.value in lang_code:
                                voice_language = tts_lang
                                break
                    
                    # Determine gender from voice metadata
                    from .base_tts import TTSVoice
                    gender = TTSVoice.NEUTRAL  # Default
                    if "gender" in labels:
                        gender_str = labels["gender"].lower()
                        if "male" in gender_str:
                            gender = TTSVoice.MALE
                        elif "female" in gender_str:
                            gender = TTSVoice.FEMALE
                    
                    voice = Voice(
                        id=voice_data["voice_id"],
                        name=voice_data["name"],
                        language=voice_language,
                        gender=gender,
                        description=voice_data.get("description", ""),
                        metadata={
                            "category": voice_data.get("category", ""),
                            "labels": labels,
                            "preview_url": voice_data.get("preview_url", ""),
                            "available_for_tiers": voice_data.get("available_for_tiers", [])
                        }
                    )
                    voices.append(voice)
                
                # Cache the voices
                self._available_voices = voices
                self._voices_cache_time = current_time
                
            except Exception as e:
                self.logger.error(f"Error fetching voices: {e}")
                return []
        
        # Filter by language if specified
        if language:
            voices = [v for v in voices if v.language == language]
        
        return voices
    
    async def synthesize_speech(self, text: str, voice_id: Optional[str] = None) -> TTSResult:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            voice_id: Optional voice ID override
            
        Returns:
            TTSResult with audio data
        """
        if not self.client:
            raise RuntimeError("Provider not initialized")
        
        effective_voice_id = voice_id or self.config.voice_id
        if not effective_voice_id:
            raise ValueError("Voice ID is required")
        
        try:
            start_time = time.time()
            
            # Prepare request data
            request_data = {
                "text": text,
                "model_id": self.config.model_id,
                "voice_settings": self.config.voice_settings
            }
            
            # Add optional parameters
            if self.config.pronunciation_dictionary_locators:
                request_data["pronunciation_dictionary_locators"] = self.config.pronunciation_dictionary_locators
            
            if self.config.seed is not None:
                request_data["seed"] = self.config.seed
            
            if self.config.previous_text:
                request_data["previous_text"] = self.config.previous_text
            
            if self.config.next_text:
                request_data["next_text"] = self.config.next_text
            
            if self.config.previous_request_ids:
                request_data["previous_request_ids"] = self.config.previous_request_ids
            
            if self.config.next_request_ids:
                request_data["next_request_ids"] = self.config.next_request_ids
            
            # Determine output format
            if self.config.audio_format == AudioFormat.MP3:
                output_format = "mp3_44100_128"
            elif self.config.audio_format == AudioFormat.WAV:
                output_format = "wav"
            else:
                output_format = "mp3_44100_128"  # Default
            
            # Make API request
            url = f"/text-to-speech/{effective_voice_id}"
            params = {"output_format": output_format}
            
            response = await self.client.post(
                url,
                json=request_data,
                params=params
            )
            response.raise_for_status()
            
            # Get audio data
            audio_data = response.content
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Update metrics
            self._request_count += 1
            self._total_characters_synthesized += len(text)
            
            # Estimate duration (rough estimate based on character count)
            estimated_duration = len(text) / 150 * 60  # ~150 chars per minute
            
            result = TTSResult(
                audio_data=audio_data,
                format=self.config.audio_format,
                sample_rate=self.config.sample_rate,
                duration=estimated_duration,
                text=text,
                voice_id=effective_voice_id,
                metadata={
                    "provider": self.provider_name,
                    "model_id": self.config.model_id,
                    "processing_time": processing_time,
                    "character_count": len(text),
                    "voice_settings": self.config.voice_settings
                }
            )
            
            self.logger.debug(f"Synthesized {len(text)} characters in {processing_time:.3f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Error synthesizing speech: {e}")
            return TTSResult(
                audio_data=b"",
                format=self.config.audio_format,
                sample_rate=self.config.sample_rate,
                metadata={"error": str(e), "provider": self.provider_name}
            )
    
    async def synthesize_streaming(
        self, 
        text: str, 
        voice_id: Optional[str] = None
    ) -> AsyncIterator[bytes]:
        """
        Synthesize speech with streaming output.
        
        Args:
            text: Text to synthesize
            voice_id: Optional voice ID override
            
        Yields:
            Audio data chunks as they become available
        """
        if not self.client:
            raise RuntimeError("Provider not initialized")
        
        effective_voice_id = voice_id or self.config.voice_id
        if not effective_voice_id:
            raise ValueError("Voice ID is required")
        
        try:
            # Prepare request data
            request_data = {
                "text": text,
                "model_id": self.config.model_id,
                "voice_settings": self.config.voice_settings
            }
            
            # Add optional parameters
            if self.config.pronunciation_dictionary_locators:
                request_data["pronunciation_dictionary_locators"] = self.config.pronunciation_dictionary_locators
            
            # Determine output format
            if self.config.audio_format == AudioFormat.MP3:
                output_format = "mp3_44100_128"
            elif self.config.audio_format == AudioFormat.WAV:
                output_format = "wav"
            else:
                output_format = "mp3_44100_128"  # Default
            
            # Make streaming API request
            url = f"/text-to-speech/{effective_voice_id}/stream"
            params = {"output_format": output_format}
            
            async with self.client.stream(
                "POST",
                url,
                json=request_data,
                params=params
            ) as response:
                response.raise_for_status()
                
                # Stream audio chunks
                async for chunk in response.aiter_bytes(chunk_size=self.config.chunk_size):
                    if chunk:
                        yield chunk
            
            # Update metrics
            self._request_count += 1
            self._total_characters_synthesized += len(text)
            
        except Exception as e:
            self.logger.error(f"Error in streaming synthesis: {e}")
            # Yield empty chunk to indicate error
            yield b""
    
    async def _test_connection(self) -> None:
        """Test the ElevenLabs API connection."""
        try:
            response = await self.client.get("/user")
            response.raise_for_status()
            
            user_data = response.json()
            self.logger.debug(f"Connected to ElevenLabs API as user: {user_data.get('username', 'Unknown')}")
            
        except Exception as e:
            raise Exception(f"ElevenLabs API connection test failed: {e}")
    
    async def clone_voice(
        self, 
        name: str, 
        description: str,
        files: List[bytes],
        labels: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Clone a voice using provided audio samples.
        
        Args:
            name: Name for the cloned voice
            description: Description of the voice
            files: List of audio file bytes for training
            labels: Optional labels for the voice
            
        Returns:
            Voice ID of the cloned voice
        """
        if not self.client:
            raise RuntimeError("Provider not initialized")
        
        try:
            # Prepare multipart form data
            files_data = []
            for i, file_bytes in enumerate(files):
                files_data.append(("files", (f"sample_{i}.wav", file_bytes, "audio/wav")))
            
            data = {
                "name": name,
                "description": description
            }
            
            if labels:
                data["labels"] = json.dumps(labels)
            
            # Make voice cloning request
            response = await self.client.post(
                "/voices/add",
                data=data,
                files=files_data
            )
            response.raise_for_status()
            
            result = response.json()
            voice_id = result["voice_id"]
            
            self.logger.info(f"Successfully cloned voice '{name}' with ID: {voice_id}")
            
            # Clear voice cache to include new voice
            self._available_voices = []
            self._voices_cache_time = 0
            
            return voice_id
            
        except Exception as e:
            self.logger.error(f"Error cloning voice: {e}")
            raise
    
    async def delete_voice(self, voice_id: str) -> bool:
        """
        Delete a cloned voice.
        
        Args:
            voice_id: ID of the voice to delete
            
        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            raise RuntimeError("Provider not initialized")
        
        try:
            response = await self.client.delete(f"/voices/{voice_id}")
            response.raise_for_status()
            
            self.logger.info(f"Successfully deleted voice: {voice_id}")
            
            # Clear voice cache
            self._available_voices = []
            self._voices_cache_time = 0
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting voice {voice_id}: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of the ElevenLabs TTS provider.
        
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
            
            # Get voice count
            voices = await self.get_voices()
            
            return {
                "status": "healthy",
                "provider": self.provider_name,
                "response_time": response_time,
                "available_voices_count": len(voices),
                "request_count": self._request_count,
                "total_characters_synthesized": self._total_characters_synthesized,
                "config": {
                    "model_id": self.config.model_id,
                    "voice_id": self.config.voice_id,
                    "audio_format": self.config.audio_format.value,
                    "sample_rate": self.config.sample_rate,
                    "voice_settings": self.config.voice_settings
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
            "total_characters_synthesized": self._total_characters_synthesized,
            "average_characters_per_request": (
                self._total_characters_synthesized / self._request_count 
                if self._request_count > 0 else 0
            ),
            "voices_cache_size": len(self._available_voices),
            "model_id": self.config.model_id,
            "voice_settings": self.config.voice_settings
        })
        return base_metrics


# Register the provider
TTSProviderFactory.register_provider("elevenlabs", ElevenLabsTTSProvider)