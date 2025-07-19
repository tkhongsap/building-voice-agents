"""
Azure Cognitive Services Text-to-Speech Provider Implementation

This module implements the Azure Cognitive Services TTS provider with multiple voice options
and neural voice support.
"""

import asyncio
import logging
import json
import time
import uuid
from typing import Optional, Dict, Any, AsyncIterator, List
import httpx
import websockets

from .base_tts import (
    BaseTTSProvider, TTSResult, TTSConfig, Voice, TTSLanguage, 
    TTSQuality, AudioFormat, TTSProviderFactory, TTSVoice
)

logger = logging.getLogger(__name__)


class AzureTTSConfig(TTSConfig):
    """Configuration specific to Azure Cognitive Services TTS."""
    
    def __init__(
        self,
        subscription_key: Optional[str] = None,
        region: str = "eastus",
        endpoint: Optional[str] = None,
        neural_voice: bool = True,
        voice_name: Optional[str] = None,  # e.g., "en-US-JennyNeural"
        speaking_style: Optional[str] = None,  # e.g., "cheerful", "empathetic"
        style_degree: float = 1.0,  # 0.01 to 2.0
        role: Optional[str] = None,  # e.g., "YoungAdultFemale", "OlderAdultMale"
        prosody_rate: Optional[str] = None,  # e.g., "+10%", "0.8", "slow"
        prosody_pitch: Optional[str] = None,  # e.g., "+10Hz", "high", "x-low"
        prosody_contour: Optional[str] = None,  # Pitch contour for fine control
        lexicon_uri: Optional[str] = None,  # Custom pronunciation dictionary
        **kwargs
    ):
        super().__init__(**kwargs)
        self.subscription_key = subscription_key
        self.region = region
        self.endpoint = endpoint or f"https://{region}.tts.speech.microsoft.com"
        self.neural_voice = neural_voice
        self.voice_name = voice_name
        self.speaking_style = speaking_style
        self.style_degree = style_degree
        self.role = role
        self.prosody_rate = prosody_rate
        self.prosody_pitch = prosody_pitch
        self.prosody_contour = prosody_contour
        self.lexicon_uri = lexicon_uri


class AzureTTSProvider(BaseTTSProvider):
    """
    Azure Cognitive Services TTS provider implementation.
    
    Provides text-to-speech functionality using Azure Cognitive Services
    with support for neural voices and advanced voice customization.
    """
    
    def __init__(self, config: AzureTTSConfig):
        super().__init__(config)
        self.config: AzureTTSConfig = config
        self.client: Optional[httpx.AsyncClient] = None
        self._available_voices: List[Voice] = []
        self._voices_cache_time = 0
        self._voices_cache_duration = 3600  # 1 hour cache
        self._request_count = 0
        self._total_characters_synthesized = 0
        self._websocket: Optional[websockets.WebSocketServerProtocol] = None
        self._connection_id = None
    
    @property
    def provider_name(self) -> str:
        """Return the name of this TTS provider."""
        return "azure_speech"
    
    @property
    def supported_languages(self) -> List[TTSLanguage]:
        """Return list of languages supported by Azure Speech."""
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
        """Return list of audio formats supported by Azure Speech."""
        return [AudioFormat.WAV, AudioFormat.MP3, AudioFormat.OGG, AudioFormat.PCM]
    
    @property
    def supported_sample_rates(self) -> List[int]:
        """Return list of supported audio sample rates."""
        return [8000, 16000, 22050, 24000, 44100, 48000]
    
    async def initialize(self) -> None:
        """Initialize the Azure TTS provider."""
        try:
            subscription_key = self.config.subscription_key
            if not subscription_key:
                import os
                subscription_key = os.getenv("AZURE_SPEECH_KEY")
            
            if not subscription_key:
                raise ValueError("Azure Speech subscription key is required")
            
            # Initialize HTTP client
            headers = {
                "Ocp-Apim-Subscription-Key": subscription_key,
                "Content-Type": "application/ssml+xml",
                "X-Microsoft-OutputFormat": self._get_output_format(),
                "User-Agent": "LiveKitVoiceAgent"
            }
            
            self.client = httpx.AsyncClient(
                base_url=self.config.endpoint,
                headers=headers,
                timeout=30.0
            )
            
            # Test the connection and load voices
            await self._test_connection()
            await self.get_voices()  # Cache voices
            
            self.logger.info("Azure TTS provider initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Azure TTS: {e}")
            raise
    
    async def cleanup(self) -> None:
        """Clean up resources and close connections."""
        if self._websocket and not self._websocket.closed:
            await self._websocket.close()
            self._websocket = None
        
        if self.client:
            await self.client.aclose()
            self.client = None
        
        self.logger.info("Azure TTS provider cleanup completed")
    
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
                response = await self.client.get(
                    "/cognitiveservices/voices/list",
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                
                voices_data = response.json()
                voices = []
                
                for voice_data in voices_data:
                    # Map Azure locale to our language enum
                    locale = voice_data.get("Locale", "")
                    voice_language = self._map_locale_to_language(locale)
                    
                    # Determine gender
                    gender_str = voice_data.get("Gender", "").lower()
                    if gender_str == "male":
                        gender = TTSVoice.MALE
                    elif gender_str == "female":
                        gender = TTSVoice.FEMALE
                    else:
                        gender = TTSVoice.NEUTRAL
                    
                    # Get voice type (Neural vs Standard)
                    voice_type = voice_data.get("VoiceType", "Standard")
                    is_neural = voice_type == "Neural"
                    
                    voice = Voice(
                        id=voice_data.get("ShortName", ""),
                        name=voice_data.get("DisplayName", ""),
                        language=voice_language,
                        gender=gender,
                        description=f"{voice_type} voice - {voice_data.get('LocalName', '')}",
                        sample_rate=24000 if is_neural else 16000,
                        metadata={
                            "locale": locale,
                            "voice_type": voice_type,
                            "is_neural": is_neural,
                            "style_list": voice_data.get("StyleList", []),
                            "role_list": voice_data.get("RolePlayList", []),
                            "secondary_locales": voice_data.get("SecondaryLocaleList", []),
                            "sample_rate_hertz": voice_data.get("SampleRateHertz", "24000"),
                            "words_per_minute": voice_data.get("WordsPerMinute", "150")
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
        
        # Sort voices: Neural first, then by name
        voices.sort(key=lambda v: (not v.metadata.get("is_neural", False), v.name))
        
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
        
        effective_voice_id = voice_id or self.config.voice_id or self.config.voice_name
        if not effective_voice_id:
            # Get default voice for the language
            default_voice = await self.get_default_voice(self.config.language)
            if default_voice:
                effective_voice_id = default_voice.id
            else:
                raise ValueError("No voice ID specified and no default voice available")
        
        try:
            start_time = time.time()
            
            # Build SSML
            ssml = self._build_ssml(text, effective_voice_id)
            
            # Make API request
            response = await self.client.post(
                "/cognitiveservices/v1",
                content=ssml,
                headers={
                    "Content-Type": "application/ssml+xml",
                    "X-Microsoft-OutputFormat": self._get_output_format()
                }
            )
            response.raise_for_status()
            
            # Get audio data
            audio_data = response.content
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Update metrics
            self._request_count += 1
            self._total_characters_synthesized += len(text)
            
            # Estimate duration based on words per minute
            words = len(text.split())
            wpm = 150  # Default words per minute
            estimated_duration = (words / wpm) * 60
            
            result = TTSResult(
                audio_data=audio_data,
                format=self.config.audio_format,
                sample_rate=self.config.sample_rate,
                duration=estimated_duration,
                text=text,
                voice_id=effective_voice_id,
                metadata={
                    "provider": self.provider_name,
                    "processing_time": processing_time,
                    "character_count": len(text),
                    "region": self.config.region,
                    "neural_voice": self.config.neural_voice,
                    "style": self.config.speaking_style,
                    "role": self.config.role
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
        
        effective_voice_id = voice_id or self.config.voice_id or self.config.voice_name
        if not effective_voice_id:
            # Get default voice for the language
            default_voice = await self.get_default_voice(self.config.language)
            if default_voice:
                effective_voice_id = default_voice.id
            else:
                raise ValueError("No voice ID specified and no default voice available")
        
        try:
            # For streaming, we'll use WebSocket connection
            await self._connect_websocket()
            
            # Send synthesis request
            await self._send_synthesis_request(text, effective_voice_id)
            
            # Stream audio chunks
            async for chunk in self._receive_audio_chunks():
                if chunk:
                    yield chunk
            
            # Update metrics
            self._request_count += 1
            self._total_characters_synthesized += len(text)
            
        except Exception as e:
            self.logger.error(f"Error in streaming synthesis: {e}")
            yield b""
        finally:
            await self._disconnect_websocket()
    
    def _build_ssml(self, text: str, voice_id: str) -> str:
        """Build SSML document for synthesis."""
        # Escape XML special characters in text
        text = text.replace("&", "&amp;")
        text = text.replace("<", "&lt;")
        text = text.replace(">", "&gt;")
        text = text.replace('"', "&quot;")
        text = text.replace("'", "&apos;")
        
        # Start building SSML
        ssml_parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"',
            '      xmlns:mstts="https://www.w3.org/2001/mstts"',
            f'      xml:lang="{self._get_locale_from_voice_id(voice_id)}">'
        ]
        
        # Add lexicon if specified
        if self.config.lexicon_uri:
            ssml_parts.append(f'  <lexicon uri="{self.config.lexicon_uri}"/>')
        
        # Start voice element
        voice_attrs = [f'name="{voice_id}"']
        
        # Add effect if neural voice features are used
        if self.config.neural_voice and (self.config.speaking_style or self.config.role):
            voice_attrs.append('effect="eq_telecomhp8k"')
        
        ssml_parts.append(f'  <voice {" ".join(voice_attrs)}>')
        
        # Add express-as for style and role
        if self.config.neural_voice and (self.config.speaking_style or self.config.role):
            express_attrs = []
            if self.config.speaking_style:
                express_attrs.append(f'style="{self.config.speaking_style}"')
            if self.config.style_degree != 1.0:
                express_attrs.append(f'styledegree="{self.config.style_degree}"')
            if self.config.role:
                express_attrs.append(f'role="{self.config.role}"')
            
            ssml_parts.append(f'    <mstts:express-as {" ".join(express_attrs)}>')
        
        # Add prosody if any attributes are set
        prosody_attrs = []
        if self.config.prosody_rate:
            prosody_attrs.append(f'rate="{self.config.prosody_rate}"')
        if self.config.prosody_pitch:
            prosody_attrs.append(f'pitch="{self.config.prosody_pitch}"')
        if self.config.prosody_contour:
            prosody_attrs.append(f'contour="{self.config.prosody_contour}"')
        if self.config.volume != 1.0:
            # Convert 0-2 scale to Azure's scale
            azure_volume = "x-soft" if self.config.volume < 0.5 else \
                          "soft" if self.config.volume < 0.75 else \
                          "medium" if self.config.volume < 1.25 else \
                          "loud" if self.config.volume < 1.75 else "x-loud"
            prosody_attrs.append(f'volume="{azure_volume}"')
        
        if prosody_attrs:
            ssml_parts.append(f'      <prosody {" ".join(prosody_attrs)}>')
            ssml_parts.append(f'        {text}')
            ssml_parts.append('      </prosody>')
        else:
            ssml_parts.append(f'      {text}')
        
        # Close express-as if used
        if self.config.neural_voice and (self.config.speaking_style or self.config.role):
            ssml_parts.append('    </mstts:express-as>')
        
        # Close voice and speak elements
        ssml_parts.append('  </voice>')
        ssml_parts.append('</speak>')
        
        return '\n'.join(ssml_parts)
    
    def _get_output_format(self) -> str:
        """Get Azure output format string based on config."""
        format_map = {
            AudioFormat.WAV: {
                8000: "riff-8khz-16bit-mono-pcm",
                16000: "riff-16khz-16bit-mono-pcm",
                22050: "riff-22050hz-16bit-mono-pcm",
                24000: "riff-24khz-16bit-mono-pcm",
                44100: "riff-44100hz-16bit-mono-pcm",
                48000: "riff-48khz-16bit-mono-pcm"
            },
            AudioFormat.MP3: {
                16000: "audio-16khz-32kbitrate-mono-mp3",
                24000: "audio-24khz-48kbitrate-mono-mp3",
                48000: "audio-48khz-96kbitrate-mono-mp3"
            },
            AudioFormat.OGG: {
                16000: "ogg-16khz-16bit-mono-opus",
                24000: "ogg-24khz-16bit-mono-opus",
                48000: "ogg-48khz-16bit-mono-opus"
            },
            AudioFormat.PCM: {
                16000: "raw-16khz-16bit-mono-pcm",
                24000: "raw-24khz-16bit-mono-pcm",
                48000: "raw-48khz-16bit-mono-pcm"
            }
        }
        
        audio_format = self.config.audio_format
        sample_rate = self.config.sample_rate
        
        if audio_format in format_map:
            rate_formats = format_map[audio_format]
            # Find closest supported sample rate
            closest_rate = min(rate_formats.keys(), key=lambda x: abs(x - sample_rate))
            return rate_formats[closest_rate]
        
        # Default format
        return "audio-24khz-48kbitrate-mono-mp3"
    
    def _get_locale_from_voice_id(self, voice_id: str) -> str:
        """Extract locale from voice ID (e.g., 'en-US' from 'en-US-JennyNeural')."""
        parts = voice_id.split('-')
        if len(parts) >= 2:
            return f"{parts[0]}-{parts[1]}"
        return "en-US"  # Default
    
    def _map_locale_to_language(self, locale: str) -> TTSLanguage:
        """Map Azure locale to our language enum."""
        locale_lower = locale.lower()
        
        if locale_lower.startswith("en"):
            return TTSLanguage.ENGLISH
        elif locale_lower.startswith("es"):
            return TTSLanguage.SPANISH
        elif locale_lower.startswith("fr"):
            return TTSLanguage.FRENCH
        elif locale_lower.startswith("de"):
            return TTSLanguage.GERMAN
        elif locale_lower.startswith("it"):
            return TTSLanguage.ITALIAN
        elif locale_lower.startswith("pt"):
            return TTSLanguage.PORTUGUESE
        elif locale_lower.startswith("ru"):
            return TTSLanguage.RUSSIAN
        elif locale_lower.startswith("ja"):
            return TTSLanguage.JAPANESE
        elif locale_lower.startswith("ko"):
            return TTSLanguage.KOREAN
        elif locale_lower.startswith("zh"):
            return TTSLanguage.CHINESE
        else:
            return TTSLanguage.ENGLISH  # Default
    
    async def _connect_websocket(self) -> None:
        """Connect to Azure Speech WebSocket for streaming."""
        self._connection_id = str(uuid.uuid4()).replace('-', '')
        
        ws_url = (
            f"wss://{self.config.region}.tts.speech.microsoft.com/"
            f"cognitiveservices/websocket/v1?TrafficType=AzureDemo"
            f"&X-ConnectionId={self._connection_id}"
        )
        
        headers = {
            "Ocp-Apim-Subscription-Key": self.config.subscription_key
        }
        
        self._websocket = await websockets.connect(ws_url, extra_headers=headers)
    
    async def _disconnect_websocket(self) -> None:
        """Disconnect WebSocket connection."""
        if self._websocket and not self._websocket.closed:
            await self._websocket.close()
            self._websocket = None
    
    async def _send_synthesis_request(self, text: str, voice_id: str) -> None:
        """Send synthesis request via WebSocket."""
        if not self._websocket:
            raise RuntimeError("WebSocket not connected")
        
        # Build SSML
        ssml = self._build_ssml(text, voice_id)
        
        # Create request message
        request = {
            "ssml": ssml,
            "outputFormat": self._get_output_format()
        }
        
        # Send request
        message = f"Path: ssml\r\nX-RequestId: {self._connection_id}\r\nContent-Type: application/json\r\n\r\n{json.dumps(request)}"
        await self._websocket.send(message)
    
    async def _receive_audio_chunks(self) -> AsyncIterator[bytes]:
        """Receive audio chunks from WebSocket."""
        if not self._websocket:
            return
        
        try:
            async for message in self._websocket:
                if isinstance(message, bytes):
                    # Audio data chunk
                    yield message
                elif isinstance(message, str):
                    # Control message
                    if "Path:turn.end" in message:
                        break
        except websockets.exceptions.ConnectionClosed:
            self.logger.warning("WebSocket connection closed during streaming")
    
    async def _test_connection(self) -> None:
        """Test the Azure Speech Services connection."""
        try:
            # Try to list voices as a connection test
            response = await self.client.get(
                "/cognitiveservices/voices/list",
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code in [200, 401, 403]:
                # Even auth errors confirm connectivity
                return
            else:
                raise Exception(f"Unexpected status code: {response.status_code}")
                
        except Exception as e:
            raise Exception(f"Azure Speech API connection test failed: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of the Azure TTS provider.
        
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
            neural_voice_count = sum(1 for v in voices if v.metadata.get("is_neural", False))
            
            return {
                "status": "healthy",
                "provider": self.provider_name,
                "response_time": response_time,
                "region": self.config.region,
                "available_voices_count": len(voices),
                "neural_voices_count": neural_voice_count,
                "request_count": self._request_count,
                "total_characters_synthesized": self._total_characters_synthesized,
                "config": {
                    "neural_voice": self.config.neural_voice,
                    "voice_id": self.config.voice_id or self.config.voice_name,
                    "audio_format": self.config.audio_format.value,
                    "sample_rate": self.config.sample_rate,
                    "speaking_style": self.config.speaking_style,
                    "role": self.config.role
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
            "region": self.config.region,
            "neural_voice": self.config.neural_voice,
            "prosody_settings": {
                "rate": self.config.prosody_rate,
                "pitch": self.config.prosody_pitch,
                "contour": self.config.prosody_contour
            },
            "style_settings": {
                "style": self.config.speaking_style,
                "style_degree": self.config.style_degree,
                "role": self.config.role
            }
        })
        return base_metrics


# Register the provider
TTSProviderFactory.register_provider("azure", AzureTTSProvider)