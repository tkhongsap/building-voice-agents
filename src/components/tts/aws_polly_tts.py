"""
AWS Polly Text-to-Speech Provider Implementation

This module implements the AWS Polly TTS provider with neural voices
and SSML support for advanced speech synthesis.
"""

import asyncio
import logging
import json
import time
import hmac
import hashlib
from datetime import datetime
from typing import Optional, Dict, Any, AsyncIterator, List
import httpx

from .base_tts import (
    BaseTTSProvider, TTSResult, TTSConfig, Voice, TTSLanguage, 
    TTSQuality, AudioFormat, TTSProviderFactory, TTSVoice
)

logger = logging.getLogger(__name__)


class AWSPollyTTSConfig(TTSConfig):
    """Configuration specific to AWS Polly TTS."""
    
    def __init__(
        self,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        session_token: Optional[str] = None,
        region: str = "us-east-1",
        engine: str = "neural",  # "standard" or "neural"
        voice_id: Optional[str] = None,  # e.g., "Joanna", "Matthew"
        language_code: Optional[str] = None,  # e.g., "en-US", "es-US"
        speech_mark_types: Optional[List[str]] = None,  # ["sentence", "word", "viseme", "ssml"]
        max_duration: int = 600,  # Maximum speech duration in seconds
        lexicon_names: Optional[List[str]] = None,  # Custom pronunciation lexicons
        neural_voice_style: Optional[str] = None,  # e.g., "conversational", "newscaster"
        **kwargs
    ):
        super().__init__(**kwargs)
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.session_token = session_token
        self.region = region
        self.engine = engine
        self.voice_id = voice_id
        self.language_code = language_code
        self.speech_mark_types = speech_mark_types or []
        self.max_duration = max_duration
        self.lexicon_names = lexicon_names or []
        self.neural_voice_style = neural_voice_style


class AWSPollyTTSProvider(BaseTTSProvider):
    """
    AWS Polly TTS provider implementation.
    
    Provides text-to-speech functionality using AWS Polly
    with support for neural voices and advanced customization.
    """
    
    def __init__(self, config: AWSPollyTTSConfig):
        super().__init__(config)
        self.config: AWSPollyTTSConfig = config
        self.client: Optional[httpx.AsyncClient] = None
        self._available_voices: List[Voice] = []
        self._voices_cache_time = 0
        self._voices_cache_duration = 3600  # 1 hour cache
        self._request_count = 0
        self._total_characters_synthesized = 0
        self._service = "polly"
        self._endpoint = f"https://polly.{config.region}.amazonaws.com"
    
    @property
    def provider_name(self) -> str:
        """Return the name of this TTS provider."""
        return "aws_polly"
    
    @property
    def supported_languages(self) -> List[TTSLanguage]:
        """Return list of languages supported by AWS Polly."""
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
        """Return list of audio formats supported by AWS Polly."""
        return [AudioFormat.MP3, AudioFormat.OGG, AudioFormat.PCM, AudioFormat.WAV]
    
    @property
    def supported_sample_rates(self) -> List[int]:
        """Return list of supported audio sample rates."""
        return [8000, 16000, 22050, 24000]
    
    async def initialize(self) -> None:
        """Initialize the AWS Polly TTS provider."""
        try:
            # Get AWS credentials
            access_key_id = self.config.access_key_id
            secret_access_key = self.config.secret_access_key
            
            if not access_key_id or not secret_access_key:
                import os
                access_key_id = access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
                secret_access_key = secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")
                self.config.session_token = self.config.session_token or os.getenv("AWS_SESSION_TOKEN")
            
            if not access_key_id or not secret_access_key:
                raise ValueError("AWS credentials are required")
            
            # Store credentials for signing
            self.config.access_key_id = access_key_id
            self.config.secret_access_key = secret_access_key
            
            # Initialize HTTP client
            self.client = httpx.AsyncClient(
                base_url=self._endpoint,
                timeout=30.0
            )
            
            # Test the connection and load voices
            await self._test_connection()
            await self.get_voices()  # Cache voices
            
            self.logger.info("AWS Polly TTS provider initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize AWS Polly TTS: {e}")
            raise
    
    async def cleanup(self) -> None:
        """Clean up resources and close connections."""
        if self.client:
            await self.client.aclose()
            self.client = None
        
        self.logger.info("AWS Polly TTS provider cleanup completed")
    
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
                # Prepare request
                method = "GET"
                path = "/v1/voices"
                headers = await self._create_aws_headers(method, path, {})
                
                response = await self.client.get(path, headers=headers)
                response.raise_for_status()
                
                voices_data = response.json()
                voices = []
                
                for voice_data in voices_data.get("Voices", []):
                    # Map language code to our enum
                    language_code = voice_data.get("LanguageCode", "")
                    voice_language = self._map_language_code_to_enum(language_code)
                    
                    # Determine gender
                    gender_str = voice_data.get("Gender", "").lower()
                    if gender_str == "male":
                        gender = TTSVoice.MALE
                    elif gender_str == "female":
                        gender = TTSVoice.FEMALE
                    else:
                        gender = TTSVoice.NEUTRAL
                    
                    # Check if neural voice
                    supported_engines = voice_data.get("SupportedEngines", ["standard"])
                    is_neural = "neural" in supported_engines
                    
                    voice = Voice(
                        id=voice_data.get("Id", ""),
                        name=voice_data.get("Name", ""),
                        language=voice_language,
                        gender=gender,
                        description=f"{voice_data.get('LanguageName', '')} - {voice_data.get('Name', '')}",
                        sample_rate=24000 if is_neural else 22050,
                        metadata={
                            "language_code": language_code,
                            "language_name": voice_data.get("LanguageName", ""),
                            "supported_engines": supported_engines,
                            "is_neural": is_neural,
                            "additional_language_codes": voice_data.get("AdditionalLanguageCodes", [])
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
        
        effective_voice_id = voice_id or self.config.voice_id
        if not effective_voice_id:
            # Get default voice for the language
            default_voice = await self.get_default_voice(self.config.language)
            if default_voice:
                effective_voice_id = default_voice.id
            else:
                effective_voice_id = "Joanna"  # AWS Polly default
        
        try:
            start_time = time.time()
            
            # Prepare request body
            request_body = {
                "OutputFormat": self._get_output_format(),
                "Text": text if not self.config.enable_ssml else self._build_ssml(text),
                "TextType": "ssml" if self.config.enable_ssml else "text",
                "VoiceId": effective_voice_id,
                "Engine": self.config.engine
            }
            
            # Add optional parameters
            if self.config.language_code:
                request_body["LanguageCode"] = self.config.language_code
            
            if self.config.sample_rate:
                request_body["SampleRate"] = str(self.config.sample_rate)
            
            if self.config.lexicon_names:
                request_body["LexiconNames"] = self.config.lexicon_names
            
            if self.config.speech_mark_types:
                request_body["SpeechMarkTypes"] = self.config.speech_mark_types
            
            # Neural voice style (for conversational neural voices)
            if self.config.engine == "neural" and self.config.neural_voice_style:
                request_body["TextType"] = "ssml"
                request_body["Text"] = self._build_ssml_with_style(text, self.config.neural_voice_style)
            
            # Prepare request
            method = "POST"
            path = "/v1/speech"
            headers = await self._create_aws_headers(method, path, request_body)
            headers["Content-Type"] = "application/json"
            
            # Make API request
            response = await self.client.post(
                path,
                json=request_body,
                headers=headers
            )
            response.raise_for_status()
            
            # Get audio data
            audio_data = response.content
            
            # Extract metadata from response headers
            request_characters = int(response.headers.get("x-amzn-RequestCharacters", len(text)))
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Update metrics
            self._request_count += 1
            self._total_characters_synthesized += request_characters
            
            # Estimate duration based on speaking rate
            words = len(text.split())
            speaking_rate = 150  # Average words per minute
            estimated_duration = (words / speaking_rate) * 60
            
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
                    "character_count": request_characters,
                    "engine": self.config.engine,
                    "region": self.config.region,
                    "neural_voice_style": self.config.neural_voice_style
                }
            )
            
            self.logger.debug(f"Synthesized {request_characters} characters in {processing_time:.3f}s")
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
        
        Note: AWS Polly doesn't support true streaming synthesis,
        so this implementation synthesizes the full text and yields chunks.
        
        Args:
            text: Text to synthesize
            voice_id: Optional voice ID override
            
        Yields:
            Audio data chunks
        """
        if not self.client:
            raise RuntimeError("Provider not initialized")
        
        try:
            # Get full synthesis result
            result = await self.synthesize_speech(text, voice_id)
            
            if result.audio_data:
                # Yield audio in chunks
                chunk_size = self.config.chunk_size
                audio_data = result.audio_data
                
                for i in range(0, len(audio_data), chunk_size):
                    chunk = audio_data[i:i + chunk_size]
                    yield chunk
                    await asyncio.sleep(0.01)  # Small delay to simulate streaming
            
        except Exception as e:
            self.logger.error(f"Error in streaming synthesis: {e}")
            yield b""
    
    def _build_ssml(self, text: str) -> str:
        """Build basic SSML document for synthesis."""
        # Escape XML special characters
        text = text.replace("&", "&amp;")
        text = text.replace("<", "&lt;")
        text = text.replace(">", "&gt;")
        text = text.replace('"', "&quot;")
        text = text.replace("'", "&apos;")
        
        ssml_parts = ['<speak>']
        
        # Add prosody if configured
        prosody_attrs = []
        if self.config.speaking_rate != 1.0:
            rate_percent = int((self.config.speaking_rate - 1.0) * 100)
            rate_str = f"+{rate_percent}%" if rate_percent > 0 else f"{rate_percent}%"
            prosody_attrs.append(f'rate="{rate_str}"')
        
        if self.config.pitch != 0.0:
            pitch_percent = int(self.config.pitch * 100 / 12)  # Convert semitones to percentage
            pitch_str = f"+{pitch_percent}%" if pitch_percent > 0 else f"{pitch_percent}%"
            prosody_attrs.append(f'pitch="{pitch_str}"')
        
        if self.config.volume != 1.0:
            volume_db = 20 * (self.config.volume - 1.0)  # Convert to dB
            volume_str = f"+{volume_db:.1f}dB" if volume_db > 0 else f"{volume_db:.1f}dB"
            prosody_attrs.append(f'volume="{volume_str}"')
        
        if prosody_attrs:
            ssml_parts.append(f'<prosody {" ".join(prosody_attrs)}>')
            ssml_parts.append(text)
            ssml_parts.append('</prosody>')
        else:
            ssml_parts.append(text)
        
        ssml_parts.append('</speak>')
        
        return ''.join(ssml_parts)
    
    def _build_ssml_with_style(self, text: str, style: str) -> str:
        """Build SSML with neural voice style."""
        # Escape XML special characters
        text = text.replace("&", "&amp;")
        text = text.replace("<", "&lt;")
        text = text.replace(">", "&gt;")
        text = text.replace('"', "&quot;")
        text = text.replace("'", "&apos;")
        
        # Use amazon:domain for neural voice styles
        return f'<speak><amazon:domain name="{style}">{text}</amazon:domain></speak>'
    
    def _get_output_format(self) -> str:
        """Get AWS Polly output format string based on config."""
        format_map = {
            AudioFormat.MP3: "mp3",
            AudioFormat.OGG: "ogg_vorbis",
            AudioFormat.PCM: "pcm",
            AudioFormat.WAV: "pcm"  # WAV is PCM with headers
        }
        
        return format_map.get(self.config.audio_format, "mp3")
    
    def _map_language_code_to_enum(self, language_code: str) -> TTSLanguage:
        """Map AWS language code to our language enum."""
        lang_prefix = language_code.split('-')[0].lower()
        
        language_map = {
            "en": TTSLanguage.ENGLISH,
            "es": TTSLanguage.SPANISH,
            "fr": TTSLanguage.FRENCH,
            "de": TTSLanguage.GERMAN,
            "it": TTSLanguage.ITALIAN,
            "pt": TTSLanguage.PORTUGUESE,
            "ru": TTSLanguage.RUSSIAN,
            "ja": TTSLanguage.JAPANESE,
            "ko": TTSLanguage.KOREAN,
            "zh": TTSLanguage.CHINESE,
            "cmn": TTSLanguage.CHINESE  # Mandarin Chinese
        }
        
        return language_map.get(lang_prefix, TTSLanguage.ENGLISH)
    
    async def _create_aws_headers(
        self, 
        method: str, 
        path: str, 
        body: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """Create AWS Signature Version 4 headers."""
        # Get current time
        t = datetime.utcnow()
        amz_date = t.strftime('%Y%m%dT%H%M%SZ')
        date_stamp = t.strftime('%Y%m%d')
        
        # Create canonical request
        canonical_uri = path
        canonical_querystring = ''
        
        headers = {
            'host': f"polly.{self.config.region}.amazonaws.com",
            'x-amz-date': amz_date
        }
        
        if self.config.session_token:
            headers['x-amz-security-token'] = self.config.session_token
        
        # Create the canonical headers and signed headers
        canonical_headers = '\n'.join([f"{k}:{v}" for k, v in sorted(headers.items())]) + '\n'
        signed_headers = ';'.join(sorted(headers.keys()))
        
        # Create payload hash
        if body:
            payload = json.dumps(body, separators=(',', ':'))
        else:
            payload = ''
        
        payload_hash = hashlib.sha256(payload.encode('utf-8')).hexdigest()
        
        # Create canonical request
        canonical_request = '\n'.join([
            method,
            canonical_uri,
            canonical_querystring,
            canonical_headers,
            signed_headers,
            payload_hash
        ])
        
        # Create the string to sign
        algorithm = 'AWS4-HMAC-SHA256'
        credential_scope = f"{date_stamp}/{self.config.region}/{self._service}/aws4_request"
        string_to_sign = '\n'.join([
            algorithm,
            amz_date,
            credential_scope,
            hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()
        ])
        
        # Create signing key
        def sign(key, msg):
            return hmac.new(key, msg.encode('utf-8'), hashlib.sha256).digest()
        
        k_date = sign(f"AWS4{self.config.secret_access_key}".encode('utf-8'), date_stamp)
        k_region = sign(k_date, self.config.region)
        k_service = sign(k_region, self._service)
        k_signing = sign(k_service, 'aws4_request')
        
        # Create signature
        signature = hmac.new(k_signing, string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()
        
        # Create authorization header
        authorization_header = (
            f"{algorithm} Credential={self.config.access_key_id}/{credential_scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}"
        )
        
        # Add authorization to headers
        headers['Authorization'] = authorization_header
        
        return headers
    
    async def _test_connection(self) -> None:
        """Test the AWS Polly connection."""
        try:
            # Try to list voices as a connection test
            method = "GET"
            path = "/v1/voices"
            headers = await self._create_aws_headers(method, path, {})
            
            response = await self.client.get(path, headers=headers)
            
            if response.status_code in [200, 403]:  # 403 means auth works but no permission
                return
            else:
                raise Exception(f"Unexpected status code: {response.status_code}")
                
        except Exception as e:
            raise Exception(f"AWS Polly connection test failed: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of the AWS Polly TTS provider.
        
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
                    "engine": self.config.engine,
                    "voice_id": self.config.voice_id,
                    "audio_format": self.config.audio_format.value,
                    "sample_rate": self.config.sample_rate,
                    "neural_voice_style": self.config.neural_voice_style
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
            "engine": self.config.engine,
            "neural_voice_style": self.config.neural_voice_style,
            "lexicon_count": len(self.config.lexicon_names),
            "speech_mark_types": self.config.speech_mark_types
        })
        return base_metrics


# Register the provider
TTSProviderFactory.register_provider("aws_polly", AWSPollyTTSProvider)