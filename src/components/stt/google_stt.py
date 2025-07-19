"""
Google Cloud Speech-to-Text Provider Implementation

This module implements the Google Cloud Speech-to-Text STT provider with streaming
recognition and enhanced accuracy features.
"""

import asyncio
import logging
import json
import time
import io
from typing import Optional, Dict, Any, AsyncIterator, List
import httpx

from .base_stt import (
    BaseSTTProvider, STTResult, STTConfig, STTLanguage, STTQuality,
    STTProviderFactory
)

logger = logging.getLogger(__name__)


class GoogleSTTConfig(STTConfig):
    """Configuration specific to Google Cloud Speech-to-Text."""
    
    def __init__(
        self,
        credentials_path: Optional[str] = None,
        project_id: Optional[str] = None,
        model: str = "latest_long",  # latest_long, latest_short, command_and_search, etc.
        use_enhanced: bool = True,
        enable_word_time_offsets: bool = True,
        enable_word_confidence: bool = True,
        enable_speaker_diarization: bool = False,
        diarization_speaker_count: int = 2,
        boost_phrases: Optional[List[str]] = None,
        speech_contexts: Optional[List[Dict[str, Any]]] = None,
        adaptation_phrase_set_references: Optional[List[str]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.credentials_path = credentials_path
        self.project_id = project_id
        self.model = model
        self.use_enhanced = use_enhanced
        self.enable_word_time_offsets = enable_word_time_offsets
        self.enable_word_confidence = enable_word_confidence
        self.enable_speaker_diarization = enable_speaker_diarization
        self.diarization_speaker_count = diarization_speaker_count
        self.boost_phrases = boost_phrases or []
        self.speech_contexts = speech_contexts or []
        self.adaptation_phrase_set_references = adaptation_phrase_set_references or []


class GoogleSTTProvider(BaseSTTProvider):
    """
    Google Cloud Speech-to-Text STT provider implementation.
    
    Provides speech-to-text functionality using Google Cloud Speech-to-Text API
    with support for streaming recognition and enhanced accuracy features.
    """
    
    def __init__(self, config: GoogleSTTConfig):
        super().__init__(config)
        self.config: GoogleSTTConfig = config
        self.client: Optional[httpx.AsyncClient] = None
        self._access_token: Optional[str] = None
        self._token_expiry: float = 0
        self._streaming_active = False
        self._stream_session_id = None
        self._recognition_config = None
    
    @property
    def provider_name(self) -> str:
        """Return the name of this STT provider."""
        return "google_speech"
    
    @property
    def supported_languages(self) -> List[STTLanguage]:
        """Return list of languages supported by Google Speech."""
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
        return [8000, 16000, 22050, 24000, 32000, 44100, 48000]
    
    async def initialize(self) -> None:
        """Initialize the Google STT provider."""
        try:
            # Set up authentication
            await self._setup_authentication()
            
            # Initialize HTTP client
            self.client = httpx.AsyncClient(
                base_url="https://speech.googleapis.com/v1",
                timeout=30.0
            )
            
            # Prepare recognition configuration
            self._recognition_config = self._build_recognition_config()
            
            # Test the connection
            await self._test_connection()
            
            self.logger.info("Google STT provider initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Google STT: {e}")
            raise
    
    async def cleanup(self) -> None:
        """Clean up resources and close connections."""
        self._streaming_active = False
        
        if self.client:
            await self.client.aclose()
            self.client = None
        
        self.logger.info("Google STT provider cleanup completed")
    
    async def transcribe_audio(self, audio_data: bytes) -> STTResult:
        """
        Transcribe audio data using Google Cloud Speech-to-Text.
        
        Args:
            audio_data: Raw audio bytes
            
        Returns:
            STTResult with transcription
        """
        if not self.client:
            raise RuntimeError("Provider not initialized")
        
        try:
            start_time = time.time()
            
            # Ensure we have a valid access token
            await self._ensure_valid_token()
            
            # Prepare request data
            request_data = {
                "config": self._recognition_config,
                "audio": {
                    "content": self._encode_audio_base64(audio_data)
                }
            }
            
            # Make API request
            headers = {"Authorization": f"Bearer {self._access_token}"}
            response = await self.client.post(
                "/speech:recognize",
                json=request_data,
                headers=headers
            )
            response.raise_for_status()
            
            # Parse response
            result_data = response.json()
            
            # Extract recognition results
            results = result_data.get("results", [])
            
            if results:
                # Get the top result
                top_result = results[0]
                alternatives = top_result.get("alternatives", [])
                
                if alternatives:
                    best_alternative = alternatives[0]
                    text = best_alternative.get("transcript", "")
                    confidence = best_alternative.get("confidence", 0.0)
                    
                    # Extract word-level information
                    words = best_alternative.get("words", [])
                    word_info = []
                    for word in words:
                        word_data = {
                            "word": word.get("word", ""),
                            "confidence": word.get("confidence", 0.0),
                            "start_time": self._parse_duration(word.get("startTime", "0s")),
                            "end_time": self._parse_duration(word.get("endTime", "0s"))
                        }
                        word_info.append(word_data)
                    
                    # Extract alternatives
                    alternative_results = []
                    for alt in alternatives[:self.config.max_alternatives]:
                        alt_data = {
                            "text": alt.get("transcript", ""),
                            "confidence": alt.get("confidence", 0.0)
                        }
                        if "words" in alt:
                            alt_data["words"] = alt["words"]
                        alternative_results.append(alt_data)
                    
                    # Extract speaker information if diarization is enabled
                    speaker_info = None
                    if self.config.enable_speaker_diarization and "words" in best_alternative:
                        speaker_info = self._extract_speaker_info(best_alternative["words"])
                    
                else:
                    text = ""
                    confidence = 0.0
                    word_info = []
                    alternative_results = []
                    speaker_info = None
            else:
                text = ""
                confidence = 0.0
                word_info = []
                alternative_results = []
                speaker_info = None
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Calculate duration from word timing
            duration = 0.0
            if word_info:
                duration = word_info[-1]["end_time"]
            
            result = STTResult(
                text=text,
                confidence=confidence,
                is_final=True,
                language=self._get_google_language_code(),
                start_time=0.0,
                end_time=duration,
                alternatives=alternative_results,
                metadata={
                    "provider": self.provider_name,
                    "model": self.config.model,
                    "processing_time": processing_time,
                    "audio_length": len(audio_data),
                    "word_info": word_info,
                    "speaker_info": speaker_info,
                    "use_enhanced": self.config.use_enhanced
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
        self._stream_session_id = str(int(time.time() * 1000))
        
        self.logger.info("Started Google STT streaming mode")
    
    async def stop_streaming(self) -> None:
        """Stop streaming transcription mode."""
        self._streaming_active = False
        self._stream_session_id = None
        
        self.logger.info("Stopped Google STT streaming mode")
    
    async def stream_audio(self, audio_chunk: bytes) -> None:
        """
        Send audio chunk for streaming transcription.
        
        Args:
            audio_chunk: Audio data chunk
        """
        if not self._streaming_active:
            await self.start_streaming()
        
        # Note: Google Cloud Speech streaming requires persistent connection
        # This is a simplified implementation that could be enhanced with
        # bidirectional streaming for better real-time performance
        pass
    
    async def get_streaming_results(self) -> AsyncIterator[STTResult]:
        """
        Get streaming transcription results.
        
        Yields:
            STTResult objects as they become available
        """
        # Implementation for streaming recognition using Google's streaming API
        # This would involve setting up a bidirectional gRPC stream
        
        if not self._streaming_active:
            return
        
        try:
            # For now, simulate streaming by processing in chunks
            # In a full implementation, this would use the streaming recognize API
            while self._streaming_active:
                await asyncio.sleep(0.1)
                # Placeholder for actual streaming implementation
                
        except Exception as e:
            self.logger.error(f"Error in streaming results: {e}")
            yield STTResult(
                text="",
                confidence=0.0,
                is_final=False,
                metadata={"error": str(e), "provider": self.provider_name}
            )
    
    def _build_recognition_config(self) -> Dict[str, Any]:
        """Build recognition configuration for Google Speech API."""
        config = {
            "encoding": "LINEAR16",  # Assuming 16-bit PCM
            "sampleRateHertz": self.config.sample_rate,
            "languageCode": self._get_google_language_code(),
            "model": self.config.model,
            "useEnhanced": self.config.use_enhanced,
            "enableWordTimeOffsets": self.config.enable_word_time_offsets,
            "enableWordConfidence": self.config.enable_word_confidence,
            "enableAutomaticPunctuation": self.config.enable_automatic_punctuation,
            "profanityFilter": self.config.enable_profanity_filter,
            "maxAlternatives": self.config.max_alternatives
        }
        
        # Add language detection for auto mode
        if self.config.language == STTLanguage.AUTO:
            config["alternativeLanguageCodes"] = [
                "en-US", "es-ES", "fr-FR", "de-DE", "it-IT",
                "pt-BR", "ru-RU", "ja-JP", "ko-KR", "zh-CN"
            ]
            config["languageCode"] = "en-US"  # Primary language for auto-detection
        
        # Add speaker diarization if enabled
        if self.config.enable_speaker_diarization:
            config["diarizationConfig"] = {
                "enableSpeakerDiarization": True,
                "minSpeakerCount": 1,
                "maxSpeakerCount": self.config.diarization_speaker_count
            }
        
        # Add speech contexts for improved accuracy
        if self.config.speech_contexts or self.config.boost_phrases:
            contexts = list(self.config.speech_contexts)
            
            if self.config.boost_phrases:
                contexts.append({
                    "phrases": self.config.boost_phrases,
                    "boost": 10.0  # Moderate boost
                })
            
            config["speechContexts"] = contexts
        
        # Add adaptation if configured
        if self.config.adaptation_phrase_set_references:
            config["adaptation"] = {
                "phraseSetReferences": self.config.adaptation_phrase_set_references
            }
        
        return config
    
    def _get_google_language_code(self) -> str:
        """Convert internal language enum to Google language code."""
        language_mapping = {
            STTLanguage.AUTO: "en-US",  # Will be handled by alternative languages
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
    
    def _encode_audio_base64(self, audio_data: bytes) -> str:
        """Encode audio data to base64 for API request."""
        import base64
        return base64.b64encode(audio_data).decode('utf-8')
    
    def _parse_duration(self, duration_str: str) -> float:
        """Parse Google duration string (e.g., '1.500s') to float."""
        if duration_str.endswith('s'):
            return float(duration_str[:-1])
        return 0.0
    
    def _extract_speaker_info(self, words: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract speaker diarization information."""
        speakers = {}
        current_speaker = None
        speaker_segments = []
        
        for word in words:
            speaker_tag = word.get("speakerTag", 1)
            
            if current_speaker != speaker_tag:
                if current_speaker is not None:
                    # End current segment
                    speaker_segments[-1]["end_time"] = self._parse_duration(
                        words[words.index(word) - 1].get("endTime", "0s")
                    )
                
                # Start new segment
                speaker_segments.append({
                    "speaker": speaker_tag,
                    "start_time": self._parse_duration(word.get("startTime", "0s")),
                    "end_time": None  # Will be set when segment ends
                })
                current_speaker = speaker_tag
            
            # Track words per speaker
            if speaker_tag not in speakers:
                speakers[speaker_tag] = {"word_count": 0, "total_confidence": 0.0}
            
            speakers[speaker_tag]["word_count"] += 1
            speakers[speaker_tag]["total_confidence"] += word.get("confidence", 0.0)
        
        # Close last segment
        if speaker_segments and words:
            speaker_segments[-1]["end_time"] = self._parse_duration(
                words[-1].get("endTime", "0s")
            )
        
        # Calculate average confidence per speaker
        for speaker_id, info in speakers.items():
            if info["word_count"] > 0:
                info["average_confidence"] = info["total_confidence"] / info["word_count"]
        
        return {
            "speakers": speakers,
            "segments": speaker_segments,
            "speaker_count": len(speakers)
        }
    
    async def _setup_authentication(self) -> None:
        """Set up Google Cloud authentication."""
        try:
            # Try to get credentials from environment or service account file
            credentials_path = self.config.credentials_path
            if not credentials_path:
                import os
                credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            
            if credentials_path:
                # Load service account credentials
                with open(credentials_path, 'r') as f:
                    credentials = json.load(f)
                
                self.config.project_id = self.config.project_id or credentials.get("project_id")
                
                # Get access token using service account
                await self._get_access_token_from_service_account(credentials)
            else:
                # Try to get access token from environment
                import os
                access_token = os.getenv("GOOGLE_CLOUD_ACCESS_TOKEN")
                if access_token:
                    self._access_token = access_token
                    self._token_expiry = time.time() + 3600  # Assume 1 hour expiry
                else:
                    raise ValueError("Google Cloud credentials not found")
                    
        except Exception as e:
            raise Exception(f"Failed to setup Google Cloud authentication: {e}")
    
    async def _get_access_token_from_service_account(self, credentials: Dict[str, Any]) -> None:
        """Get access token using service account credentials."""
        try:
            import jwt
            import time
            
            # Create JWT token
            now = int(time.time())
            payload = {
                "iss": credentials["client_email"],
                "scope": "https://www.googleapis.com/auth/cloud-platform",
                "aud": "https://oauth2.googleapis.com/token",
                "iat": now,
                "exp": now + 3600
            }
            
            # Sign JWT with private key
            token = jwt.encode(payload, credentials["private_key"], algorithm="RS256")
            
            # Exchange JWT for access token
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://oauth2.googleapis.com/token",
                    data={
                        "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                        "assertion": token
                    }
                )
                response.raise_for_status()
                
                token_data = response.json()
                self._access_token = token_data["access_token"]
                self._token_expiry = time.time() + token_data.get("expires_in", 3600)
                
        except ImportError:
            raise Exception("PyJWT library is required for service account authentication")
        except Exception as e:
            raise Exception(f"Failed to get access token: {e}")
    
    async def _ensure_valid_token(self) -> None:
        """Ensure we have a valid access token."""
        if not self._access_token or time.time() >= self._token_expiry - 300:  # Refresh 5 min early
            await self._setup_authentication()
    
    async def _test_connection(self) -> None:
        """Test the Google Cloud Speech API connection."""
        try:
            await self._ensure_valid_token()
            
            # Test with a minimal request
            headers = {"Authorization": f"Bearer {self._access_token}"}
            response = await self.client.get("/operations", headers=headers)
            
            if response.status_code in [200, 403]:  # 403 is OK, means auth works but no permission for operations
                return
            else:
                raise Exception(f"Unexpected status code: {response.status_code}")
                
        except Exception as e:
            raise Exception(f"Google Cloud Speech API connection test failed: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of the Google STT provider.
        
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
                "token_valid": self._access_token is not None and time.time() < self._token_expiry,
                "config": {
                    "model": self.config.model,
                    "language": self.config.language.value,
                    "use_enhanced": self.config.use_enhanced,
                    "sample_rate": self.config.sample_rate,
                    "project_id": self.config.project_id
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
            "use_enhanced": self.config.use_enhanced,
            "streaming_active": self._streaming_active,
            "token_expires_in": max(0, self._token_expiry - time.time()) if self._token_expiry else 0,
            "speaker_diarization_enabled": self.config.enable_speaker_diarization,
            "boost_phrases_count": len(self.config.boost_phrases),
            "speech_contexts_count": len(self.config.speech_contexts)
        })
        return base_metrics


# Register the provider
STTProviderFactory.register_provider("google", GoogleSTTProvider)