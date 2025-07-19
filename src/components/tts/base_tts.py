"""
Base Text-to-Speech (TTS) Provider Abstraction

This module defines the abstract base class that all TTS providers must implement,
ensuring a consistent interface across different text-to-speech services.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, AsyncIterator, Union, List
from enum import Enum
import asyncio
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class TTSVoice(Enum):
    """Voice types for text-to-speech."""
    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"


class TTSLanguage(Enum):
    """Supported languages for text-to-speech."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    JAPANESE = "ja"
    KOREAN = "ko"
    CHINESE = "zh"


class TTSQuality(Enum):
    """Quality levels for text-to-speech."""
    LOW = "low"      # Fastest, lowest quality
    MEDIUM = "medium"  # Balanced speed and quality
    HIGH = "high"    # Highest quality, slower


class AudioFormat(Enum):
    """Supported audio formats."""
    WAV = "wav"
    MP3 = "mp3"
    OGG = "ogg"
    FLAC = "flac"
    PCM = "pcm"


@dataclass
class Voice:
    """Represents a voice option for TTS."""
    id: str
    name: str
    language: TTSLanguage
    gender: TTSVoice
    description: Optional[str] = None
    sample_rate: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert voice to dictionary format."""
        return {
            "id": self.id,
            "name": self.name,
            "language": self.language.value,
            "gender": self.gender.value,
            "description": self.description,
            "sample_rate": self.sample_rate,
            "metadata": self.metadata or {}
        }


@dataclass
class TTSResult:
    """Represents the result of text-to-speech synthesis."""
    audio_data: bytes
    format: AudioFormat
    sample_rate: int
    duration: Optional[float] = None
    text: Optional[str] = None
    voice_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __len__(self) -> int:
        return len(self.audio_data)
    
    def save_to_file(self, file_path: str) -> None:
        """Save audio data to file."""
        with open(file_path, 'wb') as f:
            f.write(self.audio_data)


class TTSConfig:
    """Configuration for TTS providers."""
    
    def __init__(
        self,
        voice_id: Optional[str] = None,
        language: TTSLanguage = TTSLanguage.ENGLISH,
        quality: TTSQuality = TTSQuality.MEDIUM,
        audio_format: AudioFormat = AudioFormat.WAV,
        sample_rate: int = 22050,
        speaking_rate: float = 1.0,  # Speed multiplier (0.5 = slow, 2.0 = fast)
        pitch: float = 0.0,  # Pitch adjustment in semitones (-12 to +12)
        volume: float = 1.0,  # Volume multiplier (0.0 to 2.0)
        enable_streaming: bool = True,
        chunk_size: int = 1024,
        enable_ssml: bool = True,  # Support Speech Synthesis Markup Language
        stability: float = 0.5,  # Voice stability (provider-specific)
        similarity_boost: float = 0.5,  # Voice similarity boost (provider-specific)
        style: Optional[str] = None,  # Voice style (provider-specific)
        **kwargs
    ):
        self.voice_id = voice_id
        self.language = language
        self.quality = quality
        self.audio_format = audio_format
        self.sample_rate = sample_rate
        self.speaking_rate = speaking_rate
        self.pitch = pitch
        self.volume = volume
        self.enable_streaming = enable_streaming
        self.chunk_size = chunk_size
        self.enable_ssml = enable_ssml
        self.stability = stability
        self.similarity_boost = similarity_boost
        self.style = style
        
        # Store any additional provider-specific config
        for key, value in kwargs.items():
            setattr(self, key, value)


class BaseTTSProvider(ABC):
    """
    Abstract base class for all Text-to-Speech providers.
    
    All TTS implementations must inherit from this class and implement
    the required methods to ensure consistent behavior across providers.
    """
    
    def __init__(self, config: TTSConfig):
        self.config = config
        self._is_streaming = False
        self._stream_task = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of this TTS provider."""
        pass
    
    @property
    @abstractmethod
    def supported_languages(self) -> List[TTSLanguage]:
        """Return list of languages supported by this provider."""
        pass
    
    @property
    @abstractmethod
    def supported_formats(self) -> List[AudioFormat]:
        """Return list of audio formats supported by this provider."""
        pass
    
    @property
    @abstractmethod
    def supported_sample_rates(self) -> List[int]:
        """Return list of supported audio sample rates."""
        pass
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the TTS provider (authenticate, setup connections, etc.)."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources and close connections."""
        pass
    
    @abstractmethod
    async def get_voices(self, language: Optional[TTSLanguage] = None) -> List[Voice]:
        """
        Get available voices for the specified language.
        
        Args:
            language: Optional language filter
            
        Returns:
            List of available voices
        """
        pass
    
    @abstractmethod
    async def synthesize_speech(self, text: str, voice_id: Optional[str] = None) -> TTSResult:
        """
        Synthesize speech from text (batch processing).
        
        Args:
            text: Text to synthesize
            voice_id: Optional voice ID override
            
        Returns:
            TTSResult with audio data
        """
        pass
    
    @abstractmethod
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
        pass
    
    async def speak(self, text: str, voice_id: Optional[str] = None) -> TTSResult:
        """
        High-level method to synthesize speech.
        
        Args:
            text: Text to speak
            voice_id: Optional voice ID override
            
        Returns:
            TTSResult with audio data
        """
        effective_voice_id = voice_id or self.config.voice_id
        return await self.synthesize_speech(text, effective_voice_id)
    
    async def speak_streaming(
        self, 
        text: str, 
        voice_id: Optional[str] = None
    ) -> AsyncIterator[bytes]:
        """
        High-level method to synthesize speech with streaming.
        
        Args:
            text: Text to speak
            voice_id: Optional voice ID override
            
        Yields:
            Audio data chunks
        """
        effective_voice_id = voice_id or self.config.voice_id
        async for chunk in self.synthesize_streaming(text, effective_voice_id):
            yield chunk
    
    async def synthesize_ssml(self, ssml: str, voice_id: Optional[str] = None) -> TTSResult:
        """
        Synthesize speech from SSML markup.
        
        Args:
            ssml: SSML markup to synthesize
            voice_id: Optional voice ID override
            
        Returns:
            TTSResult with audio data
        """
        if not self.config.enable_ssml:
            raise ValueError("SSML is not enabled in configuration")
        
        return await self.synthesize_speech(ssml, voice_id)
    
    async def get_voice_by_name(self, name: str, language: Optional[TTSLanguage] = None) -> Optional[Voice]:
        """
        Get a voice by name.
        
        Args:
            name: Voice name to search for
            language: Optional language filter
            
        Returns:
            Voice object if found, None otherwise
        """
        voices = await self.get_voices(language)
        for voice in voices:
            if voice.name.lower() == name.lower():
                return voice
        return None
    
    async def get_default_voice(self, language: Optional[TTSLanguage] = None) -> Optional[Voice]:
        """
        Get the default voice for a language.
        
        Args:
            language: Language to get default voice for
            
        Returns:
            Default voice object if available
        """
        target_language = language or self.config.language
        voices = await self.get_voices(target_language)
        
        if not voices:
            return None
        
        # Return first voice or preferred default
        return voices[0]
    
    def estimate_speech_duration(self, text: str, speaking_rate: Optional[float] = None) -> float:
        """
        Estimate the duration of synthesized speech.
        
        Args:
            text: Text to estimate duration for
            speaking_rate: Optional speaking rate override
            
        Returns:
            Estimated duration in seconds
        """
        # Basic estimation: ~150 words per minute for normal speech
        rate = speaking_rate or self.config.speaking_rate
        words = len(text.split())
        base_duration = (words / 150) * 60  # seconds
        
        # Adjust for speaking rate
        adjusted_duration = base_duration / rate
        
        return max(0.1, adjusted_duration)  # Minimum 0.1 seconds
    
    def validate_config(self) -> bool:
        """
        Validate the current configuration.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        if self.config.language not in self.supported_languages:
            raise ValueError(f"Language {self.config.language} not supported by {self.provider_name}")
        
        if self.config.audio_format not in self.supported_formats:
            raise ValueError(f"Audio format {self.config.audio_format} not supported by {self.provider_name}")
        
        if self.config.sample_rate not in self.supported_sample_rates:
            raise ValueError(f"Sample rate {self.config.sample_rate} not supported by {self.provider_name}")
        
        if not 0.1 <= self.config.speaking_rate <= 4.0:
            raise ValueError("Speaking rate must be between 0.1 and 4.0")
        
        if not -12 <= self.config.pitch <= 12:
            raise ValueError("Pitch must be between -12 and +12 semitones")
        
        if not 0.0 <= self.config.volume <= 2.0:
            raise ValueError("Volume must be between 0.0 and 2.0")
        
        return True
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of the TTS provider.
        
        Returns:
            Dictionary with health status information
        """
        try:
            await self.initialize()
            
            # Test synthesis with short text
            test_result = await self.synthesize_speech("Hello, this is a test.")
            
            return {
                "status": "healthy",
                "provider": self.provider_name,
                "config": {
                    "language": self.config.language.value,
                    "format": self.config.audio_format.value,
                    "sample_rate": self.config.sample_rate
                },
                "test_synthesis_size": len(test_result.audio_data)
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
        return {
            "provider": self.provider_name,
            "is_streaming": self._is_streaming,
            "config": {
                "language": self.config.language.value,
                "quality": self.config.quality.value,
                "format": self.config.audio_format.value,
                "sample_rate": self.config.sample_rate,
                "speaking_rate": self.config.speaking_rate,
                "voice_id": self.config.voice_id
            }
        }
    
    async def __aenter__(self):
        """Context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.cleanup()


class TTSProviderFactory:
    """Factory for creating TTS provider instances."""
    
    _providers: Dict[str, type] = {}
    
    @classmethod
    def register_provider(cls, name: str, provider_class: type):
        """Register a new TTS provider."""
        cls._providers[name] = provider_class
    
    @classmethod
    def create_provider(cls, name: str, config: TTSConfig) -> BaseTTSProvider:
        """Create an instance of the specified TTS provider."""
        if name not in cls._providers:
            raise ValueError(f"Unknown TTS provider: {name}")
        
        provider_class = cls._providers[name]
        return provider_class(config)
    
    @classmethod
    def list_providers(cls) -> List[str]:
        """List all registered TTS providers."""
        return list(cls._providers.keys())