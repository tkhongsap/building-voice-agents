"""
Base Speech-to-Text (STT) Provider Abstraction

This module defines the abstract base class that all STT providers must implement,
ensuring a consistent interface across different speech recognition services.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, AsyncIterator, Union
from enum import Enum
import asyncio
import logging

logger = logging.getLogger(__name__)


class STTLanguage(Enum):
    """Supported languages for speech recognition."""
    AUTO = "auto"  # Auto-detect language
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


class STTQuality(Enum):
    """Quality levels for speech recognition."""
    LOW = "low"      # Fastest, lowest quality
    MEDIUM = "medium"  # Balanced speed and quality
    HIGH = "high"    # Highest quality, slower


class STTResult:
    """Represents the result of speech recognition."""
    
    def __init__(
        self,
        text: str,
        confidence: float = 0.0,
        is_final: bool = True,
        language: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        alternatives: Optional[list] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.text = text
        self.confidence = confidence
        self.is_final = is_final
        self.language = language
        self.start_time = start_time
        self.end_time = end_time
        self.alternatives = alternatives or []
        self.metadata = metadata or {}
    
    def __str__(self) -> str:
        return f"STTResult(text='{self.text}', confidence={self.confidence}, final={self.is_final})"
    
    def __repr__(self) -> str:
        return self.__str__()


class STTConfig:
    """Configuration for STT providers."""
    
    def __init__(
        self,
        language: STTLanguage = STTLanguage.AUTO,
        quality: STTQuality = STTQuality.MEDIUM,
        enable_interim_results: bool = True,
        enable_automatic_punctuation: bool = True,
        enable_profanity_filter: bool = False,
        sample_rate: int = 16000,
        chunk_size: int = 1024,
        silence_timeout: float = 2.0,
        phrase_timeout: float = 5.0,
        max_alternatives: int = 1,
        custom_vocabulary: Optional[list] = None,
        **kwargs
    ):
        self.language = language
        self.quality = quality
        self.enable_interim_results = enable_interim_results
        self.enable_automatic_punctuation = enable_automatic_punctuation
        self.enable_profanity_filter = enable_profanity_filter
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.silence_timeout = silence_timeout
        self.phrase_timeout = phrase_timeout
        self.max_alternatives = max_alternatives
        self.custom_vocabulary = custom_vocabulary or []
        
        # Store any additional provider-specific config
        for key, value in kwargs.items():
            setattr(self, key, value)


class BaseSTTProvider(ABC):
    """
    Abstract base class for all Speech-to-Text providers.
    
    All STT implementations must inherit from this class and implement
    the required methods to ensure consistent behavior across providers.
    """
    
    def __init__(self, config: STTConfig):
        self.config = config
        self._is_streaming = False
        self._stream_task = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of this STT provider."""
        pass
    
    @property
    @abstractmethod
    def supported_languages(self) -> list[STTLanguage]:
        """Return list of languages supported by this provider."""
        pass
    
    @property
    @abstractmethod
    def supported_sample_rates(self) -> list[int]:
        """Return list of supported audio sample rates."""
        pass
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the STT provider (authenticate, setup connections, etc.)."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources and close connections."""
        pass
    
    @abstractmethod
    async def transcribe_audio(self, audio_data: bytes) -> STTResult:
        """
        Transcribe audio data (batch processing).
        
        Args:
            audio_data: Raw audio bytes
            
        Returns:
            STTResult with transcription
        """
        pass
    
    @abstractmethod
    async def start_streaming(self) -> None:
        """Start streaming transcription mode."""
        pass
    
    @abstractmethod
    async def stop_streaming(self) -> None:
        """Stop streaming transcription mode."""
        pass
    
    @abstractmethod
    async def stream_audio(self, audio_chunk: bytes) -> None:
        """
        Send audio chunk for streaming transcription.
        
        Args:
            audio_chunk: Audio data chunk
        """
        pass
    
    @abstractmethod
    async def get_streaming_results(self) -> AsyncIterator[STTResult]:
        """
        Get streaming transcription results.
        
        Yields:
            STTResult objects as they become available
        """
        pass
    
    async def transcribe_file(self, file_path: str) -> STTResult:
        """
        Transcribe audio from file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            STTResult with transcription
        """
        try:
            with open(file_path, 'rb') as f:
                audio_data = f.read()
            return await self.transcribe_audio(audio_data)
        except Exception as e:
            self.logger.error(f"Error transcribing file {file_path}: {e}")
            raise
    
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
        
        if self.config.sample_rate not in self.supported_sample_rates:
            raise ValueError(f"Sample rate {self.config.sample_rate} not supported by {self.provider_name}")
        
        if self.config.chunk_size <= 0:
            raise ValueError("Chunk size must be positive")
        
        if self.config.silence_timeout <= 0:
            raise ValueError("Silence timeout must be positive")
        
        return True
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of the STT provider.
        
        Returns:
            Dictionary with health status information
        """
        try:
            await self.initialize()
            return {
                "status": "healthy",
                "provider": self.provider_name,
                "config": {
                    "language": self.config.language.value,
                    "quality": self.config.quality.value,
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
        return {
            "provider": self.provider_name,
            "is_streaming": self._is_streaming,
            "config": {
                "language": self.config.language.value,
                "quality": self.config.quality.value,
                "sample_rate": self.config.sample_rate,
                "chunk_size": self.config.chunk_size
            }
        }
    
    async def __aenter__(self):
        """Context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.cleanup()


class STTProviderFactory:
    """Factory for creating STT provider instances."""
    
    _providers: Dict[str, type] = {}
    
    @classmethod
    def register_provider(cls, name: str, provider_class: type):
        """Register a new STT provider."""
        cls._providers[name] = provider_class
    
    @classmethod
    def create_provider(cls, name: str, config: STTConfig) -> BaseSTTProvider:
        """Create an instance of the specified STT provider."""
        if name not in cls._providers:
            raise ValueError(f"Unknown STT provider: {name}")
        
        provider_class = cls._providers[name]
        return provider_class(config)
    
    @classmethod
    def list_providers(cls) -> list[str]:
        """List all registered STT providers."""
        return list(cls._providers.keys())