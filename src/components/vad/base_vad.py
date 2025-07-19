"""
Base Voice Activity Detection (VAD) Provider Abstraction

This module defines the abstract base class that all VAD providers must implement,
ensuring a consistent interface across different voice activity detection services.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, AsyncIterator, Union, List, Callable
from enum import Enum
import asyncio
import logging
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


class VADSensitivity(Enum):
    """Sensitivity levels for voice activity detection."""
    LOW = "low"          # Less sensitive, fewer false positives
    MEDIUM = "medium"    # Balanced sensitivity
    HIGH = "high"        # More sensitive, more false positives
    ADAPTIVE = "adaptive" # Automatically adjusts based on environment


class VADState(Enum):
    """Voice activity states."""
    SILENCE = "silence"
    SPEECH = "speech"
    UNCERTAIN = "uncertain"


@dataclass
class VADResult:
    """Represents the result of voice activity detection."""
    is_speech: bool
    confidence: float
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration: Optional[float] = None
    state: VADState = VADState.SILENCE
    volume: Optional[float] = None  # Audio volume level (0.0 to 1.0)
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.is_speech:
            self.state = VADState.SPEECH
        elif self.confidence < 0.3:
            self.state = VADState.UNCERTAIN
        else:
            self.state = VADState.SILENCE
    
    def __str__(self) -> str:
        return f"VADResult(speech={self.is_speech}, confidence={self.confidence:.2f}, state={self.state.value})"


@dataclass
class VADSegment:
    """Represents a segment of audio with voice activity information."""
    start_time: float
    end_time: float
    is_speech: bool
    confidence: float
    audio_data: Optional[bytes] = None
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


class VADConfig:
    """Configuration for VAD providers."""
    
    def __init__(
        self,
        sensitivity: VADSensitivity = VADSensitivity.MEDIUM,
        sample_rate: int = 16000,
        chunk_size: int = 1024,
        min_speech_duration: float = 0.1,  # Minimum duration to consider speech
        min_silence_duration: float = 0.3,  # Minimum silence before considering end of speech
        speech_threshold: float = 0.5,     # Confidence threshold for speech detection
        aggressiveness: int = 1,           # Provider-specific aggressiveness (0-3)
        frame_duration_ms: int = 30,       # Frame duration in milliseconds
        lookback_frames: int = 5,          # Number of frames to look back for context
        lookahead_frames: int = 3,         # Number of frames to look ahead
        enable_noise_reduction: bool = True,
        enable_automatic_gain_control: bool = True,
        volume_threshold: float = 0.01,    # Minimum volume threshold
        **kwargs
    ):
        self.sensitivity = sensitivity
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.min_speech_duration = min_speech_duration
        self.min_silence_duration = min_silence_duration
        self.speech_threshold = speech_threshold
        self.aggressiveness = aggressiveness
        self.frame_duration_ms = frame_duration_ms
        self.lookback_frames = lookback_frames
        self.lookahead_frames = lookahead_frames
        self.enable_noise_reduction = enable_noise_reduction
        self.enable_automatic_gain_control = enable_automatic_gain_control
        self.volume_threshold = volume_threshold
        
        # Store any additional provider-specific config
        for key, value in kwargs.items():
            setattr(self, key, value)


class BaseVADProvider(ABC):
    """
    Abstract base class for all Voice Activity Detection providers.
    
    All VAD implementations must inherit from this class and implement
    the required methods to ensure consistent behavior across providers.
    """
    
    def __init__(self, config: VADConfig):
        self.config = config
        self._is_processing = False
        self._speech_segments: List[VADSegment] = []
        self._current_segment_start: Optional[float] = None
        self._last_speech_time: Optional[float] = None
        self._frame_buffer: List[bytes] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Callbacks for speech events
        self._on_speech_start: Optional[Callable] = None
        self._on_speech_end: Optional[Callable] = None
        self._on_speech_segment: Optional[Callable] = None
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of this VAD provider."""
        pass
    
    @property
    @abstractmethod
    def supported_sample_rates(self) -> List[int]:
        """Return list of supported audio sample rates."""
        pass
    
    @property
    @abstractmethod
    def min_chunk_size(self) -> int:
        """Return minimum chunk size for processing."""
        pass
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the VAD provider."""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources and close connections."""
        pass
    
    @abstractmethod
    async def detect_voice_activity(self, audio_chunk: bytes) -> VADResult:
        """
        Detect voice activity in an audio chunk.
        
        Args:
            audio_chunk: Raw audio bytes
            
        Returns:
            VADResult with detection information
        """
        pass
    
    @abstractmethod
    async def process_audio_stream(self, audio_stream: AsyncIterator[bytes]) -> AsyncIterator[VADResult]:
        """
        Process a stream of audio chunks for voice activity.
        
        Args:
            audio_stream: Stream of audio chunks
            
        Yields:
            VADResult objects for each chunk
        """
        pass
    
    async def start_processing(self) -> None:
        """Start voice activity processing."""
        self._is_processing = True
        self._speech_segments.clear()
        self._current_segment_start = None
        self._last_speech_time = None
        self.logger.info("Started VAD processing")
    
    async def stop_processing(self) -> None:
        """Stop voice activity processing."""
        self._is_processing = False
        
        # Finalize any ongoing speech segment
        if self._current_segment_start is not None:
            await self._end_speech_segment()
        
        self.logger.info("Stopped VAD processing")
    
    async def process_audio_chunk(self, audio_chunk: bytes) -> VADResult:
        """
        Process a single audio chunk with additional logic.
        
        Args:
            audio_chunk: Raw audio bytes
            
        Returns:
            VADResult with enhanced information
        """
        if not self._is_processing:
            await self.start_processing()
        
        # Get basic VAD result
        result = await self.detect_voice_activity(audio_chunk)
        
        # Add timestamp information
        current_time = time.time()
        result.start_time = current_time
        result.end_time = current_time + (len(audio_chunk) / (self.config.sample_rate * 2))  # Assuming 16-bit audio
        
        # Track speech segments
        await self._update_speech_tracking(result)
        
        return result
    
    async def _update_speech_tracking(self, result: VADResult) -> None:
        """Update speech segment tracking based on VAD result."""
        current_time = time.time()
        
        if result.is_speech:
            self._last_speech_time = current_time
            
            # Start new speech segment if needed
            if self._current_segment_start is None:
                self._current_segment_start = current_time
                await self._start_speech_segment()
        
        else:  # No speech detected
            # Check if we should end current speech segment
            if (self._current_segment_start is not None and 
                self._last_speech_time is not None and
                current_time - self._last_speech_time >= self.config.min_silence_duration):
                
                await self._end_speech_segment()
    
    async def _start_speech_segment(self) -> None:
        """Handle start of speech segment."""
        self.logger.debug("Speech segment started")
        
        if self._on_speech_start:
            try:
                await self._on_speech_start()
            except Exception as e:
                self.logger.error(f"Error in speech start callback: {e}")
    
    async def _end_speech_segment(self) -> None:
        """Handle end of speech segment."""
        if self._current_segment_start is None:
            return
        
        current_time = time.time()
        duration = current_time - self._current_segment_start
        
        # Only consider valid speech segments
        if duration >= self.config.min_speech_duration:
            segment = VADSegment(
                start_time=self._current_segment_start,
                end_time=current_time,
                is_speech=True,
                confidence=1.0,  # Segment-level confidence
            )
            
            self._speech_segments.append(segment)
            self.logger.debug(f"Speech segment ended: {duration:.2f}s")
            
            if self._on_speech_end:
                try:
                    await self._on_speech_end(segment)
                except Exception as e:
                    self.logger.error(f"Error in speech end callback: {e}")
            
            if self._on_speech_segment:
                try:
                    await self._on_speech_segment(segment)
                except Exception as e:
                    self.logger.error(f"Error in speech segment callback: {e}")
        
        self._current_segment_start = None
    
    def set_speech_callbacks(
        self,
        on_speech_start: Optional[Callable] = None,
        on_speech_end: Optional[Callable] = None,
        on_speech_segment: Optional[Callable] = None
    ) -> None:
        """
        Set callbacks for speech events.
        
        Args:
            on_speech_start: Called when speech starts
            on_speech_end: Called when speech ends (receives VADSegment)
            on_speech_segment: Called for each complete speech segment
        """
        self._on_speech_start = on_speech_start
        self._on_speech_end = on_speech_end
        self._on_speech_segment = on_speech_segment
    
    def get_speech_segments(self) -> List[VADSegment]:
        """Get all detected speech segments."""
        return self._speech_segments.copy()
    
    def clear_speech_segments(self) -> None:
        """Clear stored speech segments."""
        self._speech_segments.clear()
    
    def is_currently_speaking(self) -> bool:
        """Check if speech is currently being detected."""
        return self._current_segment_start is not None
    
    def get_silence_duration(self) -> Optional[float]:
        """Get duration of current silence period."""
        if self._last_speech_time is None:
            return None
        
        return time.time() - self._last_speech_time
    
    def adjust_sensitivity(self, sensitivity: VADSensitivity) -> None:
        """Dynamically adjust VAD sensitivity."""
        self.config.sensitivity = sensitivity
        self.logger.info(f"Adjusted VAD sensitivity to {sensitivity.value}")
    
    def validate_config(self) -> bool:
        """
        Validate the current configuration.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ValueError: If configuration is invalid
        """
        if self.config.sample_rate not in self.supported_sample_rates:
            raise ValueError(f"Sample rate {self.config.sample_rate} not supported by {self.provider_name}")
        
        if self.config.chunk_size < self.min_chunk_size:
            raise ValueError(f"Chunk size must be at least {self.min_chunk_size}")
        
        if not 0.0 <= self.config.speech_threshold <= 1.0:
            raise ValueError("Speech threshold must be between 0.0 and 1.0")
        
        if self.config.min_speech_duration <= 0:
            raise ValueError("Minimum speech duration must be positive")
        
        if self.config.min_silence_duration <= 0:
            raise ValueError("Minimum silence duration must be positive")
        
        if not 0 <= self.config.aggressiveness <= 3:
            raise ValueError("Aggressiveness must be between 0 and 3")
        
        return True
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of the VAD provider.
        
        Returns:
            Dictionary with health status information
        """
        try:
            await self.initialize()
            
            # Test with silent audio
            silent_chunk = b'\x00' * self.config.chunk_size
            test_result = await self.detect_voice_activity(silent_chunk)
            
            return {
                "status": "healthy",
                "provider": self.provider_name,
                "config": {
                    "sensitivity": self.config.sensitivity.value,
                    "sample_rate": self.config.sample_rate,
                    "speech_threshold": self.config.speech_threshold
                },
                "test_result": {
                    "is_speech": test_result.is_speech,
                    "confidence": test_result.confidence
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
            "is_processing": self._is_processing,
            "speech_segments_count": len(self._speech_segments),
            "is_currently_speaking": self.is_currently_speaking(),
            "silence_duration": self.get_silence_duration(),
            "config": {
                "sensitivity": self.config.sensitivity.value,
                "sample_rate": self.config.sample_rate,
                "speech_threshold": self.config.speech_threshold,
                "min_speech_duration": self.config.min_speech_duration,
                "min_silence_duration": self.config.min_silence_duration
            }
        }
    
    async def __aenter__(self):
        """Context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.cleanup()


class VADProviderFactory:
    """Factory for creating VAD provider instances."""
    
    _providers: Dict[str, type] = {}
    
    @classmethod
    def register_provider(cls, name: str, provider_class: type):
        """Register a new VAD provider."""
        cls._providers[name] = provider_class
    
    @classmethod
    def create_provider(cls, name: str, config: VADConfig) -> BaseVADProvider:
        """Create an instance of the specified VAD provider."""
        if name not in cls._providers:
            raise ValueError(f"Unknown VAD provider: {name}")
        
        provider_class = cls._providers[name]
        return provider_class(config)
    
    @classmethod
    def list_providers(cls) -> List[str]:
        """List all registered VAD providers."""
        return list(cls._providers.keys())