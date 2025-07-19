"""
Silero Voice Activity Detection Provider Implementation

This module implements the Silero VAD provider for high-accuracy
voice activity detection with configurable sensitivity settings.
"""

import asyncio
import logging
import numpy as np
import time
from typing import Optional, Dict, Any, List, AsyncIterator, Union
import torch
import torchaudio
from io import BytesIO

from .base_vad import (
    BaseVADProvider, VADResult, VADSegment, VADConfig, VADSensitivity,
    VADProviderFactory
)

logger = logging.getLogger(__name__)


class SileroVADConfig(VADConfig):
    """Configuration specific to Silero VAD."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        threshold: float = 0.5,  # Voice probability threshold
        min_speech_duration_ms: int = 250,  # Minimum speech duration
        min_silence_duration_ms: int = 100,  # Minimum silence duration
        window_size_samples: int = 512,  # Analysis window size
        speech_pad_ms: int = 30,  # Padding around speech segments
        return_seconds: bool = False,  # Return timestamps in seconds vs samples
        force_reload: bool = False,  # Force model reload
        onnx: bool = False,  # Use ONNX runtime instead of PyTorch
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_path = model_path
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.window_size_samples = window_size_samples
        self.speech_pad_ms = speech_pad_ms
        self.return_seconds = return_seconds
        self.force_reload = force_reload
        self.onnx = onnx


class SileroVADProvider(BaseVADProvider):
    """
    Silero VAD provider implementation.
    
    Provides voice activity detection using Silero's pre-trained models
    with high accuracy and configurable sensitivity settings.
    """
    
    def __init__(self, config: SileroVADConfig):
        super().__init__(config)
        self.config: SileroVADConfig = config
        self.model = None
        self.utils = None
        self._model_loaded = False
        self._processing_count = 0
        self._total_audio_processed_seconds = 0.0
        self._last_processing_time = 0
        
        # Internal state for streaming
        self._audio_buffer = np.array([], dtype=np.float32)
        self._last_speech_timestamp = 0
        self._speech_state = False
        self._current_segment_start = None
    
    @property
    def provider_name(self) -> str:
        """Return the name of this VAD provider."""
        return "silero_vad"
    
    @property
    def supported_sample_rates(self) -> List[int]:
        """Return list of supported sample rates."""
        return [8000, 16000]  # Silero VAD supports 8kHz and 16kHz
    
    @property
    def supports_streaming(self) -> bool:
        """Return whether this provider supports streaming detection."""
        return True
    
    async def initialize(self) -> None:
        """Initialize the Silero VAD provider."""
        try:
            # Load Silero VAD model
            await self._load_model()
            
            # Validate configuration
            if self.config.sample_rate not in self.supported_sample_rates:
                self.logger.warning(
                    f"Sample rate {self.config.sample_rate} not optimal for Silero. "
                    f"Recommended: {self.supported_sample_rates}"
                )
            
            # Set threshold based on sensitivity
            self._set_threshold_from_sensitivity()
            
            self.logger.info("Silero VAD provider initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Silero VAD: {e}")
            raise
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        self.model = None
        self.utils = None
        self._model_loaded = False
        self._audio_buffer = np.array([], dtype=np.float32)
        
        self.logger.info("Silero VAD provider cleanup completed")
    
    async def detect_voice_activity(self, audio_data: Union[bytes, np.ndarray]) -> VADResult:
        """
        Detect voice activity in audio data.
        
        Args:
            audio_data: Audio data as bytes or numpy array
            
        Returns:
            VADResult with voice activity information
        """
        if not self._model_loaded:
            raise RuntimeError("Provider not initialized")
        
        try:
            start_time = time.time()
            
            # Convert audio data to numpy array if needed
            if isinstance(audio_data, bytes):
                audio_array = self._bytes_to_numpy(audio_data)
            else:
                audio_array = audio_data.astype(np.float32)
            
            # Ensure audio is 1-dimensional
            if audio_array.ndim > 1:
                audio_array = audio_array.flatten()
            
            # Resample if necessary
            if self.config.sample_rate != 16000:
                audio_array = self._resample_audio(audio_array, self.config.sample_rate, 16000)
            
            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio_array).unsqueeze(0)
            
            # Get voice probabilities from model
            voice_probs = self.model(audio_tensor, 16000).squeeze()
            
            # Convert probabilities to segments
            segments = self._probabilities_to_segments(
                voice_probs.numpy(), 
                16000,
                self.config.threshold
            )
            
            # Calculate overall voice activity
            total_duration = len(audio_array) / 16000
            voice_duration = sum(seg.end_time - seg.start_time for seg in segments)
            voice_activity_ratio = voice_duration / total_duration if total_duration > 0 else 0.0
            
            # Determine if voice is detected
            has_voice = voice_activity_ratio > (self.config.threshold * 0.1)  # 10% of threshold
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Update metrics
            self._processing_count += 1
            self._total_audio_processed_seconds += total_duration
            self._last_processing_time = end_time
            
            result = VADResult(
                has_voice=has_voice,
                voice_probability=float(voice_probs.max()) if len(voice_probs) > 0 else 0.0,
                segments=segments,
                audio_duration=total_duration,
                processing_time=processing_time,
                metadata={
                    "provider": self.provider_name,
                    "threshold_used": self.config.threshold,
                    "voice_activity_ratio": voice_activity_ratio,
                    "num_segments": len(segments),
                    "sample_rate": 16000,
                    "model_type": "silero_v4" if hasattr(self.model, "version") else "silero"
                }
            )
            
            self.logger.debug(
                f"Processed {total_duration:.3f}s audio in {processing_time:.3f}s, "
                f"found {len(segments)} voice segments"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error detecting voice activity: {e}")
            return VADResult(
                has_voice=False,
                voice_probability=0.0,
                segments=[],
                metadata={"error": str(e), "provider": self.provider_name}
            )
    
    async def detect_voice_activity_streaming(
        self, 
        audio_stream: AsyncIterator[Union[bytes, np.ndarray]]
    ) -> AsyncIterator[VADResult]:
        """
        Detect voice activity in streaming audio.
        
        Args:
            audio_stream: Async iterator of audio chunks
            
        Yields:
            VADResult for each processed chunk
        """
        if not self._model_loaded:
            raise RuntimeError("Provider not initialized")
        
        try:
            chunk_duration = self.config.chunk_duration_ms / 1000.0
            
            async for audio_chunk in audio_stream:
                # Convert audio chunk to numpy array
                if isinstance(audio_chunk, bytes):
                    chunk_array = self._bytes_to_numpy(audio_chunk)
                else:
                    chunk_array = audio_chunk.astype(np.float32)
                
                # Ensure 1D array
                if chunk_array.ndim > 1:
                    chunk_array = chunk_array.flatten()
                
                # Add to buffer
                self._audio_buffer = np.concatenate([self._audio_buffer, chunk_array])
                
                # Process buffer if we have enough audio
                min_samples = int(0.5 * 16000)  # 0.5 seconds minimum
                if len(self._audio_buffer) >= min_samples:
                    # Process current buffer
                    result = await self._process_streaming_buffer()
                    
                    if result:
                        yield result
                    
                    # Keep some audio in buffer for continuity
                    overlap_samples = int(0.1 * 16000)  # 0.1 second overlap
                    if len(self._audio_buffer) > overlap_samples:
                        self._audio_buffer = self._audio_buffer[-overlap_samples:]
                    
        except Exception as e:
            self.logger.error(f"Error in streaming VAD: {e}")
            yield VADResult(
                has_voice=False,
                voice_probability=0.0,
                segments=[],
                metadata={"error": str(e), "provider": self.provider_name}
            )
    
    async def _load_model(self) -> None:
        """Load the Silero VAD model."""
        try:
            # Import Silero VAD utilities
            if self.config.onnx:
                # Use ONNX runtime
                try:
                    import onnxruntime
                    model, utils = torch.hub.load(
                        repo_or_dir='snakers4/silero-vad',
                        model='silero_vad',
                        force_reload=self.config.force_reload,
                        onnx=True
                    )
                except ImportError:
                    self.logger.warning("ONNX runtime not available, falling back to PyTorch")
                    model, utils = torch.hub.load(
                        repo_or_dir='snakers4/silero-vad',
                        model='silero_vad',
                        force_reload=self.config.force_reload,
                        onnx=False
                    )
            else:
                # Use PyTorch
                model, utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=self.config.force_reload,
                    onnx=False
                )
            
            self.model = model
            self.utils = utils
            self._model_loaded = True
            
            # Set model to evaluation mode
            if hasattr(self.model, 'eval'):
                self.model.eval()
            
            self.logger.info("Silero VAD model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load Silero VAD model: {e}")
            raise RuntimeError(f"Could not load Silero VAD model: {e}")
    
    def _set_threshold_from_sensitivity(self) -> None:
        """Set detection threshold based on configured sensitivity."""
        sensitivity_thresholds = {
            VADSensitivity.LOW: 0.7,      # Less sensitive, fewer false positives
            VADSensitivity.MEDIUM: 0.5,   # Balanced
            VADSensitivity.HIGH: 0.3,     # More sensitive, more detections
            VADSensitivity.VERY_HIGH: 0.2 # Very sensitive
        }
        
        if hasattr(self.config, 'sensitivity') and self.config.sensitivity:
            self.config.threshold = sensitivity_thresholds.get(
                self.config.sensitivity, 
                0.5
            )
    
    def _bytes_to_numpy(self, audio_bytes: bytes) -> np.ndarray:
        """Convert audio bytes to numpy array."""
        # Assume 16-bit PCM audio
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        # Normalize to [-1, 1] range
        return audio_array.astype(np.float32) / 32768.0
    
    def _resample_audio(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        if orig_sr == target_sr:
            return audio
        
        # Convert to torch tensor for resampling
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)
        
        # Resample using torchaudio
        resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
        resampled = resampler(audio_tensor).squeeze(0)
        
        return resampled.numpy()
    
    def _probabilities_to_segments(
        self, 
        voice_probs: np.ndarray, 
        sample_rate: int,
        threshold: float
    ) -> List[VADSegment]:
        """Convert voice probabilities to speech segments."""
        segments = []
        
        # Find speech regions above threshold
        speech_mask = voice_probs > threshold
        
        # Find start and end points of speech segments
        speech_changes = np.diff(np.concatenate(([False], speech_mask, [False])).astype(int))
        speech_starts = np.where(speech_changes == 1)[0]
        speech_ends = np.where(speech_changes == -1)[0]
        
        # Convert sample indices to time
        window_size_samples = self.config.window_size_samples
        hop_size = window_size_samples // 2  # Typical hop size
        
        for start_idx, end_idx in zip(speech_starts, speech_ends):
            start_time = (start_idx * hop_size) / sample_rate
            end_time = (end_idx * hop_size) / sample_rate
            duration = end_time - start_time
            
            # Filter by minimum duration
            min_duration = self.config.min_speech_duration_ms / 1000.0
            if duration >= min_duration:
                # Calculate average confidence for this segment
                segment_probs = voice_probs[start_idx:end_idx]
                confidence = float(np.mean(segment_probs)) if len(segment_probs) > 0 else threshold
                
                segment = VADSegment(
                    start_time=start_time,
                    end_time=end_time,
                    confidence=confidence,
                    is_speech=True
                )
                segments.append(segment)
        
        return segments
    
    async def _process_streaming_buffer(self) -> Optional[VADResult]:
        """Process the current audio buffer for streaming detection."""
        if len(self._audio_buffer) == 0:
            return None
        
        try:
            # Resample if necessary
            audio_for_processing = self._audio_buffer
            if self.config.sample_rate != 16000:
                audio_for_processing = self._resample_audio(
                    self._audio_buffer, 
                    self.config.sample_rate, 
                    16000
                )
            
            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio_for_processing).unsqueeze(0)
            
            # Get voice probability
            voice_prob = self.model(audio_tensor, 16000).squeeze()
            max_prob = float(voice_prob.max()) if len(voice_prob) > 0 else 0.0
            
            # Determine if voice is detected
            has_voice = max_prob > self.config.threshold
            
            # Update speech state tracking
            current_time = time.time()
            if has_voice:
                if not self._speech_state:
                    # Speech started
                    self._current_segment_start = current_time
                    self._speech_state = True
                self._last_speech_timestamp = current_time
            else:
                if self._speech_state:
                    # Check if silence duration is long enough to end speech
                    silence_duration = current_time - self._last_speech_timestamp
                    min_silence = self.config.min_silence_duration_ms / 1000.0
                    if silence_duration > min_silence:
                        self._speech_state = False
                        self._current_segment_start = None
            
            # Create segments if we have an active speech segment
            segments = []
            if self._speech_state and self._current_segment_start:
                segment = VADSegment(
                    start_time=0.0,  # Relative to this chunk
                    end_time=len(audio_for_processing) / 16000,
                    confidence=max_prob,
                    is_speech=True
                )
                segments.append(segment)
            
            # Calculate duration
            duration = len(audio_for_processing) / 16000
            
            return VADResult(
                has_voice=has_voice,
                voice_probability=max_prob,
                segments=segments,
                audio_duration=duration,
                metadata={
                    "provider": self.provider_name,
                    "threshold_used": self.config.threshold,
                    "streaming": True,
                    "speech_state": self._speech_state
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error processing streaming buffer: {e}")
            return None
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of the Silero VAD provider.
        
        Returns:
            Dictionary with health status information
        """
        try:
            if not self._model_loaded:
                return {
                    "status": "unhealthy",
                    "provider": self.provider_name,
                    "error": "Model not loaded"
                }
            
            # Test with a small audio sample
            start_time = time.time()
            test_audio = np.random.randn(16000).astype(np.float32)  # 1 second of noise
            test_tensor = torch.from_numpy(test_audio).unsqueeze(0)
            
            # Test model inference
            with torch.no_grad():
                _ = self.model(test_tensor, 16000)
            
            response_time = time.time() - start_time
            
            return {
                "status": "healthy",
                "provider": self.provider_name,
                "model_loaded": self._model_loaded,
                "response_time": response_time,
                "processing_count": self._processing_count,
                "total_audio_processed": self._total_audio_processed_seconds,
                "config": {
                    "threshold": self.config.threshold,
                    "sample_rate": self.config.sample_rate,
                    "min_speech_duration_ms": self.config.min_speech_duration_ms,
                    "onnx_mode": self.config.onnx,
                    "window_size": self.config.window_size_samples
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
            "processing_count": self._processing_count,
            "total_audio_processed_seconds": self._total_audio_processed_seconds,
            "last_processing_time": self._last_processing_time,
            "average_processing_time": (
                self._total_audio_processed_seconds / self._processing_count 
                if self._processing_count > 0 else 0
            ),
            "model_loaded": self._model_loaded,
            "buffer_size": len(self._audio_buffer) if self._audio_buffer is not None else 0,
            "current_speech_state": self._speech_state,
            "silero_config": {
                "threshold": self.config.threshold,
                "min_speech_duration_ms": self.config.min_speech_duration_ms,
                "min_silence_duration_ms": self.config.min_silence_duration_ms,
                "window_size_samples": self.config.window_size_samples,
                "onnx_mode": self.config.onnx,
                "return_seconds": self.config.return_seconds
            }
        })
        return base_metrics


# Register the provider
VADProviderFactory.register_provider("silero", SileroVADProvider)