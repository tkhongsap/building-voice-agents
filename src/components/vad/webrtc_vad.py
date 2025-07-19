"""
WebRTC Voice Activity Detection Provider Implementation

This module implements the WebRTC VAD provider for fast, lightweight
voice activity detection with low latency and resource usage.
"""

import asyncio
import logging
import numpy as np
import time
from typing import Optional, Dict, Any, List, AsyncIterator, Union

from .base_vad import (
    BaseVADProvider, VADResult, VADSegment, VADConfig, VADSensitivity,
    VADProviderFactory
)

logger = logging.getLogger(__name__)


class WebRTCVADConfig(VADConfig):
    """Configuration specific to WebRTC VAD."""
    
    def __init__(
        self,
        aggressiveness: int = 1,  # 0-3, higher = more aggressive filtering
        frame_duration_ms: int = 30,  # Frame duration (10, 20, or 30 ms)
        min_speech_frames: int = 3,  # Minimum consecutive speech frames
        min_silence_frames: int = 10,  # Minimum consecutive silence frames
        speech_padding_frames: int = 2,  # Padding frames around speech
        **kwargs
    ):
        super().__init__(**kwargs)
        self.aggressiveness = aggressiveness
        self.frame_duration_ms = frame_duration_ms
        self.min_speech_frames = min_speech_frames
        self.min_silence_frames = min_silence_frames
        self.speech_padding_frames = speech_padding_frames


class WebRTCVADProvider(BaseVADProvider):
    """
    WebRTC VAD provider implementation.
    
    Provides fast, lightweight voice activity detection using the WebRTC
    VAD algorithm with configurable aggressiveness levels.
    """
    
    def __init__(self, config: WebRTCVADConfig):
        super().__init__(config)
        self.config: WebRTCVADConfig = config
        self.vad = None
        self._vad_initialized = False
        self._processing_count = 0
        self._total_audio_processed_seconds = 0.0
        self._last_processing_time = 0
        
        # Streaming state
        self._frame_buffer = []
        self._speech_frame_count = 0
        self._silence_frame_count = 0
        self._current_speech_state = False
        self._current_segment_start_frame = None
        self._frame_count = 0
    
    @property
    def provider_name(self) -> str:
        """Return the name of this VAD provider."""
        return "webrtc_vad"
    
    @property
    def supported_sample_rates(self) -> List[int]:
        """Return list of supported sample rates."""
        return [8000, 16000, 32000, 48000]  # WebRTC VAD supported rates
    
    @property
    def supports_streaming(self) -> bool:
        """Return whether this provider supports streaming detection."""
        return True
    
    async def initialize(self) -> None:
        """Initialize the WebRTC VAD provider."""
        try:
            # Import webrtcvad (requires pip install webrtcvad)
            try:
                import webrtcvad
                self.webrtcvad = webrtcvad
            except ImportError:
                raise ImportError(
                    "webrtcvad package is required. Install with: pip install webrtcvad"
                )
            
            # Create VAD instance
            self.vad = webrtcvad.Vad(self.config.aggressiveness)
            
            # Validate configuration
            if self.config.sample_rate not in self.supported_sample_rates:
                raise ValueError(
                    f"Sample rate {self.config.sample_rate} not supported. "
                    f"Supported rates: {self.supported_sample_rates}"
                )
            
            if self.config.frame_duration_ms not in [10, 20, 30]:
                raise ValueError(
                    f"Frame duration {self.config.frame_duration_ms}ms not supported. "
                    "Supported durations: 10, 20, 30 ms"
                )
            
            # Set aggressiveness based on sensitivity if not explicitly set
            if hasattr(self.config, 'sensitivity') and self.config.sensitivity:
                self._set_aggressiveness_from_sensitivity()
            
            self._vad_initialized = True
            
            self.logger.info(
                f"WebRTC VAD provider initialized with aggressiveness {self.config.aggressiveness}, "
                f"frame duration {self.config.frame_duration_ms}ms"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to initialize WebRTC VAD: {e}")
            raise
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        self.vad = None
        self._vad_initialized = False
        self._frame_buffer = []
        self._speech_frame_count = 0
        self._silence_frame_count = 0
        self._current_speech_state = False
        
        self.logger.info("WebRTC VAD provider cleanup completed")
    
    async def detect_voice_activity(self, audio_data: Union[bytes, np.ndarray]) -> VADResult:
        """
        Detect voice activity in audio data.
        
        Args:
            audio_data: Audio data as bytes or numpy array
            
        Returns:
            VADResult with voice activity information
        """
        if not self._vad_initialized:
            raise RuntimeError("Provider not initialized")
        
        try:
            start_time = time.time()
            
            # Convert audio data to bytes if needed
            if isinstance(audio_data, np.ndarray):
                audio_bytes = self._numpy_to_bytes(audio_data)
            else:
                audio_bytes = audio_data
            
            # Process audio in frames
            frames = self._split_audio_into_frames(audio_bytes)
            voice_frames = []
            
            for frame in frames:
                is_speech = self.vad.is_speech(frame, self.config.sample_rate)
                voice_frames.append(is_speech)
            
            # Analyze frame results
            segments = self._frames_to_segments(voice_frames)
            
            # Calculate overall voice activity
            total_frames = len(voice_frames)
            speech_frames = sum(voice_frames)
            voice_activity_ratio = speech_frames / total_frames if total_frames > 0 else 0.0
            
            # Determine if voice is detected (require minimum speech frames)
            has_voice = speech_frames >= self.config.min_speech_frames
            
            # Calculate total duration
            total_duration = len(audio_bytes) / (2 * self.config.sample_rate)  # 16-bit audio
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Update metrics
            self._processing_count += 1
            self._total_audio_processed_seconds += total_duration
            self._last_processing_time = end_time
            
            result = VADResult(
                has_voice=has_voice,
                voice_probability=voice_activity_ratio,  # Use ratio as probability estimate
                segments=segments,
                audio_duration=total_duration,
                processing_time=processing_time,
                metadata={
                    "provider": self.provider_name,
                    "aggressiveness": self.config.aggressiveness,
                    "total_frames": total_frames,
                    "speech_frames": speech_frames,
                    "voice_activity_ratio": voice_activity_ratio,
                    "frame_duration_ms": self.config.frame_duration_ms,
                    "sample_rate": self.config.sample_rate
                }
            )
            
            self.logger.debug(
                f"Processed {total_duration:.3f}s audio in {processing_time:.3f}s, "
                f"found {len(segments)} voice segments ({speech_frames}/{total_frames} frames)"
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
        if not self._vad_initialized:
            raise RuntimeError("Provider not initialized")
        
        try:
            async for audio_chunk in audio_stream:
                # Convert audio chunk to bytes
                if isinstance(audio_chunk, np.ndarray):
                    chunk_bytes = self._numpy_to_bytes(audio_chunk)
                else:
                    chunk_bytes = audio_chunk
                
                # Add to frame buffer
                self._frame_buffer.extend(chunk_bytes)
                
                # Process complete frames
                results = []
                while len(self._frame_buffer) >= self._get_frame_size_bytes():
                    frame_bytes = bytes(self._frame_buffer[:self._get_frame_size_bytes()])
                    self._frame_buffer = self._frame_buffer[self._get_frame_size_bytes():]
                    
                    # Process single frame
                    frame_result = await self._process_streaming_frame(frame_bytes)
                    if frame_result:
                        results.append(frame_result)
                
                # Yield results for processed frames
                for result in results:
                    yield result
                    
        except Exception as e:
            self.logger.error(f"Error in streaming VAD: {e}")
            yield VADResult(
                has_voice=False,
                voice_probability=0.0,
                segments=[],
                metadata={"error": str(e), "provider": self.provider_name}
            )
    
    def _set_aggressiveness_from_sensitivity(self) -> None:
        """Set aggressiveness based on configured sensitivity."""
        sensitivity_aggressiveness = {
            VADSensitivity.LOW: 3,        # Most aggressive filtering (less sensitive)
            VADSensitivity.MEDIUM: 2,     # Medium filtering
            VADSensitivity.HIGH: 1,       # Light filtering (more sensitive)
            VADSensitivity.VERY_HIGH: 0   # Minimal filtering (most sensitive)
        }
        
        if hasattr(self.config, 'sensitivity') and self.config.sensitivity:
            self.config.aggressiveness = sensitivity_aggressiveness.get(
                self.config.sensitivity, 
                1
            )
    
    def _numpy_to_bytes(self, audio_array: np.ndarray) -> bytes:
        """Convert numpy array to bytes (16-bit PCM)."""
        # Ensure audio is in the right format
        if audio_array.dtype != np.int16:
            # Assume float32 input, convert to int16
            if audio_array.dtype == np.float32:
                audio_array = (audio_array * 32767).astype(np.int16)
            else:
                audio_array = audio_array.astype(np.int16)
        
        # Ensure mono audio
        if audio_array.ndim > 1:
            audio_array = audio_array.flatten()
        
        return audio_array.tobytes()
    
    def _split_audio_into_frames(self, audio_bytes: bytes) -> List[bytes]:
        """Split audio into frames suitable for WebRTC VAD."""
        frame_size = self._get_frame_size_bytes()
        frames = []
        
        for i in range(0, len(audio_bytes), frame_size):
            frame = audio_bytes[i:i + frame_size]
            
            # Only process complete frames
            if len(frame) == frame_size:
                frames.append(frame)
            else:
                # Pad the last frame if it's incomplete
                padded_frame = frame + b'\x00' * (frame_size - len(frame))
                frames.append(padded_frame)
        
        return frames
    
    def _get_frame_size_bytes(self) -> int:
        """Get frame size in bytes for current configuration."""
        # Frame size = sample_rate * frame_duration_ms / 1000 * 2 (16-bit)
        samples_per_frame = (self.config.sample_rate * self.config.frame_duration_ms) // 1000
        return samples_per_frame * 2  # 2 bytes per sample (16-bit)
    
    def _frames_to_segments(self, voice_frames: List[bool]) -> List[VADSegment]:
        """Convert frame-level voice activity to speech segments."""
        segments = []
        
        # Apply minimum speech/silence frame filtering
        filtered_frames = self._apply_frame_filtering(voice_frames)
        
        # Find speech segments
        in_speech = False
        segment_start = None
        
        frame_duration_s = self.config.frame_duration_ms / 1000.0
        
        for i, is_speech in enumerate(filtered_frames):
            frame_time = i * frame_duration_s
            
            if is_speech and not in_speech:
                # Speech started
                in_speech = True
                segment_start = max(0, frame_time - self.config.speech_padding_frames * frame_duration_s)
            
            elif not is_speech and in_speech:
                # Speech ended
                in_speech = False
                segment_end = frame_time + self.config.speech_padding_frames * frame_duration_s
                
                if segment_start is not None:
                    # Calculate confidence as percentage of speech frames in segment
                    segment_frame_start = max(0, int(segment_start / frame_duration_s))
                    segment_frame_end = min(len(voice_frames), int(segment_end / frame_duration_s))
                    
                    if segment_frame_end > segment_frame_start:
                        speech_in_segment = sum(voice_frames[segment_frame_start:segment_frame_end])
                        total_in_segment = segment_frame_end - segment_frame_start
                        confidence = speech_in_segment / total_in_segment if total_in_segment > 0 else 0.0
                    else:
                        confidence = 0.0
                    
                    segment = VADSegment(
                        start_time=segment_start,
                        end_time=segment_end,
                        confidence=confidence,
                        is_speech=True
                    )
                    segments.append(segment)
                    segment_start = None
        
        # Handle case where audio ends while in speech
        if in_speech and segment_start is not None:
            segment_end = len(filtered_frames) * frame_duration_s
            
            # Calculate confidence for final segment
            segment_frame_start = max(0, int(segment_start / frame_duration_s))
            if len(voice_frames) > segment_frame_start:
                speech_in_segment = sum(voice_frames[segment_frame_start:])
                total_in_segment = len(voice_frames) - segment_frame_start
                confidence = speech_in_segment / total_in_segment if total_in_segment > 0 else 0.0
            else:
                confidence = 0.0
            
            segment = VADSegment(
                start_time=segment_start,
                end_time=segment_end,
                confidence=confidence,
                is_speech=True
            )
            segments.append(segment)
        
        return segments
    
    def _apply_frame_filtering(self, voice_frames: List[bool]) -> List[bool]:
        """Apply minimum speech/silence frame filtering."""
        if not voice_frames:
            return voice_frames
        
        filtered = voice_frames.copy()
        
        # Remove isolated speech frames (require minimum consecutive speech frames)
        for i in range(len(filtered)):
            if filtered[i]:  # If frame is marked as speech
                # Count consecutive speech frames starting from this position
                consecutive_speech = 0
                for j in range(i, min(i + self.config.min_speech_frames, len(filtered))):
                    if voice_frames[j]:
                        consecutive_speech += 1
                    else:
                        break
                
                # If not enough consecutive speech frames, mark as silence
                if consecutive_speech < self.config.min_speech_frames:
                    for j in range(i, i + consecutive_speech):
                        filtered[j] = False
        
        # Remove short silence gaps (require minimum consecutive silence frames)
        for i in range(len(filtered)):
            if not filtered[i]:  # If frame is marked as silence
                # Count consecutive silence frames
                consecutive_silence = 0
                for j in range(i, min(i + self.config.min_silence_frames, len(filtered))):
                    if not voice_frames[j]:
                        consecutive_silence += 1
                    else:
                        break
                
                # Check if this silence gap is between speech segments
                has_speech_before = i > 0 and any(filtered[max(0, i-5):i])
                has_speech_after = (i + consecutive_silence < len(filtered) and 
                                   any(filtered[i + consecutive_silence:min(len(filtered), i + consecutive_silence + 5)]))
                
                # If short silence gap between speech, mark as speech
                if (consecutive_silence < self.config.min_silence_frames and 
                    has_speech_before and has_speech_after):
                    for j in range(i, i + consecutive_silence):
                        filtered[j] = True
        
        return filtered
    
    async def _process_streaming_frame(self, frame_bytes: bytes) -> Optional[VADResult]:
        """Process a single frame for streaming detection."""
        try:
            # Get voice activity for this frame
            is_speech = self.vad.is_speech(frame_bytes, self.config.sample_rate)
            
            # Update state counters
            if is_speech:
                self._speech_frame_count += 1
                self._silence_frame_count = 0
                
                # Start new speech segment if not already in one
                if not self._current_speech_state and self._speech_frame_count >= self.config.min_speech_frames:
                    self._current_speech_state = True
                    self._current_segment_start_frame = self._frame_count - self._speech_frame_count + 1
            else:
                self._silence_frame_count += 1
                
                # End speech segment if we have enough silence
                if (self._current_speech_state and 
                    self._silence_frame_count >= self.config.min_silence_frames):
                    self._current_speech_state = False
                    self._speech_frame_count = 0
            
            self._frame_count += 1
            
            # Create result for this frame
            frame_duration_s = self.config.frame_duration_ms / 1000.0
            
            # Create segment if currently in speech
            segments = []
            if self._current_speech_state and self._current_segment_start_frame is not None:
                start_time = self._current_segment_start_frame * frame_duration_s
                end_time = self._frame_count * frame_duration_s
                
                segment = VADSegment(
                    start_time=start_time,
                    end_time=end_time,
                    confidence=1.0 if is_speech else 0.5,  # Simple confidence estimate
                    is_speech=True
                )
                segments.append(segment)
            
            return VADResult(
                has_voice=self._current_speech_state,
                voice_probability=1.0 if is_speech else 0.0,
                segments=segments,
                audio_duration=frame_duration_s,
                metadata={
                    "provider": self.provider_name,
                    "frame_number": self._frame_count,
                    "speech_frame_count": self._speech_frame_count,
                    "silence_frame_count": self._silence_frame_count,
                    "current_speech_state": self._current_speech_state,
                    "streaming": True
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error processing streaming frame: {e}")
            return None
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of the WebRTC VAD provider.
        
        Returns:
            Dictionary with health status information
        """
        try:
            if not self._vad_initialized:
                return {
                    "status": "unhealthy",
                    "provider": self.provider_name,
                    "error": "VAD not initialized"
                }
            
            # Test with a small audio sample
            start_time = time.time()
            
            # Create test frame (silence)
            frame_size = self._get_frame_size_bytes()
            test_frame = b'\x00' * frame_size
            
            # Test VAD processing
            _ = self.vad.is_speech(test_frame, self.config.sample_rate)
            
            response_time = time.time() - start_time
            
            return {
                "status": "healthy",
                "provider": self.provider_name,
                "vad_initialized": self._vad_initialized,
                "response_time": response_time,
                "processing_count": self._processing_count,
                "total_audio_processed": self._total_audio_processed_seconds,
                "config": {
                    "aggressiveness": self.config.aggressiveness,
                    "sample_rate": self.config.sample_rate,
                    "frame_duration_ms": self.config.frame_duration_ms,
                    "min_speech_frames": self.config.min_speech_frames,
                    "min_silence_frames": self.config.min_silence_frames,
                    "frame_size_bytes": frame_size
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
            "vad_initialized": self._vad_initialized,
            "frame_buffer_size": len(self._frame_buffer),
            "current_speech_state": self._current_speech_state,
            "speech_frame_count": self._speech_frame_count,
            "silence_frame_count": self._silence_frame_count,
            "total_frames_processed": self._frame_count,
            "webrtc_config": {
                "aggressiveness": self.config.aggressiveness,
                "frame_duration_ms": self.config.frame_duration_ms,
                "min_speech_frames": self.config.min_speech_frames,
                "min_silence_frames": self.config.min_silence_frames,
                "speech_padding_frames": self.config.speech_padding_frames,
                "frame_size_bytes": self._get_frame_size_bytes() if self._vad_initialized else 0
            }
        })
        return base_metrics


# Register the provider
VADProviderFactory.register_provider("webrtc", WebRTCVADProvider)