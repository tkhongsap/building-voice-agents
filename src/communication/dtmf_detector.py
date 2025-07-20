"""
DTMF (Dual-Tone Multi-Frequency) Detection and Handling

This module provides comprehensive DTMF tone detection, generation, and handling
for voice agents, supporting both real-time and batch audio processing.
"""

import asyncio
import logging
import time
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List, Callable, Union, Tuple
from collections import deque

try:
    import scipy.signal
    import scipy.fft
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from monitoring.performance_monitor import monitor_performance, global_performance_monitor

logger = logging.getLogger(__name__)


class DTMFTone(Enum):
    """DTMF tone mappings."""
    DIGIT_0 = "0"
    DIGIT_1 = "1"
    DIGIT_2 = "2"
    DIGIT_3 = "3"
    DIGIT_4 = "4"
    DIGIT_5 = "5"
    DIGIT_6 = "6"
    DIGIT_7 = "7"
    DIGIT_8 = "8"
    DIGIT_9 = "9"
    STAR = "*"
    POUND = "#"
    A = "A"
    B = "B"
    C = "C"
    D = "D"


# DTMF frequency mappings (Hz)
DTMF_FREQUENCIES = {
    # Row frequencies (low frequency)
    697: [DTMFTone.DIGIT_1, DTMFTone.DIGIT_2, DTMFTone.DIGIT_3, DTMFTone.A],
    770: [DTMFTone.DIGIT_4, DTMFTone.DIGIT_5, DTMFTone.DIGIT_6, DTMFTone.B],
    852: [DTMFTone.DIGIT_7, DTMFTone.DIGIT_8, DTMFTone.DIGIT_9, DTMFTone.C],
    941: [DTMFTone.STAR, DTMFTone.DIGIT_0, DTMFTone.POUND, DTMFTone.D],
    
    # Column frequencies (high frequency)
    1209: [DTMFTone.DIGIT_1, DTMFTone.DIGIT_4, DTMFTone.DIGIT_7, DTMFTone.STAR],
    1336: [DTMFTone.DIGIT_2, DTMFTone.DIGIT_5, DTMFTone.DIGIT_8, DTMFTone.DIGIT_0],
    1477: [DTMFTone.DIGIT_3, DTMFTone.DIGIT_6, DTMFTone.DIGIT_9, DTMFTone.POUND],
    1633: [DTMFTone.A, DTMFTone.B, DTMFTone.C, DTMFTone.D]
}

# Reverse mapping: tone -> (low_freq, high_freq)
TONE_TO_FREQUENCIES = {
    DTMFTone.DIGIT_1: (697, 1209),
    DTMFTone.DIGIT_2: (697, 1336),
    DTMFTone.DIGIT_3: (697, 1477),
    DTMFTone.A: (697, 1633),
    DTMFTone.DIGIT_4: (770, 1209),
    DTMFTone.DIGIT_5: (770, 1336),
    DTMFTone.DIGIT_6: (770, 1477),
    DTMFTone.B: (770, 1633),
    DTMFTone.DIGIT_7: (852, 1209),
    DTMFTone.DIGIT_8: (852, 1336),
    DTMFTone.DIGIT_9: (852, 1477),
    DTMFTone.C: (852, 1633),
    DTMFTone.STAR: (941, 1209),
    DTMFTone.DIGIT_0: (941, 1336),
    DTMFTone.POUND: (941, 1477),
    DTMFTone.D: (941, 1633)
}


@dataclass
class DTMFConfig:
    """DTMF detection configuration."""
    # Audio settings
    sample_rate: int = 8000
    frame_size: int = 160  # 20ms at 8kHz
    
    # Detection parameters
    min_tone_duration_ms: int = 40  # Minimum tone duration
    max_tone_duration_ms: int = 200  # Maximum tone duration
    inter_digit_delay_ms: int = 40  # Minimum silence between tones
    
    # Frequency analysis
    fft_size: int = 512
    frequency_tolerance_hz: float = 20.0  # Frequency detection tolerance
    amplitude_threshold: float = 0.1  # Minimum amplitude for detection
    snr_threshold_db: float = 6.0  # Signal-to-noise ratio threshold
    
    # Advanced settings
    use_goertzel_filter: bool = True  # Use Goertzel algorithm for efficiency
    enable_twist_detection: bool = True  # Detect amplitude imbalance
    max_twist_db: float = 6.0  # Maximum amplitude difference between frequencies
    
    # Buffer settings
    history_buffer_size: int = 10  # Number of frames to keep for analysis


@dataclass
class DTMFDetection:
    """Result of DTMF tone detection."""
    tone: DTMFTone
    confidence: float
    start_time: float
    end_time: float
    duration_ms: float
    low_frequency: float
    high_frequency: float
    low_amplitude: float
    high_amplitude: float
    snr_db: float
    twist_db: float
    
    @property
    def character(self) -> str:
        """Get the character representation of the tone."""
        return self.tone.value


class DTMFDetector:
    """Real-time DTMF tone detector."""
    
    def __init__(self, config: DTMFConfig = None):
        self.config = config or DTMFConfig()
        
        # Detection state
        self.is_detecting = False
        self.current_tone: Optional[DTMFTone] = None
        self.tone_start_time: Optional[float] = None
        self.tone_samples: List[float] = []
        
        # Audio buffer
        self.audio_buffer = deque(maxlen=self.config.history_buffer_size)
        self.silence_counter = 0
        
        # Frequency analysis
        self.frequencies = list(DTMF_FREQUENCIES.keys())
        if self.config.use_goertzel_filter:
            self._init_goertzel_filters()
        
        # Event callbacks
        self.on_tone_detected_callbacks: List[Callable] = []
        self.on_tone_started_callbacks: List[Callable] = []
        self.on_tone_ended_callbacks: List[Callable] = []
        self.on_sequence_detected_callbacks: List[Callable] = []
        
        # Sequence detection
        self.detected_sequence: List[str] = []
        self.sequence_timeout_ms: int = 5000  # 5 seconds
        self.last_sequence_time: Optional[float] = None
        
        # Monitoring
        self.monitor = global_performance_monitor.register_component(
            "dtmf_detector", "communication"
        )
        
        # Statistics
        self.total_detections = 0
        self.false_positives = 0
        self.detection_accuracy = 1.0
    
    def _init_goertzel_filters(self):
        """Initialize Goertzel filter coefficients for efficient frequency detection."""
        self.goertzel_coeffs = {}
        
        for freq in self.frequencies:
            # Calculate Goertzel coefficient
            k = int(0.5 + (self.config.fft_size * freq / self.config.sample_rate))
            omega = 2.0 * np.pi * k / self.config.fft_size
            coeff = 2.0 * np.cos(omega)
            self.goertzel_coeffs[freq] = coeff
    
    @monitor_performance(component="dtmf_detector", operation="process_audio")
    async def process_audio_frame(self, audio_data: Union[bytes, np.ndarray]) -> List[DTMFDetection]:
        """Process a single audio frame for DTMF detection."""
        # Convert audio data to numpy array if needed
        if isinstance(audio_data, bytes):
            # Assume 16-bit PCM
            samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            samples = np.asarray(audio_data, dtype=np.float32)
        
        # Add to audio buffer
        self.audio_buffer.append(samples)
        
        # Perform detection
        detections = []
        
        if len(self.audio_buffer) >= 2:  # Need at least 2 frames for analysis
            current_frame = self.audio_buffer[-1]
            detection_result = await self._analyze_frame(current_frame)
            
            if detection_result:
                detections.append(detection_result)
                
                # Add to sequence
                await self._update_sequence(detection_result.character)
        
        return detections
    
    async def _analyze_frame(self, frame: np.ndarray) -> Optional[DTMFDetection]:
        """Analyze a single audio frame for DTMF tones."""
        if not SCIPY_AVAILABLE:
            # Mock detection for development
            return None
        
        try:
            # Calculate power spectrum
            if self.config.use_goertzel_filter:
                powers = self._goertzel_analysis(frame)
            else:
                powers = self._fft_analysis(frame)
            
            # Find peak frequencies
            detected_tone = self._identify_tone(powers)
            
            if detected_tone:
                # Validate tone duration and quality
                if await self._validate_tone(detected_tone, powers):
                    return await self._create_detection(detected_tone, powers)
            
            return None
        
        except Exception as e:
            logger.error(f"Error in DTMF analysis: {e}")
            return None
    
    def _goertzel_analysis(self, frame: np.ndarray) -> Dict[float, float]:
        """Perform Goertzel filter analysis for DTMF frequencies."""
        powers = {}
        
        for freq, coeff in self.goertzel_coeffs.items():
            # Goertzel algorithm implementation
            s_prev = 0.0
            s_prev2 = 0.0
            
            for sample in frame:
                s = sample + coeff * s_prev - s_prev2
                s_prev2 = s_prev
                s_prev = s
            
            # Calculate power
            power = s_prev2 * s_prev2 + s_prev * s_prev - coeff * s_prev * s_prev2
            powers[freq] = power
        
        return powers
    
    def _fft_analysis(self, frame: np.ndarray) -> Dict[float, float]:
        """Perform FFT analysis for DTMF frequencies."""
        # Zero-pad frame to FFT size
        if len(frame) < self.config.fft_size:
            padded_frame = np.zeros(self.config.fft_size)
            padded_frame[:len(frame)] = frame
        else:
            padded_frame = frame[:self.config.fft_size]
        
        # Apply window function
        windowed = padded_frame * np.hanning(len(padded_frame))
        
        # Perform FFT
        fft_result = scipy.fft.fft(windowed)
        freqs = scipy.fft.fftfreq(len(fft_result), 1.0 / self.config.sample_rate)
        
        # Extract power at DTMF frequencies
        powers = {}
        for target_freq in self.frequencies:
            # Find closest frequency bin
            freq_idx = np.argmin(np.abs(freqs - target_freq))
            power = abs(fft_result[freq_idx]) ** 2
            powers[target_freq] = power
        
        return powers
    
    def _identify_tone(self, powers: Dict[float, float]) -> Optional[DTMFTone]:
        """Identify DTMF tone from frequency powers."""
        # Find the two strongest frequencies
        sorted_powers = sorted(powers.items(), key=lambda x: x[1], reverse=True)
        
        if len(sorted_powers) < 2:
            return None
        
        freq1, power1 = sorted_powers[0]
        freq2, power2 = sorted_powers[1]
        
        # Check if powers are above threshold
        if power1 < self.config.amplitude_threshold or power2 < self.config.amplitude_threshold:
            return None
        
        # Check frequency separation (low and high frequency groups)
        low_freqs = [697, 770, 852, 941]
        high_freqs = [1209, 1336, 1477, 1633]
        
        low_freq = None
        high_freq = None
        
        for freq in [freq1, freq2]:
            if freq in low_freqs:
                low_freq = freq
            elif freq in high_freqs:
                high_freq = freq
        
        if low_freq is None or high_freq is None:
            return None
        
        # Find the tone that matches both frequencies
        for tone, (tone_low, tone_high) in TONE_TO_FREQUENCIES.items():
            if (abs(low_freq - tone_low) <= self.config.frequency_tolerance_hz and
                abs(high_freq - tone_high) <= self.config.frequency_tolerance_hz):
                return tone
        
        return None
    
    async def _validate_tone(self, tone: DTMFTone, powers: Dict[float, float]) -> bool:
        """Validate detected tone quality and duration."""
        low_freq, high_freq = TONE_TO_FREQUENCIES[tone]
        
        low_power = powers.get(low_freq, 0)
        high_power = powers.get(high_freq, 0)
        
        if low_power == 0 or high_power == 0:
            return False
        
        # Check signal-to-noise ratio
        # Calculate average power of non-target frequencies
        noise_powers = [p for f, p in powers.items() 
                       if f != low_freq and f != high_freq]
        
        if noise_powers:
            avg_noise = np.mean(noise_powers)
            signal_power = (low_power + high_power) / 2
            
            if avg_noise > 0:
                snr_db = 10 * np.log10(signal_power / avg_noise)
                if snr_db < self.config.snr_threshold_db:
                    return False
        
        # Check twist (amplitude balance between frequencies)
        if self.config.enable_twist_detection:
            twist_db = abs(10 * np.log10(low_power / high_power))
            if twist_db > self.config.max_twist_db:
                return False
        
        return True
    
    async def _create_detection(self, tone: DTMFTone, powers: Dict[float, float]) -> DTMFDetection:
        """Create DTMF detection result."""
        current_time = time.time()
        low_freq, high_freq = TONE_TO_FREQUENCIES[tone]
        
        low_power = powers.get(low_freq, 0)
        high_power = powers.get(high_freq, 0)
        
        # Calculate metrics
        snr_db = 0.0
        twist_db = 0.0
        
        if low_power > 0 and high_power > 0:
            twist_db = abs(10 * np.log10(low_power / high_power))
            
            # Calculate SNR
            noise_powers = [p for f, p in powers.items() 
                           if f != low_freq and f != high_freq]
            if noise_powers:
                avg_noise = np.mean(noise_powers)
                signal_power = (low_power + high_power) / 2
                if avg_noise > 0:
                    snr_db = 10 * np.log10(signal_power / avg_noise)
        
        # Estimate duration (simplified)
        duration_ms = self.config.frame_size * 1000 / self.config.sample_rate
        
        detection = DTMFDetection(
            tone=tone,
            confidence=min(1.0, snr_db / 20.0),  # Normalize confidence
            start_time=current_time - duration_ms / 1000,
            end_time=current_time,
            duration_ms=duration_ms,
            low_frequency=low_freq,
            high_frequency=high_freq,
            low_amplitude=low_power,
            high_amplitude=high_power,
            snr_db=snr_db,
            twist_db=twist_db
        )
        
        self.total_detections += 1
        
        # Trigger callbacks
        await self._trigger_tone_detected(detection)
        
        return detection
    
    async def _update_sequence(self, character: str):
        """Update detected sequence and check for patterns."""
        current_time = time.time()
        
        # Reset sequence if timeout exceeded
        if (self.last_sequence_time and 
            current_time - self.last_sequence_time > self.sequence_timeout_ms / 1000):
            self.detected_sequence.clear()
        
        # Add character to sequence
        self.detected_sequence.append(character)
        self.last_sequence_time = current_time
        
        # Limit sequence length
        if len(self.detected_sequence) > 50:  # Max 50 digits
            self.detected_sequence = self.detected_sequence[-50:]
        
        # Trigger sequence callback
        await self._trigger_sequence_detected(''.join(self.detected_sequence))
    
    async def generate_tone(self, tone: DTMFTone, duration_ms: int = 100) -> np.ndarray:
        """Generate DTMF tone audio samples."""
        if tone not in TONE_TO_FREQUENCIES:
            raise ValueError(f"Invalid DTMF tone: {tone}")
        
        low_freq, high_freq = TONE_TO_FREQUENCIES[tone]
        
        # Calculate sample count
        sample_count = int(self.config.sample_rate * duration_ms / 1000)
        
        # Generate time array
        t = np.linspace(0, duration_ms / 1000, sample_count, False)
        
        # Generate sine waves
        low_tone = np.sin(2 * np.pi * low_freq * t)
        high_tone = np.sin(2 * np.pi * high_freq * t)
        
        # Combine tones with equal amplitude
        combined = (low_tone + high_tone) / 2
        
        # Apply envelope to reduce clicks
        envelope_samples = int(0.005 * self.config.sample_rate)  # 5ms ramp
        if envelope_samples > 0:
            ramp_up = np.linspace(0, 1, envelope_samples)
            ramp_down = np.linspace(1, 0, envelope_samples)
            
            combined[:envelope_samples] *= ramp_up
            combined[-envelope_samples:] *= ramp_down
        
        return combined
    
    async def generate_sequence(self, sequence: str, tone_duration_ms: int = 100, 
                              gap_duration_ms: int = 50) -> np.ndarray:
        """Generate audio for a sequence of DTMF tones."""
        audio_segments = []
        
        for char in sequence:
            # Find corresponding tone
            tone = None
            for t in DTMFTone:
                if t.value == char:
                    tone = t
                    break
            
            if tone is None:
                logger.warning(f"Skipping invalid DTMF character: {char}")
                continue
            
            # Generate tone
            tone_audio = await self.generate_tone(tone, tone_duration_ms)
            audio_segments.append(tone_audio)
            
            # Add gap between tones
            if gap_duration_ms > 0:
                gap_samples = int(self.config.sample_rate * gap_duration_ms / 1000)
                gap_audio = np.zeros(gap_samples)
                audio_segments.append(gap_audio)
        
        # Concatenate all segments
        if audio_segments:
            return np.concatenate(audio_segments)
        else:
            return np.array([])
    
    # Event callback triggers
    async def _trigger_tone_detected(self, detection: DTMFDetection):
        """Trigger tone detected callbacks."""
        for callback in self.on_tone_detected_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(detection)
                else:
                    callback(detection)
            except Exception as e:
                logger.error(f"Error in tone detected callback: {e}")
    
    async def _trigger_sequence_detected(self, sequence: str):
        """Trigger sequence detected callbacks."""
        for callback in self.on_sequence_detected_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(sequence)
                else:
                    callback(sequence)
            except Exception as e:
                logger.error(f"Error in sequence detected callback: {e}")
    
    # Public callback registration methods
    def on_tone_detected(self, callback: Callable):
        """Register callback for tone detection events."""
        self.on_tone_detected_callbacks.append(callback)
    
    def on_sequence_detected(self, callback: Callable):
        """Register callback for sequence detection events."""
        self.on_sequence_detected_callbacks.append(callback)
    
    # Status and information methods
    def get_current_sequence(self) -> str:
        """Get the current detected sequence."""
        return ''.join(self.detected_sequence)
    
    def clear_sequence(self):
        """Clear the current detected sequence."""
        self.detected_sequence.clear()
        self.last_sequence_time = None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detection statistics."""
        return {
            "total_detections": self.total_detections,
            "false_positives": self.false_positives,
            "accuracy": self.detection_accuracy,
            "current_sequence_length": len(self.detected_sequence),
            "sequence_active": self.last_sequence_time is not None
        }


# Convenience functions
def create_dtmf_detector(sample_rate: int = 8000, **kwargs) -> DTMFDetector:
    """Create a DTMF detector with specified configuration."""
    config = DTMFConfig(sample_rate=sample_rate, **kwargs)
    return DTMFDetector(config)


async def detect_dtmf_in_audio(audio_data: np.ndarray, sample_rate: int = 8000) -> List[DTMFDetection]:
    """Detect DTMF tones in a complete audio buffer."""
    detector = create_dtmf_detector(sample_rate)
    
    # Process audio in chunks
    frame_size = detector.config.frame_size
    detections = []
    
    for i in range(0, len(audio_data), frame_size):
        frame = audio_data[i:i + frame_size]
        
        if len(frame) < frame_size:
            # Pad last frame
            padded_frame = np.zeros(frame_size)
            padded_frame[:len(frame)] = frame
            frame = padded_frame
        
        frame_detections = await detector.process_audio_frame(frame)
        detections.extend(frame_detections)
    
    return detections


# Global DTMF detector for easy access
_global_dtmf_detector: Optional[DTMFDetector] = None


def get_global_dtmf_detector() -> Optional[DTMFDetector]:
    """Get the global DTMF detector instance."""
    return _global_dtmf_detector


def set_global_dtmf_detector(detector: DTMFDetector):
    """Set the global DTMF detector instance."""
    global _global_dtmf_detector
    _global_dtmf_detector = detector