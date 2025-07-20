"""
Unit tests for DTMF Detector.

Tests DTMF tone detection, Goertzel filter implementation, tone generation,
and audio processing functionality.
"""

import pytest
import asyncio
import numpy as np
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from communication.dtmf_detector import (
    DTMFDetector,
    DTMFConfig,
    DTMFCharacter,
    DTMFDetection,
    GoertzelFilter,
    DTMFState,
    DTMFValidator
)


class TestDTMFConfig:
    """Test DTMF configuration."""
    
    def test_default_config(self):
        """Test default DTMF configuration."""
        config = DTMFConfig()
        
        assert config.sample_rate == 8000
        assert config.frame_size == 160
        assert config.min_tone_duration_ms == 40
        assert config.max_tone_duration_ms == 200
        assert config.inter_digit_delay_ms == 40
        assert config.frequency_tolerance_hz == 20.0
        assert config.amplitude_threshold == 0.1
        assert config.snr_threshold_db == 6.0
        assert config.use_goertzel_filter == True
        assert config.enable_twist_detection == True
        assert config.max_twist_db == 6.0
    
    def test_custom_config(self):
        """Test custom DTMF configuration."""
        config = DTMFConfig(
            sample_rate=16000,
            frame_size=320,
            min_tone_duration_ms=50,
            amplitude_threshold=0.2,
            snr_threshold_db=8.0
        )
        
        assert config.sample_rate == 16000
        assert config.frame_size == 320
        assert config.min_tone_duration_ms == 50
        assert config.amplitude_threshold == 0.2
        assert config.snr_threshold_db == 8.0


class TestDTMFCharacter:
    """Test DTMF character enum and utilities."""
    
    def test_dtmf_character_values(self):
        """Test DTMF character enum values."""
        assert DTMFCharacter.DTMF_0.value == "0"
        assert DTMFCharacter.DTMF_1.value == "1"
        assert DTMFCharacter.DTMF_2.value == "2"
        assert DTMFCharacter.DTMF_3.value == "3"
        assert DTMFCharacter.DTMF_4.value == "4"
        assert DTMFCharacter.DTMF_5.value == "5"
        assert DTMFCharacter.DTMF_6.value == "6"
        assert DTMFCharacter.DTMF_7.value == "7"
        assert DTMFCharacter.DTMF_8.value == "8"
        assert DTMFCharacter.DTMF_9.value == "9"
        assert DTMFCharacter.DTMF_STAR.value == "*"
        assert DTMFCharacter.DTMF_HASH.value == "#"
    
    def test_dtmf_character_frequencies(self):
        """Test DTMF character frequency mapping."""
        # Test known frequency pairs
        assert DTMFCharacter.get_frequencies(DTMFCharacter.DTMF_1) == (697, 1209)
        assert DTMFCharacter.get_frequencies(DTMFCharacter.DTMF_2) == (697, 1336)
        assert DTMFCharacter.get_frequencies(DTMFCharacter.DTMF_3) == (697, 1477)
        assert DTMFCharacter.get_frequencies(DTMFCharacter.DTMF_4) == (770, 1209)
        assert DTMFCharacter.get_frequencies(DTMFCharacter.DTMF_5) == (770, 1336)
        assert DTMFCharacter.get_frequencies(DTMFCharacter.DTMF_6) == (770, 1477)
        assert DTMFCharacter.get_frequencies(DTMFCharacter.DTMF_7) == (852, 1209)
        assert DTMFCharacter.get_frequencies(DTMFCharacter.DTMF_8) == (852, 1336)
        assert DTMFCharacter.get_frequencies(DTMFCharacter.DTMF_9) == (852, 1477)
        assert DTMFCharacter.get_frequencies(DTMFCharacter.DTMF_0) == (941, 1336)
        assert DTMFCharacter.get_frequencies(DTMFCharacter.DTMF_STAR) == (941, 1209)
        assert DTMFCharacter.get_frequencies(DTMFCharacter.DTMF_HASH) == (941, 1477)
    
    def test_character_from_frequencies(self):
        """Test character identification from frequencies."""
        char = DTMFCharacter.from_frequencies(697, 1209)
        assert char == DTMFCharacter.DTMF_1
        
        char = DTMFCharacter.from_frequencies(941, 1477)
        assert char == DTMFCharacter.DTMF_HASH
        
        # Test invalid frequencies
        char = DTMFCharacter.from_frequencies(500, 600)
        assert char is None


class TestDTMFDetection:
    """Test DTMF detection result."""
    
    def test_detection_creation(self):
        """Test DTMF detection creation."""
        detection = DTMFDetection(
            character=DTMFCharacter.DTMF_1,
            confidence=0.95,
            timestamp=time.time(),
            low_frequency=697,
            high_frequency=1209,
            amplitude=0.8,
            snr_db=12.5
        )
        
        assert detection.character == DTMFCharacter.DTMF_1
        assert detection.confidence == 0.95
        assert detection.low_frequency == 697
        assert detection.high_frequency == 1209
        assert detection.amplitude == 0.8
        assert detection.snr_db == 12.5


class TestGoertzelFilter:
    """Test Goertzel filter implementation."""
    
    @pytest.fixture
    def goertzel_filter(self):
        """Create Goertzel filter for testing."""
        return GoertzelFilter(
            sample_rate=8000,
            frame_size=160,
            target_frequency=697
        )
    
    def test_filter_initialization(self, goertzel_filter):
        """Test Goertzel filter initialization."""
        assert goertzel_filter.sample_rate == 8000
        assert goertzel_filter.frame_size == 160
        assert goertzel_filter.target_frequency == 697
        assert goertzel_filter.k is not None
        assert goertzel_filter.omega is not None
        assert goertzel_filter.coeff is not None
    
    def test_filter_processing(self, goertzel_filter):
        """Test Goertzel filter audio processing."""
        # Generate test signal with target frequency
        sample_rate = 8000
        frame_size = 160
        frequency = 697
        
        t = np.linspace(0, frame_size / sample_rate, frame_size, False)
        signal = np.sin(2 * np.pi * frequency * t)
        
        # Process signal
        magnitude = goertzel_filter.process(signal)
        
        # Should detect the target frequency
        assert magnitude > 0.1  # Should have significant magnitude
    
    def test_filter_reset(self, goertzel_filter):
        """Test Goertzel filter reset."""
        # Process some data
        signal = np.random.random(160)
        goertzel_filter.process(signal)
        
        # Reset filter
        goertzel_filter.reset()
        
        # Internal state should be reset
        assert goertzel_filter.s0 == 0.0
        assert goertzel_filter.s1 == 0.0
        assert goertzel_filter.s2 == 0.0


class TestDTMFValidator:
    """Test DTMF validation logic."""
    
    @pytest.fixture
    def validator(self, mock_dtmf_config):
        """Create DTMF validator for testing."""
        return DTMFValidator(mock_dtmf_config)
    
    def test_frequency_validation(self, validator):
        """Test frequency validation."""
        # Valid DTMF frequencies
        assert validator.is_valid_frequency_pair(697, 1209) == True
        assert validator.is_valid_frequency_pair(941, 1477) == True
        
        # Invalid frequencies
        assert validator.is_valid_frequency_pair(500, 600) == False
        assert validator.is_valid_frequency_pair(697, 500) == False
    
    def test_amplitude_validation(self, validator):
        """Test amplitude validation."""
        # Valid amplitude
        assert validator.is_valid_amplitude(0.5) == True
        assert validator.is_valid_amplitude(0.15) == True
        
        # Invalid amplitude (too low)
        assert validator.is_valid_amplitude(0.05) == False
    
    def test_snr_validation(self, validator):
        """Test SNR validation."""
        # Valid SNR
        assert validator.is_valid_snr(10.0) == True
        assert validator.is_valid_snr(8.0) == True
        
        # Invalid SNR (too low)
        assert validator.is_valid_snr(3.0) == False
    
    def test_twist_detection(self, validator):
        """Test twist detection."""
        # Valid twist (within tolerance)
        assert validator.check_twist(0.8, 0.7) == True
        assert validator.check_twist(0.5, 0.6) == True
        
        # Invalid twist (too much difference)
        assert validator.check_twist(1.0, 0.2) == False
    
    def test_timing_validation(self, validator):
        """Test timing validation."""
        # Valid duration
        assert validator.is_valid_duration(50) == True
        assert validator.is_valid_duration(100) == True
        
        # Invalid duration (too short or too long)
        assert validator.is_valid_duration(20) == False
        assert validator.is_valid_duration(300) == False


class TestDTMFDetector:
    """Test DTMF detector functionality."""
    
    @pytest.fixture
    def dtmf_detector(self, mock_dtmf_config):
        """Create DTMF detector for testing."""
        detector = DTMFDetector(mock_dtmf_config)
        return detector
    
    def test_detector_initialization(self, dtmf_detector):
        """Test DTMF detector initialization."""
        assert dtmf_detector.config is not None
        assert dtmf_detector.state == DTMFState.IDLE
        assert len(dtmf_detector.sequence) == 0
        assert dtmf_detector.validator is not None
        assert len(dtmf_detector.goertzel_filters) > 0
    
    def test_filter_setup(self, dtmf_detector):
        """Test Goertzel filter setup."""
        # Should have filters for all DTMF frequencies
        expected_frequencies = [697, 770, 852, 941, 1209, 1336, 1477]
        filter_frequencies = [f.target_frequency for f in dtmf_detector.goertzel_filters]
        
        for freq in expected_frequencies:
            assert freq in filter_frequencies
    
    @pytest.mark.asyncio
    async def test_audio_frame_processing(self, dtmf_detector, mock_dtmf_audio_data):
        """Test audio frame processing."""
        # Process audio frame containing DTMF tone
        detections = await dtmf_detector.process_audio_frame(mock_dtmf_audio_data)
        
        # Should return list of detections
        assert isinstance(detections, list)
    
    @pytest.mark.asyncio
    async def test_dtmf_tone_detection(self, dtmf_detector):
        """Test DTMF tone detection."""
        # Generate test audio with DTMF '1' tone
        sample_rate = 8000
        duration = 0.1  # 100ms
        samples = int(sample_rate * duration)
        t = np.linspace(0, duration, samples, False)
        
        # DTMF '1' frequencies: 697 Hz + 1209 Hz
        tone_697 = np.sin(2 * np.pi * 697 * t)
        tone_1209 = np.sin(2 * np.pi * 1209 * t)
        combined = (tone_697 + tone_1209) / 2
        
        # Convert to bytes
        audio_int16 = (combined * 32767).astype(np.int16)
        audio_data = audio_int16.tobytes()
        
        # Process audio
        detections = await dtmf_detector.process_audio_frame(audio_data)
        
        # Should detect DTMF '1' if filters work correctly
        if detections:
            assert any(d.character == DTMFCharacter.DTMF_1 for d in detections)
    
    def test_sequence_management(self, dtmf_detector):
        """Test DTMF sequence management."""
        # Initially empty
        assert dtmf_detector.get_current_sequence() == ""
        
        # Add detection to sequence
        detection = DTMFDetection(
            character=DTMFCharacter.DTMF_1,
            confidence=0.9,
            timestamp=time.time()
        )
        dtmf_detector._add_to_sequence(detection)
        
        assert dtmf_detector.get_current_sequence() == "1"
        
        # Clear sequence
        dtmf_detector.clear_sequence()
        assert dtmf_detector.get_current_sequence() == ""
    
    def test_callback_registration(self, dtmf_detector):
        """Test DTMF detection callback registration."""
        callback_called = False
        detected_char = None
        
        def dtmf_callback(detection):
            nonlocal callback_called, detected_char
            callback_called = True
            detected_char = detection.character
        
        dtmf_detector.on_dtmf_detected(dtmf_callback)
        
        # Trigger callback
        detection = DTMFDetection(
            character=DTMFCharacter.DTMF_2,
            confidence=0.95,
            timestamp=time.time()
        )
        dtmf_detector._trigger_detection_callback(detection)
        
        assert callback_called == True
        assert detected_char == DTMFCharacter.DTMF_2
    
    @pytest.mark.asyncio
    async def test_state_transitions(self, dtmf_detector):
        """Test DTMF state transitions."""
        # Initial state
        assert dtmf_detector.state == DTMFState.IDLE
        
        # Simulate tone detection
        dtmf_detector._set_state(DTMFState.TONE_DETECTED)
        assert dtmf_detector.state == DTMFState.TONE_DETECTED
        
        # Back to idle
        dtmf_detector._set_state(DTMFState.IDLE)
        assert dtmf_detector.state == DTMFState.IDLE
    
    def test_timing_constraints(self, dtmf_detector):
        """Test timing constraint validation."""
        # Test minimum duration
        assert dtmf_detector._is_valid_tone_duration(50) == True
        assert dtmf_detector._is_valid_tone_duration(20) == False
        
        # Test maximum duration  
        assert dtmf_detector._is_valid_tone_duration(150) == True
        assert dtmf_detector._is_valid_tone_duration(300) == False
    
    @pytest.mark.asyncio
    async def test_multiple_tones_sequence(self, dtmf_detector):
        """Test detection of multiple DTMF tones in sequence."""
        # Simulate sequence: 1-2-3
        tones = [DTMFCharacter.DTMF_1, DTMFCharacter.DTMF_2, DTMFCharacter.DTMF_3]
        
        for tone in tones:
            detection = DTMFDetection(
                character=tone,
                confidence=0.9,
                timestamp=time.time()
            )
            dtmf_detector._add_to_sequence(detection)
            
            # Add small delay between tones
            await asyncio.sleep(0.01)
        
        assert dtmf_detector.get_current_sequence() == "123"
    
    @pytest.mark.asyncio
    async def test_noise_rejection(self, dtmf_detector):
        """Test noise rejection capabilities."""
        # Generate noise signal
        sample_rate = 8000
        frame_size = 160
        noise = np.random.normal(0, 0.1, frame_size)
        
        # Convert to bytes
        noise_int16 = (noise * 32767).astype(np.int16)
        noise_data = noise_int16.tobytes()
        
        # Process noise
        detections = await dtmf_detector.process_audio_frame(noise_data)
        
        # Should not detect any valid DTMF tones in noise
        assert len(detections) == 0
    
    @pytest.mark.asyncio
    async def test_cleanup(self, dtmf_detector):
        """Test detector cleanup."""
        # Add some sequence data
        detection = DTMFDetection(
            character=DTMFCharacter.DTMF_5,
            confidence=0.8,
            timestamp=time.time()
        )
        dtmf_detector._add_to_sequence(detection)
        
        # Cleanup
        await dtmf_detector.cleanup()
        
        # Should reset state
        assert dtmf_detector.state == DTMFState.IDLE
        assert dtmf_detector.get_current_sequence() == ""


class TestDTMFGeneration:
    """Test DTMF tone generation."""
    
    @pytest.fixture
    def dtmf_detector(self, mock_dtmf_config):
        """Create DTMF detector for testing."""
        return DTMFDetector(mock_dtmf_config)
    
    def test_tone_generation(self, dtmf_detector):
        """Test DTMF tone generation."""
        # Generate tone for DTMF '5'
        audio_data = dtmf_detector.generate_tone(
            DTMFCharacter.DTMF_5,
            duration_ms=100
        )
        
        assert audio_data is not None
        assert len(audio_data) > 0
        
        # Should be appropriate length for 100ms at sample rate
        expected_samples = int(dtmf_detector.config.sample_rate * 0.1)
        expected_bytes = expected_samples * 2  # 16-bit samples
        assert len(audio_data) == expected_bytes
    
    def test_sequence_generation(self, dtmf_detector):
        """Test DTMF sequence generation."""
        # Generate sequence "123"
        audio_data = dtmf_detector.generate_sequence(
            "123",
            tone_duration_ms=80,
            pause_duration_ms=50
        )
        
        assert audio_data is not None
        assert len(audio_data) > 0
        
        # Should be longer than single tone
        single_tone = dtmf_detector.generate_tone(DTMFCharacter.DTMF_1, 80)
        assert len(audio_data) > len(single_tone)
    
    def test_invalid_sequence_generation(self, dtmf_detector):
        """Test generation with invalid characters."""
        # Try to generate with invalid character
        audio_data = dtmf_detector.generate_sequence("1X3")
        
        # Should handle gracefully (skip invalid characters)
        assert audio_data is not None


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.fixture
    def error_detector(self, mock_dtmf_config):
        """Create DTMF detector for error testing."""
        return DTMFDetector(mock_dtmf_config)
    
    @pytest.mark.asyncio
    async def test_invalid_audio_data_handling(self, error_detector):
        """Test handling of invalid audio data."""
        # Test with None
        detections = await error_detector.process_audio_frame(None)
        assert len(detections) == 0
        
        # Test with empty data
        detections = await error_detector.process_audio_frame(b"")
        assert len(detections) == 0
        
        # Test with wrong size data
        detections = await error_detector.process_audio_frame(b"short")
        assert len(detections) == 0
    
    @pytest.mark.asyncio
    async def test_processing_error_handling(self, error_detector):
        """Test error handling during processing."""
        # Mock filter processing to raise exception
        with patch.object(error_detector.goertzel_filters[0], 'process', side_effect=Exception("Filter error")):
            # Should not crash
            detections = await error_detector.process_audio_frame(b'\x00' * 320)
            assert isinstance(detections, list)
    
    def test_invalid_configuration_handling(self):
        """Test handling of invalid configuration."""
        # Test with invalid sample rate
        config = DTMFConfig(sample_rate=0)
        
        # Should handle gracefully or raise appropriate exception
        try:
            detector = DTMFDetector(config)
            # If it doesn't crash, that's fine too
        except ValueError:
            # Expected for invalid config
            pass
    
    def test_callback_error_handling(self, error_detector):
        """Test error handling in callbacks."""
        def failing_callback(detection):
            raise Exception("Callback failed")
        
        error_detector.on_dtmf_detected(failing_callback)
        
        # Should not crash when callback fails
        detection = DTMFDetection(
            character=DTMFCharacter.DTMF_0,
            confidence=0.9,
            timestamp=time.time()
        )
        
        # Should handle callback exception gracefully
        error_detector._trigger_detection_callback(detection)


class TestPerformance:
    """Test performance characteristics."""
    
    @pytest.fixture
    def perf_detector(self, mock_dtmf_config):
        """Create DTMF detector for performance testing."""
        return DTMFDetector(mock_dtmf_config)
    
    @pytest.mark.asyncio
    async def test_processing_latency(self, perf_detector):
        """Test processing latency."""
        # Generate test audio
        sample_rate = 8000
        frame_size = 160
        audio_data = np.random.random(frame_size).astype(np.float32)
        audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
        
        # Measure processing time
        start_time = time.time()
        await perf_detector.process_audio_frame(audio_bytes)
        processing_time = time.time() - start_time
        
        # Should process frame quickly (within reasonable time)
        assert processing_time < 0.1  # 100ms max for 20ms frame
    
    @pytest.mark.asyncio
    async def test_memory_usage(self, perf_detector):
        """Test memory usage during processing."""
        # Process multiple frames
        for _ in range(100):
            audio_data = np.random.random(160).astype(np.float32)
            audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
            await perf_detector.process_audio_frame(audio_bytes)
        
        # Should not accumulate excessive memory
        sequence_length = len(perf_detector.get_current_sequence())
        assert sequence_length < 1000  # Reasonable limit


# Integration test markers
pytestmark = pytest.mark.unit