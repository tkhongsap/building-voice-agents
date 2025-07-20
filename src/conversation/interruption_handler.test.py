"""
Unit Tests for Interruption Handler

Tests the real-time interruption detection and handling system including
agent state management, interruption classification, and recovery mechanisms.
"""

import asyncio
import pytest
import time
import numpy as np
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from typing import List, Dict, Any

from interruption_handler import (
    InterruptionHandler, InterruptionHandlerConfig, InterruptionType,
    InterruptionSeverity, AgentState, InterruptionEvent
)

try:
    from ..components.vad.base_vad import BaseVADProvider, VADResult, VADState
    from ..components.tts.base_tts import BaseTTSProvider
    from .turn_detector import TurnDetector
except ImportError:
    # Fallback for direct imports
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from components.vad.base_vad import BaseVADProvider, VADResult, VADState
    from components.tts.base_tts import BaseTTSProvider
    from turn_detector import TurnDetector


@pytest.fixture
def mock_vad_provider():
    """Create a mock VAD provider for testing."""
    mock = AsyncMock(spec=BaseVADProvider)
    mock.process_audio_chunk = AsyncMock()
    mock.set_speech_callbacks = Mock()
    return mock


@pytest.fixture
def mock_tts_provider():
    """Create a mock TTS provider for testing."""
    mock = AsyncMock(spec=BaseTTSProvider)
    mock.stop_speaking = AsyncMock()
    return mock


@pytest.fixture
def mock_turn_detector():
    """Create a mock turn detector for testing."""
    mock = Mock(spec=TurnDetector)
    mock.set_callbacks = Mock()
    return mock


@pytest.fixture
def basic_config():
    """Create a basic interruption handler configuration."""
    return InterruptionHandlerConfig(
        interruption_confidence_threshold=0.6,
        min_interruption_duration=0.2,
        speaking_sensitivity=0.4,
        listening_sensitivity=0.7,
        enable_detailed_logging=True,
        enable_recovery=True
    )


@pytest.fixture
def interruption_handler(mock_vad_provider, basic_config):
    """Create an interruption handler instance for testing."""
    return InterruptionHandler(
        vad_provider=mock_vad_provider,
        config=basic_config
    )


@pytest.fixture
def full_interruption_handler(mock_vad_provider, mock_tts_provider, mock_turn_detector, basic_config):
    """Create a full interruption handler with all providers."""
    return InterruptionHandler(
        vad_provider=mock_vad_provider,
        tts_provider=mock_tts_provider,
        turn_detector=mock_turn_detector,
        config=basic_config
    )


class TestInterruptionHandlerConfig:
    """Test interruption handler configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = InterruptionHandlerConfig()
        assert config.interruption_confidence_threshold == 0.6
        assert config.speaking_sensitivity == 0.4
        assert config.listening_sensitivity == 0.7
        assert config.enable_recovery is True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = InterruptionHandlerConfig(
            interruption_confidence_threshold=0.8,
            speaking_sensitivity=0.3,
            immediate_response_threshold=0.9
        )
        assert config.interruption_confidence_threshold == 0.8
        assert config.speaking_sensitivity == 0.3
        assert config.immediate_response_threshold == 0.9


class TestInterruptionHandlerInitialization:
    """Test interruption handler initialization."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, interruption_handler):
        """Test basic initialization."""
        await interruption_handler.initialize()
        interruption_handler.vad_provider.set_speech_callbacks.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_full_initialization(self, full_interruption_handler):
        """Test initialization with all providers."""
        await full_interruption_handler.initialize()
        full_interruption_handler.vad_provider.set_speech_callbacks.assert_called_once()
        full_interruption_handler.turn_detector.set_callbacks.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, interruption_handler):
        """Test start and stop monitoring."""
        await interruption_handler.initialize()
        
        await interruption_handler.start_monitoring()
        assert interruption_handler._is_processing is True
        assert interruption_handler.agent_state == AgentState.LISTENING
        
        await interruption_handler.stop_monitoring()
        assert interruption_handler._is_processing is False
    
    def test_initial_state(self, interruption_handler):
        """Test initial handler state."""
        assert interruption_handler.agent_state == AgentState.IDLE
        assert interruption_handler.current_interruption is None
        assert len(interruption_handler.interruption_history) == 0


class TestAgentStateManagement:
    """Test agent state management functionality."""
    
    @pytest.mark.asyncio
    async def test_agent_speaking_notification(self, interruption_handler):
        """Test agent speaking state management."""
        await interruption_handler.initialize()
        
        await interruption_handler.notify_agent_speaking("Hello world", 3.0)
        
        assert interruption_handler.agent_state == AgentState.SPEAKING
        assert interruption_handler._agent_speech_start is not None
        assert interruption_handler._current_sensitivity == interruption_handler.config.speaking_sensitivity
    
    @pytest.mark.asyncio
    async def test_agent_stopped_speaking(self, interruption_handler):
        """Test agent stopped speaking state management."""
        await interruption_handler.initialize()
        
        # First start speaking
        await interruption_handler.notify_agent_speaking("Hello", 2.0)
        
        # Then stop
        await interruption_handler.notify_agent_stopped_speaking()
        
        assert interruption_handler.agent_state == AgentState.LISTENING
        assert interruption_handler._agent_speech_start is None
        assert interruption_handler._current_sensitivity == interruption_handler.config.listening_sensitivity
    
    @pytest.mark.asyncio
    async def test_state_change_callbacks(self, interruption_handler):
        """Test agent state change callbacks."""
        state_changes = []
        
        async def on_state_change(old_state, new_state):
            state_changes.append((old_state, new_state))
        
        interruption_handler.set_callbacks(on_agent_state_change=on_state_change)
        await interruption_handler.initialize()
        
        await interruption_handler._set_agent_state(AgentState.SPEAKING)
        await interruption_handler._set_agent_state(AgentState.INTERRUPTED)
        
        assert len(state_changes) == 2
        assert state_changes[0] == (AgentState.IDLE, AgentState.SPEAKING)
        assert state_changes[1] == (AgentState.SPEAKING, AgentState.INTERRUPTED)


class TestInterruptionDetection:
    """Test interruption detection functionality."""
    
    @pytest.mark.asyncio
    async def test_basic_interruption_detection(self, interruption_handler):
        """Test basic interruption detection during speaking."""
        await interruption_handler.initialize()
        await interruption_handler.start_monitoring()
        await interruption_handler.notify_agent_speaking("Test speech", 2.0)
        
        # Create high-confidence VAD result
        vad_result = VADResult(
            is_speech=True,
            confidence=0.8,
            state=VADState.SPEECH
        )
        interruption_handler.vad_provider.process_audio_chunk.return_value = vad_result
        
        # Create audio chunk with energy
        audio_chunk = np.random.randint(-1000, 1000, 1024, dtype=np.int16).tobytes()
        
        result = await interruption_handler.process_audio_chunk(audio_chunk)
        
        # Should detect interruption
        assert result is not None
        assert result.interruption_type in [
            InterruptionType.IMMEDIATE, InterruptionType.OVERLAP, 
            InterruptionType.BARGE_IN, InterruptionType.GRADUAL
        ]
        assert result.confidence == 0.8
    
    @pytest.mark.asyncio
    async def test_no_interruption_when_listening(self, interruption_handler):
        """Test no interruption detection when agent is listening."""
        await interruption_handler.initialize()
        await interruption_handler.start_monitoring()
        # Agent is in listening state by default
        
        vad_result = VADResult(is_speech=True, confidence=0.8, state=VADState.SPEECH)
        interruption_handler.vad_provider.process_audio_chunk.return_value = vad_result
        
        audio_chunk = np.random.randint(-1000, 1000, 1024, dtype=np.int16).tobytes()
        result = await interruption_handler.process_audio_chunk(audio_chunk)
        
        # Should not detect interruption when not speaking
        assert result is None
    
    @pytest.mark.asyncio
    async def test_low_confidence_filtering(self, interruption_handler):
        """Test filtering of low confidence interruptions."""
        await interruption_handler.initialize()
        await interruption_handler.start_monitoring()
        await interruption_handler.notify_agent_speaking("Test speech", 2.0)
        
        # Low confidence VAD result
        vad_result = VADResult(
            is_speech=True,
            confidence=0.3,  # Below speaking sensitivity threshold
            state=VADState.SPEECH
        )
        interruption_handler.vad_provider.process_audio_chunk.return_value = vad_result
        
        audio_chunk = np.random.randint(-500, 500, 1024, dtype=np.int16).tobytes()
        result = await interruption_handler.process_audio_chunk(audio_chunk)
        
        # Should not detect due to low confidence
        assert result is None
    
    @pytest.mark.asyncio
    async def test_energy_threshold_filtering(self, interruption_handler):
        """Test filtering based on audio energy threshold."""
        await interruption_handler.initialize()
        await interruption_handler.start_monitoring()
        await interruption_handler.notify_agent_speaking("Test speech", 2.0)
        
        vad_result = VADResult(is_speech=True, confidence=0.8, state=VADState.SPEECH)
        interruption_handler.vad_provider.process_audio_chunk.return_value = vad_result
        
        # Very low energy audio (silence)
        audio_chunk = np.zeros(1024, dtype=np.int16).tobytes()
        result = await interruption_handler.process_audio_chunk(audio_chunk)
        
        # Should not detect due to low energy
        assert result is None


class TestInterruptionClassification:
    """Test interruption type and severity classification."""
    
    def test_interruption_type_classification(self, interruption_handler):
        """Test interruption type classification logic."""
        # High confidence and energy = immediate
        vad_result = VADResult(is_speech=True, confidence=0.95, state=VADState.SPEECH)
        result_type = interruption_handler._classify_interruption_type(vad_result, 0.8)
        assert result_type == InterruptionType.IMMEDIATE
        
        # High energy with moderate confidence = barge-in
        vad_result = VADResult(is_speech=True, confidence=0.7, state=VADState.SPEECH)
        result_type = interruption_handler._classify_interruption_type(vad_result, 0.7)
        assert result_type == InterruptionType.BARGE_IN
    
    def test_interruption_severity_classification(self, interruption_handler):
        """Test interruption severity classification."""
        # High confidence and energy = critical
        vad_result = VADResult(is_speech=True, confidence=0.9, state=VADState.SPEECH)
        severity = interruption_handler._calculate_interruption_severity(vad_result, 0.9)
        assert severity == InterruptionSeverity.CRITICAL
        
        # Medium confidence and energy = medium
        vad_result = VADResult(is_speech=True, confidence=0.6, state=VADState.SPEECH)
        severity = interruption_handler._calculate_interruption_severity(vad_result, 0.6)
        assert severity == InterruptionSeverity.MEDIUM
        
        # Low confidence and energy = low
        vad_result = VADResult(is_speech=True, confidence=0.4, state=VADState.SPEECH)
        severity = interruption_handler._calculate_interruption_severity(vad_result, 0.4)
        assert severity == InterruptionSeverity.LOW
    
    @pytest.mark.asyncio
    async def test_gradual_interruption_detection(self, interruption_handler):
        """Test gradual interruption pattern detection."""
        await interruption_handler.initialize()
        
        # Build up confidence pattern
        confidences = [0.3, 0.5, 0.7]
        for conf in confidences:
            interruption_handler._vad_confidence_buffer.append(conf)
        
        vad_result = VADResult(is_speech=True, confidence=0.7, state=VADState.SPEECH)
        result_type = interruption_handler._classify_interruption_type(vad_result, 0.5)
        
        assert result_type == InterruptionType.GRADUAL


class TestInterruptionHandling:
    """Test interruption handling and response mechanisms."""
    
    @pytest.mark.asyncio
    async def test_critical_interruption_handling(self, full_interruption_handler):
        """Test critical interruption handling."""
        await full_interruption_handler.initialize()
        await full_interruption_handler.start_monitoring()
        await full_interruption_handler.notify_agent_speaking("Test speech", 2.0)
        
        # Create critical interruption
        interruption = InterruptionEvent(
            timestamp=time.time(),
            interruption_type=InterruptionType.IMMEDIATE,
            severity=InterruptionSeverity.CRITICAL,
            confidence=0.95,
            vad_confidence=0.95,
            agent_state_at_interruption=AgentState.SPEAKING
        )
        
        await full_interruption_handler._handle_critical_interruption(interruption)
        
        # Should stop TTS and change state
        full_interruption_handler.tts_provider.stop_speaking.assert_called_once()
        assert full_interruption_handler.agent_state == AgentState.INTERRUPTED
    
    @pytest.mark.asyncio
    async def test_medium_interruption_handling(self, interruption_handler):
        """Test medium interruption handling with delay."""
        await interruption_handler.initialize()
        await interruption_handler.start_monitoring()
        
        interruption = InterruptionEvent(
            timestamp=time.time(),
            interruption_type=InterruptionType.OVERLAP,
            severity=InterruptionSeverity.MEDIUM,
            confidence=0.7,
            vad_confidence=0.7
        )
        
        # Mock _is_interruption_still_valid to return True
        interruption_handler._is_interruption_still_valid = Mock(return_value=True)
        
        await interruption_handler._handle_medium_interruption(interruption)
        
        # Should eventually change to interrupted state
        assert interruption_handler.agent_state == AgentState.INTERRUPTED
    
    @pytest.mark.asyncio
    async def test_low_interruption_monitoring(self, interruption_handler):
        """Test low severity interruption monitoring."""
        await interruption_handler.initialize()
        
        interruption = InterruptionEvent(
            timestamp=time.time(),
            interruption_type=InterruptionType.OVERLAP,
            severity=InterruptionSeverity.LOW,
            confidence=0.5,
            vad_confidence=0.5
        )
        
        await interruption_handler._handle_low_interruption(interruption)
        
        # Should add to candidates
        assert interruption in interruption_handler._interruption_candidates
    
    @pytest.mark.asyncio
    async def test_interruption_confirmation(self, interruption_handler):
        """Test interruption confirmation process."""
        callback_called = False
        received_interruption = None
        
        async def on_interruption_confirmed(interruption):
            nonlocal callback_called, received_interruption
            callback_called = True
            received_interruption = interruption
        
        interruption_handler.set_callbacks(on_interruption_confirmed=on_interruption_confirmed)
        await interruption_handler.initialize()
        
        interruption = InterruptionEvent(
            timestamp=time.time(),
            interruption_type=InterruptionType.IMMEDIATE,
            severity=InterruptionSeverity.HIGH,
            confidence=0.8,
            vad_confidence=0.8
        )
        
        await interruption_handler._confirm_interruption(interruption)
        
        assert callback_called is True
        assert received_interruption == interruption
        assert interruption.is_processed is True
        assert len(interruption_handler.interruption_history) == 1


class TestFalsePositiveFiltering:
    """Test false positive filtering mechanisms."""
    
    @pytest.mark.asyncio
    async def test_duration_based_filtering(self, interruption_handler):
        """Test filtering based on interruption duration."""
        interruption = InterruptionEvent(
            timestamp=time.time(),
            interruption_type=InterruptionType.OVERLAP,
            severity=InterruptionSeverity.LOW,
            confidence=0.6,
            vad_confidence=0.6,
            duration=0.05  # Very short duration
        )
        
        is_false_positive = await interruption_handler._is_false_positive(interruption)
        assert is_false_positive is True
    
    @pytest.mark.asyncio
    async def test_energy_consistency_filtering(self, interruption_handler):
        """Test filtering based on energy consistency."""
        await interruption_handler.initialize()
        
        # Add consistent low energy values
        for _ in range(5):
            interruption_handler._audio_energy_buffer.append(0.1)
        
        interruption = InterruptionEvent(
            timestamp=time.time(),
            interruption_type=InterruptionType.OVERLAP,
            severity=InterruptionSeverity.LOW,
            confidence=0.6,
            vad_confidence=0.6
        )
        
        is_false_positive = await interruption_handler._is_false_positive(interruption)
        assert is_false_positive is True
    
    @pytest.mark.asyncio
    async def test_echo_cancellation_filtering(self, interruption_handler):
        """Test echo cancellation filtering."""
        await interruption_handler.initialize()
        await interruption_handler.notify_agent_speaking("Test", 2.0)
        
        # Setup similar background confidence
        for _ in range(3):
            interruption_handler._vad_confidence_buffer.append(0.5)
        
        interruption = InterruptionEvent(
            timestamp=time.time(),
            interruption_type=InterruptionType.OVERLAP,
            severity=InterruptionSeverity.LOW,
            confidence=0.52,  # Similar to background
            vad_confidence=0.52
        )
        
        is_false_positive = await interruption_handler._is_false_positive(interruption)
        assert is_false_positive is True


class TestRecoveryMechanism:
    """Test interruption recovery mechanisms."""
    
    @pytest.mark.asyncio
    async def test_recovery_process(self, interruption_handler):
        """Test complete recovery process."""
        callback_called = False
        
        async def on_recovery_complete():
            nonlocal callback_called
            callback_called = True
        
        interruption_handler.set_callbacks(on_recovery_complete=on_recovery_complete)
        await interruption_handler.initialize()
        
        await interruption_handler._start_recovery()
        
        assert callback_called is True
        assert interruption_handler.agent_state == AgentState.LISTENING
    
    @pytest.mark.asyncio
    async def test_recovery_state_transitions(self, interruption_handler):
        """Test state transitions during recovery."""
        await interruption_handler.initialize()
        
        # Start from interrupted state
        await interruption_handler._set_agent_state(AgentState.INTERRUPTED)
        
        # Start recovery
        recovery_task = asyncio.create_task(interruption_handler._start_recovery())
        
        # Briefly check recovering state
        await asyncio.sleep(0.1)
        assert interruption_handler.agent_state == AgentState.RECOVERING
        
        # Wait for completion
        await recovery_task
        assert interruption_handler.agent_state == AgentState.LISTENING


class TestAdaptiveParameters:
    """Test adaptive parameter adjustment."""
    
    @pytest.mark.asyncio
    async def test_sensitivity_adaptation(self, interruption_handler):
        """Test adaptive sensitivity adjustment."""
        await interruption_handler.initialize()
        
        # Add high-confidence interruptions to history
        for i in range(3):
            interruption = InterruptionEvent(
                timestamp=time.time() + i,
                interruption_type=InterruptionType.IMMEDIATE,
                severity=InterruptionSeverity.HIGH,
                confidence=0.9,
                vad_confidence=0.9
            )
            interruption_handler.interruption_history.append(interruption)
        
        original_sensitivity = interruption_handler._current_sensitivity
        await interruption_handler._update_adaptive_parameters(interruption_handler.interruption_history[0])
        
        # Sensitivity should be lowered for high-confidence users
        assert interruption_handler._current_sensitivity < original_sensitivity
    
    @pytest.mark.asyncio
    async def test_sensitivity_bounds(self, interruption_handler):
        """Test sensitivity stays within bounds."""
        await interruption_handler.initialize()
        
        # Set extreme value
        interruption_handler._current_sensitivity = 0.05
        
        interruption = InterruptionEvent(
            timestamp=time.time(),
            interruption_type=InterruptionType.OVERLAP,
            severity=InterruptionSeverity.LOW,
            confidence=0.3,
            vad_confidence=0.3
        )
        
        await interruption_handler._update_adaptive_parameters(interruption)
        
        # Should be clamped to minimum
        assert interruption_handler._current_sensitivity >= 0.3
    
    @pytest.mark.asyncio
    async def test_manual_sensitivity_adjustment(self, interruption_handler):
        """Test manual sensitivity adjustment."""
        await interruption_handler.adjust_sensitivity(0.8)
        assert interruption_handler._current_sensitivity == 0.8
        
        # Test bounds
        await interruption_handler.adjust_sensitivity(1.5)  # Above max
        assert interruption_handler._current_sensitivity == 1.0
        
        await interruption_handler.adjust_sensitivity(-0.1)  # Below min
        assert interruption_handler._current_sensitivity == 0.1


class TestAudioAnalysis:
    """Test audio analysis functionality."""
    
    def test_audio_energy_calculation(self, interruption_handler):
        """Test audio energy calculation."""
        # Silent audio
        silent_audio = np.zeros(1024, dtype=np.int16).tobytes()
        energy = interruption_handler._calculate_audio_energy(silent_audio)
        assert energy == 0.0
        
        # Loud audio
        loud_audio = np.full(1024, 16000, dtype=np.int16).tobytes()
        energy = interruption_handler._calculate_audio_energy(loud_audio)
        assert energy > 0.4
        
        # Random audio
        random_audio = np.random.randint(-1000, 1000, 1024, dtype=np.int16).tobytes()
        energy = interruption_handler._calculate_audio_energy(random_audio)
        assert 0.0 <= energy <= 1.0
    
    def test_interruption_validation(self, interruption_handler):
        """Test interruption validation logic."""
        # Create valid VAD result
        interruption_handler._last_vad_result = VADResult(
            is_speech=True,
            confidence=0.8,
            state=VADState.SPEECH
        )
        
        interruption = InterruptionEvent(
            timestamp=time.time(),
            interruption_type=InterruptionType.OVERLAP,
            severity=InterruptionSeverity.MEDIUM,
            confidence=0.8,
            vad_confidence=0.8
        )
        
        is_valid = interruption_handler._is_interruption_still_valid(interruption)
        assert is_valid is True
        
        # Test with no speech
        interruption_handler._last_vad_result.is_speech = False
        is_valid = interruption_handler._is_interruption_still_valid(interruption)
        assert is_valid is False


class TestMetricsAndMonitoring:
    """Test metrics collection and monitoring."""
    
    def test_metrics_collection(self, interruption_handler):
        """Test metrics collection."""
        # Add test interruptions
        interruptions = [
            InterruptionEvent(
                timestamp=time.time(),
                interruption_type=InterruptionType.IMMEDIATE,
                severity=InterruptionSeverity.HIGH,
                confidence=0.8,
                vad_confidence=0.8,
                response_latency=0.2
            ),
            InterruptionEvent(
                timestamp=time.time(),
                interruption_type=InterruptionType.GRADUAL,
                severity=InterruptionSeverity.MEDIUM,
                confidence=0.6,
                vad_confidence=0.6,
                response_latency=0.3
            )
        ]
        
        interruption_handler.interruption_history.extend(interruptions)
        
        metrics = interruption_handler.get_metrics()
        
        assert metrics['total_interruptions'] == 2
        assert metrics['average_confidence'] == 0.7
        assert metrics['average_response_latency'] == 0.25
        assert 'severity_distribution' in metrics
        assert 'agent_state' in metrics
    
    def test_state_reporting(self, interruption_handler):
        """Test current state reporting."""
        assert interruption_handler.get_current_state() == AgentState.IDLE
        assert interruption_handler.get_current_interruption() is None
        assert len(interruption_handler.get_interruption_history()) == 0
    
    @pytest.mark.asyncio
    async def test_monitoring_loop(self, interruption_handler):
        """Test monitoring loop functionality."""
        await interruption_handler.initialize()
        await interruption_handler.start_monitoring()
        
        # Let monitoring run briefly
        await asyncio.sleep(0.2)
        
        # Should be running without errors
        assert interruption_handler._is_processing is True
        
        await interruption_handler.stop_monitoring()


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_processing_without_initialization(self, interruption_handler):
        """Test processing without proper initialization."""
        # Should handle gracefully
        result = await interruption_handler.process_audio_chunk(b'audio_data')
        assert result is None
    
    @pytest.mark.asyncio
    async def test_callback_error_handling(self, interruption_handler):
        """Test error handling in callbacks."""
        def failing_callback(*args):
            raise Exception("Callback error")
        
        interruption_handler.set_callbacks(on_interruption_detected=failing_callback)
        await interruption_handler.initialize()
        await interruption_handler.start_monitoring()
        await interruption_handler.notify_agent_speaking("Test", 2.0)
        
        # Create interruption
        vad_result = VADResult(is_speech=True, confidence=0.8, state=VADState.SPEECH)
        interruption_handler.vad_provider.process_audio_chunk.return_value = vad_result
        
        audio_chunk = np.random.randint(-1000, 1000, 1024, dtype=np.int16).tobytes()
        
        # Should not raise exception despite callback failure
        result = await interruption_handler.process_audio_chunk(audio_chunk)
        # Test passes if no exception is raised
    
    @pytest.mark.asyncio
    async def test_invalid_audio_data(self, interruption_handler):
        """Test handling of invalid audio data."""
        await interruption_handler.initialize()
        
        # Invalid audio data should not crash
        energy = interruption_handler._calculate_audio_energy(b'invalid')
        assert energy == 0.0
    
    @pytest.mark.asyncio
    async def test_force_interruption_stop(self, interruption_handler):
        """Test forced interruption stop."""
        await interruption_handler.initialize()
        
        # Create active interruption
        interruption = InterruptionEvent(
            timestamp=time.time(),
            interruption_type=InterruptionType.IMMEDIATE,
            severity=InterruptionSeverity.HIGH,
            confidence=0.8,
            vad_confidence=0.8
        )
        interruption_handler.current_interruption = interruption
        
        await interruption_handler.force_interruption_stop()
        
        assert interruption_handler.current_interruption is None
        assert interruption_handler.agent_state == AgentState.LISTENING


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_complete_interruption_flow(self, full_interruption_handler):
        """Test complete interruption flow from detection to recovery."""
        await full_interruption_handler.initialize()
        await full_interruption_handler.start_monitoring()
        
        # Agent starts speaking
        await full_interruption_handler.notify_agent_speaking("Hello, how are you doing today?", 3.0)
        
        # User interrupts
        vad_result = VADResult(is_speech=True, confidence=0.9, state=VADState.SPEECH)
        full_interruption_handler.vad_provider.process_audio_chunk.return_value = vad_result
        
        audio_chunk = np.random.randint(-2000, 2000, 1024, dtype=np.int16).tobytes()
        result = await full_interruption_handler.process_audio_chunk(audio_chunk)
        
        # Should detect and handle interruption
        assert result is not None
        assert result.severity in [InterruptionSeverity.HIGH, InterruptionSeverity.CRITICAL]
        assert full_interruption_handler.agent_state == AgentState.INTERRUPTED
        
        # TTS should be stopped
        full_interruption_handler.tts_provider.stop_speaking.assert_called()
    
    @pytest.mark.asyncio
    async def test_multiple_interruption_handling(self, interruption_handler):
        """Test handling multiple rapid interruptions."""
        await interruption_handler.initialize()
        await interruption_handler.start_monitoring()
        await interruption_handler.notify_agent_speaking("Test speech", 5.0)
        
        # Create multiple interruption events
        vad_result = VADResult(is_speech=True, confidence=0.8, state=VADState.SPEECH)
        interruption_handler.vad_provider.process_audio_chunk.return_value = vad_result
        
        audio_chunk = np.random.randint(-1000, 1000, 1024, dtype=np.int16).tobytes()
        
        results = []
        for _ in range(3):
            result = await interruption_handler.process_audio_chunk(audio_chunk)
            if result:
                results.append(result)
            await asyncio.sleep(0.1)
        
        # Should handle multiple interruptions
        assert len(results) >= 1  # At least one should be detected
    
    @pytest.mark.asyncio
    async def test_interruption_escalation(self, interruption_handler):
        """Test interruption escalation from low to high severity."""
        await interruption_handler.initialize()
        await interruption_handler.start_monitoring()
        
        # Start with low severity
        low_interruption = InterruptionEvent(
            timestamp=time.time(),
            interruption_type=InterruptionType.OVERLAP,
            severity=InterruptionSeverity.LOW,
            confidence=0.5,
            vad_confidence=0.5
        )
        
        # Mock escalation conditions
        interruption_handler._last_vad_result = VADResult(
            is_speech=True,
            confidence=0.8,  # Higher than original
            state=VADState.SPEECH
        )
        
        # Monitor escalation
        await interruption_handler._monitor_interruption_escalation(low_interruption)
        
        # Should escalate to medium severity
        assert low_interruption.severity == InterruptionSeverity.MEDIUM


if __name__ == "__main__":
    pytest.main([__file__, "-v"])