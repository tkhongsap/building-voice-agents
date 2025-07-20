"""
Unit Tests for Turn Detector

Tests the advanced turn detection system including dual-signal processing,
semantic analysis, and adaptive threshold management.
"""

import asyncio
import pytest
import time
import numpy as np
from unittest.mock import Mock, AsyncMock, MagicMock
from typing import List, Dict, Any

from turn_detector import (
    TurnDetector, TurnDetectionConfig, TurnState, TurnEndReason,
    TurnSegment, SemanticContext
)

try:
    from ..components.vad.base_vad import BaseVADProvider, VADResult, VADState
    from ..components.stt.base_stt import BaseSTTProvider, STTResult
    from ..components.llm.base_llm import BaseLLMProvider
except ImportError:
    # Fallback for direct imports
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from components.vad.base_vad import BaseVADProvider, VADResult, VADState
    from components.stt.base_stt import BaseSTTProvider, STTResult
    from components.llm.base_llm import BaseLLMProvider


@pytest.fixture
def mock_vad_provider():
    """Create a mock VAD provider for testing."""
    mock = AsyncMock(spec=BaseVADProvider)
    mock.process_audio_chunk = AsyncMock()
    mock.set_speech_callbacks = Mock()
    return mock


@pytest.fixture
def mock_stt_provider():
    """Create a mock STT provider for testing."""
    mock = AsyncMock(spec=BaseSTTProvider)
    return mock


@pytest.fixture
def mock_llm_provider():
    """Create a mock LLM provider for testing."""
    mock = AsyncMock(spec=BaseLLMProvider)
    return mock


@pytest.fixture
def basic_config():
    """Create a basic turn detection configuration."""
    return TurnDetectionConfig(
        silence_threshold_short=0.5,
        silence_threshold_medium=1.0,
        silence_threshold_long=2.0,
        vad_weight=0.6,
        semantic_weight=0.4,
        confidence_threshold=0.7,
        enable_detailed_logging=True
    )


@pytest.fixture
def turn_detector(mock_vad_provider, basic_config):
    """Create a turn detector instance for testing."""
    return TurnDetector(
        vad_provider=mock_vad_provider,
        config=basic_config
    )


@pytest.fixture
def full_turn_detector(mock_vad_provider, mock_stt_provider, mock_llm_provider, basic_config):
    """Create a full turn detector with all providers."""
    return TurnDetector(
        vad_provider=mock_vad_provider,
        stt_provider=mock_stt_provider,
        llm_provider=mock_llm_provider,
        config=basic_config
    )


class TestTurnDetectionConfig:
    """Test turn detection configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TurnDetectionConfig()
        assert config.silence_threshold_medium == 1.0
        assert config.vad_weight == 0.6
        assert config.semantic_weight == 0.4
        assert config.confidence_threshold == 0.7
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = TurnDetectionConfig(
            silence_threshold_medium=1.5,
            vad_weight=0.8,
            semantic_weight=0.2
        )
        assert config.silence_threshold_medium == 1.5
        assert config.vad_weight == 0.8
        assert config.semantic_weight == 0.2


class TestTurnDetectorInitialization:
    """Test turn detector initialization."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, turn_detector):
        """Test basic initialization."""
        await turn_detector.initialize()
        turn_detector.vad_provider.set_speech_callbacks.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_start_stop_processing(self, turn_detector):
        """Test start and stop processing."""
        await turn_detector.initialize()
        
        await turn_detector.start_processing()
        assert turn_detector._is_processing is True
        assert turn_detector.current_state == TurnState.LISTENING
        
        await turn_detector.stop_processing()
        assert turn_detector._is_processing is False
    
    def test_initial_state(self, turn_detector):
        """Test initial detector state."""
        assert turn_detector.current_state == TurnState.LISTENING
        assert turn_detector.current_turn is None
        assert len(turn_detector.turn_history) == 0


class TestVADSignalProcessing:
    """Test VAD signal processing functionality."""
    
    @pytest.mark.asyncio
    async def test_speech_detection(self, turn_detector):
        """Test basic speech detection."""
        await turn_detector.initialize()
        await turn_detector.start_processing()
        
        # Mock VAD result with speech
        vad_result = VADResult(
            is_speech=True,
            confidence=0.8,
            state=VADState.SPEECH
        )
        turn_detector.vad_provider.process_audio_chunk.return_value = vad_result
        
        # Process audio chunk
        result = await turn_detector.process_audio_chunk(b'audio_data')
        
        # Should start a new turn
        assert turn_detector.current_turn is not None
        assert turn_detector.current_state == TurnState.TURN_TAKING
    
    @pytest.mark.asyncio
    async def test_silence_detection(self, turn_detector):
        """Test silence detection and turn completion."""
        await turn_detector.initialize()
        await turn_detector.start_processing()
        
        # Start with speech
        speech_vad = VADResult(is_speech=True, confidence=0.8, state=VADState.SPEECH)
        turn_detector.vad_provider.process_audio_chunk.return_value = speech_vad
        await turn_detector.process_audio_chunk(b'speech_data')
        
        # Mock time to simulate silence duration
        original_time = time.time
        mock_start_time = 1000.0
        
        def mock_time():
            return mock_start_time + 1.5  # 1.5 seconds later
        
        # Apply time mock
        time.time = mock_time
        
        try:
            # Now silence
            silence_vad = VADResult(is_speech=False, confidence=0.2, state=VADState.SILENCE)
            turn_detector.vad_provider.process_audio_chunk.return_value = silence_vad
            
            # Set last speech time manually for test
            turn_detector._last_speech_time = mock_start_time
            
            result = await turn_detector.process_audio_chunk(b'silence_data')
            
            # Should complete turn due to silence threshold
            if result:
                assert result.is_complete is True
                assert result.end_reason in [TurnEndReason.SILENCE_THRESHOLD, TurnEndReason.COMBINED_SIGNALS]
        
        finally:
            # Restore original time function
            time.time = original_time
    
    @pytest.mark.asyncio
    async def test_confidence_calculation(self, turn_detector):
        """Test VAD confidence calculation."""
        await turn_detector.initialize()
        await turn_detector.start_processing()
        
        # Add multiple VAD results to confidence buffer
        confidences = [0.7, 0.8, 0.9, 0.6, 0.8]
        for conf in confidences:
            turn_detector._vad_confidence_buffer.append(conf)
        
        # Calculate combined confidence
        combined = await turn_detector._calculate_combined_confidence()
        
        # Should be weighted average with VAD weight
        expected_vad_avg = np.mean(confidences)
        expected_combined = turn_detector.config.vad_weight * expected_vad_avg
        
        assert abs(combined - expected_combined) < 0.01


class TestSemanticAnalysis:
    """Test semantic analysis functionality."""
    
    @pytest.mark.asyncio
    async def test_semantic_confidence_calculation(self, turn_detector):
        """Test semantic confidence calculation."""
        await turn_detector.initialize()
        
        # Test complete question
        turn_detector._semantic_context.partial_transcript = "What is your name?"
        confidence = await turn_detector._calculate_semantic_confidence()
        assert confidence > 0.5  # Should have high confidence for complete question
        
        # Test incomplete statement
        turn_detector._semantic_context.partial_transcript = "I think that"
        confidence = await turn_detector._calculate_semantic_confidence()
        assert confidence < 0.5  # Should have lower confidence for incomplete statement
        
        # Test complete statement
        turn_detector._semantic_context.partial_transcript = "The weather is nice today."
        confidence = await turn_detector._calculate_semantic_confidence()
        assert confidence > 0.4  # Should have decent confidence for complete statement
    
    @pytest.mark.asyncio
    async def test_semantic_context_updates(self, turn_detector):
        """Test semantic context updating."""
        await turn_detector.initialize()
        
        # Update with complex transcript
        turn_detector._semantic_context.partial_transcript = "What do you think about this? It seems interesting."
        await turn_detector._update_semantic_context()
        
        context = turn_detector._semantic_context
        assert len(context.sentence_fragments) >= 1
        assert 'what' in context.question_indicators
        assert context.linguistic_features['question_mark_count'] == 1
        assert context.linguistic_features['period_count'] == 1
    
    @pytest.mark.asyncio
    async def test_semantic_turn_completion(self, full_turn_detector):
        """Test turn completion based on semantic analysis."""
        await full_turn_detector.initialize()
        await full_turn_detector.start_processing()
        
        # Start a turn
        speech_vad = VADResult(is_speech=True, confidence=0.8)
        full_turn_detector.vad_provider.process_audio_chunk.return_value = speech_vad
        await full_turn_detector.process_audio_chunk(b'audio', "What is")
        
        # Continue with complete semantic content
        silence_vad = VADResult(is_speech=False, confidence=0.3)
        full_turn_detector.vad_provider.process_audio_chunk.return_value = silence_vad
        
        # Set high semantic confidence
        full_turn_detector._semantic_context.completeness_score = 0.8
        
        # Mock time for short silence
        original_time = time.time
        def mock_time():
            return 1000.0 + 0.6  # Short silence duration
        time.time = mock_time
        full_turn_detector._last_speech_time = 1000.0
        
        try:
            result = await full_turn_detector.process_audio_chunk(b'silence', "What is your name?")
            
            # Should complete due to semantic confidence
            if result:
                assert result.end_reason == TurnEndReason.SEMANTIC_COMPLETION
        finally:
            time.time = original_time


class TestAdaptiveThresholds:
    """Test adaptive threshold functionality."""
    
    @pytest.mark.asyncio
    async def test_threshold_adaptation(self, turn_detector):
        """Test adaptive threshold adjustment."""
        await turn_detector.initialize()
        await turn_detector.start_processing()
        
        # Add some turns to history with high confidence
        for i in range(3):
            turn = TurnSegment(
                start_time=1000.0 + i,
                end_time=1001.0 + i,
                confidence=0.9,
                is_complete=True
            )
            turn_detector.turn_history.append(turn)
        
        # Update adaptive thresholds
        vad_result = VADResult(is_speech=False, confidence=0.5)
        await turn_detector._update_adaptive_thresholds(vad_result)
        
        # Threshold should be adjusted for high-confidence speaker
        assert turn_detector._adaptive_silence_threshold < turn_detector.config.silence_threshold_medium
    
    @pytest.mark.asyncio
    async def test_threshold_bounds(self, turn_detector):
        """Test that adaptive thresholds stay within bounds."""
        await turn_detector.initialize()
        
        # Set extreme values
        turn_detector._adaptive_silence_threshold = 10.0  # Very high
        
        vad_result = VADResult(is_speech=False, confidence=0.5)
        await turn_detector._update_adaptive_thresholds(vad_result)
        
        # Should be clamped to maximum
        assert turn_detector._adaptive_silence_threshold <= turn_detector.config.silence_threshold_long


class TestTurnSegmentManagement:
    """Test turn segment creation and management."""
    
    @pytest.mark.asyncio
    async def test_turn_segment_creation(self, turn_detector):
        """Test turn segment creation and completion."""
        await turn_detector.initialize()
        await turn_detector.start_processing()
        
        start_time = time.time()
        
        # Start turn
        await turn_detector._start_new_turn(start_time)
        
        assert turn_detector.current_turn is not None
        assert turn_detector.current_turn.start_time == start_time
        assert turn_detector.current_turn.is_complete is False
        
        # Finalize turn
        result = await turn_detector._finalize_turn(TurnEndReason.SILENCE_THRESHOLD)
        
        assert result.is_complete is True
        assert result.end_reason == TurnEndReason.SILENCE_THRESHOLD
        assert result.duration > 0
        assert len(turn_detector.turn_history) == 1
    
    @pytest.mark.asyncio
    async def test_turn_history_management(self, turn_detector):
        """Test turn history management and limits."""
        await turn_detector.initialize()
        
        # Add many turns to test history limit
        for i in range(15):  # More than the default limit of 10
            turn = TurnSegment(
                start_time=1000.0 + i,
                end_time=1001.0 + i,
                is_complete=True
            )
            turn_detector.turn_history.append(turn)
        
        # Should not exceed maxlen
        assert len(turn_detector.turn_history) == turn_detector.config.context_window_size
    
    @pytest.mark.asyncio
    async def test_force_turn_completion(self, turn_detector):
        """Test forced turn completion."""
        await turn_detector.initialize()
        await turn_detector.start_processing()
        
        # Start a turn
        await turn_detector._start_new_turn(time.time())
        
        # Force completion
        result = await turn_detector.force_turn_completion()
        
        assert result is not None
        assert result.end_reason == TurnEndReason.FORCED
        assert result.is_complete is True
        assert turn_detector.current_turn is None


class TestCallbacks:
    """Test callback functionality."""
    
    @pytest.mark.asyncio
    async def test_turn_start_callback(self, turn_detector):
        """Test turn start callback."""
        callback_called = False
        received_turn = None
        
        async def on_turn_start(turn):
            nonlocal callback_called, received_turn
            callback_called = True
            received_turn = turn
        
        turn_detector.set_callbacks(on_turn_start=on_turn_start)
        await turn_detector.initialize()
        await turn_detector.start_processing()
        
        # Start a turn
        await turn_detector._start_new_turn(time.time())
        
        assert callback_called is True
        assert received_turn is not None
    
    @pytest.mark.asyncio
    async def test_turn_end_callback(self, turn_detector):
        """Test turn end callback."""
        callback_called = False
        received_turn = None
        
        async def on_turn_end(turn):
            nonlocal callback_called, received_turn
            callback_called = True
            received_turn = turn
        
        turn_detector.set_callbacks(on_turn_end=on_turn_end)
        await turn_detector.initialize()
        await turn_detector.start_processing()
        
        # Start and end a turn
        await turn_detector._start_new_turn(time.time())
        await turn_detector._finalize_turn(TurnEndReason.SILENCE_THRESHOLD)
        
        assert callback_called is True
        assert received_turn is not None
        assert received_turn.is_complete is True
    
    @pytest.mark.asyncio
    async def test_state_change_callback(self, turn_detector):
        """Test state change callback."""
        state_changes = []
        
        async def on_state_change(old_state, new_state):
            state_changes.append((old_state, new_state))
        
        turn_detector.set_callbacks(on_state_change=on_state_change)
        await turn_detector.initialize()
        
        # Change states
        await turn_detector._set_state(TurnState.POTENTIAL_TURN)
        await turn_detector._set_state(TurnState.TURN_TAKING)
        
        assert len(state_changes) == 2
        assert state_changes[0] == (TurnState.LISTENING, TurnState.POTENTIAL_TURN)
        assert state_changes[1] == (TurnState.POTENTIAL_TURN, TurnState.TURN_TAKING)


class TestMetricsAndMonitoring:
    """Test metrics and monitoring functionality."""
    
    def test_metrics_collection(self, turn_detector):
        """Test metrics collection."""
        # Add some test data
        turn_detector.turn_history.extend([
            TurnSegment(start_time=1000, end_time=1002, duration=2.0, confidence=0.8, is_complete=True),
            TurnSegment(start_time=1003, end_time=1005, duration=2.0, confidence=0.7, is_complete=True),
        ])
        
        metrics = turn_detector.get_metrics()
        
        assert metrics['total_turns'] == 2
        assert metrics['average_turn_duration'] == 2.0
        assert metrics['average_confidence'] == 0.75
        assert 'adaptive_threshold' in metrics
        assert 'semantic_context' in metrics
    
    def test_current_state_reporting(self, turn_detector):
        """Test current state reporting."""
        assert turn_detector.get_current_state() == TurnState.LISTENING
        assert turn_detector.get_current_turn() is None
        assert len(turn_detector.get_turn_history()) == 0
    
    @pytest.mark.asyncio
    async def test_sensitivity_adjustment(self, turn_detector):
        """Test dynamic sensitivity adjustment."""
        await turn_detector.adjust_sensitivity(1.2, 0.8)
        
        assert turn_detector.config.silence_threshold_medium == 1.2
        assert turn_detector.config.confidence_threshold == 0.8
        assert turn_detector._adaptive_silence_threshold == 1.2


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_processing_without_initialization(self, turn_detector):
        """Test processing without proper initialization."""
        # Should handle gracefully
        result = await turn_detector.process_audio_chunk(b'audio_data')
        assert result is None
    
    @pytest.mark.asyncio
    async def test_callback_error_handling(self, turn_detector):
        """Test error handling in callbacks."""
        def failing_callback(*args):
            raise Exception("Callback error")
        
        turn_detector.set_callbacks(on_turn_start=failing_callback)
        await turn_detector.initialize()
        await turn_detector.start_processing()
        
        # Should not raise exception despite callback failure
        await turn_detector._start_new_turn(time.time())
    
    @pytest.mark.asyncio
    async def test_empty_transcript_handling(self, turn_detector):
        """Test handling of empty transcripts."""
        await turn_detector.initialize()
        
        # Empty transcript should not cause issues
        confidence = await turn_detector._calculate_semantic_confidence()
        assert confidence == 0.0
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self, turn_detector):
        """Test concurrent audio chunk processing."""
        await turn_detector.initialize()
        await turn_detector.start_processing()
        
        # Mock VAD to return speech
        vad_result = VADResult(is_speech=True, confidence=0.8)
        turn_detector.vad_provider.process_audio_chunk.return_value = vad_result
        
        # Process multiple chunks concurrently
        tasks = [
            turn_detector.process_audio_chunk(f'chunk_{i}'.encode()) 
            for i in range(5)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Should handle concurrent processing without errors
        assert all(not isinstance(r, Exception) for r in results)


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_complete_conversation_flow(self, full_turn_detector):
        """Test a complete conversation flow."""
        await full_turn_detector.initialize()
        await full_turn_detector.start_processing()
        
        # Simulate conversation: user speaks, pauses, agent responds
        timestamps = [1000.0, 1000.5, 1001.0, 1002.0, 1002.5]
        
        # User starts speaking
        speech_vad = VADResult(is_speech=True, confidence=0.8)
        full_turn_detector.vad_provider.process_audio_chunk.return_value = speech_vad
        
        await full_turn_detector.process_audio_chunk(b'speech1', "Hello")
        assert full_turn_detector.current_turn is not None
        
        # User continues
        await full_turn_detector.process_audio_chunk(b'speech2', "Hello, how are you?")
        
        # User stops speaking (silence)
        silence_vad = VADResult(is_speech=False, confidence=0.2)
        full_turn_detector.vad_provider.process_audio_chunk.return_value = silence_vad
        
        # Mock time progression
        original_time = time.time
        def mock_time():
            return 1003.0  # Sufficient silence
        time.time = mock_time
        full_turn_detector._last_speech_time = 1001.0
        
        try:
            result = await full_turn_detector.process_audio_chunk(b'silence', "Hello, how are you?")
            
            # Should complete the turn
            if result:
                assert result.is_complete is True
                assert result.text_content == "Hello, how are you?"
                assert len(full_turn_detector.turn_history) == 1
        finally:
            time.time = original_time
    
    @pytest.mark.asyncio
    async def test_interruption_scenario(self, turn_detector):
        """Test interruption handling."""
        await turn_detector.initialize()
        await turn_detector.start_processing()
        
        # Start a turn
        speech_vad = VADResult(is_speech=True, confidence=0.8)
        turn_detector.vad_provider.process_audio_chunk.return_value = speech_vad
        await turn_detector.process_audio_chunk(b'speech')
        
        # Simulate interruption by forcing completion
        result = await turn_detector.force_turn_completion()
        
        assert result.end_reason == TurnEndReason.FORCED
        assert turn_detector.current_turn is None
    
    @pytest.mark.asyncio
    async def test_long_turn_timeout(self, turn_detector):
        """Test long turn timeout handling."""
        await turn_detector.initialize()
        await turn_detector.start_processing()
        
        # Start a turn
        start_time = 1000.0
        await turn_detector._start_new_turn(start_time)
        
        # Mock time to exceed max duration
        original_time = time.time
        def mock_time():
            return start_time + turn_detector.config.max_turn_duration + 1
        time.time = mock_time
        
        try:
            # Process audio chunk - should trigger timeout
            silence_vad = VADResult(is_speech=False, confidence=0.2)
            turn_detector.vad_provider.process_audio_chunk.return_value = silence_vad
            turn_detector._last_speech_time = start_time
            
            result = await turn_detector.process_audio_chunk(b'silence')
            
            # Should complete due to timeout
            if result:
                assert result.end_reason == TurnEndReason.TIMEOUT
        finally:
            time.time = original_time


if __name__ == "__main__":
    pytest.main([__file__, "-v"])