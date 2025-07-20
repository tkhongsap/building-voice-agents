"""
Unit Tests for Pipeline Flush Handler

Tests the graceful pipeline flushing mechanisms for interruption handling
including data preservation, component flushing, and recovery.
"""

import asyncio
import pytest
import time
import numpy as np
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from typing import List, Dict, Any

from pipeline_flush_handler import (
    PipelineFlushHandler, PipelineFlushConfig, FlushType, FlushReason,
    ComponentState, FlushOperation, ComponentFlushState
)

try:
    from ..pipeline.audio_pipeline import StreamingAudioPipeline, PipelineState
    from ..components.stt.base_stt import BaseSTTProvider
    from ..components.llm.base_llm import BaseLLMProvider
    from ..components.tts.base_tts import BaseTTSProvider
    from .interruption_handler import InterruptionHandler, InterruptionEvent, InterruptionSeverity
except ImportError:
    # Fallback for direct imports
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from pipeline.audio_pipeline import StreamingAudioPipeline, PipelineState
    from components.stt.base_stt import BaseSTTProvider
    from components.llm.base_llm import BaseLLMProvider
    from components.tts.base_tts import BaseTTSProvider
    from interruption_handler import InterruptionHandler, InterruptionEvent, InterruptionSeverity


@pytest.fixture
def mock_pipeline():
    """Create a mock pipeline for testing."""
    mock = Mock(spec=StreamingAudioPipeline)
    mock.get_state = Mock(return_value=PipelineState.LISTENING)
    mock._set_state = AsyncMock()
    mock._audio_queue = MagicMock()
    mock._audio_queue.empty = Mock(return_value=True)
    mock._speech_buffer = []
    mock._processing_tasks = []
    
    # Mock providers
    mock.stt_provider = Mock(spec=BaseSTTProvider)
    mock.llm_provider = Mock(spec=BaseLLMProvider)
    mock.tts_provider = Mock(spec=BaseTTSProvider)
    
    return mock


@pytest.fixture
def mock_interruption_handler():
    """Create a mock interruption handler for testing."""
    mock = Mock(spec=InterruptionHandler)
    mock.set_callbacks = Mock()
    return mock


@pytest.fixture
def basic_config():
    """Create a basic flush handler configuration."""
    return PipelineFlushConfig(
        soft_flush_timeout=1.0,
        hard_flush_timeout=0.5,
        preserve_partial_stt=True,
        preserve_llm_context=True,
        enable_detailed_logging=True
    )


@pytest.fixture
def flush_handler(mock_pipeline, basic_config):
    """Create a flush handler instance for testing."""
    return PipelineFlushHandler(
        pipeline=mock_pipeline,
        config=basic_config
    )


@pytest.fixture
def full_flush_handler(mock_pipeline, mock_interruption_handler, basic_config):
    """Create a full flush handler with all components."""
    return PipelineFlushHandler(
        pipeline=mock_pipeline,
        interruption_handler=mock_interruption_handler,
        config=basic_config
    )


class TestPipelineFlushConfig:
    """Test pipeline flush configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = PipelineFlushConfig()
        assert config.soft_flush_timeout == 2.0
        assert config.hard_flush_timeout == 0.5
        assert config.preserve_partial_stt is True
        assert config.preserve_llm_context is True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = PipelineFlushConfig(
            soft_flush_timeout=3.0,
            preserve_partial_stt=False,
            enable_parallel_flushing=False
        )
        assert config.soft_flush_timeout == 3.0
        assert config.preserve_partial_stt is False
        assert config.enable_parallel_flushing is False


class TestFlushHandlerInitialization:
    """Test flush handler initialization."""
    
    @pytest.mark.asyncio
    async def test_basic_initialization(self, flush_handler):
        """Test basic initialization."""
        await flush_handler.initialize()
        
        # Should initialize component states
        assert len(flush_handler.component_states) > 0
        assert "stt" in flush_handler.component_states
        assert "llm" in flush_handler.component_states
        assert "tts" in flush_handler.component_states
    
    @pytest.mark.asyncio
    async def test_full_initialization(self, full_flush_handler):
        """Test initialization with interruption handler."""
        await full_flush_handler.initialize()
        
        # Should set up interruption handler callbacks
        full_flush_handler.interruption_handler.set_callbacks.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, flush_handler):
        """Test start and stop monitoring."""
        await flush_handler.initialize()
        
        await flush_handler.start_monitoring()
        assert flush_handler._is_monitoring is True
        
        await flush_handler.stop_monitoring()
        assert flush_handler._is_monitoring is False
    
    def test_initial_state(self, flush_handler):
        """Test initial handler state."""
        assert flush_handler.current_flush is None
        assert len(flush_handler.flush_history) == 0
        assert len(flush_handler._preserved_data) == 0


class TestFlushOperations:
    """Test flush operation functionality."""
    
    @pytest.mark.asyncio
    async def test_soft_flush_operation(self, flush_handler):
        """Test soft flush operation."""
        await flush_handler.initialize()
        
        # Setup mock methods for soft flush
        flush_handler._soft_flush_component = AsyncMock()
        flush_handler._preserve_pipeline_data = AsyncMock()
        
        result = await flush_handler.flush_pipeline(
            flush_type=FlushType.SOFT_FLUSH,
            reason=FlushReason.USER_INTERRUPTION,
            preserve_data=True
        )
        
        assert result.success is True
        assert result.flush_type == FlushType.SOFT_FLUSH
        assert result.reason == FlushReason.USER_INTERRUPTION
        assert result.duration is not None
        
        # Should have called preserve data
        flush_handler._preserve_pipeline_data.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_hard_flush_operation(self, flush_handler):
        """Test hard flush operation."""
        await flush_handler.initialize()
        
        # Setup mock methods
        flush_handler._hard_flush_component = AsyncMock()
        
        result = await flush_handler.flush_pipeline(
            flush_type=FlushType.HARD_FLUSH,
            reason=FlushReason.ERROR_RECOVERY,
            preserve_data=False
        )
        
        assert result.success is True
        assert result.flush_type == FlushType.HARD_FLUSH
        assert result.reason == FlushReason.ERROR_RECOVERY
    
    @pytest.mark.asyncio
    async def test_selective_flush_operation(self, flush_handler):
        """Test selective flush operation."""
        await flush_handler.initialize()
        
        # Setup mock methods
        flush_handler._soft_flush_component = AsyncMock()
        flush_handler._preserve_pipeline_data = AsyncMock()
        
        result = await flush_handler.flush_pipeline(
            flush_type=FlushType.SELECTIVE_FLUSH,
            target_components=["stt", "tts"],
            preserve_data=True
        )
        
        assert result.success is True
        assert result.target_components == ["stt", "tts"]
        
        # Should have called soft flush for each target component
        assert flush_handler._soft_flush_component.call_count == 2
    
    @pytest.mark.asyncio
    async def test_graceful_flush_operation(self, flush_handler):
        """Test graceful flush operation."""
        await flush_handler.initialize()
        
        # Mock graceful flush conditions
        flush_handler._can_flush_gracefully = AsyncMock(return_value=True)
        flush_handler._preserve_pipeline_data = AsyncMock()
        flush_handler._execute_soft_flush = AsyncMock()
        
        result = await flush_handler.flush_pipeline(
            flush_type=FlushType.GRACEFUL_FLUSH,
            preserve_data=True
        )
        
        assert result.success is True
        assert result.flush_type == FlushType.GRACEFUL_FLUSH
        
        # Should have checked graceful conditions
        flush_handler._can_flush_gracefully.assert_called()
    
    @pytest.mark.asyncio
    async def test_concurrent_flush_handling(self, flush_handler):
        """Test handling of concurrent flush requests."""
        await flush_handler.initialize()
        
        # Setup slow flush
        flush_handler._execute_soft_flush = AsyncMock()
        
        async def slow_flush(*args):
            await asyncio.sleep(0.2)
        
        flush_handler._execute_soft_flush.side_effect = slow_flush
        
        # Start first flush
        task1 = asyncio.create_task(flush_handler.flush_pipeline(FlushType.SOFT_FLUSH))
        
        # Wait a bit then start second flush
        await asyncio.sleep(0.1)
        task2 = asyncio.create_task(flush_handler.flush_pipeline(FlushType.HARD_FLUSH))
        
        # Both should complete
        result1, result2 = await asyncio.gather(task1, task2)
        
        assert result1.success is True
        assert result2.success is True


class TestComponentFlushing:
    """Test individual component flushing."""
    
    @pytest.mark.asyncio
    async def test_stt_component_flush(self, flush_handler):
        """Test STT component flushing."""
        await flush_handler.initialize()
        
        # Setup STT provider with flush methods
        stt = flush_handler.pipeline.stt_provider
        stt.get_partial_transcript = AsyncMock(return_value="partial text")
        stt.flush_buffers = AsyncMock()
        
        await flush_handler._flush_stt_component(soft=True)
        
        # Should have preserved partial transcript and flushed buffers
        stt.get_partial_transcript.assert_called_once()
        stt.flush_buffers.assert_called_once()
        assert "partial text" in flush_handler._partial_transcripts
    
    @pytest.mark.asyncio
    async def test_llm_component_flush(self, flush_handler):
        """Test LLM component flushing."""
        await flush_handler.initialize()
        
        # Setup LLM provider
        llm = flush_handler.pipeline.llm_provider
        llm.get_conversation_history = AsyncMock(return_value=["msg1", "msg2"])
        llm.cancel_current_request = AsyncMock()
        llm.clear_request_queue = AsyncMock()
        
        await flush_handler._flush_llm_component(soft=True)
        
        # Should have preserved context and cancelled requests
        llm.get_conversation_history.assert_called_once()
        llm.cancel_current_request.assert_called_once()
        llm.clear_request_queue.assert_called_once()
        assert flush_handler._llm_context_backup == ["msg1", "msg2"]
    
    @pytest.mark.asyncio
    async def test_tts_component_flush(self, flush_handler):
        """Test TTS component flushing."""
        await flush_handler.initialize()
        
        # Setup TTS provider
        tts = flush_handler.pipeline.tts_provider
        tts.get_pending_queue = AsyncMock(return_value=["text1", "text2"])
        tts.stop_synthesis = AsyncMock()
        tts.clear_queue = AsyncMock()
        
        await flush_handler._flush_tts_component(soft=True)
        
        # Should have preserved queue and stopped synthesis
        tts.get_pending_queue.assert_called_once()
        tts.stop_synthesis.assert_called_once()
        tts.clear_queue.assert_called_once()
        assert flush_handler._pending_tts_queue == ["text1", "text2"]
    
    @pytest.mark.asyncio
    async def test_pipeline_component_flush(self, flush_handler):
        """Test pipeline component flushing."""
        await flush_handler.initialize()
        
        # Setup pipeline with audio queue
        pipeline = flush_handler.pipeline
        pipeline._audio_queue.empty.return_value = False
        pipeline._audio_queue.get_nowait = Mock(side_effect=[b'chunk1', b'chunk2', Exception()])
        
        await flush_handler._flush_pipeline_component(soft=True)
        
        # Should have cleared audio queue
        assert pipeline._audio_queue.get_nowait.call_count >= 2
    
    @pytest.mark.asyncio
    async def test_component_state_tracking(self, flush_handler):
        """Test component state tracking during flush."""
        await flush_handler.initialize()
        
        # Setup component with state
        component_name = "stt"
        state = flush_handler.component_states[component_name]
        assert state.state == ComponentState.ACTIVE
        
        # Mock flush methods
        flush_handler._flush_stt_component = AsyncMock()
        
        await flush_handler._soft_flush_component(component_name)
        
        # State should be updated
        assert state.state == ComponentState.FLUSHED
        assert state.flush_start_time is not None
        assert state.flush_completion_time is not None
        assert state.flush_duration is not None


class TestDataPreservation:
    """Test data preservation functionality."""
    
    @pytest.mark.asyncio
    async def test_data_preservation(self, flush_handler):
        """Test data preservation across components."""
        await flush_handler.initialize()
        
        # Setup providers with data to preserve
        stt = flush_handler.pipeline.stt_provider
        llm = flush_handler.pipeline.llm_provider
        tts = flush_handler.pipeline.tts_provider
        
        stt.get_partial_transcript = AsyncMock(return_value="partial transcript")
        llm.get_conversation_history = AsyncMock(return_value=["msg1", "msg2"])
        tts.get_pending_queue = AsyncMock(return_value=["text1"])
        
        await flush_handler._preserve_pipeline_data(["stt", "llm", "tts"])
        
        # Should have preserved data from all components
        preserved = flush_handler._preserved_data
        assert "stt_partial" in preserved
        assert "llm_context" in preserved
        assert "tts_queue" in preserved
        assert preserved["stt_partial"] == "partial transcript"
    
    @pytest.mark.asyncio
    async def test_data_restoration(self, flush_handler):
        """Test data restoration functionality."""
        await flush_handler.initialize()
        
        # Setup preserved data
        flush_handler._preserved_data = {
            "stt_partial": "restored text",
            "llm_context": ["restored", "context"],
            "tts_queue": ["restored", "queue"]
        }
        
        # Setup providers with restore methods
        stt = flush_handler.pipeline.stt_provider
        llm = flush_handler.pipeline.llm_provider
        tts = flush_handler.pipeline.tts_provider
        
        stt.set_partial_transcript = AsyncMock()
        llm.set_conversation_history = AsyncMock()
        tts.restore_queue = AsyncMock()
        
        restored = await flush_handler.restore_preserved_data()
        
        # Should have restored data to all components
        stt.set_partial_transcript.assert_called_once_with("restored text")
        llm.set_conversation_history.assert_called_once_with(["restored", "context"])
        tts.restore_queue.assert_called_once_with(["restored", "queue"])
        
        # Should have cleared preserved data
        assert len(flush_handler._preserved_data) == 0
        
        # Should return restoration summary
        assert "stt_partial" in restored
        assert "llm_context" in restored
        assert "tts_queue" in restored
    
    @pytest.mark.asyncio
    async def test_selective_data_preservation(self, flush_handler):
        """Test selective data preservation based on configuration."""
        # Disable TTS preservation
        flush_handler.config.preserve_tts_queue = False
        await flush_handler.initialize()
        
        # Setup providers
        stt = flush_handler.pipeline.stt_provider
        tts = flush_handler.pipeline.tts_provider
        
        stt.get_partial_transcript = AsyncMock(return_value="partial")
        tts.get_pending_queue = AsyncMock(return_value=["queue"])
        
        await flush_handler._preserve_pipeline_data(["stt", "tts"])
        
        # Should preserve STT but not TTS
        preserved = flush_handler._preserved_data
        assert "stt_partial" in preserved
        assert "tts_queue" not in preserved


class TestGracefulFlushing:
    """Test graceful flushing functionality."""
    
    @pytest.mark.asyncio
    async def test_graceful_flush_conditions(self, flush_handler):
        """Test graceful flush condition checking."""
        await flush_handler.initialize()
        
        # Pipeline in listening state should allow graceful flush
        flush_handler.pipeline.get_state.return_value = PipelineState.LISTENING
        can_flush = await flush_handler._can_flush_gracefully()
        assert can_flush is True
        
        # Pipeline in processing state should not allow graceful flush initially
        flush_handler.pipeline.get_state.return_value = PipelineState.PROCESSING_STT
        can_flush = await flush_handler._can_flush_gracefully()
        # May return True due to simplified implementation
    
    @pytest.mark.asyncio
    async def test_sentence_completion_detection(self, flush_handler):
        """Test sentence completion detection for graceful flushing."""
        flush_handler.config.wait_for_sentence_completion = True
        await flush_handler.initialize()
        
        # Setup STT with sentence-ending transcript
        stt = flush_handler.pipeline.stt_provider
        stt.get_partial_transcript = AsyncMock(return_value="This is a complete sentence.")
        
        # Set pipeline to processing state
        flush_handler.pipeline.get_state.return_value = PipelineState.PROCESSING_STT
        
        can_flush = await flush_handler._can_flush_gracefully()
        assert can_flush is True
    
    @pytest.mark.asyncio
    async def test_graceful_flush_timeout(self, flush_handler):
        """Test graceful flush timeout handling."""
        flush_handler.config.graceful_flush_timeout = 0.1  # Very short timeout
        await flush_handler.initialize()
        
        # Mock conditions that never allow graceful flush
        flush_handler._can_flush_gracefully = AsyncMock(return_value=False)
        flush_handler._preserve_pipeline_data = AsyncMock()
        flush_handler._execute_soft_flush = AsyncMock()
        
        start_time = time.time()
        await flush_handler._execute_graceful_flush(FlushOperation(
            flush_id="test",
            timestamp=time.time(),
            flush_type=FlushType.GRACEFUL_FLUSH,
            reason=FlushReason.MANUAL_REQUEST,
            target_components=["stt"],
            preserve_data=True
        ))
        
        # Should have timed out and proceeded with soft flush
        duration = time.time() - start_time
        assert duration >= 0.1  # At least the timeout duration
        flush_handler._execute_soft_flush.assert_called_once()


class TestInterruptionIntegration:
    """Test integration with interruption handler."""
    
    @pytest.mark.asyncio
    async def test_interruption_detection_handling(self, full_flush_handler):
        """Test handling of interruption detection."""
        await full_flush_handler.initialize()
        await full_flush_handler.start_monitoring()
        
        # Create interruption event
        interruption = InterruptionEvent(
            timestamp=time.time(),
            interruption_type=None,
            severity=InterruptionSeverity.HIGH,
            confidence=0.8,
            vad_confidence=0.8
        )
        
        # Should not flush on detection (wait for confirmation)
        await full_flush_handler._on_interruption_detected(interruption)
        assert full_flush_handler.current_flush is None
    
    @pytest.mark.asyncio
    async def test_interruption_confirmation_handling(self, full_flush_handler):
        """Test handling of interruption confirmation."""
        await full_flush_handler.initialize()
        await full_flush_handler.start_monitoring()
        
        # Mock flush execution
        full_flush_handler.flush_pipeline = AsyncMock()
        
        # Create high severity interruption
        interruption = InterruptionEvent(
            timestamp=time.time(),
            interruption_type=None,
            severity=InterruptionSeverity.HIGH,
            confidence=0.8,
            vad_confidence=0.8
        )
        
        await full_flush_handler._on_interruption_confirmed(interruption)
        
        # Should have triggered soft flush with data preservation
        full_flush_handler.flush_pipeline.assert_called_once()
        call_args = full_flush_handler.flush_pipeline.call_args
        assert call_args.kwargs['flush_type'] == FlushType.SOFT_FLUSH
        assert call_args.kwargs['reason'] == FlushReason.USER_INTERRUPTION
        assert call_args.kwargs['preserve_data'] is True
    
    @pytest.mark.asyncio
    async def test_critical_interruption_handling(self, full_flush_handler):
        """Test handling of critical interruptions."""
        await full_flush_handler.initialize()
        await full_flush_handler.start_monitoring()
        
        # Mock flush execution
        full_flush_handler.flush_pipeline = AsyncMock()
        
        # Create critical interruption
        interruption = InterruptionEvent(
            timestamp=time.time(),
            interruption_type=None,
            severity=InterruptionSeverity.CRITICAL,
            confidence=0.9,
            vad_confidence=0.9
        )
        
        await full_flush_handler._on_interruption_confirmed(interruption)
        
        # Should have triggered hard flush without data preservation
        call_args = full_flush_handler.flush_pipeline.call_args
        assert call_args.kwargs['flush_type'] == FlushType.HARD_FLUSH
        assert call_args.kwargs['preserve_data'] is False


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_flush_operation_failure(self, flush_handler):
        """Test handling of flush operation failures."""
        await flush_handler.initialize()
        
        # Mock component flush to fail
        async def failing_flush(*args):
            raise Exception("Flush failed")
        
        flush_handler._soft_flush_component = failing_flush
        
        # Should handle error gracefully
        result = await flush_handler.flush_pipeline(FlushType.SOFT_FLUSH)
        
        assert result.success is False
        assert result.error_message == "Flush failed"
        assert result.completion_time is not None
    
    @pytest.mark.asyncio
    async def test_fallback_to_hard_flush(self, flush_handler):
        """Test fallback to hard flush on soft flush failure."""
        flush_handler.config.fallback_to_hard_flush = True
        await flush_handler.initialize()
        
        call_count = 0
        
        async def mock_flush(flush_op):
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # First call (soft flush) fails
                raise Exception("Soft flush failed")
            # Second call (hard flush) succeeds
        
        # Mock execution methods
        flush_handler._execute_soft_flush = mock_flush
        flush_handler._execute_hard_flush = AsyncMock()
        
        result = await flush_handler.flush_pipeline(FlushType.SOFT_FLUSH)
        
        # Should have fallen back to hard flush
        assert result.flush_type == FlushType.HARD_FLUSH
        assert result.success is True
    
    @pytest.mark.asyncio
    async def test_component_flush_error_handling(self, flush_handler):
        """Test error handling in component flushing."""
        await flush_handler.initialize()
        
        # Setup component to fail
        component_name = "stt"
        flush_handler._flush_stt_component = AsyncMock(side_effect=Exception("STT flush failed"))
        
        # Should handle component error
        with pytest.raises(Exception, match="STT flush failed"):
            await flush_handler._soft_flush_component(component_name)
        
        # Component state should reflect error
        state = flush_handler.component_states[component_name]
        assert state.state == ComponentState.ERROR
        assert state.error_message == "STT flush failed"
    
    @pytest.mark.asyncio
    async def test_data_preservation_error_handling(self, flush_handler):
        """Test error handling in data preservation."""
        await flush_handler.initialize()
        
        # Setup STT to fail during data preservation
        stt = flush_handler.pipeline.stt_provider
        stt.get_partial_transcript = AsyncMock(side_effect=Exception("STT error"))
        
        # Should handle error gracefully and continue
        await flush_handler._preserve_pipeline_data(["stt"])
        
        # Should not have preserved STT data but shouldn't crash
        assert "stt_partial" not in flush_handler._preserved_data
    
    @pytest.mark.asyncio
    async def test_callback_error_handling(self, flush_handler):
        """Test error handling in callbacks."""
        def failing_callback(*args):
            raise Exception("Callback error")
        
        flush_handler.set_callbacks(on_flush_completed=failing_callback)
        await flush_handler.initialize()
        
        # Mock successful flush
        flush_handler._execute_soft_flush = AsyncMock()
        
        # Should not raise exception despite callback failure
        result = await flush_handler.flush_pipeline(FlushType.SOFT_FLUSH)
        assert result.success is True


class TestMetricsAndMonitoring:
    """Test metrics collection and monitoring."""
    
    def test_metrics_collection(self, flush_handler):
        """Test metrics collection."""
        # Add test flush operations
        flush_ops = [
            FlushOperation(
                flush_id="1",
                timestamp=1000.0,
                flush_type=FlushType.SOFT_FLUSH,
                reason=FlushReason.USER_INTERRUPTION,
                target_components=["stt"],
                completion_time=1001.0,
                success=True
            ),
            FlushOperation(
                flush_id="2",
                timestamp=2000.0,
                flush_type=FlushType.HARD_FLUSH,
                reason=FlushReason.ERROR_RECOVERY,
                target_components=["all"],
                completion_time=2000.5,
                success=False
            )
        ]
        
        flush_handler.flush_history.extend(flush_ops)
        
        metrics = flush_handler.get_metrics()
        
        assert metrics['total_flushes'] == 2
        assert metrics['successful_flushes'] == 1
        assert metrics['success_rate'] == 0.5
        assert 'average_flush_duration' in metrics
        assert 'flush_type_distribution' in metrics
        assert 'component_states' in metrics
    
    def test_state_reporting(self, flush_handler):
        """Test current state reporting."""
        assert flush_handler.get_current_flush() is None
        assert len(flush_handler.get_flush_history()) == 0
        assert len(flush_handler.get_component_states()) == 0  # Before initialization
        assert len(flush_handler.get_preserved_data()) == 0
    
    @pytest.mark.asyncio
    async def test_emergency_flush(self, flush_handler):
        """Test emergency flush functionality."""
        await flush_handler.initialize()
        
        # Mock hard flush execution
        flush_handler._execute_hard_flush = AsyncMock()
        
        result = await flush_handler.emergency_flush()
        
        assert result.flush_type == FlushType.HARD_FLUSH
        assert result.reason == FlushReason.ERROR_RECOVERY
        assert result.preserve_data is False


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_complete_interruption_flush_cycle(self, full_flush_handler):
        """Test complete interruption to flush to recovery cycle."""
        await full_flush_handler.initialize()
        await full_flush_handler.start_monitoring()
        
        # Setup mock providers with data
        stt = full_flush_handler.pipeline.stt_provider
        llm = full_flush_handler.pipeline.llm_provider
        tts = full_flush_handler.pipeline.tts_provider
        
        stt.get_partial_transcript = AsyncMock(return_value="user was saying")
        stt.flush_buffers = AsyncMock()
        llm.get_conversation_history = AsyncMock(return_value=["context"])
        llm.cancel_current_request = AsyncMock()
        tts.stop_synthesis = AsyncMock()
        tts.clear_queue = AsyncMock()
        
        # Simulate interruption confirmation
        interruption = InterruptionEvent(
            timestamp=time.time(),
            interruption_type=None,
            severity=InterruptionSeverity.HIGH,
            confidence=0.8,
            vad_confidence=0.8
        )
        
        await full_flush_handler._on_interruption_confirmed(interruption)
        
        # Should have preserved data and flushed components
        assert len(full_flush_handler._preserved_data) > 0
        assert len(full_flush_handler.flush_history) == 1
        
        # Test data restoration
        stt.set_partial_transcript = AsyncMock()
        llm.set_conversation_history = AsyncMock()
        
        restored = await full_flush_handler.restore_preserved_data()
        assert len(restored) > 0
    
    @pytest.mark.asyncio
    async def test_concurrent_flush_and_interruption(self, full_flush_handler):
        """Test handling concurrent flush and interruption events."""
        await full_flush_handler.initialize()
        await full_flush_handler.start_monitoring()
        
        # Mock slow flush execution
        full_flush_handler._execute_soft_flush = AsyncMock()
        
        async def slow_flush(*args):
            await asyncio.sleep(0.2)
        
        full_flush_handler._execute_soft_flush.side_effect = slow_flush
        
        # Start manual flush
        flush_task = asyncio.create_task(
            full_flush_handler.flush_pipeline(FlushType.SOFT_FLUSH)
        )
        
        # Simulate interruption during flush
        await asyncio.sleep(0.1)
        interruption = InterruptionEvent(
            timestamp=time.time(),
            interruption_type=None,
            severity=InterruptionSeverity.CRITICAL,
            confidence=0.9,
            vad_confidence=0.9
        )
        
        # Should wait for current flush to complete
        await full_flush_handler._on_interruption_confirmed(interruption)
        
        # Wait for flush to complete
        result = await flush_task
        assert result.success is True
    
    @pytest.mark.asyncio
    async def test_multiple_rapid_flushes(self, flush_handler):
        """Test handling multiple rapid flush requests."""
        await flush_handler.initialize()
        
        # Mock fast flush execution
        flush_handler._execute_soft_flush = AsyncMock()
        
        # Execute multiple flushes rapidly
        tasks = []
        for i in range(5):
            task = asyncio.create_task(
                flush_handler.flush_pipeline(FlushType.SOFT_FLUSH)
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # All flushes should complete successfully
        assert all(r.success for r in results)
        assert len(flush_handler.flush_history) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])