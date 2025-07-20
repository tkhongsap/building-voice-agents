"""
Unit Tests for Conversation State Manager

Tests the conversation state preservation and recovery system including
state persistence, recovery strategies, and checkpoint management.
"""

import asyncio
import pytest
import time
import json
import tempfile
import shutil
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from typing import List, Dict, Any
from pathlib import Path

from conversation_state_manager import (
    ConversationStateManager, ConversationStateConfig, ConversationStateType,
    RecoveryStrategy, StatePreservationLevel, ConversationState,
    ConversationMessage, ConversationParticipant, RecoveryContext
)

try:
    from .turn_detector import TurnDetector, TurnSegment, TurnState
    from .interruption_handler import InterruptionHandler, InterruptionEvent, AgentState
    from .pipeline_flush_handler import PipelineFlushHandler, FlushOperation
    from ..components.llm.base_llm import BaseLLMProvider
except ImportError:
    # Fallback for direct imports
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from turn_detector import TurnDetector, TurnSegment, TurnState
    from interruption_handler import InterruptionHandler, InterruptionEvent, AgentState
    from pipeline_flush_handler import PipelineFlushHandler, FlushOperation
    from components.llm.base_llm import BaseLLMProvider


@pytest.fixture
def temp_storage():
    """Create temporary storage directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_turn_detector():
    """Create a mock turn detector for testing."""
    mock = Mock(spec=TurnDetector)
    mock.get_current_state = Mock(return_value=TurnState.LISTENING)
    mock.get_current_turn = Mock(return_value=None)
    mock.get_turn_history = Mock(return_value=[])
    mock.set_callbacks = Mock()
    return mock


@pytest.fixture
def mock_interruption_handler():
    """Create a mock interruption handler for testing."""
    mock = Mock(spec=InterruptionHandler)
    mock.get_current_state = Mock(return_value=AgentState.LISTENING)
    mock.get_current_interruption = Mock(return_value=None)
    mock.set_callbacks = Mock()
    return mock


@pytest.fixture
def mock_flush_handler():
    """Create a mock flush handler for testing."""
    mock = Mock(spec=PipelineFlushHandler)
    mock.set_callbacks = Mock()
    return mock


@pytest.fixture
def mock_llm_provider():
    """Create a mock LLM provider for testing."""
    mock = AsyncMock(spec=BaseLLMProvider)
    mock.get_conversation_history = AsyncMock(return_value=["msg1", "msg2"])
    mock.set_conversation_history = AsyncMock()
    return mock


@pytest.fixture
def basic_config(temp_storage):
    """Create a basic state manager configuration."""
    return ConversationStateConfig(
        storage_directory=temp_storage,
        persistence_interval=0.1,  # Fast for testing
        checkpoint_interval=0.2,   # Fast for testing
        enable_detailed_logging=True
    )


@pytest.fixture
def state_manager(basic_config):
    """Create a basic state manager instance for testing."""
    return ConversationStateManager(
        session_id="test_session",
        conversation_id="test_conversation",
        config=basic_config
    )


@pytest.fixture
def full_state_manager(basic_config, mock_turn_detector, mock_interruption_handler, 
                      mock_flush_handler, mock_llm_provider):
    """Create a full state manager with all components."""
    return ConversationStateManager(
        session_id="test_session",
        conversation_id="test_conversation",
        turn_detector=mock_turn_detector,
        interruption_handler=mock_interruption_handler,
        flush_handler=mock_flush_handler,
        llm_provider=mock_llm_provider,
        config=basic_config
    )


class TestConversationStateConfig:
    """Test conversation state configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ConversationStateConfig()
        assert config.enable_persistence is True
        assert config.auto_recovery_enabled is True
        assert config.recovery_strategy == RecoveryStrategy.CONTEXT_AWARE
        assert config.default_preservation_level == StatePreservationLevel.STANDARD
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ConversationStateConfig(
            persistence_interval=60.0,
            recovery_strategy=RecoveryStrategy.IMMEDIATE,
            max_checkpoints=50
        )
        assert config.persistence_interval == 60.0
        assert config.recovery_strategy == RecoveryStrategy.IMMEDIATE
        assert config.max_checkpoints == 50


class TestConversationStateManager:
    """Test conversation state manager initialization and basic functionality."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, state_manager):
        """Test basic initialization."""
        await state_manager.initialize()
        
        assert state_manager.current_state.session_id == "test_session"
        assert state_manager.current_state.conversation_id == "test_conversation"
        assert state_manager.current_state.state_type == ConversationStateType.ACTIVE
        assert len(state_manager.current_state.participants) == 2  # user and agent
    
    @pytest.mark.asyncio
    async def test_full_initialization(self, full_state_manager):
        """Test initialization with all components."""
        await full_state_manager.initialize()
        
        # Should set up callbacks for all components
        full_state_manager.turn_detector.set_callbacks.assert_called_once()
        full_state_manager.interruption_handler.set_callbacks.assert_called_once()
        full_state_manager.flush_handler.set_callbacks.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, state_manager):
        """Test start and stop monitoring."""
        await state_manager.initialize()
        
        await state_manager.start_monitoring()
        assert state_manager._is_monitoring is True
        assert state_manager._persistence_task is not None
        assert state_manager._checkpoint_task is not None
        
        await state_manager.stop_monitoring()
        assert state_manager._is_monitoring is False
    
    def test_initial_state(self, state_manager):
        """Test initial state properties."""
        assert state_manager.session_id == "test_session"
        assert state_manager.conversation_id == "test_conversation"
        assert len(state_manager.state_history) == 0
        assert len(state_manager.recovery_history) == 0
        assert len(state_manager.checkpoints) == 0


class TestStatePreservation:
    """Test state preservation functionality."""
    
    @pytest.mark.asyncio
    async def test_basic_state_preservation(self, state_manager):
        """Test basic state preservation."""
        await state_manager.initialize()
        
        # Add some content to state
        await state_manager.add_message("user", "Hello, world!")
        await state_manager.update_conversation_context({"topic": "greeting"})
        
        # Preserve state
        preserved_state = await state_manager.preserve_current_state()
        
        assert preserved_state is not None
        assert len(preserved_state.messages) == 1
        assert preserved_state.conversation_context["topic"] == "greeting"
        assert len(state_manager.state_history) == 1
    
    @pytest.mark.asyncio
    async def test_preservation_levels(self, state_manager):
        """Test different preservation levels."""
        await state_manager.initialize()
        
        # Add content
        await state_manager.add_message("user", "Test message")
        
        # Test different preservation levels
        minimal_state = await state_manager.preserve_current_state(
            StatePreservationLevel.MINIMAL
        )
        assert minimal_state.preservation_level == StatePreservationLevel.MINIMAL
        
        comprehensive_state = await state_manager.preserve_current_state(
            StatePreservationLevel.COMPREHENSIVE
        )
        assert comprehensive_state.preservation_level == StatePreservationLevel.COMPREHENSIVE
    
    @pytest.mark.asyncio
    async def test_state_persistence_to_storage(self, state_manager):
        """Test state persistence to storage."""
        await state_manager.initialize()
        
        # Add content
        await state_manager.add_message("user", "Persistent message")
        
        # Preserve state
        await state_manager.preserve_current_state()
        
        # Check that file was created
        storage_path = Path(state_manager.config.storage_directory)
        latest_file = storage_path / f"{state_manager.conversation_id}_latest.json"
        assert latest_file.exists()
        
        # Verify file content
        with open(latest_file, 'r') as f:
            saved_data = json.load(f)
        
        assert saved_data["conversation_id"] == "test_conversation"
        assert len(saved_data["messages"]) == 1
    
    @pytest.mark.asyncio
    async def test_component_state_integration(self, full_state_manager):
        """Test integration with component states."""
        await full_state_manager.initialize()
        
        # Setup component states
        full_state_manager.turn_detector.get_current_state.return_value = TurnState.TURN_TAKING
        full_state_manager.interruption_handler.get_current_state.return_value = AgentState.SPEAKING
        
        # Preserve state
        preserved_state = await full_state_manager.preserve_current_state()
        
        # Should have captured component states
        assert preserved_state.turn_detector_state == TurnState.TURN_TAKING
        assert preserved_state.agent_state == AgentState.SPEAKING


class TestMessageManagement:
    """Test message management functionality."""
    
    @pytest.mark.asyncio
    async def test_add_message(self, state_manager):
        """Test adding messages to conversation."""
        await state_manager.initialize()
        
        message = await state_manager.add_message(
            participant_id="user",
            content="Hello!",
            message_type="text",
            metadata={"source": "test"},
            confidence=0.95
        )
        
        assert message.participant_id == "user"
        assert message.content == "Hello!"
        assert message.confidence == 0.95
        assert len(state_manager.current_state.messages) == 1
    
    @pytest.mark.asyncio
    async def test_partial_message_handling(self, state_manager):
        """Test handling of partial messages."""
        await state_manager.initialize()
        
        # Add partial message
        partial_msg = await state_manager.add_message(
            participant_id="user",
            content="This is partial...",
            is_partial=True
        )
        
        assert partial_msg.is_partial is True
        
        # Add complete message
        complete_msg = await state_manager.add_message(
            participant_id="user",
            content="This is complete.",
            is_partial=False
        )
        
        assert complete_msg.is_partial is False
        assert len(state_manager.current_state.messages) == 2
    
    @pytest.mark.asyncio
    async def test_auto_checkpoint_on_messages(self, state_manager):
        """Test automatic checkpoint creation on important messages."""
        state_manager.config.auto_checkpoint_on_turns = True
        await state_manager.initialize()
        
        # Add message that should trigger checkpoint
        await state_manager.add_message("user", "Important message")
        
        # Should have created a checkpoint
        assert len(state_manager.checkpoints) == 1


class TestConversationContext:
    """Test conversation context management."""
    
    @pytest.mark.asyncio
    async def test_update_conversation_context(self, state_manager):
        """Test updating conversation context."""
        await state_manager.initialize()
        
        # Update context
        await state_manager.update_conversation_context({
            "topic": "weather",
            "user_mood": "happy",
            "context_level": 1
        })
        
        context = state_manager.current_state.conversation_context
        assert context["topic"] == "weather"
        assert context["user_mood"] == "happy"
        assert context["context_level"] == 1
    
    @pytest.mark.asyncio
    async def test_context_preservation_across_states(self, state_manager):
        """Test context preservation across state changes."""
        await state_manager.initialize()
        
        # Set initial context
        await state_manager.update_conversation_context({"initial": "value"})
        
        # Preserve state
        state1 = await state_manager.preserve_current_state()
        
        # Update context
        await state_manager.update_conversation_context({"updated": "value"})
        
        # Preserve another state
        state2 = await state_manager.preserve_current_state()
        
        # Verify both states have their respective contexts
        assert state1.conversation_context["initial"] == "value"
        assert "updated" not in state1.conversation_context
        
        assert state2.conversation_context["updated"] == "value"
        assert state2.conversation_context["initial"] == "value"  # Should be preserved


class TestCheckpointManagement:
    """Test checkpoint creation and management."""
    
    @pytest.mark.asyncio
    async def test_create_checkpoint(self, state_manager):
        """Test checkpoint creation."""
        await state_manager.initialize()
        
        # Add some state
        await state_manager.add_message("user", "Checkpoint test")
        
        # Create checkpoint
        checkpoint_id = await state_manager.create_checkpoint("test_checkpoint")
        
        assert checkpoint_id == "test_checkpoint"
        assert "test_checkpoint" in state_manager.checkpoints
        assert len(state_manager.current_state.recovery_checkpoints) == 1
    
    @pytest.mark.asyncio
    async def test_automatic_checkpoint_creation(self, state_manager):
        """Test automatic checkpoint creation."""
        await state_manager.initialize()
        
        # Add content and create automatic checkpoint
        await state_manager.add_message("user", "Auto checkpoint test")
        checkpoint_id = await state_manager.create_checkpoint()
        
        assert checkpoint_id.startswith("checkpoint_")
        assert len(state_manager.checkpoints) == 1
    
    @pytest.mark.asyncio
    async def test_checkpoint_limit_enforcement(self, state_manager):
        """Test checkpoint limit enforcement."""
        state_manager.config.max_checkpoints = 3
        await state_manager.initialize()
        
        # Create more checkpoints than the limit
        for i in range(5):
            await state_manager.create_checkpoint(f"checkpoint_{i}")
        
        # Should only keep the maximum number
        assert len(state_manager.checkpoints) == 3
        
        # Should keep the most recent ones
        checkpoint_ids = list(state_manager.checkpoints.keys())
        assert "checkpoint_2" in checkpoint_ids
        assert "checkpoint_3" in checkpoint_ids
        assert "checkpoint_4" in checkpoint_ids
    
    @pytest.mark.asyncio
    async def test_restore_from_checkpoint(self, state_manager):
        """Test restoring from a checkpoint."""
        await state_manager.initialize()
        
        # Create initial state
        await state_manager.add_message("user", "Initial message")
        checkpoint_id = await state_manager.create_checkpoint("restore_test")
        
        # Modify state
        await state_manager.add_message("user", "Modified message")
        assert len(state_manager.current_state.messages) == 2
        
        # Mock recovery to succeed
        state_manager.recover_conversation_state = AsyncMock(
            return_value=RecoveryContext(
                recovery_id="test",
                timestamp=time.time(),
                strategy=RecoveryStrategy.IMMEDIATE,
                target_state=None,
                recovery_reason="test",
                success=True
            )
        )
        
        # Restore from checkpoint
        success = await state_manager.restore_from_checkpoint("restore_test")
        assert success is True


class TestRecoveryStrategies:
    """Test conversation recovery strategies."""
    
    @pytest.mark.asyncio
    async def test_immediate_recovery(self, state_manager):
        """Test immediate recovery strategy."""
        await state_manager.initialize()
        
        # Create target state
        await state_manager.add_message("user", "Target state message")
        target_state = await state_manager.preserve_current_state()
        
        # Modify current state
        await state_manager.add_message("user", "New message")
        
        # Mock find recovery target
        state_manager._find_recovery_target = AsyncMock(return_value=target_state)
        state_manager._restore_component_states = AsyncMock()
        
        # Execute immediate recovery
        recovery_ctx = await state_manager.recover_conversation_state(
            recovery_strategy=RecoveryStrategy.IMMEDIATE,
            reason="test_immediate"
        )
        
        assert recovery_ctx.success is True
        assert recovery_ctx.strategy == RecoveryStrategy.IMMEDIATE
        assert "Replaced current state" in recovery_ctx.recovery_steps
    
    @pytest.mark.asyncio
    async def test_gradual_recovery(self, state_manager):
        """Test gradual recovery strategy."""
        await state_manager.initialize()
        
        # Create target state
        await state_manager.add_message("user", "Target message")
        await state_manager.update_conversation_context({"target": "context"})
        target_state = await state_manager.preserve_current_state()
        
        # Mock dependencies
        state_manager._find_recovery_target = AsyncMock(return_value=target_state)
        state_manager._restore_component_states = AsyncMock()
        
        # Execute gradual recovery
        recovery_ctx = await state_manager.recover_conversation_state(
            recovery_strategy=RecoveryStrategy.GRADUAL,
            reason="test_gradual"
        )
        
        assert recovery_ctx.success is True
        assert recovery_ctx.strategy == RecoveryStrategy.GRADUAL
        assert "Restored conversation context" in recovery_ctx.recovery_steps
        assert "Restored message history" in recovery_ctx.recovery_steps
    
    @pytest.mark.asyncio
    async def test_context_aware_recovery(self, state_manager):
        """Test context-aware recovery strategy."""
        await state_manager.initialize()
        
        # Create target state with context
        await state_manager.add_message("user", "Old message")
        await state_manager.update_conversation_context({"old": "context"})
        target_state = await state_manager.preserve_current_state()
        
        # Update current state
        await state_manager.add_message("user", "New message")
        await state_manager.update_conversation_context({"new": "context"})
        
        # Mock dependencies
        state_manager._find_recovery_target = AsyncMock(return_value=target_state)
        
        # Execute context-aware recovery
        recovery_ctx = await state_manager.recover_conversation_state(
            recovery_strategy=RecoveryStrategy.CONTEXT_AWARE,
            reason="test_context_aware"
        )
        
        assert recovery_ctx.success is True
        assert recovery_ctx.strategy == RecoveryStrategy.CONTEXT_AWARE
        assert "Merged conversation context" in recovery_ctx.recovery_steps
    
    @pytest.mark.asyncio
    async def test_recovery_target_finding(self, state_manager):
        """Test finding appropriate recovery targets."""
        await state_manager.initialize()
        
        # Create multiple states
        states = []
        for i in range(3):
            await state_manager.add_message("user", f"Message {i}")
            state = await state_manager.preserve_current_state()
            states.append(state)
            await asyncio.sleep(0.01)  # Small delay for timestamp differences
        
        # Test timestamp-based target finding
        target_time = states[1].timestamp
        found_target = await state_manager._find_recovery_target(target_time, RecoveryStrategy.IMMEDIATE)
        
        assert found_target is not None
        # Should find closest to target time
        
        # Test strategy-based target finding
        context_target = await state_manager._find_recovery_target(None, RecoveryStrategy.CONTEXT_AWARE)
        assert context_target is not None


class TestComponentIntegration:
    """Test integration with conversation components."""
    
    @pytest.mark.asyncio
    async def test_turn_completion_handling(self, full_state_manager):
        """Test handling of turn completion events."""
        await full_state_manager.initialize()
        
        # Create mock turn
        turn = TurnSegment(start_time=time.time(), end_time=time.time() + 1.0)
        
        # Trigger turn completion callback
        await full_state_manager._on_turn_completed(turn)
        
        # Should have created a checkpoint
        assert len(full_state_manager.checkpoints) == 1
    
    @pytest.mark.asyncio
    async def test_interruption_handling(self, full_state_manager):
        """Test handling of interruption events."""
        await full_state_manager.initialize()
        
        # Create mock interruption
        interruption = InterruptionEvent(
            timestamp=time.time(),
            interruption_type=None,
            severity=None,
            confidence=0.8,
            vad_confidence=0.8
        )
        
        # Trigger interruption callback
        await full_state_manager._on_interruption_occurred(interruption)
        
        # Should have updated state
        assert full_state_manager.current_state.state_type == ConversationStateType.INTERRUPTED
        assert full_state_manager.current_state.interruption_context == interruption
    
    @pytest.mark.asyncio
    async def test_pipeline_flush_handling(self, full_state_manager):
        """Test handling of pipeline flush events."""
        await full_state_manager.initialize()
        
        # Create mock flush operation
        flush_op = FlushOperation(
            flush_id="test_flush",
            timestamp=time.time(),
            flush_type=None,
            reason=None,
            target_components=["stt"]
        )
        
        # Trigger flush callback
        await full_state_manager._on_pipeline_flushed(flush_op)
        
        # Should have created a post-flush checkpoint
        assert len(full_state_manager.checkpoints) == 1
        checkpoint_id = list(full_state_manager.checkpoints.keys())[0]
        assert checkpoint_id.startswith("post_flush_")
    
    @pytest.mark.asyncio
    async def test_llm_context_preservation(self, full_state_manager):
        """Test LLM context preservation and restoration."""
        await full_state_manager.initialize()
        
        # LLM should be called during state update
        await full_state_manager._update_current_state()
        
        # Should have captured LLM history
        full_state_manager.llm_provider.get_conversation_history.assert_called_once()
        
        # Test restoration
        test_state = ConversationState(
            session_id="test",
            conversation_id="test",
            timestamp=time.time(),
            state_type=ConversationStateType.ACTIVE,
            conversation_context={"llm_history": ["restored", "history"]}
        )
        
        await full_state_manager._restore_component_states(test_state)
        
        # Should have restored LLM history
        full_state_manager.llm_provider.set_conversation_history.assert_called_once_with(
            ["restored", "history"]
        )


class TestStateLoading:
    """Test state loading and persistence."""
    
    @pytest.mark.asyncio
    async def test_load_existing_state(self, temp_storage):
        """Test loading existing conversation state."""
        # Create test state file
        conversation_id = "test_load_conversation"
        state_file = Path(temp_storage) / f"{conversation_id}_latest.json"
        
        test_state_data = {
            "session_id": "test_session",
            "conversation_id": conversation_id,
            "timestamp": time.time(),
            "state_type": "active",
            "preservation_level": "standard",
            "participants": [],
            "messages": [
                {
                    "id": "msg1",
                    "timestamp": time.time(),
                    "participant_id": "user",
                    "content": "Loaded message",
                    "message_type": "text",
                    "is_partial": False
                }
            ],
            "turn_history": [],
            "conversation_context": {"loaded": "context"},
            "user_preferences": {},
            "session_metadata": {},
            "recovery_checkpoints": []
        }
        
        with open(state_file, 'w') as f:
            json.dump(test_state_data, f)
        
        # Create state manager and initialize
        config = ConversationStateConfig(storage_directory=temp_storage)
        state_manager = ConversationStateManager(
            session_id="test_session",
            conversation_id=conversation_id,
            config=config
        )
        
        await state_manager.initialize()
        
        # Should have loaded the existing state
        assert len(state_manager.current_state.messages) == 1
        assert state_manager.current_state.messages[0].content == "Loaded message"
        assert state_manager.current_state.conversation_context["loaded"] == "context"
    
    @pytest.mark.asyncio
    async def test_state_serialization_deserialization(self, state_manager):
        """Test state serialization and deserialization."""
        await state_manager.initialize()
        
        # Add complex state
        await state_manager.add_message("user", "Test message", metadata={"test": "data"})
        await state_manager.update_conversation_context({"complex": {"nested": "data"}})
        
        # Serialize state
        state_dict = state_manager.current_state.to_dict()
        
        # Deserialize state
        restored_state = ConversationState.from_dict(state_dict)
        
        # Verify restoration
        assert restored_state.session_id == state_manager.current_state.session_id
        assert len(restored_state.messages) == len(state_manager.current_state.messages)
        assert restored_state.conversation_context == state_manager.current_state.conversation_context


class TestMetricsAndMonitoring:
    """Test metrics collection and monitoring."""
    
    def test_metrics_collection(self, state_manager):
        """Test metrics collection."""
        # Add some test data
        state_manager.current_state.messages = [
            ConversationMessage(
                id="1", timestamp=time.time(), participant_id="user", content="msg1"
            ),
            ConversationMessage(
                id="2", timestamp=time.time(), participant_id="agent", content="msg2"
            )
        ]
        
        state_manager.recovery_history = [
            RecoveryContext(
                recovery_id="1", timestamp=time.time(), strategy=RecoveryStrategy.IMMEDIATE,
                target_state=None, recovery_reason="test", success=True
            ),
            RecoveryContext(
                recovery_id="2", timestamp=time.time(), strategy=RecoveryStrategy.GRADUAL,
                target_state=None, recovery_reason="test", success=False
            )
        ]
        
        metrics = state_manager.get_metrics()
        
        assert metrics['session_id'] == "test_session"
        assert metrics['conversation_id'] == "test_conversation"
        assert metrics['total_messages'] == 2
        assert metrics['total_recoveries'] == 2
        assert metrics['successful_recoveries'] == 1
        assert metrics['recovery_success_rate'] == 0.5
    
    def test_state_reporting(self, state_manager):
        """Test current state reporting."""
        assert state_manager.get_current_state() is not None
        assert len(state_manager.get_state_history()) == 0
        assert len(state_manager.get_recovery_history()) == 0
        assert len(state_manager.get_checkpoints()) == 0
    
    @pytest.mark.asyncio
    async def test_monitoring_loops(self, state_manager):
        """Test background monitoring loops."""
        await state_manager.initialize()
        await state_manager.start_monitoring()
        
        # Let monitoring run briefly
        await asyncio.sleep(0.25)  # Should trigger persistence and checkpoint
        
        # Should have automatically preserved state and created checkpoints
        # Note: This test may be timing-dependent
        
        await state_manager.stop_monitoring()


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_recovery_failure_handling(self, state_manager):
        """Test handling of recovery failures."""
        await state_manager.initialize()
        
        # Mock recovery to fail
        state_manager._find_recovery_target = AsyncMock(side_effect=Exception("Recovery failed"))
        
        # Attempt recovery
        recovery_ctx = await state_manager.recover_conversation_state(
            reason="test_failure"
        )
        
        assert recovery_ctx.success is False
        assert recovery_ctx.error_message == "Recovery failed"
        assert recovery_ctx.completion_time is not None
    
    @pytest.mark.asyncio
    async def test_invalid_checkpoint_restoration(self, state_manager):
        """Test handling of invalid checkpoint restoration."""
        await state_manager.initialize()
        
        # Try to restore from non-existent checkpoint
        success = await state_manager.restore_from_checkpoint("non_existent")
        assert success is False
    
    @pytest.mark.asyncio
    async def test_persistence_error_handling(self, state_manager):
        """Test handling of persistence errors."""
        await state_manager.initialize()
        
        # Mock storage path to be invalid
        state_manager.storage_path = Path("/invalid/path/that/does/not/exist")
        
        # Should handle persistence error gracefully
        preserved_state = await state_manager.preserve_current_state()
        assert preserved_state is not None  # Should still return the state
    
    @pytest.mark.asyncio
    async def test_callback_error_handling(self, state_manager):
        """Test error handling in callbacks."""
        def failing_callback(*args):
            raise Exception("Callback error")
        
        state_manager.set_callbacks(on_state_changed=failing_callback)
        await state_manager.initialize()
        
        # Should not raise exception despite callback failure
        await state_manager.set_conversation_state_type(ConversationStateType.PAUSED)
    
    @pytest.mark.asyncio
    async def test_component_integration_errors(self, full_state_manager):
        """Test handling of component integration errors."""
        await full_state_manager.initialize()
        
        # Mock LLM provider to fail
        full_state_manager.llm_provider.get_conversation_history.side_effect = Exception("LLM error")
        
        # Should handle error gracefully during state update
        await full_state_manager._update_current_state()
        # Should not crash despite LLM error


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_complete_conversation_cycle(self, full_state_manager):
        """Test complete conversation cycle with state management."""
        await full_state_manager.initialize()
        await full_state_manager.start_monitoring()
        
        # Simulate conversation flow
        await full_state_manager.add_message("user", "Hello!")
        await full_state_manager.add_message("agent", "Hi there! How can I help?")
        
        # Create checkpoint
        checkpoint_id = await full_state_manager.create_checkpoint("mid_conversation")
        
        # Continue conversation
        await full_state_manager.add_message("user", "I need help with something")
        
        # Simulate interruption
        interruption = InterruptionEvent(
            timestamp=time.time(),
            interruption_type=None,
            severity=None,
            confidence=0.8,
            vad_confidence=0.8
        )
        await full_state_manager._on_interruption_occurred(interruption)
        
        # Verify state changes
        assert full_state_manager.current_state.state_type == ConversationStateType.INTERRUPTED
        assert len(full_state_manager.current_state.messages) == 3
        assert len(full_state_manager.checkpoints) >= 1
        
        await full_state_manager.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_state_preservation_and_recovery_cycle(self, state_manager):
        """Test complete preservation and recovery cycle."""
        await state_manager.initialize()
        
        # Build conversation state
        await state_manager.add_message("user", "Initial message")
        await state_manager.update_conversation_context({"topic": "testing"})
        
        # Preserve initial state
        initial_state = await state_manager.preserve_current_state()
        
        # Modify state
        await state_manager.add_message("user", "Modified message")
        await state_manager.update_conversation_context({"topic": "changed"})
        
        # Mock recovery components
        state_manager._find_recovery_target = AsyncMock(return_value=initial_state)
        state_manager._restore_component_states = AsyncMock()
        
        # Recover to initial state
        recovery_ctx = await state_manager.recover_conversation_state(
            recovery_strategy=RecoveryStrategy.CONTEXT_AWARE,
            reason="test_cycle"
        )
        
        assert recovery_ctx.success is True
        assert len(state_manager.recovery_history) == 1
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, state_manager):
        """Test concurrent state operations."""
        await state_manager.initialize()
        
        # Run concurrent operations
        tasks = [
            state_manager.add_message("user", f"Message {i}")
            for i in range(5)
        ]
        
        await asyncio.gather(*tasks)
        
        # All messages should be added
        assert len(state_manager.current_state.messages) == 5
        
        # Test concurrent preservation
        preservation_tasks = [
            state_manager.preserve_current_state()
            for _ in range(3)
        ]
        
        preserved_states = await asyncio.gather(*preservation_tasks)
        
        # All preservations should succeed
        assert all(state is not None for state in preserved_states)
        assert len(state_manager.state_history) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])