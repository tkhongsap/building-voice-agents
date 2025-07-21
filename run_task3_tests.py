#!/usr/bin/env python3
"""
Simple test runner for Task 3.0 Advanced Conversation Management System
Runs tests without pytest dependency
"""

import sys
import os
import asyncio
import traceback
from typing import List, Tuple

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def run_test_function(test_func, test_name: str) -> Tuple[bool, str]:
    """Run a single test function and return success status and error message."""
    try:
        if asyncio.iscoroutinefunction(test_func):
            asyncio.run(test_func())
        else:
            test_func()
        return True, ""
    except Exception as e:
        error_msg = f"{test_name}: {str(e)}\n{traceback.format_exc()}"
        return False, error_msg

def test_context_manager_basic():
    """Test basic context manager functionality."""
    from conversation.context_manager import (
        ConversationContextManager, ContextManagerConfig, ContextItem,
        ContextType, ContextScope, ContextRelevance
    )
    
    # Test configuration
    config = ContextManagerConfig(
        immediate_window_size=5,
        enable_detailed_logging=False,
        enable_async_processing=False
    )
    assert config.immediate_window_size == 5
    
    # Test context manager creation
    manager = ConversationContextManager(
        conversation_id="test_conversation",
        config=config
    )
    assert manager.conversation_id == "test_conversation"
    
    # Test context item creation
    item = ContextItem(
        id="test_item",
        content="test content",
        context_type=ContextType.MESSAGE,
        scope=ContextScope.SHORT_TERM,
        relevance=ContextRelevance.HIGH,
        timestamp=1234567890.0,
        source="test"
    )
    assert item.id == "test_item"
    assert item.content == "test content"
    assert item.context_type == ContextType.MESSAGE
    
    print("✓ Context manager basic tests passed")

def test_turn_detector_basic():
    """Test basic turn detector functionality."""
    from conversation.turn_detector import (
        TurnDetector, TurnDetectionConfig, TurnDetectionState,
        TurnSegment, TurnConfidence
    )
    
    # Test configuration
    config = TurnDetectionConfig(
        vad_threshold=0.5,
        silence_threshold=1.0,
        enable_detailed_logging=False
    )
    assert config.vad_threshold == 0.5
    
    # Test turn detector creation
    detector = TurnDetector(config=config)
    assert detector.config.vad_threshold == 0.5
    assert detector.current_state == TurnDetectionState.LISTENING
    
    # Test turn segment
    segment = TurnSegment(
        segment_id="test_segment",
        start_time=1234567890.0,
        end_time=1234567895.0,
        confidence=TurnConfidence.HIGH,
        participant_id="user"
    )
    assert segment.segment_id == "test_segment"
    assert segment.duration == 5.0
    
    print("✓ Turn detector basic tests passed")

def test_interruption_handler_basic():
    """Test basic interruption handler functionality."""
    from conversation.interruption_handler import (
        InterruptionHandler, InterruptionHandlerConfig, InterruptionEvent,
        InterruptionType, InterruptionSeverity
    )
    
    # Test configuration
    config = InterruptionHandlerConfig(
        detection_threshold=0.7,
        enable_detailed_logging=False
    )
    assert config.detection_threshold == 0.7
    
    # Test handler creation
    handler = InterruptionHandler(config=config)
    assert handler.config.detection_threshold == 0.7
    
    # Test interruption event
    event = InterruptionEvent(
        event_id="test_event",
        timestamp=1234567890.0,
        interruption_type=InterruptionType.BARGE_IN,
        severity=InterruptionSeverity.HIGH,
        participant_id="user"
    )
    assert event.event_id == "test_event"
    assert event.interruption_type == InterruptionType.BARGE_IN
    
    print("✓ Interruption handler basic tests passed")

def test_pipeline_flush_handler_basic():
    """Test basic pipeline flush handler functionality."""
    from conversation.pipeline_flush_handler import (
        PipelineFlushHandler, PipelineFlushConfig, FlushOperation,
        FlushType, FlushReason, FlushStatus
    )
    
    # Test configuration
    config = PipelineFlushConfig(
        default_flush_timeout=5.0,
        enable_detailed_logging=False
    )
    assert config.default_flush_timeout == 5.0
    
    # Test handler creation
    handler = PipelineFlushHandler(config=config)
    assert handler.config.default_flush_timeout == 5.0
    
    # Test flush operation
    operation = FlushOperation(
        operation_id="test_operation",
        flush_type=FlushType.SOFT_FLUSH,
        reason=FlushReason.INTERRUPTION,
        status=FlushStatus.PENDING,
        timestamp=1234567890.0
    )
    assert operation.operation_id == "test_operation"
    assert operation.flush_type == FlushType.SOFT_FLUSH
    
    print("✓ Pipeline flush handler basic tests passed")

def test_conversation_state_manager_basic():
    """Test basic conversation state manager functionality."""
    from conversation.conversation_state_manager import (
        ConversationStateManager, ConversationStateConfig, ConversationState,
        ConversationParticipant, ConversationMessage, ParticipantRole
    )
    
    # Test configuration
    config = ConversationStateConfig(
        max_history_size=100,
        enable_detailed_logging=False
    )
    assert config.max_history_size == 100
    
    # Test state manager creation
    manager = ConversationStateManager(
        conversation_id="test_conversation",
        config=config
    )
    assert manager.conversation_id == "test_conversation"
    
    # Test participant
    participant = ConversationParticipant(
        participant_id="user_1",
        name="Test User",
        role=ParticipantRole.USER
    )
    assert participant.participant_id == "user_1"
    assert participant.role == ParticipantRole.USER
    
    # Test message
    message = ConversationMessage(
        id="msg_1",
        timestamp=1234567890.0,
        participant_id="user_1",
        content="Hello, world!"
    )
    assert message.id == "msg_1"
    assert message.content == "Hello, world!"
    
    print("✓ Conversation state manager basic tests passed")

async def test_context_manager_async():
    """Test async context manager functionality."""
    from conversation.context_manager import (
        ConversationContextManager, ContextManagerConfig
    )
    
    config = ContextManagerConfig(
        enable_async_processing=False,  # Disable for simpler testing
        enable_detailed_logging=False
    )
    
    manager = ConversationContextManager(
        conversation_id="test_async",
        config=config
    )
    
    # Test initialization
    await manager.initialize()
    assert manager.conversation_id == "test_async"
    
    # Test adding context item
    item = await manager.add_context_item(
        content="test content",
        context_type=manager.ContextType.MESSAGE,
        scope=manager.ContextScope.SHORT_TERM,
        relevance=manager.ContextRelevance.HIGH,
        source="test"
    )
    assert item.content == "test content"
    
    # Test getting relevant context
    context_items = await manager.get_relevant_context()
    assert len(context_items) >= 1
    
    # Test cleanup
    await manager.cleanup()
    
    print("✓ Context manager async tests passed")

def main():
    """Run all Task 3.0 tests."""
    print("=" * 60)
    print("Running Task 3.0 Advanced Conversation Management System Tests")
    print("=" * 60)
    
    test_functions = [
        (test_context_manager_basic, "Context Manager Basic"),
        (test_turn_detector_basic, "Turn Detector Basic"),
        (test_interruption_handler_basic, "Interruption Handler Basic"),
        (test_pipeline_flush_handler_basic, "Pipeline Flush Handler Basic"),
        (test_conversation_state_manager_basic, "Conversation State Manager Basic"),
        (test_context_manager_async, "Context Manager Async"),
    ]
    
    passed = 0
    failed = 0
    errors = []
    
    for test_func, test_name in test_functions:
        print(f"\nRunning {test_name}...")
        success, error_msg = run_test_function(test_func, test_name)
        
        if success:
            passed += 1
            print(f"✓ {test_name} PASSED")
        else:
            failed += 1
            print(f"✗ {test_name} FAILED")
            errors.append(error_msg)
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Total tests: {passed + failed}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if errors:
        print("\nERROR DETAILS:")
        print("-" * 40)
        for error in errors:
            print(error)
            print("-" * 40)
    
    print("\n✓ Task 3.0 test execution completed")
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)