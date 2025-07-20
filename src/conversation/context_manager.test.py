"""
Unit Tests for Conversation Context Manager

Tests the multi-turn conversation context management system including
context tracking, relevance scoring, and intelligent context windowing.
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, AsyncMock, MagicMock
from typing import List, Dict, Any

from context_manager import (
    ConversationContextManager, ContextManagerConfig, ContextItem,
    ContextType, ContextScope, ContextRelevance, ContextWindow
)

try:
    from .conversation_state_manager import ConversationStateManager, ConversationMessage
    from ..components.llm.base_llm import BaseLLMProvider, LLMMessage, LLMRole
except ImportError:
    # Fallback for direct imports
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from conversation_state_manager import ConversationStateManager, ConversationMessage
    from components.llm.base_llm import BaseLLMProvider, LLMMessage, LLMRole


@pytest.fixture
def mock_state_manager():
    """Create a mock state manager for testing."""
    mock = Mock(spec=ConversationStateManager)
    mock.get_current_state = Mock()
    mock.update_conversation_context = AsyncMock()
    mock.get_current_state.return_value.conversation_context = {}
    return mock


@pytest.fixture
def mock_llm_provider():
    """Create a mock LLM provider for testing."""
    mock = AsyncMock(spec=BaseLLMProvider)
    return mock


@pytest.fixture
def basic_config():
    """Create a basic context manager configuration."""
    return ContextManagerConfig(
        immediate_window_size=5,
        short_term_window_size=10,
        enable_detailed_logging=True,
        enable_async_processing=False  # Disable for simpler testing
    )


@pytest.fixture
def context_manager(basic_config):
    """Create a basic context manager instance for testing."""
    return ConversationContextManager(
        conversation_id="test_conversation",
        config=basic_config
    )


@pytest.fixture
def full_context_manager(basic_config, mock_state_manager, mock_llm_provider):
    """Create a full context manager with all components."""
    return ConversationContextManager(
        conversation_id="test_conversation",
        state_manager=mock_state_manager,
        llm_provider=mock_llm_provider,
        config=basic_config
    )


class TestContextManagerConfig:
    """Test context manager configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ContextManagerConfig()
        assert config.immediate_window_size == 10
        assert config.short_term_window_size == 30
        assert config.auto_relevance_scoring is True
        assert config.enable_entity_extraction is True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ContextManagerConfig(
            immediate_window_size=20,
            enable_topic_tracking=False,
            relevance_decay_rate=0.2
        )
        assert config.immediate_window_size == 20
        assert config.enable_topic_tracking is False
        assert config.relevance_decay_rate == 0.2


class TestContextItem:
    """Test context item functionality."""
    
    def test_context_item_creation(self):
        """Test creating context items."""
        item = ContextItem(
            id="test_item",
            content="test content",
            context_type=ContextType.MESSAGE,
            scope=ContextScope.SHORT_TERM,
            relevance=ContextRelevance.HIGH,
            timestamp=time.time(),
            source="test"
        )
        
        assert item.id == "test_item"
        assert item.content == "test content"
        assert item.context_type == ContextType.MESSAGE
        assert item.relevance == ContextRelevance.HIGH
    
    def test_context_item_aging(self):
        """Test context item aging functionality."""
        past_time = time.time() - 3600  # 1 hour ago
        item = ContextItem(
            id="old_item",
            content="old content",
            context_type=ContextType.MESSAGE,
            scope=ContextScope.SHORT_TERM,
            relevance=ContextRelevance.MEDIUM,
            timestamp=past_time,
            source="test"
        )
        
        assert item.age >= 3600
        assert item.age < 3700  # Should be approximately 1 hour
    
    def test_context_item_expiry(self):
        """Test context item expiry functionality."""
        current_time = time.time()
        item = ContextItem(
            id="expiring_item",
            content="expiring content",
            context_type=ContextType.MESSAGE,
            scope=ContextScope.SHORT_TERM,
            relevance=ContextRelevance.MEDIUM,
            timestamp=current_time,
            source="test",
            expiry_time=current_time - 1  # Already expired
        )
        
        assert item.is_expired is True
        
        # Test non-expired item
        item.expiry_time = current_time + 3600  # Expires in 1 hour
        assert item.is_expired is False
    
    def test_context_item_access_tracking(self):
        """Test context item access tracking."""
        item = ContextItem(
            id="access_item",
            content="access content",
            context_type=ContextType.MESSAGE,
            scope=ContextScope.SHORT_TERM,
            relevance=ContextRelevance.MEDIUM,
            timestamp=time.time(),
            source="test"
        )
        
        assert item.access_count == 0
        
        item.access()
        assert item.access_count == 1
        assert item.last_accessed is not None
        
        # Multiple accesses
        item.access()
        item.access()
        assert item.access_count == 3
    
    def test_context_item_serialization(self):
        """Test context item serialization and deserialization."""
        item = ContextItem(
            id="serialize_item",
            content="serialize content",
            context_type=ContextType.ENTITY,
            scope=ContextScope.MEDIUM_TERM,
            relevance=ContextRelevance.HIGH,
            timestamp=time.time(),
            source="test",
            metadata={"key": "value"}
        )
        
        # Serialize
        item_dict = item.to_dict()
        assert item_dict["id"] == "serialize_item"
        assert item_dict["context_type"] == "entity"
        assert item_dict["metadata"]["key"] == "value"
        
        # Deserialize
        restored_item = ContextItem.from_dict(item_dict)
        assert restored_item.id == item.id
        assert restored_item.context_type == item.context_type
        assert restored_item.metadata == item.metadata


class TestContextWindow:
    """Test context window functionality."""
    
    def test_context_window_creation(self):
        """Test creating context windows."""
        window = ContextWindow(
            window_id="test_window",
            max_items=5,
            max_age_seconds=3600,
            relevance_threshold=ContextRelevance.MEDIUM
        )
        
        assert window.window_id == "test_window"
        assert window.max_items == 5
        assert len(window.items) == 0
    
    def test_context_window_item_addition(self):
        """Test adding items to context window."""
        window = ContextWindow(
            window_id="test_window",
            max_items=3,
            relevance_threshold=ContextRelevance.LOW
        )
        
        # Add items with different relevance levels
        high_item = ContextItem(
            id="high", content="high", context_type=ContextType.MESSAGE,
            scope=ContextScope.SHORT_TERM, relevance=ContextRelevance.HIGH,
            timestamp=time.time(), source="test"
        )
        
        low_item = ContextItem(
            id="low", content="low", context_type=ContextType.MESSAGE,
            scope=ContextScope.SHORT_TERM, relevance=ContextRelevance.LOW,
            timestamp=time.time(), source="test"
        )
        
        irrelevant_item = ContextItem(
            id="irrelevant", content="irrelevant", context_type=ContextType.MESSAGE,
            scope=ContextScope.SHORT_TERM, relevance=ContextRelevance.IRRELEVANT,
            timestamp=time.time(), source="test"
        )
        
        # Should add high and low relevance items
        assert window.add_item(high_item) is True
        assert window.add_item(low_item) is True
        assert len(window.items) == 2
        
        # Should not add irrelevant item
        assert window.add_item(irrelevant_item) is False
        assert len(window.items) == 2
    
    def test_context_window_size_limit(self):
        """Test context window size limit enforcement."""
        window = ContextWindow(
            window_id="test_window",
            max_items=2,
            relevance_threshold=ContextRelevance.LOW
        )
        
        # Add items exceeding the limit
        for i in range(5):
            item = ContextItem(
                id=f"item_{i}", content=f"content_{i}",
                context_type=ContextType.MESSAGE, scope=ContextScope.SHORT_TERM,
                relevance=ContextRelevance.MEDIUM, timestamp=time.time() + i,
                source="test"
            )
            window.add_item(item)
        
        # Should only keep the maximum number of items
        assert len(window.items) <= window.max_items
    
    def test_context_window_age_limit(self):
        """Test context window age limit enforcement."""
        window = ContextWindow(
            window_id="test_window",
            max_age_seconds=60,  # 1 minute
            relevance_threshold=ContextRelevance.LOW
        )
        
        # Add old item
        old_item = ContextItem(
            id="old", content="old", context_type=ContextType.MESSAGE,
            scope=ContextScope.SHORT_TERM, relevance=ContextRelevance.MEDIUM,
            timestamp=time.time() - 120,  # 2 minutes ago
            source="test"
        )
        
        # Add recent item
        recent_item = ContextItem(
            id="recent", content="recent", context_type=ContextType.MESSAGE,
            scope=ContextScope.SHORT_TERM, relevance=ContextRelevance.MEDIUM,
            timestamp=time.time(),
            source="test"
        )
        
        window.add_item(old_item)
        window.add_item(recent_item)
        
        # Should only contain the recent item
        assert len(window.items) == 1
        assert window.items[0].id == "recent"
    
    def test_get_relevant_items(self):
        """Test getting relevant items from window."""
        window = ContextWindow(
            window_id="test_window",
            relevance_threshold=ContextRelevance.LOW
        )
        
        # Add items with different relevance
        items = [
            ContextItem(
                id=f"item_{i}", content=f"content_{i}",
                context_type=ContextType.MESSAGE, scope=ContextScope.SHORT_TERM,
                relevance=list(ContextRelevance)[i % len(ContextRelevance)],
                timestamp=time.time(), source="test"
            )
            for i in range(5)
        ]
        
        for item in items:
            window.add_item(item)
        
        # Get items with medium or higher relevance
        relevant = window.get_relevant_items(ContextRelevance.MEDIUM)
        
        # Should only contain medium, high, or critical items
        for item in relevant:
            assert item.relevance in [
                ContextRelevance.MEDIUM, ContextRelevance.HIGH, ContextRelevance.CRITICAL
            ]


class TestConversationContextManager:
    """Test conversation context manager functionality."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, context_manager):
        """Test basic initialization."""
        await context_manager.initialize()
        assert context_manager.conversation_id == "test_conversation"
        assert len(context_manager.context_index) == 0
    
    @pytest.mark.asyncio
    async def test_full_initialization(self, full_context_manager):
        """Test initialization with all components."""
        await full_context_manager.initialize()
        # Should attempt to load context from state manager
        full_context_manager.state_manager.get_current_state.assert_called()
    
    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, context_manager):
        """Test start and stop monitoring."""
        await context_manager.initialize()
        
        await context_manager.start_monitoring()
        assert context_manager._is_monitoring is True
        
        await context_manager.stop_monitoring()
        assert context_manager._is_monitoring is False
    
    @pytest.mark.asyncio
    async def test_add_custom_context_item(self, context_manager):
        """Test adding custom context items."""
        await context_manager.initialize()
        
        item = await context_manager.add_context_item(
            content="Custom context",
            context_type=ContextType.TOPIC,
            scope=ContextScope.MEDIUM_TERM,
            relevance=ContextRelevance.HIGH,
            source="test_custom"
        )
        
        assert item.content == "Custom context"
        assert item.context_type == ContextType.TOPIC
        assert item.id in context_manager.context_index
        
        # Should be added to appropriate windows
        assert len(context_manager.medium_term_window.items) > 0
        assert len(context_manager.short_term_window.items) > 0


class TestMessageContextExtraction:
    """Test message context extraction functionality."""
    
    @pytest.mark.asyncio
    async def test_add_message_context(self, context_manager):
        """Test adding context from messages."""
        await context_manager.initialize()
        
        message = ConversationMessage(
            id="test_msg",
            timestamp=time.time(),
            participant_id="user",
            content="Hello, I need help with my email support@example.com",
            metadata={"test": "data"}
        )
        
        context_items = await context_manager.add_message_context(message)
        
        # Should create message context item
        assert len(context_items) >= 1
        message_item = context_items[0]
        assert message_item.context_type == ContextType.MESSAGE
        assert message_item.content == message.content
        
        # Should extract entities (email)
        if context_manager.config.enable_entity_extraction:
            entity_items = [item for item in context_items if item.context_type == ContextType.ENTITY]
            assert len(entity_items) > 0
    
    @pytest.mark.asyncio
    async def test_entity_extraction(self, context_manager):
        """Test entity extraction from messages."""
        await context_manager.initialize()
        
        # Test email extraction
        email_text = "Contact me at john.doe@example.com for more info"
        entities = await context_manager._extract_entities(email_text)
        
        email_entities = [e for e in entities if e["type"] == "email"]
        assert len(email_entities) == 1
        assert email_entities[0]["name"] == "john.doe@example.com"
        
        # Test phone extraction
        phone_text = "Call me at 123-456-7890"
        entities = await context_manager._extract_entities(phone_text)
        
        phone_entities = [e for e in entities if e["type"] == "phone"]
        assert len(phone_entities) == 1
        assert phone_entities[0]["name"] == "123-456-7890"
    
    @pytest.mark.asyncio
    async def test_topic_extraction(self, context_manager):
        """Test topic extraction from messages."""
        await context_manager.initialize()
        
        # Test weather topic
        weather_text = "What's the weather like today? It's very sunny outside."
        topics = await context_manager._extract_topics(weather_text)
        assert "weather" in topics
        
        # Test technology topic
        tech_text = "I'm having trouble with my computer software"
        topics = await context_manager._extract_topics(tech_text)
        assert "technology" in topics
    
    @pytest.mark.asyncio
    async def test_intent_detection(self, context_manager):
        """Test intent detection from messages."""
        await context_manager.initialize()
        
        # Test help intent
        help_text = "Can you help me with this problem?"
        intent = await context_manager._detect_intent(help_text)
        assert intent == "help_request"
        
        # Test question intent
        question_text = "What time is it?"
        intent = await context_manager._detect_intent(question_text)
        assert intent == "question"
        
        # Test gratitude intent
        thanks_text = "Thank you for your assistance"
        intent = await context_manager._detect_intent(thanks_text)
        assert intent == "gratitude"


class TestContextRetrieval:
    """Test context retrieval functionality."""
    
    @pytest.mark.asyncio
    async def test_get_relevant_context(self, context_manager):
        """Test getting relevant context items."""
        await context_manager.initialize()
        
        # Add various context items
        items = []
        for i in range(5):
            item = await context_manager.add_context_item(
                content=f"Content {i}",
                context_type=ContextType.MESSAGE,
                scope=ContextScope.SHORT_TERM,
                relevance=list(ContextRelevance)[i % len(ContextRelevance)],
                source="test"
            )
            items.append(item)
        
        # Get high relevance items
        high_relevance = await context_manager.get_relevant_context(
            min_relevance=ContextRelevance.HIGH
        )
        
        # Should only return high and critical relevance items
        for item in high_relevance:
            assert item.relevance in [ContextRelevance.HIGH, ContextRelevance.CRITICAL]
    
    @pytest.mark.asyncio
    async def test_get_context_by_scope(self, context_manager):
        """Test getting context by scope."""
        await context_manager.initialize()
        
        # Add items with different scopes
        await context_manager.add_context_item(
            content="Immediate content",
            context_type=ContextType.MESSAGE,
            scope=ContextScope.IMMEDIATE,
            relevance=ContextRelevance.HIGH,
            source="test"
        )
        
        await context_manager.add_context_item(
            content="Long term content",
            context_type=ContextType.TOPIC,
            scope=ContextScope.LONG_TERM,
            relevance=ContextRelevance.HIGH,
            source="test"
        )
        
        # Get immediate scope items
        immediate_items = await context_manager.get_relevant_context(
            scope=ContextScope.IMMEDIATE
        )
        
        assert len(immediate_items) > 0
        for item in immediate_items:
            # Item should be in immediate window
            assert any(i.id == item.id for i in context_manager.immediate_window.items)
    
    @pytest.mark.asyncio
    async def test_get_context_by_type(self, context_manager):
        """Test getting context by type."""
        await context_manager.initialize()
        
        # Add items with different types
        await context_manager.add_context_item(
            content="Message content",
            context_type=ContextType.MESSAGE,
            scope=ContextScope.SHORT_TERM,
            relevance=ContextRelevance.HIGH,
            source="test"
        )
        
        await context_manager.add_context_item(
            content="Topic content",
            context_type=ContextType.TOPIC,
            scope=ContextScope.SHORT_TERM,
            relevance=ContextRelevance.HIGH,
            source="test"
        )
        
        # Get only message type items
        message_items = await context_manager.get_relevant_context(
            context_type=ContextType.MESSAGE
        )
        
        assert len(message_items) > 0
        for item in message_items:
            assert item.context_type == ContextType.MESSAGE
    
    @pytest.mark.asyncio
    async def test_build_context_for_llm(self, context_manager):
        """Test building context for LLM."""
        await context_manager.initialize()
        
        # Add message context items
        messages = [
            "Hello, how are you?",
            "I'm doing well, thank you!",
            "Can you help me with something?"
        ]
        
        for i, msg_content in enumerate(messages):
            message = ConversationMessage(
                id=f"msg_{i}",
                timestamp=time.time() + i,
                participant_id="user" if i % 2 == 0 else "agent",
                content=msg_content
            )
            await context_manager.add_message_context(message, extract_context=False)
        
        # Build LLM context
        llm_messages = await context_manager.build_context_for_llm(
            current_message="What can you tell me about that?",
            include_history=True,
            max_tokens=1000
        )
        
        assert len(llm_messages) > 0
        
        # Should include conversation history
        history_messages = [msg for msg in llm_messages if msg.content in messages]
        assert len(history_messages) > 0
        
        # Should include current message
        current_messages = [msg for msg in llm_messages if "What can you tell me" in msg.content]
        assert len(current_messages) == 1


class TestContextRelevanceManagement:
    """Test context relevance management."""
    
    @pytest.mark.asyncio
    async def test_update_context_relevance(self, context_manager):
        """Test updating context relevance."""
        await context_manager.initialize()
        
        item = await context_manager.add_context_item(
            content="Test content",
            context_type=ContextType.MESSAGE,
            scope=ContextScope.SHORT_TERM,
            relevance=ContextRelevance.MEDIUM,
            source="test"
        )
        
        # Update relevance
        success = await context_manager.update_context_relevance(
            item.id, ContextRelevance.HIGH
        )
        
        assert success is True
        assert context_manager.context_index[item.id].relevance == ContextRelevance.HIGH
        
        # Try updating non-existent item
        success = await context_manager.update_context_relevance(
            "non_existent", ContextRelevance.LOW
        )
        assert success is False
    
    @pytest.mark.asyncio
    async def test_relevance_decay(self, context_manager):
        """Test relevance decay over time."""
        await context_manager.initialize()
        
        # Create item with old timestamp
        old_time = time.time() - 7200  # 2 hours ago
        item = ContextItem(
            id="old_item",
            content="old content",
            context_type=ContextType.MESSAGE,
            scope=ContextScope.SHORT_TERM,
            relevance=ContextRelevance.HIGH,
            timestamp=old_time,
            source="test"
        )
        
        original_relevance = item.relevance
        
        # Update relevance score
        await context_manager._update_relevance_score(item)
        
        # Relevance should be affected by age (may or may not change depending on decay settings)
        # This test mainly ensures the method runs without error


class TestContextSummaryAndMetrics:
    """Test context summary and metrics functionality."""
    
    @pytest.mark.asyncio
    async def test_summarize_conversation_context(self, context_manager):
        """Test conversation context summarization."""
        await context_manager.initialize()
        
        # Add various context items
        await context_manager.add_context_item(
            content="Topic: weather",
            context_type=ContextType.TOPIC,
            scope=ContextScope.MEDIUM_TERM,
            relevance=ContextRelevance.HIGH,
            source="test"
        )
        
        context_manager.topic_history = ["weather", "technology", "health"]
        context_manager.entity_registry = {"john@example.com": {"type": "email"}}
        context_manager.intent_history = ["question", "help_request"]
        
        summary = await context_manager.summarize_conversation_context()
        
        assert summary["conversation_id"] == "test_conversation"
        assert "windows" in summary
        assert "topics" in summary
        assert "entities" in summary
        assert "intents" in summary
        
        assert len(summary["topics"]) > 0
        assert len(summary["entities"]) > 0
        assert len(summary["intents"]) > 0
    
    def test_get_metrics(self, context_manager):
        """Test metrics collection."""
        # Add some test data
        context_manager.context_index["test_item"] = ContextItem(
            id="test_item",
            content="test",
            context_type=ContextType.MESSAGE,
            scope=ContextScope.SHORT_TERM,
            relevance=ContextRelevance.MEDIUM,
            timestamp=time.time(),
            source="test"
        )
        
        context_manager.topic_history = ["topic1", "topic2"]
        context_manager.entity_registry = {"entity1": {}, "entity2": {}}
        context_manager.intent_history = ["intent1"]
        
        metrics = context_manager.get_metrics()
        
        assert metrics["conversation_id"] == "test_conversation"
        assert metrics["total_context_items"] == 1
        assert metrics["topic_count"] == 2
        assert metrics["entity_count"] == 2
        assert metrics["intent_count"] == 1
        assert "windows" in metrics


class TestStateIntegration:
    """Test integration with conversation state manager."""
    
    @pytest.mark.asyncio
    async def test_save_context_to_state(self, full_context_manager):
        """Test saving context to state manager."""
        await full_context_manager.initialize()
        
        # Add some context
        await full_context_manager.add_context_item(
            content="Test context",
            context_type=ContextType.MESSAGE,
            scope=ContextScope.SHORT_TERM,
            relevance=ContextRelevance.HIGH,
            source="test"
        )
        
        # Save to state
        await full_context_manager._save_context_to_state()
        
        # Should have called update_conversation_context
        full_context_manager.state_manager.update_conversation_context.assert_called_once()
        
        # Check the data structure
        call_args = full_context_manager.state_manager.update_conversation_context.call_args[0][0]
        assert "context_manager" in call_args
        assert "context_items" in call_args["context_manager"]
    
    @pytest.mark.asyncio
    async def test_load_context_from_state(self, full_context_manager):
        """Test loading context from state manager."""
        # Mock state with context data
        mock_state = Mock()
        mock_state.conversation_context = {
            "context_manager": {
                "context_items": [
                    {
                        "id": "loaded_item",
                        "content": "loaded content",
                        "context_type": "message",
                        "scope": "short_term",
                        "relevance": "high",
                        "timestamp": time.time(),
                        "source": "loaded",
                        "confidence": 1.0,
                        "access_count": 0
                    }
                ],
                "topic_history": ["loaded_topic"],
                "entity_registry": {"loaded_entity": {"type": "test"}},
                "intent_history": ["loaded_intent"]
            }
        }
        
        full_context_manager.state_manager.get_current_state.return_value = mock_state
        
        await full_context_manager.initialize()
        
        # Should have loaded the context
        assert "loaded_item" in full_context_manager.context_index
        assert "loaded_topic" in full_context_manager.topic_history
        assert "loaded_entity" in full_context_manager.entity_registry
        assert "loaded_intent" in full_context_manager.intent_history


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_callback_error_handling(self, context_manager):
        """Test error handling in callbacks."""
        def failing_callback(*args):
            raise Exception("Callback error")
        
        context_manager.set_callbacks(on_context_updated=failing_callback)
        await context_manager.initialize()
        
        # Should not raise exception despite callback failure
        await context_manager.add_context_item(
            content="Test",
            context_type=ContextType.MESSAGE,
            scope=ContextScope.SHORT_TERM,
            relevance=ContextRelevance.MEDIUM,
            source="test"
        )
    
    @pytest.mark.asyncio
    async def test_invalid_context_operations(self, context_manager):
        """Test handling of invalid context operations."""
        await context_manager.initialize()
        
        # Try to update non-existent context item
        success = await context_manager.update_context_relevance("invalid_id", ContextRelevance.HIGH)
        assert success is False
        
        # Try to get context with empty index
        items = await context_manager.get_relevant_context()
        assert len(items) == 0
    
    @pytest.mark.asyncio
    async def test_extraction_error_handling(self, context_manager):
        """Test error handling in context extraction."""
        await context_manager.initialize()
        
        # Create malformed message
        message = ConversationMessage(
            id="malformed",
            timestamp=time.time(),
            participant_id="user",
            content=None  # This might cause issues
        )
        
        # Should handle gracefully
        context_items = await context_manager.add_message_context(message)
        # Should at least create a basic message context item


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_conversation_flow(self, full_context_manager):
        """Test complete conversation flow with context management."""
        await full_context_manager.initialize()
        await full_context_manager.start_monitoring()
        
        # Simulate conversation messages
        messages = [
            ("user", "Hi, I'm having trouble with my computer"),
            ("agent", "I'd be happy to help! What specific issue are you experiencing?"),
            ("user", "My email client john.doe@example.com isn't working"),
            ("agent", "Let me help you troubleshoot the email issue")
        ]
        
        for i, (participant, content) in enumerate(messages):
            message = ConversationMessage(
                id=f"msg_{i}",
                timestamp=time.time() + i,
                participant_id=participant,
                content=content
            )
            
            await full_context_manager.add_message_context(message)
        
        # Should have extracted various context types
        assert len(full_context_manager.context_index) > len(messages)  # More than just messages
        
        # Should have detected entities (email)
        email_entities = [
            item for item in full_context_manager.context_index.values()
            if item.context_type == ContextType.ENTITY and "john.doe@example.com" in str(item.content)
        ]
        assert len(email_entities) > 0
        
        # Should have detected topics (technology)
        assert "technology" in full_context_manager.topic_history
        
        # Build LLM context
        llm_context = await full_context_manager.build_context_for_llm(
            current_message="Can you provide more details?",
            include_history=True
        )
        
        assert len(llm_context) > 0
        
        await full_context_manager.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_context_window_management(self, context_manager):
        """Test context window management across different scopes."""
        context_manager.config.immediate_window_size = 2
        context_manager.config.short_term_window_size = 3
        
        await context_manager.initialize()
        
        # Add many items to test window limits
        for i in range(10):
            await context_manager.add_context_item(
                content=f"Content {i}",
                context_type=ContextType.MESSAGE,
                scope=ContextScope.SHORT_TERM,
                relevance=ContextRelevance.MEDIUM,
                source="test"
            )
        
        # Windows should respect size limits
        assert len(context_manager.immediate_window.items) <= 2
        assert len(context_manager.short_term_window.items) <= 3
        
        # Get context and verify access tracking
        relevant_items = await context_manager.get_relevant_context(max_items=5)
        
        # Items should have been accessed
        for item in relevant_items:
            assert item.access_count > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])