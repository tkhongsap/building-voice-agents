"""
Multi-Turn Conversation Context Management

This module implements comprehensive context management for multi-turn conversations,
including context tracking, relevance scoring, and intelligent context windowing.
"""

import asyncio
import logging
import time
import json
from typing import Optional, Dict, Any, List, Callable, Union, Tuple
from enum import Enum
from dataclasses import dataclass, field, asdict
from collections import deque
import uuid

try:
    from .conversation_state_manager import ConversationStateManager, ConversationMessage, ConversationParticipant
    from ..components.llm.base_llm import BaseLLMProvider, LLMMessage, LLMRole
except ImportError:
    # Fallback for direct imports
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from conversation_state_manager import ConversationStateManager, ConversationMessage, ConversationParticipant
    from components.llm.base_llm import BaseLLMProvider, LLMMessage, LLMRole

logger = logging.getLogger(__name__)


class ContextRelevance(Enum):
    """Relevance levels for context items."""
    CRITICAL = "critical"       # Essential for conversation
    HIGH = "high"              # Very relevant
    MEDIUM = "medium"          # Moderately relevant
    LOW = "low"                # Minimally relevant
    IRRELEVANT = "irrelevant"  # Not relevant


class ContextType(Enum):
    """Types of context information."""
    MESSAGE = "message"
    TOPIC = "topic"
    ENTITY = "entity"
    INTENT = "intent"
    PREFERENCE = "preference"
    METADATA = "metadata"
    SYSTEM = "system"


class ContextScope(Enum):
    """Scope of context information."""
    IMMEDIATE = "immediate"     # Current turn only
    SHORT_TERM = "short_term"   # Last few turns
    MEDIUM_TERM = "medium_term" # Current conversation segment
    LONG_TERM = "long_term"     # Entire conversation
    SESSION = "session"         # Across conversation sessions
    GLOBAL = "global"           # User profile level


@dataclass
class ContextItem:
    """Represents a single context item."""
    id: str
    content: Any
    context_type: ContextType
    scope: ContextScope
    relevance: ContextRelevance
    timestamp: float
    source: str  # Where this context came from
    confidence: float = 1.0
    expiry_time: Optional[float] = None
    access_count: int = 0
    last_accessed: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.last_accessed is None:
            self.last_accessed = self.timestamp
    
    @property
    def age(self) -> float:
        """Get age of context item in seconds."""
        return time.time() - self.timestamp
    
    @property
    def is_expired(self) -> bool:
        """Check if context item has expired."""
        if self.expiry_time is None:
            return False
        return time.time() >= self.expiry_time
    
    def access(self) -> None:
        """Mark context item as accessed."""
        self.access_count += 1
        self.last_accessed = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['context_type'] = self.context_type.value
        data['scope'] = self.scope.value
        data['relevance'] = self.relevance.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContextItem':
        """Create from dictionary."""
        data['context_type'] = ContextType(data['context_type'])
        data['scope'] = ContextScope(data['scope'])
        data['relevance'] = ContextRelevance(data['relevance'])
        return cls(**data)


@dataclass
class ContextWindow:
    """Represents a context window for conversation processing."""
    window_id: str
    items: List[ContextItem] = field(default_factory=list)
    max_items: int = 50
    max_age_seconds: float = 3600.0  # 1 hour
    relevance_threshold: ContextRelevance = ContextRelevance.LOW
    total_tokens: int = 0
    max_tokens: int = 4000
    
    def add_item(self, item: ContextItem) -> bool:
        """Add item to context window."""
        if item.relevance.value == self.relevance_threshold.value or self._is_more_relevant(item.relevance):
            self.items.append(item)
            self._maintain_window_constraints()
            return True
        return False
    
    def _is_more_relevant(self, relevance: ContextRelevance) -> bool:
        """Check if relevance is higher than threshold."""
        relevance_order = [
            ContextRelevance.IRRELEVANT,
            ContextRelevance.LOW,
            ContextRelevance.MEDIUM,
            ContextRelevance.HIGH,
            ContextRelevance.CRITICAL
        ]
        return relevance_order.index(relevance) >= relevance_order.index(self.relevance_threshold)
    
    def _maintain_window_constraints(self) -> None:
        """Maintain window size and age constraints."""
        current_time = time.time()
        
        # Remove expired items
        self.items = [item for item in self.items if not item.is_expired]
        
        # Remove items older than max age
        self.items = [
            item for item in self.items 
            if current_time - item.timestamp <= self.max_age_seconds
        ]
        
        # Sort by relevance and recency
        self.items.sort(
            key=lambda x: (
                list(ContextRelevance).index(x.relevance),
                x.timestamp
            ),
            reverse=True
        )
        
        # Limit number of items
        if len(self.items) > self.max_items:
            self.items = self.items[:self.max_items]
    
    def get_relevant_items(self, min_relevance: ContextRelevance = ContextRelevance.LOW) -> List[ContextItem]:
        """Get items with at least the specified relevance."""
        return [
            item for item in self.items 
            if self._is_more_relevant(item.relevance) or item.relevance == min_relevance
        ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'window_id': self.window_id,
            'items': [item.to_dict() for item in self.items],
            'max_items': self.max_items,
            'max_age_seconds': self.max_age_seconds,
            'relevance_threshold': self.relevance_threshold.value,
            'total_tokens': self.total_tokens,
            'max_tokens': self.max_tokens
        }


@dataclass
class ContextManagerConfig:
    """Configuration for context manager."""
    
    # Window settings
    immediate_window_size: int = 10
    short_term_window_size: int = 30
    medium_term_window_size: int = 100
    long_term_window_size: int = 500
    
    # Token limits
    immediate_token_limit: int = 1000
    short_term_token_limit: int = 2000
    medium_term_token_limit: int = 4000
    long_term_token_limit: int = 8000
    
    # Time limits (seconds)
    immediate_time_limit: float = 60.0      # 1 minute
    short_term_time_limit: float = 600.0    # 10 minutes
    medium_term_time_limit: float = 3600.0  # 1 hour
    long_term_time_limit: float = 86400.0   # 24 hours
    
    # Relevance settings
    auto_relevance_scoring: bool = True
    relevance_decay_rate: float = 0.1       # Per hour
    min_relevance_for_retention: ContextRelevance = ContextRelevance.LOW
    
    # Context extraction
    enable_entity_extraction: bool = True
    enable_topic_tracking: bool = True
    enable_intent_detection: bool = True
    enable_sentiment_analysis: bool = True
    
    # Performance optimization
    enable_async_processing: bool = True
    batch_processing_size: int = 10
    max_concurrent_operations: int = 5
    enable_caching: bool = True
    
    # Persistence
    enable_context_persistence: bool = True
    persistence_interval: float = 300.0     # 5 minutes
    
    # Monitoring
    enable_detailed_logging: bool = False
    track_context_metrics: bool = True


class ConversationContextManager:
    """
    Manages multi-turn conversation context with intelligent windowing and relevance scoring.
    
    Tracks conversation context across different time scales and maintains relevant
    information for improved conversation understanding and response generation.
    """
    
    def __init__(
        self,
        conversation_id: str,
        state_manager: Optional[ConversationStateManager] = None,
        llm_provider: Optional[BaseLLMProvider] = None,
        config: Optional[ContextManagerConfig] = None
    ):
        self.conversation_id = conversation_id
        self.state_manager = state_manager
        self.llm_provider = llm_provider
        self.config = config or ContextManagerConfig()
        
        # Context windows
        self.immediate_window = ContextWindow(
            window_id="immediate",
            max_items=self.config.immediate_window_size,
            max_age_seconds=self.config.immediate_time_limit,
            max_tokens=self.config.immediate_token_limit
        )
        self.short_term_window = ContextWindow(
            window_id="short_term",
            max_items=self.config.short_term_window_size,
            max_age_seconds=self.config.short_term_time_limit,
            max_tokens=self.config.short_term_token_limit
        )
        self.medium_term_window = ContextWindow(
            window_id="medium_term",
            max_items=self.config.medium_term_window_size,
            max_age_seconds=self.config.medium_term_time_limit,
            max_tokens=self.config.medium_term_token_limit
        )
        self.long_term_window = ContextWindow(
            window_id="long_term",
            max_items=self.config.long_term_window_size,
            max_age_seconds=self.config.long_term_time_limit,
            max_tokens=self.config.long_term_token_limit
        )
        
        # Context tracking
        self.context_index: Dict[str, ContextItem] = {}
        self.topic_history: List[str] = []
        self.entity_registry: Dict[str, Dict[str, Any]] = {}
        self.intent_history: List[str] = []
        
        # Processing
        self._is_monitoring = False
        self._processing_queue: asyncio.Queue = asyncio.Queue()
        self._processing_task: Optional[asyncio.Task] = None
        
        # Callbacks
        self._on_context_updated: Optional[Callable] = None
        self._on_topic_changed: Optional[Callable] = None
        self._on_entity_detected: Optional[Callable] = None
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def initialize(self) -> None:
        """Initialize the context manager."""
        try:
            self.logger.info("Initializing conversation context manager...")
            
            # Load existing context if available
            if self.state_manager:
                await self._load_context_from_state()
            
            self.logger.info("Conversation context manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize context manager: {e}")
            raise
    
    async def start_monitoring(self) -> None:
        """Start context monitoring and processing."""
        if self._is_monitoring:
            return
        
        self._is_monitoring = True
        
        # Start processing task
        if self.config.enable_async_processing:
            self._processing_task = asyncio.create_task(self._processing_loop())
        
        self.logger.info("Context monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop context monitoring."""
        if not self._is_monitoring:
            return
        
        self._is_monitoring = False
        
        # Cancel processing task
        if self._processing_task and not self._processing_task.done():
            self._processing_task.cancel()
        
        # Save context if persistence enabled
        if self.config.enable_context_persistence and self.state_manager:
            await self._save_context_to_state()
        
        self.logger.info("Context monitoring stopped")
    
    async def add_message_context(
        self,
        message: ConversationMessage,
        extract_context: bool = True
    ) -> List[ContextItem]:
        """
        Add context from a conversation message.
        
        Args:
            message: The conversation message
            extract_context: Whether to extract additional context
            
        Returns:
            List of context items created
        """
        context_items = []
        
        # Create basic message context
        message_context = ContextItem(
            id=f"msg_{message.id}",
            content=message.content,
            context_type=ContextType.MESSAGE,
            scope=ContextScope.IMMEDIATE,
            relevance=ContextRelevance.HIGH,
            timestamp=message.timestamp,
            source=f"message_{message.participant_id}",
            metadata=message.metadata
        )
        
        context_items.append(message_context)
        await self._add_context_item(message_context)
        
        # Extract additional context if enabled
        if extract_context:
            extracted_items = await self._extract_context_from_message(message)
            context_items.extend(extracted_items)
        
        return context_items
    
    async def add_context_item(
        self,
        content: Any,
        context_type: ContextType,
        scope: ContextScope = ContextScope.SHORT_TERM,
        relevance: ContextRelevance = ContextRelevance.MEDIUM,
        source: str = "manual",
        metadata: Optional[Dict[str, Any]] = None
    ) -> ContextItem:
        """Add a custom context item."""
        context_item = ContextItem(
            id=str(uuid.uuid4()),
            content=content,
            context_type=context_type,
            scope=scope,
            relevance=relevance,
            timestamp=time.time(),
            source=source,
            metadata=metadata
        )
        
        await self._add_context_item(context_item)
        return context_item
    
    async def get_relevant_context(
        self,
        scope: Optional[ContextScope] = None,
        context_type: Optional[ContextType] = None,
        min_relevance: ContextRelevance = ContextRelevance.LOW,
        max_items: Optional[int] = None
    ) -> List[ContextItem]:
        """
        Get relevant context items based on criteria.
        
        Args:
            scope: Filter by context scope
            context_type: Filter by context type
            min_relevance: Minimum relevance level
            max_items: Maximum number of items to return
            
        Returns:
            List of relevant context items
        """
        relevant_items = []
        
        # Get items from appropriate windows
        if scope == ContextScope.IMMEDIATE:
            relevant_items = self.immediate_window.get_relevant_items(min_relevance)
        elif scope == ContextScope.SHORT_TERM:
            relevant_items = self.short_term_window.get_relevant_items(min_relevance)
        elif scope == ContextScope.MEDIUM_TERM:
            relevant_items = self.medium_term_window.get_relevant_items(min_relevance)
        elif scope == ContextScope.LONG_TERM:
            relevant_items = self.long_term_window.get_relevant_items(min_relevance)
        else:
            # Get from all windows
            relevant_items.extend(self.immediate_window.get_relevant_items(min_relevance))
            relevant_items.extend(self.short_term_window.get_relevant_items(min_relevance))
            relevant_items.extend(self.medium_term_window.get_relevant_items(min_relevance))
            relevant_items.extend(self.long_term_window.get_relevant_items(min_relevance))
        
        # Filter by context type if specified
        if context_type:
            relevant_items = [item for item in relevant_items if item.context_type == context_type]
        
        # Sort by relevance and recency
        relevant_items.sort(
            key=lambda x: (
                list(ContextRelevance).index(x.relevance),
                x.timestamp
            ),
            reverse=True
        )
        
        # Limit number of items if specified
        if max_items:
            relevant_items = relevant_items[:max_items]
        
        # Mark items as accessed
        for item in relevant_items:
            item.access()
        
        return relevant_items
    
    async def build_context_for_llm(
        self,
        current_message: Optional[str] = None,
        include_history: bool = True,
        max_tokens: Optional[int] = None
    ) -> List[LLMMessage]:
        """
        Build context for LLM from conversation history.
        
        Args:
            current_message: Current user message
            include_history: Whether to include conversation history
            max_tokens: Maximum tokens for context
            
        Returns:
            List of LLM messages for context
        """
        llm_messages = []
        token_count = 0
        max_token_limit = max_tokens or self.config.short_term_token_limit
        
        if include_history:
            # Get relevant context items
            context_items = await self.get_relevant_context(
                scope=ContextScope.SHORT_TERM,
                min_relevance=ContextRelevance.MEDIUM,
                max_items=20
            )
            
            # Convert context items to LLM messages
            for item in context_items:
                if item.context_type == ContextType.MESSAGE:
                    # Estimate token count (rough approximation)
                    estimated_tokens = len(str(item.content).split()) * 1.3
                    
                    if token_count + estimated_tokens <= max_token_limit:
                        # Determine role based on source
                        role = LLMRole.USER if "user" in item.source else LLMRole.ASSISTANT
                        
                        llm_message = LLMMessage(
                            role=role,
                            content=str(item.content)
                        )
                        llm_messages.append(llm_message)
                        token_count += estimated_tokens
                    else:
                        break
        
        # Add current message if provided
        if current_message:
            llm_messages.append(LLMMessage(
                role=LLMRole.USER,
                content=current_message
            ))
        
        return llm_messages
    
    async def update_context_relevance(
        self,
        context_id: str,
        new_relevance: ContextRelevance
    ) -> bool:
        """Update relevance of a context item."""
        if context_id in self.context_index:
            item = self.context_index[context_id]
            old_relevance = item.relevance
            item.relevance = new_relevance
            
            self.logger.debug(
                f"Updated context relevance: {context_id} "
                f"{old_relevance.value} -> {new_relevance.value}"
            )
            return True
        
        return False
    
    async def summarize_conversation_context(self) -> Dict[str, Any]:
        """Create a summary of current conversation context."""
        summary = {
            "conversation_id": self.conversation_id,
            "timestamp": time.time(),
            "windows": {
                "immediate": {
                    "item_count": len(self.immediate_window.items),
                    "token_count": self.immediate_window.total_tokens
                },
                "short_term": {
                    "item_count": len(self.short_term_window.items),
                    "token_count": self.short_term_window.total_tokens
                },
                "medium_term": {
                    "item_count": len(self.medium_term_window.items),
                    "token_count": self.medium_term_window.total_tokens
                },
                "long_term": {
                    "item_count": len(self.long_term_window.items),
                    "token_count": self.long_term_window.total_tokens
                }
            },
            "topics": self.topic_history[-5:],  # Last 5 topics
            "entities": list(self.entity_registry.keys())[:10],  # Top 10 entities
            "intents": self.intent_history[-5:]  # Last 5 intents
        }
        
        return summary
    
    async def _add_context_item(self, item: ContextItem) -> None:
        """Add context item to appropriate windows."""
        # Add to index
        self.context_index[item.id] = item
        
        # Add to appropriate windows based on scope
        if item.scope == ContextScope.IMMEDIATE:
            self.immediate_window.add_item(item)
        elif item.scope == ContextScope.SHORT_TERM:
            self.short_term_window.add_item(item)
            self.immediate_window.add_item(item)
        elif item.scope == ContextScope.MEDIUM_TERM:
            self.medium_term_window.add_item(item)
            self.short_term_window.add_item(item)
            self.immediate_window.add_item(item)
        elif item.scope == ContextScope.LONG_TERM:
            self.long_term_window.add_item(item)
            self.medium_term_window.add_item(item)
            self.short_term_window.add_item(item)
            self.immediate_window.add_item(item)
        
        # Queue for async processing if enabled
        if self.config.enable_async_processing:
            await self._processing_queue.put(('process_item', item))
        
        # Notify callbacks
        if self._on_context_updated:
            try:
                await self._on_context_updated(item)
            except Exception as e:
                self.logger.error(f"Error in context updated callback: {e}")
    
    async def _extract_context_from_message(self, message: ConversationMessage) -> List[ContextItem]:
        """Extract additional context from a message."""
        context_items = []
        
        try:
            if self.config.enable_entity_extraction:
                entities = await self._extract_entities(message.content)
                for entity in entities:
                    context_item = ContextItem(
                        id=f"entity_{entity['name']}_{int(time.time())}",
                        content=entity,
                        context_type=ContextType.ENTITY,
                        scope=ContextScope.MEDIUM_TERM,
                        relevance=ContextRelevance.MEDIUM,
                        timestamp=message.timestamp,
                        source=f"extraction_{message.id}",
                        metadata={"entity_type": entity.get("type")}
                    )
                    context_items.append(context_item)
                    await self._add_context_item(context_item)
            
            if self.config.enable_topic_tracking:
                topics = await self._extract_topics(message.content)
                for topic in topics:
                    if topic not in self.topic_history:
                        self.topic_history.append(topic)
                        
                        context_item = ContextItem(
                            id=f"topic_{topic}_{int(time.time())}",
                            content=topic,
                            context_type=ContextType.TOPIC,
                            scope=ContextScope.MEDIUM_TERM,
                            relevance=ContextRelevance.HIGH,
                            timestamp=message.timestamp,
                            source=f"topic_extraction_{message.id}"
                        )
                        context_items.append(context_item)
                        await self._add_context_item(context_item)
                        
                        # Notify topic change
                        if self._on_topic_changed:
                            try:
                                await self._on_topic_changed(topic)
                            except Exception as e:
                                self.logger.error(f"Error in topic changed callback: {e}")
            
            if self.config.enable_intent_detection:
                intent = await self._detect_intent(message.content)
                if intent:
                    self.intent_history.append(intent)
                    
                    context_item = ContextItem(
                        id=f"intent_{intent}_{int(time.time())}",
                        content=intent,
                        context_type=ContextType.INTENT,
                        scope=ContextScope.SHORT_TERM,
                        relevance=ContextRelevance.HIGH,
                        timestamp=message.timestamp,
                        source=f"intent_detection_{message.id}"
                    )
                    context_items.append(context_item)
                    await self._add_context_item(context_item)
        
        except Exception as e:
            self.logger.warning(f"Error extracting context from message: {e}")
        
        return context_items
    
    async def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text (simplified implementation)."""
        # This is a simplified entity extraction
        # In a real implementation, you would use NLP libraries like spaCy or transformers
        entities = []
        
        # Simple pattern matching for common entities
        import re
        
        # Email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        for match in re.finditer(email_pattern, text):
            entities.append({
                "name": match.group(),
                "type": "email",
                "start": match.start(),
                "end": match.end()
            })
        
        # Phone numbers (simple pattern)
        phone_pattern = r'\b\d{3}-\d{3}-\d{4}\b'
        for match in re.finditer(phone_pattern, text):
            entities.append({
                "name": match.group(),
                "type": "phone",
                "start": match.start(),
                "end": match.end()
            })
        
        # Update entity registry
        for entity in entities:
            self.entity_registry[entity["name"]] = entity
        
        return entities
    
    async def _extract_topics(self, text: str) -> List[str]:
        """Extract topics from text (simplified implementation)."""
        # This is a simplified topic extraction
        # In a real implementation, you would use topic modeling or LLM-based extraction
        
        # Simple keyword-based topic detection
        topic_keywords = {
            "weather": ["weather", "temperature", "rain", "snow", "sunny", "cloudy"],
            "technology": ["computer", "software", "AI", "programming", "code", "app"],
            "health": ["doctor", "medicine", "hospital", "symptom", "treatment"],
            "travel": ["trip", "vacation", "flight", "hotel", "destination"],
            "food": ["restaurant", "recipe", "cooking", "meal", "dinner"]
        }
        
        text_lower = text.lower()
        detected_topics = []
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_topics.append(topic)
        
        return detected_topics
    
    async def _detect_intent(self, text: str) -> Optional[str]:
        """Detect intent from text (simplified implementation)."""
        # This is a simplified intent detection
        # In a real implementation, you would use intent classification models
        
        text_lower = text.lower()
        
        # Simple pattern-based intent detection
        if any(word in text_lower for word in ["help", "assist", "support"]):
            return "help_request"
        elif any(word in text_lower for word in ["book", "schedule", "appointment"]):
            return "booking"
        elif any(word in text_lower for word in ["cancel", "stop", "end"]):
            return "cancellation"
        elif "?" in text:
            return "question"
        elif any(word in text_lower for word in ["thank", "thanks", "appreciate"]):
            return "gratitude"
        
        return None
    
    async def _processing_loop(self) -> None:
        """Background processing loop for context items."""
        while self._is_monitoring:
            try:
                # Process items from queue
                action, item = await asyncio.wait_for(
                    self._processing_queue.get(),
                    timeout=1.0
                )
                
                if action == 'process_item':
                    await self._process_context_item(item)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _process_context_item(self, item: ContextItem) -> None:
        """Process a context item for additional insights."""
        try:
            # Update relevance based on age and access patterns
            if self.config.auto_relevance_scoring:
                await self._update_relevance_score(item)
            
            # Additional processing could include:
            # - Similarity scoring with other items
            # - Clustering related items
            # - Extracting additional metadata
            
        except Exception as e:
            self.logger.warning(f"Error processing context item {item.id}: {e}")
    
    async def _update_relevance_score(self, item: ContextItem) -> None:
        """Update relevance score based on age and usage."""
        # Decay relevance over time
        age_hours = item.age / 3600
        decay_factor = 1.0 - (self.config.relevance_decay_rate * age_hours)
        decay_factor = max(0.1, decay_factor)  # Minimum relevance
        
        # Boost relevance based on access count
        access_boost = min(0.2, item.access_count * 0.05)
        
        # Calculate new relevance (simplified scoring)
        base_relevance = list(ContextRelevance).index(item.relevance)
        adjusted_relevance = base_relevance * decay_factor + access_boost
        
        # Map back to enum
        relevance_values = list(ContextRelevance)
        new_relevance_index = max(0, min(len(relevance_values) - 1, int(adjusted_relevance)))
        item.relevance = relevance_values[new_relevance_index]
    
    async def _load_context_from_state(self) -> None:
        """Load context from conversation state manager."""
        if not self.state_manager:
            return
        
        try:
            current_state = self.state_manager.get_current_state()
            context_data = current_state.conversation_context.get("context_manager")
            
            if context_data:
                # Restore context items
                for item_data in context_data.get("context_items", []):
                    item = ContextItem.from_dict(item_data)
                    await self._add_context_item(item)
                
                # Restore topic history
                self.topic_history = context_data.get("topic_history", [])
                
                # Restore entity registry
                self.entity_registry = context_data.get("entity_registry", {})
                
                # Restore intent history
                self.intent_history = context_data.get("intent_history", [])
                
                self.logger.info("Loaded context from conversation state")
        
        except Exception as e:
            self.logger.warning(f"Could not load context from state: {e}")
    
    async def _save_context_to_state(self) -> None:
        """Save context to conversation state manager."""
        if not self.state_manager:
            return
        
        try:
            # Prepare context data
            context_data = {
                "context_items": [
                    item.to_dict() for item in list(self.context_index.values())[-100:]  # Last 100 items
                ],
                "topic_history": self.topic_history[-20:],  # Last 20 topics
                "entity_registry": dict(list(self.entity_registry.items())[-50:]),  # Last 50 entities
                "intent_history": self.intent_history[-20:]  # Last 20 intents
            }
            
            # Save to conversation context
            await self.state_manager.update_conversation_context({
                "context_manager": context_data
            })
            
            self.logger.debug("Saved context to conversation state")
        
        except Exception as e:
            self.logger.error(f"Error saving context to state: {e}")
    
    def set_callbacks(
        self,
        on_context_updated: Optional[Callable] = None,
        on_topic_changed: Optional[Callable] = None,
        on_entity_detected: Optional[Callable] = None
    ) -> None:
        """Set callback functions for context events."""
        self._on_context_updated = on_context_updated
        self._on_topic_changed = on_topic_changed
        self._on_entity_detected = on_entity_detected
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get context manager metrics."""
        return {
            "conversation_id": self.conversation_id,
            "total_context_items": len(self.context_index),
            "windows": {
                "immediate": {
                    "items": len(self.immediate_window.items),
                    "tokens": self.immediate_window.total_tokens
                },
                "short_term": {
                    "items": len(self.short_term_window.items),
                    "tokens": self.short_term_window.total_tokens
                },
                "medium_term": {
                    "items": len(self.medium_term_window.items),
                    "tokens": self.medium_term_window.total_tokens
                },
                "long_term": {
                    "items": len(self.long_term_window.items),
                    "tokens": self.long_term_window.total_tokens
                }
            },
            "topic_count": len(self.topic_history),
            "entity_count": len(self.entity_registry),
            "intent_count": len(self.intent_history),
            "is_monitoring": self._is_monitoring
        }
    
    async def cleanup(self) -> None:
        """Clean up context manager resources."""
        await self.stop_monitoring()
        
        # Clear context data
        self.context_index.clear()
        self.topic_history.clear()
        self.entity_registry.clear()
        self.intent_history.clear()
        
        self.logger.info("Context manager cleanup completed")