"""
Conversation State Preservation and Recovery

This module implements comprehensive conversation state management with
preservation, recovery, and persistence capabilities for robust conversations.
"""

import asyncio
import logging
import time
import json
import uuid
from typing import Optional, Dict, Any, List, Callable, Union
from enum import Enum
from dataclasses import dataclass, field, asdict
from collections import deque
import pickle
import os
from pathlib import Path

try:
    from .turn_detector import TurnDetector, TurnSegment, TurnState
    from .interruption_handler import InterruptionHandler, InterruptionEvent, AgentState
    from .pipeline_flush_handler import PipelineFlushHandler, FlushOperation
    from ..components.llm.base_llm import BaseLLMProvider, LLMMessage, LLMRole
except ImportError:
    # Fallback for direct imports
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from turn_detector import TurnDetector, TurnSegment, TurnState
    from interruption_handler import InterruptionHandler, InterruptionEvent, AgentState
    from pipeline_flush_handler import PipelineFlushHandler, FlushOperation
    from components.llm.base_llm import BaseLLMProvider, LLMMessage, LLMRole

logger = logging.getLogger(__name__)


class ConversationStateType(Enum):
    """Types of conversation states."""
    ACTIVE = "active"
    PAUSED = "paused"
    INTERRUPTED = "interrupted"
    RECOVERING = "recovering"
    ENDED = "ended"
    ERROR = "error"
    SUSPENDED = "suspended"


class RecoveryStrategy(Enum):
    """Recovery strategies for conversation state."""
    IMMEDIATE = "immediate"        # Immediate recovery
    GRADUAL = "gradual"           # Gradual state restoration
    USER_INITIATED = "user_initiated"  # Wait for user to initiate
    CONTEXT_AWARE = "context_aware"    # Recover based on context
    SMART_RESUME = "smart_resume"      # Intelligent resume point


class StatePreservationLevel(Enum):
    """Levels of state preservation."""
    MINIMAL = "minimal"           # Basic conversation info
    STANDARD = "standard"         # Full conversation state
    COMPREHENSIVE = "comprehensive"  # Everything including metadata
    CUSTOM = "custom"             # User-defined preservation


@dataclass
class ConversationParticipant:
    """Represents a conversation participant."""
    id: str
    name: Optional[str] = None
    role: str = "user"  # user, agent, system
    metadata: Optional[Dict[str, Any]] = None
    preferences: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConversationMessage:
    """Represents a message in conversation state."""
    id: str
    timestamp: float
    participant_id: str
    content: str
    message_type: str = "text"  # text, audio, system
    metadata: Optional[Dict[str, Any]] = None
    confidence: Optional[float] = None
    is_partial: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationMessage':
        return cls(**data)


@dataclass
class ConversationState:
    """Represents the complete conversation state."""
    session_id: str
    conversation_id: str
    timestamp: float
    state_type: ConversationStateType
    participants: List[ConversationParticipant] = field(default_factory=list)
    messages: List[ConversationMessage] = field(default_factory=list)
    
    # Turn and interaction state
    current_turn: Optional[TurnSegment] = None
    turn_history: List[TurnSegment] = field(default_factory=list)
    interruption_context: Optional[InterruptionEvent] = None
    
    # Component states
    agent_state: Optional[AgentState] = None
    turn_detector_state: Optional[TurnState] = None
    pipeline_state: Optional[str] = None
    
    # Context and memory
    conversation_context: Dict[str, Any] = field(default_factory=dict)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    session_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Recovery information
    last_stable_state: Optional[float] = None
    recovery_checkpoints: List[float] = field(default_factory=list)
    preservation_level: StatePreservationLevel = StatePreservationLevel.STANDARD
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Handle enum serialization
        data['state_type'] = self.state_type.value
        data['preservation_level'] = self.preservation_level.value
        if self.agent_state:
            data['agent_state'] = self.agent_state.value
        if self.turn_detector_state:
            data['turn_detector_state'] = self.turn_detector_state.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationState':
        """Create from dictionary."""
        # Handle enum deserialization
        data['state_type'] = ConversationStateType(data['state_type'])
        data['preservation_level'] = StatePreservationLevel(data['preservation_level'])
        if data.get('agent_state'):
            data['agent_state'] = AgentState(data['agent_state'])
        if data.get('turn_detector_state'):
            data['turn_detector_state'] = TurnState(data['turn_detector_state'])
        
        # Convert nested objects
        if 'participants' in data:
            data['participants'] = [
                ConversationParticipant(**p) if isinstance(p, dict) else p
                for p in data['participants']
            ]
        if 'messages' in data:
            data['messages'] = [
                ConversationMessage.from_dict(m) if isinstance(m, dict) else m
                for m in data['messages']
            ]
        
        return cls(**data)


@dataclass
class RecoveryContext:
    """Context for conversation recovery."""
    recovery_id: str
    timestamp: float
    strategy: RecoveryStrategy
    target_state: ConversationState
    recovery_reason: str
    estimated_duration: Optional[float] = None
    recovery_steps: List[str] = field(default_factory=list)
    success: bool = False
    error_message: Optional[str] = None
    completion_time: Optional[float] = None
    
    @property
    def duration(self) -> Optional[float]:
        if self.completion_time:
            return self.completion_time - self.timestamp
        return None


@dataclass
class ConversationStateConfig:
    """Configuration for conversation state management."""
    
    # State persistence
    enable_persistence: bool = True
    persistence_interval: float = 30.0  # Save state every 30 seconds
    max_states_in_memory: int = 100
    state_retention_days: int = 30
    
    # Recovery settings
    auto_recovery_enabled: bool = True
    recovery_timeout: float = 10.0
    max_recovery_attempts: int = 3
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.CONTEXT_AWARE
    
    # Checkpoint settings
    checkpoint_interval: float = 60.0   # Create checkpoint every minute
    max_checkpoints: int = 20
    auto_checkpoint_on_turns: bool = True
    
    # Preservation settings
    default_preservation_level: StatePreservationLevel = StatePreservationLevel.STANDARD
    preserve_partial_messages: bool = True
    preserve_interruption_context: bool = True
    preserve_user_preferences: bool = True
    
    # Storage settings
    storage_directory: str = "./conversation_states"
    use_compression: bool = True
    use_encryption: bool = False
    
    # Performance optimization
    enable_lazy_loading: bool = True
    max_concurrent_operations: int = 5
    enable_async_persistence: bool = True
    
    # Monitoring
    enable_detailed_logging: bool = False
    track_state_metrics: bool = True
    log_recovery_operations: bool = True


class ConversationStateManager:
    """
    Manages conversation state preservation and recovery.
    
    Provides comprehensive state management capabilities including automatic
    preservation, intelligent recovery, and persistence across sessions.
    """
    
    def __init__(
        self,
        session_id: str,
        conversation_id: Optional[str] = None,
        turn_detector: Optional[TurnDetector] = None,
        interruption_handler: Optional[InterruptionHandler] = None,
        flush_handler: Optional[PipelineFlushHandler] = None,
        llm_provider: Optional[BaseLLMProvider] = None,
        config: Optional[ConversationStateConfig] = None
    ):
        self.session_id = session_id
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.turn_detector = turn_detector
        self.interruption_handler = interruption_handler
        self.flush_handler = flush_handler
        self.llm_provider = llm_provider
        self.config = config or ConversationStateConfig()
        
        # Current state
        self.current_state = ConversationState(
            session_id=self.session_id,
            conversation_id=self.conversation_id,
            timestamp=time.time(),
            state_type=ConversationStateType.ACTIVE,
            preservation_level=self.config.default_preservation_level
        )
        
        # State management
        self.state_history: deque = deque(maxlen=self.config.max_states_in_memory)
        self.recovery_history: List[RecoveryContext] = []
        self.checkpoints: Dict[str, ConversationState] = {}
        
        # Processing
        self._is_monitoring = False
        self._persistence_task: Optional[asyncio.Task] = None
        self._checkpoint_task: Optional[asyncio.Task] = None
        self._state_lock = asyncio.Lock()
        
        # Callbacks
        self._on_state_changed: Optional[Callable] = None
        self._on_state_preserved: Optional[Callable] = None
        self._on_recovery_started: Optional[Callable] = None
        self._on_recovery_completed: Optional[Callable] = None
        
        # Storage
        self.storage_path = Path(self.config.storage_directory)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def initialize(self) -> None:
        """Initialize the state manager."""
        try:
            self.logger.info("Initializing conversation state manager...")
            
            # Setup component callbacks
            await self._setup_component_callbacks()
            
            # Load existing state if available
            await self._load_existing_state()
            
            # Initialize participants
            await self._initialize_participants()
            
            self.logger.info("Conversation state manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize state manager: {e}")
            raise
    
    async def start_monitoring(self) -> None:
        """Start state monitoring and automatic preservation."""
        if self._is_monitoring:
            return
        
        self._is_monitoring = True
        
        # Start persistence task
        if self.config.enable_persistence and self.config.enable_async_persistence:
            self._persistence_task = asyncio.create_task(self._persistence_loop())
        
        # Start checkpoint task
        self._checkpoint_task = asyncio.create_task(self._checkpoint_loop())
        
        self.logger.info("Conversation state monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop monitoring and save final state."""
        if not self._is_monitoring:
            return
        
        self._is_monitoring = False
        
        # Cancel tasks
        if self._persistence_task and not self._persistence_task.done():
            self._persistence_task.cancel()
        if self._checkpoint_task and not self._checkpoint_task.done():
            self._checkpoint_task.cancel()
        
        # Save final state
        await self.preserve_current_state()
        
        self.logger.info("Conversation state monitoring stopped")
    
    async def preserve_current_state(
        self,
        preservation_level: Optional[StatePreservationLevel] = None
    ) -> ConversationState:
        """
        Preserve the current conversation state.
        
        Args:
            preservation_level: Level of detail to preserve
            
        Returns:
            Preserved conversation state
        """
        async with self._state_lock:
            # Update preservation level if specified
            if preservation_level:
                self.current_state.preservation_level = preservation_level
            
            # Update current state with latest information
            await self._update_current_state()
            
            # Create state snapshot
            state_snapshot = self._create_state_snapshot()
            
            # Add to history
            self.state_history.append(state_snapshot)
            
            # Persist to storage if enabled
            if self.config.enable_persistence:
                await self._persist_state(state_snapshot)
            
            self.logger.debug(f"Preserved conversation state: {state_snapshot.state_type.value}")
            
            # Notify callbacks
            if self._on_state_preserved:
                try:
                    await self._on_state_preserved(state_snapshot)
                except Exception as e:
                    self.logger.error(f"Error in state preserved callback: {e}")
            
            return state_snapshot
    
    async def recover_conversation_state(
        self,
        target_timestamp: Optional[float] = None,
        recovery_strategy: Optional[RecoveryStrategy] = None,
        reason: str = "manual_recovery"
    ) -> RecoveryContext:
        """
        Recover conversation to a previous state.
        
        Args:
            target_timestamp: Specific timestamp to recover to
            recovery_strategy: Strategy to use for recovery
            reason: Reason for recovery
            
        Returns:
            Recovery context with results
        """
        strategy = recovery_strategy or self.config.recovery_strategy
        recovery_id = str(uuid.uuid4())
        
        recovery_ctx = RecoveryContext(
            recovery_id=recovery_id,
            timestamp=time.time(),
            strategy=strategy,
            target_state=None,
            recovery_reason=reason
        )
        
        try:
            self.logger.info(f"Starting conversation recovery: {strategy.value}")
            
            # Find target state
            target_state = await self._find_recovery_target(target_timestamp, strategy)
            if not target_state:
                raise ValueError("No suitable recovery target found")
            
            recovery_ctx.target_state = target_state
            
            # Notify recovery started
            if self._on_recovery_started:
                await self._on_recovery_started(recovery_ctx)
            
            # Execute recovery based on strategy
            if strategy == RecoveryStrategy.IMMEDIATE:
                await self._execute_immediate_recovery(recovery_ctx)
            elif strategy == RecoveryStrategy.GRADUAL:
                await self._execute_gradual_recovery(recovery_ctx)
            elif strategy == RecoveryStrategy.CONTEXT_AWARE:
                await self._execute_context_aware_recovery(recovery_ctx)
            elif strategy == RecoveryStrategy.SMART_RESUME:
                await self._execute_smart_resume_recovery(recovery_ctx)
            else:  # USER_INITIATED
                await self._prepare_user_initiated_recovery(recovery_ctx)
            
            recovery_ctx.success = True
            recovery_ctx.completion_time = time.time()
            
            self.logger.info(
                f"Recovery completed successfully in {recovery_ctx.duration:.3f}s"
            )
            
        except Exception as e:
            recovery_ctx.success = False
            recovery_ctx.error_message = str(e)
            recovery_ctx.completion_time = time.time()
            
            self.logger.error(f"Recovery failed: {e}")
        
        finally:
            self.recovery_history.append(recovery_ctx)
            
            # Notify recovery completed
            if self._on_recovery_completed:
                try:
                    await self._on_recovery_completed(recovery_ctx)
                except Exception as e:
                    self.logger.error(f"Error in recovery completed callback: {e}")
        
        return recovery_ctx
    
    async def create_checkpoint(self, checkpoint_name: Optional[str] = None) -> str:
        """
        Create a recovery checkpoint.
        
        Args:
            checkpoint_name: Optional name for the checkpoint
            
        Returns:
            Checkpoint ID
        """
        checkpoint_id = checkpoint_name or f"checkpoint_{int(time.time())}"
        
        # Preserve current state as checkpoint
        checkpoint_state = await self.preserve_current_state()
        self.checkpoints[checkpoint_id] = checkpoint_state
        
        # Update current state checkpoints list
        self.current_state.recovery_checkpoints.append(checkpoint_state.timestamp)
        
        # Limit number of checkpoints
        if len(self.checkpoints) > self.config.max_checkpoints:
            # Remove oldest checkpoint
            oldest_id = min(self.checkpoints.keys(), 
                          key=lambda k: self.checkpoints[k].timestamp)
            del self.checkpoints[oldest_id]
        
        self.logger.debug(f"Created checkpoint: {checkpoint_id}")
        return checkpoint_id
    
    async def restore_from_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Restore conversation from a specific checkpoint.
        
        Args:
            checkpoint_id: ID of checkpoint to restore
            
        Returns:
            True if restoration was successful
        """
        if checkpoint_id not in self.checkpoints:
            self.logger.error(f"Checkpoint not found: {checkpoint_id}")
            return False
        
        try:
            target_state = self.checkpoints[checkpoint_id]
            
            recovery_ctx = await self.recover_conversation_state(
                target_timestamp=target_state.timestamp,
                recovery_strategy=RecoveryStrategy.IMMEDIATE,
                reason=f"checkpoint_restore_{checkpoint_id}"
            )
            
            return recovery_ctx.success
            
        except Exception as e:
            self.logger.error(f"Failed to restore from checkpoint {checkpoint_id}: {e}")
            return False
    
    async def add_message(
        self,
        participant_id: str,
        content: str,
        message_type: str = "text",
        metadata: Optional[Dict[str, Any]] = None,
        confidence: Optional[float] = None,
        is_partial: bool = False
    ) -> ConversationMessage:
        """Add a message to the conversation state."""
        message = ConversationMessage(
            id=str(uuid.uuid4()),
            timestamp=time.time(),
            participant_id=participant_id,
            content=content,
            message_type=message_type,
            metadata=metadata,
            confidence=confidence,
            is_partial=is_partial
        )
        
        async with self._state_lock:
            self.current_state.messages.append(message)
            
            # Auto-checkpoint on important messages
            if (self.config.auto_checkpoint_on_turns and 
                not is_partial and 
                message_type == "text"):
                await self.create_checkpoint()
        
        return message
    
    async def update_conversation_context(self, context_updates: Dict[str, Any]) -> None:
        """Update conversation context."""
        async with self._state_lock:
            self.current_state.conversation_context.update(context_updates)
            self.current_state.timestamp = time.time()
    
    async def set_conversation_state_type(self, state_type: ConversationStateType) -> None:
        """Update conversation state type."""
        if self.current_state.state_type != state_type:
            old_state = self.current_state.state_type
            
            async with self._state_lock:
                self.current_state.state_type = state_type
                self.current_state.timestamp = time.time()
            
            self.logger.debug(f"Conversation state: {old_state.value} -> {state_type.value}")
            
            # Notify state change
            if self._on_state_changed:
                try:
                    await self._on_state_changed(old_state, state_type)
                except Exception as e:
                    self.logger.error(f"Error in state changed callback: {e}")
    
    async def _update_current_state(self) -> None:
        """Update current state with latest information from components."""
        self.current_state.timestamp = time.time()
        
        # Update component states
        if self.turn_detector:
            self.current_state.turn_detector_state = self.turn_detector.get_current_state()
            self.current_state.current_turn = self.turn_detector.get_current_turn()
            self.current_state.turn_history = self.turn_detector.get_turn_history()
        
        if self.interruption_handler:
            self.current_state.agent_state = self.interruption_handler.get_current_state()
            self.current_state.interruption_context = self.interruption_handler.get_current_interruption()
        
        # Update LLM context if available
        if self.llm_provider:
            try:
                if hasattr(self.llm_provider, 'get_conversation_history'):
                    llm_history = await self.llm_provider.get_conversation_history()
                    if llm_history:
                        self.current_state.conversation_context['llm_history'] = llm_history
            except Exception as e:
                self.logger.warning(f"Could not get LLM history: {e}")
    
    def _create_state_snapshot(self) -> ConversationState:
        """Create a deep copy snapshot of current state."""
        # Use JSON serialization for deep copy
        state_dict = self.current_state.to_dict()
        return ConversationState.from_dict(state_dict)
    
    async def _find_recovery_target(
        self,
        target_timestamp: Optional[float],
        strategy: RecoveryStrategy
    ) -> Optional[ConversationState]:
        """Find appropriate state for recovery."""
        
        if target_timestamp:
            # Find closest state to target timestamp
            closest_state = None
            min_diff = float('inf')
            
            for state in self.state_history:
                diff = abs(state.timestamp - target_timestamp)
                if diff < min_diff:
                    min_diff = diff
                    closest_state = state
            
            return closest_state
        
        # Find based on strategy
        if strategy == RecoveryStrategy.IMMEDIATE:
            # Return most recent stable state
            for state in reversed(self.state_history):
                if state.state_type == ConversationStateType.ACTIVE:
                    return state
        
        elif strategy == RecoveryStrategy.CONTEXT_AWARE:
            # Find state with good conversation context
            for state in reversed(self.state_history):
                if (state.state_type == ConversationStateType.ACTIVE and
                    len(state.messages) > 0 and
                    state.conversation_context):
                    return state
        
        # Default to most recent state
        return self.state_history[-1] if self.state_history else None
    
    async def _execute_immediate_recovery(self, recovery_ctx: RecoveryContext) -> None:
        """Execute immediate recovery to target state."""
        target_state = recovery_ctx.target_state
        
        # Replace current state with target state
        async with self._state_lock:
            self.current_state = self._create_state_snapshot_from(target_state)
            self.current_state.timestamp = time.time()
        
        # Restore component states
        await self._restore_component_states(target_state)
        
        recovery_ctx.recovery_steps = [
            "Replaced current state",
            "Restored component states"
        ]
    
    async def _execute_gradual_recovery(self, recovery_ctx: RecoveryContext) -> None:
        """Execute gradual recovery with smooth transitions."""
        target_state = recovery_ctx.target_state
        
        steps = []
        
        # Gradually restore state elements
        async with self._state_lock:
            # First restore conversation context
            self.current_state.conversation_context = target_state.conversation_context.copy()
            steps.append("Restored conversation context")
            await asyncio.sleep(0.1)
            
            # Then restore messages
            self.current_state.messages = target_state.messages.copy()
            steps.append("Restored message history")
            await asyncio.sleep(0.1)
            
            # Finally restore state type
            self.current_state.state_type = target_state.state_type
            self.current_state.timestamp = time.time()
            steps.append("Restored conversation state")
        
        # Restore components gradually
        await self._restore_component_states(target_state)
        steps.append("Restored component states")
        
        recovery_ctx.recovery_steps = steps
    
    async def _execute_context_aware_recovery(self, recovery_ctx: RecoveryContext) -> None:
        """Execute context-aware recovery with intelligent state merging."""
        target_state = recovery_ctx.target_state
        
        steps = []
        
        async with self._state_lock:
            # Merge conversation context intelligently
            merged_context = self.current_state.conversation_context.copy()
            merged_context.update(target_state.conversation_context)
            self.current_state.conversation_context = merged_context
            steps.append("Merged conversation context")
            
            # Preserve recent messages but restore older context
            current_messages = self.current_state.messages
            target_messages = target_state.messages
            
            # Keep recent messages, restore older context
            if current_messages and target_messages:
                last_target_time = target_messages[-1].timestamp if target_messages else 0
                recent_messages = [m for m in current_messages if m.timestamp > last_target_time]
                self.current_state.messages = target_messages + recent_messages
            else:
                self.current_state.messages = target_messages
            
            steps.append("Intelligently merged message history")
            
            # Update state type if appropriate
            if target_state.state_type == ConversationStateType.ACTIVE:
                self.current_state.state_type = target_state.state_type
            
            self.current_state.timestamp = time.time()
        
        recovery_ctx.recovery_steps = steps
    
    async def _execute_smart_resume_recovery(self, recovery_ctx: RecoveryContext) -> None:
        """Execute smart resume with optimal recovery point."""
        target_state = recovery_ctx.target_state
        
        # Find optimal resume point in conversation
        resume_point = await self._find_optimal_resume_point(target_state)
        
        # Restore to resume point
        await self._execute_context_aware_recovery(recovery_ctx)
        
        recovery_ctx.recovery_steps.append(f"Smart resume from optimal point: {resume_point}")
    
    async def _prepare_user_initiated_recovery(self, recovery_ctx: RecoveryContext) -> None:
        """Prepare for user-initiated recovery."""
        # Just mark target state as prepared
        recovery_ctx.recovery_steps = [
            "Prepared target state for user-initiated recovery",
            "Waiting for user confirmation"
        ]
    
    async def _find_optimal_resume_point(self, state: ConversationState) -> str:
        """Find optimal point to resume conversation."""
        if not state.messages:
            return "conversation_start"
        
        # Look for natural break points
        for i, message in enumerate(reversed(state.messages)):
            # Check for question or complete statement
            if message.content.strip().endswith(('?', '.', '!')):
                return f"message_{len(state.messages) - i - 1}"
        
        return "last_message"
    
    async def _restore_component_states(self, target_state: ConversationState) -> None:
        """Restore component states from target state."""
        try:
            # Restore LLM context
            if self.llm_provider and 'llm_history' in target_state.conversation_context:
                if hasattr(self.llm_provider, 'set_conversation_history'):
                    await self.llm_provider.set_conversation_history(
                        target_state.conversation_context['llm_history']
                    )
            
            # Restore other component states as needed
            # This would integrate with the specific components
            
        except Exception as e:
            self.logger.warning(f"Error restoring component states: {e}")
    
    def _create_state_snapshot_from(self, source_state: ConversationState) -> ConversationState:
        """Create a new state snapshot from source state."""
        state_dict = source_state.to_dict()
        return ConversationState.from_dict(state_dict)
    
    async def _setup_component_callbacks(self) -> None:
        """Setup callbacks with integrated components."""
        if self.turn_detector:
            self.turn_detector.set_callbacks(
                on_turn_end=self._on_turn_completed,
                on_state_change=self._on_turn_state_changed
            )
        
        if self.interruption_handler:
            self.interruption_handler.set_callbacks(
                on_interruption_confirmed=self._on_interruption_occurred,
                on_agent_state_change=self._on_agent_state_changed
            )
        
        if self.flush_handler:
            self.flush_handler.set_callbacks(
                on_flush_completed=self._on_pipeline_flushed
            )
    
    async def _load_existing_state(self) -> None:
        """Load existing conversation state if available."""
        state_file = self.storage_path / f"{self.conversation_id}_latest.json"
        
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    state_data = json.load(f)
                
                loaded_state = ConversationState.from_dict(state_data)
                
                # Validate loaded state
                if loaded_state.conversation_id == self.conversation_id:
                    self.current_state = loaded_state
                    self.logger.info(f"Loaded existing conversation state from {state_file}")
                
            except Exception as e:
                self.logger.warning(f"Could not load existing state: {e}")
    
    async def _initialize_participants(self) -> None:
        """Initialize conversation participants."""
        if not self.current_state.participants:
            # Add default participants
            user_participant = ConversationParticipant(
                id="user",
                name="User",
                role="user"
            )
            agent_participant = ConversationParticipant(
                id="agent",
                name="Assistant",
                role="agent"
            )
            
            self.current_state.participants = [user_participant, agent_participant]
    
    async def _persistence_loop(self) -> None:
        """Background task for periodic state persistence."""
        while self._is_monitoring:
            try:
                await asyncio.sleep(self.config.persistence_interval)
                await self.preserve_current_state()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in persistence loop: {e}")
                await asyncio.sleep(5.0)  # Wait before retrying
    
    async def _checkpoint_loop(self) -> None:
        """Background task for automatic checkpointing."""
        while self._is_monitoring:
            try:
                await asyncio.sleep(self.config.checkpoint_interval)
                await self.create_checkpoint()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in checkpoint loop: {e}")
                await asyncio.sleep(10.0)  # Wait before retrying
    
    async def _persist_state(self, state: ConversationState) -> None:
        """Persist state to storage."""
        try:
            # Save latest state
            latest_file = self.storage_path / f"{self.conversation_id}_latest.json"
            with open(latest_file, 'w') as f:
                json.dump(state.to_dict(), f, indent=2)
            
            # Save timestamped state
            timestamp_file = self.storage_path / f"{self.conversation_id}_{int(state.timestamp)}.json"
            with open(timestamp_file, 'w') as f:
                json.dump(state.to_dict(), f, indent=2)
            
        except Exception as e:
            self.logger.error(f"Error persisting state: {e}")
    
    # Component event handlers
    async def _on_turn_completed(self, turn: TurnSegment) -> None:
        """Handle turn completion."""
        if self.config.auto_checkpoint_on_turns:
            await self.create_checkpoint(f"turn_{turn.start_time}")
    
    async def _on_turn_state_changed(self, old_state, new_state) -> None:
        """Handle turn state changes."""
        self.current_state.turn_detector_state = new_state
    
    async def _on_interruption_occurred(self, interruption: InterruptionEvent) -> None:
        """Handle interruption events."""
        self.current_state.interruption_context = interruption
        await self.set_conversation_state_type(ConversationStateType.INTERRUPTED)
    
    async def _on_agent_state_changed(self, old_state, new_state) -> None:
        """Handle agent state changes."""
        self.current_state.agent_state = new_state
    
    async def _on_pipeline_flushed(self, flush_op: FlushOperation) -> None:
        """Handle pipeline flush completion."""
        # Create recovery checkpoint after flush
        await self.create_checkpoint(f"post_flush_{flush_op.flush_id}")
    
    def set_callbacks(
        self,
        on_state_changed: Optional[Callable] = None,
        on_state_preserved: Optional[Callable] = None,
        on_recovery_started: Optional[Callable] = None,
        on_recovery_completed: Optional[Callable] = None
    ) -> None:
        """Set callback functions for state management events."""
        self._on_state_changed = on_state_changed
        self._on_state_preserved = on_state_preserved
        self._on_recovery_started = on_recovery_started
        self._on_recovery_completed = on_recovery_completed
    
    def get_current_state(self) -> ConversationState:
        """Get current conversation state."""
        return self.current_state
    
    def get_state_history(self) -> List[ConversationState]:
        """Get conversation state history."""
        return list(self.state_history)
    
    def get_recovery_history(self) -> List[RecoveryContext]:
        """Get recovery operation history."""
        return self.recovery_history.copy()
    
    def get_checkpoints(self) -> Dict[str, ConversationState]:
        """Get available checkpoints."""
        return self.checkpoints.copy()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get state management metrics."""
        total_recoveries = len(self.recovery_history)
        successful_recoveries = sum(1 for r in self.recovery_history if r.success)
        
        return {
            "session_id": self.session_id,
            "conversation_id": self.conversation_id,
            "current_state_type": self.current_state.state_type.value,
            "total_messages": len(self.current_state.messages),
            "state_history_count": len(self.state_history),
            "checkpoint_count": len(self.checkpoints),
            "total_recoveries": total_recoveries,
            "successful_recoveries": successful_recoveries,
            "recovery_success_rate": successful_recoveries / total_recoveries if total_recoveries > 0 else 0.0,
            "uptime": time.time() - self.current_state.timestamp,
            "preservation_level": self.current_state.preservation_level.value
        }
    
    async def cleanup(self) -> None:
        """Clean up state manager resources."""
        await self.stop_monitoring()
        
        # Final state preservation
        await self.preserve_current_state()
        
        self.logger.info("Conversation state manager cleanup completed")