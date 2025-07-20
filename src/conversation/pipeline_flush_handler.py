"""
Graceful Pipeline Flushing for Interruptions

This module implements graceful pipeline flushing mechanisms that ensure
smooth interruption handling without data loss or audio artifacts.
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, Callable, List, Tuple
from enum import Enum
from dataclasses import dataclass, field
from collections import deque
import json
import numpy as np

try:
    from ..pipeline.audio_pipeline import StreamingAudioPipeline, PipelineState
    from ..components.stt.base_stt import BaseSTTProvider, STTResult
    from ..components.llm.base_llm import BaseLLMProvider, LLMResponse
    from ..components.tts.base_tts import BaseTTSProvider, TTSResult
    from .interruption_handler import InterruptionHandler, InterruptionEvent, AgentState
except ImportError:
    # Fallback for direct imports
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from pipeline.audio_pipeline import StreamingAudioPipeline, PipelineState
    from components.stt.base_stt import BaseSTTProvider, STTResult
    from components.llm.base_llm import BaseLLMProvider, LLMResponse
    from components.tts.base_tts import BaseTTSProvider, TTSResult
    from interruption_handler import InterruptionHandler, InterruptionEvent, AgentState

logger = logging.getLogger(__name__)


class FlushType(Enum):
    """Types of pipeline flushes."""
    SOFT_FLUSH = "soft_flush"           # Gentle flush, preserve partial data
    HARD_FLUSH = "hard_flush"           # Complete flush, discard all data
    SELECTIVE_FLUSH = "selective_flush" # Flush specific components only
    GRACEFUL_FLUSH = "graceful_flush"   # Wait for natural breakpoints


class FlushReason(Enum):
    """Reasons for pipeline flush."""
    USER_INTERRUPTION = "user_interruption"
    AGENT_INTERRUPTION = "agent_interruption"
    ERROR_RECOVERY = "error_recovery"
    TIMEOUT = "timeout"
    MANUAL_REQUEST = "manual_request"
    CONVERSATION_END = "conversation_end"


class ComponentState(Enum):
    """States of individual components during flush."""
    ACTIVE = "active"
    FLUSHING = "flushing"
    FLUSHED = "flushed"
    ERROR = "error"
    PRESERVING = "preserving"


@dataclass
class FlushOperation:
    """Represents a pipeline flush operation."""
    flush_id: str
    timestamp: float
    flush_type: FlushType
    reason: FlushReason
    target_components: List[str]
    preserve_data: bool = False
    completion_time: Optional[float] = None
    success: bool = False
    preserved_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    
    @property
    def duration(self) -> Optional[float]:
        if self.completion_time:
            return self.completion_time - self.timestamp
        return None


@dataclass
class ComponentFlushState:
    """State of a component during flush operation."""
    component_name: str
    state: ComponentState
    data_to_preserve: Optional[Any] = None
    flush_start_time: Optional[float] = None
    flush_completion_time: Optional[float] = None
    error_message: Optional[str] = None
    
    @property
    def flush_duration(self) -> Optional[float]:
        if self.flush_start_time and self.flush_completion_time:
            return self.flush_completion_time - self.flush_start_time
        return None


@dataclass
class PipelineFlushConfig:
    """Configuration for pipeline flushing."""
    
    # Flush timeouts
    soft_flush_timeout: float = 2.0        # Timeout for soft flush
    hard_flush_timeout: float = 0.5        # Timeout for hard flush
    graceful_flush_timeout: float = 5.0    # Timeout for graceful flush
    
    # Component-specific timeouts
    stt_flush_timeout: float = 1.0
    llm_flush_timeout: float = 2.0
    tts_flush_timeout: float = 0.5
    
    # Data preservation settings
    preserve_partial_stt: bool = True      # Preserve partial STT transcripts
    preserve_llm_context: bool = True      # Preserve LLM conversation context
    preserve_tts_queue: bool = False       # Preserve TTS synthesis queue
    
    # Graceful flush settings
    wait_for_sentence_completion: bool = True
    wait_for_word_completion: bool = True
    min_graceful_flush_duration: float = 0.1
    
    # Error handling
    max_flush_retries: int = 3
    retry_delay: float = 0.1
    fallback_to_hard_flush: bool = True
    
    # Performance optimization
    enable_parallel_flushing: bool = True
    enable_predictive_flushing: bool = True
    flush_buffer_size: int = 10
    
    # Monitoring
    enable_detailed_logging: bool = False
    log_preserved_data: bool = False
    track_flush_metrics: bool = True


class PipelineFlushHandler:
    """
    Graceful pipeline flushing handler for interruptions.
    
    Manages pipeline state during interruptions to ensure smooth transitions
    without data loss or audio artifacts.
    """
    
    def __init__(
        self,
        pipeline: StreamingAudioPipeline,
        interruption_handler: Optional[InterruptionHandler] = None,
        config: Optional[PipelineFlushConfig] = None
    ):
        self.pipeline = pipeline
        self.interruption_handler = interruption_handler
        self.config = config or PipelineFlushConfig()
        
        # Flush state tracking
        self.current_flush: Optional[FlushOperation] = None
        self.flush_history: deque = deque(maxlen=self.config.flush_buffer_size)
        self.component_states: Dict[str, ComponentFlushState] = {}
        
        # Data preservation
        self._preserved_data: Dict[str, Any] = {}
        self._partial_transcripts: List[str] = []
        self._llm_context_backup: Optional[List] = None
        self._pending_tts_queue: List[str] = []
        
        # Processing
        self._flush_lock = asyncio.Lock()
        self._active_flush_tasks: List[asyncio.Task] = []
        self._is_monitoring = False
        
        # Callbacks
        self._on_flush_started: Optional[Callable] = None
        self._on_flush_completed: Optional[Callable] = None
        self._on_data_preserved: Optional[Callable] = None
        self._on_flush_error: Optional[Callable] = None
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def initialize(self) -> None:
        """Initialize the flush handler."""
        try:
            self.logger.info("Initializing pipeline flush handler...")
            
            # Setup interruption handler callbacks if available
            if self.interruption_handler:
                self.interruption_handler.set_callbacks(
                    on_interruption_detected=self._on_interruption_detected,
                    on_interruption_confirmed=self._on_interruption_confirmed
                )
            
            # Initialize component states
            await self._initialize_component_states()
            
            self.logger.info("Pipeline flush handler initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize flush handler: {e}")
            raise
    
    async def start_monitoring(self) -> None:
        """Start monitoring for flush requirements."""
        if self._is_monitoring:
            return
        
        self._is_monitoring = True
        self.logger.info("Pipeline flush monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop monitoring and cleanup."""
        if not self._is_monitoring:
            return
        
        self._is_monitoring = False
        
        # Cancel active flush tasks
        for task in self._active_flush_tasks:
            if not task.done():
                task.cancel()
        
        # Complete any active flush
        if self.current_flush:
            await self._complete_flush_operation(self.current_flush, success=False)
        
        self.logger.info("Pipeline flush monitoring stopped")
    
    async def flush_pipeline(
        self,
        flush_type: FlushType = FlushType.SOFT_FLUSH,
        reason: FlushReason = FlushReason.MANUAL_REQUEST,
        target_components: Optional[List[str]] = None,
        preserve_data: bool = True
    ) -> FlushOperation:
        """
        Flush the pipeline with specified parameters.
        
        Args:
            flush_type: Type of flush to perform
            reason: Reason for the flush
            target_components: Specific components to flush (None = all)
            preserve_data: Whether to preserve data during flush
            
        Returns:
            FlushOperation object with results
        """
        async with self._flush_lock:
            if self.current_flush:
                self.logger.warning("Flush already in progress, queuing request")
                await self._wait_for_current_flush()
            
            # Create flush operation
            flush_op = FlushOperation(
                flush_id=f"flush_{int(time.time() * 1000)}",
                timestamp=time.time(),
                flush_type=flush_type,
                reason=reason,
                target_components=target_components or self._get_all_components(),
                preserve_data=preserve_data
            )
            
            self.current_flush = flush_op
            
            try:
                # Execute flush based on type
                if flush_type == FlushType.SOFT_FLUSH:
                    await self._execute_soft_flush(flush_op)
                elif flush_type == FlushType.HARD_FLUSH:
                    await self._execute_hard_flush(flush_op)
                elif flush_type == FlushType.SELECTIVE_FLUSH:
                    await self._execute_selective_flush(flush_op)
                elif flush_type == FlushType.GRACEFUL_FLUSH:
                    await self._execute_graceful_flush(flush_op)
                
                # Mark as successful
                flush_op.success = True
                flush_op.completion_time = time.time()
                
                self.logger.info(
                    f"Pipeline flush completed: {flush_op.flush_type.value} "
                    f"in {flush_op.duration:.3f}s"
                )
                
            except Exception as e:
                flush_op.success = False
                flush_op.error_message = str(e)
                flush_op.completion_time = time.time()
                
                self.logger.error(f"Pipeline flush failed: {e}")
                
                if self._on_flush_error:
                    try:
                        await self._on_flush_error(flush_op, e)
                    except Exception as callback_error:
                        self.logger.error(f"Error in flush error callback: {callback_error}")
                
                # Fallback to hard flush if enabled
                if (self.config.fallback_to_hard_flush and 
                    flush_type != FlushType.HARD_FLUSH):
                    self.logger.info("Falling back to hard flush")
                    return await self.flush_pipeline(
                        FlushType.HARD_FLUSH, reason, target_components, False
                    )
                
                raise
            
            finally:
                await self._complete_flush_operation(flush_op, flush_op.success)
            
            return flush_op
    
    async def _execute_soft_flush(self, flush_op: FlushOperation) -> None:
        """Execute soft flush - preserve data where possible."""
        self.logger.debug("Executing soft flush")
        
        # Preserve data first
        if flush_op.preserve_data:
            await self._preserve_pipeline_data(flush_op.target_components)
        
        # Flush components gently
        flush_tasks = []
        for component in flush_op.target_components:
            task = asyncio.create_task(
                self._soft_flush_component(component)
            )
            flush_tasks.append(task)
        
        # Wait for completion with timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*flush_tasks),
                timeout=self.config.soft_flush_timeout
            )
        except asyncio.TimeoutError:
            self.logger.warning("Soft flush timeout, cancelling remaining tasks")
            for task in flush_tasks:
                if not task.done():
                    task.cancel()
    
    async def _execute_hard_flush(self, flush_op: FlushOperation) -> None:
        """Execute hard flush - immediate termination."""
        self.logger.debug("Executing hard flush")
        
        # Hard flush all components immediately
        flush_tasks = []
        for component in flush_op.target_components:
            task = asyncio.create_task(
                self._hard_flush_component(component)
            )
            flush_tasks.append(task)
        
        # Wait with short timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*flush_tasks),
                timeout=self.config.hard_flush_timeout
            )
        except asyncio.TimeoutError:
            self.logger.warning("Hard flush timeout - forcing termination")
            # Force cancel all tasks
            for task in flush_tasks:
                task.cancel()
    
    async def _execute_selective_flush(self, flush_op: FlushOperation) -> None:
        """Execute selective flush - only specified components."""
        self.logger.debug(f"Executing selective flush for: {flush_op.target_components}")
        
        # Preserve data for non-target components if needed
        if flush_op.preserve_data:
            all_components = self._get_all_components()
            preserve_components = [c for c in all_components if c not in flush_op.target_components]
            if preserve_components:
                await self._preserve_pipeline_data(preserve_components)
        
        # Flush only target components
        for component in flush_op.target_components:
            await self._soft_flush_component(component)
    
    async def _execute_graceful_flush(self, flush_op: FlushOperation) -> None:
        """Execute graceful flush - wait for natural breakpoints."""
        self.logger.debug("Executing graceful flush")
        
        start_time = time.time()
        
        # Wait for natural breakpoints
        while time.time() - start_time < self.config.graceful_flush_timeout:
            # Check if we can flush gracefully
            if await self._can_flush_gracefully():
                break
            
            await asyncio.sleep(0.1)
        
        # Preserve data before flush
        if flush_op.preserve_data:
            await self._preserve_pipeline_data(flush_op.target_components)
        
        # Execute soft flush
        await self._execute_soft_flush(flush_op)
    
    async def _soft_flush_component(self, component_name: str) -> None:
        """Perform soft flush on a specific component."""
        state = self.component_states.get(component_name)
        if not state:
            return
        
        state.state = ComponentState.FLUSHING
        state.flush_start_time = time.time()
        
        try:
            if component_name == "stt":
                await self._flush_stt_component(soft=True)
            elif component_name == "llm":
                await self._flush_llm_component(soft=True)
            elif component_name == "tts":
                await self._flush_tts_component(soft=True)
            elif component_name == "pipeline":
                await self._flush_pipeline_component(soft=True)
            
            state.state = ComponentState.FLUSHED
            state.flush_completion_time = time.time()
            
        except Exception as e:
            state.state = ComponentState.ERROR
            state.error_message = str(e)
            self.logger.error(f"Error flushing {component_name}: {e}")
            raise
    
    async def _hard_flush_component(self, component_name: str) -> None:
        """Perform hard flush on a specific component."""
        state = self.component_states.get(component_name)
        if not state:
            return
        
        state.state = ComponentState.FLUSHING
        state.flush_start_time = time.time()
        
        try:
            if component_name == "stt":
                await self._flush_stt_component(soft=False)
            elif component_name == "llm":
                await self._flush_llm_component(soft=False)
            elif component_name == "tts":
                await self._flush_tts_component(soft=False)
            elif component_name == "pipeline":
                await self._flush_pipeline_component(soft=False)
            
            state.state = ComponentState.FLUSHED
            state.flush_completion_time = time.time()
            
        except Exception as e:
            state.state = ComponentState.ERROR
            state.error_message = str(e)
            self.logger.error(f"Error hard flushing {component_name}: {e}")
            raise
    
    async def _flush_stt_component(self, soft: bool = True) -> None:
        """Flush STT component."""
        if not hasattr(self.pipeline, 'stt_provider'):
            return
        
        stt = self.pipeline.stt_provider
        
        if soft and self.config.preserve_partial_stt:
            # Try to get partial transcript before flushing
            if hasattr(stt, 'get_partial_transcript'):
                try:
                    partial = await stt.get_partial_transcript()
                    if partial:
                        self._partial_transcripts.append(partial)
                except Exception as e:
                    self.logger.warning(f"Could not get partial transcript: {e}")
        
        # Flush STT buffers
        if hasattr(stt, 'flush_buffers'):
            await stt.flush_buffers()
        elif hasattr(stt, 'reset'):
            await stt.reset()
        
        # Clear speech buffer in pipeline
        if hasattr(self.pipeline, '_speech_buffer'):
            self.pipeline._speech_buffer.clear()
    
    async def _flush_llm_component(self, soft: bool = True) -> None:
        """Flush LLM component."""
        if not hasattr(self.pipeline, 'llm_provider'):
            return
        
        llm = self.pipeline.llm_provider
        
        if soft and self.config.preserve_llm_context:
            # Backup conversation context
            if hasattr(llm, 'get_conversation_history'):
                try:
                    self._llm_context_backup = await llm.get_conversation_history()
                except Exception as e:
                    self.logger.warning(f"Could not backup LLM context: {e}")
        
        # Cancel any ongoing LLM requests
        if hasattr(llm, 'cancel_current_request'):
            await llm.cancel_current_request()
        
        # Clear any pending requests
        if hasattr(llm, 'clear_request_queue'):
            await llm.clear_request_queue()
    
    async def _flush_tts_component(self, soft: bool = True) -> None:
        """Flush TTS component."""
        if not hasattr(self.pipeline, 'tts_provider'):
            return
        
        tts = self.pipeline.tts_provider
        
        if soft and self.config.preserve_tts_queue:
            # Backup pending TTS queue
            if hasattr(tts, 'get_pending_queue'):
                try:
                    self._pending_tts_queue = await tts.get_pending_queue()
                except Exception as e:
                    self.logger.warning(f"Could not backup TTS queue: {e}")
        
        # Stop current synthesis
        if hasattr(tts, 'stop_synthesis'):
            await tts.stop_synthesis()
        elif hasattr(tts, 'stop_speaking'):
            await tts.stop_speaking()
        
        # Clear synthesis queue
        if hasattr(tts, 'clear_queue'):
            await tts.clear_queue()
    
    async def _flush_pipeline_component(self, soft: bool = True) -> None:
        """Flush main pipeline component."""
        # Clear audio queues
        if hasattr(self.pipeline, '_audio_queue'):
            while not self.pipeline._audio_queue.empty():
                try:
                    self.pipeline._audio_queue.get_nowait()
                except:
                    break
        
        # Reset pipeline state if needed
        if not soft:
            # Force reset to listening state
            await self.pipeline._set_state(PipelineState.LISTENING)
        
        # Clear processing tasks if hard flush
        if not soft and hasattr(self.pipeline, '_processing_tasks'):
            for task in self.pipeline._processing_tasks:
                if not task.done():
                    task.cancel()
    
    async def _preserve_pipeline_data(self, components: List[str]) -> None:
        """Preserve data from specified components."""
        preserved = {}
        
        for component in components:
            try:
                if component == "stt" and self.config.preserve_partial_stt:
                    if hasattr(self.pipeline.stt_provider, 'get_partial_transcript'):
                        partial = await self.pipeline.stt_provider.get_partial_transcript()
                        if partial:
                            preserved["stt_partial"] = partial
                
                elif component == "llm" and self.config.preserve_llm_context:
                    if hasattr(self.pipeline.llm_provider, 'get_conversation_history'):
                        context = await self.pipeline.llm_provider.get_conversation_history()
                        if context:
                            preserved["llm_context"] = context
                
                elif component == "tts" and self.config.preserve_tts_queue:
                    if hasattr(self.pipeline.tts_provider, 'get_pending_queue'):
                        queue = await self.pipeline.tts_provider.get_pending_queue()
                        if queue:
                            preserved["tts_queue"] = queue
                            
            except Exception as e:
                self.logger.warning(f"Error preserving data for {component}: {e}")
        
        if preserved:
            self._preserved_data.update(preserved)
            
            if self.config.log_preserved_data:
                self.logger.debug(f"Preserved data: {list(preserved.keys())}")
            
            if self._on_data_preserved:
                try:
                    await self._on_data_preserved(preserved)
                except Exception as e:
                    self.logger.error(f"Error in data preserved callback: {e}")
    
    async def restore_preserved_data(self) -> Dict[str, Any]:
        """Restore previously preserved data."""
        if not self._preserved_data:
            return {}
        
        restored = {}
        
        try:
            # Restore STT partial transcript
            if "stt_partial" in self._preserved_data:
                partial = self._preserved_data["stt_partial"]
                if hasattr(self.pipeline.stt_provider, 'set_partial_transcript'):
                    await self.pipeline.stt_provider.set_partial_transcript(partial)
                    restored["stt_partial"] = partial
            
            # Restore LLM context
            if "llm_context" in self._preserved_data:
                context = self._preserved_data["llm_context"]
                if hasattr(self.pipeline.llm_provider, 'set_conversation_history'):
                    await self.pipeline.llm_provider.set_conversation_history(context)
                    restored["llm_context"] = len(context) if isinstance(context, list) else str(context)
            
            # Restore TTS queue
            if "tts_queue" in self._preserved_data:
                queue = self._preserved_data["tts_queue"]
                if hasattr(self.pipeline.tts_provider, 'restore_queue'):
                    await self.pipeline.tts_provider.restore_queue(queue)
                    restored["tts_queue"] = len(queue) if isinstance(queue, list) else str(queue)
            
            self.logger.info(f"Restored preserved data: {list(restored.keys())}")
            
        except Exception as e:
            self.logger.error(f"Error restoring preserved data: {e}")
        
        finally:
            # Clear preserved data after restoration
            self._preserved_data.clear()
        
        return restored
    
    async def _can_flush_gracefully(self) -> bool:
        """Check if pipeline can be flushed gracefully."""
        # Check if we're at a natural breakpoint
        pipeline_state = self.pipeline.get_state()
        
        # Good times to flush
        if pipeline_state in [PipelineState.LISTENING, PipelineState.STOPPED]:
            return True
        
        # Check for sentence completion if enabled
        if self.config.wait_for_sentence_completion:
            if hasattr(self.pipeline.stt_provider, 'get_partial_transcript'):
                try:
                    partial = await self.pipeline.stt_provider.get_partial_transcript()
                    if partial and (partial.endswith('.') or partial.endswith('?') or partial.endswith('!')):
                        return True
                except:
                    pass
        
        # Check for word completion if enabled
        if self.config.wait_for_word_completion:
            # Simple heuristic - if we've been in the same state for a short time
            return True  # Simplified for now
        
        return False
    
    async def _wait_for_current_flush(self) -> None:
        """Wait for current flush operation to complete."""
        if not self.current_flush:
            return
        
        start_time = time.time()
        while self.current_flush and time.time() - start_time < 10.0:  # 10 second timeout
            await asyncio.sleep(0.1)
    
    async def _complete_flush_operation(self, flush_op: FlushOperation, success: bool) -> None:
        """Complete and finalize flush operation."""
        flush_op.success = success
        if not flush_op.completion_time:
            flush_op.completion_time = time.time()
        
        # Add to history
        self.flush_history.append(flush_op)
        
        # Reset current flush
        self.current_flush = None
        
        # Reset component states
        for state in self.component_states.values():
            if state.state == ComponentState.FLUSHING:
                state.state = ComponentState.ACTIVE
        
        # Notify completion
        if self._on_flush_completed:
            try:
                await self._on_flush_completed(flush_op)
            except Exception as e:
                self.logger.error(f"Error in flush completed callback: {e}")
    
    async def _initialize_component_states(self) -> None:
        """Initialize component states."""
        components = self._get_all_components()
        for component in components:
            self.component_states[component] = ComponentFlushState(
                component_name=component,
                state=ComponentState.ACTIVE
            )
    
    def _get_all_components(self) -> List[str]:
        """Get list of all flushable components."""
        return ["stt", "llm", "tts", "pipeline"]
    
    async def _on_interruption_detected(self, interruption: InterruptionEvent) -> None:
        """Handle interruption detection."""
        if not self._is_monitoring:
            return
        
        # Determine flush type based on interruption severity
        if interruption.severity.value in ["critical", "high"]:
            flush_type = FlushType.HARD_FLUSH
        else:
            flush_type = FlushType.SOFT_FLUSH
        
        self.logger.debug(f"Interruption detected, preparing {flush_type.value}")
        
        # Don't flush immediately - wait for confirmation
        # This prevents false positive flushes
    
    async def _on_interruption_confirmed(self, interruption: InterruptionEvent) -> None:
        """Handle confirmed interruption."""
        if not self._is_monitoring:
            return
        
        # Determine flush parameters
        if interruption.severity.value == "critical":
            flush_type = FlushType.HARD_FLUSH
            preserve_data = False
        elif interruption.severity.value == "high":
            flush_type = FlushType.SOFT_FLUSH
            preserve_data = True
        else:
            flush_type = FlushType.GRACEFUL_FLUSH
            preserve_data = True
        
        try:
            # Execute flush
            await self.flush_pipeline(
                flush_type=flush_type,
                reason=FlushReason.USER_INTERRUPTION,
                preserve_data=preserve_data
            )
            
        except Exception as e:
            self.logger.error(f"Error during interruption flush: {e}")
    
    def set_callbacks(
        self,
        on_flush_started: Optional[Callable] = None,
        on_flush_completed: Optional[Callable] = None,
        on_data_preserved: Optional[Callable] = None,
        on_flush_error: Optional[Callable] = None
    ) -> None:
        """Set callback functions for flush events."""
        self._on_flush_started = on_flush_started
        self._on_flush_completed = on_flush_completed
        self._on_data_preserved = on_data_preserved
        self._on_flush_error = on_flush_error
    
    def get_current_flush(self) -> Optional[FlushOperation]:
        """Get current active flush operation."""
        return self.current_flush
    
    def get_flush_history(self) -> List[FlushOperation]:
        """Get history of flush operations."""
        return list(self.flush_history)
    
    def get_component_states(self) -> Dict[str, ComponentFlushState]:
        """Get current component states."""
        return self.component_states.copy()
    
    def get_preserved_data(self) -> Dict[str, Any]:
        """Get currently preserved data."""
        return self._preserved_data.copy()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get flush handler metrics."""
        total_flushes = len(self.flush_history)
        successful_flushes = sum(1 for f in self.flush_history if f.success)
        
        if total_flushes > 0:
            avg_duration = np.mean([f.duration for f in self.flush_history if f.duration])
            flush_types = {t.value: 0 for t in FlushType}
            for f in self.flush_history:
                flush_types[f.flush_type.value] += 1
        else:
            avg_duration = 0.0
            flush_types = {t.value: 0 for t in FlushType}
        
        return {
            "total_flushes": total_flushes,
            "successful_flushes": successful_flushes,
            "success_rate": successful_flushes / total_flushes if total_flushes > 0 else 0.0,
            "average_flush_duration": avg_duration,
            "flush_type_distribution": flush_types,
            "active_flush": self.current_flush is not None,
            "preserved_data_count": len(self._preserved_data),
            "component_states": {
                name: state.state.value 
                for name, state in self.component_states.items()
            }
        }
    
    async def emergency_flush(self) -> FlushOperation:
        """Perform emergency hard flush of entire pipeline."""
        return await self.flush_pipeline(
            flush_type=FlushType.HARD_FLUSH,
            reason=FlushReason.ERROR_RECOVERY,
            preserve_data=False
        )
    
    async def cleanup(self) -> None:
        """Clean up flush handler resources."""
        await self.stop_monitoring()
        
        # Clear preserved data
        self._preserved_data.clear()
        self._partial_transcripts.clear()
        self._pending_tts_queue.clear()
        
        self.logger.info("Pipeline flush handler cleanup completed")