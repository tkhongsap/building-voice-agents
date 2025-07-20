"""
Advanced Turn Detection Implementation

This module implements state-of-the-art turn detection using a dual-signal approach
that combines voice activity detection (VAD) with semantic analysis for accurate
turn completion prediction.
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, AsyncIterator, Callable, List, Tuple
from enum import Enum
from dataclasses import dataclass, field
import numpy as np
from collections import deque

try:
    from ..components.vad.base_vad import BaseVADProvider, VADResult, VADState
    from ..components.stt.base_stt import BaseSTTProvider, STTResult
    from ..components.llm.base_llm import BaseLLMProvider, LLMMessage, LLMRole
except ImportError:
    # Fallback for direct imports
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from components.vad.base_vad import BaseVADProvider, VADResult, VADState
    from components.stt.base_stt import BaseSTTProvider, STTResult
    from components.llm.base_llm import BaseLLMProvider, LLMMessage, LLMRole

logger = logging.getLogger(__name__)


class TurnState(Enum):
    """States of turn detection."""
    LISTENING = "listening"
    POTENTIAL_TURN = "potential_turn"
    TURN_CONFIRMED = "turn_confirmed"
    TURN_TAKING = "turn_taking"
    INTERRUPTED = "interrupted"
    UNCERTAIN = "uncertain"


class TurnEndReason(Enum):
    """Reasons for turn ending."""
    SILENCE_THRESHOLD = "silence_threshold"
    SEMANTIC_COMPLETION = "semantic_completion"
    COMBINED_SIGNALS = "combined_signals"
    INTERRUPTION = "interruption"
    TIMEOUT = "timeout"
    FORCED = "forced"


@dataclass
class TurnSegment:
    """Represents a detected turn segment."""
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    confidence: float = 0.0
    end_reason: Optional[TurnEndReason] = None
    text_content: Optional[str] = None
    is_complete: bool = False
    vad_confidence: float = 0.0
    semantic_confidence: float = 0.0
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.end_time and self.start_time:
            self.duration = self.end_time - self.start_time


@dataclass
class SemanticContext:
    """Context for semantic analysis."""
    partial_transcript: str = ""
    sentence_fragments: List[str] = field(default_factory=list)
    intent_confidence: float = 0.0
    completeness_score: float = 0.0
    question_indicators: List[str] = field(default_factory=list)
    statement_indicators: List[str] = field(default_factory=list)
    linguistic_features: Dict[str, float] = field(default_factory=dict)


@dataclass
class TurnDetectionConfig:
    """Configuration for turn detection."""
    
    # Basic thresholds
    silence_threshold_short: float = 0.5   # Short pause threshold
    silence_threshold_medium: float = 1.0  # Medium pause threshold  
    silence_threshold_long: float = 2.0    # Long pause threshold
    min_turn_duration: float = 0.3         # Minimum turn duration
    max_turn_duration: float = 30.0        # Maximum turn duration
    
    # Dual-signal weights
    vad_weight: float = 0.6               # Weight for VAD signal
    semantic_weight: float = 0.4          # Weight for semantic signal
    confidence_threshold: float = 0.7     # Combined confidence threshold
    
    # Semantic analysis settings
    enable_semantic_analysis: bool = True
    semantic_chunk_size: int = 50         # Characters for partial analysis
    intent_confidence_threshold: float = 0.6
    completeness_threshold: float = 0.7
    
    # Adaptive settings
    enable_adaptive_thresholds: bool = True
    adaptation_rate: float = 0.1          # How quickly to adapt thresholds
    context_window_size: int = 10         # Number of recent turns to consider
    
    # Performance optimization
    enable_parallel_processing: bool = True
    max_concurrent_analyses: int = 3
    analysis_timeout: float = 0.5         # Timeout for semantic analysis
    
    # Debug and monitoring
    enable_detailed_logging: bool = False
    log_confidence_scores: bool = False


class TurnDetector:
    """
    State-of-the-art turn detection using dual-signal approach.
    
    Combines voice activity detection with semantic analysis to accurately
    predict turn completion and handle conversation flow.
    """
    
    def __init__(
        self,
        vad_provider: BaseVADProvider,
        stt_provider: Optional[BaseSTTProvider] = None,
        llm_provider: Optional[BaseLLMProvider] = None,
        config: Optional[TurnDetectionConfig] = None
    ):
        self.vad_provider = vad_provider
        self.stt_provider = stt_provider
        self.llm_provider = llm_provider
        self.config = config or TurnDetectionConfig()
        
        # Turn state tracking
        self.current_state = TurnState.LISTENING
        self.current_turn: Optional[TurnSegment] = None
        self.turn_history: deque = deque(maxlen=self.config.context_window_size)
        
        # Signal tracking
        self._last_speech_time: Optional[float] = None
        self._silence_start_time: Optional[float] = None
        self._vad_confidence_buffer: deque = deque(maxlen=20)
        self._semantic_context = SemanticContext()
        
        # Adaptive thresholds
        self._adaptive_silence_threshold = self.config.silence_threshold_medium
        self._speaker_patterns: Dict[str, Dict[str, float]] = {}
        
        # Processing
        self._is_processing = False
        self._processing_lock = asyncio.Lock()
        self._analysis_tasks: List[asyncio.Task] = []
        
        # Callbacks
        self._on_turn_start: Optional[Callable] = None
        self._on_turn_end: Optional[Callable] = None
        self._on_state_change: Optional[Callable] = None
        self._on_potential_turn: Optional[Callable] = None
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def initialize(self) -> None:
        """Initialize the turn detector."""
        try:
            self.logger.info("Initializing turn detector...")
            
            # Setup VAD callbacks
            self.vad_provider.set_speech_callbacks(
                on_speech_start=self._on_vad_speech_start,
                on_speech_end=self._on_vad_speech_end
            )
            
            # Initialize semantic analysis patterns
            await self._initialize_semantic_patterns()
            
            self.logger.info("Turn detector initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize turn detector: {e}")
            raise
    
    async def start_processing(self) -> None:
        """Start turn detection processing."""
        if self._is_processing:
            self.logger.warning("Turn detector is already processing")
            return
        
        self._is_processing = True
        self.current_state = TurnState.LISTENING
        self._reset_state()
        
        self.logger.info("Turn detector started")
    
    async def stop_processing(self) -> None:
        """Stop turn detection processing."""
        if not self._is_processing:
            return
        
        self._is_processing = False
        
        # Cancel any running analysis tasks
        for task in self._analysis_tasks:
            if not task.done():
                task.cancel()
        
        # Finalize current turn if active
        if self.current_turn and not self.current_turn.is_complete:
            await self._finalize_turn(TurnEndReason.FORCED)
        
        self.logger.info("Turn detector stopped")
    
    async def process_audio_chunk(
        self, 
        audio_chunk: bytes, 
        partial_transcript: Optional[str] = None
    ) -> Optional[TurnSegment]:
        """
        Process audio chunk and update turn detection state.
        
        Args:
            audio_chunk: Raw audio bytes
            partial_transcript: Optional partial transcript from STT
            
        Returns:
            TurnSegment if turn is detected and completed
        """
        if not self._is_processing:
            return None
        
        async with self._processing_lock:
            # Get VAD result
            vad_result = await self.vad_provider.process_audio_chunk(audio_chunk)
            
            # Update semantic context if transcript available
            if partial_transcript:
                self._semantic_context.partial_transcript = partial_transcript
                await self._update_semantic_context()
            
            # Process the dual signals
            turn_segment = await self._process_dual_signals(vad_result)
            
            # Update adaptive thresholds
            if self.config.enable_adaptive_thresholds:
                await self._update_adaptive_thresholds(vad_result)
            
            return turn_segment
    
    async def _process_dual_signals(self, vad_result: VADResult) -> Optional[TurnSegment]:
        """Process VAD and semantic signals together."""
        current_time = time.time()
        
        # Update VAD confidence buffer
        self._vad_confidence_buffer.append(vad_result.confidence)
        
        # Handle speech detection
        if vad_result.is_speech:
            await self._handle_speech_detected(current_time, vad_result)
        else:
            await self._handle_silence_detected(current_time, vad_result)
        
        # Check for turn completion
        return await self._check_turn_completion(current_time)
    
    async def _handle_speech_detected(self, current_time: float, vad_result: VADResult):
        """Handle when speech is detected."""
        self._last_speech_time = current_time
        self._silence_start_time = None
        
        # Start new turn if needed
        if self.current_turn is None:
            await self._start_new_turn(current_time)
        
        # Update turn confidence with VAD signal
        if self.current_turn:
            self.current_turn.vad_confidence = vad_result.confidence
    
    async def _handle_silence_detected(self, current_time: float, vad_result: VADResult):
        """Handle when silence is detected."""
        if self._silence_start_time is None and self._last_speech_time is not None:
            self._silence_start_time = current_time
            
            # Check if this might be a turn boundary
            silence_duration = current_time - self._last_speech_time
            if silence_duration >= self.config.silence_threshold_short:
                await self._handle_potential_turn_boundary(silence_duration)
    
    async def _handle_potential_turn_boundary(self, silence_duration: float):
        """Handle potential turn boundary detection."""
        if self.current_state == TurnState.LISTENING:
            await self._set_state(TurnState.POTENTIAL_TURN)
            
            if self._on_potential_turn:
                try:
                    await self._on_potential_turn(silence_duration)
                except Exception as e:
                    self.logger.error(f"Error in potential turn callback: {e}")
    
    async def _check_turn_completion(self, current_time: float) -> Optional[TurnSegment]:
        """Check if current turn should be completed."""
        if not self.current_turn:
            return None
        
        # Calculate silence duration
        silence_duration = 0.0
        if self._silence_start_time and self._last_speech_time:
            silence_duration = current_time - self._last_speech_time
        
        # Get combined confidence score
        combined_confidence = await self._calculate_combined_confidence()
        
        # Check completion conditions
        should_complete, reason = await self._should_complete_turn(
            silence_duration, combined_confidence
        )
        
        if should_complete:
            return await self._finalize_turn(reason)
        
        return None
    
    async def _calculate_combined_confidence(self) -> float:
        """Calculate combined confidence from VAD and semantic signals."""
        # VAD confidence (from recent buffer)
        vad_conf = np.mean(list(self._vad_confidence_buffer)) if self._vad_confidence_buffer else 0.0
        
        # Semantic confidence
        semantic_conf = 0.0
        if self.config.enable_semantic_analysis:
            semantic_conf = await self._calculate_semantic_confidence()
        
        # Combined weighted confidence
        combined = (
            self.config.vad_weight * vad_conf + 
            self.config.semantic_weight * semantic_conf
        )
        
        if self.config.log_confidence_scores:
            self.logger.debug(
                f"Confidence scores - VAD: {vad_conf:.3f}, "
                f"Semantic: {semantic_conf:.3f}, Combined: {combined:.3f}"
            )
        
        return combined
    
    async def _calculate_semantic_confidence(self) -> float:
        """Calculate semantic completion confidence."""
        if not self._semantic_context.partial_transcript:
            return 0.0
        
        try:
            # Quick linguistic analysis
            text = self._semantic_context.partial_transcript.strip()
            
            if not text:
                return 0.0
            
            # Basic completeness indicators
            completeness_score = 0.0
            
            # Check for sentence endings
            if text.endswith(('.', '!', '?')):
                completeness_score += 0.4
            
            # Check for question completion
            if any(text.lower().startswith(q) for q in ['what', 'how', 'why', 'when', 'where', 'who']):
                if '?' in text or text.endswith('?'):
                    completeness_score += 0.3
                elif len(text.split()) >= 3:  # Minimum question length
                    completeness_score += 0.2
            
            # Check for statement completion
            if len(text.split()) >= 3 and any(
                text.lower().startswith(s) for s in ['i', 'the', 'this', 'that', 'it', 'we']
            ):
                completeness_score += 0.2
            
            # Check for natural pauses
            if text.endswith(','):
                completeness_score += 0.1
            
            # Length-based scoring
            word_count = len(text.split())
            if word_count >= 5:
                completeness_score += min(0.2, word_count * 0.02)
            
            # Store for context
            self._semantic_context.completeness_score = completeness_score
            
            return min(1.0, completeness_score)
            
        except Exception as e:
            self.logger.warning(f"Error calculating semantic confidence: {e}")
            return 0.0
    
    async def _should_complete_turn(
        self, 
        silence_duration: float, 
        combined_confidence: float
    ) -> Tuple[bool, TurnEndReason]:
        """Determine if turn should be completed and why."""
        
        # Timeout check
        if self.current_turn and self.current_turn.start_time:
            turn_duration = time.time() - self.current_turn.start_time
            if turn_duration >= self.config.max_turn_duration:
                return True, TurnEndReason.TIMEOUT
        
        # Silence threshold check with adaptive adjustment
        if silence_duration >= self._adaptive_silence_threshold:
            if combined_confidence >= self.config.confidence_threshold:
                return True, TurnEndReason.COMBINED_SIGNALS
            else:
                return True, TurnEndReason.SILENCE_THRESHOLD
        
        # Semantic completion check (if available)
        if (self.config.enable_semantic_analysis and 
            self._semantic_context.completeness_score >= self.config.completeness_threshold):
            if silence_duration >= self.config.silence_threshold_short:
                return True, TurnEndReason.SEMANTIC_COMPLETION
        
        # High confidence with medium pause
        if (combined_confidence >= 0.9 and 
            silence_duration >= self.config.silence_threshold_short):
            return True, TurnEndReason.COMBINED_SIGNALS
        
        return False, TurnEndReason.SILENCE_THRESHOLD
    
    async def _start_new_turn(self, start_time: float):
        """Start a new turn."""
        self.current_turn = TurnSegment(
            start_time=start_time,
            confidence=0.0,
            metadata={"detector_version": "1.0", "config": self.config.__dict__}
        )
        
        await self._set_state(TurnState.TURN_TAKING)
        
        if self._on_turn_start:
            try:
                await self._on_turn_start(self.current_turn)
            except Exception as e:
                self.logger.error(f"Error in turn start callback: {e}")
        
        self.logger.debug("New turn started")
    
    async def _finalize_turn(self, reason: TurnEndReason) -> TurnSegment:
        """Finalize the current turn."""
        if not self.current_turn:
            return None
        
        current_time = time.time()
        self.current_turn.end_time = current_time
        self.current_turn.duration = current_time - self.current_turn.start_time
        self.current_turn.end_reason = reason
        self.current_turn.is_complete = True
        self.current_turn.text_content = self._semantic_context.partial_transcript
        
        # Calculate final confidence
        self.current_turn.confidence = await self._calculate_combined_confidence()
        
        # Store in history
        self.turn_history.append(self.current_turn)
        
        await self._set_state(TurnState.LISTENING)
        
        if self._on_turn_end:
            try:
                await self._on_turn_end(self.current_turn)
            except Exception as e:
                self.logger.error(f"Error in turn end callback: {e}")
        
        self.logger.debug(
            f"Turn completed: {self.current_turn.duration:.2f}s, "
            f"reason: {reason.value}, confidence: {self.current_turn.confidence:.3f}"
        )
        
        completed_turn = self.current_turn
        self.current_turn = None
        self._reset_state()
        
        return completed_turn
    
    async def _update_adaptive_thresholds(self, vad_result: VADResult):
        """Update adaptive thresholds based on speaker patterns."""
        if not self.turn_history:
            return
        
        # Calculate average turn duration from recent history
        recent_turns = list(self.turn_history)[-5:]  # Last 5 turns
        if not recent_turns:
            return
        
        avg_duration = np.mean([turn.duration for turn in recent_turns if turn.duration])
        avg_confidence = np.mean([turn.confidence for turn in recent_turns])
        
        # Adjust silence threshold based on patterns
        if avg_confidence > 0.8:  # High confidence speaker
            adjustment_factor = 0.9  # Shorter thresholds
        elif avg_confidence < 0.5:  # Low confidence speaker
            adjustment_factor = 1.2  # Longer thresholds
        else:
            adjustment_factor = 1.0
        
        # Apply gradual adjustment
        target_threshold = self.config.silence_threshold_medium * adjustment_factor
        self._adaptive_silence_threshold += (
            (target_threshold - self._adaptive_silence_threshold) * 
            self.config.adaptation_rate
        )
        
        # Clamp to reasonable bounds
        self._adaptive_silence_threshold = max(
            self.config.silence_threshold_short,
            min(self.config.silence_threshold_long, self._adaptive_silence_threshold)
        )
    
    async def _update_semantic_context(self):
        """Update semantic context with new transcript information."""
        if not self.config.enable_semantic_analysis:
            return
        
        text = self._semantic_context.partial_transcript
        if not text:
            return
        
        # Update sentence fragments
        sentences = text.split('.')
        self._semantic_context.sentence_fragments = [s.strip() for s in sentences if s.strip()]
        
        # Identify question and statement indicators
        text_lower = text.lower()
        self._semantic_context.question_indicators = [
            word for word in ['what', 'how', 'why', 'when', 'where', 'who', 'which']
            if word in text_lower
        ]
        
        self._semantic_context.statement_indicators = [
            word for word in ['i', 'the', 'this', 'that', 'it', 'we', 'they']
            if text_lower.startswith(word)
        ]
        
        # Calculate linguistic features
        words = text.split()
        self._semantic_context.linguistic_features = {
            'word_count': len(words),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'question_mark_count': text.count('?'),
            'comma_count': text.count(','),
            'period_count': text.count('.'),
        }
    
    async def _initialize_semantic_patterns(self):
        """Initialize semantic analysis patterns."""
        # Common conversation patterns could be loaded here
        # For now, we use basic heuristics
        pass
    
    async def _set_state(self, new_state: TurnState):
        """Update turn detection state."""
        if self.current_state != new_state:
            old_state = self.current_state
            self.current_state = new_state
            
            if self.config.enable_detailed_logging:
                self.logger.debug(f"Turn state: {old_state.value} -> {new_state.value}")
            
            if self._on_state_change:
                try:
                    await self._on_state_change(old_state, new_state)
                except Exception as e:
                    self.logger.error(f"Error in state change callback: {e}")
    
    def _reset_state(self):
        """Reset internal state for new processing session."""
        self._last_speech_time = None
        self._silence_start_time = None
        self._vad_confidence_buffer.clear()
        self._semantic_context = SemanticContext()
    
    async def _on_vad_speech_start(self):
        """Handle VAD speech start event."""
        if self.config.enable_detailed_logging:
            self.logger.debug("VAD speech start detected")
    
    async def _on_vad_speech_end(self, segment):
        """Handle VAD speech end event."""
        if self.config.enable_detailed_logging:
            self.logger.debug(f"VAD speech end detected: {segment.duration:.2f}s")
    
    def set_callbacks(
        self,
        on_turn_start: Optional[Callable] = None,
        on_turn_end: Optional[Callable] = None,
        on_state_change: Optional[Callable] = None,
        on_potential_turn: Optional[Callable] = None
    ):
        """Set callback functions for turn detection events."""
        self._on_turn_start = on_turn_start
        self._on_turn_end = on_turn_end
        self._on_state_change = on_state_change
        self._on_potential_turn = on_potential_turn
    
    def get_current_state(self) -> TurnState:
        """Get current turn detection state."""
        return self.current_state
    
    def get_current_turn(self) -> Optional[TurnSegment]:
        """Get current active turn."""
        return self.current_turn
    
    def get_turn_history(self) -> List[TurnSegment]:
        """Get history of completed turns."""
        return list(self.turn_history)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get turn detection metrics."""
        total_turns = len(self.turn_history)
        avg_duration = np.mean([t.duration for t in self.turn_history if t.duration]) if self.turn_history else 0
        avg_confidence = np.mean([t.confidence for t in self.turn_history]) if self.turn_history else 0
        
        return {
            "state": self.current_state.value,
            "total_turns": total_turns,
            "average_turn_duration": avg_duration,
            "average_confidence": avg_confidence,
            "adaptive_threshold": self._adaptive_silence_threshold,
            "current_turn_active": self.current_turn is not None,
            "semantic_context": {
                "partial_transcript_length": len(self._semantic_context.partial_transcript),
                "completeness_score": self._semantic_context.completeness_score,
                "intent_confidence": self._semantic_context.intent_confidence,
            }
        }
    
    async def force_turn_completion(self) -> Optional[TurnSegment]:
        """Force completion of current turn."""
        if self.current_turn:
            return await self._finalize_turn(TurnEndReason.FORCED)
        return None
    
    async def adjust_sensitivity(self, silence_threshold: float, confidence_threshold: float):
        """Dynamically adjust detection sensitivity."""
        self.config.silence_threshold_medium = silence_threshold
        self.config.confidence_threshold = confidence_threshold
        self._adaptive_silence_threshold = silence_threshold
        
        self.logger.info(
            f"Adjusted turn detection sensitivity - "
            f"silence: {silence_threshold}, confidence: {confidence_threshold}"
        )
    
    async def cleanup(self):
        """Clean up turn detector resources."""
        await self.stop_processing()
        self.logger.info("Turn detector cleanup completed")