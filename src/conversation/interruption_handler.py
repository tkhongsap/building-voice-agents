"""
Real-Time Interruption Detection and Handling

This module implements sophisticated interruption detection during agent speech,
allowing for natural conversation flow and responsive interaction handling.
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, Callable, List, Tuple
from enum import Enum
from dataclasses import dataclass, field
import numpy as np
from collections import deque

try:
    from ..components.vad.base_vad import BaseVADProvider, VADResult, VADState
    from ..components.tts.base_tts import BaseTTSProvider
    from .turn_detector import TurnDetector, TurnSegment, TurnState
except ImportError:
    # Fallback for direct imports
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from components.vad.base_vad import BaseVADProvider, VADResult, VADState
    from components.tts.base_tts import BaseTTSProvider
    from turn_detector import TurnDetector, TurnSegment, TurnState

logger = logging.getLogger(__name__)


class InterruptionType(Enum):
    """Types of interruptions."""
    IMMEDIATE = "immediate"        # Instant interruption (high confidence)
    GRADUAL = "gradual"           # Gradual interruption building up
    OVERLAP = "overlap"           # Overlapping speech
    BARGE_IN = "barge_in"         # Clear barge-in intent
    FALSE_POSITIVE = "false_positive"  # Detected but filtered out


class InterruptionSeverity(Enum):
    """Severity levels of interruptions."""
    LOW = "low"           # Minor interruption, continue speaking
    MEDIUM = "medium"     # Moderate interruption, pause briefly
    HIGH = "high"         # Strong interruption, stop speaking
    CRITICAL = "critical" # Immediate stop required


class AgentState(Enum):
    """States of the agent during conversation."""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"
    PAUSED = "paused"
    RECOVERING = "recovering"


@dataclass
class InterruptionEvent:
    """Represents an interruption event."""
    timestamp: float
    interruption_type: InterruptionType
    severity: InterruptionSeverity
    confidence: float
    vad_confidence: float
    duration: float = 0.0
    agent_state_at_interruption: Optional[AgentState] = None
    agent_speech_position: Optional[float] = None  # Position in agent speech when interrupted
    user_audio_energy: Optional[float] = None
    context: Optional[Dict[str, Any]] = None
    is_processed: bool = False
    response_latency: Optional[float] = None  # Time to respond to interruption


@dataclass
class InterruptionHandlerConfig:
    """Configuration for interruption handling."""
    
    # Detection thresholds
    interruption_confidence_threshold: float = 0.6
    min_interruption_duration: float = 0.2
    max_false_positive_duration: float = 0.1
    
    # Agent state sensitivity
    speaking_sensitivity: float = 0.4      # Lower threshold when agent is speaking
    listening_sensitivity: float = 0.7     # Higher threshold when listening
    
    # Interruption response timing
    immediate_response_threshold: float = 0.8  # High confidence immediate stop
    gradual_response_delay: float = 0.3        # Delay for gradual interruptions
    max_response_latency: float = 0.5          # Maximum acceptable response time
    
    # Audio analysis
    energy_threshold: float = 0.3             # Audio energy threshold
    spectral_threshold: float = 0.4           # Spectral analysis threshold
    enable_echo_cancellation: bool = True     # Cancel agent's own speech
    
    # Adaptive behavior
    enable_adaptive_sensitivity: bool = True
    adaptation_window: int = 10               # Number of recent interactions to consider
    user_pattern_learning: bool = True        # Learn user interruption patterns
    
    # Recovery settings
    enable_recovery: bool = True
    recovery_pause_duration: float = 0.5     # Pause before resuming
    max_recovery_attempts: int = 3            # Maximum recovery attempts
    
    # Performance optimization
    enable_parallel_processing: bool = True
    processing_buffer_size: int = 50
    enable_prediction: bool = True            # Predict interruptions
    
    # Debug and monitoring
    enable_detailed_logging: bool = False
    log_all_events: bool = False
    monitor_response_times: bool = True


class InterruptionHandler:
    """
    Real-time interruption detection and handling system.
    
    Detects when users interrupt agent speech and handles the interruption
    gracefully with appropriate responses and recovery mechanisms.
    """
    
    def __init__(
        self,
        vad_provider: BaseVADProvider,
        tts_provider: Optional[BaseTTSProvider] = None,
        turn_detector: Optional[TurnDetector] = None,
        config: Optional[InterruptionHandlerConfig] = None
    ):
        self.vad_provider = vad_provider
        self.tts_provider = tts_provider
        self.turn_detector = turn_detector
        self.config = config or InterruptionHandlerConfig()
        
        # State tracking
        self.agent_state = AgentState.IDLE
        self.current_interruption: Optional[InterruptionEvent] = None
        self.interruption_history: deque = deque(maxlen=self.config.adaptation_window)
        
        # Speech tracking
        self._agent_speech_start: Optional[float] = None
        self._agent_speech_position: float = 0.0
        self._user_speech_start: Optional[float] = None
        self._last_vad_result: Optional[VADResult] = None
        
        # Audio analysis buffers
        self._audio_energy_buffer: deque = deque(maxlen=20)
        self._vad_confidence_buffer: deque = deque(maxlen=15)
        self._interruption_candidates: List[InterruptionEvent] = []
        
        # Adaptive parameters
        self._current_sensitivity = self.config.interruption_confidence_threshold
        self._user_interruption_patterns: Dict[str, float] = {}
        
        # Processing
        self._is_processing = False
        self._processing_lock = asyncio.Lock()
        self._monitoring_task: Optional[asyncio.Task] = None
        
        # Callbacks
        self._on_interruption_detected: Optional[Callable] = None
        self._on_interruption_confirmed: Optional[Callable] = None
        self._on_agent_state_change: Optional[Callable] = None
        self._on_recovery_complete: Optional[Callable] = None
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def initialize(self) -> None:
        """Initialize the interruption handler."""
        try:
            self.logger.info("Initializing interruption handler...")
            
            # Setup VAD callbacks for interruption detection
            self.vad_provider.set_speech_callbacks(
                on_speech_start=self._on_user_speech_start,
                on_speech_end=self._on_user_speech_end
            )
            
            # Connect to turn detector if available
            if self.turn_detector:
                self.turn_detector.set_callbacks(
                    on_state_change=self._on_turn_state_change
                )
            
            self.logger.info("Interruption handler initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize interruption handler: {e}")
            raise
    
    async def start_monitoring(self) -> None:
        """Start interruption monitoring."""
        if self._is_processing:
            self.logger.warning("Interruption handler is already monitoring")
            return
        
        self._is_processing = True
        self.agent_state = AgentState.LISTENING
        
        # Start monitoring task
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        self.logger.info("Interruption monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop interruption monitoring."""
        if not self._is_processing:
            return
        
        self._is_processing = False
        
        # Cancel monitoring task
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Finalize any active interruption
        if self.current_interruption and not self.current_interruption.is_processed:
            await self._finalize_interruption()
        
        self.logger.info("Interruption monitoring stopped")
    
    async def notify_agent_speaking(self, speech_content: str, expected_duration: float) -> None:
        """Notify handler that agent started speaking."""
        await self._set_agent_state(AgentState.SPEAKING)
        self._agent_speech_start = time.time()
        self._agent_speech_position = 0.0
        
        # Adjust sensitivity for speaking state
        self._current_sensitivity = self.config.speaking_sensitivity
        
        self.logger.debug(f"Agent started speaking: {expected_duration:.2f}s expected")
    
    async def notify_agent_stopped_speaking(self) -> None:
        """Notify handler that agent stopped speaking."""
        await self._set_agent_state(AgentState.LISTENING)
        self._agent_speech_start = None
        self._agent_speech_position = 0.0
        
        # Reset sensitivity for listening state
        self._current_sensitivity = self.config.listening_sensitivity
        
        self.logger.debug("Agent stopped speaking")
    
    async def process_audio_chunk(
        self, 
        audio_chunk: bytes,
        timestamp: Optional[float] = None
    ) -> Optional[InterruptionEvent]:
        """
        Process audio chunk for interruption detection.
        
        Args:
            audio_chunk: Raw audio bytes
            timestamp: Optional timestamp for the audio
            
        Returns:
            InterruptionEvent if interruption is detected and confirmed
        """
        if not self._is_processing:
            return None
        
        current_time = timestamp or time.time()
        
        async with self._processing_lock:
            # Get VAD result
            vad_result = await self.vad_provider.process_audio_chunk(audio_chunk)
            self._last_vad_result = vad_result
            
            # Calculate audio energy
            audio_energy = self._calculate_audio_energy(audio_chunk)
            self._audio_energy_buffer.append(audio_energy)
            self._vad_confidence_buffer.append(vad_result.confidence)
            
            # Check for interruption during agent speech
            if self.agent_state == AgentState.SPEAKING:
                interruption = await self._detect_interruption(vad_result, current_time, audio_energy)
                if interruption:
                    return await self._handle_interruption(interruption)
            
            # Update agent speech position if speaking
            if self._agent_speech_start:
                self._agent_speech_position = current_time - self._agent_speech_start
        
        return None
    
    async def _detect_interruption(
        self, 
        vad_result: VADResult, 
        timestamp: float,
        audio_energy: float
    ) -> Optional[InterruptionEvent]:
        """Detect potential interruption from VAD and audio analysis."""
        
        # Skip if no speech detected
        if not vad_result.is_speech or vad_result.state != VADState.SPEECH:
            return None
        
        # Skip if confidence too low
        if vad_result.confidence < self._current_sensitivity:
            return None
        
        # Skip if audio energy too low (might be echo)
        if audio_energy < self.config.energy_threshold:
            return None
        
        # Determine interruption type and severity
        interruption_type = self._classify_interruption_type(vad_result, audio_energy)
        severity = self._calculate_interruption_severity(vad_result, audio_energy)
        
        # Create interruption event
        interruption = InterruptionEvent(
            timestamp=timestamp,
            interruption_type=interruption_type,
            severity=severity,
            confidence=vad_result.confidence,
            vad_confidence=vad_result.confidence,
            agent_state_at_interruption=self.agent_state,
            agent_speech_position=self._agent_speech_position,
            user_audio_energy=audio_energy,
            context={
                "vad_state": vad_result.state.value,
                "agent_speaking_duration": self._agent_speech_position,
                "recent_energy_avg": np.mean(list(self._audio_energy_buffer)) if self._audio_energy_buffer else 0
            }
        )
        
        # Filter false positives
        if await self._is_false_positive(interruption):
            interruption.interruption_type = InterruptionType.FALSE_POSITIVE
            return None
        
        return interruption
    
    async def _handle_interruption(self, interruption: InterruptionEvent) -> InterruptionEvent:
        """Handle detected interruption event."""
        self.current_interruption = interruption
        
        # Log detection
        if self.config.enable_detailed_logging:
            self.logger.debug(
                f"Interruption detected: {interruption.interruption_type.value}, "
                f"severity: {interruption.severity.value}, confidence: {interruption.confidence:.3f}"
            )
        
        # Notify callbacks
        if self._on_interruption_detected:
            try:
                await self._on_interruption_detected(interruption)
            except Exception as e:
                self.logger.error(f"Error in interruption detected callback: {e}")
        
        # Handle based on severity
        if interruption.severity == InterruptionSeverity.CRITICAL:
            await self._handle_critical_interruption(interruption)
        elif interruption.severity == InterruptionSeverity.HIGH:
            await self._handle_high_interruption(interruption)
        elif interruption.severity == InterruptionSeverity.MEDIUM:
            await self._handle_medium_interruption(interruption)
        else:  # LOW severity
            await self._handle_low_interruption(interruption)
        
        return interruption
    
    async def _handle_critical_interruption(self, interruption: InterruptionEvent):
        """Handle critical interruption - immediate stop."""
        await self._set_agent_state(AgentState.INTERRUPTED)
        
        # Stop TTS immediately if available
        if self.tts_provider and hasattr(self.tts_provider, 'stop_speaking'):
            try:
                await self.tts_provider.stop_speaking()
            except Exception as e:
                self.logger.error(f"Error stopping TTS: {e}")
        
        # Record response latency
        interruption.response_latency = time.time() - interruption.timestamp
        
        await self._confirm_interruption(interruption)
    
    async def _handle_high_interruption(self, interruption: InterruptionEvent):
        """Handle high severity interruption - stop with brief delay."""
        # Brief delay to confirm
        await asyncio.sleep(0.1)
        
        # Check if still valid
        if self._is_interruption_still_valid(interruption):
            await self._set_agent_state(AgentState.INTERRUPTED)
            
            if self.tts_provider and hasattr(self.tts_provider, 'stop_speaking'):
                try:
                    await self.tts_provider.stop_speaking()
                except Exception as e:
                    self.logger.error(f"Error stopping TTS: {e}")
            
            interruption.response_latency = time.time() - interruption.timestamp
            await self._confirm_interruption(interruption)
    
    async def _handle_medium_interruption(self, interruption: InterruptionEvent):
        """Handle medium severity interruption - pause and evaluate."""
        await self._set_agent_state(AgentState.PAUSED)
        
        # Wait for gradual response delay
        await asyncio.sleep(self.config.gradual_response_delay)
        
        # Re-evaluate interruption
        if self._is_interruption_still_valid(interruption):
            await self._set_agent_state(AgentState.INTERRUPTED)
            await self._confirm_interruption(interruption)
        else:
            # Resume speaking
            await self._set_agent_state(AgentState.SPEAKING)
    
    async def _handle_low_interruption(self, interruption: InterruptionEvent):
        """Handle low severity interruption - monitor and continue."""
        # Add to candidates for monitoring
        self._interruption_candidates.append(interruption)
        
        # Continue speaking but monitor for escalation
        await self._monitor_interruption_escalation(interruption)
    
    async def _confirm_interruption(self, interruption: InterruptionEvent):
        """Confirm and finalize interruption handling."""
        interruption.is_processed = True
        interruption.duration = time.time() - interruption.timestamp
        
        # Add to history
        self.interruption_history.append(interruption)
        
        # Update adaptive parameters
        if self.config.enable_adaptive_sensitivity:
            await self._update_adaptive_parameters(interruption)
        
        # Notify confirmation callback
        if self._on_interruption_confirmed:
            try:
                await self._on_interruption_confirmed(interruption)
            except Exception as e:
                self.logger.error(f"Error in interruption confirmed callback: {e}")
        
        self.logger.debug(f"Interruption confirmed and processed: {interruption.duration:.3f}s")
        
        # Start recovery if enabled
        if self.config.enable_recovery:
            await self._start_recovery()
    
    async def _start_recovery(self):
        """Start recovery process after interruption."""
        await self._set_agent_state(AgentState.RECOVERING)
        
        # Recovery pause
        await asyncio.sleep(self.config.recovery_pause_duration)
        
        # Return to listening state
        await self._set_agent_state(AgentState.LISTENING)
        
        if self._on_recovery_complete:
            try:
                await self._on_recovery_complete()
            except Exception as e:
                self.logger.error(f"Error in recovery complete callback: {e}")
        
        self.logger.debug("Recovery completed")
    
    def _classify_interruption_type(self, vad_result: VADResult, audio_energy: float) -> InterruptionType:
        """Classify the type of interruption based on signals."""
        
        # High confidence and energy = immediate
        if vad_result.confidence > 0.9 and audio_energy > 0.7:
            return InterruptionType.IMMEDIATE
        
        # Moderate confidence with rising pattern = gradual
        if len(self._vad_confidence_buffer) >= 3:
            recent_confidences = list(self._vad_confidence_buffer)[-3:]
            if all(recent_confidences[i] <= recent_confidences[i+1] for i in range(len(recent_confidences)-1)):
                return InterruptionType.GRADUAL
        
        # High energy with moderate confidence = barge-in
        if audio_energy > 0.6 and vad_result.confidence > 0.7:
            return InterruptionType.BARGE_IN
        
        # Default to overlap
        return InterruptionType.OVERLAP
    
    def _calculate_interruption_severity(self, vad_result: VADResult, audio_energy: float) -> InterruptionSeverity:
        """Calculate interruption severity based on multiple factors."""
        
        # Base severity from confidence and energy
        confidence_score = vad_result.confidence
        energy_score = audio_energy
        
        # Combined score
        combined_score = (confidence_score + energy_score) / 2
        
        # Adjust based on agent state and speech position
        if self.agent_state == AgentState.SPEAKING:
            # Early interruption in speech is more severe
            if self._agent_speech_position < 1.0:  # First second
                combined_score *= 1.2
        
        # Determine severity
        if combined_score >= 0.9:
            return InterruptionSeverity.CRITICAL
        elif combined_score >= 0.7:
            return InterruptionSeverity.HIGH
        elif combined_score >= 0.5:
            return InterruptionSeverity.MEDIUM
        else:
            return InterruptionSeverity.LOW
    
    async def _is_false_positive(self, interruption: InterruptionEvent) -> bool:
        """Check if interruption is likely a false positive."""
        
        # Check duration - very short interruptions are likely false positives
        if interruption.duration > 0 and interruption.duration < self.config.max_false_positive_duration:
            return True
        
        # Check energy consistency
        if len(self._audio_energy_buffer) >= 3:
            recent_energies = list(self._audio_energy_buffer)[-3:]
            if np.std(recent_energies) < 0.1:  # Very consistent low energy
                return True
        
        # Check if echo cancellation indicates agent's own voice
        if self.config.enable_echo_cancellation and self.agent_state == AgentState.SPEAKING:
            # Simple echo detection - if interruption confidence is similar to background
            background_confidence = np.mean(list(self._vad_confidence_buffer)[:-1]) if len(self._vad_confidence_buffer) > 1 else 0
            if abs(interruption.confidence - background_confidence) < 0.2:
                return True
        
        return False
    
    def _is_interruption_still_valid(self, interruption: InterruptionEvent) -> bool:
        """Check if interruption is still valid after delay."""
        if not self._last_vad_result:
            return False
        
        # Check if speech is still detected
        if not self._last_vad_result.is_speech:
            return False
        
        # Check if confidence is still high enough
        if self._last_vad_result.confidence < interruption.confidence * 0.8:
            return False
        
        return True
    
    async def _monitor_interruption_escalation(self, interruption: InterruptionEvent):
        """Monitor low severity interruption for potential escalation."""
        start_time = time.time()
        
        while time.time() - start_time < 1.0:  # Monitor for 1 second
            await asyncio.sleep(0.1)
            
            if not self._last_vad_result or not self._last_vad_result.is_speech:
                break
            
            # Check for escalation
            if self._last_vad_result.confidence > interruption.confidence * 1.3:
                # Escalate to medium severity
                interruption.severity = InterruptionSeverity.MEDIUM
                await self._handle_medium_interruption(interruption)
                break
    
    def _calculate_audio_energy(self, audio_chunk: bytes) -> float:
        """Calculate normalized audio energy from raw bytes."""
        try:
            # Convert bytes to numpy array (assuming 16-bit PCM)
            audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
            
            # Calculate RMS energy
            rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
            
            # Normalize to 0-1 range
            normalized_energy = min(1.0, rms / 32767.0)
            
            return normalized_energy
            
        except Exception as e:
            self.logger.warning(f"Error calculating audio energy: {e}")
            return 0.0
    
    async def _update_adaptive_parameters(self, interruption: InterruptionEvent):
        """Update adaptive parameters based on interruption patterns."""
        if not self.interruption_history:
            return
        
        # Calculate average interruption characteristics
        recent_interruptions = list(self.interruption_history)[-5:]  # Last 5 interruptions
        avg_confidence = np.mean([i.confidence for i in recent_interruptions])
        avg_severity = np.mean([
            list(InterruptionSeverity).index(i.severity) for i in recent_interruptions
        ])
        
        # Adjust sensitivity based on patterns
        if avg_confidence > 0.8:  # User tends to interrupt with high confidence
            self._current_sensitivity *= 0.95  # Slightly lower threshold
        elif avg_confidence < 0.6:  # User tends to have lower confidence interruptions
            self._current_sensitivity *= 1.05  # Slightly higher threshold
        
        # Clamp to reasonable bounds
        self._current_sensitivity = max(0.3, min(0.9, self._current_sensitivity))
    
    async def _monitoring_loop(self):
        """Main monitoring loop for interruption handling."""
        while self._is_processing:
            try:
                await asyncio.sleep(0.1)  # Monitor every 100ms
                
                # Clean up old interruption candidates
                current_time = time.time()
                self._interruption_candidates = [
                    i for i in self._interruption_candidates 
                    if current_time - i.timestamp < 2.0  # Keep for 2 seconds
                ]
                
                # Monitor response times
                if self.config.monitor_response_times and self.current_interruption:
                    if (not self.current_interruption.is_processed and 
                        current_time - self.current_interruption.timestamp > self.config.max_response_latency):
                        
                        self.logger.warning(
                            f"Interruption response latency exceeded: "
                            f"{current_time - self.current_interruption.timestamp:.3f}s"
                        )
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(0.5)
    
    async def _finalize_interruption(self):
        """Finalize any pending interruption."""
        if self.current_interruption and not self.current_interruption.is_processed:
            self.current_interruption.is_processed = True
            self.current_interruption.duration = time.time() - self.current_interruption.timestamp
            self.interruption_history.append(self.current_interruption)
        
        self.current_interruption = None
    
    async def _set_agent_state(self, new_state: AgentState):
        """Update agent state and notify callbacks."""
        if self.agent_state != new_state:
            old_state = self.agent_state
            self.agent_state = new_state
            
            if self.config.enable_detailed_logging:
                self.logger.debug(f"Agent state: {old_state.value} -> {new_state.value}")
            
            if self._on_agent_state_change:
                try:
                    await self._on_agent_state_change(old_state, new_state)
                except Exception as e:
                    self.logger.error(f"Error in agent state change callback: {e}")
    
    async def _on_user_speech_start(self):
        """Handle user speech start from VAD."""
        self._user_speech_start = time.time()
        
        if self.config.enable_detailed_logging:
            self.logger.debug("User speech started")
    
    async def _on_user_speech_end(self, segment):
        """Handle user speech end from VAD."""
        self._user_speech_start = None
        
        if self.config.enable_detailed_logging:
            self.logger.debug(f"User speech ended: {segment.duration:.2f}s")
    
    async def _on_turn_state_change(self, old_state, new_state):
        """Handle turn detector state changes."""
        if self.config.enable_detailed_logging:
            self.logger.debug(f"Turn state changed: {old_state.value} -> {new_state.value}")
    
    def set_callbacks(
        self,
        on_interruption_detected: Optional[Callable] = None,
        on_interruption_confirmed: Optional[Callable] = None,
        on_agent_state_change: Optional[Callable] = None,
        on_recovery_complete: Optional[Callable] = None
    ):
        """Set callback functions for interruption events."""
        self._on_interruption_detected = on_interruption_detected
        self._on_interruption_confirmed = on_interruption_confirmed
        self._on_agent_state_change = on_agent_state_change
        self._on_recovery_complete = on_recovery_complete
    
    def get_current_state(self) -> AgentState:
        """Get current agent state."""
        return self.agent_state
    
    def get_current_interruption(self) -> Optional[InterruptionEvent]:
        """Get current active interruption."""
        return self.current_interruption
    
    def get_interruption_history(self) -> List[InterruptionEvent]:
        """Get history of interruptions."""
        return list(self.interruption_history)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get interruption handling metrics."""
        total_interruptions = len(self.interruption_history)
        
        if total_interruptions > 0:
            avg_confidence = np.mean([i.confidence for i in self.interruption_history])
            avg_response_latency = np.mean([
                i.response_latency for i in self.interruption_history 
                if i.response_latency is not None
            ])
            severity_distribution = {
                severity.value: sum(1 for i in self.interruption_history if i.severity == severity)
                for severity in InterruptionSeverity
            }
        else:
            avg_confidence = 0.0
            avg_response_latency = 0.0
            severity_distribution = {s.value: 0 for s in InterruptionSeverity}
        
        return {
            "agent_state": self.agent_state.value,
            "current_sensitivity": self._current_sensitivity,
            "total_interruptions": total_interruptions,
            "average_confidence": avg_confidence,
            "average_response_latency": avg_response_latency,
            "severity_distribution": severity_distribution,
            "active_interruption": self.current_interruption is not None,
            "interruption_candidates": len(self._interruption_candidates),
            "speech_position": self._agent_speech_position
        }
    
    async def force_interruption_stop(self) -> None:
        """Force stop any active interruption handling."""
        if self.current_interruption:
            await self._finalize_interruption()
        
        await self._set_agent_state(AgentState.LISTENING)
        self.logger.info("Forced interruption stop")
    
    async def adjust_sensitivity(self, new_sensitivity: float) -> None:
        """Dynamically adjust interruption sensitivity."""
        self._current_sensitivity = max(0.1, min(1.0, new_sensitivity))
        self.logger.info(f"Adjusted interruption sensitivity to {self._current_sensitivity:.3f}")
    
    async def cleanup(self) -> None:
        """Clean up interruption handler resources."""
        await self.stop_monitoring()
        self.logger.info("Interruption handler cleanup completed")