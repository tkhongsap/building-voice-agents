"""
Streaming Audio Pipeline Implementation

This module implements a low-latency streaming audio pipeline that coordinates
STT, LLM, TTS, and VAD components for real-time voice agent interactions.
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, AsyncIterator, Callable, List
from enum import Enum
from dataclasses import dataclass, field
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

try:
    from ..components.stt.base_stt import BaseSTTProvider, STTResult
    from ..components.llm.base_llm import BaseLLMProvider, LLMResponse, LLMMessage, LLMRole
    from ..components.tts.base_tts import BaseTTSProvider, TTSResult
    from ..components.vad.base_vad import BaseVADProvider, VADResult, VADState
except ImportError:
    # Fallback for direct imports
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from components.stt.base_stt import BaseSTTProvider, STTResult
    from components.llm.base_llm import BaseLLMProvider, LLMResponse, LLMMessage, LLMRole
    from components.tts.base_tts import BaseTTSProvider, TTSResult
    from components.vad.base_vad import BaseVADProvider, VADResult, VADState

logger = logging.getLogger(__name__)


class PipelineState(Enum):
    """States of the audio processing pipeline."""
    STOPPED = "stopped"
    STARTING = "starting"
    LISTENING = "listening"
    PROCESSING_STT = "processing_stt"
    PROCESSING_LLM = "processing_llm" 
    PROCESSING_TTS = "processing_tts"
    SPEAKING = "speaking"
    ERROR = "error"


class PipelineMode(Enum):
    """Operating modes for the pipeline."""
    CONTINUOUS = "continuous"     # Continuous listening and processing
    PUSH_TO_TALK = "push_to_talk" # Manual activation required
    VOICE_ACTIVATED = "voice_activated" # Activated by voice activity


@dataclass
class PipelineMetrics:
    """Metrics for pipeline performance monitoring."""
    total_latency: float = 0.0
    stt_latency: float = 0.0
    llm_latency: float = 0.0
    tts_latency: float = 0.0
    vad_latency: float = 0.0
    
    audio_chunks_processed: int = 0
    speech_segments_detected: int = 0
    responses_generated: int = 0
    errors_encountered: int = 0
    
    pipeline_start_time: Optional[float] = None
    last_interaction_time: Optional[float] = None
    
    # Performance targets
    target_latency: float = 0.236  # 236ms target latency
    
    def reset(self):
        """Reset metrics counters."""
        self.audio_chunks_processed = 0
        self.speech_segments_detected = 0
        self.responses_generated = 0
        self.errors_encountered = 0
        self.pipeline_start_time = time.time()
    
    def is_meeting_latency_target(self) -> bool:
        """Check if pipeline is meeting latency targets."""
        return self.total_latency <= self.target_latency
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_latency": self.total_latency,
            "stt_latency": self.stt_latency,
            "llm_latency": self.llm_latency,
            "tts_latency": self.tts_latency,
            "vad_latency": self.vad_latency,
            "audio_chunks_processed": self.audio_chunks_processed,
            "speech_segments_detected": self.speech_segments_detected,
            "responses_generated": self.responses_generated,
            "errors_encountered": self.errors_encountered,
            "pipeline_uptime": time.time() - self.pipeline_start_time if self.pipeline_start_time else 0,
            "meeting_latency_target": self.is_meeting_latency_target()
        }


@dataclass
class PipelineConfig:
    """Configuration for the audio pipeline."""
    
    # Audio settings
    sample_rate: int = 16000
    chunk_size: int = 1024
    buffer_size: int = 4096
    max_silence_duration: float = 2.0
    min_speech_duration: float = 0.5
    
    # Pipeline behavior
    mode: PipelineMode = PipelineMode.CONTINUOUS
    enable_interruption: bool = True
    enable_echo_cancellation: bool = True
    enable_noise_suppression: bool = True
    
    # Latency optimization
    enable_streaming_stt: bool = True
    enable_streaming_llm: bool = True
    enable_streaming_tts: bool = True
    max_concurrent_requests: int = 3
    
    # Error handling
    max_retries: int = 3
    retry_delay: float = 0.1
    fallback_enabled: bool = True
    
    # Performance monitoring
    enable_metrics: bool = True
    metrics_interval: float = 1.0
    log_performance: bool = True


class StreamingAudioPipeline:
    """
    Low-latency streaming audio pipeline for voice agent interactions.
    
    Coordinates STT, LLM, TTS, and VAD components to provide real-time
    voice conversation capabilities with minimal latency.
    """
    
    def __init__(
        self,
        stt_provider: BaseSTTProvider,
        llm_provider: BaseLLMProvider,
        tts_provider: BaseTTSProvider,
        vad_provider: BaseVADProvider,
        config: PipelineConfig = None
    ):
        self.stt_provider = stt_provider
        self.llm_provider = llm_provider
        self.tts_provider = tts_provider
        self.vad_provider = vad_provider
        self.config = config or PipelineConfig()
        
        # Pipeline state
        self.state = PipelineState.STOPPED
        self.metrics = PipelineMetrics()
        self._running = False
        self._processing_lock = asyncio.Lock()
        
        # Audio buffers and queues
        self._audio_queue = asyncio.Queue(maxsize=self.config.max_concurrent_requests * 2)
        self._speech_buffer = []
        self._current_speech_start = None
        
        # Callbacks
        self._on_state_change: Optional[Callable] = None
        self._on_speech_detected: Optional[Callable] = None
        self._on_response_ready: Optional[Callable] = None
        self._on_error: Optional[Callable] = None
        
        # Performance monitoring
        self._metrics_task: Optional[asyncio.Task] = None
        self._last_metrics_report = time.time()
        
        # Processing tasks
        self._processing_tasks: List[asyncio.Task] = []
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def initialize(self) -> None:
        """Initialize all pipeline components."""
        try:
            self.logger.info("Initializing audio pipeline components...")
            
            # Initialize all providers in parallel for speed
            await asyncio.gather(
                self.stt_provider.initialize(),
                self.llm_provider.initialize(), 
                self.tts_provider.initialize(),
                self.vad_provider.initialize()
            )
            
            # Setup VAD callbacks
            self.vad_provider.set_speech_callbacks(
                on_speech_start=self._on_speech_start,
                on_speech_end=self._on_speech_end
            )
            
            self.metrics.reset()
            self.logger.info("Audio pipeline initialized successfully")
            
        except Exception as e:
            await self._handle_error(f"Failed to initialize pipeline: {e}")
            raise
    
    async def start(self) -> None:
        """Start the audio processing pipeline."""
        if self._running:
            self.logger.warning("Pipeline is already running")
            return
        
        try:
            await self._set_state(PipelineState.STARTING)
            
            # Start all providers
            await asyncio.gather(
                self.stt_provider.start_streaming(),
                self.vad_provider.start_processing()
            )
            
            self._running = True
            
            # Start processing tasks
            self._processing_tasks = [
                asyncio.create_task(self._audio_processing_loop()),
                asyncio.create_task(self._pipeline_monitoring_loop())
            ]
            
            await self._set_state(PipelineState.LISTENING)
            self.logger.info("Audio pipeline started")
            
        except Exception as e:
            await self._handle_error(f"Failed to start pipeline: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the audio processing pipeline."""
        if not self._running:
            return
        
        self.logger.info("Stopping audio pipeline...")
        self._running = False
        
        # Cancel processing tasks
        for task in self._processing_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self._processing_tasks:
            await asyncio.gather(*self._processing_tasks, return_exceptions=True)
        
        # Stop providers
        await asyncio.gather(
            self.stt_provider.stop_streaming(),
            self.vad_provider.stop_processing(),
            return_exceptions=True
        )
        
        await self._set_state(PipelineState.STOPPED)
        self.logger.info("Audio pipeline stopped")
    
    async def process_audio_chunk(self, audio_chunk: bytes) -> None:
        """
        Process an incoming audio chunk.
        
        Args:
            audio_chunk: Raw audio bytes to process
        """
        if not self._running:
            return
        
        try:
            # Add to processing queue
            await self._audio_queue.put(audio_chunk)
            self.metrics.audio_chunks_processed += 1
            
        except asyncio.QueueFull:
            self.logger.warning("Audio queue is full, dropping chunk")
            self.metrics.errors_encountered += 1
    
    async def _audio_processing_loop(self) -> None:
        """Main audio processing loop."""
        while self._running:
            try:
                # Get audio chunk from queue
                audio_chunk = await asyncio.wait_for(
                    self._audio_queue.get(), 
                    timeout=0.1
                )
                
                # Process with VAD first
                vad_start = time.time()
                vad_result = await self.vad_provider.process_audio_chunk(audio_chunk)
                vad_latency = time.time() - vad_start
                self.metrics.vad_latency = vad_latency
                
                # Handle speech detection
                if vad_result.is_speech and vad_result.state == VADState.SPEECH:
                    await self._handle_speech_chunk(audio_chunk, vad_result)
                
                # Process with STT if we have speech buffer
                if self._speech_buffer and len(self._speech_buffer) > 0:
                    await self._process_speech_buffer()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                await self._handle_error(f"Error in audio processing loop: {e}")
                await asyncio.sleep(0.1)
    
    async def _handle_speech_chunk(self, audio_chunk: bytes, vad_result: VADResult) -> None:
        """Handle a chunk containing speech."""
        # Add to speech buffer
        self._speech_buffer.append(audio_chunk)
        
        # Track speech start time
        if self._current_speech_start is None:
            self._current_speech_start = time.time()
            await self._set_state(PipelineState.PROCESSING_STT)
    
    async def _process_speech_buffer(self) -> None:
        """Process accumulated speech buffer through STT."""
        if not self._speech_buffer:
            return
        
        async with self._processing_lock:
            try:
                # Combine speech chunks
                speech_audio = b''.join(self._speech_buffer)
                
                # Process with STT
                stt_start = time.time()
                
                if self.config.enable_streaming_stt:
                    # Stream audio to STT
                    await self.stt_provider.stream_audio(speech_audio)
                    
                    # Get results
                    async for stt_result in self.stt_provider.get_streaming_results():
                        if stt_result.is_final and stt_result.text.strip():
                            stt_latency = time.time() - stt_start
                            self.metrics.stt_latency = stt_latency
                            
                            # Process with LLM
                            await self._process_with_llm(stt_result.text)
                            break
                else:
                    # Batch STT processing
                    stt_result = await self.stt_provider.transcribe_audio(speech_audio)
                    stt_latency = time.time() - stt_start
                    self.metrics.stt_latency = stt_latency
                    
                    if stt_result.text.strip():
                        await self._process_with_llm(stt_result.text)
                
                # Clear speech buffer
                self._speech_buffer.clear()
                self._current_speech_start = None
                
            except Exception as e:
                await self._handle_error(f"Error processing speech buffer: {e}")
    
    async def _process_with_llm(self, text: str) -> None:
        """Process text with LLM and generate response."""
        try:
            await self._set_state(PipelineState.PROCESSING_LLM)
            
            llm_start = time.time()
            
            if self.config.enable_streaming_llm:
                # Streaming LLM response
                response_parts = []
                async for chunk in self.llm_provider.chat_streaming(text):
                    response_parts.append(chunk.content)
                
                llm_response = ''.join(response_parts)
            else:
                # Batch LLM processing
                llm_result = await self.llm_provider.chat(text)
                llm_response = llm_result.content
            
            llm_latency = time.time() - llm_start
            self.metrics.llm_latency = llm_latency
            
            # Process with TTS
            if llm_response.strip():
                await self._process_with_tts(llm_response)
            
        except Exception as e:
            await self._handle_error(f"Error processing with LLM: {e}")
    
    async def _process_with_tts(self, text: str) -> None:
        """Process text with TTS and output audio."""
        try:
            await self._set_state(PipelineState.PROCESSING_TTS)
            
            tts_start = time.time()
            
            if self.config.enable_streaming_tts:
                # Streaming TTS
                await self._set_state(PipelineState.SPEAKING)
                
                async for audio_chunk in self.tts_provider.speak_streaming(text):
                    # Output audio chunk
                    if self._on_response_ready:
                        await self._on_response_ready(audio_chunk)
            else:
                # Batch TTS
                tts_result = await self.tts_provider.speak(text)
                
                await self._set_state(PipelineState.SPEAKING)
                
                # Output complete audio
                if self._on_response_ready:
                    await self._on_response_ready(tts_result.audio_data)
            
            tts_latency = time.time() - tts_start
            self.metrics.tts_latency = tts_latency
            
            # Calculate total latency
            if self._current_speech_start:
                self.metrics.total_latency = time.time() - self._current_speech_start
            
            self.metrics.responses_generated += 1
            self.metrics.last_interaction_time = time.time()
            
            await self._set_state(PipelineState.LISTENING)
            
        except Exception as e:
            await self._handle_error(f"Error processing with TTS: {e}")
    
    async def _on_speech_start(self) -> None:
        """Handle VAD speech start event."""
        self.logger.debug("Speech started")
        self._current_speech_start = time.time()
        
        if self._on_speech_detected:
            await self._on_speech_detected(True)
    
    async def _on_speech_end(self, segment) -> None:
        """Handle VAD speech end event."""
        self.logger.debug(f"Speech ended: {segment.duration:.2f}s")
        self.metrics.speech_segments_detected += 1
        
        if self._on_speech_detected:
            await self._on_speech_detected(False)
    
    async def _set_state(self, new_state: PipelineState) -> None:
        """Update pipeline state and notify callbacks."""
        if self.state != new_state:
            old_state = self.state
            self.state = new_state
            
            self.logger.debug(f"Pipeline state: {old_state.value} -> {new_state.value}")
            
            if self._on_state_change:
                await self._on_state_change(old_state, new_state)
    
    async def _handle_error(self, error_message: str) -> None:
        """Handle pipeline errors."""
        self.logger.error(error_message)
        self.metrics.errors_encountered += 1
        
        await self._set_state(PipelineState.ERROR)
        
        if self._on_error:
            await self._on_error(error_message)
    
    async def _pipeline_monitoring_loop(self) -> None:
        """Monitor pipeline performance and health."""
        while self._running:
            try:
                await asyncio.sleep(self.config.metrics_interval)
                
                if self.config.log_performance:
                    # Log performance metrics periodically
                    if time.time() - self._last_metrics_report >= 10.0:  # Every 10 seconds
                        metrics_dict = self.metrics.to_dict()
                        self.logger.info(f"Pipeline metrics: {metrics_dict}")
                        self._last_metrics_report = time.time()
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
    
    def set_callbacks(
        self,
        on_state_change: Optional[Callable] = None,
        on_speech_detected: Optional[Callable] = None,
        on_response_ready: Optional[Callable] = None,
        on_error: Optional[Callable] = None
    ) -> None:
        """Set callback functions for pipeline events."""
        self._on_state_change = on_state_change
        self._on_speech_detected = on_speech_detected
        self._on_response_ready = on_response_ready
        self._on_error = on_error
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current pipeline metrics."""
        return self.metrics.to_dict()
    
    def get_state(self) -> PipelineState:
        """Get current pipeline state."""
        return self.state
    
    def is_running(self) -> bool:
        """Check if pipeline is running."""
        return self._running
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health_status = {
            "pipeline": {
                "status": "healthy" if self._running else "stopped",
                "state": self.state.value,
                "uptime": time.time() - self.metrics.pipeline_start_time if self.metrics.pipeline_start_time else 0
            }
        }
        
        # Check component health
        try:
            component_checks = await asyncio.gather(
                self.stt_provider.health_check(),
                self.llm_provider.health_check(),
                self.tts_provider.health_check(),
                self.vad_provider.health_check(),
                return_exceptions=True
            )
            
            health_status["stt"] = component_checks[0]
            health_status["llm"] = component_checks[1]
            health_status["tts"] = component_checks[2]
            health_status["vad"] = component_checks[3]
            
        except Exception as e:
            health_status["error"] = f"Health check failed: {e}"
        
        return health_status
    
    async def cleanup(self) -> None:
        """Clean up pipeline resources."""
        await self.stop()
        
        # Cleanup providers
        await asyncio.gather(
            self.stt_provider.cleanup(),
            self.llm_provider.cleanup(),
            self.tts_provider.cleanup(),
            self.vad_provider.cleanup(),
            return_exceptions=True
        )
        
        self.logger.info("Pipeline cleanup completed")
    
    async def __aenter__(self):
        """Context manager entry."""
        await self.initialize()
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.cleanup()