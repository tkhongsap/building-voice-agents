"""
Voice Pipeline with Comprehensive Structured Logging

This module provides an enhanced voice pipeline that demonstrates end-to-end
correlation tracking across STT, LLM, and TTS components with structured logging.
"""

import asyncio
import time
import uuid
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from contextlib import asynccontextmanager

from ..components.stt.base_stt import BaseSTTProvider, STTResult
from ..components.llm.base_llm import BaseLLMProvider, LLMResponse
from ..components.tts.base_tts import BaseTTSProvider, TTSResult
from ..components.vad.base_vad import BaseVADProvider
from ..monitoring.structured_logging import (
    StructuredLogger,
    CorrelationIdManager,
    timed_operation
)
from ..monitoring.logging_middleware import (
    pipeline_tracker,
    livekit_middleware,
    correlation_context
)


@dataclass
class PipelineConfig:
    """Configuration for the voice pipeline."""
    enable_vad: bool = True
    enable_interruption: bool = True
    max_processing_time: float = 30.0  # seconds
    enable_streaming: bool = True
    log_intermediate_results: bool = True
    performance_tracking: bool = True


@dataclass
class PipelineMetrics:
    """Metrics for pipeline execution."""
    total_duration: float
    stt_duration: float
    llm_duration: float
    tts_duration: float
    vad_duration: float
    audio_input_size: int
    text_length: int
    audio_output_size: int
    success: bool
    error: Optional[str] = None


class VoicePipelineWithLogging:
    """
    Enhanced voice pipeline with comprehensive structured logging and correlation tracking.
    
    This pipeline orchestrates the flow: Audio -> VAD -> STT -> LLM -> TTS -> Audio
    with full correlation tracking and performance monitoring.
    """
    
    def __init__(
        self,
        stt_provider: BaseSTTProvider,
        llm_provider: BaseLLMProvider,
        tts_provider: BaseTTSProvider,
        vad_provider: Optional[BaseVADProvider] = None,
        config: Optional[PipelineConfig] = None,
        session_id: Optional[str] = None
    ):
        self.stt_provider = stt_provider
        self.llm_provider = llm_provider
        self.tts_provider = tts_provider
        self.vad_provider = vad_provider
        self.config = config or PipelineConfig()
        self.session_id = session_id or CorrelationIdManager.generate_session_id()
        
        # Initialize logger
        self.logger = StructuredLogger(__name__, "pipeline")
        
        # Set session context
        CorrelationIdManager.set_session_id(self.session_id)
        
        # Pipeline state
        self.is_initialized = False
        self.active_pipelines: Dict[str, Dict[str, Any]] = {}
        
        self.logger.info(
            "Voice pipeline initialized",
            extra_data={
                "session_id": self.session_id,
                "stt_provider": self.stt_provider.provider_name,
                "llm_provider": self.llm_provider.provider_name,
                "tts_provider": self.tts_provider.provider_name,
                "vad_enabled": self.vad_provider is not None,
                "config": {
                    "enable_vad": self.config.enable_vad,
                    "enable_interruption": self.config.enable_interruption,
                    "max_processing_time": self.config.max_processing_time,
                    "enable_streaming": self.config.enable_streaming
                }
            }
        )
    
    async def initialize(self):
        """Initialize all pipeline components."""
        if self.is_initialized:
            return
        
        self.logger.info("Initializing pipeline components")
        
        try:
            # Initialize all providers
            await self.stt_provider.initialize()
            await self.llm_provider.initialize()
            await self.tts_provider.initialize()
            
            if self.vad_provider:
                await self.vad_provider.initialize()
            
            self.is_initialized = True
            
            self.logger.info(
                "Pipeline components initialized successfully",
                extra_data={
                    "stt_provider": self.stt_provider.provider_name,
                    "llm_provider": self.llm_provider.provider_name,
                    "tts_provider": self.tts_provider.provider_name,
                    "vad_provider": self.vad_provider.provider_name if self.vad_provider else None
                }
            )
            
        except Exception as e:
            self.logger.exception(
                "Failed to initialize pipeline components",
                extra_data={
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
            raise
    
    async def cleanup(self):
        """Cleanup all pipeline components."""
        if not self.is_initialized:
            return
        
        self.logger.info("Cleaning up pipeline components")
        
        try:
            # Cleanup all providers
            await self.stt_provider.cleanup()
            await self.llm_provider.cleanup()
            await self.tts_provider.cleanup()
            
            if self.vad_provider:
                await self.vad_provider.cleanup()
            
            self.is_initialized = False
            
            self.logger.info("Pipeline components cleaned up successfully")
            
        except Exception as e:
            self.logger.exception(
                "Error during pipeline cleanup",
                extra_data={
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
    
    async def process_audio(
        self,
        audio_data: bytes,
        user_id: Optional[str] = None,
        conversation_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[TTSResult, PipelineMetrics]:
        """
        Process audio through the complete voice pipeline with full correlation tracking.
        
        Args:
            audio_data: Input audio data
            user_id: Optional user identifier
            conversation_context: Optional conversation context
            
        Returns:
            Tuple of (TTS result, pipeline metrics)
        """
        if not self.is_initialized:
            await self.initialize()
        
        # Generate pipeline ID and correlation ID
        pipeline_id = await pipeline_tracker.start_pipeline(
            self.session_id,
            f"Audio input ({len(audio_data)} bytes)",
            None
        )
        
        correlation_id = CorrelationIdManager.generate_correlation_id()
        
        # Set correlation context
        async with correlation_context(
            correlation_id=correlation_id,
            session_id=self.session_id,
            user_id=user_id,
            component="pipeline"
        ):
            self.logger.info(
                f"Starting voice pipeline processing: {pipeline_id}",
                extra_data={
                    "pipeline_id": pipeline_id,
                    "audio_size_bytes": len(audio_data),
                    "user_id": user_id,
                    "conversation_context": conversation_context
                }
            )
            
            pipeline_start_time = time.time()
            metrics = PipelineMetrics(
                total_duration=0,
                stt_duration=0,
                llm_duration=0,
                tts_duration=0,
                vad_duration=0,
                audio_input_size=len(audio_data),
                text_length=0,
                audio_output_size=0,
                success=False
            )
            
            try:
                # Step 1: Voice Activity Detection (if enabled)
                processed_audio = audio_data
                if self.config.enable_vad and self.vad_provider:
                    processed_audio, vad_duration = await self._process_vad(
                        audio_data, pipeline_id
                    )
                    metrics.vad_duration = vad_duration
                
                # Step 2: Speech-to-Text
                stt_result, stt_duration = await self._process_stt(
                    processed_audio, pipeline_id
                )
                metrics.stt_duration = stt_duration
                metrics.text_length = len(stt_result.text)
                
                # Step 3: Large Language Model Processing
                llm_response, llm_duration = await self._process_llm(
                    stt_result.text, pipeline_id, conversation_context
                )
                metrics.llm_duration = llm_duration
                
                # Step 4: Text-to-Speech
                tts_result, tts_duration = await self._process_tts(
                    llm_response.content, pipeline_id
                )
                metrics.tts_duration = tts_duration
                metrics.audio_output_size = len(tts_result.audio_data)
                
                # Calculate total duration
                metrics.total_duration = time.time() - pipeline_start_time
                metrics.success = True
                
                # Log pipeline completion
                await pipeline_tracker.end_pipeline(
                    pipeline_id,
                    final_output=llm_response.content
                )
                
                self.logger.info(
                    f"Voice pipeline completed successfully: {pipeline_id}",
                    extra_data={
                        "pipeline_id": pipeline_id,
                        "total_duration_ms": metrics.total_duration * 1000,
                        "stt_duration_ms": metrics.stt_duration * 1000,
                        "llm_duration_ms": metrics.llm_duration * 1000,
                        "tts_duration_ms": metrics.tts_duration * 1000,
                        "vad_duration_ms": metrics.vad_duration * 1000,
                        "input_audio_bytes": metrics.audio_input_size,
                        "output_audio_bytes": metrics.audio_output_size,
                        "text_length": metrics.text_length,
                        "recognized_text": stt_result.text,
                        "llm_response": llm_response.content
                    }
                )
                
                # Log performance metrics
                self.logger.log_performance_metrics(
                    "voice_pipeline",
                    {
                        "total_duration_ms": metrics.total_duration * 1000,
                        "stt_duration_ms": metrics.stt_duration * 1000,
                        "llm_duration_ms": metrics.llm_duration * 1000,
                        "tts_duration_ms": metrics.tts_duration * 1000,
                        "vad_duration_ms": metrics.vad_duration * 1000,
                        "audio_processing_ratio": metrics.total_duration / (tts_result.duration or 1),
                        "throughput_chars_per_second": metrics.text_length / metrics.total_duration if metrics.total_duration > 0 else 0,
                        "pipeline_efficiency": (metrics.stt_duration + metrics.llm_duration + metrics.tts_duration) / metrics.total_duration if metrics.total_duration > 0 else 0
                    }
                )
                
                return tts_result, metrics
                
            except Exception as e:
                metrics.total_duration = time.time() - pipeline_start_time
                metrics.success = False
                metrics.error = str(e)
                
                # Log pipeline failure
                await pipeline_tracker.end_pipeline(pipeline_id)
                
                self.logger.exception(
                    f"Voice pipeline failed: {pipeline_id}",
                    extra_data={
                        "pipeline_id": pipeline_id,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "partial_metrics": {
                            "stt_duration_ms": metrics.stt_duration * 1000,
                            "llm_duration_ms": metrics.llm_duration * 1000,
                            "tts_duration_ms": metrics.tts_duration * 1000,
                            "vad_duration_ms": metrics.vad_duration * 1000
                        }
                    }
                )
                
                raise
    
    async def _process_vad(
        self, 
        audio_data: bytes, 
        pipeline_id: str
    ) -> Tuple[bytes, float]:
        """Process Voice Activity Detection."""
        start_time = time.time()
        
        try:
            # For demonstration - actual VAD implementation would go here
            # This is a placeholder that just returns the input audio
            processed_audio = audio_data
            
            duration = time.time() - start_time
            
            self.logger.debug(
                "VAD processing completed",
                extra_data={
                    "pipeline_id": pipeline_id,
                    "input_size_bytes": len(audio_data),
                    "output_size_bytes": len(processed_audio),
                    "duration_ms": duration * 1000,
                    "speech_detected": True  # Placeholder
                }
            )
            
            return processed_audio, duration
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.exception(
                "VAD processing failed",
                extra_data={
                    "pipeline_id": pipeline_id,
                    "error": str(e),
                    "duration_ms": duration * 1000
                }
            )
            raise
    
    async def _process_stt(
        self, 
        audio_data: bytes, 
        pipeline_id: str
    ) -> Tuple[STTResult, float]:
        """Process Speech-to-Text."""
        start_time = time.time()
        
        try:
            result = await self.stt_provider.transcribe_audio_with_logging(
                audio_data, pipeline_id
            )
            
            duration = time.time() - start_time
            return result, duration
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.exception(
                "STT processing failed",
                extra_data={
                    "pipeline_id": pipeline_id,
                    "error": str(e),
                    "duration_ms": duration * 1000
                }
            )
            raise
    
    async def _process_llm(
        self, 
        text: str, 
        pipeline_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[LLMResponse, float]:
        """Process Large Language Model."""
        start_time = time.time()
        
        try:
            # Add context to the message if provided
            if context:
                enhanced_text = f"Context: {context}\n\nUser input: {text}"
            else:
                enhanced_text = text
            
            response = await self.llm_provider.chat_with_logging(
                enhanced_text, pipeline_id
            )
            
            duration = time.time() - start_time
            return response, duration
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.exception(
                "LLM processing failed",
                extra_data={
                    "pipeline_id": pipeline_id,
                    "error": str(e),
                    "input_text_length": len(text),
                    "duration_ms": duration * 1000
                }
            )
            raise
    
    async def _process_tts(
        self, 
        text: str, 
        pipeline_id: str
    ) -> Tuple[TTSResult, float]:
        """Process Text-to-Speech."""
        start_time = time.time()
        
        try:
            result = await self.tts_provider.speak_with_logging(
                text, None, pipeline_id
            )
            
            duration = time.time() - start_time
            return result, duration
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.exception(
                "TTS processing failed",
                extra_data={
                    "pipeline_id": pipeline_id,
                    "error": str(e),
                    "text_length": len(text),
                    "duration_ms": duration * 1000
                }
            )
            raise
    
    async def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and metrics."""
        return {
            "session_id": self.session_id,
            "is_initialized": self.is_initialized,
            "active_pipelines": len(self.active_pipelines),
            "components": {
                "stt": {
                    "provider": self.stt_provider.provider_name,
                    "metrics": self.stt_provider.get_metrics()
                },
                "llm": {
                    "provider": self.llm_provider.provider_name,
                    "metrics": self.llm_provider.get_metrics()
                },
                "tts": {
                    "provider": self.tts_provider.provider_name,
                    "metrics": self.tts_provider.get_metrics()
                },
                "vad": {
                    "provider": self.vad_provider.provider_name if self.vad_provider else None,
                    "metrics": self.vad_provider.get_metrics() if self.vad_provider else None
                }
            },
            "config": {
                "enable_vad": self.config.enable_vad,
                "enable_interruption": self.config.enable_interruption,
                "max_processing_time": self.config.max_processing_time,
                "enable_streaming": self.config.enable_streaming
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check of all pipeline components."""
        self.logger.info("Performing pipeline health check")
        
        health_status = {
            "pipeline": "healthy",
            "session_id": self.session_id,
            "timestamp": time.time(),
            "components": {}
        }
        
        try:
            # Check STT provider
            stt_health = await self.stt_provider.health_check()
            health_status["components"]["stt"] = stt_health
            
            # Check LLM provider
            llm_health = await self.llm_provider.health_check()
            health_status["components"]["llm"] = llm_health
            
            # Check TTS provider
            tts_health = await self.tts_provider.health_check()
            health_status["components"]["tts"] = tts_health
            
            # Check VAD provider if available
            if self.vad_provider:
                vad_health = await self.vad_provider.health_check()
                health_status["components"]["vad"] = vad_health
            
            # Determine overall health
            all_healthy = all(
                component.get("status") == "healthy"
                for component in health_status["components"].values()
            )
            
            if not all_healthy:
                health_status["pipeline"] = "degraded"
            
            self.logger.info(
                f"Pipeline health check completed: {health_status['pipeline']}",
                extra_data=health_status
            )
            
            return health_status
            
        except Exception as e:
            health_status["pipeline"] = "unhealthy"
            health_status["error"] = str(e)
            
            self.logger.exception(
                "Pipeline health check failed",
                extra_data={
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
            
            return health_status
    
    async def __aenter__(self):
        """Context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.cleanup()