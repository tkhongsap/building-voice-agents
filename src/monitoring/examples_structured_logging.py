"""
Examples and Usage Guide for Structured Logging System

This module provides comprehensive examples demonstrating how to use
the structured logging system across different components and scenarios.
"""

import asyncio
import time
from typing import Optional, Dict, Any
from pathlib import Path

# Import structured logging components
from .structured_logging import (
    StructuredLogger,
    CorrelationIdManager,
    log_with_correlation,
    timed_operation
)
from .logging_config import (
    setup_logging,
    get_logger,
    LoggingConfig,
    LogFormat,
    LogOutput
)
from .logging_middleware import (
    correlation_context,
    livekit_middleware,
    pipeline_tracker
)

# Mock components for examples
from ..components.stt.base_stt import BaseSTTProvider, STTResult, STTConfig
from ..components.llm.base_llm import BaseLLMProvider, LLMResponse, LLMConfig
from ..components.tts.base_tts import BaseTTSProvider, TTSResult, TTSConfig


class ExampleSTTProvider(BaseSTTProvider):
    """Example STT provider for demonstration."""
    
    @property
    def provider_name(self) -> str:
        return "example_stt"
    
    @property
    def supported_languages(self) -> list:
        return ["en"]
    
    @property
    def supported_sample_rates(self) -> list:
        return [16000, 44100]
    
    async def initialize(self) -> None:
        self.logger.info("STT provider initialized")
    
    async def cleanup(self) -> None:
        self.logger.info("STT provider cleaned up")
    
    async def transcribe_audio(self, audio_data: bytes) -> STTResult:
        # Simulate processing time
        await asyncio.sleep(0.1)
        return STTResult(
            text="Hello, this is a transcribed message",
            confidence=0.95,
            is_final=True
        )
    
    async def start_streaming(self) -> None:
        pass
    
    async def stop_streaming(self) -> None:
        pass
    
    async def stream_audio(self, audio_chunk: bytes) -> None:
        pass
    
    async def get_streaming_results(self):
        pass


class ExampleLLMProvider(BaseLLMProvider):
    """Example LLM provider for demonstration."""
    
    @property
    def provider_name(self) -> str:
        return "example_llm"
    
    @property
    def supported_models(self) -> list:
        return ["example-model"]
    
    @property
    def supports_function_calling(self) -> bool:
        return False
    
    @property
    def supports_streaming(self) -> bool:
        return True
    
    async def initialize(self) -> None:
        self.logger.info("LLM provider initialized")
    
    async def cleanup(self) -> None:
        self.logger.info("LLM provider cleaned up")
    
    async def generate_response(self, messages, functions=None) -> LLMResponse:
        # Simulate processing time
        await asyncio.sleep(0.2)
        return LLMResponse(
            content="This is a response from the LLM",
            model="example-model",
            usage={"total_tokens": 50}
        )
    
    async def generate_streaming_response(self, messages, functions=None):
        # Simulate streaming response
        chunks = ["This ", "is ", "a ", "streaming ", "response"]
        for chunk in chunks:
            await asyncio.sleep(0.05)
            yield LLMResponse(content=chunk, is_streaming=True)


class ExampleTTSProvider(BaseTTSProvider):
    """Example TTS provider for demonstration."""
    
    @property
    def provider_name(self) -> str:
        return "example_tts"
    
    @property
    def supported_languages(self) -> list:
        return ["en"]
    
    @property
    def supported_formats(self) -> list:
        return ["wav", "mp3"]
    
    @property
    def supported_sample_rates(self) -> list:
        return [22050, 44100]
    
    async def initialize(self) -> None:
        self.logger.info("TTS provider initialized")
    
    async def cleanup(self) -> None:
        self.logger.info("TTS provider cleaned up")
    
    async def get_voices(self, language=None) -> list:
        return []
    
    async def synthesize_speech(self, text: str, voice_id=None) -> TTSResult:
        # Simulate processing time
        await asyncio.sleep(0.15)
        return TTSResult(
            audio_data=b"fake_audio_data" * 100,
            format="wav",
            sample_rate=22050,
            duration=2.5,
            text=text
        )
    
    async def synthesize_streaming(self, text: str, voice_id=None):
        # Simulate streaming audio chunks
        for i in range(5):
            await asyncio.sleep(0.03)
            yield b"audio_chunk_" + str(i).encode()


async def example_basic_structured_logging():
    """
    Example 1: Basic structured logging setup and usage.
    """
    print("\n=== Example 1: Basic Structured Logging ===")
    
    # Configure structured logging
    config = setup_logging(
        level="DEBUG",
        format_type=LogFormat.JSON,
        output=LogOutput.CONSOLE,
        correlation_tracking=True
    )
    
    print(f"Configured logging: {config.format_type.value} format, {config.output.value} output")
    
    # Create a structured logger
    logger = get_logger("example.basic", component="demo")
    
    # Set correlation context
    correlation_id = CorrelationIdManager.set_correlation_id()
    session_id = CorrelationIdManager.set_session_id()
    
    print(f"Generated correlation_id: {correlation_id}")
    print(f"Generated session_id: {session_id}")
    
    # Basic logging
    logger.info("Starting basic logging example")
    logger.debug("Debug information", extra_data={"step": 1, "data": "sample"})
    logger.warning("This is a warning", extra_data={"level": "moderate"})
    
    # Log with performance metrics
    logger.log_performance_metrics("demo_operation", {
        "duration_ms": 150,
        "success_rate": 0.95,
        "throughput": 100
    })
    
    # Log operation lifecycle
    logger.log_operation_start("example_operation", {"input": "test_data"})
    await asyncio.sleep(0.1)  # Simulate work
    logger.log_operation_end("example_operation", 0.1, True, {"output": "result"})
    
    # Error logging
    try:
        raise ValueError("Example error for demonstration")
    except ValueError as e:
        logger.exception("Caught an error", extra_data={"error_code": "DEMO_001"})
    
    print("Basic logging example completed\n")


async def example_correlation_tracking():
    """
    Example 2: Correlation ID tracking across async operations.
    """
    print("\n=== Example 2: Correlation Tracking ===")
    
    # Setup logging
    setup_logging(format_type=LogFormat.JSON, output=LogOutput.CONSOLE)
    logger = get_logger("example.correlation", component="correlation_demo")
    
    @log_with_correlation(
        correlation_id="demo-correlation-123",
        user_id="user-456",
        component="async_operation"
    )
    async def async_operation_1():
        op_logger = get_logger("example.operation1", component="operation1")
        op_logger.info("Executing async operation 1")
        await asyncio.sleep(0.1)
        return "result_1"
    
    @log_with_correlation(component="async_operation")
    async def async_operation_2():
        op_logger = get_logger("example.operation2", component="operation2")
        op_logger.info("Executing async operation 2")
        await asyncio.sleep(0.1)
        return "result_2"
    
    logger.info("Starting correlation tracking example")
    
    # Operations will inherit correlation context
    result1 = await async_operation_1()
    result2 = await async_operation_2()
    
    logger.info("Correlation tracking example completed", extra_data={
        "result1": result1,
        "result2": result2
    })
    
    print("Correlation tracking example completed\n")


async def example_middleware_integration():
    """
    Example 3: Middleware integration for session and pipeline tracking.
    """
    print("\n=== Example 3: Middleware Integration ===")
    
    # Setup logging
    setup_logging(format_type=LogFormat.JSON, output=LogOutput.CONSOLE)
    logger = get_logger("example.middleware", component="middleware_demo")
    
    # Simulate LiveKit session
    session_id = await livekit_middleware.session_start(
        room_name="demo_room",
        participant_identity="demo_user",
        session_metadata={"demo": True, "version": "1.0"}
    )
    
    logger.info(f"Started LiveKit session: {session_id}")
    
    # Simulate pipeline processing
    pipeline_id = await pipeline_tracker.start_pipeline(
        session_id,
        "User said: Hello world"
    )
    
    logger.info(f"Started pipeline: {pipeline_id}")
    
    # Simulate STT stage
    await pipeline_tracker.log_stage_start(
        pipeline_id, "stt", "openai_whisper", {"audio_duration": 2.5}
    )
    await asyncio.sleep(0.1)  # Simulate processing
    await pipeline_tracker.log_stage_end(
        pipeline_id, "stt", True, {"text": "Hello world", "confidence": 0.95}
    )
    
    # Simulate LLM stage
    await pipeline_tracker.log_stage_start(
        pipeline_id, "llm", "gpt-4", {"input_tokens": 10}
    )
    await asyncio.sleep(0.2)  # Simulate processing
    await pipeline_tracker.log_stage_end(
        pipeline_id, "llm", True, {"response": "Hello! How can I help you?", "output_tokens": 15}
    )
    
    # Simulate TTS stage
    await pipeline_tracker.log_stage_start(
        pipeline_id, "tts", "elevenlabs", {"text_length": 25}
    )
    await asyncio.sleep(0.15)  # Simulate processing
    await pipeline_tracker.log_stage_end(
        pipeline_id, "tts", True, {"audio_duration": 3.0, "audio_size": 48000}
    )
    
    # End pipeline and session
    await pipeline_tracker.end_pipeline(pipeline_id, "Hello! How can I help you?")
    await livekit_middleware.session_end(session_id, "completed")
    
    logger.info("Middleware integration example completed")
    print("Middleware integration example completed\n")


async def example_voice_pipeline_integration():
    """
    Example 4: Full voice pipeline with structured logging.
    """
    print("\n=== Example 4: Voice Pipeline Integration ===")
    
    # Setup logging
    setup_logging(format_type=LogFormat.JSON, output=LogOutput.CONSOLE)
    logger = get_logger("example.pipeline", component="pipeline_demo")
    
    # Create example providers
    stt_provider = ExampleSTTProvider(STTConfig())
    llm_provider = ExampleLLMProvider(LLMConfig())
    tts_provider = ExampleTTSProvider(TTSConfig())
    
    # Initialize providers
    await stt_provider.initialize()
    await llm_provider.initialize()
    await tts_provider.initialize()
    
    logger.info("Initialized voice pipeline providers")
    
    # Create correlation context for the pipeline
    async with correlation_context(
        correlation_id=CorrelationIdManager.generate_correlation_id(),
        session_id=CorrelationIdManager.generate_session_id(),
        user_id="demo_user",
        component="voice_pipeline"
    ):
        # Start pipeline
        pipeline_id = await pipeline_tracker.start_pipeline(
            CorrelationIdManager.get_all_context()["session_id"],
            "Demo audio input"
        )
        
        logger.info(f"Starting voice pipeline: {pipeline_id}")
        
        # Simulate audio data
        fake_audio = b"fake_audio_data" * 100
        
        try:
            # STT Processing
            logger.info("Processing STT stage")
            stt_result = await stt_provider.transcribe_audio_with_logging(
                fake_audio, pipeline_id
            )
            
            # LLM Processing
            logger.info("Processing LLM stage")
            llm_response = await llm_provider.chat_with_logging(
                stt_result.text, pipeline_id
            )
            
            # TTS Processing
            logger.info("Processing TTS stage")
            tts_result = await tts_provider.speak_with_logging(
                llm_response.content, None, pipeline_id
            )
            
            # Complete pipeline
            await pipeline_tracker.end_pipeline(
                pipeline_id, llm_response.content
            )
            
            logger.info("Voice pipeline completed successfully", extra_data={
                "stt_text": stt_result.text,
                "llm_response": llm_response.content,
                "tts_audio_size": len(tts_result.audio_data)
            })
            
        except Exception as e:
            logger.exception("Voice pipeline failed", extra_data={
                "pipeline_id": pipeline_id,
                "error": str(e)
            })
            await pipeline_tracker.end_pipeline(pipeline_id)
    
    # Cleanup providers
    await stt_provider.cleanup()
    await llm_provider.cleanup()
    await tts_provider.cleanup()
    
    print("Voice pipeline integration example completed\n")


async def example_performance_monitoring():
    """
    Example 5: Performance monitoring and metrics collection.
    """
    print("\n=== Example 5: Performance Monitoring ===")
    
    # Setup logging with performance tracking
    setup_logging(
        format_type=LogFormat.JSON,
        output=LogOutput.CONSOLE,
        performance_tracking=True
    )
    logger = get_logger("example.performance", component="perf_demo")
    
    @timed_operation(operation_name="cpu_intensive_task", component="performance")
    async def cpu_intensive_task(size: int):
        """Simulate CPU-intensive work."""
        # Simulate some computation
        await asyncio.sleep(0.1 * size)
        return f"Processed {size} items"
    
    @timed_operation(operation_name="io_intensive_task", component="performance")
    async def io_intensive_task(delay: float):
        """Simulate I/O-intensive work."""
        await asyncio.sleep(delay)
        return f"Completed I/O operation in {delay}s"
    
    logger.info("Starting performance monitoring example")
    
    # Run tasks with automatic timing
    tasks = [
        cpu_intensive_task(1),
        cpu_intensive_task(2),
        cpu_intensive_task(3),
        io_intensive_task(0.05),
        io_intensive_task(0.1),
        io_intensive_task(0.15)
    ]
    
    start_time = time.time()
    results = await asyncio.gather(*tasks)
    total_duration = time.time() - start_time
    
    # Log overall performance metrics
    logger.log_performance_metrics("batch_processing", {
        "total_duration_ms": total_duration * 1000,
        "tasks_completed": len(results),
        "average_task_duration_ms": (total_duration / len(results)) * 1000,
        "throughput_tasks_per_second": len(results) / total_duration
    })
    
    logger.info("Performance monitoring example completed", extra_data={
        "results": results,
        "total_duration": total_duration
    })
    
    print("Performance monitoring example completed\n")


async def example_error_handling_and_recovery():
    """
    Example 6: Error handling and recovery with structured logging.
    """
    print("\n=== Example 6: Error Handling and Recovery ===")
    
    # Setup logging
    setup_logging(format_type=LogFormat.JSON, output=LogOutput.CONSOLE)
    logger = get_logger("example.error_handling", component="error_demo")
    
    async def failing_operation(should_fail: bool = True):
        """Operation that may fail."""
        logger.info("Starting operation", extra_data={"will_fail": should_fail})
        
        if should_fail:
            raise RuntimeError("Simulated operation failure")
        
        return "Operation succeeded"
    
    async def resilient_operation_with_retry(max_retries: int = 3):
        """Operation with retry logic and structured logging."""
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}/{max_retries}")
                
                # Simulate success on final attempt
                should_fail = attempt < max_retries - 1
                result = await failing_operation(should_fail)
                
                logger.info("Operation succeeded", extra_data={
                    "attempt": attempt + 1,
                    "result": result
                })
                return result
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed", extra_data={
                    "attempt": attempt + 1,
                    "error": str(e),
                    "remaining_attempts": max_retries - attempt - 1
                })
                
                if attempt == max_retries - 1:
                    logger.error("All retry attempts failed", extra_data={
                        "total_attempts": max_retries,
                        "final_error": str(e)
                    })
                    raise
                
                # Wait before retry
                await asyncio.sleep(0.1 * (attempt + 1))
    
    logger.info("Starting error handling example")
    
    # Demonstrate successful recovery
    try:
        result = await resilient_operation_with_retry(3)
        logger.info("Resilient operation completed", extra_data={"result": result})
    except Exception as e:
        logger.exception("Resilient operation ultimately failed")
    
    # Demonstrate failure tracking
    try:
        await failing_operation(True)
    except Exception as e:
        logger.exception("Expected failure occurred", extra_data={
            "error_code": "DEMO_FAIL_001",
            "recovery_action": "Log and continue"
        })
    
    logger.info("Error handling example completed")
    print("Error handling example completed\n")


async def run_all_examples():
    """Run all structured logging examples."""
    print("🚀 Running Structured Logging Examples")
    print("=" * 50)
    
    examples = [
        example_basic_structured_logging,
        example_correlation_tracking,
        example_middleware_integration,
        example_voice_pipeline_integration,
        example_performance_monitoring,
        example_error_handling_and_recovery
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n📋 Running Example {i}: {example.__name__}")
        try:
            await example()
            print(f"✅ Example {i} completed successfully")
        except Exception as e:
            print(f"❌ Example {i} failed: {e}")
    
    print("\n🎉 All examples completed!")
    print("=" * 50)


def print_configuration_guide():
    """Print configuration guide for structured logging."""
    print("\n📖 Configuration Guide for Structured Logging")
    print("=" * 50)
    
    guide = """
1. Basic Setup:
   ```python
   from src.monitoring.logging_config import setup_logging, LogFormat, LogOutput
   
   # Console JSON logging
   setup_logging(level="INFO", format_type=LogFormat.JSON, output=LogOutput.CONSOLE)
   
   # File logging with rotation
   setup_logging(
       level="DEBUG",
       format_type=LogFormat.JSON,
       output=LogOutput.FILE,
       log_file="logs/voice_agents.log",
       max_file_size=10*1024*1024,  # 10MB
       backup_count=5
   )
   ```

2. Environment Variables:
   - LOG_LEVEL: Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
   - LOG_FORMAT: Set format (json, plain, colored)
   - LOG_OUTPUT: Set output (console, file, both, syslog)
   - LOG_FILE: Set log file path
   - LOG_CORRELATION_TRACKING: Enable correlation tracking (true/false)
   - LOG_PERFORMANCE_TRACKING: Enable performance tracking (true/false)

3. Correlation Context:
   ```python
   from src.monitoring.structured_logging import CorrelationIdManager
   from src.monitoring.logging_middleware import correlation_context
   
   # Set global context
   CorrelationIdManager.set_correlation_id("my-correlation-id")
   CorrelationIdManager.set_session_id("my-session-id")
   
   # Use context manager
   async with correlation_context(
       correlation_id="corr-123",
       session_id="sess-456",
       user_id="user-789"
   ):
       # All logging within this context will include these IDs
       logger.info("This will include correlation IDs")
   ```

4. FastAPI Integration:
   ```python
   from fastapi import FastAPI
   from src.monitoring.logging_middleware import add_correlation_middleware
   
   app = FastAPI()
   add_correlation_middleware(app, log_requests=True, log_responses=True)
   ```

5. Voice Pipeline Integration:
   ```python
   from src.pipeline.voice_pipeline_with_logging import VoicePipelineWithLogging
   
   pipeline = VoicePipelineWithLogging(
       stt_provider=my_stt,
       llm_provider=my_llm,
       tts_provider=my_tts,
       session_id="my-session"
   )
   
   # Automatic correlation tracking across STT->LLM->TTS
   result, metrics = await pipeline.process_audio(audio_data, user_id="user123")
   ```

6. Performance Monitoring:
   ```python
   from src.monitoring.structured_logging import timed_operation
   
   @timed_operation(operation_name="my_operation", component="my_component")
   async def my_function():
       # Automatic timing and performance logging
       return "result"
   ```

7. Custom Logging:
   ```python
   from src.monitoring.logging_config import get_logger
   
   logger = get_logger("my.module", component="my_component")
   
   # Structured logging with extra data
   logger.info("User action", extra_data={
       "action": "login",
       "user_id": "123",
       "ip_address": "192.168.1.1"
   })
   
   # Performance metrics
   logger.log_performance_metrics("api_call", {
       "duration_ms": 150,
       "status_code": 200,
       "bytes_transferred": 1024
   })
   ```
"""
    print(guide)


if __name__ == "__main__":
    print_configuration_guide()
    asyncio.run(run_all_examples())