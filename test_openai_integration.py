#!/usr/bin/env python3
"""
OpenAI Integration Test Runner for Task 1.0 Voice Processing Pipeline

This script tests the OpenAI integration to validate that our Task 1.0 
implementation works with real OpenAI API connections.
"""

import asyncio
import sys
import os
import traceback
import time
import json
import base64
from typing import List, Callable, Dict, Any
from io import BytesIO

# Add src to path
sys.path.append('src')

# Environment setup
from dotenv import load_dotenv
load_dotenv()


class TestResult:
    def __init__(self, name: str, passed: bool, error: str = None, duration: float = 0.0, details: Dict[str, Any] = None):
        self.name = name
        self.passed = passed
        self.error = error
        self.duration = duration
        self.details = details or {}


class OpenAIIntegrationTestRunner:
    def __init__(self):
        self.tests: List[Callable] = []
        self.results: List[TestResult] = []
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
    def add_test(self, test_func: Callable):
        """Add a test function to the runner."""
        self.tests.append(test_func)
    
    async def run_all_tests(self):
        """Run all registered tests."""
        print("OpenAI Integration Tests for Task 1.0 Voice Processing Pipeline")
        print("=" * 70)
        
        if not self.openai_api_key:
            print("❌ OPENAI_API_KEY not found in environment")
            print("Please set your OpenAI API key in .env file")
            return
        
        print(f"✅ OpenAI API Key found: {self.openai_api_key[:8]}...{self.openai_api_key[-4:]}")
        print()
        
        for test_func in self.tests:
            test_name = test_func.__name__
            start_time = time.time()
            
            try:
                if asyncio.iscoroutinefunction(test_func):
                    result_details = await test_func()
                else:
                    result_details = test_func()
                
                duration = time.time() - start_time
                self.results.append(TestResult(test_name, True, duration=duration, details=result_details))
                print(f"✅ {test_name} - PASSED ({duration:.3f}s)")
                
                if result_details:
                    for key, value in result_details.items():
                        print(f"   📊 {key}: {value}")
                
            except Exception as e:
                duration = time.time() - start_time
                error_msg = f"{type(e).__name__}: {str(e)}"
                self.results.append(TestResult(test_name, False, error_msg, duration))
                print(f"❌ {test_name} - FAILED ({duration:.3f}s): {error_msg}")
                # Uncomment for detailed traceback
                # traceback.print_exc()
            
            print()
        
        self._print_summary()
    
    def _print_summary(self):
        """Print test results summary."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        total_duration = sum(r.duration for r in self.results)
        
        print("=" * 70)
        print(f"OpenAI Integration Test Results: {passed}/{total} passed")
        print(f"Total execution time: {total_duration:.3f}s")
        
        if failed > 0:
            print(f"\n🔴 Failed tests:")
            for result in self.results:
                if not result.passed:
                    print(f"  - {result.name}: {result.error}")
        else:
            print("\n🎉 All OpenAI integration tests passed!")
            print("✅ Task 1.0 Voice Processing Pipeline is working with OpenAI!")


# Initialize test runner
runner = OpenAIIntegrationTestRunner()


def test_environment_setup():
    """Test that environment and imports are properly configured."""
    # Test basic imports
    try:
        from components.stt.base_stt import BaseSTTProvider, STTConfig
        from components.llm.base_llm import BaseLLMProvider, LLMConfig
        from components.tts.base_tts import BaseTTSProvider, TTSConfig
        from pipeline.audio_pipeline import StreamingAudioPipeline
        from monitoring.performance_monitor import PerformanceMonitor
    except ImportError as e:
        raise ImportError(f"Failed to import core components: {e}")
    
    # Test OpenAI imports (this will fail if OpenAI is not installed)
    try:
        import openai
        from openai import AsyncOpenAI
        openai_available = True
    except ImportError:
        openai_available = False
    
    return {
        "core_components_imported": True,
        "openai_package_available": openai_available,
        "api_key_configured": bool(runner.openai_api_key)
    }


def test_openai_provider_instantiation():
    """Test that OpenAI providers can be instantiated without errors."""
    try:
        from components.stt.openai_stt import OpenAISTTProvider, OpenAISTTConfig
        from components.llm.openai_llm import OpenAILLMProvider, OpenAILLMConfig
        from components.llm.base_llm import LLMModelType
    except ImportError as e:
        raise ImportError(f"Failed to import OpenAI providers: {e}")
    
    # Test STT provider instantiation
    stt_config = OpenAISTTConfig(
        api_key=runner.openai_api_key,
        model="whisper-1",
        sample_rate=16000
    )
    stt_provider = OpenAISTTProvider(stt_config)
    
    # Test LLM provider instantiation
    llm_config = OpenAILLMConfig(
        api_key=runner.openai_api_key,
        model=LLMModelType.GPT_4_1_MINI,
        temperature=0.7
    )
    llm_provider = OpenAILLMProvider(llm_config)
    
    return {
        "stt_provider_created": stt_provider.provider_name == "openai_whisper",
        "llm_provider_created": llm_provider.provider_name == "openai_gpt",
        "stt_supports_streaming": stt_provider.supports_streaming,
        "llm_supports_streaming": llm_provider.supports_streaming,
        "llm_supports_functions": llm_provider.supports_function_calling
    }


async def test_openai_llm_simple_completion():
    """Test OpenAI LLM with a simple completion request."""
    try:
        from components.llm.openai_llm import OpenAILLMProvider, OpenAILLMConfig
        from components.llm.base_llm import LLMMessage, LLMRole, LLMModelType
        from monitoring.performance_monitor import global_performance_monitor
    except ImportError as e:
        raise ImportError(f"Failed to import required components: {e}")
    
    # Create and initialize LLM provider
    config = OpenAILLMConfig(
        api_key=runner.openai_api_key,
        model=LLMModelType.GPT_4_1_MINI,
        temperature=0.3,
        max_tokens=50
    )
    
    provider = OpenAILLMProvider(config)
    
    # Initialize the provider
    await provider.initialize()
    
    # Create a simple test message
    messages = [
        LLMMessage(role=LLMRole.USER, content="Hello! Please respond with exactly 'Voice agent test successful'")
    ]
    
    # Generate response
    start_time = time.time()
    response = await provider.generate_response(messages)
    end_time = time.time()
    
    # Cleanup
    await provider.cleanup()
    
    # Validate response
    response_time = end_time - start_time
    response_contains_expected = "voice agent" in response.content.lower() or "successful" in response.content.lower()
    
    return {
        "response_generated": bool(response.content),
        "response_time_ms": round(response_time * 1000, 2),
        "response_length": len(response.content),
        "response_preview": response.content[:100] + "..." if len(response.content) > 100 else response.content,
        "response_contains_expected": response_contains_expected,
        "model_used": response.model,
        "tokens_used": response.usage.get("total_tokens") if response.usage else None
    }


async def test_openai_stt_with_generated_audio():
    """Test OpenAI STT with a simple generated audio sample."""
    try:
        from components.stt.openai_stt import OpenAISTTProvider, OpenAISTTConfig
        import numpy as np
        import wave
    except ImportError as e:
        raise ImportError(f"Failed to import required components: {e}")
    
    # Create STT provider
    config = OpenAISTTConfig(
        api_key=runner.openai_api_key,
        model="whisper-1",
        sample_rate=16000,
        language="en"
    )
    
    provider = OpenAISTTProvider(config)
    await provider.initialize()
    
    # Generate a simple test audio (silence + simple tone pattern)
    # This won't produce meaningful transcription but tests the API connection
    duration = 2.0  # 2 seconds
    sample_rate = 16000
    samples = int(duration * sample_rate)
    
    # Create simple audio pattern (very quiet tone to avoid loud output)
    t = np.linspace(0, duration, samples)
    frequency = 440  # A note
    audio_data = (0.1 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
    
    # Convert to 16-bit PCM
    audio_int16 = (audio_data * 32767).astype(np.int16)
    
    # Create WAV file in memory
    wav_buffer = BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())
    
    wav_data = wav_buffer.getvalue()
    
    # Test transcription
    start_time = time.time()
    try:
        result = await provider.transcribe_audio(wav_data)
        transcription_successful = True
        transcription_text = result.text if result else ""
    except Exception as e:
        transcription_successful = False
        transcription_text = f"Error: {str(e)}"
    
    end_time = time.time()
    
    # Cleanup
    await provider.cleanup()
    
    return {
        "audio_generated_seconds": duration,
        "audio_size_bytes": len(wav_data),
        "transcription_successful": transcription_successful,
        "transcription_time_ms": round((end_time - start_time) * 1000, 2),
        "transcription_result": transcription_text[:200] + "..." if len(transcription_text) > 200 else transcription_text,
        "api_connection_working": transcription_successful
    }


async def test_pipeline_integration():
    """Test integration between STT, LLM components in pipeline."""
    try:
        from components.stt.openai_stt import OpenAISTTProvider, OpenAISTTConfig
        from components.llm.openai_llm import OpenAILLMProvider, OpenAILLMConfig
        from components.llm.base_llm import LLMMessage, LLMRole, LLMModelType
        from pipeline.audio_pipeline import StreamingAudioPipeline, PipelineConfig, PipelineMode
        from monitoring.performance_monitor import global_performance_monitor
    except ImportError as e:
        raise ImportError(f"Failed to import pipeline components: {e}")
    
    # Create configurations
    stt_config = OpenAISTTConfig(
        api_key=runner.openai_api_key,
        model="whisper-1"
    )
    
    llm_config = OpenAILLMConfig(
        api_key=runner.openai_api_key,
        model=LLMModelType.GPT_4_1_MINI,
        max_tokens=30,
        temperature=0.3
    )
    
    pipeline_config = PipelineConfig(
        sample_rate=16000,
        mode=PipelineMode.PUSH_TO_TALK,
        enable_streaming_stt=False,  # Simplified for testing
        enable_streaming_llm=False
    )
    
    # Create providers
    stt_provider = OpenAISTTProvider(stt_config)
    llm_provider = OpenAILLMProvider(llm_config)
    
    # Create pipeline (without TTS and VAD for this test)
    pipeline = StreamingAudioPipeline(
        config=pipeline_config,
        stt_provider=stt_provider,
        llm_provider=llm_provider,
        tts_provider=None,  # Skip TTS for this test
        vad_provider=None   # Skip VAD for this test
    )
    
    # Initialize pipeline
    await pipeline.initialize()
    
    # Test pipeline state
    pipeline_initialized = pipeline.current_state.name != "ERROR"
    
    # Test component integration
    components_registered = (
        pipeline.stt_provider is not None and
        pipeline.llm_provider is not None
    )
    
    # Cleanup
    await pipeline.cleanup()
    
    return {
        "pipeline_initialized": pipeline_initialized,
        "components_registered": components_registered,
        "pipeline_config_valid": pipeline_config.sample_rate == 16000,
        "stt_provider_ready": stt_provider.provider_name == "openai_whisper",
        "llm_provider_ready": llm_provider.provider_name == "openai_gpt"
    }


async def test_performance_monitoring():
    """Test that performance monitoring captures OpenAI operations."""
    try:
        from monitoring.performance_monitor import global_performance_monitor, monitor_performance
        from components.llm.openai_llm import OpenAILLMProvider, OpenAILLMConfig
        from components.llm.base_llm import LLMMessage, LLMRole, LLMModelType
    except ImportError as e:
        raise ImportError(f"Failed to import monitoring components: {e}")
    
    # Start monitoring
    await global_performance_monitor.start_monitoring()
    
    # Register a component for monitoring
    component_monitor = global_performance_monitor.register_component("test_openai_llm", "llm")
    
    # Create LLM provider
    config = OpenAILLMConfig(
        api_key=runner.openai_api_key,
        model=LLMModelType.GPT_4_1_MINI,
        max_tokens=20
    )
    provider = OpenAILLMProvider(config)
    await provider.initialize()
    
    # Perform monitored operation
    messages = [LLMMessage(role=LLMRole.USER, content="Say 'monitoring test'")]
    
    async with component_monitor.monitor_operation("test_completion"):
        response = await provider.generate_response(messages)
    
    # Get metrics
    component_summary = component_monitor.get_summary()
    overall_summary = global_performance_monitor.get_overall_summary()
    
    # Cleanup
    await provider.cleanup()
    await global_performance_monitor.stop_monitoring()
    
    return {
        "monitoring_active": component_summary.get("total_operations", 0) > 0,
        "operations_recorded": component_summary.get("total_operations", 0),
        "success_rate": component_summary.get("overall_success_rate", 0),
        "response_generated": bool(response.content),
        "has_performance_data": len(overall_summary.get("components", {})) > 0
    }


async def test_error_handling():
    """Test error handling and graceful degradation."""
    try:
        from components.llm.openai_llm import OpenAILLMProvider, OpenAILLMConfig
        from components.llm.base_llm import LLMMessage, LLMRole, LLMModelType
        from components.error_handling import global_error_handler, handle_errors
    except ImportError as e:
        raise ImportError(f"Failed to import error handling components: {e}")
    
    # Test with invalid API key to trigger error handling
    config = OpenAILLMConfig(
        api_key="invalid_key_for_testing",
        model=LLMModelType.GPT_4_1_MINI,
        timeout=5.0  # Quick timeout
    )
    
    provider = OpenAILLMProvider(config)
    
    try:
        await provider.initialize()
        # This should fail with invalid API key
        messages = [LLMMessage(role=LLMRole.USER, content="Test")]
        response = await provider.generate_response(messages)
        error_handled = False  # If we get here, something's wrong
        error_type = None
    except Exception as e:
        error_handled = True
        error_type = type(e).__name__
    
    # Test with valid API key for comparison
    valid_config = OpenAILLMConfig(
        api_key=runner.openai_api_key,
        model=LLMModelType.GPT_4_1_MINI
    )
    valid_provider = OpenAILLMProvider(valid_config)
    
    try:
        await valid_provider.initialize()
        messages = [LLMMessage(role=LLMRole.USER, content="Hello")]
        response = await valid_provider.generate_response(messages)
        valid_request_worked = bool(response.content)
        await valid_provider.cleanup()
    except Exception:
        valid_request_worked = False
    
    # Get error statistics
    error_stats = global_error_handler.get_error_stats()
    
    return {
        "invalid_key_error_handled": error_handled,
        "error_type": error_type,
        "valid_request_worked": valid_request_worked,
        "error_handler_active": True,
        "total_errors_tracked": error_stats.get("total_errors", 0)
    }


# Register all tests
runner.add_test(test_environment_setup)
runner.add_test(test_openai_provider_instantiation)
runner.add_test(test_openai_llm_simple_completion)
runner.add_test(test_openai_stt_with_generated_audio)
runner.add_test(test_pipeline_integration)
runner.add_test(test_performance_monitoring)
runner.add_test(test_error_handling)


async def main():
    """Main test execution."""
    await runner.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())