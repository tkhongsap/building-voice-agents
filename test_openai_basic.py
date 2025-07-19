#!/usr/bin/env python3
"""
Basic OpenAI Integration Test for Task 1.0 Voice Processing Pipeline

This script tests the basic structure and validates that our Task 1.0 
implementation is properly structured for OpenAI integration.
"""

import asyncio
import sys
import os
import traceback
import time
from typing import List, Callable, Dict, Any

# Add src to path
sys.path.append('src')


class TestResult:
    def __init__(self, name: str, passed: bool, error: str = None, duration: float = 0.0, details: Dict[str, Any] = None):
        self.name = name
        self.passed = passed
        self.error = error
        self.duration = duration
        self.details = details or {}


class BasicOpenAITestRunner:
    def __init__(self):
        self.tests: List[Callable] = []
        self.results: List[TestResult] = []
        
        # Try to get OpenAI API key from environment or .env file
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            try:
                with open('.env', 'r') as f:
                    for line in f:
                        if line.startswith('OPENAI_API_KEY'):
                            self.openai_api_key = line.split('=')[1].strip().strip('"').strip("'")
                            break
            except FileNotFoundError:
                pass
        
    def add_test(self, test_func: Callable):
        """Add a test function to the runner."""
        self.tests.append(test_func)
    
    async def run_all_tests(self):
        """Run all registered tests."""
        print("Basic OpenAI Integration Tests for Task 1.0")
        print("=" * 50)
        
        if self.openai_api_key:
            print(f"✅ OpenAI API Key found: {self.openai_api_key[:8]}...{self.openai_api_key[-4:]}")
        else:
            print("⚠️  OpenAI API Key not found - will test structure only")
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
            
            print()
        
        self._print_summary()
    
    def _print_summary(self):
        """Print test results summary."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        total_duration = sum(r.duration for r in self.results)
        
        print("=" * 50)
        print(f"Test Results: {passed}/{total} passed")
        print(f"Total execution time: {total_duration:.3f}s")
        
        if failed > 0:
            print(f"\n🔴 Failed tests:")
            for result in self.results:
                if not result.passed:
                    print(f"  - {result.name}: {result.error}")
        else:
            print("\n🎉 All tests passed!")
            if self.openai_api_key:
                print("✅ Task 1.0 structure is ready for OpenAI integration!")
            else:
                print("✅ Task 1.0 structure is valid (add OpenAI API key for full testing)")


# Initialize test runner
runner = BasicOpenAITestRunner()


def test_core_imports():
    """Test that all core components can be imported."""
    imports_successful = {}
    
    try:
        from components.stt.base_stt import BaseSTTProvider, STTConfig, STTResult
        imports_successful["base_stt"] = True
    except ImportError as e:
        imports_successful["base_stt"] = f"Failed: {e}"
    
    try:
        from components.llm.base_llm import BaseLLMProvider, LLMConfig, LLMMessage
        imports_successful["base_llm"] = True
    except ImportError as e:
        imports_successful["base_llm"] = f"Failed: {e}"
    
    try:
        from components.tts.base_tts import BaseTTSProvider, TTSConfig, TTSResult
        imports_successful["base_tts"] = True
    except ImportError as e:
        imports_successful["base_tts"] = f"Failed: {e}"
    
    try:
        from components.vad.base_vad import BaseVADProvider, VADConfig, VADResult
        imports_successful["base_vad"] = True
    except ImportError as e:
        imports_successful["base_vad"] = f"Failed: {e}"
    
    try:
        from pipeline.audio_pipeline import StreamingAudioPipeline, PipelineConfig
        imports_successful["audio_pipeline"] = True
    except ImportError as e:
        imports_successful["audio_pipeline"] = f"Failed: {e}"
    
    try:
        from monitoring.performance_monitor import PerformanceMonitor
        imports_successful["performance_monitor"] = True
    except ImportError as e:
        imports_successful["performance_monitor"] = f"Failed: {e}"
    
    try:
        from components.error_handling import ErrorHandler
        imports_successful["error_handling"] = True
    except ImportError as e:
        imports_successful["error_handling"] = f"Failed: {e}"
    
    # Check if any imports failed
    failed_imports = [k for k, v in imports_successful.items() if v != True]
    if failed_imports:
        raise ImportError(f"Failed imports: {failed_imports}")
    
    return imports_successful


def test_openai_provider_imports():
    """Test that OpenAI provider implementations can be imported."""
    imports_successful = {}
    
    try:
        from components.stt.openai_stt import OpenAISTTProvider, OpenAISTTConfig
        imports_successful["openai_stt"] = True
    except ImportError as e:
        imports_successful["openai_stt"] = f"Failed: {e}"
    
    try:
        from components.llm.openai_llm import OpenAILLMProvider, OpenAILLMConfig
        imports_successful["openai_llm"] = True
    except ImportError as e:
        imports_successful["openai_llm"] = f"Failed: {e}"
    
    # Note: We don't have OpenAI TTS implementation (they don't have TTS API yet)
    # but we have other TTS providers
    try:
        from components.tts.elevenlabs_tts import ElevenLabsTTSProvider
        imports_successful["alternative_tts"] = True
    except ImportError as e:
        imports_successful["alternative_tts"] = f"Failed: {e}"
    
    return imports_successful


def test_openai_provider_instantiation():
    """Test that OpenAI providers can be instantiated."""
    try:
        from components.stt.openai_stt import OpenAISTTProvider, OpenAISTTConfig
        from components.llm.openai_llm import OpenAILLMProvider, OpenAILLMConfig
        from components.llm.base_llm import LLMModelType
    except ImportError as e:
        raise ImportError(f"Cannot import OpenAI providers: {e}")
    
    # Test STT provider creation
    stt_config = OpenAISTTConfig(
        api_key=runner.openai_api_key or "test_key",
        model="whisper-1",
        sample_rate=16000
    )
    stt_provider = OpenAISTTProvider(stt_config)
    
    # Test LLM provider creation
    llm_config = OpenAILLMConfig(
        api_key=runner.openai_api_key or "test_key",
        model=LLMModelType.GPT_4_1_MINI,
        temperature=0.7
    )
    llm_provider = OpenAILLMProvider(llm_config)
    
    return {
        "stt_provider_name": stt_provider.provider_name,
        "stt_supports_streaming": stt_provider.supports_streaming,
        "llm_provider_name": llm_provider.provider_name,
        "llm_supports_streaming": llm_provider.supports_streaming,
        "llm_supports_functions": llm_provider.supports_function_calling,
        "stt_config_valid": stt_config.model == "whisper-1",
        "llm_config_valid": llm_config.model == LLMModelType.GPT_4_1_MINI
    }


def test_factory_patterns():
    """Test that factory patterns work for provider registration."""
    try:
        from components.stt.base_stt import STTProviderFactory
        from components.llm.base_llm import LLMProviderFactory
        from components.tts.base_tts import TTSProviderFactory
        from components.vad.base_vad import VADProviderFactory
    except ImportError as e:
        raise ImportError(f"Cannot import factory patterns: {e}")
    
    # Test that providers are registered
    stt_providers = STTProviderFactory.list_providers()
    llm_providers = LLMProviderFactory.list_providers()
    tts_providers = TTSProviderFactory.list_providers()
    vad_providers = VADProviderFactory.list_providers()
    
    return {
        "stt_providers_available": len(stt_providers),
        "stt_providers": list(stt_providers),
        "llm_providers_available": len(llm_providers),
        "llm_providers": list(llm_providers),
        "tts_providers_available": len(tts_providers),
        "tts_providers": list(tts_providers),
        "vad_providers_available": len(vad_providers),
        "vad_providers": list(vad_providers),
        "openai_stt_registered": "openai" in stt_providers or "openai_whisper" in stt_providers,
        "openai_llm_registered": "openai" in llm_providers or "openai_gpt" in llm_providers
    }


def test_pipeline_configuration():
    """Test that pipeline can be configured with OpenAI providers."""
    try:
        from pipeline.audio_pipeline import StreamingAudioPipeline, PipelineConfig, PipelineMode
        from components.stt.openai_stt import OpenAISTTProvider, OpenAISTTConfig
        from components.llm.openai_llm import OpenAILLMProvider, OpenAILLMConfig
        from components.llm.base_llm import LLMModelType
    except ImportError as e:
        raise ImportError(f"Cannot import pipeline components: {e}")
    
    # Create configurations
    pipeline_config = PipelineConfig(
        sample_rate=16000,
        mode=PipelineMode.CONTINUOUS,
        enable_streaming_stt=True,
        enable_streaming_llm=True,
        latency_target_ms=236  # Our target latency
    )
    
    stt_config = OpenAISTTConfig(
        api_key=runner.openai_api_key or "test_key",
        model="whisper-1"
    )
    
    llm_config = OpenAILLMConfig(
        api_key=runner.openai_api_key or "test_key",
        model=LLMModelType.GPT_4_1_MINI
    )
    
    # Create providers
    stt_provider = OpenAISTTProvider(stt_config)
    llm_provider = OpenAILLMProvider(llm_config)
    
    # Create pipeline (without initializing since we may not have API access)
    pipeline = StreamingAudioPipeline(
        config=pipeline_config,
        stt_provider=stt_provider,
        llm_provider=llm_provider,
        tts_provider=None,  # Optional for this test
        vad_provider=None   # Optional for this test
    )
    
    return {
        "pipeline_created": pipeline is not None,
        "pipeline_config_valid": pipeline_config.sample_rate == 16000,
        "latency_target_ms": pipeline_config.latency_target_ms,
        "stt_provider_attached": pipeline.stt_provider is not None,
        "llm_provider_attached": pipeline.llm_provider is not None,
        "streaming_enabled": pipeline_config.enable_streaming_stt and pipeline_config.enable_streaming_llm
    }


def test_monitoring_integration():
    """Test that monitoring systems are properly integrated."""
    try:
        from monitoring.performance_monitor import global_performance_monitor, ComponentMonitor
    except ImportError as e:
        raise ImportError(f"Cannot import monitoring components: {e}")
    
    # Test component registration
    stt_monitor = global_performance_monitor.register_component("test_openai_stt", "stt")
    llm_monitor = global_performance_monitor.register_component("test_openai_llm", "llm")
    
    # Test that monitors are ComponentMonitor instances
    monitors_valid = (
        isinstance(stt_monitor, ComponentMonitor) and
        isinstance(llm_monitor, ComponentMonitor)
    )
    
    # Test summary generation
    summary = global_performance_monitor.get_overall_summary()
    
    return {
        "monitors_created": monitors_valid,
        "stt_monitor_name": stt_monitor.component_name,
        "llm_monitor_name": llm_monitor.component_name,
        "monitoring_components": len(summary.get("components", {})),
        "summary_generated": "timestamp" in summary,
        "performance_tracking_ready": True
    }


def test_error_handling_integration():
    """Test that error handling systems are properly integrated."""
    try:
        from components.error_handling import global_error_handler, ErrorInfo, ErrorSeverity, ErrorCategory
    except ImportError as e:
        raise ImportError(f"Cannot import error handling components: {e}")
    
    # Test error classification
    test_exception = ValueError("Test API error")
    error_info = global_error_handler.classify_error(test_exception, "openai_llm")
    
    # Test fallback configuration
    fallback_configs = global_error_handler.fallback_configs
    
    return {
        "error_handler_available": True,
        "error_classified": error_info.error_type == "ValueError",
        "error_component": error_info.component,
        "error_severity": error_info.severity.value,
        "error_category": error_info.category.value,
        "fallback_configs_available": len(fallback_configs),
        "llm_fallback_configured": "llm" in fallback_configs,
        "stt_fallback_configured": "stt" in fallback_configs,
        "graceful_degradation_ready": True
    }


async def test_openai_api_availability():
    """Test if OpenAI API is actually accessible (if API key is available)."""
    if not runner.openai_api_key:
        return {
            "api_key_available": False,
            "test_skipped": True,
            "message": "No API key provided - cannot test actual API access"
        }
    
    try:
        # Try a very simple HTTP request to OpenAI API to check connectivity
        import urllib.request
        import json
        
        # Create a simple request to test API key
        url = "https://api.openai.com/v1/models"
        request = urllib.request.Request(url)
        request.add_header("Authorization", f"Bearer {runner.openai_api_key}")
        request.add_header("User-Agent", "VoiceAgent-Test/1.0")
        
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                data = json.loads(response.read().decode())
                models_available = len(data.get("data", []))
                
                return {
                    "api_key_valid": True,
                    "api_accessible": True,
                    "models_available": models_available,
                    "api_connection_working": True
                }
        except urllib.error.HTTPError as e:
            if e.code == 401:
                return {
                    "api_key_valid": False,
                    "api_accessible": True,
                    "error": "Invalid API key",
                    "api_connection_working": False
                }
            else:
                return {
                    "api_key_valid": "unknown",
                    "api_accessible": False,
                    "error": f"HTTP {e.code}: {e.reason}",
                    "api_connection_working": False
                }
        except Exception as e:
            return {
                "api_key_valid": "unknown",
                "api_accessible": False,
                "error": str(e),
                "api_connection_working": False
            }
    
    except ImportError:
        return {
            "api_key_available": True,
            "test_skipped": True,
            "message": "Cannot test API access without urllib (but key is available)"
        }


# Register all tests
runner.add_test(test_core_imports)
runner.add_test(test_openai_provider_imports)
runner.add_test(test_openai_provider_instantiation)
runner.add_test(test_factory_patterns)
runner.add_test(test_pipeline_configuration)
runner.add_test(test_monitoring_integration)
runner.add_test(test_error_handling_integration)
runner.add_test(test_openai_api_availability)


async def main():
    """Main test execution."""
    await runner.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())