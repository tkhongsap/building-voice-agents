#!/usr/bin/env python3
"""
Simple test runner for the Voice Agents Platform implementation.
This script validates the core components without requiring pytest installation.
"""

import asyncio
import sys
import traceback
from typing import List, Callable


class TestResult:
    def __init__(self, name: str, passed: bool, error: str = None):
        self.name = name
        self.passed = passed
        self.error = error


class SimpleTestRunner:
    def __init__(self):
        self.tests: List[Callable] = []
        self.results: List[TestResult] = []
    
    def add_test(self, test_func: Callable):
        """Add a test function to the runner."""
        self.tests.append(test_func)
    
    async def run_all_tests(self):
        """Run all registered tests."""
        print("Running Voice Agents Platform Tests...")
        print("=" * 50)
        
        for test_func in self.tests:
            test_name = test_func.__name__
            try:
                if asyncio.iscoroutinefunction(test_func):
                    await test_func()
                else:
                    test_func()
                
                self.results.append(TestResult(test_name, True))
                print(f"✅ {test_name} - PASSED")
                
            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                self.results.append(TestResult(test_name, False, error_msg))
                print(f"❌ {test_name} - FAILED: {error_msg}")
                # Uncomment for detailed traceback
                # traceback.print_exc()
        
        self._print_summary()
    
    def _print_summary(self):
        """Print test results summary."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        
        print("\n" + "=" * 50)
        print(f"Test Results: {passed}/{total} passed")
        
        if failed > 0:
            print(f"\nFailed tests:")
            for result in self.results:
                if not result.passed:
                    print(f"  - {result.name}: {result.error}")
            sys.exit(1)
        else:
            print("🎉 All tests passed!")


# Initialize test runner
runner = SimpleTestRunner()


# Test 1: Base STT Provider
def test_stt_config_creation():
    """Test STT configuration creation."""
    import sys
    sys.path.append('src')
    
    from components.stt.base_stt import STTConfig, STTLanguage, STTQuality
    
    config = STTConfig(
        language=STTLanguage.ENGLISH,
        quality=STTQuality.HIGH,
        sample_rate=16000
    )
    
    assert config.language == STTLanguage.ENGLISH
    assert config.quality == STTQuality.HIGH
    assert config.sample_rate == 16000


def test_stt_result_creation():
    """Test STT result creation."""
    import sys
    sys.path.append('src')
    
    from components.stt.base_stt import STTResult
    
    result = STTResult(
        text="Test transcription",
        confidence=0.85,
        is_final=True,
        language="en"
    )
    
    assert result.text == "Test transcription"
    assert result.confidence == 0.85
    assert result.is_final is True


# Test 2: Base LLM Provider  
def test_llm_config_creation():
    """Test LLM configuration creation."""
    import sys
    sys.path.append('src')
    
    from components.llm.base_llm import LLMConfig, LLMModelType
    
    config = LLMConfig(
        model=LLMModelType.GPT_4O_MINI,
        temperature=0.7,
        max_tokens=1000
    )
    
    assert config.model == LLMModelType.GPT_4O_MINI
    assert config.temperature == 0.7
    assert config.max_tokens == 1000


def test_llm_message_creation():
    """Test LLM message creation."""
    import sys
    sys.path.append('src')
    
    from components.llm.base_llm import LLMMessage, LLMRole
    
    message = LLMMessage(
        role=LLMRole.USER,
        content="Hello, how are you?"
    )
    
    assert message.role == LLMRole.USER
    assert message.content == "Hello, how are you?"
    
    # Test message to dict conversion
    msg_dict = message.to_dict()
    assert msg_dict["role"] == "user"
    assert msg_dict["content"] == "Hello, how are you?"


# Test 3: Base TTS Provider
def test_tts_config_creation():
    """Test TTS configuration creation."""
    import sys
    sys.path.append('src')
    
    from components.tts.base_tts import TTSConfig, TTSLanguage, AudioFormat
    
    config = TTSConfig(
        language=TTSLanguage.ENGLISH,
        audio_format=AudioFormat.WAV,
        sample_rate=22050
    )
    
    assert config.language == TTSLanguage.ENGLISH
    assert config.audio_format == AudioFormat.WAV
    assert config.sample_rate == 22050


def test_voice_creation():
    """Test Voice object creation."""
    import sys
    sys.path.append('src')
    
    from components.tts.base_tts import Voice, TTSLanguage, TTSVoice
    
    voice = Voice(
        id="test_voice_id",
        name="Test Voice",
        language=TTSLanguage.ENGLISH,
        gender=TTSVoice.FEMALE,
        description="A test voice"
    )
    
    assert voice.id == "test_voice_id"
    assert voice.name == "Test Voice"
    assert voice.language == TTSLanguage.ENGLISH
    assert voice.gender == TTSVoice.FEMALE


# Test 4: Base VAD Provider
def test_vad_config_creation():
    """Test VAD configuration creation."""
    import sys
    sys.path.append('src')
    
    from components.vad.base_vad import VADConfig, VADSensitivity
    
    config = VADConfig(
        sensitivity=VADSensitivity.MEDIUM,
        sample_rate=16000,
        speech_threshold=0.5
    )
    
    assert config.sensitivity == VADSensitivity.MEDIUM
    assert config.sample_rate == 16000
    assert config.speech_threshold == 0.5


def test_vad_result_creation():
    """Test VAD result creation."""
    import sys
    sys.path.append('src')
    
    from components.vad.base_vad import VADResult, VADState
    
    result = VADResult(
        is_speech=True,
        confidence=0.9,
        volume=0.7
    )
    
    assert result.is_speech is True
    assert result.confidence == 0.9
    assert result.state == VADState.SPEECH  # Should be set automatically


# Test 5: Audio Pipeline
def test_pipeline_config_creation():
    """Test audio pipeline configuration."""
    import sys
    sys.path.append('src')
    
    from pipeline.audio_pipeline import PipelineConfig, PipelineMode
    
    config = PipelineConfig(
        sample_rate=16000,
        mode=PipelineMode.CONTINUOUS,
        enable_streaming_stt=True
    )
    
    assert config.sample_rate == 16000
    assert config.mode == PipelineMode.CONTINUOUS
    assert config.enable_streaming_stt is True


def test_pipeline_metrics():
    """Test pipeline metrics creation."""
    import sys
    sys.path.append('src')
    
    from pipeline.audio_pipeline import PipelineMetrics
    
    metrics = PipelineMetrics()
    metrics.total_latency = 0.200  # 200ms
    metrics.stt_latency = 0.050
    metrics.llm_latency = 0.100
    metrics.tts_latency = 0.050
    
    assert metrics.total_latency == 0.200
    assert metrics.is_meeting_latency_target() is True  # Under 236ms target
    
    metrics.total_latency = 0.300  # 300ms
    assert metrics.is_meeting_latency_target() is False  # Over target


# Test 6: Provider Import Tests
def test_openai_stt_import():
    """Test OpenAI STT provider import."""
    import sys
    sys.path.append('src')
    
    from components.stt.openai_stt import OpenAISTTProvider, OpenAISTTConfig
    
    config = OpenAISTTConfig(api_key="test_key", model="whisper-1")
    provider = OpenAISTTProvider(config)
    
    assert provider.provider_name == "openai_whisper"
    assert config.model == "whisper-1"


def test_openai_llm_import():
    """Test OpenAI LLM provider import."""
    import sys
    sys.path.append('src')
    
    from components.llm.openai_llm import OpenAILLMProvider, OpenAILLMConfig
    from components.llm.base_llm import LLMModelType
    
    config = OpenAILLMConfig(api_key="test_key", model=LLMModelType.GPT_4O_MINI)
    provider = OpenAILLMProvider(config)
    
    assert provider.provider_name == "openai_gpt"
    assert provider.supports_function_calling is True
    assert provider.supports_streaming is True


def test_elevenlabs_tts_import():
    """Test ElevenLabs TTS provider import."""
    import sys
    sys.path.append('src')
    
    from components.tts.elevenlabs_tts import ElevenLabsTTSProvider, ElevenLabsTTSConfig
    
    config = ElevenLabsTTSConfig(api_key="test_key", voice_id="test_voice")
    provider = ElevenLabsTTSProvider(config)
    
    assert provider.provider_name == "elevenlabs"
    assert config.voice_id == "test_voice"


# Register all tests
runner.add_test(test_stt_config_creation)
runner.add_test(test_stt_result_creation)
runner.add_test(test_llm_config_creation)
runner.add_test(test_llm_message_creation)
runner.add_test(test_tts_config_creation)
runner.add_test(test_voice_creation)
runner.add_test(test_vad_config_creation)
runner.add_test(test_vad_result_creation)
runner.add_test(test_pipeline_config_creation)
runner.add_test(test_pipeline_metrics)
runner.add_test(test_openai_stt_import)
runner.add_test(test_openai_llm_import)
runner.add_test(test_elevenlabs_tts_import)


async def main():
    """Main test execution."""
    await runner.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())