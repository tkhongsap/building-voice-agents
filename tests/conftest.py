"""
Shared test fixtures and configurations for Task 1.0 Voice Processing Pipeline tests.

This module provides common test fixtures, mock objects, and configuration
that can be reused across different test files.
"""

import pytest
import asyncio
import sys
import os
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List, Optional

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

# Import communication test fixtures
pytest_plugins = [
    "tests.unit.test_communication.conftest_communication"
]


# ============================================================================
# TEST DATA FIXTURES
# ============================================================================

@pytest.fixture
def test_audio_data() -> bytes:
    """Provide test audio data for testing."""
    return b'\x00\x01' * 1024  # 2KB of test audio data


@pytest.fixture
def test_text() -> str:
    """Provide test text for testing."""
    return "Hello, this is a test message for voice processing."


@pytest.fixture
def test_wav_header() -> bytes:
    """Provide a minimal WAV file header for testing."""
    # Minimal 44-byte WAV header for 16-bit mono 16kHz
    return bytes([
        0x52, 0x49, 0x46, 0x46,  # "RIFF"
        0x24, 0x08, 0x00, 0x00,  # File size - 8
        0x57, 0x41, 0x56, 0x45,  # "WAVE"
        0x66, 0x6D, 0x74, 0x20,  # "fmt "
        0x10, 0x00, 0x00, 0x00,  # fmt chunk size
        0x01, 0x00,              # Audio format (PCM)
        0x01, 0x00,              # Number of channels (mono)
        0x80, 0x3E, 0x00, 0x00,  # Sample rate (16000)
        0x00, 0x7D, 0x00, 0x00,  # Byte rate
        0x02, 0x00,              # Block align
        0x10, 0x00,              # Bits per sample
        0x64, 0x61, 0x74, 0x61,  # "data"
        0x00, 0x08, 0x00, 0x00   # Data chunk size
    ])


# ============================================================================
# MOCK PROVIDER FIXTURES
# ============================================================================

@pytest.fixture
def mock_stt_config():
    """Create a mock STT configuration."""
    config = Mock()
    config.api_key = "test_stt_key"
    config.model = "test_model"
    config.language = "en-US"
    config.sample_rate = 16000
    config.chunk_size = 1024
    config.enable_interim_results = True
    return config


@pytest.fixture
def mock_llm_config():
    """Create a mock LLM configuration."""
    config = Mock()
    config.api_key = "test_llm_key"
    config.model = "test_model"
    config.temperature = 0.7
    config.max_tokens = 1000
    config.enable_function_calling = True
    config.enable_streaming = True
    return config


@pytest.fixture
def mock_tts_config():
    """Create a mock TTS configuration."""
    config = Mock()
    config.api_key = "test_tts_key"
    config.voice_id = "test_voice"
    config.model = "test_model"
    config.sample_rate = 22050
    config.quality = "high"
    return config


@pytest.fixture
def mock_vad_config():
    """Create a mock VAD configuration."""
    config = Mock()
    config.sensitivity = 0.7
    config.sample_rate = 16000
    config.frame_size = 480
    config.aggressiveness = 3
    return config


# ============================================================================
# MOCK PROVIDER CLASSES
# ============================================================================

class MockSTTProvider:
    """Mock STT provider for testing."""
    
    def __init__(self, config):
        self.config = config
        self.provider_name = "mock_stt"
        self.supported_languages = ["en-US", "es-ES"]
        self.supported_sample_rates = [16000, 22050]
        self.supports_streaming = True
        self._streaming_active = False
    
    async def initialize(self):
        pass
    
    async def cleanup(self):
        pass
    
    async def transcribe_audio(self, audio_data: bytes):
        from components.stt.base_stt import STTResult
        return STTResult(
            text="Mock transcription result",
            confidence=0.95,
            is_final=True,
            metadata={"provider": self.provider_name}
        )
    
    async def start_streaming(self):
        self._streaming_active = True
    
    async def stop_streaming(self):
        self._streaming_active = False
    
    async def stream_audio(self, audio_chunk: bytes):
        pass
    
    async def get_streaming_results(self):
        from components.stt.base_stt import STTResult
        if self._streaming_active:
            yield STTResult(
                text="Mock streaming result",
                confidence=0.8,
                is_final=False
            )
    
    async def health_check(self):
        return {"status": "healthy", "provider": self.provider_name}


class MockLLMProvider:
    """Mock LLM provider for testing."""
    
    def __init__(self, config):
        self.config = config
        self.provider_name = "mock_llm"
        self.supported_models = ["test_model"]
        self.supports_function_calling = True
        self.supports_streaming = True
    
    async def initialize(self):
        pass
    
    async def cleanup(self):
        pass
    
    async def generate_response(self, messages, functions=None):
        from components.llm.base_llm import LLMResponse, LLMRole
        return LLMResponse(
            content="Mock LLM response",
            role=LLMRole.ASSISTANT,
            finish_reason="stop",
            usage={"total_tokens": 50},
            metadata={"provider": self.provider_name}
        )
    
    async def generate_streaming_response(self, messages, functions=None):
        from components.llm.base_llm import LLMResponse, LLMRole
        chunks = ["Mock ", "streaming ", "response"]
        for chunk in chunks:
            yield LLMResponse(
                content=chunk,
                role=LLMRole.ASSISTANT,
                is_streaming=True,
                metadata={"provider": self.provider_name}
            )
    
    async def health_check(self):
        return {"status": "healthy", "provider": self.provider_name}


class MockTTSProvider:
    """Mock TTS provider for testing."""
    
    def __init__(self, config):
        self.config = config
        self.provider_name = "mock_tts"
        self.available_voices = ["test_voice"]
        self.supported_sample_rates = [22050, 44100]
        self.supports_streaming = True
    
    async def initialize(self):
        pass
    
    async def cleanup(self):
        pass
    
    async def speak(self, text: str):
        from components.tts.base_tts import TTSResult
        return TTSResult(
            audio_data=b'mock_audio_data' * 100,
            sample_rate=22050,
            duration=len(text) * 0.1,  # Rough estimate
            metadata={"provider": self.provider_name}
        )
    
    async def speak_streaming(self, text: str):
        chunks = [b'mock_chunk_1', b'mock_chunk_2', b'mock_chunk_3']
        for chunk in chunks:
            yield chunk
    
    async def health_check(self):
        return {"status": "healthy", "provider": self.provider_name}


class MockVADProvider:
    """Mock VAD provider for testing."""
    
    def __init__(self, config):
        self.config = config
        self.provider_name = "mock_vad"
        self._processing_active = False
        self._speech_callbacks = {}
    
    async def initialize(self):
        pass
    
    async def cleanup(self):
        pass
    
    async def process_audio_chunk(self, audio_chunk: bytes):
        from components.vad.base_vad import VADResult, VADState
        # Mock speech detection based on chunk size
        is_speech = len(audio_chunk) > 512
        return VADResult(
            is_speech=is_speech,
            confidence=0.9 if is_speech else 0.1,
            state=VADState.SPEECH if is_speech else VADState.SILENCE,
            metadata={"provider": self.provider_name}
        )
    
    async def start_processing(self):
        self._processing_active = True
    
    async def stop_processing(self):
        self._processing_active = False
    
    def set_speech_callbacks(self, on_speech_start=None, on_speech_end=None):
        self._speech_callbacks = {
            "on_speech_start": on_speech_start,
            "on_speech_end": on_speech_end
        }
    
    async def health_check(self):
        return {"status": "healthy", "provider": self.provider_name}


# ============================================================================
# PROVIDER FIXTURES
# ============================================================================

@pytest.fixture
def mock_stt_provider(mock_stt_config):
    """Create a mock STT provider."""
    return MockSTTProvider(mock_stt_config)


@pytest.fixture
def mock_llm_provider(mock_llm_config):
    """Create a mock LLM provider."""
    return MockLLMProvider(mock_llm_config)


@pytest.fixture
def mock_tts_provider(mock_tts_config):
    """Create a mock TTS provider."""
    return MockTTSProvider(mock_tts_config)


@pytest.fixture
def mock_vad_provider(mock_vad_config):
    """Create a mock VAD provider."""
    return MockVADProvider(mock_vad_config)


# ============================================================================
# PIPELINE FIXTURES
# ============================================================================

@pytest.fixture
def mock_pipeline_config():
    """Create a mock pipeline configuration."""
    config = Mock()
    config.sample_rate = 16000
    config.chunk_size = 1024
    config.buffer_size = 4096
    config.max_silence_duration = 2.0
    config.min_speech_duration = 0.5
    config.enable_streaming_stt = True
    config.enable_streaming_llm = True
    config.enable_streaming_tts = True
    config.enable_interruption = True
    config.max_concurrent_requests = 3
    config.enable_metrics = True
    return config


@pytest.fixture
def mock_complete_pipeline(
    mock_pipeline_config,
    mock_stt_provider,
    mock_llm_provider,
    mock_tts_provider,
    mock_vad_provider
):
    """Create a complete mock pipeline with all providers."""
    pipeline_data = {
        "config": mock_pipeline_config,
        "stt_provider": mock_stt_provider,
        "llm_provider": mock_llm_provider,
        "tts_provider": mock_tts_provider,
        "vad_provider": mock_vad_provider
    }
    return pipeline_data


# ============================================================================
# ASYNC TEST UTILITIES
# ============================================================================

@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def async_context():
    """Provide an async context for testing."""
    class AsyncContext:
        async def __aenter__(self):
            return self
        
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
    
    return AsyncContext()


# ============================================================================
# ERROR HANDLING FIXTURES
# ============================================================================

@pytest.fixture
def mock_error_handler():
    """Create a mock error handler."""
    handler = Mock()
    handler.classify_error = Mock(return_value=Mock(
        error_type="TestError",
        severity=Mock(value="medium"),
        category=Mock(value="test"),
        component="test_component"
    ))
    handler.handle_error = AsyncMock(return_value=None)
    handler.get_error_stats = Mock(return_value={
        "total_errors": 0,
        "recent_errors": 0,
        "errors_by_severity": {},
        "errors_by_category": {}
    })
    return handler


# ============================================================================
# PERFORMANCE MONITORING FIXTURES
# ============================================================================

@pytest.fixture
def mock_performance_monitor():
    """Create a mock performance monitor."""
    monitor = Mock()
    monitor.register_component = Mock(return_value=Mock(
        monitor_operation=Mock(return_value=Mock(
            __aenter__=AsyncMock(),
            __aexit__=AsyncMock()
        )),
        get_summary=Mock(return_value={
            "total_operations": 1,
            "overall_success_rate": 1.0,
            "average_latency": 0.1
        })
    ))
    monitor.record_metric = Mock()
    monitor.get_overall_summary = Mock(return_value={
        "timestamp": 1234567890,
        "components": {},
        "total_operations": 1
    })
    monitor.start_monitoring = AsyncMock()
    monitor.stop_monitoring = AsyncMock()
    return monitor


# ============================================================================
# FILE SYSTEM MOCKS
# ============================================================================

@pytest.fixture
def mock_file_exists():
    """Mock os.path.exists to return True for all test files."""
    with patch('os.path.exists', return_value=True):
        yield


@pytest.fixture
def temp_test_env():
    """Create a temporary test environment."""
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    original_cwd = os.getcwd()
    
    try:
        os.chdir(temp_dir)
        yield temp_dir
    finally:
        os.chdir(original_cwd)
        shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================================
# CONFIGURATION HELPERS
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "requires_api: mark test as requiring external API"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add unit marker to all tests by default
        if not any(marker.name in ['integration', 'slow', 'requires_api'] 
                   for marker in item.iter_markers()):
            item.add_marker(pytest.mark.unit)


# ============================================================================
# CLEANUP HELPERS
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_globals():
    """Clean up global state after each test."""
    yield
    # Reset any global state that might affect other tests
    import gc
    gc.collect()