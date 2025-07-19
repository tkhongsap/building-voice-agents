"""
Unit tests for Base STT Provider
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from .base_stt import (
    BaseSTTProvider, STTResult, STTConfig, STTLanguage, 
    STTQuality, STTProviderFactory
)


class MockSTTProvider(BaseSTTProvider):
    """Mock STT provider for testing."""
    
    @property
    def provider_name(self) -> str:
        return "mock_stt"
    
    @property
    def supported_languages(self) -> list:
        return [STTLanguage.ENGLISH, STTLanguage.SPANISH]
    
    @property
    def supported_sample_rates(self) -> list:
        return [16000, 22050]
    
    async def initialize(self) -> None:
        pass
    
    async def cleanup(self) -> None:
        pass
    
    async def transcribe_audio(self, audio_data: bytes) -> STTResult:
        return STTResult(
            text="Hello world",
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
        yield STTResult(text="Streaming result", confidence=0.9, is_final=False)


def test_stt_config_creation():
    """Test STT configuration creation."""
    config = STTConfig(
        language=STTLanguage.ENGLISH,
        quality=STTQuality.HIGH,
        sample_rate=16000
    )
    
    assert config.language == STTLanguage.ENGLISH
    assert config.quality == STTQuality.HIGH
    assert config.sample_rate == 16000
    assert config.enable_interim_results is True


def test_stt_result_creation():
    """Test STT result creation."""
    result = STTResult(
        text="Test transcription",
        confidence=0.85,
        is_final=True,
        language="en"
    )
    
    assert result.text == "Test transcription"
    assert result.confidence == 0.85
    assert result.is_final is True
    assert result.language == "en"


@pytest.mark.asyncio
async def test_mock_provider_functionality():
    """Test mock provider basic functionality."""
    config = STTConfig(language=STTLanguage.ENGLISH)
    provider = MockSTTProvider(config)
    
    # Test initialization
    await provider.initialize()
    
    # Test transcription
    result = await provider.transcribe_audio(b"fake audio data")
    assert result.text == "Hello world"
    assert result.confidence == 0.95
    
    # Test streaming
    await provider.start_streaming()
    results = []
    async for result in provider.get_streaming_results():
        results.append(result)
        break  # Just get first result
    
    assert len(results) == 1
    assert results[0].text == "Streaming result"
    
    await provider.stop_streaming()
    await provider.cleanup()


@pytest.mark.asyncio
async def test_provider_validation():
    """Test provider configuration validation."""
    config = STTConfig(
        language=STTLanguage.ENGLISH,
        sample_rate=16000,
        chunk_size=1024
    )
    
    provider = MockSTTProvider(config)
    
    # Valid configuration should pass
    assert provider.validate_config() is True
    
    # Test invalid sample rate
    config.sample_rate = 999999  # Not supported
    provider.config = config
    
    with pytest.raises(ValueError):
        provider.validate_config()


@pytest.mark.asyncio 
async def test_health_check():
    """Test provider health check."""
    config = STTConfig()
    provider = MockSTTProvider(config)
    
    health = await provider.health_check()
    
    assert health["status"] == "healthy"
    assert health["provider"] == "mock_stt"


def test_stt_provider_factory():
    """Test STT provider factory."""
    # Register mock provider
    STTProviderFactory.register_provider("mock", MockSTTProvider)
    
    # Test provider creation
    config = STTConfig()
    provider = STTProviderFactory.create_provider("mock", config)
    
    assert isinstance(provider, MockSTTProvider)
    assert provider.provider_name == "mock_stt"
    
    # Test listing providers
    providers = STTProviderFactory.list_providers()
    assert "mock" in providers


if __name__ == "__main__":
    pytest.main([__file__])