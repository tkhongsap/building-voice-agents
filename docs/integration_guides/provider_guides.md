# AI Provider Integration Guides

This comprehensive guide covers integration with all major AI service providers supported by the LiveKit Voice Agents Platform. Each provider offers unique capabilities and pricing models, allowing you to choose the best combination for your use case.

## Table of Contents

- [Quick Start](#quick-start)
- [Speech-to-Text (STT) Providers](#speech-to-text-stt-providers)
- [Large Language Model (LLM) Providers](#large-language-model-llm-providers)
- [Text-to-Speech (TTS) Providers](#text-to-speech-tts-providers)
- [Voice Activity Detection (VAD) Providers](#voice-activity-detection-vad-providers)
- [Provider Comparison & Selection](#provider-comparison--selection)
- [Cost Optimization](#cost-optimization)
- [Troubleshooting](#troubleshooting)

## Quick Start

To use any AI provider with the Voice Agents Platform, you'll need:

1. **API Keys**: Obtain credentials from your chosen providers
2. **Configuration**: Set up provider-specific configurations
3. **Environment Variables**: Securely store your API keys
4. **SDK Integration**: Use the builder pattern to configure providers

### Basic Setup Example

```python
from voice_agents.sdk import VoiceAgentSDK, initialize_sdk

# Initialize SDK
sdk = await initialize_sdk({
    "project_name": "my_voice_agent",
    "environment": "production"
})

# Create agent with multiple providers
agent = (sdk.create_builder()
    .with_name("My Assistant")
    .with_stt("openai", language="en")           # OpenAI Whisper
    .with_llm("anthropic", model="claude-3")     # Anthropic Claude
    .with_tts("elevenlabs", voice="rachel")      # ElevenLabs TTS
    .with_vad("silero")                          # Silero VAD
    .build())
```

---

## Speech-to-Text (STT) Providers

### OpenAI Whisper

**Best for**: High accuracy, multiple languages, real-time streaming

#### Setup

```bash
# Install dependencies
pip install openai

# Set environment variable
export OPENAI_API_KEY="sk-your-api-key"
```

#### Configuration

```python
# Basic configuration
agent = builder.with_stt("openai", 
    language="en",
    model="whisper-1"
)

# Advanced configuration
agent = builder.with_stt("openai", {
    "language": "en",
    "model": "whisper-1",
    "temperature": 0.0,
    "response_format": "text",
    "timestamp_granularities": ["word"],
    "prompt": "Custom context for better accuracy"
})
```

#### Features
- **Languages**: 50+ languages supported
- **Real-time**: Streaming transcription available
- **Accuracy**: Industry-leading accuracy
- **Cost**: $0.006 per minute

#### Best Practices

```python
# For phone calls (lower quality audio)
stt_config = {
    "model": "whisper-1",
    "language": "en",
    "temperature": 0.2,  # Higher for noisy audio
    "prompt": "Phone conversation with potential background noise"
}

# For high-quality studio recordings
stt_config = {
    "model": "whisper-1", 
    "language": "en",
    "temperature": 0.0,  # Lower for clean audio
    "timestamp_granularities": ["word", "segment"]
}
```

### Azure Speech Services

**Best for**: Enterprise customers, regional compliance, custom models

#### Setup

```bash
pip install azure-cognitiveservices-speech

export AZURE_SPEECH_KEY="your-speech-key"
export AZURE_SPEECH_REGION="eastus"
```

#### Configuration

```python
# Basic Azure Speech
agent = builder.with_stt("azure", {
    "language": "en-US",
    "speech_key": os.getenv("AZURE_SPEECH_KEY"),
    "speech_region": os.getenv("AZURE_SPEECH_REGION")
})

# With custom model
agent = builder.with_stt("azure", {
    "language": "en-US",
    "endpoint_id": "your-custom-model-id",
    "phrase_list": ["custom", "vocabulary", "terms"],
    "profanity_filter": True
})
```

#### Features
- **Custom Models**: Train on your domain-specific data
- **Real-time**: Low-latency streaming
- **Compliance**: SOC, ISO, HIPAA certified
- **Cost**: $1 per hour (standard), custom pricing for models

### Google Cloud Speech-to-Text

**Best for**: Advanced features, automatic punctuation, speaker diarization

#### Setup

```bash
pip install google-cloud-speech

# Service account authentication
export GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account.json"
```

#### Configuration

```python
# Basic Google Speech
agent = builder.with_stt("google", {
    "language_code": "en-US",
    "model": "latest_long",
    "use_enhanced": True
})

# Advanced features
agent = builder.with_stt("google", {
    "language_code": "en-US",
    "model": "phone_call",
    "enable_automatic_punctuation": True,
    "enable_speaker_diarization": True,
    "diarization_speaker_count": 2,
    "enable_word_time_offsets": True,
    "profanity_filter": False,
    "speech_contexts": [{
        "phrases": ["LiveKit", "voice agents", "real-time"],
        "boost": 20.0
    }]
})
```

#### Features
- **Speaker Diarization**: Identify multiple speakers
- **Auto Punctuation**: Intelligent punctuation insertion
- **Enhanced Models**: Domain-specific optimizations
- **Cost**: $0.006-$0.009 per 15 seconds

---

## Large Language Model (LLM) Providers

### OpenAI GPT Models

**Best for**: General intelligence, function calling, consistent quality

#### Setup

```bash
pip install openai

export OPENAI_API_KEY="sk-your-api-key"
```

#### Configuration

```python
# GPT-4 Turbo (recommended)
agent = builder.with_llm("openai", {
    "model": "gpt-4-turbo",
    "temperature": 0.7,
    "max_tokens": 500,
    "top_p": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0
})

# GPT-4o for multimodal
agent = builder.with_llm("openai", {
    "model": "gpt-4o",
    "temperature": 0.7,
    "max_tokens": 300,
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    }
                }
            }
        }
    ]
})

# GPT-3.5 Turbo (cost-effective)
agent = builder.with_llm("openai", {
    "model": "gpt-3.5-turbo",
    "temperature": 0.5,
    "max_tokens": 150  # Lower for cost optimization
})
```

#### Model Selection Guide

| Model | Best Use Case | Cost | Speed | Quality |
|-------|--------------|------|-------|---------|
| GPT-4o | Multimodal, complex reasoning | High | Medium | Excellent |
| GPT-4 Turbo | Complex conversations | High | Fast | Excellent |
| GPT-3.5 Turbo | Simple conversations | Low | Very Fast | Good |

### Anthropic Claude

**Best for**: Safety, reasoning, large context windows

#### Setup

```bash
pip install anthropic

export ANTHROPIC_API_KEY="sk-ant-your-api-key"
```

#### Configuration

```python
# Claude 3 Sonnet (balanced)
agent = builder.with_llm("anthropic", {
    "model": "claude-3-sonnet-20240229",
    "max_tokens": 500,
    "temperature": 0.7,
    "top_p": 1.0,
    "top_k": 250
})

# Claude 3 Opus (highest quality)
agent = builder.with_llm("anthropic", {
    "model": "claude-3-opus-20240229",
    "max_tokens": 300,
    "temperature": 0.5,
    "system": "You are a helpful customer service representative."
})

# Claude 3 Haiku (fastest)
agent = builder.with_llm("anthropic", {
    "model": "claude-3-haiku-20240307",
    "max_tokens": 200,
    "temperature": 0.8
})
```

#### Features
- **Large Context**: Up to 200k tokens
- **Safety**: Built-in safety mechanisms
- **Reasoning**: Strong analytical capabilities
- **Cost**: Competitive pricing across models

### Google Gemini

**Best for**: Multimodal capabilities, Google ecosystem integration

#### Setup

```bash
pip install google-generativeai

export GOOGLE_API_KEY="your-gemini-api-key"
```

#### Configuration

```python
# Gemini Pro
agent = builder.with_llm("google", {
    "model": "gemini-pro",
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 500
})

# Gemini Pro Vision (multimodal)
agent = builder.with_llm("google", {
    "model": "gemini-pro-vision",
    "temperature": 0.4,
    "max_output_tokens": 300
})
```

### Local/Open Source Models

**Best for**: Data privacy, cost control, customization

#### Setup with Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull models
ollama pull llama2
ollama pull mistral
ollama pull codellama
```

#### Configuration

```python
# Llama 2
agent = builder.with_llm("local", {
    "model": "llama2",
    "base_url": "http://localhost:11434",
    "temperature": 0.8,
    "max_tokens": 400
})

# Mistral
agent = builder.with_llm("local", {
    "model": "mistral",
    "base_url": "http://localhost:11434",
    "temperature": 0.7,
    "num_predict": 300
})
```

---

## Text-to-Speech (TTS) Providers

### ElevenLabs

**Best for**: Voice quality, voice cloning, emotional expression

#### Setup

```bash
pip install elevenlabs

export ELEVENLABS_API_KEY="your-api-key"
```

#### Configuration

```python
# Pre-built voices
agent = builder.with_tts("elevenlabs", {
    "voice": "Rachel",  # or voice_id
    "model": "eleven_multilingual_v2",
    "stability": 0.5,
    "similarity_boost": 0.5,
    "style": 0.0,
    "use_speaker_boost": True
})

# Custom voice
agent = builder.with_tts("elevenlabs", {
    "voice_id": "your-custom-voice-id",
    "model": "eleven_turbo_v2",
    "stability": 0.75,
    "similarity_boost": 0.75,
    "optimize_streaming_latency": 1
})
```

#### Voice Selection

| Voice | Character | Best For |
|-------|-----------|----------|
| Rachel | Professional | Business, customer service |
| Josh | Casual | Friendly conversations |
| Arnold | Authoritative | Announcements, presentations |
| Bella | Warm | Healthcare, support |
| Antoni | Energetic | Marketing, sales |

### OpenAI TTS

**Best for**: Cost-effectiveness, quality balance, reliability

#### Setup

```bash
# Already included with OpenAI package
export OPENAI_API_KEY="sk-your-api-key"
```

#### Configuration

```python
# Standard voices
agent = builder.with_tts("openai", {
    "voice": "nova",  # alloy, echo, fable, onyx, nova, shimmer
    "model": "tts-1",
    "speed": 1.0,
    "response_format": "mp3"
})

# HD quality (higher cost)
agent = builder.with_tts("openai", {
    "voice": "nova",
    "model": "tts-1-hd",
    "speed": 0.95,
    "response_format": "opus"
})
```

### Azure Speech Services

**Best for**: Enterprise features, SSML support, neural voices

#### Setup

```bash
export AZURE_SPEECH_KEY="your-speech-key"
export AZURE_SPEECH_REGION="eastus"
```

#### Configuration

```python
# Neural voices
agent = builder.with_tts("azure", {
    "voice": "en-US-JennyNeural",
    "speech_key": os.getenv("AZURE_SPEECH_KEY"),
    "speech_region": os.getenv("AZURE_SPEECH_REGION"),
    "speech_synthesis_voice_name": "en-US-JennyNeural",
    "speech_synthesis_output_format": "audio-16khz-32kbitrate-mono-mp3"
})

# With SSML control
agent = builder.with_tts("azure", {
    "voice": "en-US-AriaNeural",
    "use_ssml": True,
    "rate": "+10%",
    "pitch": "+2Hz",
    "volume": "loud"
})
```

---

## Voice Activity Detection (VAD) Providers

### Silero VAD

**Best for**: Accuracy, multiple languages, offline capability

#### Configuration

```python
# Standard configuration
agent = builder.with_vad("silero", {
    "threshold": 0.5,
    "min_speech_duration_ms": 250,
    "max_speech_duration_s": 30,
    "min_silence_duration_ms": 100,
    "speech_pad_ms": 30
})

# Sensitive detection (for quiet speakers)
agent = builder.with_vad("silero", {
    "threshold": 0.3,
    "min_speech_duration_ms": 150,
    "min_silence_duration_ms": 50
})
```

### WebRTC VAD

**Best for**: Low latency, lightweight, web compatibility

#### Configuration

```python
# Basic WebRTC VAD
agent = builder.with_vad("webrtc", {
    "aggressiveness": 2,  # 0-3, higher = more aggressive
    "sample_rate": 16000,
    "frame_duration_ms": 30
})

# Aggressive mode (noisy environments)
agent = builder.with_vad("webrtc", {
    "aggressiveness": 3,
    "sample_rate": 16000,
    "frame_duration_ms": 10
})
```

---

## Provider Comparison & Selection

### Cost Comparison (USD, as of 2024)

| Provider | Service | Pricing |
|----------|---------|---------|
| **STT** |
| OpenAI | Whisper | $0.006/minute |
| Azure | Speech | $1.00/hour |
| Google | Speech-to-Text | $0.006/15s |
| **LLM** |
| OpenAI | GPT-4 Turbo | $0.01/1k input, $0.03/1k output |
| OpenAI | GPT-3.5 Turbo | $0.0005/1k input, $0.0015/1k output |
| Anthropic | Claude 3 Sonnet | $0.003/1k input, $0.015/1k output |
| Google | Gemini Pro | $0.0005/1k input, $0.0015/1k output |
| **TTS** |
| ElevenLabs | Standard | $0.18/1k characters |
| OpenAI | TTS | $0.015/1k characters |
| Azure | Neural TTS | $4.00/1M characters |

### Selection Matrix

#### For Customer Service
```python
# Recommended configuration
agent = (builder
    .with_stt("openai", language="en")      # High accuracy
    .with_llm("openai", model="gpt-4-turbo") # Reliable responses
    .with_tts("openai", voice="nova")        # Cost-effective
    .with_vad("silero", threshold=0.5)       # Accurate detection
)
```

#### For Healthcare/Telehealth
```python
# HIPAA-compliant configuration
agent = (builder
    .with_stt("azure", region="eastus")     # HIPAA compliance
    .with_llm("azure", model="gpt-4")       # Enterprise security
    .with_tts("azure", voice="JennyNeural") # Professional voice
    .with_vad("silero", threshold=0.4)      # Sensitive detection
)
```

#### For International/Multilingual
```python
# Multi-language configuration
agent = (builder
    .with_stt("google", language="auto")    # Auto language detection
    .with_llm("anthropic", model="claude-3") # Large context
    .with_tts("elevenlabs", voice="Rachel")  # High quality voices
    .with_vad("silero", threshold=0.5)       # Multi-language VAD
)
```

#### For Cost-Optimized
```python
# Budget-friendly configuration
agent = (builder
    .with_stt("openai", model="whisper-1")
    .with_llm("openai", model="gpt-3.5-turbo", max_tokens=150)
    .with_tts("openai", voice="alloy")
    .with_vad("webrtc", aggressiveness=2)
)
```

---

## Cost Optimization

### Token Management

```python
# Optimize LLM costs
llm_config = {
    "model": "gpt-3.5-turbo",
    "max_tokens": 100,        # Limit response length
    "temperature": 0.3,       # More focused responses
    "top_p": 0.9,            # Reduce randomness
    "frequency_penalty": 0.1, # Reduce repetition
    "presence_penalty": 0.1   # Encourage conciseness
}
```

### Caching Strategies

```python
# Enable response caching
sdk_config = {
    "enable_response_cache": True,
    "cache_ttl_seconds": 3600,
    "cache_similar_queries": True,
    "similarity_threshold": 0.85
}
```

### Provider Failover

```python
# Set up provider fallback
agent = (builder
    .with_llm("openai", model="gpt-4-turbo")
    .with_fallback_llm("openai", model="gpt-3.5-turbo")  # Cheaper fallback
    .with_tts("elevenlabs", voice="Rachel")
    .with_fallback_tts("openai", voice="nova")           # Backup TTS
)
```

### Usage Monitoring

```python
# Monitor usage and costs
from voice_agents.monitoring import CostTracker

cost_tracker = CostTracker()

# Set budget alerts
cost_tracker.set_budget_alert(
    monthly_limit=100.00,  # $100/month
    alert_thresholds=[0.5, 0.8, 0.9]  # 50%, 80%, 90%
)

# Track usage by provider
usage_report = cost_tracker.get_usage_report(
    period="current_month",
    group_by="provider"
)
```

---

## Troubleshooting

### Common Issues and Solutions

#### Authentication Errors

```python
# Verify API keys
import os

def check_api_keys():
    required_keys = {
        "OpenAI": "OPENAI_API_KEY",
        "Anthropic": "ANTHROPIC_API_KEY", 
        "ElevenLabs": "ELEVENLABS_API_KEY",
        "Azure Speech": "AZURE_SPEECH_KEY"
    }
    
    for provider, key_name in required_keys.items():
        if not os.getenv(key_name):
            print(f"❌ Missing {provider} API key: {key_name}")
        else:
            print(f"✅ {provider} API key found")

check_api_keys()
```

#### Rate Limiting

```python
# Configure rate limiting and retries
sdk_config = {
    "rate_limiting": {
        "requests_per_minute": 60,
        "retry_attempts": 3,
        "backoff_factor": 2,
        "respect_provider_limits": True
    }
}
```

#### Quality Issues

```python
# Debug poor transcription
stt_debug_config = {
    "model": "whisper-1",
    "language": "en",
    "temperature": 0.0,  # Lower for noisy audio
    "prompt": "Include context about expected content",
    "enable_debug_logging": True
}

# Debug poor responses
llm_debug_config = {
    "model": "gpt-4-turbo",
    "temperature": 0.1,  # Lower for consistency
    "max_tokens": 200,
    "enable_debug_logging": True,
    "log_prompts": True  # Log full prompts for debugging
}
```

#### Latency Optimization

```python
# Optimize for speed
speed_config = {
    "stt": {
        "provider": "openai",
        "streaming": True,
        "chunk_size": 1024
    },
    "llm": {
        "provider": "openai", 
        "model": "gpt-3.5-turbo",
        "max_tokens": 100,
        "stream": True
    },
    "tts": {
        "provider": "openai",
        "model": "tts-1",  # Faster model
        "optimize_streaming_latency": True
    }
}
```

### Provider Health Checks

```python
async def health_check_providers():
    """Check health of all configured providers."""
    from voice_agents.health import ProviderHealthChecker
    
    checker = ProviderHealthChecker()
    
    results = await checker.check_all([
        "openai_stt",
        "openai_llm", 
        "openai_tts",
        "elevenlabs_tts",
        "anthropic_llm"
    ])
    
    for provider, status in results.items():
        if status["healthy"]:
            print(f"✅ {provider}: {status['latency']:.2f}ms")
        else:
            print(f"❌ {provider}: {status['error']}")
```

### Logging and Debugging

```python
# Enable comprehensive logging
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('voice_agent.log'),
        logging.StreamHandler()
    ]
)

# Provider-specific debug settings
debug_config = {
    "openai": {
        "log_requests": True,
        "log_responses": True,
        "log_audio_metadata": True
    },
    "elevenlabs": {
        "log_voice_settings": True,
        "log_generation_metadata": True
    }
}
```

---

## Best Practices Summary

1. **Start Simple**: Begin with OpenAI providers for quick setup
2. **Test Extensively**: Use different audio conditions and languages
3. **Monitor Costs**: Set up budget alerts and usage tracking
4. **Plan for Scale**: Configure rate limiting and caching
5. **Handle Failures**: Implement fallback providers
6. **Optimize Quality**: Fine-tune parameters for your use case
7. **Stay Updated**: Providers frequently release new models and features

For the latest provider updates and new integrations, check our [changelog](../CHANGELOG.md) and [community forum](https://community.livekit.io).