# OpenAI-Focused Voice Processing Pipeline Configuration

This guide shows how to configure Task 1.0 Core Voice Processing Pipeline to work primarily with OpenAI services for cost-effective voice agent development.

## 🔑 Required API Keys

For a minimal OpenAI-focused setup, you only need:

- **OpenAI API Key** - For STT (Whisper) and LLM (GPT-4.1-mini)

## 📱 Recommended Models

### STT (Speech-to-Text)
- **Primary**: `whisper-1` (OpenAI Whisper)
- **Cost**: $0.006 per minute of audio
- **Features**: 99+ languages, high accuracy, streaming support

### LLM (Large Language Model) 
- **Primary**: `gpt-4.1-mini` ⭐ **RECOMMENDED**
- **Cost**: ~60% cheaper than GPT-4o
- **Features**: Better performance, multimodal capabilities, function calling
- **Alternative**: `gpt-4o-mini` (older but still good)

### TTS (Text-to-Speech)
- **Primary**: ElevenLabs (best quality, requires ElevenLabs API key)
- **Fallback**: OpenAI TTS (cost-effective, uses same OpenAI API key)

### VAD (Voice Activity Detection)
- **Primary**: WebRTC VAD (no API key needed)
- **Alternative**: Silero VAD (local model, no API key needed)

## ⚙️ Configuration Examples

### Minimal OpenAI Setup

```python
from components.stt.openai_stt import OpenAISTTProvider, OpenAISTTConfig
from components.llm.openai_llm import OpenAILLMProvider, OpenAILLMConfig
from components.vad.webrtc_vad import WebRTCVADProvider, WebRTCVADConfig
from components.llm.base_llm import LLMModelType

# STT Configuration
stt_config = OpenAISTTConfig(
    api_key="your-openai-api-key",
    model="whisper-1",
    language="en",
    sample_rate=16000
)

# LLM Configuration  
llm_config = OpenAILLMConfig(
    api_key="your-openai-api-key",
    model=LLMModelType.GPT_4_1_MINI,  # Latest and most cost-effective
    temperature=0.7,
    max_tokens=1000
)

# VAD Configuration (no API key needed)
vad_config = WebRTCVADConfig(
    aggressiveness=2,  # 0-3, higher = more aggressive
    sample_rate=16000
)

# TTS Configuration - ElevenLabs primary, OpenAI fallback
from components.tts.elevenlabs_tts import ElevenLabsTTSProvider, ElevenLabsTTSConfig
from components.tts.openai_tts import OpenAITTSProvider, OpenAITTSConfig

# Primary TTS: ElevenLabs (best quality)
elevenlabs_config = ElevenLabsTTSConfig(
    api_key="your-elevenlabs-api-key",
    voice_id="your-preferred-voice-id",
    model="eleven_monolingual_v1"
)

# Fallback TTS: OpenAI (cost-effective)
openai_tts_config = OpenAITTSConfig(
    api_key="your-openai-api-key",  # Same as STT/LLM
    model="tts-1",  # or "tts-1-hd" for higher quality
    voice="alloy",  # alloy, echo, fable, onyx, nova, shimmer
    response_format="mp3"
)

# Create providers
stt_provider = OpenAISTTProvider(stt_config)
llm_provider = OpenAILLMProvider(llm_config)  
tts_provider = ElevenLabsTTSProvider(elevenlabs_config)  # Primary
tts_fallback = OpenAITTSProvider(openai_tts_config)     # Fallback
vad_provider = WebRTCVADProvider(vad_config)
```

### Environment Configuration (.env)

```bash
# Required for OpenAI pipeline
OPENAI_API_KEY=sk-proj-your-openai-api-key-here

# Required for best TTS quality
ELEVENLABS_API_KEY=your-elevenlabs-key
# OpenAI TTS automatically uses OPENAI_API_KEY as fallback
```

## 🚀 Complete Pipeline Setup

```python
from pipeline.audio_pipeline import StreamingAudioPipeline, PipelineConfig, PipelineMode

# Pipeline configuration
pipeline_config = PipelineConfig(
    sample_rate=16000,
    mode=PipelineMode.CONTINUOUS,
    enable_streaming_stt=True,
    enable_streaming_llm=True,
    latency_target_ms=236  # Our performance target
)

# Create the pipeline
pipeline = StreamingAudioPipeline(
    config=pipeline_config,
    stt_provider=stt_provider,      # OpenAI Whisper
    llm_provider=llm_provider,      # OpenAI GPT-4.1-mini
    tts_provider=tts_provider,      # ElevenLabs primary
    vad_provider=vad_provider       # WebRTC VAD
)

# Initialize and start
await pipeline.initialize()
await pipeline.start()
```

## 💰 Cost Optimization Tips

### Model Selection
- **GPT-4.1-mini**: Latest, best performance/cost ratio
- **Whisper**: Cost-effective STT, charge per minute of audio
- **WebRTC VAD**: Free, no API costs

### Usage Optimization
```python
# Optimize for cost
llm_config = OpenAILLMConfig(
    model=LLMModelType.GPT_4_1_MINI,
    max_tokens=500,        # Limit response length
    temperature=0.3,       # Lower temperature = more focused responses
    top_p=0.9             # Nucleus sampling for efficiency
)

# Optimize STT usage
stt_config = OpenAISTTConfig(
    model="whisper-1",
    language="en",         # Specify language for better accuracy
    response_format="json" # Structured response
)
```

## 🛡️ Fallback Configuration

Even with an OpenAI-focused setup, configure fallbacks for production:

```python
from components.error_handling import global_error_handler

# Configure fallback order
global_error_handler.fallback_configs["llm"].fallback_providers = [
    "openai_gpt",      # Primary
    "anthropic_claude", # Fallback 1 (if you have Anthropic API key)
    "local_llm"        # Fallback 2 (local model)
]

global_error_handler.fallback_configs["tts"].fallback_providers = [
    "elevenlabs",      # Primary (best quality)
    "openai"           # Fallback (cost-effective, same API key)
]

global_error_handler.fallback_configs["stt"].fallback_providers = [
    "openai_whisper"   # Primary (can add others if needed)
]
```

## 📊 Performance Monitoring

```python
from monitoring.performance_monitor import global_performance_monitor

# Start monitoring
await global_performance_monitor.start_monitoring()

# Register components
stt_monitor = global_performance_monitor.register_component("openai_stt", "stt")
llm_monitor = global_performance_monitor.register_component("openai_llm", "llm")

# Get performance metrics
async with llm_monitor.monitor_operation("generate_response"):
    response = await llm_provider.generate_response(messages)

# View metrics
summary = global_performance_monitor.get_overall_summary()
print(f"Average latency: {summary['components']['openai_llm']['avg_duration_ms']}ms")
```

## 🔧 Testing Your Setup

Run our validation script to test your OpenAI configuration:

```bash
# Test basic structure and API connectivity
python3 test_task1_structure.py

# Test with real OpenAI API (requires dependencies)
python3 test_openai_integration.py
```

## 📈 Expected Performance

With this OpenAI-focused configuration, you should achieve:

- **STT Latency**: ~100-300ms (Whisper)
- **LLM Latency**: ~200-800ms (GPT-4.1-mini)  
- **Total Pipeline**: <236ms target (excluding TTS)
- **Accuracy**: 95%+ STT accuracy, high-quality LLM responses
- **Cost**: Significantly lower than GPT-4o while maintaining quality

## 🌟 Benefits of This Setup

✅ **Best Quality**: ElevenLabs for premium voice synthesis
✅ **Cost-Effective**: OpenAI TTS fallback using same API key  
✅ **High Performance**: GPT-4.1-mini ~60% cheaper than GPT-4o
✅ **Simplified Stack**: Only 2 TTS providers instead of 4
✅ **Minimal API Keys**: OpenAI + ElevenLabs covers everything
✅ **Production Ready**: Full error handling and monitoring
✅ **Optimal Fallback**: Quality → Cost-effective progression

This configuration gives you the best of both worlds: **premium quality with cost-effective fallbacks** in a simplified, production-ready stack.