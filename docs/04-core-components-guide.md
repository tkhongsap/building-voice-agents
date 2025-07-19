# Core Components Implementation Guide

## Overview

This guide provides detailed implementation instructions for each core component of a voice agent pipeline: Speech-to-Text (STT), Large Language Models (LLM), Text-to-Speech (TTS), and Voice Activity Detection (VAD).

## Component Architecture Overview

```python
# Basic voice agent structure
class VoiceAgent(Agent):
    def __init__(self):
        # Initialize all components
        stt = openai.STT()           # Speech-to-Text
        llm = openai.LLM()           # Large Language Model  
        tts = elevenlabs.TTS()       # Text-to-Speech
        vad = silero.VAD.load()      # Voice Activity Detection
        
        super().__init__(
            instructions="Your agent instructions",
            stt=stt,
            llm=llm, 
            tts=tts,
            vad=vad
        )
```

## 1. Speech-to-Text (STT) Implementation

### Component Purpose
- Converts spoken audio into text transcription
- Handles real-time streaming audio processing
- Supports multiple languages and domain vocabularies

### Implementation Options

#### OpenAI Whisper (Recommended)
```python
from livekit.plugins import openai

# Basic configuration
stt = openai.STT(
    model="whisper-1",
    language=None,  # Auto-detect language
)

# Advanced configuration
stt = openai.STT(
    model="whisper-1",
    language="en",  # Force English
    temperature=0.0,  # Deterministic output
)
```

#### Configuration Parameters
- **model**: Model version ("whisper-1" is current)
- **language**: Language code or None for auto-detection
- **temperature**: Controls randomness (0.0-1.0)

### STT Best Practices

#### Language Detection
```python
# Auto-detection (recommended for multi-language)
stt = openai.STT(language=None)

# Fixed language (better accuracy for single language)
stt = openai.STT(language="en")  # English only
```

#### Domain-Specific Optimization
For specialized vocabularies (medical, legal, technical):
```python
# Use domain-specific prompts in LLM layer
instructions = """
You are a medical voice assistant. 
Common terms include: stethoscope, diagnosis, symptoms, prescription...
"""
```

### Troubleshooting STT Issues

**Problem**: Inaccurate transcription
**Solutions**:
- Use fixed language instead of auto-detection
- Ensure good audio quality and minimal background noise
- Consider domain-specific fine-tuning

**Problem**: High latency
**Solutions**:
- Verify streaming is enabled
- Check network connectivity
- Consider switching to faster STT providers

## 2. Large Language Model (LLM) Implementation

### Component Purpose
- Generates intelligent responses based on transcribed text
- Maintains conversation context and history
- Handles reasoning and task-specific logic

### Implementation Options

#### OpenAI GPT Models
```python
from livekit.plugins import openai

# Standard configuration (balanced performance)
llm = openai.LLM(
    model="gpt-4o-mini",  # Fast, cost-effective
    temperature=0.7,
)

# High-quality configuration
llm = openai.LLM(
    model="gpt-4o",  # Best quality, higher latency
    temperature=0.7,
)

# Low-latency configuration
llm = openai.LLM(
    model="gpt-4o-mini",
    temperature=0.3,  # More deterministic
)
```

### Model Selection Guide

| Model | Latency | Quality | Cost | Use Case |
|-------|---------|---------|------|----------|
| gpt-4o-mini | Fast (~400ms TTFT) | Good | Low | General conversation |
| gpt-4o | Slower (~800ms TTFT) | Excellent | High | Complex reasoning |

### LLM Configuration Best Practices

#### Instructions Design
```python
# Good: Specific, voice-optimized instructions
instructions = """
You are a helpful AI assistant communicating via voice.
Keep responses conversational and under 50 words.
Be warm, professional, and helpful.
If you don't understand, ask for clarification.
"""

# Avoid: Long instructions that increase latency
```

#### Context Management
```python
# The Agent class automatically handles:
# - Conversation history
# - Message threading
# - Context window management
# - Turn-taking state
```

### Advanced LLM Features

#### Custom Response Handling
```python
class CustomVoiceAssistant(Agent):
    async def before_llm_response(self, user_input: str) -> Optional[str]:
        """Pre-process user input before LLM"""
        # Handle specific commands
        if user_input.lower().strip() == "what is your name":
            return "I'm your friendly AI voice assistant!"
        return None
    
    async def after_llm_response(self, response: str) -> str:
        """Post-process LLM response before TTS"""
        # Add response filtering or modification
        return response
```

## 3. Text-to-Speech (TTS) Implementation

### Component Purpose
- Converts generated text into natural-sounding speech
- Supports voice customization and personality
- Handles real-time audio streaming

### Implementation Options

#### ElevenLabs TTS (Recommended)
```python
from livekit.plugins import elevenlabs

# Default voice (Rachel)
tts = elevenlabs.TTS()

# Custom voice selection
tts = elevenlabs.TTS(
    voice_id="CwhRBWXzGAHq8TQ4Fs17",  # Roger
    model="eleven_monolingual_v1",
    encoding="mp3_44100_128",
)
```

### Available Voice Options

```python
# ElevenLabs voice IDs from course
VOICES = {
    "rachel": "21m00Tcm4TlvDq8ikWAM",  # Default, natural female
    "roger": "CwhRBWXzGAHq8TQ4Fs17",   # Professional male
    "sarah": "EXAVITQu4vr4xnSDxMaL",   # Warm female
    "laura": "FGY2WhTYpPnrIDTdsKH5",   # Friendly female
    "george": "JBFqnCBsd6RMkjVDRZzb",  # Authoritative male
}

# Usage
tts = elevenlabs.TTS(voice_id=VOICES["roger"])
```

### Voice Customization

#### Voice Selection Criteria
- **Natural Conversation**: Rachel (default)
- **Professional/Business**: Roger
- **Warm/Empathetic**: Sarah
- **Friendly/Casual**: Laura
- **Authoritative**: George

#### Custom Voice Training
```python
# For custom voice cloning (enterprise feature)
tts = elevenlabs.TTS(
    voice_id="your_custom_voice_id",
    model="eleven_multilingual_v2",  # For multiple languages
)
```

### TTS Performance Optimization

#### Key Metrics
- **Time to First Byte (TTFB)**: Critical for perceived responsiveness
- **Streaming**: Essential for real-time conversation
- **Audio Quality**: Balance quality vs latency

#### Configuration Tips
```python
# Optimized for speed
tts = elevenlabs.TTS(
    voice_id="21m00Tcm4TlvDq8ikWAM",
    model="eleven_monolingual_v1",  # Faster than multilingual
    encoding="mp3_44100_128",       # Good quality/speed balance
)
```

## 4. Voice Activity Detection (VAD) Implementation

### Component Purpose
- Identifies when human speech is present in audio
- Reduces false transcriptions and processing costs
- Enables turn detection and interruption handling

### Implementation

#### Silero VAD (Recommended)
```python
from livekit.plugins import silero

# Basic configuration
vad = silero.VAD.load()

# Advanced configuration
vad = silero.VAD.load(
    min_speech_duration=0.1,    # Minimum speech to trigger (100ms)
    min_silence_duration=0.5,   # Silence before turn end (500ms)  
    speech_threshold=0.5,       # Confidence threshold
)
```

### VAD Configuration Parameters

#### Timing Parameters
- **min_speech_duration**: Minimum speech length to consider valid
  - Too low: False positives from noise
  - Too high: Misses short utterances
  - Recommended: 0.1s (100ms)

- **min_silence_duration**: Silence duration before ending turn
  - Too low: Cuts off speakers who pause
  - Too high: Delayed responses
  - Recommended: 0.3-0.5s

- **speech_threshold**: Confidence threshold for speech detection
  - Too low: Background noise triggers
  - Too high: Misses quiet speech
  - Recommended: 0.5

### VAD Optimization

#### Environment-Specific Tuning
```python
# Noisy environment
vad = silero.VAD.load(
    speech_threshold=0.7,       # Higher threshold
    min_speech_duration=0.2,    # Longer minimum
)

# Quiet environment  
vad = silero.VAD.load(
    speech_threshold=0.3,       # Lower threshold
    min_speech_duration=0.05,   # Shorter minimum
)
```

## Complete Integration Example

### Basic Voice Agent
```python
import logging
from dotenv import load_dotenv
from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions
from livekit.plugins import openai, elevenlabs, silero

# Load environment variables
load_dotenv()

class VoiceAssistant(Agent):
    def __init__(self) -> None:
        # Initialize components with optimized settings
        llm = openai.LLM(
            model="gpt-4o-mini",  # Fast response
            temperature=0.7,
        )
        
        stt = openai.STT(
            model="whisper-1",
            language=None,  # Auto-detect
        )
        
        tts = elevenlabs.TTS(
            voice_id="21m00Tcm4TlvDq8ikWAM",  # Rachel
        )
        
        vad = silero.VAD.load(
            min_speech_duration=0.1,
            min_silence_duration=0.5,
            speech_threshold=0.5,
        )

        super().__init__(
            instructions="""
                You are a helpful assistant communicating via voice.
                Keep responses conversational and under 50 words.
                Be warm, professional, and helpful.
            """,
            stt=stt,
            llm=llm,
            tts=tts,
            vad=vad,
        )

# Entry point function
async def entrypoint(ctx: JobContext):
    await ctx.connect()
    session = AgentSession()
    await session.start(
        room=ctx.room,
        agent=VoiceAssistant()
    )
```

### Voice Customization Example
```python
class CustomVoiceAssistant(VoiceAssistant):
    def __init__(self, voice_persona="professional"):
        # Voice persona configurations
        voice_configs = {
            "professional": {
                "voice_id": "CwhRBWXzGAHq8TQ4Fs17",  # Roger
                "instructions": "You are a professional business assistant.",
            },
            "friendly": {
                "voice_id": "EXAVITQu4vr4xnSDxMaL",  # Sarah
                "instructions": "You are a warm, friendly assistant.",
            },
            "authoritative": {
                "voice_id": "JBFqnCBsd6RMkjVDRZzb",  # George
                "instructions": "You are an authoritative expert assistant.",
            }
        }
        
        config = voice_configs.get(voice_persona, voice_configs["professional"])
        
        # Override TTS with selected voice
        self.tts = elevenlabs.TTS(voice_id=config["voice_id"])
        
        super().__init__()
        self.instructions = config["instructions"]
```

## Provider Alternatives

### STT Alternatives
- **Google Cloud Speech-to-Text**: Good accuracy, competitive pricing
- **Azure Speech Services**: Strong enterprise features
- **AWS Transcribe**: Good AWS ecosystem integration

### LLM Alternatives
- **Anthropic Claude**: Strong safety features
- **Groq**: Ultra-fast inference
- **Together AI**: Cost-effective open-source models
- **Cerebras**: Extremely fast inference

### TTS Alternatives
- **OpenAI TTS**: Simple integration, good quality
- **Google Cloud Text-to-Speech**: Wide language support
- **Azure Speech Services**: Enterprise features
- **AWS Polly**: Good AWS integration

### VAD Alternatives
- **WebRTC VAD**: Built into browsers
- **Custom models**: Domain-specific training

## Troubleshooting Common Issues

### Audio Quality Problems
**Symptoms**: Poor transcription, robotic voice output
**Solutions**:
- Check microphone quality and positioning
- Minimize background noise
- Verify audio encoding settings
- Test with different devices

### Latency Issues
**Symptoms**: Delayed responses, choppy audio
**Solutions**:
- Switch to faster models (gpt-4o-mini vs gpt-4o)
- Optimize network connectivity
- Use edge deployments
- Enable streaming for all components

### Context Loss
**Symptoms**: Agent forgets previous conversation
**Solutions**:
- Verify Agent class is maintaining history
- Check context window limits
- Implement custom context management if needed

## Next Steps

1. **Performance Optimization**: Advanced metrics and optimization in [Performance Guide](07-performance-optimization.md)
2. **Turn Detection**: Deep dive into conversation flow in [Turn Detection Guide](06-turn-detection-guide.md)
3. **Complete Tutorial**: Step-by-step implementation in [Implementation Tutorial](05-implementation-tutorial.md)
4. **Real-World Applications**: Domain-specific guides in [Applications Guide](08-applications-guide.md)

## Key Takeaways

- **Component Selection**: Balance performance, quality, and cost based on use case
- **Configuration Matters**: Fine-tune parameters for your specific environment
- **Voice Personality**: TTS voice selection significantly impacts user experience
- **Real-time Processing**: Enable streaming for all components to minimize latency
- **Environment Variables**: Use secure configuration for API keys and settings