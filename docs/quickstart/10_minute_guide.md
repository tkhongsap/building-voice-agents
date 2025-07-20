# 🚀 10-Minute Quickstart: Build Your First Voice Agent

Welcome! In just 10 minutes, you'll have a working voice agent that can have conversations using speech-to-text, language models, and text-to-speech. Let's get started!

## Prerequisites ⚡

- Python 3.8+
- API keys for:
  - OpenAI (for LLM and STT)
  - ElevenLabs (for TTS) *or* OpenAI TTS
- 5-10 minutes of your time

## Step 1: Installation (30 seconds)

```bash
pip install livekit-voice-agents
```

## Step 2: Setup Environment (1 minute)

Create a `.env` file in your project directory:

```bash
# Required API Keys
OPENAI_API_KEY=sk-your-openai-key-here
ELEVENLABS_API_KEY=your-elevenlabs-key-here

# Optional: Customize default settings
VOICE_AGENT_LOG_LEVEL=INFO
VOICE_AGENT_PROJECT_NAME=my-first-agent
```

## Step 3: Hello World Agent (2 minutes)

Create `hello_world.py`:

```python
import asyncio
from livekit_voice_agents import initialize_sdk, VoiceAgentBuilder

async def main():
    # Initialize the SDK
    await initialize_sdk()
    
    # Build your first agent
    agent = (VoiceAgentBuilder()
        .with_name("Hello World Agent")
        .with_stt("openai", language="en")
        .with_llm("openai", model="gpt-4")
        .with_tts("elevenlabs")
        .with_system_prompt("You are a friendly AI assistant. Keep responses short and conversational.")
        .build())
    
    # Start the agent
    await agent.start()
    print("🎉 Your voice agent is running!")
    print("Say something and watch the magic happen...")
    
    # Keep running (Ctrl+C to stop)
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("Stopping agent...")
        await agent.stop()
        await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
```

Run it:
```bash
python hello_world.py
```

🎉 **Congratulations!** You now have a working voice agent!

## Step 4: Add Conversation Features (3 minutes)

Let's make it smarter with interruption handling and context management:

```python
import asyncio
from livekit_voice_agents import (
    initialize_sdk, 
    VoiceAgentBuilder, 
    AgentCapability
)

async def main():
    await initialize_sdk()
    
    # Enhanced agent with conversation features
    agent = (VoiceAgentBuilder()
        .with_name("Smart Conversation Agent")
        .with_stt("openai", language="en")
        .with_llm("openai", model="gpt-4", temperature=0.7)
        .with_tts("elevenlabs", voice_id="21m00Tcm4TlvDq8ikWAM")
        .with_vad("silero")  # Voice activity detection
        .with_capabilities(
            AgentCapability.TURN_DETECTION,
            AgentCapability.INTERRUPTION_HANDLING,
            AgentCapability.CONTEXT_MANAGEMENT
        )
        .with_system_prompt(
            "You are a helpful assistant. Remember our conversation context "
            "and be responsive to interruptions. Keep responses conversational."
        )
        .build())
    
    # Add event callbacks
    agent.on_user_speech(lambda text: print(f"User said: {text}"))
    agent.on_agent_speech(lambda text: print(f"Agent responding: {text}"))
    
    await agent.start()
    print("🤖 Smart agent ready! Try interrupting it mid-sentence.")
    
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        await agent.stop()
        await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
```

## Step 5: Use Quick Templates (2 minutes)

For common use cases, use our pre-built templates:

```python
import asyncio
from livekit_voice_agents import initialize_sdk, QuickBuilder

async def main():
    await initialize_sdk()
    
    # Choose a pre-configured template
    agent = QuickBuilder.customer_service_agent(
        name="Support Bot",
        language="en"
    ).build()
    
    await agent.start()
    print("📞 Customer service agent ready!")
    
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        await agent.stop()
        await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
```

### Available Quick Templates:
- `QuickBuilder.customer_service_agent()` - Customer support
- `QuickBuilder.telehealth_agent()` - Medical consultation assistant  
- `QuickBuilder.translator_agent()` - Real-time translation

## Step 6: Configuration & Customization (2 minutes)

Create a `config.yaml` for advanced settings:

```yaml
# config.yaml
project_name: "my-voice-agent"
environment: "development"
log_level: "DEBUG"

# Pipeline settings
pipeline:
  sample_rate: 16000
  latency_mode: "low"
  enable_echo_cancellation: true

# Provider settings
openai:
  model: "gpt-4"
  temperature: 0.7
  max_tokens: 150

elevenlabs:
  voice_id: "21m00Tcm4TlvDq8ikWAM"
  optimize_streaming_latency: 2

# Server settings (for web deployment)
server:
  host: "0.0.0.0"
  port: 8080
  enable_cors: true
```

Use the config file:

```python
import asyncio
from livekit_voice_agents import initialize_sdk, VoiceAgentBuilder

async def main():
    # Initialize with custom config
    await initialize_sdk(config="config.yaml")
    
    agent = (VoiceAgentBuilder()
        .with_name("Configured Agent")
        .with_stt("openai")
        .with_llm("openai")
        .with_tts("elevenlabs")
        .build())
    
    await agent.start()
    print("Agent running with custom configuration!")
    
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        await agent.stop()
        await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
```

## 🎯 What You've Accomplished

In just 10 minutes, you've:

✅ **Built a working voice agent** with speech recognition and synthesis  
✅ **Added smart conversation features** like interruption handling  
✅ **Used quick templates** for common scenarios  
✅ **Customized configuration** for your specific needs  

## 🚀 Next Steps

Ready to go deeper? Check out:

- **[Complete Examples](../examples/)** - More complex use cases
- **[Provider Guides](../integration_guides/)** - Integrate different AI services
- **[API Reference](../api/)** - Full SDK documentation
- **[Recipes](../recipes/)** - Pre-built solutions for specific industries

## 📚 Common Patterns

### Error Handling
```python
try:
    await agent.start()
except ConfigurationError as e:
    print(f"Configuration error: {e}")
except AuthenticationError as e:
    print(f"API key issue: {e}")
```

### Custom Callbacks
```python
async def on_error(error):
    print(f"Agent error: {error}")
    # Add your error handling logic

agent = (VoiceAgentBuilder()
    .with_name("Error-Handled Agent")
    # ... other config
    .on_error(on_error)
    .build())
```

### Development Server
```python
from livekit_voice_agents import create_dev_server

# For web development
server = create_dev_server(agent)
await server.run(host="localhost", port=8080)
```

## 🔧 Troubleshooting

**Agent not responding?**
- Check your API keys in `.env`
- Verify your microphone permissions
- Check the logs with `log_level: "DEBUG"`

**Audio issues?**
- Try different audio devices
- Adjust `sample_rate` in pipeline config
- Enable echo cancellation

**Performance slow?**
- Use `latency_mode: "low"` in pipeline
- Try OpenAI TTS instead of ElevenLabs for faster response
- Reduce `max_tokens` for shorter responses

## 💡 Pro Tips

1. **Start Simple**: Begin with basic STT→LLM→TTS, then add features
2. **Use Templates**: QuickBuilder templates save setup time
3. **Configure Gradually**: Start with defaults, then customize
4. **Monitor Performance**: Use built-in metrics and logging
5. **Test Different Voices**: Try various TTS voices for your use case

---

**Need help?** Check our [troubleshooting guide](../troubleshooting/) or join our community Discord.

**Ready for production?** See our [deployment guide](../deployment/) for scaling your agent.