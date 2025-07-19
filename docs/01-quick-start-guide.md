# Quick Start Guide

**📖 Learning Path Navigation:**
[← Complete Guide](00-complete-voice-agent-guide.md) | **01 Quick Start** | [02 Setup Guide →](02-setup-guide.md)

**📚 All Guides:** [Complete Guide](00-complete-voice-agent-guide.md) | [01 Quick Start](01-quick-start-guide.md) | [02 Setup](02-setup-guide.md) | [03 Architecture](03-voice-agent-architecture.md) | [04 Components](04-core-components-guide.md) | [05 Implementation](05-implementation-tutorial.md) | [06 Turn Detection](06-turn-detection-guide.md) | [07 Performance](07-performance-optimization.md) | [08 Applications](08-applications-guide.md) | [09 LiveKit Reference](09-livekit-reference.md)

## Overview

Get your first voice agent running in under 15 minutes! This guide provides the fastest path from zero to a working voice agent using the LiveKit Agents framework, based on the DeepLearning.AI course materials.

## Prerequisites

- Python 3.10.11 or later
- OpenAI API key
- ElevenLabs API key
- LiveKit server (we'll use cloud instance)

## 1. Environment Setup (3 minutes)

### Install Python Dependencies
```bash
# Create virtual environment
python -m venv voice-agent-env
source voice-agent-env/bin/activate  # On Windows: voice-agent-env\Scripts\activate

# Install LiveKit Agents
pip install livekit-agents[all]==1.0.11

# Verify installation
python -c "import livekit.agents; print('✅ Installation successful')"
```

### Set Environment Variables
```bash
# Create .env file
cat > .env << EOF
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your-api-key
LIVEKIT_API_SECRET=your-api-secret
OPENAI_API_KEY=your-openai-key
ELEVENLABS_API_KEY=your-elevenlabs-key
EOF

# Load environment variables
export $(cat .env | xargs)
```

### Get Free LiveKit Cloud Account
1. Visit [livekit.io](https://livekit.io) and sign up
2. Create a new project
3. Copy the WebSocket URL, API key, and secret

## 2. Minimal Voice Agent (5 minutes)

### Create Basic Agent
```python
# agent.py
import asyncio
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli
from livekit.agents.llm import openai
from livekit.agents.stt import openai as openai_stt
from livekit.agents.tts import elevenlabs
from livekit.agents.vad import silero
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.agents.llm import ChatContext, ChatMessage

async def entrypoint(ctx: JobContext):
    """Main agent entry point"""
    
    # Initialize components
    initial_ctx = ChatContext(
        messages=[
            ChatMessage(
                role="system",
                content="You are a helpful voice assistant. Keep responses short and natural."
            )
        ]
    )
    
    # Create voice assistant
    assistant = VoiceAssistant(
        vad=silero.VAD.load(),
        stt=openai_stt.STT(),
        llm=openai.LLM(),
        tts=elevenlabs.TTS(),
        chat_ctx=initial_ctx,
    )
    
    # Connect to room
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    
    # Start assistant
    assistant.start(ctx.room)
    
    # Greet user
    await assistant.say("Hi there! I'm your voice assistant. How can I help you today?")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
```

### Test Your Agent
```bash
# Start the agent
python agent.py start

# You should see:
# ✅ Agent started successfully
# 📞 Waiting for connections on wss://your-project.livekit.cloud
```

## 3. Test with Web Client (3 minutes)

### Simple HTML Test Client
```html
<!-- test-client.html -->
<!DOCTYPE html>
<html>
<head>
    <title>Voice Agent Test</title>
    <script src="https://unpkg.com/livekit-client/dist/livekit-client.umd.js"></script>
</head>
<body>
    <h1>Voice Agent Test</h1>
    <button id="connect-btn">Connect to Agent</button>
    <button id="disconnect-btn" disabled>Disconnect</button>
    <div id="status">Disconnected</div>

    <script>
        const connectBtn = document.getElementById('connect-btn');
        const disconnectBtn = document.getElementById('disconnect-btn');
        const status = document.getElementById('status');
        
        let room = null;

        connectBtn.addEventListener('click', async () => {
            try {
                // Replace with your LiveKit details
                const wsURL = 'wss://your-project.livekit.cloud';
                const token = 'your-client-token'; // Generate from LiveKit dashboard
                
                room = new LiveKit.Room();
                
                // Connect to room
                await room.connect(wsURL, token);
                
                // Enable microphone
                await room.localParticipant.enableCameraAndMicrophone(false, true);
                
                status.textContent = 'Connected! Start talking...';
                connectBtn.disabled = true;
                disconnectBtn.disabled = false;
                
            } catch (error) {
                status.textContent = `Error: ${error.message}`;
            }
        });

        disconnectBtn.addEventListener('click', () => {
            if (room) {
                room.disconnect();
                status.textContent = 'Disconnected';
                connectBtn.disabled = false;
                disconnectBtn.disabled = true;
            }
        });
    </script>
</body>
</html>
```

### Generate Client Token
```python
# generate_token.py
import jwt
import time
from datetime import datetime, timedelta

def generate_client_token():
    api_key = "your-api-key"
    api_secret = "your-api-secret"
    
    now = datetime.now()
    exp = now + timedelta(hours=1)
    
    token = jwt.encode({
        'iss': api_key,
        'exp': int(exp.timestamp()),
        'nbf': int(now.timestamp()),
        'sub': 'test-user',
        'video': {
            'roomJoin': True,
            'room': 'test-room'
        }
    }, api_secret, algorithm='HS256')
    
    print(f"Client token: {token}")
    return token

if __name__ == "__main__":
    generate_client_token()
```

## 4. Enhanced Agent with Functions (4 minutes)

### Add Function Calling
```python
# enhanced_agent.py
import asyncio
import json
from datetime import datetime
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli
from livekit.agents.llm import openai, ChatContext, ChatMessage, FunctionContext
from livekit.agents.stt import openai as openai_stt
from livekit.agents.tts import elevenlabs
from livekit.agents.vad import silero
from livekit.agents.voice_assistant import VoiceAssistant

async def get_weather(location: str) -> str:
    """Mock weather function"""
    return json.dumps({
        "location": location,
        "temperature": "22°C",
        "condition": "Sunny",
        "humidity": "65%"
    })

async def get_time() -> str:
    """Get current time"""
    return datetime.now().strftime("The current time is %I:%M %p")

async def entrypoint(ctx: JobContext):
    """Enhanced agent with function calling"""
    
    # Create function context
    fnc_ctx = FunctionContext()
    fnc_ctx.ai_functions = [get_weather, get_time]
    
    # Enhanced system prompt
    initial_ctx = ChatContext(
        messages=[
            ChatMessage(
                role="system",
                content="""You are a helpful voice assistant with access to weather and time information.
                
Available functions:
- get_weather(location): Get weather for any location
- get_time(): Get current time

Keep responses conversational and natural for voice interaction.
Use functions when users ask about weather or time."""
            )
        ]
    )
    
    # Create voice assistant
    assistant = VoiceAssistant(
        vad=silero.VAD.load(),
        stt=openai_stt.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=elevenlabs.TTS(
            voice="Rachel",
            model="eleven_turbo_v2"  # Faster model
        ),
        chat_ctx=initial_ctx,
        fnc_ctx=fnc_ctx,
    )
    
    # Connect and start
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    assistant.start(ctx.room)
    
    await assistant.say("Hello! I can help with weather, time, and general questions. What would you like to know?")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
```

## 5. Quick Testing Commands

### Test Voice Agent
```bash
# Start enhanced agent
python enhanced_agent.py start

# Test with curl (if you have REST API endpoint)
curl -X POST http://localhost:8080/test \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the weather in London?"}'
```

### Common Test Phrases
Try these phrases with your voice agent:
- "Hello, how are you?"
- "What time is it?"
- "What's the weather like in New York?"
- "Tell me a joke"
- "What can you help me with?"

## 6. Monitor Performance

### Add Basic Metrics
```python
# Add to your agent.py
import time
from livekit.agents.voice_assistant import VoiceAssistant

class MetricsVoiceAssistant(VoiceAssistant):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = {
            "conversations": 0,
            "messages": 0,
            "avg_response_time": 0
        }
    
    async def say(self, message, **kwargs):
        start_time = time.time()
        result = await super().say(message, **kwargs)
        
        # Track metrics
        response_time = (time.time() - start_time) * 1000
        self.metrics["messages"] += 1
        self.metrics["avg_response_time"] = (
            (self.metrics["avg_response_time"] * (self.metrics["messages"] - 1) + response_time) 
            / self.metrics["messages"]
        )
        
        print(f"Response time: {response_time:.1f}ms | Avg: {self.metrics['avg_response_time']:.1f}ms")
        return result

# Use MetricsVoiceAssistant instead of VoiceAssistant in your entrypoint
```

## 7. Troubleshooting

### Common Issues

#### "Connection refused"
```bash
# Check environment variables
echo $LIVEKIT_URL
echo $LIVEKIT_API_KEY

# Verify LiveKit server is accessible
curl -I $LIVEKIT_URL
```

#### "Authentication failed"
```bash
# Regenerate API credentials in LiveKit dashboard
# Update .env file
# Restart agent
```

#### "Module not found"
```bash
# Ensure virtual environment is activated
which python
pip list | grep livekit

# Reinstall if needed
pip install --upgrade livekit-agents[all]
```

#### "Audio not working"
- Check microphone permissions in browser
- Verify WebRTC connectivity
- Test with different browsers
- Check firewall settings

### Debug Mode
```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Add to your agent
logger = logging.getLogger(__name__)
logger.info("Agent starting...")
```

## 8. Next Steps

### Immediate Next Steps (Choose One)
1. **Add More Functions**: Implement calendar, email, or database functions
2. **Improve Personality**: Customize system prompts and TTS voices
3. **Add Memory**: Implement conversation context persistence
4. **Deploy to Production**: Use Docker and Kubernetes deployment

### Recommended Learning Path
1. **Core Components**: Read [Core Components Guide](04-core-components-guide.md)
2. **Architecture**: Understand design in [Architecture Guide](03-voice-agent-architecture.md)
3. **Performance**: Optimize with [Performance Guide](07-performance-optimization.md)
4. **Real Applications**: Explore [Applications Guide](08-applications-guide.md)
5. **Advanced Features**: Study [Turn Detection Guide](06-turn-detection-guide.md)
6. **Complete Tutorial**: Follow [Implementation Tutorial](05-implementation-tutorial.md)
7. **Framework Details**: Reference [LiveKit Guide](09-livekit-reference.md)

### Quick Customizations

#### Change Voice Personality
```python
# Modify system prompt in ChatContext
ChatMessage(
    role="system", 
    content="You are a friendly, enthusiastic assistant who loves helping people. Use casual language and show excitement about solving problems!"
)

# Change TTS voice
tts=elevenlabs.TTS(voice="Antoni")  # Male voice
tts=elevenlabs.TTS(voice="Bella")   # Child-like voice
```

#### Adjust Response Speed
```python
# Faster responses
llm=openai.LLM(model="gpt-4o-mini", max_tokens=50)
tts=elevenlabs.TTS(model="eleven_turbo_v2")

# More thoughtful responses  
llm=openai.LLM(model="gpt-4o", max_tokens=200)
tts=elevenlabs.TTS(model="eleven_multilingual_v2")
```

#### Language Support
```python
# Spanish support
stt=openai_stt.STT(language="es")
ChatMessage(
    role="system",
    content="Eres un asistente de voz útil. Responde en español."
)
```

## 9. Production Checklist

Before deploying to production:

- [ ] **Security**: Rotate API keys, use secrets management
- [ ] **Monitoring**: Add comprehensive logging and metrics
- [ ] **Error Handling**: Implement retry logic and graceful failures
- [ ] **Scaling**: Test with multiple concurrent users
- [ ] **Backup**: Have fallback systems for critical functions
- [ ] **Compliance**: Review data retention and privacy policies
- [ ] **Performance**: Load test and optimize response times
- [ ] **Documentation**: Document API endpoints and configuration

## Key Takeaways

✅ **Under 15 Minutes**: From zero to working voice agent  
✅ **Minimal Dependencies**: Just Python and a few API keys  
✅ **Real Conversations**: Natural voice interactions with function calling  
✅ **Production Ready**: Foundation for scalable voice applications  
✅ **Extensible**: Easy to add custom functions and behaviors  

## Support & Resources

- **Documentation**: Complete guides in this repository
- **LiveKit Docs**: [docs.livekit.io](https://docs.livekit.io)
- **OpenAI API**: [platform.openai.com/docs](https://platform.openai.com/docs)
- **ElevenLabs API**: [docs.elevenlabs.io](https://docs.elevenlabs.io)
- **Community**: LiveKit Discord and GitHub discussions

**🎉 Congratulations!** You now have a working voice agent. Start experimenting and building amazing voice-powered applications!