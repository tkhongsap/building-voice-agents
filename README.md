# Voice Agent Application

A modern, real-time voice conversation application built with LiveKit, featuring an AI assistant with natural speech interactions and an animated avatar interface.

## Features

- 🎤 **Real-time Voice Chat**: Natural conversation with AI using speech-to-text and text-to-speech
- 🤖 **Animated Avatar**: Visual representation of the AI assistant with expressions and animations
- 📊 **Performance Monitoring**: Real-time metrics for latency optimization
- 🔊 **Voice Activity Detection**: Intelligent turn-taking and interruption handling
- 💬 **Conversation History**: Visual chat interface with message history
- ⚡ **Low Latency**: Optimized for responsive, natural conversations
- 🎛️ **Customizable**: Multiple voice options and configurable settings

## Architecture

The application is built with a modern, scalable architecture:

### Backend (Python)
- **LiveKit Agents**: Core voice agent framework
- **FastAPI**: REST API for session management
- **OpenAI**: GPT-4o for language understanding, Whisper for STT
- **ElevenLabs**: High-quality text-to-speech synthesis
- **Silero VAD**: Voice activity detection

### Frontend (React/Next.js)
- **Next.js 14**: Modern React framework with App Router
- **TypeScript**: Type-safe development
- **Tailwind CSS**: Utility-first styling
- **LiveKit SDK**: WebRTC integration for real-time communication
- **Canvas API**: Custom avatar animations

## Prerequisites

Before running the application, you need:

1. **Python 3.10+** (3.10.11 recommended)
2. **Node.js 18+** and npm
3. **API Keys**:
   - OpenAI API key
   - ElevenLabs API key
   - LiveKit credentials (or local server)

## Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd building-voice-agents
```

### 2. Backend Setup

```bash
cd backend

# Copy environment template
cp .env.example .env

# Edit .env with your API keys
# OPENAI_API_KEY=sk-...
# ELEVEN_LABS_API_KEY=...
# LIVEKIT_API_KEY=...
# LIVEKIT_API_SECRET=...

# Install dependencies (create virtual environment first)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Frontend Setup

```bash
cd frontend

# Copy environment template
cp .env.local.example .env.local

# Edit .env.local if needed
# NEXT_PUBLIC_API_URL=http://localhost:8000

# Install dependencies
npm install
```

### 4. Run the Application

Start both backend and frontend:

```bash
# Terminal 1: Start backend
cd backend
python main.py

# Terminal 2: Start frontend
cd frontend
npm run dev
```

Visit `http://localhost:3000` to use the voice agent!

## Detailed Setup Guide

### Getting API Keys

#### OpenAI API Key
1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Sign up or log in
3. Navigate to API keys section
4. Create a new API key
5. Add to your `.env` file

#### ElevenLabs API Key
1. Go to [ElevenLabs](https://elevenlabs.io/)
2. Sign up for an account
3. Navigate to your profile settings
4. Find the API key section
5. Add to your `.env` file

#### LiveKit Setup
For development, you can either:

**Option A: Use LiveKit Cloud**
1. Sign up at [LiveKit Cloud](https://livekit.io/cloud)
2. Create a new project
3. Get your API key and secret

**Option B: Run LiveKit Server Locally**
```bash
# Install LiveKit server
docker run --rm -p 7880:7880 -p 7881:7881 -p 7882:7882/udp livekit/livekit-server --dev
```

### Environment Variables

#### Backend `.env`
```env
# LiveKit Configuration
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key

# ElevenLabs Configuration
ELEVEN_LABS_API_KEY=your_elevenlabs_api_key
ELEVEN_LABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM  # Rachel voice

# Model Configuration
LLM_MODEL=gpt-4o-mini
STT_MODEL=whisper-1

# Server Configuration
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=info
```

#### Frontend `.env.local`
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Usage

### Basic Voice Chat
1. Open the application in your browser
2. Click "Connect" to start a session
3. Allow microphone permissions when prompted
4. Start speaking - the AI will listen and respond
5. You can interrupt the AI while it's speaking

### Avatar Interactions
- **Green indicator**: AI is speaking
- **Blue indicator**: Listening for your input
- **Yellow indicator**: Processing your request
- **Animations**: The avatar reacts to voice activity

### Performance Monitoring
- Click the chart icon in the header
- View real-time metrics for STT, LLM, and TTS
- Monitor latency and performance statistics

### Conversation Management
- View chat history in the right panel
- Clear conversation with the trash icon
- Messages show timestamps and participant type

## Customization

### Voice Selection
Change the voice by updating the `ELEVEN_LABS_VOICE_ID` in your `.env`:

```env
# Available voices:
ELEVEN_LABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM  # Rachel (default)
# ELEVEN_LABS_VOICE_ID=CwhRBWXzGAHq8TQ4Fs17  # Roger
# ELEVEN_LABS_VOICE_ID=EXAVITQu4vr4xnSDxMaL  # Sarah
```

### Model Selection
Choose between different LLM models:

```env
LLM_MODEL=gpt-4o-mini      # Faster, lower cost
# LLM_MODEL=gpt-4o         # Higher quality, higher cost
```

### Assistant Instructions
Modify the assistant's behavior in `backend/agents/voice_agent.py`:

```python
instructions = """You are a helpful AI assistant specializing in [your domain].
Keep responses conversational and under 50 words for voice chat."""
```

## Development

### Project Structure
```
building-voice-agents/
├── backend/                 # Python backend
│   ├── agents/             # Voice agent implementations
│   ├── api/                # FastAPI server
│   ├── config/             # Configuration files
│   └── utils/              # Utility functions
├── frontend/               # React frontend
│   ├── src/
│   │   ├── app/           # Next.js app router
│   │   ├── components/    # React components
│   │   ├── hooks/         # Custom React hooks
│   │   ├── lib/           # Utilities and API client
│   │   └── types/         # TypeScript type definitions
├── docs/                   # Documentation
└── deeplearning-ai/        # Course materials from DeepLearning.AI
```

### Adding Features

#### New Voice Commands
1. Extend the `CustomVoiceAssistant` class in `backend/agents/voice_agent.py`
2. Add command handling in `before_llm_response` method

#### Custom Avatar Animations
1. Modify the `Avatar` component in `frontend/src/components/Avatar.tsx`
2. Add new animation states and expressions

#### Additional Metrics
1. Update the `AdvancedVoiceAssistant` class for new metrics
2. Extend the `MetricsDashboard` component to display them

### Testing

#### Backend Testing
```bash
cd backend
pytest tests/
```

#### Frontend Testing
```bash
cd frontend
npm test
```

#### Manual Testing
1. Test voice conversation flow
2. Verify interruption handling
3. Check metrics accuracy
4. Test different voice commands

## Troubleshooting

### Common Issues

#### "Permission denied" for microphone
- Ensure browser has microphone permissions
- Check if other applications are using the microphone
- Try refreshing the page and allowing permissions again

#### High latency in conversations
- Check your internet connection
- Monitor the performance dashboard
- Consider switching to `gpt-4o-mini` for faster responses
- Verify LiveKit server connectivity

#### Avatar not animating
- Check browser console for errors
- Ensure WebGL is supported
- Try refreshing the page

#### Connection issues
- Verify all API keys are correctly set
- Check that the backend server is running
- Ensure LiveKit server is accessible
- Check firewall settings

### Debug Mode
Enable debug logging:

```env
LOG_LEVEL=debug
```

View logs in the browser console and backend terminal.

## Performance Optimization

### Latency Optimization
1. **Model Selection**: Use `gpt-4o-mini` for faster responses
2. **Streaming**: Ensure streaming is enabled for all components
3. **Network**: Use a reliable internet connection
4. **Server Location**: Choose LiveKit server close to your location

### Resource Management
1. **Memory**: Monitor memory usage in production
2. **CPU**: Optimize audio processing settings
3. **Bandwidth**: Adjust audio quality settings if needed

## Production Deployment

### Backend Deployment
1. Use a production WSGI server (e.g., Gunicorn)
2. Set up proper environment variables
3. Configure logging and monitoring
4. Use PostgreSQL or Redis for session storage

### Frontend Deployment
1. Build the production bundle: `npm run build`
2. Deploy to Vercel, Netlify, or similar
3. Configure environment variables
4. Set up proper CORS headers

### Security Considerations
1. Use HTTPS in production
2. Implement rate limiting
3. Validate all user inputs
4. Store API keys securely
5. Implement proper authentication

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [LiveKit](https://livekit.io/) for real-time communication
- Uses [OpenAI](https://openai.com/) for language understanding
- Powered by [ElevenLabs](https://elevenlabs.io/) for voice synthesis
- Based on the [DeepLearning.AI Building Voice Agents](https://www.deeplearning.ai/) course

## 📚 Documentation & Learning Resources

### Complete Documentation Suite

This repository includes comprehensive documentation for learning and implementing voice agents:

#### 🚀 **Quick Learning Path**
1. **[Complete Voice Agent Guide](docs/00-complete-voice-agent-guide.md)** - Comprehensive combined reference
2. **[Quick Start Guide](docs/01-quick-start-guide.md)** - Get running in 15 minutes
3. **[Setup Guide](docs/02-setup-guide.md)** - Environment configuration

#### 🏗️ **Architecture & Design**
4. **[Voice Agent Architecture](docs/03-voice-agent-architecture.md)** - System design patterns
5. **[Core Components Guide](docs/04-core-components-guide.md)** - STT, LLM, TTS, VAD deep dive

#### 🔧 **Implementation & Advanced Topics**
6. **[Implementation Tutorial](docs/05-implementation-tutorial.md)** - Complete step-by-step walkthrough
7. **[Turn Detection Guide](docs/06-turn-detection-guide.md)** - Conversation management
8. **[Performance Optimization](docs/07-performance-optimization.md)** - Production tuning

#### 🌐 **Production & Applications**
9. **[Applications Guide](docs/08-applications-guide.md)** - Industry use cases
10. **[LiveKit Reference](docs/09-livekit-reference.md)** - Complete API documentation

### 📖 **Learning Recommendations**

**For Beginners:**
- Start with [Quick Start Guide](docs/01-quick-start-guide.md) to get immediate results
- Follow [Setup Guide](docs/02-setup-guide.md) for proper configuration
- Study [Architecture Guide](docs/03-voice-agent-architecture.md) for foundational understanding

**For Developers:**
- Deep dive into [Core Components](docs/04-core-components-guide.md) for technical details
- Follow [Implementation Tutorial](docs/05-implementation-tutorial.md) for complete walkthrough
- Master [Turn Detection](docs/06-turn-detection-guide.md) for natural conversations

**For Production:**
- Optimize with [Performance Guide](docs/07-performance-optimization.md)
- Explore [Applications Guide](docs/08-applications-guide.md) for industry patterns
- Reference [LiveKit Documentation](docs/09-livekit-reference.md) for complete API details

### 🎓 **Course Materials**

This repository also contains the original course materials from DeepLearning.AI in the `deeplearning-ai/` folder:

- **Lesson 4**: Voice Agent Components (`deeplearning-ai/L4/`)
- **Lesson 5**: Optimizing Latency (`deeplearning-ai/L5/`)
- **Course Transcript**: Complete spoken content (`deeplearning-ai/recorded-courses.txt`)
- **Appendix**: Tips and Help (`deeplearning-ai/Appendix–Tips_Help_and_Download/`)

These materials provide the educational foundation for understanding voice agent development and complement the practical implementation guides.