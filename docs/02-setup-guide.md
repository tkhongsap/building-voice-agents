# Voice Agent Setup Guide

**📖 Learning Path Navigation:**
[← 01 Quick Start](01-quick-start-guide.md) | **02 Setup Guide** | [03 Architecture →](03-voice-agent-architecture.md)

**📚 All Guides:** [Complete Guide](00-complete-voice-agent-guide.md) | [01 Quick Start](01-quick-start-guide.md) | [02 Setup](02-setup-guide.md) | [03 Architecture](03-voice-agent-architecture.md) | [04 Components](04-core-components-guide.md) | [05 Implementation](05-implementation-tutorial.md) | [06 Turn Detection](06-turn-detection-guide.md) | [07 Performance](07-performance-optimization.md) | [08 Applications](08-applications-guide.md) | [09 LiveKit Reference](09-livekit-reference.md)

## Prerequisites

Before starting, you need to obtain API keys from the following services:

### 1. OpenAI API Key
- Go to [OpenAI Platform](https://platform.openai.com/)
- Sign up or log in
- Navigate to API keys section
- Create a new API key
- Copy the key for your `.env` file

### 2. ElevenLabs API Key
- Go to [ElevenLabs](https://elevenlabs.io/)
- Sign up for an account
- Navigate to your profile settings
- Find the API key section
- Copy the key for your `.env` file

### 3. LiveKit Configuration
For local development, you can use LiveKit's local server:
- Install LiveKit server locally or use their cloud service
- For local setup: Follow [LiveKit Getting Started](https://docs.livekit.io/realtime/self-hosting/local/)
- For cloud: Sign up at [LiveKit Cloud](https://livekit.io/cloud)

## Environment Setup

1. Copy the `.env.example` file to `.env`:
```bash
cp backend/.env.example backend/.env
```

2. Edit the `.env` file and add your API keys:
```
OPENAI_API_KEY=sk-...
ELEVEN_LABS_API_KEY=...
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...
```

## Installation

1. Create a Python virtual environment:
```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Start the LiveKit server (if running locally)
2. Start the voice agent backend:
```bash
cd backend
python main.py
```

3. Start the frontend (in a new terminal):
```bash
cd frontend
npm install
npm run dev
```

## Testing

Visit `http://localhost:3000` in your browser to test the voice agent.