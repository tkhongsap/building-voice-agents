# Building Voice Agents

This repository contains course materials from the **DeepLearning.AI** course "Building Voice Agents" - a comprehensive guide to creating real-time voice conversation agents using modern AI technologies.

## Overview

Learn how to build voice agents that can have natural, real-time conversations using:
- **LiveKit Agents Framework** for real-time communication
- **OpenAI** models (GPT-4o, Whisper) for language understanding and speech-to-text
- **ElevenLabs** for natural text-to-speech synthesis
- **Silero VAD** for voice activity detection

## Course Structure

### Lesson 4: Voice Agent Components
Introduction to the core components of a voice agent:
- Speech-to-Text (STT) using OpenAI Whisper
- Large Language Model (LLM) using GPT-4o
- Text-to-Speech (TTS) using ElevenLabs
- Voice Activity Detection (VAD) using Silero

### Lesson 5: Optimizing Latency
Advanced topics on performance optimization:
- Measuring and monitoring latency metrics
- Understanding TTFT (Time to First Token) and TTFB (Time to First Byte)
- Optimizing end-to-end conversation latency
- Performance debugging techniques

### Appendix
Additional resources and tips for using the course platform and troubleshooting common issues.

## Prerequisites

- Python 3.10.11
- Basic knowledge of Python and asynchronous programming
- API keys for OpenAI and ElevenLabs services

## Installation

1. Clone this repository:
```bash
git clone https://github.com/tkhongsap/building-voice-agents.git
cd building-voice-agents
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables:
```bash
# Create a .env file with your API keys
OPENAI_API_KEY=your_openai_api_key
ELEVEN_LABS_API_KEY=your_elevenlabs_api_key
```

## Key Features Demonstrated

- **Real-time Conversations**: Build agents capable of natural, flowing dialogue
- **Multi-language Support**: Automatic language detection and response
- **Voice Customization**: Multiple voice personas (Roger, Sarah, Laura, George)
- **Performance Monitoring**: Built-in metrics collection and analysis
- **Jupyter Integration**: Interactive development and testing environment

## Tech Stack

- **LiveKit Agents** (v1.0.11): Core framework for voice agents
- **OpenAI Services**: Whisper for STT, GPT-4o for LLM
- **ElevenLabs**: High-quality text-to-speech
- **Silero VAD**: Voice activity detection
- **FastAPI/Uvicorn**: Web server functionality
- **Python AsyncIO**: Asynchronous programming

## Course Content

All course materials are provided as Jupyter notebooks with inline explanations and working code examples:
- `deeplearning-ai/L4/Lesson4.ipynb`: Voice Agent Components
- `deeplearning-ai/L5/Lesson5.ipynb`: Optimizing Latency
- `deeplearning-ai/Appendix–Tips_Help_and_Download/Appendix.ipynb`: Help and tips

## Learning Outcomes

By completing this course, you will:
1. Understand the architecture of modern voice agents
2. Build functional voice agents from scratch
3. Implement real-time speech processing pipelines
4. Optimize voice agent performance for production use
5. Monitor and debug voice agent systems

## License

This repository contains educational materials from DeepLearning.AI. Please refer to the original course for licensing information.

## Acknowledgments

Course materials provided by [DeepLearning.AI](https://www.deeplearning.ai/) - Building Voice Agents course.