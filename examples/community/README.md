# Community Examples and Sample Applications

This directory contains practical examples and sample applications built with the LiveKit Voice Agents Platform. These examples demonstrate real-world use cases and serve as starting points for building your own voice applications.

## 📚 Available Examples

### 🤖 Basic Voice Bot (`basic_voice_bot.py`)
A simple voice assistant that demonstrates core platform functionality.

**Features:**
- Basic speech-to-text and text-to-speech
- Simple conversation flow with function calling
- Weather queries, time information, math calculations
- Joke telling and general Q&A

**Usage:**
```bash
export OPENAI_API_KEY="your-openai-key"
python basic_voice_bot.py
```

**Perfect for:**
- Learning the SDK basics
- Understanding voice agent architecture
- Quick prototyping and experimentation

---

### 📅 Meeting Assistant (`meeting_assistant.py`)
An advanced meeting productivity tool with transcription and action item tracking.

**Features:**
- Real-time meeting transcription with speaker identification
- Automatic action item detection and tracking
- Meeting summarization and analytics
- Session recording and playback
- Participant management

**Usage:**
```bash
export OPENAI_API_KEY="your-openai-key"
export AZURE_SPEECH_KEY="your-azure-key"
export ELEVENLABS_API_KEY="your-elevenlabs-key"

python meeting_assistant.py --mode live    # Live meeting
python meeting_assistant.py --mode demo    # Demo mode
python meeting_assistant.py --mode list    # List saved meetings
```

**Perfect for:**
- Business meetings and conferences
- Interview transcription
- Educational lectures
- Remote team collaboration

---

### 🏢 Customer Service Bot (`customer_service_bot.py`)
Enterprise-grade customer support agent with ticket management and escalation.

**Features:**
- Intelligent query routing and classification
- Ticket creation and status tracking
- Knowledge base search and integration
- Customer authentication and account management
- Escalation handling and sentiment analysis
- Multi-tier customer support (standard, premium, enterprise)

**Usage:**
```bash
export ANTHROPIC_API_KEY="your-anthropic-key"
export AZURE_SPEECH_KEY="your-azure-key"
export ELEVENLABS_API_KEY="your-elevenlabs-key"

python customer_service_bot.py --mode live      # Live support
python customer_service_bot.py --mode demo      # Demo interactions
python customer_service_bot.py --mode training  # Training scenarios
```

**Perfect for:**
- Customer support centers
- Help desk automation
- Technical support
- E-commerce customer service

---

### 🌍 Language Tutor (`language_tutor.py`)
Interactive language learning assistant with pronunciation feedback and cultural context.

**Features:**
- Multi-language support (Spanish, French, German, Italian, Chinese)
- Adaptive difficulty levels (beginner to advanced)
- Pronunciation analysis and feedback
- Interactive conversation practice scenarios
- Grammar explanations and exercises
- Progress tracking and learning analytics
- Cultural context integration

**Usage:**
```bash
export OPENAI_API_KEY="your-openai-key"
export GOOGLE_CLOUD_KEY="your-google-key"
export ELEVENLABS_API_KEY="your-elevenlabs-key"

python language_tutor.py --language spanish --level beginner --mode live
python language_tutor.py --language french --level intermediate --mode demo
```

**Supported Languages:**
- Spanish (España/Latin America)
- French (France)
- German (Deutschland)
- Italian (Italia)
- Mandarin Chinese (中文)

**Perfect for:**
- Language learning applications
- Educational platforms
- Cultural exchange programs
- Travel preparation

---

## 🚀 Getting Started

### Prerequisites

1. **Python 3.8+** with asyncio support
2. **API Keys** for the services you want to use:
   - OpenAI API key (GPT models)
   - Anthropic API key (Claude models)
   - Azure Speech Services key
   - Google Cloud Speech API key
   - ElevenLabs API key
3. **LiveKit Voice Agents SDK** installed

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/building-voice-agents.git
cd building-voice-agents

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export AZURE_SPEECH_KEY="your-azure-speech-key"
export AZURE_SPEECH_REGION="eastus"
export GOOGLE_CLOUD_KEY="your-google-cloud-key"
export ELEVENLABS_API_KEY="your-elevenlabs-api-key"
```

### Quick Start

1. **Try the Basic Voice Bot:**
   ```bash
   cd examples/community
   python basic_voice_bot.py
   ```

2. **Run a demo:**
   Choose option 2 for text-based demo if you don't have a microphone set up.

3. **Explore other examples:**
   Each example has demo modes that don't require live audio.

---

## 🛠️ Customization Guide

### Adapting Examples for Your Use Case

#### 1. **Modify System Prompts**
Each example includes customizable system prompts in the LLM configuration:

```python
llm_config = {
    "system_prompt": """Your custom instructions here..."""
}
```

#### 2. **Add Custom Functions**
Register new functions for specific capabilities:

```python
@self.agent.function(
    name="your_custom_function",
    description="What this function does",
    parameters={
        "param1": {"type": "string", "description": "Parameter description"}
    }
)
async def your_custom_function(param1: str) -> str:
    # Your implementation
    return "Response to user"
```

#### 3. **Change Voice Providers**
Swap out components based on your needs:

```python
# Use different STT provider
stt_provider="azure"  # or "google", "openai"

# Use different LLM
llm_provider="anthropic"  # or "openai", "local"

# Use different TTS
tts_provider="elevenlabs"  # or "openai", "azure"
```

#### 4. **Modify Data Storage**
Examples use JSON files for simplicity. For production:

```python
# Replace with your database
def _save_data(self):
    # Save to PostgreSQL, MongoDB, etc.
    pass
```

---

## 🎯 Use Case Scenarios

### Business Applications
- **Internal Tools:** Meeting assistants, documentation helpers
- **Customer Facing:** Support bots, sales assistants, appointment scheduling
- **Training:** Employee onboarding, compliance training

### Educational Applications
- **Language Learning:** Conversation practice, pronunciation coaching
- **Tutoring:** Subject-specific AI tutors, homework assistance
- **Accessibility:** Voice-controlled learning tools

### Healthcare Applications
- **Patient Support:** Appointment scheduling, medication reminders
- **Mental Health:** Therapy chatbots, wellness check-ins
- **Medical Training:** Symptom simulators, case studies

### Entertainment Applications
- **Gaming:** NPCs with voice interaction, game masters
- **Storytelling:** Interactive narratives, character voices
- **Social:** Virtual companions, conversation practice

---

## 🧪 Testing and Development

### Running Tests
```bash
# Run all example tests
python -m pytest examples/community/tests/

# Test specific example
python -m pytest examples/community/tests/test_basic_voice_bot.py
```

### Debug Mode
All examples support debug mode with detailed logging:

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Local Development
Use text mode for development without audio setup:

```bash
python basic_voice_bot.py
# Choose option 2: Text demo mode
```

---

## 📊 Performance Considerations

### Latency Optimization
- **STT**: Use streaming when available (Azure, Google)
- **LLM**: Choose faster models for real-time interaction
- **TTS**: Enable streaming synthesis for immediate response
- **VAD**: Tune sensitivity for your environment

### Cost Optimization
- **OpenAI**: Use `gpt-4o-mini` for cost-effective responses
- **Azure**: Regional deployment for lower latency
- **ElevenLabs**: Voice cloning vs. pre-built voices
- **Caching**: Implement response caching for common queries

### Scaling Considerations
- **Concurrent Users**: Connection pooling and load balancing
- **Data Storage**: Move from JSON to proper databases
- **Monitoring**: Implement proper logging and metrics
- **Error Handling**: Graceful degradation and fallback providers

---

## 🤝 Contributing

### Adding New Examples

1. **Create your example file:**
   ```bash
   cp basic_voice_bot.py your_example.py
   ```

2. **Follow the structure:**
   - Clear docstring with features and usage
   - Configuration management
   - Function registration
   - Demo mode support
   - Error handling

3. **Add documentation:**
   - Update this README
   - Include usage examples
   - Document required API keys

4. **Test thoroughly:**
   - Demo mode works without audio
   - All API integrations function
   - Error cases handled gracefully

### Code Style Guidelines

- **Async/await**: Use proper async patterns
- **Type hints**: Include type annotations
- **Docstrings**: Document all functions
- **Error handling**: Graceful degradation
- **Configuration**: Externalize all settings

---

## 🆘 Troubleshooting

### Common Issues

#### API Key Errors
```bash
❌ Error: Missing required environment variables
```
**Solution:** Ensure all required API keys are set:
```bash
export OPENAI_API_KEY="your-key-here"
```

#### Audio Device Issues
```bash
❌ Error: No audio input device found
```
**Solution:** Use text demo mode or check microphone permissions.

#### Import Errors
```bash
❌ ModuleNotFoundError: No module named 'src'
```
**Solution:** Run from the project root directory:
```bash
cd building-voice-agents
python examples/community/basic_voice_bot.py
```

#### Rate Limiting
```bash
❌ Error: Rate limit exceeded
```
**Solution:** Implement retry logic or use different API tiers.

### Getting Help

1. **Check the logs:** Enable debug mode for detailed output
2. **Review documentation:** Each example has extensive docstrings
3. **Try demo mode:** Test functionality without live audio
4. **Check API status:** Verify your API keys and service status

---

## 📝 License

These examples are provided under the MIT License. See the main project LICENSE file for details.

---

## 🙏 Acknowledgments

- **LiveKit Team** for the excellent WebRTC infrastructure
- **OpenAI** for powerful language models
- **Anthropic** for Claude's reasoning capabilities
- **Azure Speech Services** for robust STT/TTS
- **ElevenLabs** for high-quality voice synthesis
- **Community contributors** for feedback and improvements

---

**Happy building! 🚀**

For more examples and advanced patterns, check out the main documentation and API reference.