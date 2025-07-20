# Voice Agents Platform API Documentation

Comprehensive API reference for the LiveKit Voice Agents Platform, providing detailed documentation for all classes, methods, and configuration options.

## Table of Contents

- [Quick Start](#quick-start)
- [Core SDK](#core-sdk)
- [Agent Builder](#agent-builder)
- [Configuration Management](#configuration-management)
- [Component APIs](#component-apis)
- [Conversation Management](#conversation-management)
- [Monitoring & Debugging](#monitoring--debugging)
- [Error Handling](#error-handling)
- [Examples](#examples)

---

## Quick Start

### Installation

```bash
pip install livekit-voice-agents
```

### Basic Usage

```python
import asyncio
from voice_agents import VoiceAgentSDK, initialize_sdk

async def main():
    # Initialize SDK
    sdk = await initialize_sdk({
        "project_name": "my_voice_agent",
        "environment": "production"
    })
    
    # Create agent
    agent = (sdk.create_builder()
        .with_name("My Assistant")
        .with_stt("openai", language="en")
        .with_llm("openai", model="gpt-4-turbo")
        .with_tts("openai", voice="nova")
        .build())
    
    # Start agent
    await agent.start()
    
    try:
        # Keep running
        await asyncio.sleep(3600)  # Run for 1 hour
    finally:
        await agent.stop()
        await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Core SDK

### VoiceAgentSDK

Main SDK class for managing voice agents globally.

#### Class: `VoiceAgentSDK`

**Singleton Pattern**: The SDK uses a singleton pattern to maintain global state.

##### Methods

###### `initialize(config=None, auto_discover=True)`

Initialize the SDK with configuration.

**Parameters:**
- `config` (optional): Configuration object, dict, or file path
  - Type: `SDKConfig | Dict[str, Any] | str | Path`
  - Default: `None` (uses defaults)
- `auto_discover` (bool): Whether to auto-discover components
  - Default: `True`

**Returns:** `None`

**Raises:**
- `ConfigurationError`: If configuration is invalid
- `VoiceAgentError`: If initialization fails

**Example:**
```python
# Initialize with defaults
await sdk.initialize()

# Initialize with configuration dict
await sdk.initialize({
    "project_name": "my_project",
    "environment": "production",
    "log_level": "INFO"
})

# Initialize with config file
await sdk.initialize("config.yaml")
```

###### `create_builder()`

Create a new agent builder instance.

**Returns:** `VoiceAgentBuilder`

**Example:**
```python
builder = sdk.create_builder()
agent = (builder
    .with_name("My Agent")
    .with_stt("openai")
    .with_llm("openai", model="gpt-4")
    .build())
```

###### `get_agent(name)`

Retrieve a registered agent by name.

**Parameters:**
- `name` (str): Agent name

**Returns:** `Optional[VoiceAgent]`

**Example:**
```python
agent = sdk.get_agent("my_agent")
if agent:
    print(f"Agent status: {agent.is_running}")
```

###### `list_agents()`

Get list of all registered agent names.

**Returns:** `List[str]`

**Example:**
```python
agents = sdk.list_agents()
print(f"Registered agents: {agents}")
```

###### `register_component(component_type, name, component_class)`

Register a custom component.

**Parameters:**
- `component_type` (str): Component type ("stt", "llm", "tts", "vad")
- `name` (str): Component name
- `component_class` (Type): Component class

**Example:**
```python
# Register custom STT provider
sdk.register_component("stt", "my_stt", MyCustomSTTProvider)
```

##### Properties

###### `is_initialized`

Check if SDK is initialized.

**Type:** `bool`

###### `config`

Get current SDK configuration.

**Type:** `SDKConfig`

**Raises:** `SDKNotInitializedError` if SDK not initialized

###### `registry`

Get component registry.

**Type:** `ComponentRegistry`

### initialize_sdk()

Convenience function for SDK initialization.

**Parameters:**
- `config` (optional): Configuration
- `auto_discover` (bool): Auto-discover components

**Returns:** `VoiceAgentSDK`

**Example:**
```python
sdk = await initialize_sdk({
    "project_name": "my_app",
    "environment": "development"
})
```

---

## Agent Builder

### VoiceAgentBuilder

Fluent API for building voice agents with method chaining.

#### Class: `VoiceAgentBuilder`

##### Core Configuration Methods

###### `with_name(name)`

Set agent name.

**Parameters:**
- `name` (str): Agent display name

**Returns:** `VoiceAgentBuilder`

**Example:**
```python
builder = builder.with_name("Customer Service Agent")
```

###### `with_description(description)`

Set agent description.

**Parameters:**
- `description` (str): Agent description

**Returns:** `VoiceAgentBuilder`

###### `with_system_prompt(prompt)`

Set system prompt for the LLM.

**Parameters:**
- `prompt` (str): System prompt text

**Returns:** `VoiceAgentBuilder`

**Example:**
```python
builder = builder.with_system_prompt("""
    You are a helpful customer service representative.
    Always be polite and professional.
""")
```

##### Provider Configuration Methods

###### `with_stt(provider, **config)`

Configure Speech-to-Text provider.

**Parameters:**
- `provider` (str): Provider name ("openai", "azure", "google")
- `**config`: Provider-specific configuration

**Returns:** `VoiceAgentBuilder`

**Examples:**
```python
# OpenAI Whisper
builder = builder.with_stt("openai", 
    language="en",
    model="whisper-1",
    temperature=0.0
)

# Azure Speech
builder = builder.with_stt("azure",
    language="en-US", 
    speech_key="your-key",
    speech_region="eastus"
)

# Google Speech
builder = builder.with_stt("google",
    language_code="en-US",
    model="latest_long",
    enable_automatic_punctuation=True
)
```

###### `with_llm(provider, **config)`

Configure Large Language Model provider.

**Parameters:**
- `provider` (str): Provider name ("openai", "anthropic", "google", "local")
- `**config`: Provider-specific configuration

**Returns:** `VoiceAgentBuilder`

**Examples:**
```python
# OpenAI GPT
builder = builder.with_llm("openai",
    model="gpt-4-turbo",
    temperature=0.7,
    max_tokens=500,
    top_p=1.0
)

# Anthropic Claude
builder = builder.with_llm("anthropic",
    model="claude-3-sonnet-20240229",
    max_tokens=300,
    temperature=0.5
)

# Google Gemini
builder = builder.with_llm("google",
    model="gemini-pro",
    temperature=0.7,
    top_p=0.95
)

# Local model
builder = builder.with_llm("local",
    model="llama2",
    base_url="http://localhost:11434"
)
```

###### `with_tts(provider, **config)`

Configure Text-to-Speech provider.

**Parameters:**
- `provider` (str): Provider name ("openai", "elevenlabs", "azure")
- `**config`: Provider-specific configuration

**Returns:** `VoiceAgentBuilder`

**Examples:**
```python
# OpenAI TTS
builder = builder.with_tts("openai",
    voice="nova",
    model="tts-1-hd",
    speed=1.0
)

# ElevenLabs
builder = builder.with_tts("elevenlabs",
    voice="Rachel",
    model="eleven_multilingual_v2",
    stability=0.5,
    similarity_boost=0.75
)

# Azure Speech
builder = builder.with_tts("azure",
    voice="en-US-JennyNeural",
    speech_key="your-key",
    speech_region="eastus"
)
```

###### `with_vad(provider, **config)`

Configure Voice Activity Detection provider.

**Parameters:**
- `provider` (str): Provider name ("silero", "webrtc")
- `**config`: Provider-specific configuration

**Returns:** `VoiceAgentBuilder`

**Examples:**
```python
# Silero VAD
builder = builder.with_vad("silero",
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=100
)

# WebRTC VAD
builder = builder.with_vad("webrtc",
    aggressiveness=2,
    sample_rate=16000
)
```

##### Capability Methods

###### `with_capability(capability)`

Enable agent capability.

**Parameters:**
- `capability` (str | AgentCapability): Capability to enable

**Returns:** `VoiceAgentBuilder`

**Available Capabilities:**
- `"turn_detection"`: Advanced turn detection
- `"interruption_handling"`: Handle interruptions gracefully
- `"context_management"`: Multi-turn context management
- `"conversation_state"`: State preservation and recovery

**Example:**
```python
builder = (builder
    .with_capability("turn_detection")
    .with_capability("interruption_handling")
    .with_capability("context_management")
)
```

##### Callback Methods

###### `with_callback(event, callback)`

Register event callback.

**Parameters:**
- `event` (str): Event name
- `callback` (Callable): Callback function

**Returns:** `VoiceAgentBuilder`

**Available Events:**
- `"on_start"`: Agent started
- `"on_stop"`: Agent stopped
- `"on_error"`: Error occurred
- `"on_user_speech"`: User spoke
- `"on_agent_speech"`: Agent spoke
- `"on_turn_end"`: Turn completed
- `"on_interruption"`: Interruption detected

**Example:**
```python
async def on_user_speech(text, confidence=None):
    print(f"User said: {text} (confidence: {confidence})")

async def on_error(error):
    print(f"Error: {error}")

builder = (builder
    .with_callback("on_user_speech", on_user_speech)
    .with_callback("on_error", on_error)
)
```

##### Advanced Methods

###### `with_pipeline_config(config)`

Configure audio pipeline settings.

**Parameters:**
- `config` (Dict): Pipeline configuration

**Example:**
```python
builder = builder.with_pipeline_config({
    "sample_rate": 16000,
    "channels": 1,
    "buffer_size": 1024
})
```

###### `with_custom_component(name, component)`

Add custom component.

**Parameters:**
- `name` (str): Component name
- `component` (Any): Component instance

**Returns:** `VoiceAgentBuilder`

###### `with_fallback_provider(provider_type, provider, **config)`

Set fallback provider.

**Parameters:**
- `provider_type` (str): Provider type
- `provider` (str): Provider name
- `**config`: Provider configuration

**Example:**
```python
# Use GPT-3.5 as fallback if GPT-4 fails
builder = (builder
    .with_llm("openai", model="gpt-4-turbo")
    .with_fallback_provider("llm", "openai", model="gpt-3.5-turbo")
)
```

##### Build Method

###### `build()`

Build the voice agent.

**Returns:** `VoiceAgent`

**Raises:**
- `ConfigurationError`: If configuration is invalid
- `VoiceAgentError`: If build fails

**Example:**
```python
agent = builder.build()
```

##### Utility Methods

###### `to_dict()`

Export configuration as dictionary.

**Returns:** `Dict[str, Any]`

###### `from_dict(config_dict)`

Load configuration from dictionary.

**Parameters:**
- `config_dict` (Dict): Configuration dictionary

**Returns:** `VoiceAgentBuilder`

### QuickBuilder

Simplified builder for common use cases.

#### Class: `QuickBuilder`

##### Quick Methods

###### `basic_agent(name)`

Create basic conversational agent.

**Example:**
```python
agent = QuickBuilder.basic_agent("My Assistant")
```

###### `customer_service_agent(company_name)`

Create customer service agent.

**Example:**
```python
agent = QuickBuilder.customer_service_agent("TechCorp")
```

###### `multilingual_agent(languages)`

Create multilingual agent.

**Example:**
```python
agent = QuickBuilder.multilingual_agent(["en", "es", "fr"])
```

---

## Configuration Management

### SDKConfig

SDK configuration class.

#### Class: `SDKConfig`

##### Properties

- `project_name` (str): Project name
- `environment` (Environment): Environment (DEVELOPMENT, STAGING, PRODUCTION)
- `log_level` (LogLevel): Logging level
- `enable_monitoring` (bool): Enable monitoring
- `enable_auto_discovery` (bool): Auto-discover components
- `api_timeout` (float): API timeout in seconds
- `max_retries` (int): Maximum retry attempts

**Example:**
```python
config = SDKConfig(
    project_name="my_voice_app",
    environment=Environment.PRODUCTION,
    log_level=LogLevel.INFO,
    enable_monitoring=True
)
```

### ConfigManager

Configuration management class.

#### Class: `ConfigManager`

##### Methods

###### `__init__(config=None)`

Initialize configuration manager.

**Parameters:**
- `config` (optional): Configuration source

###### `validate()`

Validate configuration.

**Returns:** `List[str]` (list of validation errors)

###### `to_dict()`

Export configuration as dictionary.

**Returns:** `Dict[str, Any]`

###### `save(file_path)`

Save configuration to file.

**Parameters:**
- `file_path` (str | Path): File path

###### `load(file_path)`

Load configuration from file.

**Parameters:**
- `file_path` (str | Path): File path

**Returns:** `ConfigManager`

---

## Component APIs

### Speech-to-Text (STT)

#### Base STT Interface

```python
from components.stt import BaseSTT, STTConfig, STTResult

class BaseSTT:
    async def initialize(self) -> None: ...
    async def transcribe(self, audio: bytes) -> STTResult: ...
    async def transcribe_stream(self, audio_stream) -> AsyncIterator[STTResult]: ...
    async def cleanup(self) -> None: ...
```

#### STT Configuration

```python
@dataclass
class STTConfig:
    language: str = "en"
    model: Optional[str] = None
    sample_rate: int = 16000
    channels: int = 1
    enable_vad: bool = True
    confidence_threshold: float = 0.8
```

#### STT Result

```python
@dataclass
class STTResult:
    text: str
    confidence: float
    is_final: bool
    language: Optional[str] = None
    timestamps: Optional[List[Tuple[float, float]]] = None
```

### Large Language Model (LLM)

#### Base LLM Interface

```python
from components.llm import BaseLLM, LLMConfig, LLMResponse

class BaseLLM:
    async def initialize(self) -> None: ...
    async def generate(self, prompt: str, **kwargs) -> LLMResponse: ...
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncIterator[str]: ...
    async def cleanup(self) -> None: ...
```

#### LLM Configuration

```python
@dataclass
class LLMConfig:
    model: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: Optional[List[str]] = None
```

#### LLM Response

```python
@dataclass
class LLMResponse:
    text: str
    usage: Dict[str, int]
    model: str
    finish_reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### Text-to-Speech (TTS)

#### Base TTS Interface

```python
from components.tts import BaseTTS, TTSConfig, TTSResult

class BaseTTS:
    async def initialize(self) -> None: ...
    async def synthesize(self, text: str, **kwargs) -> TTSResult: ...
    async def synthesize_stream(self, text: str, **kwargs) -> AsyncIterator[bytes]: ...
    async def cleanup(self) -> None: ...
```

#### TTS Configuration

```python
@dataclass
class TTSConfig:
    voice: str
    speed: float = 1.0
    pitch: float = 0.0
    volume: float = 1.0
    sample_rate: int = 22050
    audio_format: str = "mp3"
```

#### TTS Result

```python
@dataclass
class TTSResult:
    audio: bytes
    duration_ms: float
    sample_rate: int
    channels: int
    format: str
```

### Voice Activity Detection (VAD)

#### Base VAD Interface

```python
from components.vad import BaseVAD, VADConfig, VADResult

class BaseVAD:
    async def initialize(self) -> None: ...
    async def detect(self, audio: bytes) -> VADResult: ...
    async def cleanup(self) -> None: ...
```

#### VAD Configuration

```python
@dataclass
class VADConfig:
    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 100
    sample_rate: int = 16000
```

#### VAD Result

```python
@dataclass
class VADResult:
    is_speech: bool
    confidence: float
    energy: float
    timestamp: float
```

---

## Conversation Management

### ConversationTurn

Represents a single turn in conversation.

#### Class: `ConversationTurn`

##### Properties

- `turn_id` (str): Unique turn identifier
- `speaker` (str): Speaker ("user" or "agent")
- `start_time` (datetime): Turn start time
- `end_time` (Optional[datetime]): Turn end time
- `text` (Optional[str]): Turn text content
- `audio_duration_ms` (Optional[float]): Audio duration
- `confidence` (Optional[float]): Recognition confidence
- `metadata` (Dict[str, Any]): Additional metadata

##### Methods

###### `duration`

Get turn duration.

**Returns:** `Optional[timedelta]`

###### `add_metadata(key, value)`

Add metadata to turn.

**Parameters:**
- `key` (str): Metadata key
- `value` (Any): Metadata value

### ConversationManager

Manages conversation state and context.

#### Class: `ConversationManager`

##### Methods

###### `start_conversation(conversation_id=None)`

Start new conversation.

**Parameters:**
- `conversation_id` (optional): Conversation ID

**Returns:** `str` (conversation ID)

###### `end_conversation()`

End current conversation.

###### `add_turn(speaker, text, **metadata)`

Add turn to conversation.

**Parameters:**
- `speaker` (str): Speaker name
- `text` (str): Turn text
- `**metadata`: Additional metadata

**Returns:** `ConversationTurn`

###### `get_context(max_turns=10)`

Get conversation context.

**Parameters:**
- `max_turns` (int): Maximum turns to include

**Returns:** `List[ConversationTurn]`

###### `get_summary()`

Get conversation summary.

**Returns:** `str`

---

## Monitoring & Debugging

### ConversationInspector

Advanced conversation inspection and debugging.

#### Class: `ConversationInspector`

##### Methods

###### `start_monitoring()`

Start conversation monitoring.

###### `stop_monitoring()`

Stop conversation monitoring.

###### `get_insights()`

Get conversation insights.

**Returns:** `Dict[str, Any]`

###### `detect_anomalies()`

Detect conversation anomalies.

**Returns:** `List[Dict[str, Any]]`

###### `export_session(file_path, format="json")`

Export session data.

**Parameters:**
- `file_path` (str): Export file path
- `format` (str): Export format ("json", "csv")

### PerformanceMonitor

Monitor agent performance metrics.

#### Class: `PerformanceMonitor`

##### Methods

###### `start_monitoring(agent)`

Start monitoring agent.

**Parameters:**
- `agent` (VoiceAgent): Agent to monitor

###### `get_metrics()`

Get performance metrics.

**Returns:** `Dict[str, float]`

###### `get_latency_breakdown()`

Get latency breakdown by component.

**Returns:** `Dict[str, float]`

---

## Error Handling

### Exception Hierarchy

```python
VoiceAgentError
├── ConfigurationError
├── ValidationError
├── ProviderError
│   ├── STTError
│   ├── LLMError
│   ├── TTSError
│   └── VADError
├── PipelineError
├── ConversationError
└── SDKNotInitializedError
```

### Exception Classes

#### VoiceAgentError

Base exception class.

**Properties:**
- `message` (str): Error message
- `details` (Dict): Error details
- `timestamp` (datetime): Error timestamp

#### ConfigurationError

Configuration-related errors.

#### ValidationError

Input validation errors.

**Properties:**
- `field` (str): Invalid field
- `value` (Any): Invalid value
- `reason` (str): Validation failure reason

### Error Handling Best Practices

```python
from voice_agents.exceptions import VoiceAgentError, ConfigurationError

try:
    agent = builder.build()
    await agent.start()
except ConfigurationError as e:
    print(f"Configuration error: {e.message}")
    print(f"Details: {e.details}")
except VoiceAgentError as e:
    print(f"Agent error: {e.message}")
    # Handle gracefully
except Exception as e:
    print(f"Unexpected error: {e}")
    # Log and report
```

---

## Examples

### Complete Voice Agent Examples

#### 1. Basic Conversational Agent

```python
import asyncio
from voice_agents import initialize_sdk

async def create_basic_agent():
    # Initialize SDK
    sdk = await initialize_sdk({
        "project_name": "basic_agent_demo"
    })
    
    # Create agent
    agent = (sdk.create_builder()
        .with_name("Basic Assistant")
        .with_stt("openai", language="en")
        .with_llm("openai", 
                 model="gpt-3.5-turbo",
                 temperature=0.7,
                 max_tokens=200)
        .with_tts("openai", voice="nova")
        .with_vad("silero", threshold=0.5)
        .with_system_prompt("""
            You are a friendly and helpful assistant. 
            Respond concisely and helpfully to user questions.
        """)
        .build())
    
    return agent

# Usage
async def main():
    agent = await create_basic_agent()
    await agent.start()
    
    try:
        # Agent is now ready to handle conversations
        print("Agent started. Press Ctrl+C to stop.")
        await asyncio.sleep(3600)  # Run for 1 hour
    except KeyboardInterrupt:
        print("Stopping agent...")
    finally:
        await agent.stop()
        await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
```

#### 2. Customer Service Agent with Callbacks

```python
import asyncio
from voice_agents import initialize_sdk
from voice_agents.exceptions import VoiceAgentError

async def on_customer_speech(text, confidence=None):
    """Handle customer speech."""
    print(f"Customer: {text}")
    
    # Log high-confidence interactions
    if confidence and confidence > 0.9:
        print(f"High confidence interaction: {confidence:.2f}")

async def on_agent_response(text):
    """Handle agent responses."""
    print(f"Agent: {text}")

async def on_error(error):
    """Handle errors gracefully."""
    print(f"Error occurred: {error}")
    
    # Could implement error reporting here
    # await report_error_to_monitoring(error)

async def create_customer_service_agent():
    sdk = await initialize_sdk({
        "project_name": "customer_service",
        "environment": "production"
    })
    
    agent = (sdk.create_builder()
        .with_name("Customer Service Representative")
        .with_stt("openai", 
                 language="en",
                 model="whisper-1")
        .with_llm("openai",
                 model="gpt-4-turbo",
                 temperature=0.3,  # Lower for consistency
                 max_tokens=300)
        .with_tts("openai", 
                 voice="nova",
                 speed=0.9)  # Slightly slower for clarity
        .with_vad("silero", threshold=0.5)
        .with_capability("turn_detection")
        .with_capability("interruption_handling")
        .with_capability("context_management")
        .with_system_prompt("""
            You are a professional customer service representative.
            
            Guidelines:
            - Always be polite, empathetic, and helpful
            - Listen carefully to customer concerns
            - Provide clear and accurate information
            - Escalate complex issues when appropriate
            - Maintain a professional tone throughout
            
            Start each conversation by greeting the customer warmly
            and asking how you can assist them today.
        """)
        .with_callback("on_user_speech", on_customer_speech)
        .with_callback("on_agent_speech", on_agent_response)
        .with_callback("on_error", on_error)
        .build())
    
    return agent

async def main():
    try:
        agent = await create_customer_service_agent()
        await agent.start()
        
        print("Customer service agent is running...")
        print("Press Ctrl+C to stop.")
        
        # Keep running
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        print("\nShutting down customer service agent...")
    except VoiceAgentError as e:
        print(f"Agent error: {e.message}")
    finally:
        if 'agent' in locals():
            await agent.stop()
            await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
```

#### 3. Multi-Language Agent with Auto-Detection

```python
import asyncio
from voice_agents import initialize_sdk

# Language-specific responses
GREETINGS = {
    "en": "Hello! I can help you in multiple languages.",
    "es": "¡Hola! Puedo ayudarte en varios idiomas.",
    "fr": "Bonjour! Je peux vous aider dans plusieurs langues.",
    "de": "Hallo! Ich kann Ihnen in mehreren Sprachen helfen.",
    "it": "Ciao! Posso aiutarti in più lingue."
}

async def on_language_detected(text, language=None):
    """Handle language detection."""
    if language and language in GREETINGS:
        print(f"Detected language: {language}")
        print(f"Appropriate greeting: {GREETINGS[language]}")

async def create_multilingual_agent():
    sdk = await initialize_sdk({
        "project_name": "multilingual_assistant"
    })
    
    agent = (sdk.create_builder()
        .with_name("Multilingual Assistant")
        .with_stt("google", 
                 language="auto",  # Auto-detect language
                 enable_automatic_punctuation=True)
        .with_llm("anthropic",
                 model="claude-3-sonnet-20240229",
                 temperature=0.5,
                 max_tokens=400)
        .with_tts("elevenlabs",
                 voice="Rachel",
                 model="eleven_multilingual_v2")
        .with_vad("silero", threshold=0.6)
        .with_capability("context_management")
        .with_system_prompt("""
            You are a multilingual assistant fluent in English, Spanish, 
            French, German, and Italian.
            
            Instructions:
            - Automatically detect the user's language
            - Respond in the same language they use
            - If unsure about language, ask politely for clarification
            - Maintain conversation context across language switches
            - Be culturally appropriate for each language
            
            Start by greeting in multiple languages to show your capabilities.
        """)
        .with_callback("on_user_speech", on_language_detected)
        .build())
    
    return agent

async def main():
    agent = await create_multilingual_agent()
    
    try:
        await agent.start()
        print("Multilingual agent started. Try speaking in different languages!")
        await asyncio.sleep(3600)
    except KeyboardInterrupt:
        print("Stopping multilingual agent...")
    finally:
        await agent.stop()
        await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
```

#### 4. Function Calling Agent

```python
import asyncio
import json
from datetime import datetime
from voice_agents import initialize_sdk

# Mock functions for demonstration
async def get_weather(location: str) -> str:
    """Get weather for a location."""
    # In real implementation, call weather API
    return f"The weather in {location} is sunny, 72°F with light winds."

async def set_reminder(task: str, time: str) -> str:
    """Set a reminder."""
    # In real implementation, integrate with calendar/reminder system
    return f"Reminder set: '{task}' at {time}"

async def search_knowledge(query: str) -> str:
    """Search knowledge base."""
    # In real implementation, integrate with knowledge base
    return f"Here's what I found about '{query}': [Mock search result]"

# Function registry
AVAILABLE_FUNCTIONS = {
    "get_weather": get_weather,
    "set_reminder": set_reminder,
    "search_knowledge": search_knowledge
}

async def handle_function_call(function_name: str, arguments: dict) -> str:
    """Handle function calls from the LLM."""
    if function_name not in AVAILABLE_FUNCTIONS:
        return f"Function '{function_name}' is not available."
    
    try:
        # Call the function with provided arguments
        result = await AVAILABLE_FUNCTIONS[function_name](**arguments)
        return result
    except Exception as e:
        return f"Error calling {function_name}: {str(e)}"

async def create_function_calling_agent():
    sdk = await initialize_sdk({
        "project_name": "function_calling_demo"
    })
    
    # Define available tools for the LLM
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather information for a specific location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city or location to get weather for"
                        }
                    },
                    "required": ["location"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "set_reminder",
                "description": "Set a reminder for a specific task and time",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "The task to be reminded about"
                        },
                        "time": {
                            "type": "string",
                            "description": "When to remind (e.g., '3 PM', 'tomorrow morning')"
                        }
                    },
                    "required": ["task", "time"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_knowledge",
                "description": "Search the knowledge base for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]
    
    agent = (sdk.create_builder()
        .with_name("Function Calling Assistant")
        .with_stt("openai", language="en")
        .with_llm("openai",
                 model="gpt-4-turbo",
                 temperature=0.1,  # Lower for more consistent function calls
                 tools=tools)
        .with_tts("openai", voice="echo")
        .with_vad("silero")
        .with_system_prompt("""
            You are an AI assistant with access to several helpful functions:
            
            1. get_weather - Check weather for any location
            2. set_reminder - Set reminders for tasks
            3. search_knowledge - Search for information
            
            When users ask for these services, use the appropriate function
            to help them. Always confirm what you're doing and provide
            clear, helpful responses based on the function results.
        """)
        .with_function_handler(handle_function_call)
        .build())
    
    return agent

async def main():
    agent = await create_function_calling_agent()
    
    try:
        await agent.start()
        print("Function calling agent started!")
        print("Try asking about:")
        print("- Weather in any city")
        print("- Setting reminders")
        print("- Searching for information")
        
        await asyncio.sleep(3600)
    except KeyboardInterrupt:
        print("Stopping function calling agent...")
    finally:
        await agent.stop()
        await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
```

#### 5. Production-Ready Agent with Monitoring

```python
import asyncio
import logging
from pathlib import Path
from voice_agents import initialize_sdk
from voice_agents.monitoring import ConversationInspector, PerformanceMonitor
from voice_agents.exceptions import VoiceAgentError

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('voice_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def create_production_agent():
    """Create production-ready voice agent with monitoring."""
    
    # Load configuration from file
    config_file = Path("production_config.yaml")
    
    sdk = await initialize_sdk(config_file if config_file.exists() else {
        "project_name": "production_voice_agent",
        "environment": "production",
        "log_level": "INFO",
        "enable_monitoring": True
    })
    
    # Create conversation inspector
    inspector = ConversationInspector({
        "enable_audio_analysis": True,
        "enable_prompt_logging": False,  # Disable in production for privacy
        "buffer_size": 1000
    })
    
    # Create performance monitor
    monitor = PerformanceMonitor()
    
    # Error handling callback
    async def on_agent_error(error):
        logger.error(f"Agent error: {error}")
        # Could integrate with error reporting service
        
    # Performance monitoring callback
    async def on_turn_complete(turn_data):
        latency = turn_data.get('total_latency', 0)
        if latency > 5000:  # 5 seconds
            logger.warning(f"High latency detected: {latency}ms")
    
    agent = (sdk.create_builder()
        .with_name("Production Voice Assistant")
        .with_stt("openai",
                 language="en",
                 model="whisper-1")
        .with_llm("openai",
                 model="gpt-4-turbo",
                 temperature=0.7,
                 max_tokens=300)
        .with_tts("openai",
                 voice="nova",
                 speed=1.0)
        .with_vad("silero",
                 threshold=0.5)
        .with_capability("turn_detection")
        .with_capability("interruption_handling")
        .with_capability("context_management")
        .with_capability("conversation_state")
        .with_fallback_provider("llm", "openai", model="gpt-3.5-turbo")
        .with_fallback_provider("tts", "openai", voice="alloy")
        .with_system_prompt("""
            You are a professional AI assistant. Provide helpful,
            accurate, and concise responses to user queries.
        """)
        .with_callback("on_error", on_agent_error)
        .with_callback("on_turn_end", on_turn_complete)
        .build())
    
    # Setup monitoring
    await inspector.start_monitoring()
    await monitor.start_monitoring(agent)
    
    # Store monitoring instances for cleanup
    agent._inspector = inspector
    agent._monitor = monitor
    
    return agent

async def main():
    """Main application entry point."""
    agent = None
    
    try:
        logger.info("Starting production voice agent...")
        agent = await create_production_agent()
        await agent.start()
        
        logger.info("Voice agent is running in production mode")
        
        # Health check loop
        while True:
            await asyncio.sleep(60)  # Check every minute
            
            # Basic health checks
            if not agent.is_running:
                logger.error("Agent stopped unexpectedly")
                break
                
            # Log performance metrics
            if hasattr(agent, '_monitor'):
                metrics = agent._monitor.get_metrics()
                logger.info(f"Performance metrics: {metrics}")
            
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except VoiceAgentError as e:
        logger.error(f"Voice agent error: {e.message}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
    finally:
        if agent:
            logger.info("Shutting down voice agent...")
            
            # Stop monitoring
            if hasattr(agent, '_inspector'):
                await agent._inspector.stop_monitoring()
            if hasattr(agent, '_monitor'):
                await agent._monitor.stop_monitoring()
            
            # Stop agent
            await agent.stop()
            await agent.cleanup()
            
        logger.info("Shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## API Reference Summary

### Core Classes

| Class | Purpose | Key Methods |
|-------|---------|-------------|
| `VoiceAgentSDK` | Main SDK interface | `initialize()`, `create_builder()` |
| `VoiceAgentBuilder` | Fluent agent builder | `with_*()` methods, `build()` |
| `VoiceAgent` | Agent instance | `start()`, `stop()`, `cleanup()` |
| `ConfigManager` | Configuration management | `validate()`, `save()`, `load()` |

### Provider Interfaces

| Interface | Purpose | Key Methods |
|-----------|---------|-------------|
| `BaseSTT` | Speech-to-Text | `transcribe()`, `transcribe_stream()` |
| `BaseLLM` | Large Language Model | `generate()`, `generate_stream()` |
| `BaseTTS` | Text-to-Speech | `synthesize()`, `synthesize_stream()` |
| `BaseVAD` | Voice Activity Detection | `detect()` |

### Monitoring Classes

| Class | Purpose | Key Methods |
|-------|---------|-------------|
| `ConversationInspector` | Conversation debugging | `start_monitoring()`, `get_insights()` |
| `PerformanceMonitor` | Performance tracking | `start_monitoring()`, `get_metrics()` |

### Configuration Types

| Type | Purpose | Key Properties |
|------|---------|----------------|
| `SDKConfig` | SDK configuration | `project_name`, `environment`, `log_level` |
| `STTConfig` | STT configuration | `language`, `model`, `sample_rate` |
| `LLMConfig` | LLM configuration | `model`, `temperature`, `max_tokens` |
| `TTSConfig` | TTS configuration | `voice`, `speed`, `audio_format` |

---

For more examples and advanced usage patterns, see the [Examples Repository](https://github.com/livekit/voice-agents-examples) and [Community Cookbook](https://community.livekit.io/voice-agents).

**Need help?** Join our [Discord community](https://discord.gg/livekit) or check the [troubleshooting guide](./troubleshooting.md).