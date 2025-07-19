# Complete Implementation Tutorial

## Overview

This tutorial provides a step-by-step walkthrough for building a complete voice agent from scratch. You'll start with a basic agent and progressively add advanced features like metrics collection, voice customization, and production optimizations.

## Prerequisites

### Required Knowledge
- Basic Python programming
- Understanding of asynchronous programming (async/await)
- Familiarity with environment variables and API keys

### System Requirements
- **Python**: 3.10.11 (recommended version from course)
- **Node.js**: 18+ (for frontend, if needed)
- **Operating System**: Windows, macOS, or Linux
- **Memory**: 8GB RAM minimum (16GB recommended)
- **Network**: Stable internet connection for API calls

### Required API Keys
- **OpenAI API Key**: For STT and LLM services
- **ElevenLabs API Key**: For TTS services
- **LiveKit Credentials**: For real-time communication (optional for local development)

## Step 1: Environment Setup

### 1.1 Create Project Directory
```bash
mkdir my-voice-agent
cd my-voice-agent
```

### 1.2 Create Python Virtual Environment
```bash
# Create virtual environment
python3.10 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 1.3 Install Dependencies
```bash
# Install core LiveKit agents with plugins
pip install "livekit-agents[openai,silero,elevenlabs]==1.0.11"

# Install additional dependencies
pip install fastapi==0.115.8 uvicorn==0.34.0 python-dotenv==1.0.1 httpx==0.28.1

# For development and testing
pip install ipython==8.13.2
```

### 1.4 Create Environment File
Create `.env` file in your project root:
```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# ElevenLabs Configuration  
ELEVEN_LABS_API_KEY=your_elevenlabs_api_key_here

# LiveKit Configuration (for production)
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret

# Model Configuration
LLM_MODEL=gpt-4o-mini
STT_MODEL=whisper-1

# Server Configuration
HOST=0.0.0.0
PORT=8080
LOG_LEVEL=info
```

### 1.5 Project Structure
Create the following directory structure:
```
my-voice-agent/
├── .env
├── requirements.txt
├── main.py
├── agents/
│   ├── __init__.py
│   └── voice_agent.py
└── tests/
    ├── __init__.py
    └── test_agent.py
```

## Step 2: Basic Voice Agent Implementation

### 2.1 Create Basic Agent (`agents/voice_agent.py`)
```python
"""
Basic Voice Agent Implementation
"""

import logging
from typing import Optional
from livekit.agents import Agent
from livekit.plugins import openai, elevenlabs, silero
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

class VoiceAssistant(Agent):
    """
    A basic voice assistant with STT, LLM, TTS, and VAD components.
    """
    
    def __init__(
        self,
        *,
        instructions: str = None,
        model: str = None,
        voice_id: str = None,
        temperature: float = 0.7,
    ) -> None:
        """
        Initialize the voice assistant.
        
        Args:
            instructions: Custom instructions for the assistant
            model: LLM model to use
            voice_id: ElevenLabs voice ID
            temperature: LLM temperature
        """
        # Set default instructions
        if instructions is None:
            instructions = \"\"\"You are a helpful and friendly AI voice assistant. 
            You engage in natural, conversational dialogue with users.
            Keep your responses concise and conversational.
            If you don't understand something, ask for clarification.
            Be warm, professional, and helpful.\"\"\"
        
        # Get configuration from environment
        model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        voice_id = voice_id or os.getenv("ELEVEN_LABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
        
        logger.info(f"Initializing VoiceAssistant with model={model}, voice_id={voice_id}")
        
        # Initialize components
        stt_component = openai.STT(
            model="whisper-1",
            language=None,  # Auto-detect language
        )
        
        llm_component = openai.LLM(
            model=model,
            temperature=temperature,
        )
        
        tts_component = elevenlabs.TTS(
            voice_id=voice_id,
            model="eleven_monolingual_v1",
            encoding="mp3_44100_128",
        )
        
        vad_component = silero.VAD.load(
            min_speech_duration=0.1,
            min_silence_duration=0.5,
            speech_threshold=0.5,
        )
        
        # Initialize parent Agent class
        super().__init__(
            instructions=instructions,
            stt=stt_component,
            llm=llm_component,
            tts=tts_component,
            vad=vad_component,
            interrupt_speech=True,  # Enable interruptions
        )
        
        logger.info("VoiceAssistant initialized successfully")
    
    async def on_participant_connected(self, participant):
        """Handle when a participant joins."""
        logger.info(f"Participant connected: {participant.identity}")
    
    async def on_participant_disconnected(self, participant):
        """Handle when a participant leaves."""
        logger.info(f"Participant disconnected: {participant.identity}")
```

### 2.2 Create Main Entry Point (`main.py`)
```python
"""
Main entry point for the Voice Agent application.
"""

import asyncio
import logging
import os
from dotenv import load_dotenv

from livekit import agents
from livekit.agents import JobContext, JobRequest, WorkerOptions

from agents.voice_agent import VoiceAssistant

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def entrypoint(ctx: JobContext):
    """
    Main entry point for each agent session.
    
    Args:
        ctx: JobContext containing room information
    """
    logger.info(f"Agent entrypoint called for room: {ctx.room.name}")
    
    # Connect to the LiveKit room
    await ctx.connect()
    logger.info(f"Connected to room: {ctx.room.name}")
    
    # Create the voice assistant
    assistant = VoiceAssistant(
        instructions=\"\"\"You are a helpful AI assistant in a voice conversation.
        Be friendly, concise, and natural in your responses.
        Remember this is a voice conversation, so avoid long monologues.\"\"\"
    )
    
    # Start the agent session
    session = agents.AgentSession(
        agent=assistant,
    )
    
    # Start the session with the room
    await session.start(ctx.room)
    logger.info("Agent session started")
    
    # Keep the session running
    await session.wait()
    logger.info("Agent session ended")

def request_handler(req: JobRequest):
    """Handle incoming job requests."""
    logger.info(f"Received job request for room: {req.room.name}")
    return entrypoint

def main():
    """Main function to start the LiveKit agent worker."""
    logger.info("Starting Voice Agent application...")
    
    # Create worker options
    opts = WorkerOptions(
        entrypoint_fnc=entrypoint,
        request_fnc=request_handler,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8080)),
        ws_url=os.getenv("LIVEKIT_URL", "ws://localhost:7880"),
    )
    
    # Run the agent worker
    agents.run_app(opts)

if __name__ == "__main__":
    main()
```

### 2.3 Test Basic Agent
```bash
# Activate virtual environment
source venv/bin/activate

# Run the agent
python main.py
```

## Step 3: Adding Metrics Collection

### 3.1 Enhanced Agent with Metrics (`agents/metrics_agent.py`)
```python
"""
Voice Agent with comprehensive metrics collection.
"""

import logging
import asyncio
from typing import Dict, Any
from livekit.agents import Agent
from livekit.plugins import openai, elevenlabs, silero
from livekit.agents.metrics import LLMMetrics, STTMetrics, TTSMetrics, EOUMetrics
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class MetricsVoiceAgent(Agent):
    """Voice agent with comprehensive performance metrics."""
    
    def __init__(self, **kwargs):
        # Initialize components with default settings
        model = kwargs.get("model", os.getenv("LLM_MODEL", "gpt-4o-mini"))
        voice_id = kwargs.get("voice_id", os.getenv("ELEVEN_LABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"))
        
        llm = openai.LLM(model=model, temperature=0.7)
        stt = openai.STT(model="whisper-1")
        tts = elevenlabs.TTS(voice_id=voice_id)
        vad = silero.VAD.load()
        
        super().__init__(
            instructions=kwargs.get("instructions", "You are a helpful voice assistant."),
            stt=stt,
            llm=llm,
            tts=tts,
            vad=vad,
        )

        # Set up metrics collection
        self.setup_metrics_collection(llm, stt, tts)
        
        # Store metrics data
        self.metrics_data = {
            "llm": [],
            "stt": [],
            "tts": [],
            "eou": []
        }

    def setup_metrics_collection(self, llm, stt, tts):
        """Configure metrics collection for all components."""
        
        # LLM metrics
        def llm_metrics_wrapper(metrics: LLMMetrics):
            asyncio.create_task(self.on_llm_metrics_collected(metrics))
        llm.on("metrics_collected", llm_metrics_wrapper)

        # STT metrics
        def stt_metrics_wrapper(metrics: STTMetrics):
            asyncio.create_task(self.on_stt_metrics_collected(metrics))
        stt.on("metrics_collected", stt_metrics_wrapper)

        # End of Utterance metrics
        def eou_metrics_wrapper(metrics: EOUMetrics):
            asyncio.create_task(self.on_eou_metrics_collected(metrics))
        stt.on("eou_metrics_collected", eou_metrics_wrapper)

        # TTS metrics
        def tts_metrics_wrapper(metrics: TTSMetrics):
            asyncio.create_task(self.on_tts_metrics_collected(metrics))
        tts.on("metrics_collected", tts_metrics_wrapper)

    async def on_llm_metrics_collected(self, metrics: LLMMetrics) -> None:
        """Handle LLM performance metrics."""
        print("\n--- LLM Metrics ---")
        print(f"Prompt Tokens: {metrics.prompt_tokens}")
        print(f"Completion Tokens: {metrics.completion_tokens}")
        print(f"Tokens per Second: {metrics.tokens_per_second:.4f}")
        print(f"Time to First Token: {metrics.ttft:.4f}s")
        print("------------------\n")
        
        # Store metrics
        self.metrics_data["llm"].append({
            "prompt_tokens": metrics.prompt_tokens,
            "completion_tokens": metrics.completion_tokens,
            "tokens_per_second": metrics.tokens_per_second,
            "ttft": metrics.ttft,
            "timestamp": asyncio.get_event_loop().time()
        })

    async def on_stt_metrics_collected(self, metrics: STTMetrics) -> None:
        """Handle STT performance metrics."""
        print("\n--- STT Metrics ---")
        print(f"Duration: {metrics.duration:.4f}s")
        print(f"Audio Duration: {metrics.audio_duration:.4f}s")
        print(f"Streamed: {'Yes' if metrics.streamed else 'No'}")
        print("------------------\n")
        
        self.metrics_data["stt"].append({
            "duration": metrics.duration,
            "audio_duration": metrics.audio_duration,
            "streamed": metrics.streamed,
            "timestamp": asyncio.get_event_loop().time()
        })

    async def on_eou_metrics_collected(self, metrics: EOUMetrics) -> None:
        """Handle End of Utterance metrics."""
        print("\n--- End of Utterance Metrics ---")
        print(f"End of Utterance Delay: {metrics.end_of_utterance_delay:.4f}s")
        print(f"Transcription Delay: {metrics.transcription_delay:.4f}s")
        print("--------------------------------\n")
        
        self.metrics_data["eou"].append({
            "end_of_utterance_delay": metrics.end_of_utterance_delay,
            "transcription_delay": metrics.transcription_delay,
            "timestamp": asyncio.get_event_loop().time()
        })

    async def on_tts_metrics_collected(self, metrics: TTSMetrics) -> None:
        """Handle TTS performance metrics."""
        print("\n--- TTS Metrics ---")
        print(f"Time to First Byte: {metrics.ttfb:.4f}s")
        print(f"Duration: {metrics.duration:.4f}s")
        print(f"Audio Duration: {metrics.audio_duration:.4f}s")
        print(f"Streamed: {'Yes' if metrics.streamed else 'No'}")
        print("------------------\n")
        
        self.metrics_data["tts"].append({
            "ttfb": metrics.ttfb,
            "duration": metrics.duration,
            "audio_duration": metrics.audio_duration,
            "streamed": metrics.streamed,
            "timestamp": asyncio.get_event_loop().time()
        })
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get performance metrics summary."""
        summary = {}
        
        # LLM metrics summary
        if self.metrics_data["llm"]:
            llm_metrics = self.metrics_data["llm"]
            summary["llm"] = {
                "total_requests": len(llm_metrics),
                "avg_ttft": sum(m["ttft"] for m in llm_metrics) / len(llm_metrics),
                "avg_tokens_per_second": sum(m["tokens_per_second"] for m in llm_metrics) / len(llm_metrics),
                "total_tokens": sum(m["completion_tokens"] for m in llm_metrics),
            }
        
        # TTS metrics summary
        if self.metrics_data["tts"]:
            tts_metrics = self.metrics_data["tts"]
            summary["tts"] = {
                "total_requests": len(tts_metrics),
                "avg_ttfb": sum(m["ttfb"] for m in tts_metrics) / len(tts_metrics),
                "avg_duration": sum(m["duration"] for m in tts_metrics) / len(tts_metrics),
            }
        
        return summary
```

### 3.2 Update Main Entry Point for Metrics
```python
# In main.py, replace the assistant creation with:

assistant = MetricsVoiceAgent(
    instructions=\"\"\"You are a helpful AI assistant in a voice conversation.
    Be friendly, concise, and natural in your responses.
    Remember this is a voice conversation, so avoid long monologues.\"\"\",
    model="gpt-4o-mini"  # Fast model for testing
)
```

## Step 4: Voice Customization

### 4.1 Multiple Voice Personas (`agents/persona_agent.py`)
```python
"""
Voice agent with multiple personality configurations.
"""

from agents.voice_agent import VoiceAssistant

class PersonaVoiceAgent(VoiceAssistant):
    """Voice agent with customizable personas."""
    
    # Available voice personas
    PERSONAS = {
        "professional": {
            "voice_id": "CwhRBWXzGAHq8TQ4Fs17",  # Roger
            "instructions": \"\"\"You are a professional business assistant. 
            Speak clearly and formally. Provide structured, helpful responses.
            Be efficient and focus on delivering value.\"\"\",
            "temperature": 0.3
        },
        "friendly": {
            "voice_id": "EXAVITQu4vr4xnSDxMaL",  # Sarah
            "instructions": \"\"\"You are a warm, friendly assistant. 
            Be conversational and empathetic. Use a casual, approachable tone.
            Make users feel comfortable and welcomed.\"\"\",
            "temperature": 0.7
        },
        "authoritative": {
            "voice_id": "JBFqnCBsd6RMkjVDRZzb",  # George
            "instructions": \"\"\"You are an authoritative expert assistant.
            Provide confident, knowledgeable responses. Be direct and decisive.
            Users come to you for expertise and guidance.\"\"\",
            "temperature": 0.4
        },
        "casual": {
            "voice_id": "FGY2WhTYpPnrIDTdsKH5",  # Laura
            "instructions": \"\"\"You are a casual, laid-back assistant.
            Keep things relaxed and fun. Use informal language and be personable.
            Make conversations enjoyable and stress-free.\"\"\",
            "temperature": 0.8
        }
    }
    
    def __init__(self, persona="professional", **kwargs):
        """
        Initialize agent with specific persona.
        
        Args:
            persona: One of the available personas
            **kwargs: Additional configuration options
        """
        if persona not in self.PERSONAS:
            raise ValueError(f"Unknown persona: {persona}. Available: {list(self.PERSONAS.keys())}")
        
        persona_config = self.PERSONAS[persona]
        
        # Override with persona settings
        kwargs.update({
            "voice_id": persona_config["voice_id"],
            "instructions": persona_config["instructions"],
            "temperature": persona_config["temperature"]
        })
        
        super().__init__(**kwargs)
        self.persona = persona
        
        print(f"Initialized {persona} persona with voice {persona_config['voice_id']}")
    
    @classmethod
    def list_personas(cls):
        """List available personas with descriptions."""
        for name, config in cls.PERSONAS.items():
            print(f"{name}: {config['instructions'][:50]}...")
```

### 4.2 Voice Testing Script (`test_voices.py`)
```python
"""
Script to test different voice configurations.
"""

import asyncio
from agents.persona_agent import PersonaVoiceAgent

async def test_voice_personas():
    """Test different voice personas."""
    
    test_message = "Hello! How can I help you today?"
    
    for persona_name in PersonaVoiceAgent.PERSONAS.keys():
        print(f"\n--- Testing {persona_name.title()} Persona ---")
        
        agent = PersonaVoiceAgent(persona=persona_name)
        
        # In a real implementation, you would:
        # 1. Connect to LiveKit room
        # 2. Send test message through TTS
        # 3. Evaluate voice characteristics
        
        print(f"Voice ID: {agent.tts.voice_id}")
        print(f"Instructions: {agent.instructions[:100]}...")
        print(f"Temperature: {agent.llm.temperature}")

if __name__ == "__main__":
    asyncio.run(test_voice_personas())
```

## Step 5: Testing and Validation

### 5.1 Unit Tests (`tests/test_agent.py`)
```python
"""
Unit tests for voice agent components.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from agents.voice_agent import VoiceAssistant
from agents.metrics_agent import MetricsVoiceAgent

class TestVoiceAssistant:
    """Test suite for VoiceAssistant."""
    
    def test_agent_initialization(self):
        """Test basic agent initialization."""
        agent = VoiceAssistant()
        
        assert agent is not None
        assert agent.stt is not None
        assert agent.llm is not None
        assert agent.tts is not None
        assert agent.vad is not None
    
    def test_custom_instructions(self):
        """Test custom instructions setting."""
        custom_instructions = "You are a test assistant."
        agent = VoiceAssistant(instructions=custom_instructions)
        
        assert custom_instructions in agent.instructions
    
    def test_model_configuration(self):
        """Test model configuration."""
        agent = VoiceAssistant(model="gpt-4o-mini", temperature=0.5)
        
        assert agent.llm.model == "gpt-4o-mini"
        assert agent.llm.temperature == 0.5

class TestMetricsCollection:
    """Test suite for metrics collection."""
    
    def test_metrics_initialization(self):
        """Test metrics agent initialization."""
        agent = MetricsVoiceAgent()
        
        assert agent.metrics_data is not None
        assert "llm" in agent.metrics_data
        assert "stt" in agent.metrics_data
        assert "tts" in agent.metrics_data
        assert "eou" in agent.metrics_data
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self):
        """Test metrics collection handlers."""
        agent = MetricsVoiceAgent()
        
        # Mock metrics objects
        mock_llm_metrics = Mock()
        mock_llm_metrics.prompt_tokens = 10
        mock_llm_metrics.completion_tokens = 20
        mock_llm_metrics.tokens_per_second = 15.5
        mock_llm_metrics.ttft = 0.3
        
        # Test metrics collection
        await agent.on_llm_metrics_collected(mock_llm_metrics)
        
        assert len(agent.metrics_data["llm"]) == 1
        assert agent.metrics_data["llm"][0]["prompt_tokens"] == 10

# Run tests
if __name__ == "__main__":
    pytest.main([__file__])
```

### 5.2 Performance Testing Script (`test_performance.py`)
```python
"""
Performance testing script for voice agent.
"""

import asyncio
import time
import statistics
from agents.metrics_agent import MetricsVoiceAgent

class PerformanceTester:
    """Performance testing utilities."""
    
    def __init__(self):
        self.results = []
    
    async def test_response_latency(self, agent, test_inputs):
        """Test response latency for various inputs."""
        
        print("Starting latency tests...")
        
        for i, test_input in enumerate(test_inputs):
            print(f"Test {i+1}/{len(test_inputs)}: '{test_input[:30]}...'")
            
            start_time = time.time()
            
            # Simulate processing (in real implementation, would send to agent)
            await asyncio.sleep(0.1)  # Simulate processing time
            
            end_time = time.time()
            latency = (end_time - start_time) * 1000  # Convert to ms
            
            self.results.append({
                "input": test_input,
                "latency_ms": latency,
                "timestamp": start_time
            })
            
            print(f"  Latency: {latency:.2f}ms")
    
    def analyze_results(self):
        """Analyze performance test results."""
        if not self.results:
            print("No results to analyze")
            return
        
        latencies = [r["latency_ms"] for r in self.results]
        
        print("\n--- Performance Analysis ---")
        print(f"Total tests: {len(latencies)}")
        print(f"Average latency: {statistics.mean(latencies):.2f}ms")
        print(f"Median latency: {statistics.median(latencies):.2f}ms")
        print(f"Min latency: {min(latencies):.2f}ms")
        print(f"Max latency: {max(latencies):.2f}ms")
        
        if len(latencies) > 1:
            print(f"Std deviation: {statistics.stdev(latencies):.2f}ms")
        
        # Performance benchmarks
        avg_latency = statistics.mean(latencies)
        if avg_latency < 500:
            print("✅ Excellent performance (< 500ms)")
        elif avg_latency < 1000:
            print("⚠️  Good performance (500-1000ms)")
        else:
            print("❌ Needs optimization (> 1000ms)")

async def run_performance_tests():
    """Run comprehensive performance tests."""
    
    test_inputs = [
        "Hello, how are you?",
        "What's the weather like today?",
        "Can you explain artificial intelligence in simple terms?",
        "Tell me a joke.",
        "What are the benefits of renewable energy?",
        "How do I make a good first impression in a job interview?",
        "What's the difference between machine learning and deep learning?",
    ]
    
    tester = PerformanceTester()
    agent = MetricsVoiceAgent()
    
    await tester.test_response_latency(agent, test_inputs)
    tester.analyze_results()

if __name__ == "__main__":
    asyncio.run(run_performance_tests())
```

## Step 6: Production Optimizations

### 6.1 Production Configuration (`config/production.py`)
```python
"""
Production configuration settings.
"""

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class ProductionConfig:
    """Production configuration settings."""
    
    # Model settings for production
    llm_model: str = "gpt-4o-mini"  # Faster model for production
    stt_model: str = "whisper-1"
    voice_id: str = "21m00Tcm4TlvDq8ikWAM"  # Rachel voice
    
    # Performance settings
    temperature: float = 0.3  # Lower temperature for consistency
    timeout_seconds: int = 30  # Request timeout
    max_concurrent_sessions: int = 100
    
    # VAD settings optimized for production
    vad_speech_threshold: float = 0.5
    vad_min_speech_duration: float = 0.1
    vad_min_silence_duration: float = 0.5
    
    # Monitoring settings
    enable_metrics: bool = True
    log_level: str = "INFO"
    metrics_export_interval: int = 60  # seconds
    
    # Security settings
    rate_limit_requests_per_minute: int = 60
    max_message_length: int = 1000
    allowed_origins: list = None  # Set to specific domains in production
    
    @classmethod
    def from_environment(cls):
        """Create configuration from environment variables."""
        return cls(
            llm_model=os.getenv("LLM_MODEL", cls.llm_model),
            voice_id=os.getenv("ELEVEN_LABS_VOICE_ID", cls.voice_id),
            temperature=float(os.getenv("LLM_TEMPERATURE", cls.temperature)),
            log_level=os.getenv("LOG_LEVEL", cls.log_level),
            enable_metrics=os.getenv("ENABLE_METRICS", "true").lower() == "true",
        )

class DevelopmentConfig(ProductionConfig):
    """Development configuration with relaxed settings."""
    
    log_level: str = "DEBUG"
    rate_limit_requests_per_minute: int = 1000  # Higher limit for testing
    allowed_origins: list = ["*"]  # Allow all origins in development
```

### 6.2 Error Handling and Resilience (`utils/error_handling.py`)
```python
"""
Error handling and resilience utilities.
"""

import logging
import asyncio
from functools import wraps
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

class VoiceAgentError(Exception):
    """Base exception for voice agent errors."""
    pass

class ComponentError(VoiceAgentError):
    """Error in a specific component (STT, LLM, TTS, VAD)."""
    
    def __init__(self, component: str, message: str, original_error: Exception = None):
        self.component = component
        self.original_error = original_error
        super().__init__(f"{component} error: {message}")

def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator to retry functions on failure."""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Function {func.__name__} failed after {max_retries} attempts: {e}")
                        raise
                    
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {delay}s...")
                    await asyncio.sleep(delay)
            
        return wrapper
    return decorator

class CircuitBreaker:
    """Circuit breaker pattern for API calls."""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        
        if self.state == "open":
            if self.last_failure_time and (
                asyncio.get_event_loop().time() - self.last_failure_time > self.timeout
            ):
                self.state = "half-open"
            else:
                raise VoiceAgentError("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs)
            self.on_success()
            return result
        except Exception as e:
            self.on_failure()
            raise
    
    def on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        self.state = "closed"
    
    def on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = asyncio.get_event_loop().time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"

# Global circuit breakers for external services
openai_circuit_breaker = CircuitBreaker()
elevenlabs_circuit_breaker = CircuitBreaker()
```

### 6.3 Monitoring and Logging (`utils/monitoring.py`)
```python
"""
Monitoring and observability utilities.
"""

import logging
import time
import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
from enum import Enum

class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

@dataclass
class Metric:
    name: str
    value: float
    type: MetricType
    timestamp: float
    tags: Dict[str, str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class MetricsCollector:
    """Centralized metrics collection."""
    
    def __init__(self):
        self.metrics = []
        self.logger = logging.getLogger("metrics")
    
    def record_metric(self, name: str, value: float, metric_type: MetricType, tags: Dict[str, str] = None):
        """Record a metric."""
        metric = Metric(
            name=name,
            value=value,
            type=metric_type,
            timestamp=time.time(),
            tags=tags or {}
        )
        
        self.metrics.append(metric)
        self.logger.info(f"Metric recorded: {metric.name}={metric.value}")
    
    def get_metrics(self, since: Optional[float] = None) -> list:
        """Get metrics since timestamp."""
        if since is None:
            return self.metrics
        
        return [m for m in self.metrics if m.timestamp >= since]
    
    def export_metrics(self) -> str:
        """Export metrics in JSON format."""
        return json.dumps([m.to_dict() for m in self.metrics], indent=2)
    
    def clear_metrics(self):
        """Clear stored metrics."""
        self.metrics.clear()

# Global metrics collector
metrics = MetricsCollector()

class PerformanceMonitor:
    """Monitor performance of voice agent operations."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
    
    def time_operation(self, operation_name: str, tags: Dict[str, str] = None):
        """Context manager to time operations."""
        return TimingContext(operation_name, self.metrics, tags)

class TimingContext:
    """Context manager for timing operations."""
    
    def __init__(self, operation_name: str, metrics_collector: MetricsCollector, tags: Dict[str, str] = None):
        self.operation_name = operation_name
        self.metrics = metrics_collector
        self.tags = tags or {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.metrics.record_metric(
                name=f"{self.operation_name}_duration_seconds",
                value=duration,
                metric_type=MetricType.TIMER,
                tags=self.tags
            )

# Global performance monitor
performance_monitor = PerformanceMonitor(metrics)
```

## Step 7: Deployment and Testing

### 7.1 Create Deployment Script (`deploy.py`)
```python
"""
Deployment script for voice agent.
"""

import os
import subprocess
import sys
from pathlib import Path

def check_environment():
    """Check deployment environment."""
    required_env_vars = [
        "OPENAI_API_KEY",
        "ELEVEN_LABS_API_KEY",
    ]
    
    missing_vars = []
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        return False
    
    print("✅ Environment variables configured")
    return True

def install_dependencies():
    """Install production dependencies."""
    print("Installing dependencies...")
    
    result = subprocess.run([
        sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Failed to install dependencies: {result.stderr}")
        return False
    
    print("✅ Dependencies installed")
    return True

def run_tests():
    """Run test suite."""
    print("Running tests...")
    
    result = subprocess.run([
        sys.executable, "-m", "pytest", "tests/", "-v"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Tests failed: {result.stderr}")
        return False
    
    print("✅ All tests passed")
    return True

def start_application():
    """Start the voice agent application."""
    print("Starting voice agent application...")
    
    try:
        subprocess.run([sys.executable, "main.py"])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")

def main():
    """Main deployment function."""
    print("🚀 Deploying Voice Agent Application")
    print("=" * 40)
    
    if not check_environment():
        sys.exit(1)
    
    if not install_dependencies():
        sys.exit(1)
    
    if not run_tests():
        print("⚠️  Tests failed, but continuing deployment...")
    
    start_application()

if __name__ == "__main__":
    main()
```

### 7.2 Create Requirements File
```txt
# Core LiveKit and voice agent dependencies
livekit-agents[openai,silero,elevenlabs]==1.0.11
fastapi==0.115.8
uvicorn==0.34.0
python-dotenv==1.0.1
httpx==0.28.1

# Additional utilities
websockets==12.0
aiofiles==23.2.1
pydantic==2.5.3

# Development and testing
ipython==8.13.2
pytest==7.4.4
pytest-asyncio==0.23.3
```

## Step 8: Usage Examples

### 8.1 Basic Usage
```bash
# Activate environment
source venv/bin/activate

# Set environment variables
export OPENAI_API_KEY="your_key_here"
export ELEVEN_LABS_API_KEY="your_key_here"

# Run basic agent
python main.py
```

### 8.2 Metrics Agent Usage
```python
# In main.py, use metrics agent
from agents.metrics_agent import MetricsVoiceAgent

assistant = MetricsVoiceAgent(
    model="gpt-4o-mini",
    persona="professional"
)

# Get performance summary after conversations
summary = assistant.get_metrics_summary()
print(f"Average LLM TTFT: {summary['llm']['avg_ttft']:.3f}s")
```

### 8.3 Persona Agent Usage
```python
# In main.py, use persona agent
from agents.persona_agent import PersonaVoiceAgent

# Create different personality agents
professional_agent = PersonaVoiceAgent(persona="professional")
friendly_agent = PersonaVoiceAgent(persona="friendly")
casual_agent = PersonaVoiceAgent(persona="casual")
```

## Troubleshooting Common Issues

### Issue: Import Errors
```bash
# Solution: Ensure virtual environment is activated
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: API Key Errors
```bash
# Solution: Verify environment variables
echo $OPENAI_API_KEY
echo $ELEVEN_LABS_API_KEY

# Or check .env file exists and has correct format
```

### Issue: Connection Errors
```bash
# Solution: Check network connectivity and firewall settings
# Verify LiveKit server is running (if using local server)
```

### Issue: Poor Performance
```python
# Solution: Switch to faster model
assistant = VoiceAssistant(model="gpt-4o-mini")

# Monitor metrics to identify bottlenecks
summary = assistant.get_metrics_summary()
```

## Next Steps

1. **Turn Detection**: Advanced conversation management in [Turn Detection Guide](06-turn-detection-guide.md)
2. **Performance**: Optimize your agent with [Performance Guide](07-performance-optimization.md)
3. **Applications**: Domain-specific implementations in [Applications Guide](08-applications-guide.md)
4. **Framework**: Deep dive into [LiveKit Reference](09-livekit-reference.md)

## Key Takeaways

- **Start Simple**: Begin with basic agent and add features incrementally
- **Environment Setup**: Proper configuration is crucial for success
- **Metrics Matter**: Monitor performance from the beginning
- **Test Thoroughly**: Implement comprehensive testing before deployment
- **Production Ready**: Use proper error handling and monitoring for production
- **Persona Selection**: Voice and personality significantly impact user experience