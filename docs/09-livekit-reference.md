# LiveKit Agents Framework Reference

## Overview

This comprehensive reference guide covers the LiveKit Agents framework v1.0.11, providing detailed API documentation, implementation patterns, and best practices based on the DeepLearning.AI course materials and production deployment experience.

## Framework Architecture

### Core Components Hierarchy
```
LiveKit Agents Framework
├── Agent Base Class
│   ├── Speech-to-Text (STT)
│   ├── Large Language Model (LLM)
│   ├── Text-to-Speech (TTS)
│   └── Voice Activity Detection (VAD)
├── Room Management
│   ├── WebRTC Connection
│   ├── Track Publishing
│   └── Participant Management
├── Pipeline Processing
│   ├── Audio Processing
│   ├── Function Calling
│   └── Context Management
└── Utilities
    ├── Logging
    ├── Metrics
    └── Configuration
```

## Installation & Setup

### Basic Installation
```bash
# Install LiveKit Agents framework
pip install livekit-agents==1.0.11

# Install providers
pip install livekit-agents[openai]     # For OpenAI STT/LLM
pip install livekit-agents[elevenlabs] # For ElevenLabs TTS
pip install livekit-agents[silero]     # For Silero VAD

# Install all providers
pip install livekit-agents[all]
```

### Environment Configuration
```python
import os
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli

# Environment variables
os.environ["LIVEKIT_URL"] = "wss://your-livekit-server.com"
os.environ["LIVEKIT_API_KEY"] = "your-api-key"
os.environ["LIVEKIT_API_SECRET"] = "your-api-secret"
os.environ["OPENAI_API_KEY"] = "your-openai-key"
os.environ["ELEVENLABS_API_KEY"] = "your-elevenlabs-key"

# Worker configuration
worker_options = WorkerOptions(
    entrypoint_fnc="main",
    prewarm_fnc="prewarm",
    rooms_auto_subscribe=AutoSubscribe.AUDIO_ONLY,
    worker_type="agent"
)
```

## Agent Base Class

### Basic Agent Implementation
```python
from livekit.agents import Agent, AutoSubscribe, JobContext, WorkerOptions, cli
from livekit.agents.llm import ChatContext, ChatMessage
from livekit.agents.voice_assistant import VoiceAssistant
from livekit import rtc

class BasicVoiceAgent(Agent):
    def __init__(self, ctx: JobContext):
        super().__init__(ctx)
        
        # Initialize components
        self.stt = openai.STT(
            model="whisper-1",
            language="en"
        )
        
        self.llm = openai.LLM(
            model="gpt-4o-mini",
            temperature=0.7
        )
        
        self.tts = elevenlabs.TTS(
            voice="Rachel",
            model="eleven_turbo_v2"
        )
        
        self.vad = silero.VAD.load()
        
        # Create voice assistant
        self.assistant = VoiceAssistant(
            vad=self.vad,
            stt=self.stt, 
            llm=self.llm,
            tts=self.tts,
            chat_ctx=ChatContext(
                messages=[
                    ChatMessage(
                        role="system",
                        content="You are a helpful voice assistant."
                    )
                ]
            )
        )
    
    async def entrypoint(self, ctx: JobContext):
        """Main entry point when agent joins room"""
        # Connect to room
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
        
        # Start voice assistant
        self.assistant.start(ctx.room)
        
        # Wait for participants
        await self.assistant.say("Hello! I'm ready to help.", allow_interruptions=True)
```

### Advanced Agent with Custom Logic
```python
class AdvancedVoiceAgent(Agent):
    def __init__(self, ctx: JobContext):
        super().__init__(ctx)
        self.setup_components()
        self.setup_function_calling()
        self.setup_metrics()
    
    def setup_components(self):
        """Initialize all voice components"""
        self.stt = openai.STT(
            model="whisper-1",
            language="en",
            detect_language=True  # Auto-detect language
        )
        
        self.llm = openai.LLM(
            model="gpt-4o",
            temperature=0.7,
            max_tokens=150,  # Control response length
            functions=[
                self.get_weather_function(),
                self.schedule_meeting_function(),
                self.search_knowledge_base_function()
            ]
        )
        
        self.tts = elevenlabs.TTS(
            voice="Rachel",
            model="eleven_turbo_v2",
            latency=1  # Optimize for speed
        )
        
        self.vad = silero.VAD.load(
            min_speech_duration=0.1,    # 100ms
            min_silence_duration=0.5,   # 500ms
            speech_threshold=0.5        # Sensitivity
        )
    
    def setup_function_calling(self):
        """Setup LLM function calling capabilities"""
        self.functions = {
            "get_weather": self.get_weather,
            "schedule_meeting": self.schedule_meeting,
            "search_knowledge": self.search_knowledge_base
        }
    
    def setup_metrics(self):
        """Initialize metrics collection"""
        self.metrics = MetricsCollector()
        self.conversation_start = time.time()
    
    async def entrypoint(self, ctx: JobContext):
        """Enhanced entry point with error handling"""
        try:
            await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
            
            # Setup assistant with custom callbacks
            self.assistant = VoiceAssistant(
                vad=self.vad,
                stt=self.stt,
                llm=self.llm,
                tts=self.tts,
                chat_ctx=self.create_chat_context(),
                fnc_ctx=FunctionContext(),  # Enable function calling
                interrupt_speech_duration=0.3,  # Allow quick interruptions
                interrupt_min_words=2,           # Minimum words to interrupt
                min_endpointing_delay=0.5,       # End-of-turn detection
                preemptive_synthesis=True        # Start TTS early
            )
            
            # Setup event handlers
            self.setup_event_handlers()
            
            # Start assistant
            self.assistant.start(ctx.room)
            
            # Initial greeting
            await self.assistant.say(
                "Hello! I'm your AI assistant. I can help with weather, scheduling, and general questions.",
                allow_interruptions=True
            )
            
        except Exception as e:
            logger.error(f"Error in agent entrypoint: {e}")
            await self.handle_error(e)
    
    def setup_event_handlers(self):
        """Setup event handlers for assistant lifecycle"""
        
        @self.assistant.on("user_speech_committed")
        def on_user_speech(message: ChatMessage):
            """Called when user speech is finalized"""
            self.metrics.track_user_speech(message.content)
            logger.info(f"User said: {message.content}")
        
        @self.assistant.on("agent_speech_committed") 
        def on_agent_speech(message: ChatMessage):
            """Called when agent speech is finalized"""
            self.metrics.track_agent_speech(message.content)
            logger.info(f"Agent said: {message.content}")
        
        @self.assistant.on("function_calls_finished")
        def on_function_calls(called_functions):
            """Called when function calls complete"""
            for func in called_functions:
                self.metrics.track_function_call(func.name, func.result)
                logger.info(f"Function called: {func.name}")
        
        @self.assistant.on("user_started_speaking")
        def on_user_started():
            """Called when user starts speaking"""
            self.metrics.track_user_speech_start()
        
        @self.assistant.on("user_stopped_speaking") 
        def on_user_stopped():
            """Called when user stops speaking"""
            self.metrics.track_user_speech_end()
    
    def create_chat_context(self):
        """Create initial chat context with system prompt"""
        return ChatContext(
            messages=[
                ChatMessage(
                    role="system",
                    content="""You are a helpful AI voice assistant. You can:
                    1. Provide weather information using the get_weather function
                    2. Schedule meetings using the schedule_meeting function  
                    3. Search the knowledge base using the search_knowledge function
                    4. Answer general questions conversationally
                    
                    Keep responses concise and natural for voice interaction.
                    Always use functions when appropriate rather than making up information."""
                )
            ]
        )
```

## Component Deep Dive

### Speech-to-Text (STT) Configuration

#### OpenAI Whisper STT
```python
from livekit.agents.stt import openai

# Basic configuration
stt = openai.STT(
    model="whisper-1",
    language="en"
)

# Advanced configuration
stt = openai.STT(
    model="whisper-1",
    language="en",
    detect_language=True,           # Auto-detect language
    word_timestamps=True,           # Get word-level timestamps
    no_speech_threshold=0.6,        # Threshold for silence
    temperature=0.0,                # Deterministic output
    compression_ratio_threshold=2.4, # Quality control
    logprob_threshold=-1.0,         # Confidence filtering
    initial_prompt="This is a conversation with an AI assistant."
)

# Custom STT implementation
class CustomSTT(STT):
    def __init__(self, custom_model_url: str):
        super().__init__()
        self.model_url = custom_model_url
        self.session = aiohttp.ClientSession()
    
    async def recognize(self, buffer: AudioBuffer) -> SpeechEvent:
        """Custom recognition implementation"""
        audio_data = buffer.data
        
        # Send to custom STT service
        async with self.session.post(
            self.model_url,
            data=audio_data,
            headers={"Content-Type": "audio/wav"}
        ) as response:
            result = await response.json()
            
            return SpeechEvent(
                type=SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[
                    SpeechData(
                        text=result["transcript"],
                        confidence=result["confidence"]
                    )
                ]
            )
```

#### Azure Speech STT
```python
from livekit.agents.stt import azure

stt = azure.STT(
    speech_key="your-azure-key",
    speech_region="your-region",
    language="en-US",
    recognition_mode="Conversation",  # or "Interactive", "Dictation"
    profanity_filter=True,
    output_format="Simple"  # or "Detailed"
)
```

### Large Language Model (LLM) Configuration

#### OpenAI LLM
```python
from livekit.agents.llm import openai

# Basic configuration
llm = openai.LLM(
    model="gpt-4o-mini",
    temperature=0.7
)

# Advanced configuration with function calling
llm = openai.LLM(
    model="gpt-4o",
    temperature=0.7,
    max_tokens=150,
    top_p=0.9,
    frequency_penalty=0.1,
    presence_penalty=0.1,
    functions=[
        FunctionDefinition(
            name="get_current_weather",
            description="Get current weather for a location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name or coordinates"
                    },
                    "units": {
                        "type": "string", 
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature units"
                    }
                },
                "required": ["location"]
            }
        )
    ]
)
```

#### Custom LLM Implementation
```python
from livekit.agents.llm import LLM, ChatContext, ChatMessage

class CustomLLM(LLM):
    def __init__(self, api_endpoint: str, model_name: str):
        super().__init__()
        self.api_endpoint = api_endpoint
        self.model_name = model_name
        self.session = aiohttp.ClientSession()
    
    async def chat(
        self, 
        chat_ctx: ChatContext, 
        fnc_ctx: Optional[FunctionContext] = None
    ) -> ChatResponse:
        """Custom LLM chat implementation"""
        
        # Prepare messages for your API format
        messages = [
            {
                "role": msg.role,
                "content": msg.content
            } 
            for msg in chat_ctx.messages
        ]
        
        # Call custom API
        async with self.session.post(
            f"{self.api_endpoint}/chat/completions",
            json={
                "model": self.model_name,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 150
            }
        ) as response:
            result = await response.json()
            
            content = result["choices"][0]["message"]["content"]
            
            return ChatResponse(
                content=content,
                function_calls=[]  # Add function call parsing if needed
            )
```

### Text-to-Speech (TTS) Configuration

#### ElevenLabs TTS
```python
from livekit.agents.tts import elevenlabs

# Basic configuration
tts = elevenlabs.TTS(
    voice="Rachel",
    model="eleven_turbo_v2"
)

# Advanced configuration
tts = elevenlabs.TTS(
    voice="Rachel",
    model="eleven_turbo_v2", 
    latency=1,               # Optimize for speed
    stability=0.5,           # Voice stability (0-1)
    similarity_boost=0.75,   # Voice similarity (0-1)
    style=0.0,              # Style exaggeration (0-1)
    use_speaker_boost=True   # Enhance speaker clarity
)

# Multi-voice configuration
class MultiVoiceTTS:
    def __init__(self):
        self.voices = {
            "assistant": elevenlabs.TTS(voice="Rachel", model="eleven_turbo_v2"),
            "narrator": elevenlabs.TTS(voice="Antoni", model="eleven_turbo_v2"),
            "child": elevenlabs.TTS(voice="Bella", model="eleven_turbo_v2")
        }
        self.current_voice = "assistant"
    
    async def synthesize(self, text: str, voice_type: str = None) -> AudioBuffer:
        voice = voice_type or self.current_voice
        return await self.voices[voice].synthesize(text)
```

#### Azure TTS
```python
from livekit.agents.tts import azure

tts = azure.TTS(
    speech_key="your-azure-key",
    speech_region="your-region",
    voice="en-US-JennyNeural",
    language="en-US",
    speaking_rate=1.0,      # Speed adjustment
    pitch="+0Hz",           # Pitch adjustment
    volume="+0%"            # Volume adjustment
)
```

### Voice Activity Detection (VAD)

#### Silero VAD
```python
from livekit.agents.vad import silero

# Basic configuration
vad = silero.VAD.load()

# Advanced configuration
vad = silero.VAD.load(
    min_speech_duration=0.1,    # Minimum speech duration (100ms)
    min_silence_duration=0.5,   # Minimum silence duration (500ms)
    speech_threshold=0.5,       # Speech detection threshold (0-1)
    sample_rate=16000,          # Audio sample rate
    window_size_samples=512     # Analysis window size
)

# Custom VAD implementation
class CustomVAD(VAD):
    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        self.is_speaking = False
    
    async def stream_audio(self, audio_stream: AudioStream) -> VADStream:
        """Process audio stream for voice activity"""
        async for frame in audio_stream:
            # Custom voice activity detection logic
            energy = self.calculate_audio_energy(frame)
            
            if energy > self.threshold and not self.is_speaking:
                self.is_speaking = True
                yield VADEvent(type=VADEventType.START_OF_SPEECH)
            elif energy <= self.threshold and self.is_speaking:
                self.is_speaking = False  
                yield VADEvent(type=VADEventType.END_OF_SPEECH)
    
    def calculate_audio_energy(self, frame: AudioFrame) -> float:
        """Calculate RMS energy of audio frame"""
        return float(np.sqrt(np.mean(frame.data ** 2)))
```

## Room Management

### Connection Management
```python
from livekit import rtc

class RoomManager:
    def __init__(self, room: rtc.Room):
        self.room = room
        self.participants = {}
        self.setup_event_handlers()
    
    def setup_event_handlers(self):
        """Setup room event handlers"""
        
        @self.room.on("participant_connected")
        def on_participant_connected(participant: rtc.RemoteParticipant):
            logger.info(f"Participant {participant.identity} connected")
            self.participants[participant.identity] = participant
            
            # Subscribe to participant's audio tracks
            for publication in participant.track_publications.values():
                if publication.track and publication.track.kind == rtc.TrackKind.KIND_AUDIO:
                    publication.set_subscribed(True)
        
        @self.room.on("participant_disconnected")
        def on_participant_disconnected(participant: rtc.RemoteParticipant):
            logger.info(f"Participant {participant.identity} disconnected")
            if participant.identity in self.participants:
                del self.participants[participant.identity]
        
        @self.room.on("track_published")
        def on_track_published(publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
            logger.info(f"Track {publication.sid} published by {participant.identity}")
            
            # Auto-subscribe to audio tracks
            if publication.kind == rtc.TrackKind.KIND_AUDIO:
                publication.set_subscribed(True)
        
        @self.room.on("track_subscribed")
        def on_track_subscribed(track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
            logger.info(f"Subscribed to {track.kind} track from {participant.identity}")
            
            if track.kind == rtc.TrackKind.KIND_AUDIO:
                # Process incoming audio
                self.process_participant_audio(track, participant)
    
    async def process_participant_audio(self, track: rtc.AudioTrack, participant: rtc.RemoteParticipant):
        """Process audio from a participant"""
        audio_stream = rtc.AudioStream(track)
        
        async for frame in audio_stream:
            # Process audio frame (send to VAD, STT, etc.)
            await self.handle_audio_frame(frame, participant)
    
    async def handle_audio_frame(self, frame: rtc.AudioFrame, participant: rtc.RemoteParticipant):
        """Handle individual audio frame"""
        # Send to voice assistant for processing
        await self.voice_assistant.process_audio(frame, participant.identity)
```

### Audio Track Publishing
```python
class AudioPublisher:
    def __init__(self, room: rtc.Room):
        self.room = room
        self.audio_source = None
        self.audio_track = None
    
    async def start_publishing(self):
        """Start publishing audio to the room"""
        # Create audio source
        self.audio_source = rtc.AudioSource(
            sample_rate=16000,
            num_channels=1
        )
        
        # Create audio track
        self.audio_track = rtc.LocalAudioTrack.create_audio_track(
            "agent-audio",
            self.audio_source
        )
        
        # Publish to room
        options = rtc.TrackPublishOptions(
            source=rtc.TrackSource.SOURCE_MICROPHONE
        )
        
        await self.room.local_participant.publish_track(
            self.audio_track,
            options
        )
    
    async def play_audio(self, audio_data: bytes):
        """Play audio through the published track"""
        if self.audio_source:
            # Convert audio data to AudioFrame
            frame = rtc.AudioFrame(
                data=audio_data,
                sample_rate=16000,
                num_channels=1,
                samples_per_channel=len(audio_data) // 2  # 16-bit audio
            )
            
            await self.audio_source.capture_frame(frame)
    
    async def stop_publishing(self):
        """Stop publishing audio"""
        if self.audio_track:
            await self.room.local_participant.unpublish_track(self.audio_track)
            self.audio_track = None
            self.audio_source = None
```

## Pipeline Processing

### Audio Processing Pipeline
```python
class AudioProcessingPipeline:
    def __init__(self):
        self.vad = silero.VAD.load()
        self.stt = openai.STT()
        self.processors = []
        self.buffer = AudioBuffer()
        
    def add_processor(self, processor):
        """Add audio processor to pipeline"""
        self.processors.append(processor)
    
    async def process_audio_stream(self, audio_stream: AudioStream):
        """Process continuous audio stream"""
        async for frame in audio_stream:
            # Add to buffer
            self.buffer.append(frame)
            
            # Voice activity detection
            vad_result = await self.vad.process_frame(frame)
            
            if vad_result.is_speech:
                # Process speech frame
                await self.process_speech_frame(frame)
            else:
                # Process silence/noise frame
                await self.process_silence_frame(frame)
            
            # Run additional processors
            for processor in self.processors:
                frame = await processor.process(frame)
    
    async def process_speech_frame(self, frame: AudioFrame):
        """Process frame containing speech"""
        # Add to speech buffer for STT
        self.speech_buffer.append(frame)
        
        # Check if we have enough audio for STT
        if self.speech_buffer.duration > 0.5:  # 500ms
            # Send to STT for streaming recognition
            await self.stt.stream_audio(self.speech_buffer.data)
    
    async def process_silence_frame(self, frame: AudioFrame):
        """Process frame containing silence"""
        # Check if this ends a speech segment
        if self.speech_buffer.has_data():
            # Finalize STT recognition
            final_transcript = await self.stt.finalize()
            
            if final_transcript:
                await self.handle_transcript(final_transcript)
            
            # Clear speech buffer
            self.speech_buffer.clear()
```

### Function Calling System
```python
from livekit.agents.llm import FunctionContext, FunctionDefinition

class FunctionCallManager:
    def __init__(self):
        self.functions = {}
        self.context = FunctionContext()
        
    def register_function(self, name: str, func: callable, definition: dict):
        """Register a function for LLM calling"""
        self.functions[name] = func
        
        func_def = FunctionDefinition(
            name=name,
            description=definition.get("description", ""),
            parameters=definition.get("parameters", {})
        )
        
        self.context.add_function(func_def)
    
    async def execute_function(self, function_call) -> any:
        """Execute a function call from LLM"""
        function_name = function_call.name
        arguments = function_call.arguments
        
        if function_name not in self.functions:
            raise ValueError(f"Unknown function: {function_name}")
        
        try:
            # Execute function with provided arguments
            result = await self.functions[function_name](**arguments)
            return result
        except Exception as e:
            logger.error(f"Error executing function {function_name}: {e}")
            return {"error": str(e)}
    
    # Example function implementations
    async def get_weather(self, location: str, units: str = "celsius") -> dict:
        """Get current weather for location"""
        # Mock weather API call
        return {
            "location": location,
            "temperature": 22,
            "units": units,
            "condition": "sunny",
            "humidity": 65
        }
    
    async def schedule_meeting(self, title: str, datetime: str, participants: list) -> dict:
        """Schedule a meeting"""
        # Mock calendar API call
        meeting_id = f"meeting_{int(time.time())}"
        
        return {
            "meeting_id": meeting_id,
            "title": title,
            "datetime": datetime,
            "participants": participants,
            "status": "scheduled"
        }
    
    async def search_knowledge_base(self, query: str, limit: int = 5) -> dict:
        """Search internal knowledge base"""
        # Mock knowledge base search
        return {
            "query": query,
            "results": [
                {"title": "Result 1", "content": "Content 1", "score": 0.95},
                {"title": "Result 2", "content": "Content 2", "score": 0.87}
            ],
            "total_results": 2
        }

# Register functions with the manager
function_manager = FunctionCallManager()

function_manager.register_function(
    "get_weather",
    function_manager.get_weather,
    {
        "description": "Get current weather information for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name or location"
                },
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature units"
                }
            },
            "required": ["location"]
        }
    }
)
```

## Metrics & Monitoring

### Built-in Metrics Collection
```python
from livekit.agents.metrics import MetricsCollector, LatencyMetric, CounterMetric

class AgentMetrics:
    def __init__(self):
        self.collector = MetricsCollector()
        
        # Define metrics
        self.stt_latency = LatencyMetric("stt_processing_time")
        self.llm_latency = LatencyMetric("llm_response_time") 
        self.tts_latency = LatencyMetric("tts_synthesis_time")
        self.total_latency = LatencyMetric("total_response_time")
        
        self.conversation_count = CounterMetric("conversations_started")
        self.message_count = CounterMetric("messages_processed")
        self.function_calls = CounterMetric("function_calls_executed")
        
        # Register metrics
        self.collector.register_metric(self.stt_latency)
        self.collector.register_metric(self.llm_latency)
        self.collector.register_metric(self.tts_latency)
        self.collector.register_metric(self.total_latency)
        self.collector.register_metric(self.conversation_count)
        self.collector.register_metric(self.message_count)
        self.collector.register_metric(self.function_calls)
    
    async def track_stt_processing(self, start_time: float, end_time: float):
        """Track STT processing time"""
        latency = (end_time - start_time) * 1000  # Convert to ms
        await self.stt_latency.record(latency)
    
    async def track_llm_response(self, start_time: float, end_time: float):
        """Track LLM response time"""
        latency = (end_time - start_time) * 1000
        await self.llm_latency.record(latency)
    
    async def track_tts_synthesis(self, start_time: float, end_time: float):
        """Track TTS synthesis time"""
        latency = (end_time - start_time) * 1000
        await self.tts_latency.record(latency)
    
    async def track_total_response(self, start_time: float, end_time: float):
        """Track total response time"""
        latency = (end_time - start_time) * 1000
        await self.total_latency.record(latency)
    
    async def increment_conversation(self):
        """Increment conversation counter"""
        await self.conversation_count.increment()
    
    async def increment_message(self):
        """Increment message counter"""
        await self.message_count.increment()
    
    async def increment_function_call(self):
        """Increment function call counter"""
        await self.function_calls.increment()
    
    async def get_metrics_summary(self) -> dict:
        """Get summary of all metrics"""
        return {
            "stt_avg_latency": await self.stt_latency.get_average(),
            "llm_avg_latency": await self.llm_latency.get_average(),
            "tts_avg_latency": await self.tts_latency.get_average(),
            "total_avg_latency": await self.total_latency.get_average(),
            "total_conversations": await self.conversation_count.get_value(),
            "total_messages": await self.message_count.get_value(),
            "total_function_calls": await self.function_calls.get_value()
        }
```

### Custom Metrics Export
```python
import prometheus_client
from livekit.agents.metrics import MetricsExporter

class PrometheusMetricsExporter(MetricsExporter):
    def __init__(self, port: int = 8000):
        self.port = port
        
        # Create Prometheus metrics
        self.response_time_histogram = prometheus_client.Histogram(
            'agent_response_time_seconds',
            'Time taken for agent to respond',
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
        )
        
        self.conversation_counter = prometheus_client.Counter(
            'agent_conversations_total',
            'Total number of conversations'
        )
        
        self.error_counter = prometheus_client.Counter(
            'agent_errors_total',
            'Total number of errors',
            ['error_type']
        )
        
        # Start metrics server
        prometheus_client.start_http_server(self.port)
    
    async def export_response_time(self, latency_ms: float):
        """Export response time metric"""
        self.response_time_histogram.observe(latency_ms / 1000)
    
    async def export_conversation_start(self):
        """Export conversation start metric"""
        self.conversation_counter.inc()
    
    async def export_error(self, error_type: str):
        """Export error metric"""
        self.error_counter.labels(error_type=error_type).inc()
```

## Configuration Management

### Environment-Based Configuration
```python
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class AgentConfig:
    # LiveKit configuration
    livekit_url: str
    livekit_api_key: str
    livekit_api_secret: str
    
    # OpenAI configuration
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    openai_temperature: float = 0.7
    
    # ElevenLabs configuration
    elevenlabs_api_key: str
    elevenlabs_voice: str = "Rachel"
    elevenlabs_model: str = "eleven_turbo_v2"
    
    # VAD configuration
    vad_threshold: float = 0.5
    min_speech_duration: float = 0.1
    min_silence_duration: float = 0.5
    
    # Agent behavior
    interrupt_speech_duration: float = 0.3
    interrupt_min_words: int = 2
    min_endpointing_delay: float = 0.5
    preemptive_synthesis: bool = True
    
    # Metrics
    enable_metrics: bool = True
    metrics_port: int = 8000
    
    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Create configuration from environment variables"""
        return cls(
            livekit_url=os.environ["LIVEKIT_URL"],
            livekit_api_key=os.environ["LIVEKIT_API_KEY"],
            livekit_api_secret=os.environ["LIVEKIT_API_SECRET"],
            openai_api_key=os.environ["OPENAI_API_KEY"],
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            openai_temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.7")),
            elevenlabs_api_key=os.environ["ELEVENLABS_API_KEY"],
            elevenlabs_voice=os.getenv("ELEVENLABS_VOICE", "Rachel"),
            elevenlabs_model=os.getenv("ELEVENLABS_MODEL", "eleven_turbo_v2"),
            vad_threshold=float(os.getenv("VAD_THRESHOLD", "0.5")),
            min_speech_duration=float(os.getenv("MIN_SPEECH_DURATION", "0.1")),
            min_silence_duration=float(os.getenv("MIN_SILENCE_DURATION", "0.5")),
            interrupt_speech_duration=float(os.getenv("INTERRUPT_SPEECH_DURATION", "0.3")),
            interrupt_min_words=int(os.getenv("INTERRUPT_MIN_WORDS", "2")),
            min_endpointing_delay=float(os.getenv("MIN_ENDPOINTING_DELAY", "0.5")),
            preemptive_synthesis=os.getenv("PREEMPTIVE_SYNTHESIS", "true").lower() == "true",
            enable_metrics=os.getenv("ENABLE_METRICS", "true").lower() == "true",
            metrics_port=int(os.getenv("METRICS_PORT", "8000"))
        )

# Usage
config = AgentConfig.from_env()
```

## Error Handling & Recovery

### Robust Error Handling
```python
import asyncio
from typing import Dict, Any
from livekit.agents import Agent, JobContext

class RobustVoiceAgent(Agent):
    def __init__(self, ctx: JobContext):
        super().__init__(ctx)
        self.error_counts = {}
        self.max_retries = 3
        self.recovery_strategies = {
            "stt_error": self.recover_stt,
            "llm_error": self.recover_llm,
            "tts_error": self.recover_tts,
            "connection_error": self.recover_connection
        }
    
    async def entrypoint(self, ctx: JobContext):
        """Entry point with comprehensive error handling"""
        try:
            await self.initialize_components()
            await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
            await self.start_assistant()
            
        except Exception as e:
            await self.handle_critical_error(e)
    
    async def initialize_components(self):
        """Initialize components with error handling"""
        retry_count = 0
        
        while retry_count < self.max_retries:
            try:
                self.stt = openai.STT(model="whisper-1")
                self.llm = openai.LLM(model="gpt-4o-mini")
                self.tts = elevenlabs.TTS(voice="Rachel")
                self.vad = silero.VAD.load()
                break
                
            except Exception as e:
                retry_count += 1
                logger.error(f"Component initialization failed (attempt {retry_count}): {e}")
                
                if retry_count >= self.max_retries:
                    raise
                
                # Exponential backoff
                await asyncio.sleep(2 ** retry_count)
    
    async def handle_error(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Handle errors with recovery strategies"""
        error_type = self.classify_error(error)
        
        # Track error frequency
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Check if we've exceeded error threshold
        if self.error_counts[error_type] > 10:  # More than 10 errors of same type
            await self.escalate_error(error, error_type)
            return False
        
        # Attempt recovery
        recovery_strategy = self.recovery_strategies.get(error_type)
        if recovery_strategy:
            try:
                success = await recovery_strategy(error, context)
                if success:
                    logger.info(f"Successfully recovered from {error_type}")
                    return True
            except Exception as recovery_error:
                logger.error(f"Recovery failed for {error_type}: {recovery_error}")
        
        return False
    
    def classify_error(self, error: Exception) -> str:
        """Classify error type for appropriate recovery"""
        if "openai" in str(error).lower() or "stt" in str(error).lower():
            return "stt_error"
        elif "llm" in str(error).lower() or "gpt" in str(error).lower():
            return "llm_error"
        elif "elevenlabs" in str(error).lower() or "tts" in str(error).lower():
            return "tts_error"
        elif "connection" in str(error).lower() or "network" in str(error).lower():
            return "connection_error"
        else:
            return "unknown_error"
    
    async def recover_stt(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Recover from STT errors"""
        try:
            # Reinitialize STT with fallback model
            self.stt = openai.STT(model="whisper-1", language="en")
            await asyncio.sleep(1)  # Brief pause
            return True
        except:
            return False
    
    async def recover_llm(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Recover from LLM errors"""
        try:
            # Try different model or reduce parameters
            self.llm = openai.LLM(
                model="gpt-4o-mini",
                max_tokens=100,  # Reduce token limit
                temperature=0.5  # Reduce creativity
            )
            return True
        except:
            return False
    
    async def recover_tts(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Recover from TTS errors"""
        try:
            # Reinitialize with different voice or model
            self.tts = elevenlabs.TTS(
                voice="Antoni",  # Different voice
                model="eleven_turbo_v2"
            )
            return True
        except:
            return False
    
    async def recover_connection(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Recover from connection errors"""
        try:
            # Attempt to reconnect
            await self.ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
            return True
        except:
            return False
    
    async def escalate_error(self, error: Exception, error_type: str):
        """Escalate persistent errors"""
        logger.critical(f"Escalating {error_type} after repeated failures: {error}")
        
        # Could send to monitoring system, page on-call, etc.
        # For now, attempt graceful shutdown
        await self.graceful_shutdown()
    
    async def graceful_shutdown(self):
        """Perform graceful shutdown"""
        try:
            if hasattr(self, 'assistant'):
                await self.assistant.say("I'm experiencing technical difficulties. Please try again later.")
                await self.assistant.stop()
            
            if hasattr(self, 'ctx') and self.ctx.room:
                await self.ctx.room.disconnect()
                
        except Exception as e:
            logger.error(f"Error during graceful shutdown: {e}")
```

## Testing Framework

### Unit Testing Components
```python
import unittest
from unittest.mock import AsyncMock, MagicMock
from livekit.agents.test import TestCase, MockAudioStream, MockRoom

class TestVoiceAgent(TestCase):
    async def setUp(self):
        """Set up test environment"""
        self.mock_room = MockRoom()
        self.mock_ctx = MagicMock()
        self.mock_ctx.room = self.mock_room
        
        self.agent = VoiceAgent(self.mock_ctx)
    
    async def test_stt_processing(self):
        """Test STT component"""
        # Create mock audio
        mock_audio = MockAudioStream.create_speech("Hello, how are you?")
        
        # Process through STT
        result = await self.agent.stt.recognize(mock_audio.buffer)
        
        # Verify result
        self.assertIn("hello", result.text.lower())
        self.assertGreater(result.confidence, 0.8)
    
    async def test_llm_response(self):
        """Test LLM component"""
        # Create chat context
        chat_ctx = ChatContext(messages=[
            ChatMessage(role="user", content="What's the weather like?")
        ])
        
        # Get LLM response
        response = await self.agent.llm.chat(chat_ctx)
        
        # Verify response
        self.assertIsNotNone(response.content)
        self.assertGreater(len(response.content), 0)
    
    async def test_tts_synthesis(self):
        """Test TTS component"""
        text = "Hello, this is a test."
        
        # Synthesize audio
        audio_buffer = await self.agent.tts.synthesize(text)
        
        # Verify audio properties
        self.assertIsNotNone(audio_buffer)
        self.assertGreater(len(audio_buffer.data), 0)
        self.assertEqual(audio_buffer.sample_rate, 16000)
    
    async def test_function_calling(self):
        """Test function calling capability"""
        # Mock function
        async def mock_get_weather(location: str) -> dict:
            return {"temperature": 25, "condition": "sunny"}
        
        # Register function
        self.agent.function_manager.register_function(
            "get_weather", 
            mock_get_weather,
            {"description": "Get weather", "parameters": {"type": "object"}}
        )
        
        # Test function call
        result = await self.agent.function_manager.execute_function(
            MockFunctionCall("get_weather", {"location": "New York"})
        )
        
        # Verify result
        self.assertEqual(result["temperature"], 25)
        self.assertEqual(result["condition"], "sunny")
    
    async def test_error_recovery(self):
        """Test error recovery mechanisms"""
        # Simulate STT error
        self.agent.stt = AsyncMock(side_effect=Exception("STT failed"))
        
        # Attempt recovery
        success = await self.agent.handle_error(
            Exception("STT failed"), 
            {"component": "stt"}
        )
        
        # Verify recovery attempt
        self.assertTrue(success)
```

### Integration Testing
```python
class TestVoiceAgentIntegration(TestCase):
    async def setUp(self):
        """Set up integration test environment"""
        self.test_room = await self.create_test_room()
        self.agent = VoiceAgent(self.test_room.context)
        await self.agent.initialize()
    
    async def test_full_conversation_flow(self):
        """Test complete conversation flow"""
        # Start agent
        await self.agent.start()
        
        # Simulate user speech
        user_audio = MockAudioStream.create_speech("What's the weather in London?")
        await self.test_room.simulate_user_audio(user_audio)
        
        # Wait for agent response
        response = await self.wait_for_agent_response(timeout=5.0)
        
        # Verify response
        self.assertIsNotNone(response)
        self.assertIn("weather", response.content.lower())
        
        # Verify metrics
        metrics = await self.agent.metrics.get_summary()
        self.assertGreater(metrics["total_messages"], 0)
        self.assertLess(metrics["total_avg_latency"], 2000)  # Under 2 seconds
    
    async def test_interruption_handling(self):
        """Test user interruption scenarios"""
        # Start agent speaking
        await self.agent.start_speaking("This is a long response that will be interrupted...")
        
        # Simulate user interruption
        interrupt_audio = MockAudioStream.create_speech("Wait, I have a question")
        await self.test_room.simulate_user_audio(interrupt_audio)
        
        # Verify agent stops and responds
        self.assertFalse(self.agent.is_speaking)
        
        response = await self.wait_for_agent_response(timeout=2.0)
        self.assertIsNotNone(response)
    
    async def test_concurrent_users(self):
        """Test handling multiple concurrent users"""
        num_users = 5
        tasks = []
        
        for i in range(num_users):
            task = self.simulate_user_conversation(f"user_{i}")
            tasks.append(task)
        
        # Run concurrent conversations
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all conversations completed successfully
        for result in results:
            self.assertIsNotInstance(result, Exception)
    
    async def simulate_user_conversation(self, user_id: str) -> dict:
        """Simulate a complete user conversation"""
        conversation_steps = [
            "Hello, how are you?",
            "What's the weather like?", 
            "Thank you, goodbye!"
        ]
        
        responses = []
        for step in conversation_steps:
            audio = MockAudioStream.create_speech(step)
            await self.test_room.simulate_user_audio(audio, user_id)
            
            response = await self.wait_for_agent_response(timeout=3.0)
            responses.append(response)
        
        return {"user_id": user_id, "responses": responses}
```

## Deployment Scripts

### Production Deployment
```python
#!/usr/bin/env python3
"""
Production deployment script for LiveKit Voice Agent
"""

import asyncio
import os
import signal
import sys
from livekit.agents import cli, WorkerOptions, AutoSubscribe

# Import your agent
from voice_agent import VoiceAgent

async def prewarm(ctx):
    """Prewarm function to initialize resources"""
    logger.info("Prewarming agent resources...")
    
    # Prewarm models
    await prewarm_stt_model()
    await prewarm_llm_model() 
    await prewarm_tts_model()
    await prewarm_vad_model()
    
    logger.info("Prewarm complete")

async def prewarm_stt_model():
    """Prewarm STT model"""
    stt = openai.STT(model="whisper-1")
    # Process dummy audio to load model
    dummy_audio = generate_dummy_audio(duration=1.0)
    await stt.recognize(dummy_audio)

async def prewarm_llm_model():
    """Prewarm LLM model"""
    llm = openai.LLM(model="gpt-4o-mini")
    # Send dummy prompt to initialize
    dummy_context = ChatContext(messages=[
        ChatMessage(role="user", content="Hello")
    ])
    await llm.chat(dummy_context)

async def prewarm_tts_model():
    """Prewarm TTS model"""
    tts = elevenlabs.TTS(voice="Rachel")
    # Synthesize dummy text to load model
    await tts.synthesize("Hello")

async def prewarm_vad_model():
    """Prewarm VAD model"""
    vad = silero.VAD.load()
    # Process dummy audio to load model
    dummy_audio = generate_dummy_audio(duration=0.5)
    await vad.process_audio(dummy_audio)

def generate_dummy_audio(duration: float) -> AudioBuffer:
    """Generate dummy audio for prewarming"""
    # Generate silence
    sample_rate = 16000
    samples = int(duration * sample_rate)
    data = np.zeros(samples, dtype=np.int16)
    
    return AudioBuffer(data=data, sample_rate=sample_rate)

async def entrypoint(ctx):
    """Main entrypoint for the agent"""
    agent = VoiceAgent(ctx)
    await agent.entrypoint(ctx)

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    sys.exit(0)

if __name__ == "__main__":
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Configure worker options
    worker_options = WorkerOptions(
        entrypoint_fnc=entrypoint,
        prewarm_fnc=prewarm,
        rooms_auto_subscribe=AutoSubscribe.AUDIO_ONLY,
        worker_type="agent",
        max_retry_count=3,
        retry_delay=2.0
    )
    
    # Start the agent
    cli.run_app(worker_options)
```

### Docker Configuration
```dockerfile
# Dockerfile for LiveKit Voice Agent
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 agent && chown -R agent:agent /app
USER agent

# Expose metrics port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the agent
CMD ["python", "-m", "livekit.agents", "start", "agent.py"]
```

### Kubernetes Deployment
```yaml
# kubernetes-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: voice-agent
  labels:
    app: voice-agent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: voice-agent
  template:
    metadata:
      labels:
        app: voice-agent
    spec:
      containers:
      - name: voice-agent
        image: voice-agent:latest
        ports:
        - containerPort: 8000
          name: metrics
        env:
        - name: LIVEKIT_URL
          value: "wss://your-livekit-server.com"
        - name: LIVEKIT_API_KEY
          valueFrom:
            secretKeyRef:
              name: livekit-secrets
              key: api-key
        - name: LIVEKIT_API_SECRET
          valueFrom:
            secretKeyRef:
              name: livekit-secrets
              key: api-secret
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: openai-secrets
              key: api-key
        - name: ELEVENLABS_API_KEY
          valueFrom:
            secretKeyRef:
              name: elevenlabs-secrets
              key: api-key
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10

---
apiVersion: v1
kind: Service
metadata:
  name: voice-agent-service
spec:
  selector:
    app: voice-agent
  ports:
  - name: metrics
    port: 8000
    targetPort: 8000
  type: ClusterIP

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: voice-agent-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: voice-agent
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## Best Practices

### Performance Optimization
1. **Prewarming**: Load models during startup to reduce first-response latency
2. **Streaming**: Use streaming STT and TTS for better responsiveness
3. **Caching**: Cache frequent responses and model outputs
4. **Resource Management**: Monitor and optimize CPU/memory usage
5. **Concurrent Processing**: Handle multiple conversations efficiently

### Security Considerations
1. **API Key Management**: Use secure secret management systems
2. **Audio Encryption**: Encrypt audio data in transit and at rest
3. **Access Control**: Implement proper authentication and authorization
4. **Audit Logging**: Track all interactions for security analysis
5. **Data Retention**: Implement appropriate data retention policies

### Monitoring & Observability
1. **Metrics Collection**: Track latency, error rates, and usage patterns
2. **Logging**: Implement structured logging with correlation IDs
3. **Alerting**: Set up alerts for critical failures and performance degradation
4. **Distributed Tracing**: Track requests across service boundaries
5. **Health Checks**: Implement comprehensive health check endpoints

### Scalability Patterns
1. **Horizontal Scaling**: Design for multiple concurrent agent instances
2. **Load Balancing**: Distribute traffic across agent instances
3. **Resource Isolation**: Isolate compute resources per conversation
4. **Async Processing**: Use async/await patterns throughout
5. **Circuit Breakers**: Implement circuit breakers for external dependencies

## Next Steps

1. **Quick Start**: Begin with [Quick Start Guide](quick-start-guide.md)
2. **Applications**: Explore use cases in [Applications Guide](applications-guide.md)
3. **Performance**: Optimize with [Performance Guide](performance-optimization.md)
4. **Implementation**: Complete tutorial in [Implementation Guide](implementation-tutorial.md)

## Key Takeaways

- **Component Architecture**: Modular design with STT, LLM, TTS, and VAD components
- **Event-Driven Design**: Async event handling for responsive interactions
- **Error Recovery**: Comprehensive error handling and recovery strategies
- **Metrics Integration**: Built-in metrics collection and export capabilities
- **Production Ready**: Robust deployment patterns for production environments
- **Extensibility**: Framework designed for customization and extension