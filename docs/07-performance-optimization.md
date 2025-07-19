# Performance Optimization & Metrics Guide

## Overview

Voice agents live or die by latency. This guide covers comprehensive performance optimization strategies, metrics collection, and monitoring techniques to ensure your voice agent feels fast and responsive to users.

## Human Conversation Expectations

### Response Time Baselines
- **Average Human Response**: 236 milliseconds
- **Natural Variability**: Up to 520 milliseconds
- **User Notice Threshold**: ~1 second becomes noticeably slow
- **Language Variations**: Response times vary significantly by language

### Why Latency Matters
Users expect voice conversations to feel natural. Even small delays can make interactions feel unnatural and break the conversational flow.

## Voice Agent Latency Breakdown

### Pipeline Component Latencies

| Component | Best Case | Typical | Worst Case | Notes |
|-----------|-----------|---------|------------|-------|
| **VAD Confirmation** | 15ms | 50ms | 100ms | Detection delay |
| **Speech-to-Text** | 50ms | 200ms | 500ms | Depends on utterance length |
| **LLM Processing** | 200ms | 600ms | 2000ms | Most variable component |
| **Text-to-Speech** | 100ms | 300ms | 800ms | Voice complexity dependent |
| **Network/Transport** | 20ms | 100ms | 300ms | Geographic/connectivity |
| **Total End-to-End** | **385ms** | **1250ms** | **3700ms** | Sum of all components |

### Critical Metrics to Track

#### Time to First Token (TTFT)
- **Definition**: Time from LLM receiving input to first token generation
- **Impact**: Determines perceived responsiveness
- **Target**: < 400ms for good experience
- **Optimization**: Most important metric to optimize

#### Time to First Byte (TTFB)
- **Definition**: Time from TTS receiving text to first audio byte
- **Impact**: When user hears agent start speaking
- **Target**: < 200ms for natural flow
- **Optimization**: Critical for conversational feel

## Implementing Performance Metrics

### Metrics Collection Setup

```python
import asyncio
from livekit.agents.metrics import LLMMetrics, STTMetrics, TTSMetrics, EOUMetrics

class MetricsAgent(Agent):
    def __init__(self) -> None:
        # Initialize components
        llm = openai.LLM(model="gpt-4o-mini")
        stt = openai.STT(model="whisper-1")
        tts = elevenlabs.TTS()
        vad = silero.VAD.load()
        
        super().__init__(
            instructions="You are a helpful assistant communicating via voice",
            stt=stt,
            llm=llm,
            tts=tts,
            vad=vad,
        )

        # Set up metrics collection
        self.setup_metrics_collection(llm, stt, tts)

    def setup_metrics_collection(self, llm, stt, tts):
        """Configure metrics collection for all components"""
        
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
```

### Metrics Handler Implementation

```python
    async def on_llm_metrics_collected(self, metrics: LLMMetrics) -> None:
        """Handle LLM performance metrics"""
        print("\n--- LLM Metrics ---")
        print(f"Prompt Tokens: {metrics.prompt_tokens}")
        print(f"Completion Tokens: {metrics.completion_tokens}")
        print(f"Tokens per Second: {metrics.tokens_per_second:.4f}")
        print(f"Time to First Token: {metrics.ttft:.4f}s")
        print("------------------\n")
        
        # Log for analysis
        self.log_metric("llm_ttft", metrics.ttft)
        self.log_metric("llm_tokens_per_second", metrics.tokens_per_second)

    async def on_stt_metrics_collected(self, metrics: STTMetrics) -> None:
        """Handle STT performance metrics"""
        print("\n--- STT Metrics ---")
        print(f"Duration: {metrics.duration:.4f}s")
        print(f"Audio Duration: {metrics.audio_duration:.4f}s")
        print(f"Streamed: {'Yes' if metrics.streamed else 'No'}")
        print("------------------\n")
        
        # Calculate real-time factor
        real_time_factor = metrics.duration / metrics.audio_duration
        self.log_metric("stt_real_time_factor", real_time_factor)

    async def on_eou_metrics_collected(self, metrics: EOUMetrics) -> None:
        """Handle End of Utterance metrics"""
        print("\n--- End of Utterance Metrics ---")
        print(f"End of Utterance Delay: {metrics.end_of_utterance_delay:.4f}s")
        print(f"Transcription Delay: {metrics.transcription_delay:.4f}s")
        print("--------------------------------\n")
        
        self.log_metric("eou_delay", metrics.end_of_utterance_delay)

    async def on_tts_metrics_collected(self, metrics: TTSMetrics) -> None:
        """Handle TTS performance metrics"""
        print("\n--- TTS Metrics ---")
        print(f"Time to First Byte: {metrics.ttfb:.4f}s")
        print(f"Duration: {metrics.duration:.4f}s")
        print(f"Audio Duration: {metrics.audio_duration:.4f}s")
        print(f"Streamed: {'Yes' if metrics.streamed else 'No'}")
        print("------------------\n")
        
        self.log_metric("tts_ttfb", metrics.ttfb)
```

## Optimization Strategies

### 1. LLM Optimization

#### Model Selection Impact
```python
# Performance comparison from course testing
MODELS = {
    "gpt-4o": {
        "ttft": "~800ms",
        "quality": "excellent", 
        "cost": "high"
    },
    "gpt-4o-mini": {
        "ttft": "~400ms",  # Almost 2x faster
        "quality": "good",
        "cost": "low"
    }
}

# Optimization example
llm = openai.LLM(
    model="gpt-4o-mini",  # 50% faster TTFT
    temperature=0.3,      # More deterministic = faster
)
```

#### Prompt Optimization
```python
# Good: Concise, specific instructions
instructions = """You are a helpful assistant communicating via voice.
Keep responses under 50 words. Be conversational and direct."""

# Avoid: Long instructions that increase processing time
```

#### Context Management
```python
# Automatic context optimization
class OptimizedAgent(Agent):
    def __init__(self):
        super().__init__(
            # Agent automatically manages:
            # - Context window limits
            # - Message history pruning
            # - Token count optimization
        )
```

### 2. STT Optimization

#### Streaming Configuration
```python
# Optimized STT setup
stt = openai.STT(
    model="whisper-1",
    language="en",        # Fixed language is faster than auto-detect
    temperature=0.0,      # Deterministic for consistency
)
```

#### Language-Specific Optimization
```python
# Single language (faster)
stt = openai.STT(language="en")

# Multi-language (more flexible, slower)
stt = openai.STT(language=None)  # Auto-detect
```

### 3. TTS Optimization

#### Voice and Model Selection
```python
# Optimized for speed
tts = elevenlabs.TTS(
    voice_id="21m00Tcm4TlvDq8ikWAM",  # Rachel - well-optimized voice
    model="eleven_monolingual_v1",    # Faster than multilingual
    encoding="mp3_44100_128",         # Good quality/speed balance
)
```

#### Sentence-by-Sentence Processing
The agent automatically processes TTS sentence-by-sentence rather than waiting for complete responses, reducing perceived latency.

### 4. VAD Optimization

#### Environment-Tuned Settings
```python
# Low-latency configuration
vad = silero.VAD.load(
    min_speech_duration=0.1,    # Quick detection
    min_silence_duration=0.3,   # Shorter silence tolerance
    speech_threshold=0.5,       # Balanced sensitivity
)

# High-accuracy configuration  
vad = silero.VAD.load(
    min_speech_duration=0.2,    # Avoid false positives
    min_silence_duration=0.5,   # Natural pause tolerance
    speech_threshold=0.6,       # Higher confidence
)
```

## Streaming Optimization

### Pipeline Streaming Strategy

```
User Speech → VAD → STT (streaming) → LLM (streaming) → TTS (streaming) → Audio
```

#### Key Principles:
1. **Never Wait for Completion**: Process as data becomes available
2. **Parallel Processing**: Multiple components work simultaneously  
3. **Sentence-Level TTS**: Start speaking before complete response ready
4. **Token-Level LLM**: Stream tokens to TTS as soon as available

### Implementation Example
```python
class StreamingOptimizedAgent(Agent):
    def __init__(self):
        # All components configured for streaming
        llm = openai.LLM(
            model="gpt-4o-mini",
            stream=True,  # Enable token streaming
        )
        
        stt = openai.STT(
            model="whisper-1",
            stream=True,  # Enable audio streaming
        )
        
        tts = elevenlabs.TTS(
            stream=True,  # Enable audio streaming
        )
        
        super().__init__(
            stt=stt,
            llm=llm,
            tts=tts,
            # Agent handles streaming coordination
        )
```

## Performance Monitoring

### Real-Time Dashboards

```python
class PerformanceMonitor:
    def __init__(self):
        self.metrics = {
            "llm_ttft": [],
            "tts_ttfb": [],
            "end_to_end": [],
            "user_interruptions": 0,
        }
    
    def log_metric(self, metric_name: str, value: float):
        """Log performance metric with timestamp"""
        if metric_name in self.metrics:
            self.metrics[metric_name].append({
                "value": value,
                "timestamp": time.time()
            })
    
    def get_performance_summary(self):
        """Generate performance summary"""
        summary = {}
        for metric, values in self.metrics.items():
            if values and isinstance(values, list):
                recent_values = [v["value"] for v in values[-10:]]  # Last 10
                summary[metric] = {
                    "avg": sum(recent_values) / len(recent_values),
                    "min": min(recent_values),
                    "max": max(recent_values),
                    "count": len(values)
                }
        return summary
```

### Performance Alerts
```python
class PerformanceAlerting:
    THRESHOLDS = {
        "llm_ttft": 500,      # 500ms threshold
        "tts_ttfb": 300,      # 300ms threshold
        "end_to_end": 1000,   # 1s total threshold
    }
    
    def check_performance(self, metric_name: str, value: float):
        """Alert on performance degradation"""
        threshold = self.THRESHOLDS.get(metric_name)
        if threshold and value > threshold:
            self.alert(f"Performance issue: {metric_name} = {value}ms > {threshold}ms")
```

## Network and Infrastructure Optimization

### WebRTC Optimization
```python
# LiveKit handles WebRTC optimization automatically:
# - Adaptive bitrate
# - Network condition adaptation  
# - Jitter buffer management
# - Packet loss recovery
```

### Geographic Distribution
- Use CDN/edge deployments closer to users
- Consider multi-region LiveKit deployments
- Monitor network latency between components

### Connection Quality Monitoring
```python
async def monitor_connection_quality(room):
    """Monitor WebRTC connection quality"""
    stats = await room.get_stats()
    
    # Key metrics to monitor:
    # - Round-trip time (RTT)
    # - Packet loss percentage
    # - Jitter
    # - Bandwidth utilization
```

## Performance Testing

### Load Testing Setup
```python
import asyncio
import time

class PerformanceTestSuite:
    async def test_response_latency(self, agent, test_phrases):
        """Test end-to-end response latency"""
        results = []
        
        for phrase in test_phrases:
            start_time = time.time()
            
            # Simulate user input
            await agent.process_speech_input(phrase)
            
            # Wait for response start
            await agent.wait_for_response_start()
            
            end_time = time.time()
            latency = (end_time - start_time) * 1000  # Convert to ms
            
            results.append({
                "phrase": phrase,
                "latency_ms": latency
            })
        
        return results
    
    async def test_concurrent_users(self, agent_factory, num_users=10):
        """Test performance under concurrent load"""
        tasks = []
        
        for i in range(num_users):
            agent = agent_factory()
            task = asyncio.create_task(self.simulate_conversation(agent))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return self.analyze_concurrent_performance(results)
```

## Common Performance Issues

### High LLM Latency
**Symptoms**: > 1 second response times
**Solutions**:
- Switch to gpt-4o-mini instead of gpt-4o
- Reduce instruction length
- Use faster inference providers (Groq, Cerebras)
- Implement prompt caching

### Poor Audio Quality
**Symptoms**: Robotic or choppy audio
**Solutions**:
- Check network bandwidth
- Verify audio encoding settings
- Monitor TTS streaming performance
- Test with different voice models

### Frequent Interruptions
**Symptoms**: Agent stops speaking unexpectedly
**Solutions**:
- Tune VAD sensitivity settings
- Check microphone input levels
- Adjust interruption thresholds
- Monitor background noise levels

### Context Loss
**Symptoms**: Agent forgets previous conversation
**Solutions**:
- Verify conversation history is maintained
- Check context window limits
- Monitor memory usage
- Implement context compression

## Best Practices Summary

### 1. Model Selection
- **Development**: Start with gpt-4o-mini for faster iteration
- **Production**: Balance quality needs vs latency requirements
- **Testing**: Compare models with real user scenarios

### 2. Streaming Everything
- Enable streaming for all components (STT, LLM, TTS)
- Process data as it becomes available
- Never wait for complete responses

### 3. Monitor Continuously
- Track key metrics (TTFT, TTFB, end-to-end)
- Set up alerting for performance degradation
- Analyze performance trends over time

### 4. Optimize for Environment
- Tune VAD settings for your acoustic environment
- Consider network conditions and geography
- Test with real user devices and conditions

### 5. User Experience Focus
- Prioritize perceived responsiveness over absolute speed
- Test with actual users to validate performance
- Consider conversation flow and natural pauses

## Performance Targets

### Good Performance Benchmarks
- **LLM TTFT**: < 400ms
- **TTS TTFB**: < 200ms  
- **End-to-End**: < 800ms
- **User Satisfaction**: Feels natural and responsive

### Excellent Performance Benchmarks
- **LLM TTFT**: < 250ms
- **TTS TTFB**: < 150ms
- **End-to-End**: < 500ms
- **User Satisfaction**: Indistinguishable from human response time

## Next Steps

1. **Turn Detection**: Advanced conversation management in [Turn Detection Guide](turn-detection-guide.md)
2. **Implementation**: Complete setup in [Implementation Tutorial](implementation-tutorial.md)
3. **Applications**: Domain-specific optimization in [Applications Guide](applications-guide.md)
4. **Framework**: LiveKit specifics in [LiveKit Reference](livekit-reference.md)