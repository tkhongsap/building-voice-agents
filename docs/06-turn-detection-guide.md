# Turn Detection & Conversation Management Guide

## Overview

Turn detection is one of the hardest problems in building convincing voice agents. This guide covers the sophisticated mechanisms that enable natural conversation flow, interruption handling, and context management in voice agents.

## Understanding Turn-Taking in Human Conversation

### Natural Human Behavior
In human conversation, people intuitively know when to speak or listen. This automatic alternating between speaking and listening is called **turn-taking**. Voice agents must replicate this behavior to feel natural.

### Challenges for AI Systems
- **Timing Uncertainty**: How long should the agent wait before responding?
- **Pause Interpretation**: Is silence the end of a turn or just a pause for thought?
- **Interruption Handling**: How to gracefully handle when users interrupt?
- **Context Preservation**: Maintaining conversation state during interruptions

## Turn Detection Architecture

### Dual-Signal Approach

Turn detection systems combine two complementary signals:

```
User Audio → VAD (Signal Processing) → Turn Decision
           ↓
User Audio → STT → Semantic Model → Turn Decision
```

#### 1. Signal Processing (VAD)
- **Function**: Analyzes raw audio signal
- **Output**: Binary classification (speech detected / no speech)
- **Speed**: Very fast (real-time)
- **Limitation**: No understanding of content

#### 2. Semantic Processing
- **Function**: Analyzes transcribed content meaning
- **Output**: Prediction if user is finished speaking
- **Speed**: Requires transcription first
- **Advantage**: Understands conversational context

### Implementation Details

#### VAD-Based Turn Detection
```python
class TurnDetector:
    def __init__(self):
        self.vad = silero.VAD.load(
            min_speech_duration=0.1,    # 100ms minimum speech
            min_silence_duration=0.5,   # 500ms silence threshold
            speech_threshold=0.5,       # Confidence threshold
        )
        
        self.timer = None
        self.is_user_speaking = False
    
    def process_audio(self, audio_frame):
        """Process incoming audio for turn detection"""
        speech_detected = self.vad.detect(audio_frame)
        
        if speech_detected:
            self.is_user_speaking = True
            if self.timer:
                self.timer.cancel()  # Reset timer if speech resumes
                
        else:  # Silence detected
            if self.is_user_speaking and not self.timer:
                # Start end-of-turn timer
                self.timer = self.start_timer(self.silence_duration)
    
    def start_timer(self, duration):
        """Start timer for end-of-turn detection"""
        def fire_end_of_turn():
            self.is_user_speaking = False
            self.on_turn_end()
            
        return Timer(duration, fire_end_of_turn)
```

#### Semantic Turn Detection
```python
class SemanticTurnDetector:
    def __init__(self):
        # LiveKit's trained transformer model
        self.semantic_model = self.load_semantic_model()
        self.conversation_history = []
    
    def analyze_turn_completion(self, current_transcription):
        """Analyze if user has completed their turn"""
        # Combine current transcription with recent history
        context = self.get_conversation_context()
        
        # Model considers:
        # - Grammatical completeness
        # - Conversational patterns
        # - Question vs statement structure
        # - Previous turn patterns
        
        completion_probability = self.semantic_model.predict(
            current_text=current_transcription,
            conversation_context=context
        )
        
        return completion_probability > 0.7  # Threshold for completion
    
    def get_conversation_context(self):
        """Get last 3-4 turns for context"""
        return self.conversation_history[-4:]
```

### Combined Turn Detection Logic

```python
class AdvancedTurnDetector:
    def __init__(self):
        self.vad_detector = TurnDetector()
        self.semantic_detector = SemanticTurnDetector()
        self.override_delay = 200  # 200ms additional wait
    
    def process_turn_signals(self, audio, transcription):
        """Combine VAD and semantic signals"""
        
        # Process audio through VAD
        vad_says_done = self.vad_detector.process_audio(audio)
        
        # Process transcription through semantic model
        semantic_says_done = self.semantic_detector.analyze_turn_completion(
            transcription
        )
        
        if vad_says_done and semantic_says_done:
            # Both agree - end turn immediately
            self.end_turn()
            
        elif vad_says_done and not semantic_says_done:
            # VAD says done but semantic says continue
            # User might be pausing between thoughts
            self.delay_turn_end(self.override_delay)
            
        elif not vad_says_done:
            # User still speaking - continue listening
            self.continue_listening()
    
    def delay_turn_end(self, delay_ms):
        """Delay turn end when semantic model disagrees with VAD"""
        def delayed_end():
            # Re-evaluate after delay
            if not self.vad_detector.is_user_speaking:
                self.end_turn()
        
        Timer(delay_ms / 1000, delayed_end).start()
```

## Interruption Handling

### Interruption Detection

```python
class InterruptionHandler:
    def __init__(self, agent):
        self.agent = agent
        self.vad = silero.VAD.load()
        self.agent_speaking = False
        
    def monitor_for_interruptions(self, audio_stream):
        """Continuously monitor for user interruptions"""
        while self.agent_speaking:
            speech_detected = self.vad.detect(audio_stream.get_frame())
            
            if speech_detected:
                self.handle_interruption()
                break
    
    def handle_interruption(self):
        """Handle user interruption of agent speech"""
        print("User interruption detected - stopping agent")
        
        # 1. Stop all downstream processing
        self.flush_pipeline()
        
        # 2. Reset conversation state
        self.synchronize_context()
        
        # 3. Prepare for new user input
        self.prepare_for_user_input()
    
    def flush_pipeline(self):
        """Stop all pipeline components immediately"""
        # Stop LLM inference if running
        if self.agent.llm.is_generating():
            self.agent.llm.stop_generation()
            
        # Stop TTS generation
        if self.agent.tts.is_generating():
            self.agent.tts.stop_generation()
            
        # Clear audio buffers
        self.agent.clear_audio_buffers()
```

### Pipeline Flushing

When an interruption occurs, the entire pipeline must be flushed:

```
Interruption Detected
        ↓
Stop LLM Generation → Stop TTS Generation → Clear Audio Buffers
        ↓
Synchronize Context → Reset Turn State → Ready for New Input
```

#### Implementation Example
```python
async def flush_pipeline_on_interruption(self):
    """Comprehensive pipeline flush"""
    
    # 1. Immediate stops
    await asyncio.gather(
        self.stop_llm_generation(),
        self.stop_tts_generation(),
        self.clear_audio_pipeline()
    )
    
    # 2. Context synchronization
    last_heard_timestamp = self.get_last_heard_timestamp()
    self.conversation_context.sync_to_timestamp(last_heard_timestamp)
    
    # 3. Reset state
    self.turn_state = "listening"
    self.agent_speaking = False
    
    # 4. Optional acknowledgment
    # await self.say_brief_acknowledgment("Yes?")
```

## Context Management

### Conversation State Tracking

```python
class ConversationContextManager:
    def __init__(self):
        self.messages = []
        self.current_turn = None
        self.last_user_heard_timestamp = None
        
    def add_message(self, role, content, timestamp=None):
        """Add message to conversation history"""
        message = {
            "role": role,  # "user", "assistant", "system"
            "content": content,
            "timestamp": timestamp or time.time(),
            "heard_by_user": role == "user"  # Users hear their own speech
        }
        self.messages.append(message)
    
    def sync_context_on_interruption(self):
        """Synchronize context when user interrupts"""
        # Find last message user actually heard
        cutoff_timestamp = self.last_user_heard_timestamp
        
        # Remove or mark unheard messages
        for message in reversed(self.messages):
            if (message["role"] == "assistant" and 
                message["timestamp"] > cutoff_timestamp):
                message["heard_by_user"] = False
    
    def get_context_for_llm(self):
        """Get conversation context for LLM"""
        # Only include messages the user actually heard
        return [
            msg for msg in self.messages 
            if msg.get("heard_by_user", True)
        ]
```

### Timestamp Synchronization

```python
class TimestampSynchronizer:
    def __init__(self):
        self.audio_timestamps = {}
        self.playback_timestamps = {}
    
    def track_audio_playback(self, audio_chunk, timestamp):
        """Track when audio is played to user"""
        self.audio_timestamps[audio_chunk.id] = timestamp
    
    def get_last_heard_timestamp(self):
        """Calculate what user last heard during interruption"""
        # Account for:
        # - Network latency
        # - Audio buffer delays  
        # - Playback position
        
        current_time = time.time()
        audio_latency = self.estimate_audio_latency()
        
        return current_time - audio_latency
    
    def estimate_audio_latency(self):
        """Estimate total audio latency"""
        return {
            "network_latency": 50,      # ms
            "audio_buffer": 100,        # ms  
            "processing_delay": 30,     # ms
            "total": 180                # ms
        }["total"] / 1000  # Convert to seconds
```

## Advanced Turn Detection Patterns

### Conversation Flow States

```python
class ConversationFlowManager:
    def __init__(self):
        self.state = "idle"
        self.states = {
            "idle": self.handle_idle,
            "listening": self.handle_listening,
            "processing": self.handle_processing,
            "speaking": self.handle_speaking,
            "interrupted": self.handle_interrupted
        }
    
    def transition_state(self, new_state, context=None):
        """Manage state transitions"""
        old_state = self.state
        self.state = new_state
        
        print(f"State transition: {old_state} → {new_state}")
        
        # Execute state-specific logic
        handler = self.states.get(new_state)
        if handler:
            handler(context)
    
    def handle_listening(self, context):
        """User is speaking - agent is listening"""
        self.enable_vad_monitoring()
        self.enable_stt_streaming()
        self.prepare_for_turn_end()
    
    def handle_processing(self, context):
        """Processing user input through LLM"""
        self.disable_interruption_monitoring()  # Brief processing window
        self.start_llm_inference(context["user_input"])
    
    def handle_speaking(self, context):
        """Agent is speaking to user"""
        self.enable_interruption_monitoring()
        self.start_tts_playback(context["response"])
    
    def handle_interrupted(self, context):
        """User interrupted agent"""
        self.flush_pipeline()
        self.sync_conversation_context()
        self.transition_state("listening")
```

### Multi-Turn Conversation Patterns

```python
class MultiTurnHandler:
    def __init__(self):
        self.conversation_patterns = {
            "question_answer": self.handle_qa_pattern,
            "clarification": self.handle_clarification_pattern,
            "interruption_recovery": self.handle_interruption_recovery
        }
    
    def detect_conversation_pattern(self, message_history):
        """Detect current conversation pattern"""
        if self.is_follow_up_question(message_history):
            return "clarification"
        elif self.is_interruption_recovery(message_history):
            return "interruption_recovery"
        else:
            return "question_answer"
    
    def handle_clarification_pattern(self, context):
        """Handle follow-up clarification requests"""
        # Shorter timeout for clarifications
        self.set_turn_timeout(300)  # 300ms instead of 500ms
        
        # Reference previous context more heavily
        self.boost_context_weight(previous_turns=2)
    
    def handle_interruption_recovery(self, context):
        """Handle recovery after interruption"""
        # Ask if user wants to continue where left off
        recovery_prompt = (
            "I was saying... would you like me to continue "
            "or did you have a question?"
        )
        return recovery_prompt
```

## Configuration and Tuning

### Environment-Specific Tuning

```python
class TurnDetectionConfig:
    @staticmethod
    def get_config_for_environment(environment_type):
        """Get optimized config for different environments"""
        
        configs = {
            "quiet_office": {
                "speech_threshold": 0.3,
                "min_silence_duration": 0.4,
                "min_speech_duration": 0.05
            },
            "noisy_environment": {
                "speech_threshold": 0.7,
                "min_silence_duration": 0.6,
                "min_speech_duration": 0.15
            },
            "phone_call": {
                "speech_threshold": 0.5,
                "min_silence_duration": 0.8,  # Longer for phone delays
                "min_speech_duration": 0.1
            },
            "elderly_users": {
                "speech_threshold": 0.4,
                "min_silence_duration": 1.0,   # Longer pauses
                "min_speech_duration": 0.2
            }
        }
        
        return configs.get(environment_type, configs["quiet_office"])
```

### User-Specific Adaptation

```python
class AdaptiveTurnDetection:
    def __init__(self):
        self.user_patterns = {}
        self.adaptation_enabled = True
    
    def learn_user_patterns(self, user_id, conversation_data):
        """Learn user-specific conversation patterns"""
        if user_id not in self.user_patterns:
            self.user_patterns[user_id] = {
                "avg_pause_duration": 0.5,
                "speech_rate": 150,  # words per minute
                "interruption_tendency": 0.3  # 0-1 scale
            }
        
        # Update patterns based on conversation data
        self.update_user_profile(user_id, conversation_data)
    
    def get_adaptive_config(self, user_id):
        """Get user-adapted configuration"""
        if user_id in self.user_patterns:
            patterns = self.user_patterns[user_id]
            
            return {
                "min_silence_duration": patterns["avg_pause_duration"] * 1.2,
                "interruption_sensitivity": 1.0 - patterns["interruption_tendency"],
                "response_urgency": 1.0 / patterns["avg_pause_duration"]
            }
        
        return self.get_default_config()
```

## Testing Turn Detection

### Test Scenarios

```python
class TurnDetectionTester:
    def __init__(self):
        self.test_scenarios = [
            "natural_conversation",
            "rapid_back_and_forth", 
            "long_pauses",
            "frequent_interruptions",
            "overlapping_speech",
            "background_noise"
        ]
    
    async def test_natural_conversation(self, agent):
        """Test normal conversation flow"""
        test_cases = [
            {
                "user_input": "Hello, how are you today?",
                "expected_behavior": "wait_for_complete_question",
                "max_response_time": 800  # ms
            },
            {
                "user_input": "Tell me about... um... artificial intelligence",
                "expected_behavior": "handle_hesitation_pause",
                "pause_duration": 1200  # ms pause
            }
        ]
        
        for case in test_cases:
            result = await self.run_test_case(agent, case)
            self.validate_result(result, case)
    
    async def test_interruption_handling(self, agent):
        """Test interruption scenarios"""
        # Start agent speaking
        agent.start_response("This is a long response that will be interrupted...")
        
        # Simulate user interruption after 500ms
        await asyncio.sleep(0.5)
        agent.simulate_user_speech("Wait, I have a question")
        
        # Verify agent stops and responds appropriately
        assert agent.speaking_stopped
        assert agent.ready_for_user_input
```

## Common Issues and Solutions

### Issue: Premature Turn Ending
**Symptoms**: Agent responds before user finishes speaking
**Solutions**:
- Increase `min_silence_duration`
- Lower `speech_threshold` for better detection
- Enable semantic turn detection
- Tune for specific user patterns

### Issue: Delayed Responses  
**Symptoms**: Long pauses before agent responds
**Solutions**:
- Decrease `min_silence_duration`
- Optimize semantic model inference time
- Implement parallel processing
- Use faster VAD models

### Issue: Missed Interruptions
**Symptoms**: Agent continues speaking when user tries to interrupt
**Solutions**:
- Lower interruption threshold
- Increase VAD sensitivity during agent speech
- Reduce audio buffer sizes
- Implement real-time interruption detection

### Issue: False Interruption Detection
**Symptoms**: Agent stops unnecessarily during normal pauses
**Solutions**:
- Tune VAD threshold for environment
- Implement semantic confirmation
- Add brief delay before stopping
- Filter background noise sources

## Production Deployment Considerations

### Scalability
- Turn detection models must scale across concurrent users
- Consider edge deployment for reduced latency
- Implement efficient resource sharing

### Monitoring
- Track turn detection accuracy metrics
- Monitor false positive/negative rates
- Alert on performance degradation
- A/B test different configurations

### User Experience
- Provide fallback mechanisms for detection failures
- Implement graceful degradation
- Offer manual controls when needed
- Test with diverse user populations

## Next Steps

1. **Implementation Tutorial**: Complete setup in [Implementation Tutorial](implementation-tutorial.md)
2. **Performance Optimization**: Advanced optimization in [Performance Guide](performance-optimization.md)
3. **Framework Reference**: LiveKit specifics in [LiveKit Reference](livekit-reference.md)
4. **Real-World Applications**: Domain-specific usage in [Applications Guide](applications-guide.md)

## Key Takeaways

- **Dual-Signal Approach**: Combine VAD and semantic processing for robust turn detection
- **Interruption is Critical**: Users expect natural interruption capability
- **Context Preservation**: Maintain conversation state through interruptions
- **Environment Matters**: Tune parameters for specific acoustic conditions
- **User Adaptation**: Learn and adapt to individual conversation patterns
- **Testing is Essential**: Comprehensive testing across diverse scenarios ensures reliability