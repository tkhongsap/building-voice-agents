# Error Handling Guide

This comprehensive guide covers error handling patterns, troubleshooting steps, and best practices for building robust voice agents with the LiveKit Voice Agents Platform.

## Table of Contents

1. [Error Types and Categories](#error-types-and-categories)
2. [Common Error Scenarios](#common-error-scenarios)
3. [Error Handling Patterns](#error-handling-patterns)
4. [Debugging Strategies](#debugging-strategies)
5. [Monitoring and Alerting](#monitoring-and-alerting)
6. [Recovery Mechanisms](#recovery-mechanisms)
7. [Best Practices](#best-practices)

---

## Error Types and Categories

### 1. **Network and Connectivity Errors**

#### WebRTC Connection Issues
```python
class WebRTCConnectionError(Exception):
    """WebRTC connection failed or lost."""
    def __init__(self, message: str, error_code: str = None):
        self.error_code = error_code
        super().__init__(message)

# Example handling
try:
    await agent.connect_to_room(room_url, token)
except WebRTCConnectionError as e:
    logger.error(f"WebRTC connection failed: {e}, code: {e.error_code}")
    await agent.retry_connection(max_attempts=3)
```

#### Network Timeout Errors
```python
import asyncio
from typing import Optional

async def robust_api_call(api_func, timeout: float = 30.0, retries: int = 3) -> Optional[Any]:
    """Make API call with timeout and retry logic."""
    for attempt in range(retries):
        try:
            return await asyncio.wait_for(api_func(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"API call timeout, attempt {attempt + 1}/{retries}")
            if attempt == retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
        except Exception as e:
            logger.error(f"API call failed: {e}")
            raise
```

### 2. **AI Provider Errors**

#### STT (Speech-to-Text) Errors
```python
class STTError(Exception):
    """Speech-to-text processing error."""
    pass

class STTTimeoutError(STTError):
    """STT processing timeout."""
    pass

class STTQuotaExceededError(STTError):
    """STT quota or rate limit exceeded."""
    pass

# Example STT error handler
async def handle_stt_error(stt_provider, audio_data: bytes, error: Exception):
    """Handle STT errors with fallback logic."""
    if isinstance(error, STTTimeoutError):
        logger.warning("STT timeout, retrying with shorter audio chunk")
        # Split audio into smaller chunks
        return await process_audio_chunks(stt_provider, audio_data)
    
    elif isinstance(error, STTQuotaExceededError):
        logger.error("STT quota exceeded, switching to fallback provider")
        fallback_provider = get_fallback_stt_provider()
        return await fallback_provider.transcribe(audio_data)
    
    else:
        logger.error(f"Unexpected STT error: {error}")
        return None  # Return None to signal failure
```

#### LLM (Language Model) Errors
```python
class LLMError(Exception):
    """Language model processing error."""
    pass

class LLMContextLengthError(LLMError):
    """Context length exceeded."""
    pass

class LLMRateLimitError(LLMError):
    """Rate limit exceeded."""
    pass

# Example LLM error handler
async def robust_llm_call(llm_provider, prompt: str, **kwargs):
    """Make LLM call with error handling."""
    try:
        return await llm_provider.generate(prompt, **kwargs)
    
    except LLMContextLengthError:
        # Truncate context and retry
        logger.warning("Context too long, truncating conversation history")
        truncated_prompt = truncate_context(prompt, max_tokens=4000)
        return await llm_provider.generate(truncated_prompt, **kwargs)
    
    except LLMRateLimitError:
        # Wait and retry with exponential backoff
        logger.warning("Rate limit hit, waiting before retry")
        await asyncio.sleep(random.uniform(1, 5))
        return await llm_provider.generate(prompt, **kwargs)
    
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        # Return a fallback response
        return "I'm sorry, I'm having trouble processing that right now. Could you try again?"
```

#### TTS (Text-to-Speech) Errors
```python
class TTSError(Exception):
    """Text-to-speech processing error."""
    pass

class TTSInvalidTextError(TTSError):
    """Invalid text for TTS processing."""
    pass

# Example TTS error handler
async def robust_tts_call(tts_provider, text: str):
    """Make TTS call with error handling."""
    try:
        # Sanitize text before TTS
        sanitized_text = sanitize_tts_text(text)
        return await tts_provider.synthesize(sanitized_text)
    
    except TTSInvalidTextError:
        # Clean text and retry
        logger.warning("Invalid TTS text, cleaning and retrying")
        cleaned_text = clean_text_for_tts(text)
        return await tts_provider.synthesize(cleaned_text)
    
    except Exception as e:
        logger.error(f"TTS call failed: {e}")
        # Return silent audio or error message
        return generate_error_audio("Sorry, I couldn't speak that response.")

def sanitize_tts_text(text: str) -> str:
    """Sanitize text for TTS processing."""
    # Remove or replace problematic characters
    import re
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove control characters
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    
    # Limit length
    if len(text) > 1000:
        text = text[:997] + "..."
    
    return text
```

### 3. **Audio Processing Errors**

#### VAD (Voice Activity Detection) Errors
```python
class VADError(Exception):
    """Voice activity detection error."""
    pass

# Example VAD error handler
async def robust_vad_processing(vad_provider, audio_stream):
    """Process audio with VAD error handling."""
    try:
        return await vad_provider.detect_speech(audio_stream)
    except VADError as e:
        logger.warning(f"VAD error: {e}, using fallback detection")
        # Use simple energy-based detection as fallback
        return simple_energy_vad(audio_stream)
    except Exception as e:
        logger.error(f"Unexpected VAD error: {e}")
        # Assume speech is present to avoid missing user input
        return True

def simple_energy_vad(audio_data: bytes, threshold: float = 0.01) -> bool:
    """Simple energy-based voice activity detection fallback."""
    import numpy as np
    
    try:
        # Convert audio to numpy array
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
        # Calculate RMS energy
        rms = np.sqrt(np.mean(audio_array.astype(float) ** 2))
        
        # Normalize and compare to threshold
        normalized_rms = rms / 32767.0  # For 16-bit audio
        return normalized_rms > threshold
    
    except Exception:
        # If all else fails, assume speech
        return True
```

### 4. **Configuration and Setup Errors**

```python
class ConfigurationError(Exception):
    """Configuration or setup error."""
    pass

class MissingAPIKeyError(ConfigurationError):
    """Required API key is missing."""
    pass

class InvalidConfigurationError(ConfigurationError):
    """Configuration values are invalid."""
    pass

def validate_configuration(config: Dict[str, Any]) -> None:
    """Validate agent configuration."""
    required_keys = ["stt_provider", "llm_provider", "tts_provider"]
    
    for key in required_keys:
        if key not in config:
            raise InvalidConfigurationError(f"Missing required config key: {key}")
    
    # Validate API keys based on providers
    if config["stt_provider"] == "openai" and not config.get("openai_api_key"):
        raise MissingAPIKeyError("OpenAI API key required for OpenAI STT")
    
    if config["llm_provider"] == "anthropic" and not config.get("anthropic_api_key"):
        raise MissingAPIKeyError("Anthropic API key required for Claude")
    
    # Validate numeric values
    if "temperature" in config:
        temp = config["temperature"]
        if not 0.0 <= temp <= 2.0:
            raise InvalidConfigurationError(f"Temperature must be between 0.0 and 2.0, got {temp}")
```

---

## Common Error Scenarios

### Scenario 1: Microphone Permission Denied

**Problem:** User denies microphone access or browser blocks audio input.

**Symptoms:**
- No audio input detected
- WebRTC connection fails
- "Permission denied" errors in browser console

**Solution:**
```python
async def handle_microphone_permission():
    """Handle microphone permission issues."""
    try:
        # Check if microphone is available
        devices = await get_audio_devices()
        if not devices:
            raise AudioDeviceError("No audio input devices found")
        
        # Request permission
        await request_microphone_permission()
        
    except PermissionDeniedError:
        # Provide user guidance
        show_permission_help_dialog()
        raise UserNotificationError(
            "Microphone access is required. Please enable microphone permission and refresh the page."
        )
    
    except AudioDeviceError as e:
        logger.error(f"Audio device error: {e}")
        # Offer text-based alternative
        enable_text_mode()
        raise UserNotificationError(
            "No microphone detected. You can use text input instead."
        )

def show_permission_help_dialog():
    """Show user-friendly permission help."""
    help_text = """
    To enable microphone access:
    1. Click the microphone icon in your browser's address bar
    2. Select "Always allow" for this site
    3. Refresh the page
    
    Or check your browser settings:
    - Chrome: Settings > Privacy & Security > Site Settings > Microphone
    - Firefox: Settings > Privacy & Security > Permissions > Microphone
    """
    print(help_text)  # In a real app, show in UI
```

### Scenario 2: API Rate Limiting

**Problem:** AI provider rate limits are exceeded during high usage.

**Symptoms:**
- HTTP 429 errors
- Delayed responses
- Service degradation

**Solution:**
```python
import asyncio
from typing import Optional
import time

class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, rate: float, burst: int):
        self.rate = rate  # tokens per second
        self.burst = burst  # maximum tokens
        self.tokens = burst
        self.last_update = time.time()
        self.lock = asyncio.Lock()
    
    async def acquire(self) -> bool:
        """Acquire a token, returns True if available."""
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
            self.last_update = now
            
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False

class RateLimitedProvider:
    """Provider wrapper with rate limiting."""
    
    def __init__(self, provider, rate_limiter: RateLimiter):
        self.provider = provider
        self.rate_limiter = rate_limiter
        self.retry_queue = asyncio.Queue()
    
    async def call_with_rate_limiting(self, method_name: str, *args, **kwargs):
        """Call provider method with rate limiting."""
        max_wait = 30  # Maximum wait time in seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            if await self.rate_limiter.acquire():
                try:
                    method = getattr(self.provider, method_name)
                    return await method(*args, **kwargs)
                except RateLimitError:
                    # Even with our limiting, we hit the API limit
                    logger.warning("API rate limit hit despite local limiting")
                    await asyncio.sleep(1)
                    continue
            else:
                # Wait for tokens to become available
                await asyncio.sleep(0.1)
        
        raise TimeoutError("Rate limiting timeout exceeded")

# Usage example
rate_limiter = RateLimiter(rate=10.0, burst=20)  # 10 requests/sec, burst of 20
limited_provider = RateLimitedProvider(openai_provider, rate_limiter)
```

### Scenario 3: Poor Network Conditions

**Problem:** Unstable internet connection causes frequent dropouts.

**Symptoms:**
- Audio cutting out
- Delayed responses
- Connection timeouts

**Solution:**
```python
class NetworkQualityMonitor:
    """Monitor network quality and adapt behavior."""
    
    def __init__(self):
        self.latency_samples = []
        self.packet_loss_rate = 0.0
        self.quality_score = 1.0  # 0.0 = poor, 1.0 = excellent
    
    def update_metrics(self, latency: float, packet_loss: float):
        """Update network quality metrics."""
        self.latency_samples.append(latency)
        if len(self.latency_samples) > 10:
            self.latency_samples.pop(0)
        
        self.packet_loss_rate = packet_loss
        self._calculate_quality_score()
    
    def _calculate_quality_score(self):
        """Calculate overall quality score."""
        if not self.latency_samples:
            return
        
        avg_latency = sum(self.latency_samples) / len(self.latency_samples)
        
        # Score based on latency (lower is better)
        latency_score = max(0, 1 - (avg_latency - 100) / 500)  # Good if < 100ms
        
        # Score based on packet loss (lower is better)
        loss_score = max(0, 1 - self.packet_loss_rate * 10)
        
        self.quality_score = (latency_score + loss_score) / 2
    
    def get_adaptive_settings(self) -> Dict[str, Any]:
        """Get adaptive settings based on network quality."""
        if self.quality_score > 0.8:
            return {
                "audio_quality": "high",
                "chunk_size": 1024,
                "timeout": 30,
                "retry_attempts": 3
            }
        elif self.quality_score > 0.5:
            return {
                "audio_quality": "medium", 
                "chunk_size": 512,
                "timeout": 45,
                "retry_attempts": 5
            }
        else:
            return {
                "audio_quality": "low",
                "chunk_size": 256,
                "timeout": 60,
                "retry_attempts": 8
            }

# Network-adaptive agent behavior
async def adaptive_voice_processing(agent, network_monitor: NetworkQualityMonitor):
    """Adapt voice processing based on network conditions."""
    settings = network_monitor.get_adaptive_settings()
    
    # Adjust audio chunk size
    agent.set_audio_chunk_size(settings["chunk_size"])
    
    # Adjust timeouts
    agent.set_timeout(settings["timeout"])
    
    # Use appropriate audio quality
    if settings["audio_quality"] == "low":
        # Use more aggressive compression
        agent.enable_audio_compression(level=8)
    else:
        agent.enable_audio_compression(level=3)
```

---

## Error Handling Patterns

### 1. **Circuit Breaker Pattern**

Prevent cascading failures by temporarily stopping calls to failing services.

```python
import time
from enum import Enum
from typing import Callable, Any

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered

class CircuitBreaker:
    """Circuit breaker for preventing cascading failures."""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                raise CircuitBreakerOpenError("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        
        except Exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit."""
        if self.last_failure_time is None:
            return False
        return time.time() - self.last_failure_time >= self.timeout
    
    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN

class CircuitBreakerOpenError(Exception):
    """Circuit breaker is open, blocking requests."""
    pass

# Usage example
stt_circuit_breaker = CircuitBreaker(failure_threshold=3, timeout=30.0)

async def protected_stt_call(stt_provider, audio_data):
    """STT call protected by circuit breaker."""
    try:
        return await stt_circuit_breaker.call(stt_provider.transcribe, audio_data)
    except CircuitBreakerOpenError:
        logger.warning("STT circuit breaker is open, using cached response")
        return get_cached_transcription(audio_data)
```

### 2. **Retry with Exponential Backoff**

Retry failed operations with increasing delays to avoid overwhelming failing services.

```python
import asyncio
import random
from typing import Callable, Any, Optional

async def retry_with_backoff(
    func: Callable,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_factor: float = 2.0,
    jitter: bool = True,
    exceptions: tuple = (Exception,)
) -> Any:
    """Retry function with exponential backoff."""
    
    for attempt in range(max_attempts):
        try:
            return await func()
        
        except exceptions as e:
            if attempt == max_attempts - 1:
                # Last attempt, re-raise the exception
                raise e
            
            # Calculate delay
            delay = min(base_delay * (exponential_factor ** attempt), max_delay)
            
            # Add jitter to prevent thundering herd
            if jitter:
                delay *= (0.5 + random.random() * 0.5)
            
            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s")
            await asyncio.sleep(delay)

# Usage examples
async def reliable_llm_call(llm_provider, prompt: str):
    """LLM call with retry logic."""
    return await retry_with_backoff(
        lambda: llm_provider.generate(prompt),
        max_attempts=3,
        exceptions=(LLMRateLimitError, asyncio.TimeoutError)
    )

async def reliable_api_call(api_func):
    """Generic API call with retry."""
    return await retry_with_backoff(
        api_func,
        max_attempts=5,
        base_delay=0.5,
        exponential_factor=1.5,
        exceptions=(aiohttp.ClientError, asyncio.TimeoutError)
    )
```

### 3. **Graceful Degradation**

Provide reduced functionality when full features are unavailable.

```python
class DegradationLevel(Enum):
    FULL = "full"
    REDUCED = "reduced"
    MINIMAL = "minimal"
    EMERGENCY = "emergency"

class GracefulDegradationManager:
    """Manage graceful degradation of voice agent features."""
    
    def __init__(self):
        self.current_level = DegradationLevel.FULL
        self.feature_status = {
            "stt": True,
            "llm": True,
            "tts": True,
            "vad": True,
            "function_calling": True
        }
    
    def update_feature_status(self, feature: str, is_working: bool):
        """Update feature availability."""
        self.feature_status[feature] = is_working
        self._recalculate_degradation_level()
    
    def _recalculate_degradation_level(self):
        """Recalculate current degradation level."""
        working_features = sum(self.feature_status.values())
        total_features = len(self.feature_status)
        
        if working_features == total_features:
            self.current_level = DegradationLevel.FULL
        elif working_features >= total_features * 0.8:
            self.current_level = DegradationLevel.REDUCED
        elif working_features >= total_features * 0.5:
            self.current_level = DegradationLevel.MINIMAL
        else:
            self.current_level = DegradationLevel.EMERGENCY
    
    def get_available_features(self) -> List[str]:
        """Get list of currently available features."""
        return [feature for feature, status in self.feature_status.items() if status]
    
    def can_provide_voice_interaction(self) -> bool:
        """Check if voice interaction is possible."""
        return self.feature_status["stt"] and self.feature_status["tts"]
    
    def get_fallback_response_method(self) -> str:
        """Get appropriate response method based on available features."""
        if self.feature_status["tts"]:
            return "voice"
        elif self.feature_status["llm"]:
            return "text"
        else:
            return "static"

# Usage in voice agent
async def handle_user_input_with_degradation(agent, user_input, degradation_manager):
    """Handle user input with graceful degradation."""
    
    if degradation_manager.current_level == DegradationLevel.EMERGENCY:
        return "I'm experiencing technical difficulties. Please try again later."
    
    response_text = None
    
    # Try full LLM processing
    if degradation_manager.feature_status["llm"]:
        try:
            response_text = await agent.llm.generate(user_input)
        except Exception as e:
            logger.error(f"LLM failed: {e}")
            degradation_manager.update_feature_status("llm", False)
    
    # Fallback to predefined responses
    if not response_text:
        response_text = get_fallback_response(user_input)
    
    # Try TTS
    if degradation_manager.feature_status["tts"]:
        try:
            audio_response = await agent.tts.synthesize(response_text)
            return {"text": response_text, "audio": audio_response}
        except Exception as e:
            logger.error(f"TTS failed: {e}")
            degradation_manager.update_feature_status("tts", False)
    
    # Text-only fallback
    return {"text": response_text}

def get_fallback_response(user_input: str) -> str:
    """Get predefined fallback response."""
    fallback_responses = {
        "hello": "Hello! I'm experiencing some technical issues but I'm here to help.",
        "help": "I can provide basic assistance. Please describe what you need help with.",
        "goodbye": "Thank you for your patience. Goodbye!",
        "default": "I'm sorry, I'm having technical difficulties right now. Please try again later."
    }
    
    user_input_lower = user_input.lower()
    for keyword, response in fallback_responses.items():
        if keyword in user_input_lower:
            return response
    
    return fallback_responses["default"]
```

---

## Debugging Strategies

### 1. **Comprehensive Logging**

```python
import logging
import json
from datetime import datetime
from typing import Any, Dict

class VoiceAgentLogger:
    """Specialized logger for voice agent debugging."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(detailed_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler for persistent logs
        file_handler = logging.FileHandler('voice_agent.log')
        file_handler.setFormatter(detailed_formatter)
        self.logger.addHandler(file_handler)
    
    def log_conversation_event(self, event_type: str, data: Dict[str, Any]):
        """Log conversation events with structured data."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "data": data
        }
        self.logger.info(f"CONVERSATION_EVENT: {json.dumps(log_entry)}")
    
    def log_performance_metric(self, metric_name: str, value: float, context: Dict[str, Any] = None):
        """Log performance metrics."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "metric": metric_name,
            "value": value,
            "context": context or {}
        }
        self.logger.info(f"PERFORMANCE_METRIC: {json.dumps(log_entry)}")
    
    def log_error_with_context(self, error: Exception, context: Dict[str, Any] = None):
        """Log errors with full context."""
        import traceback
        
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "context": context or {}
        }
        self.logger.error(f"ERROR_WITH_CONTEXT: {json.dumps(error_entry)}")

# Usage throughout the application
logger = VoiceAgentLogger("voice_agent")

# Log conversation events
logger.log_conversation_event("user_input", {
    "text": user_input,
    "confidence": 0.95,
    "language": "en",
    "duration_ms": 1500
})

# Log performance metrics
logger.log_performance_metric("stt_latency", 250.5, {
    "provider": "openai",
    "audio_length_ms": 2000
})

# Log errors with context
try:
    result = await stt_provider.transcribe(audio_data)
except Exception as e:
    logger.log_error_with_context(e, {
        "provider": "openai",
        "audio_size_bytes": len(audio_data),
        "attempt_number": 2
    })
```

### 2. **Real-time Debugging Tools**

```python
import asyncio
from typing import Dict, List, Callable

class DebugEventEmitter:
    """Emit debug events for real-time monitoring."""
    
    def __init__(self):
        self.listeners: Dict[str, List[Callable]] = {}
        self.event_history: List[Dict] = []
    
    def on(self, event_type: str, callback: Callable):
        """Register event listener."""
        if event_type not in self.listeners:
            self.listeners[event_type] = []
        self.listeners[event_type].append(callback)
    
    def emit(self, event_type: str, data: Dict):
        """Emit debug event."""
        event = {
            "type": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        
        # Store in history
        self.event_history.append(event)
        if len(self.event_history) > 1000:
            self.event_history.pop(0)
        
        # Notify listeners
        for callback in self.listeners.get(event_type, []):
            try:
                asyncio.create_task(callback(event))
            except Exception as e:
                logger.error(f"Debug event callback failed: {e}")
    
    def get_recent_events(self, event_type: str = None, limit: int = 50) -> List[Dict]:
        """Get recent debug events."""
        events = self.event_history
        
        if event_type:
            events = [e for e in events if e["type"] == event_type]
        
        return events[-limit:]

# Debug-enabled voice agent
class DebugVoiceAgent:
    """Voice agent with debugging capabilities."""
    
    def __init__(self):
        self.debug_emitter = DebugEventEmitter()
        self.performance_tracker = {}
        
        # Set up debug listeners
        self.debug_emitter.on("performance_warning", self._handle_performance_warning)
        self.debug_emitter.on("error", self._handle_error_event)
    
    async def _handle_performance_warning(self, event):
        """Handle performance warnings."""
        data = event["data"]
        logger.warning(f"Performance warning: {data['component']} took {data['duration_ms']}ms")
        
        # Take corrective action
        if data["component"] == "stt" and data["duration_ms"] > 5000:
            logger.info("Switching to faster STT model")
            await self.switch_stt_provider("faster_model")
    
    async def _handle_error_event(self, event):
        """Handle error events."""
        data = event["data"]
        logger.error(f"Debug error event: {data}")
        
        # Implement automatic recovery
        if "connection" in data.get("error_type", "").lower():
            logger.info("Attempting connection recovery")
            await self.reconnect()
    
    async def process_with_debug(self, user_input: str):
        """Process user input with debug tracking."""
        start_time = time.time()
        
        try:
            # Emit input event
            self.debug_emitter.emit("user_input", {
                "text": user_input,
                "length": len(user_input)
            })
            
            # Process with timing
            stt_start = time.time()
            transcription = await self.stt.transcribe(user_input)
            stt_duration = (time.time() - stt_start) * 1000
            
            self.debug_emitter.emit("stt_complete", {
                "duration_ms": stt_duration,
                "text": transcription
            })
            
            # Check for performance issues
            if stt_duration > 3000:  # 3 second threshold
                self.debug_emitter.emit("performance_warning", {
                    "component": "stt",
                    "duration_ms": stt_duration,
                    "threshold_ms": 3000
                })
            
            # Continue with LLM processing...
            llm_start = time.time()
            response = await self.llm.generate(transcription)
            llm_duration = (time.time() - llm_start) * 1000
            
            self.debug_emitter.emit("llm_complete", {
                "duration_ms": llm_duration,
                "input_tokens": len(transcription.split()),
                "output_tokens": len(response.split())
            })
            
            total_duration = (time.time() - start_time) * 1000
            self.debug_emitter.emit("request_complete", {
                "total_duration_ms": total_duration,
                "stt_duration_ms": stt_duration,
                "llm_duration_ms": llm_duration
            })
            
            return response
            
        except Exception as e:
            self.debug_emitter.emit("error", {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "component": "processing"
            })
            raise
```

### 3. **Audio Debugging Tools**

```python
import numpy as np
import wave
from typing import Optional

class AudioDebugger:
    """Tools for debugging audio processing issues."""
    
    @staticmethod
    def save_audio_for_debug(audio_data: bytes, filename: str, sample_rate: int = 16000):
        """Save audio data to file for inspection."""
        try:
            with wave.open(filename, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data)
            
            logger.debug(f"Audio saved for debugging: {filename}")
        except Exception as e:
            logger.error(f"Failed to save debug audio: {e}")
    
    @staticmethod
    def analyze_audio_quality(audio_data: bytes) -> Dict[str, Any]:
        """Analyze audio quality metrics."""
        try:
            # Convert to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Calculate metrics
            rms = np.sqrt(np.mean(audio_array.astype(float) ** 2))
            peak = np.max(np.abs(audio_array))
            snr = 20 * np.log10(rms / (np.std(audio_array) + 1e-10))
            
            # Detect clipping
            clipping_samples = np.sum(np.abs(audio_array) >= 32767)
            clipping_percentage = (clipping_samples / len(audio_array)) * 100
            
            # Detect silence
            silence_threshold = 100  # Adjust based on your needs
            silence_samples = np.sum(np.abs(audio_array) < silence_threshold)
            silence_percentage = (silence_samples / len(audio_array)) * 100
            
            analysis = {
                "rms_level": float(rms),
                "peak_level": float(peak),
                "snr_db": float(snr),
                "clipping_percentage": float(clipping_percentage),
                "silence_percentage": float(silence_percentage),
                "duration_ms": len(audio_array) * 1000 / 16000,  # Assuming 16kHz
                "quality_score": AudioDebugger._calculate_quality_score(
                    snr, clipping_percentage, silence_percentage
                )
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Audio analysis failed: {e}")
            return {"error": str(e)}
    
    @staticmethod
    def _calculate_quality_score(snr: float, clipping: float, silence: float) -> float:
        """Calculate overall audio quality score (0-1)."""
        # SNR score (good SNR > 20dB)
        snr_score = min(1.0, max(0.0, (snr - 10) / 20))
        
        # Clipping penalty (no clipping = 1.0)
        clipping_score = max(0.0, 1.0 - clipping / 10)
        
        # Silence penalty (some silence ok, too much is bad)
        if silence < 10:
            silence_score = 1.0
        elif silence < 50:
            silence_score = 1.0 - (silence - 10) / 40 * 0.5
        else:
            silence_score = 0.5 - (silence - 50) / 50 * 0.5
        
        return (snr_score + clipping_score + silence_score) / 3

# Usage in voice agent
async def debug_audio_processing(agent, audio_data: bytes):
    """Process audio with debugging."""
    # Analyze audio quality
    quality_analysis = AudioDebugger.analyze_audio_quality(audio_data)
    
    logger.debug(f"Audio quality analysis: {quality_analysis}")
    
    # Save problematic audio for inspection
    if quality_analysis.get("quality_score", 1.0) < 0.5:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"debug_audio_poor_quality_{timestamp}.wav"
        AudioDebugger.save_audio_for_debug(audio_data, filename)
        
        logger.warning(f"Poor audio quality detected, saved to {filename}")
    
    # Warn about specific issues
    if quality_analysis.get("clipping_percentage", 0) > 1:
        logger.warning(f"Audio clipping detected: {quality_analysis['clipping_percentage']:.1f}%")
    
    if quality_analysis.get("silence_percentage", 0) > 80:
        logger.warning(f"Mostly silence detected: {quality_analysis['silence_percentage']:.1f}%")
    
    return quality_analysis
```

---

## Best Practices

### 1. **Error Prevention**

```python
# Input validation
def validate_audio_input(audio_data: bytes) -> bool:
    """Validate audio input before processing."""
    if not audio_data:
        raise ValueError("Empty audio data")
    
    if len(audio_data) < 1000:  # Minimum audio length
        raise ValueError("Audio too short for processing")
    
    if len(audio_data) > 10_000_000:  # Maximum audio length (10MB)
        raise ValueError("Audio too large for processing")
    
    return True

# Configuration validation
def validate_environment():
    """Validate environment and configuration on startup."""
    required_env_vars = [
        "OPENAI_API_KEY",
        "LIVEKIT_URL", 
        "LIVEKIT_API_KEY"
    ]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        raise ConfigurationError(f"Missing environment variables: {missing_vars}")
    
    # Test API connectivity
    try:
        # Make test calls to verify API keys work
        pass
    except Exception as e:
        raise ConfigurationError(f"API connectivity test failed: {e}")
```

### 2. **Resource Management**

```python
from contextlib import asynccontextmanager
import weakref

class ResourceManager:
    """Manage voice agent resources and cleanup."""
    
    def __init__(self):
        self.active_connections = weakref.WeakSet()
        self.cleanup_tasks = []
    
    @asynccontextmanager
    async def managed_connection(self, connection):
        """Context manager for connection lifecycle."""
        self.active_connections.add(connection)
        try:
            yield connection
        finally:
            await self._cleanup_connection(connection)
    
    async def _cleanup_connection(self, connection):
        """Clean up connection resources."""
        try:
            if hasattr(connection, 'close'):
                await connection.close()
            if hasattr(connection, 'disconnect'):
                await connection.disconnect()
        except Exception as e:
            logger.error(f"Connection cleanup failed: {e}")
    
    async def cleanup_all(self):
        """Clean up all managed resources."""
        # Clean up active connections
        for connection in list(self.active_connections):
            await self._cleanup_connection(connection)
        
        # Run cleanup tasks
        for task in self.cleanup_tasks:
            try:
                await task()
            except Exception as e:
                logger.error(f"Cleanup task failed: {e}")
    
    def register_cleanup_task(self, coro):
        """Register cleanup coroutine."""
        self.cleanup_tasks.append(coro)

# Usage
resource_manager = ResourceManager()

async def main():
    try:
        # Main application logic
        async with resource_manager.managed_connection(voice_agent) as agent:
            await agent.start_conversation()
    finally:
        await resource_manager.cleanup_all()
```

### 3. **Monitoring and Alerting**

```python
class HealthChecker:
    """Monitor voice agent health and send alerts."""
    
    def __init__(self, alert_callback: Callable = None):
        self.alert_callback = alert_callback
        self.health_metrics = {}
        self.alert_thresholds = {
            "error_rate": 0.05,  # 5%
            "avg_latency_ms": 3000,  # 3 seconds
            "memory_usage_mb": 1000,  # 1GB
            "cpu_usage_percent": 80   # 80%
        }
    
    async def check_health(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "components": {},
            "alerts": []
        }
        
        # Check individual components
        components = ["stt", "llm", "tts", "webrtc"]
        for component in components:
            status = await self._check_component_health(component)
            health_status["components"][component] = status
            
            if not status["healthy"]:
                health_status["overall_status"] = "unhealthy"
                health_status["alerts"].append({
                    "severity": "error",
                    "component": component,
                    "message": status["error"]
                })
        
        # Check performance metrics
        performance_alerts = await self._check_performance_metrics()
        health_status["alerts"].extend(performance_alerts)
        
        if performance_alerts:
            health_status["overall_status"] = "degraded"
        
        # Send alerts if callback is provided
        if self.alert_callback and health_status["alerts"]:
            await self.alert_callback(health_status)
        
        return health_status
    
    async def _check_component_health(self, component: str) -> Dict[str, Any]:
        """Check health of individual component."""
        try:
            # Simulate component health check
            # In real implementation, make actual test calls
            await asyncio.sleep(0.1)
            
            return {
                "healthy": True,
                "response_time_ms": 100,
                "last_check": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "last_check": datetime.now().isoformat()
            }
    
    async def _check_performance_metrics(self) -> List[Dict[str, Any]]:
        """Check performance metrics against thresholds."""
        alerts = []
        
        # Check error rate
        error_rate = self.health_metrics.get("error_rate", 0)
        if error_rate > self.alert_thresholds["error_rate"]:
            alerts.append({
                "severity": "warning",
                "metric": "error_rate",
                "value": error_rate,
                "threshold": self.alert_thresholds["error_rate"],
                "message": f"High error rate: {error_rate:.2%}"
            })
        
        # Check latency
        avg_latency = self.health_metrics.get("avg_latency_ms", 0)
        if avg_latency > self.alert_thresholds["avg_latency_ms"]:
            alerts.append({
                "severity": "warning",
                "metric": "avg_latency_ms", 
                "value": avg_latency,
                "threshold": self.alert_thresholds["avg_latency_ms"],
                "message": f"High latency: {avg_latency:.0f}ms"
            })
        
        return alerts

# Alert notification
async def send_alert_notification(health_status: Dict[str, Any]):
    """Send alert notifications."""
    for alert in health_status["alerts"]:
        if alert["severity"] == "error":
            # Send to on-call engineer
            await send_pager_alert(alert)
        elif alert["severity"] == "warning":
            # Send to monitoring channel
            await send_slack_alert(alert)

async def send_pager_alert(alert: Dict[str, Any]):
    """Send high-priority alert."""
    logger.critical(f"CRITICAL ALERT: {alert['message']}")
    # Integrate with PagerDuty, OpsGenie, etc.

async def send_slack_alert(alert: Dict[str, Any]):
    """Send team notification."""
    logger.warning(f"WARNING ALERT: {alert['message']}")
    # Integrate with Slack, Teams, etc.
```

This comprehensive error handling guide provides the foundation for building robust, production-ready voice agents that can handle failures gracefully and recover automatically.