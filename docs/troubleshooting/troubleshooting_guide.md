# Troubleshooting Guide

This guide helps you diagnose and resolve common issues when building and deploying voice agents with the LiveKit Voice Agents Platform.

## Table of Contents

1. [Quick Diagnostic Steps](#quick-diagnostic-steps)
2. [Common Issues and Solutions](#common-issues-and-solutions)
3. [Audio Issues](#audio-issues)
4. [API and Provider Issues](#api-and-provider-issues)
5. [Performance Issues](#performance-issues)
6. [Deployment Issues](#deployment-issues)
7. [Development Environment Issues](#development-environment-issues)
8. [Debugging Tools](#debugging-tools)

---

## Quick Diagnostic Steps

Before diving into specific issues, run through these quick diagnostic steps:

### 1. **System Health Check**

```bash
# Check Python version (requires 3.8+)
python --version

# Check installed packages
pip list | grep -E "(livekit|openai|anthropic|azure|elevenlabs)"

# Check environment variables
echo $OPENAI_API_KEY | cut -c1-10  # Should show first 10 chars
echo $LIVEKIT_URL
echo $LIVEKIT_API_KEY | cut -c1-10

# Test network connectivity
ping api.openai.com
ping api.anthropic.com
ping api.elevenlabs.io
```

### 2. **Basic Functionality Test**

```python
# test_basic_functionality.py
import asyncio
import os

async def test_basic_setup():
    """Test basic voice agent setup."""
    try:
        from src.sdk.python_sdk import VoiceAgentSDK
        
        # Create SDK instance
        sdk = VoiceAgentSDK()
        print("✅ SDK initialization successful")
        
        # Test basic configuration
        config = {
            "stt_provider": "openai",
            "llm_provider": "openai", 
            "tts_provider": "openai",
            "openai_api_key": os.getenv("OPENAI_API_KEY")
        }
        
        if not config["openai_api_key"]:
            print("❌ OPENAI_API_KEY not set")
            return False
        
        print("✅ Basic configuration valid")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Setup error: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_basic_setup())
```

---

## Common Issues and Solutions

### Issue 1: "ModuleNotFoundError: No module named 'src'"

**Symptoms:**
```bash
ModuleNotFoundError: No module named 'src'
```

**Cause:** Running Python from wrong directory or incorrect Python path.

**Solutions:**

1. **Run from project root:**
   ```bash
   cd /path/to/building-voice-agents
   python examples/basic_voice_bot.py
   ```

2. **Add to Python path:**
   ```bash
   export PYTHONPATH=$PYTHONPATH:/path/to/building-voice-agents
   python examples/basic_voice_bot.py
   ```

3. **Use absolute imports:**
   ```python
   # In your scripts, use absolute imports
   import sys
   sys.path.append('/path/to/building-voice-agents')
   from src.sdk.python_sdk import VoiceAgentSDK
   ```

### Issue 2: "API Key Authentication Failed"

**Symptoms:**
```bash
AuthenticationError: Incorrect API key provided
HTTP 401: Unauthorized
```

**Diagnostic Steps:**

1. **Check API key format:**
   ```bash
   echo $OPENAI_API_KEY | wc -c  # Should be ~51 characters for OpenAI
   echo $ANTHROPIC_API_KEY | head -c 10  # Should start with 'sk-ant-'
   ```

2. **Test API key directly:**
   ```python
   # test_api_keys.py
   import os
   import asyncio
   import aiohttp

   async def test_openai_key():
       api_key = os.getenv("OPENAI_API_KEY")
       if not api_key:
           print("❌ OPENAI_API_KEY not set")
           return
       
       headers = {"Authorization": f"Bearer {api_key}"}
       async with aiohttp.ClientSession() as session:
           async with session.get("https://api.openai.com/v1/models", headers=headers) as response:
               if response.status == 200:
                   print("✅ OpenAI API key valid")
               else:
                   print(f"❌ OpenAI API key invalid: {response.status}")

   async def test_anthropic_key():
       api_key = os.getenv("ANTHROPIC_API_KEY")
       if not api_key:
           print("❌ ANTHROPIC_API_KEY not set")
           return
       
       headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
       async with aiohttp.ClientSession() as session:
           async with session.post(
               "https://api.anthropic.com/v1/messages",
               headers=headers,
               json={
                   "model": "claude-3-haiku-20240307",
                   "max_tokens": 1,
                   "messages": [{"role": "user", "content": "Hi"}]
               }
           ) as response:
               if response.status in [200, 400]:  # 400 is ok for test
                   print("✅ Anthropic API key valid")
               else:
                   print(f"❌ Anthropic API key invalid: {response.status}")

   async def main():
       await test_openai_key()
       await test_anthropic_key()

   if __name__ == "__main__":
       asyncio.run(main())
   ```

**Solutions:**

1. **Set environment variables correctly:**
   ```bash
   # For current session
   export OPENAI_API_KEY="sk-..."
   export ANTHROPIC_API_KEY="sk-ant-..."
   
   # Permanently (add to ~/.bashrc or ~/.zshrc)
   echo 'export OPENAI_API_KEY="sk-..."' >> ~/.bashrc
   source ~/.bashrc
   ```

2. **Use .env file:**
   ```bash
   # Create .env file in project root
   cat > .env << EOF
   OPENAI_API_KEY=sk-...
   ANTHROPIC_API_KEY=sk-ant-...
   AZURE_SPEECH_KEY=your-azure-key
   ELEVENLABS_API_KEY=your-elevenlabs-key
   EOF
   
   # Load in Python
   from dotenv import load_dotenv
   load_dotenv()
   ```

### Issue 3: "Rate Limit Exceeded"

**Symptoms:**
```bash
RateLimitError: Rate limit reached for requests
HTTP 429: Too Many Requests
```

**Diagnostic Steps:**

1. **Check current usage:**
   ```python
   # check_usage.py
   import openai
   import os

   openai.api_key = os.getenv("OPENAI_API_KEY")
   
   try:
       # This will show rate limit headers
       response = openai.ChatCompletion.create(
           model="gpt-3.5-turbo",
           messages=[{"role": "user", "content": "Hi"}],
           max_tokens=1
       )
       print("✅ API call successful")
   except openai.error.RateLimitError as e:
       print(f"❌ Rate limit: {e}")
   ```

**Solutions:**

1. **Implement exponential backoff:**
   ```python
   import asyncio
   import random

   async def api_call_with_backoff(api_func, max_retries=5):
       for attempt in range(max_retries):
           try:
               return await api_func()
           except RateLimitError:
               if attempt == max_retries - 1:
                   raise
               
               # Exponential backoff with jitter
               delay = (2 ** attempt) + random.uniform(0, 1)
               print(f"Rate limited, waiting {delay:.2f} seconds...")
               await asyncio.sleep(delay)
   ```

2. **Use multiple API keys (round-robin):**
   ```python
   import itertools

   class APIKeyRotator:
       def __init__(self, api_keys):
           self.keys = itertools.cycle(api_keys)
           self.current_key = next(self.keys)
       
       def rotate(self):
           self.current_key = next(self.keys)
           return self.current_key
   
   # Usage
   keys = [os.getenv("OPENAI_API_KEY_1"), os.getenv("OPENAI_API_KEY_2")]
   rotator = APIKeyRotator([k for k in keys if k])
   ```

3. **Upgrade API tier:**
   - Check your OpenAI account tier at https://platform.openai.com/account/limits
   - Consider upgrading to higher tier for increased limits

---

## Audio Issues

### Issue: No Audio Input Detected

**Symptoms:**
- Voice agent doesn't respond to speech
- WebRTC connection fails
- Browser shows microphone permission denied

**Diagnostic Steps:**

1. **Check browser permissions:**
   ```javascript
   // Run in browser console
   navigator.mediaDevices.getUserMedia({ audio: true })
     .then(stream => {
       console.log("✅ Microphone access granted");
       stream.getTracks().forEach(track => track.stop());
     })
     .catch(err => console.log("❌ Microphone access denied:", err));
   ```

2. **Test audio devices:**
   ```python
   # test_audio.py
   import asyncio
   
   async def test_audio_devices():
       try:
           # This would use your audio detection logic
           devices = await get_audio_input_devices()
           print(f"Found {len(devices)} audio devices:")
           for device in devices:
               print(f"  - {device['name']} (ID: {device['id']})")
       except Exception as e:
           print(f"❌ Audio device detection failed: {e}")
   
   # Mock function - replace with actual implementation
   async def get_audio_input_devices():
       return [{"name": "Default Microphone", "id": "default"}]
   
   if __name__ == "__main__":
       asyncio.run(test_audio_devices())
   ```

**Solutions:**

1. **Enable microphone permissions:**
   - **Chrome:** Settings → Privacy & Security → Site Settings → Microphone
   - **Firefox:** Settings → Privacy & Security → Permissions → Microphone
   - **Safari:** Preferences → Websites → Microphone

2. **Check system audio settings:**
   ```bash
   # Linux
   arecord -l  # List audio devices
   pulseaudio --check  # Check PulseAudio
   
   # macOS
   system_profiler SPAudioDataType  # List audio devices
   
   # Test recording
   arecord -f cd -t wav -d 5 test.wav  # Record 5 seconds
   ```

3. **Use fallback text mode:**
   ```python
   class VoiceAgentWithFallback:
       def __init__(self):
           self.audio_available = False
           self.text_mode = False
       
       async def start_conversation(self):
           try:
               await self.setup_audio()
               self.audio_available = True
           except AudioError:
               print("Audio not available, switching to text mode")
               self.text_mode = True
           
           if self.text_mode:
               await self.start_text_conversation()
           else:
               await self.start_voice_conversation()
   ```

### Issue: Poor Audio Quality

**Symptoms:**
- Garbled speech recognition
- Audio cutting out
- High latency

**Diagnostic Steps:**

1. **Analyze audio quality:**
   ```python
   # audio_quality_test.py
   import numpy as np
   
   def analyze_audio_quality(audio_data):
       """Analyze audio for common quality issues."""
       audio_array = np.frombuffer(audio_data, dtype=np.int16)
       
       # Check for clipping
       clipped_samples = np.sum(np.abs(audio_array) >= 32767)
       clipping_percent = (clipped_samples / len(audio_array)) * 100
       
       # Check RMS level
       rms = np.sqrt(np.mean(audio_array.astype(float) ** 2))
       rms_db = 20 * np.log10(rms / 32767) if rms > 0 else -80
       
       # Check for silence
       silence_samples = np.sum(np.abs(audio_array) < 100)
       silence_percent = (silence_samples / len(audio_array)) * 100
       
       return {
           "clipping_percent": clipping_percent,
           "rms_db": rms_db,
           "silence_percent": silence_percent,
           "sample_count": len(audio_array)
       }
   ```

**Solutions:**

1. **Adjust microphone levels:**
   ```python
   # Audio preprocessing
   def preprocess_audio(audio_data):
       audio_array = np.frombuffer(audio_data, dtype=np.int16)
       
       # Normalize audio
       if np.max(np.abs(audio_array)) > 0:
           audio_array = audio_array / np.max(np.abs(audio_array)) * 0.8
       
       # Apply noise gate
       noise_floor = np.percentile(np.abs(audio_array), 10)
       audio_array[np.abs(audio_array) < noise_floor * 3] = 0
       
       return audio_array.astype(np.int16).tobytes()
   ```

2. **Use better codec settings:**
   ```python
   webrtc_config = {
       "audio": {
           "codec": "opus",
           "bitrate": 64000,
           "sample_rate": 48000,
           "channels": 1,
           "echo_cancellation": True,
           "noise_suppression": True,
           "auto_gain_control": True
       }
   }
   ```

---

## API and Provider Issues

### Issue: "Connection Timeout" Errors

**Symptoms:**
```bash
asyncio.TimeoutError: Connection timeout
aiohttp.ClientConnectorError: Cannot connect to host
```

**Diagnostic Steps:**

1. **Test network connectivity:**
   ```bash
   # Test DNS resolution
   nslookup api.openai.com
   nslookup api.anthropic.com
   
   # Test HTTP connectivity
   curl -I https://api.openai.com/v1/models
   curl -I https://api.anthropic.com/v1/messages
   
   # Check for proxy/firewall issues
   echo $HTTP_PROXY
   echo $HTTPS_PROXY
   ```

2. **Test with different timeouts:**
   ```python
   import asyncio
   import aiohttp

   async def test_connection_timeouts():
       timeout_configs = [5, 10, 30, 60]  # seconds
       
       for timeout in timeout_configs:
           try:
               timeout_obj = aiohttp.ClientTimeout(total=timeout)
               async with aiohttp.ClientSession(timeout=timeout_obj) as session:
                   async with session.get("https://api.openai.com/v1/models") as response:
                       print(f"✅ Connection successful with {timeout}s timeout")
                       break
           except asyncio.TimeoutError:
               print(f"❌ Timeout with {timeout}s timeout")
           except Exception as e:
               print(f"❌ Error with {timeout}s timeout: {e}")
   ```

**Solutions:**

1. **Configure appropriate timeouts:**
   ```python
   # Adaptive timeout configuration
   class AdaptiveHTTPClient:
       def __init__(self):
           self.base_timeout = 30
           self.max_timeout = 120
           self.current_timeout = self.base_timeout
       
       async def make_request(self, url, **kwargs):
           for attempt in range(3):
               try:
                   timeout = aiohttp.ClientTimeout(total=self.current_timeout)
                   async with aiohttp.ClientSession(timeout=timeout) as session:
                       async with session.get(url, **kwargs) as response:
                           # Success - reduce timeout for next time
                           self.current_timeout = max(
                               self.base_timeout,
                               self.current_timeout * 0.9
                           )
                           return response
               
               except asyncio.TimeoutError:
                   # Increase timeout for next attempt
                   self.current_timeout = min(
                       self.max_timeout,
                       self.current_timeout * 1.5
                   )
                   if attempt < 2:
                       await asyncio.sleep(2 ** attempt)
   ```

2. **Configure corporate proxy/firewall:**
   ```python
   # For corporate environments
   proxy_config = {
       "http": os.getenv("HTTP_PROXY"),
       "https": os.getenv("HTTPS_PROXY")
   }
   
   # SSL verification issues
   ssl_context = ssl.create_default_context()
   if os.getenv("SKIP_SSL_VERIFY"):
       ssl_context.check_hostname = False
       ssl_context.verify_mode = ssl.CERT_NONE
   
   connector = aiohttp.TCPConnector(ssl=ssl_context)
   session = aiohttp.ClientSession(connector=connector)
   ```

### Issue: Inconsistent AI Provider Responses

**Symptoms:**
- Sometimes good responses, sometimes errors
- Varying response quality
- Intermittent failures

**Diagnostic Steps:**

1. **Log provider responses:**
   ```python
   import json
   import logging

   # Set up detailed logging
   logging.basicConfig(level=logging.DEBUG)
   logger = logging.getLogger(__name__)

   async def log_provider_call(provider_name, request_data, response_data, error=None):
       log_entry = {
           "timestamp": datetime.now().isoformat(),
           "provider": provider_name,
           "request": request_data,
           "response": response_data,
           "error": str(error) if error else None
       }
       
       if error:
           logger.error(f"Provider call failed: {json.dumps(log_entry)}")
       else:
           logger.info(f"Provider call success: {json.dumps(log_entry)}")
   ```

2. **Monitor provider status:**
   ```python
   # provider_status_checker.py
   import asyncio
   import aiohttp

   async def check_provider_status():
       providers = {
           "OpenAI": "https://status.openai.com/api/v2/status.json",
           "Anthropic": "https://status.anthropic.com/api/v2/status.json",
           "Azure": "https://status.azure.com/en-us/status"
       }
       
       for name, url in providers.items():
           try:
               async with aiohttp.ClientSession() as session:
                   async with session.get(url) as response:
                       if response.status == 200:
                           data = await response.json()
                           status = data.get("status", {}).get("indicator", "unknown")
                           print(f"{name}: {status}")
                       else:
                           print(f"{name}: HTTP {response.status}")
           except Exception as e:
               print(f"{name}: Error - {e}")
   ```

**Solutions:**

1. **Implement provider fallback:**
   ```python
   class ProviderChain:
       def __init__(self, providers):
           self.providers = providers
           self.current_index = 0
       
       async def call_with_fallback(self, method_name, *args, **kwargs):
           last_error = None
           
           for i in range(len(self.providers)):
               provider = self.providers[self.current_index]
               try:
                   method = getattr(provider, method_name)
                   result = await method(*args, **kwargs)
                   return result
               
               except Exception as e:
                   last_error = e
                   logger.warning(f"Provider {provider.__class__.__name__} failed: {e}")
                   
                   # Try next provider
                   self.current_index = (self.current_index + 1) % len(self.providers)
           
           # All providers failed
           raise Exception(f"All providers failed. Last error: {last_error}")
   ```

---

## Performance Issues

### Issue: High Latency

**Symptoms:**
- Slow response times (>5 seconds)
- Users complaining about delays
- Timeout errors

**Diagnostic Steps:**

1. **Measure component latencies:**
   ```python
   import time
   import asyncio

   class LatencyProfiler:
       def __init__(self):
           self.measurements = {}
       
       async def measure(self, component_name, func, *args, **kwargs):
           start_time = time.perf_counter()
           try:
               result = await func(*args, **kwargs)
               duration = (time.perf_counter() - start_time) * 1000
               
               if component_name not in self.measurements:
                   self.measurements[component_name] = []
               self.measurements[component_name].append(duration)
               
               print(f"{component_name}: {duration:.1f}ms")
               return result
           
           except Exception as e:
               duration = (time.perf_counter() - start_time) * 1000
               print(f"{component_name}: FAILED after {duration:.1f}ms - {e}")
               raise
       
       def get_stats(self):
           stats = {}
           for component, times in self.measurements.items():
               stats[component] = {
                   "avg": sum(times) / len(times),
                   "min": min(times),
                   "max": max(times),
                   "count": len(times)
               }
           return stats

   # Usage
   profiler = LatencyProfiler()

   async def profile_voice_agent():
       # Profile STT
       await profiler.measure("STT", stt_provider.transcribe, audio_data)
       
       # Profile LLM
       await profiler.measure("LLM", llm_provider.generate, prompt)
       
       # Profile TTS
       await profiler.measure("TTS", tts_provider.synthesize, text)
       
       print("\nPerformance Summary:")
       for component, stats in profiler.get_stats().items():
           print(f"{component}: avg={stats['avg']:.1f}ms, max={stats['max']:.1f}ms")
   ```

**Solutions:**

1. **Optimize model selection:**
   ```python
   # Use faster models for real-time interaction
   fast_config = {
       "stt_model": "whisper-1",  # Instead of whisper-large
       "llm_model": "gpt-4o-mini",  # Instead of gpt-4
       "tts_model": "tts-1",  # Instead of tts-1-hd
   }

   # Switch models based on use case
   async def get_optimal_model(use_case, latency_requirement):
       if latency_requirement < 1000:  # Sub-second requirement
           return {
               "llm": "gpt-3.5-turbo",
               "tts": "tts-1",
               "stt": "whisper-1"
           }
       elif latency_requirement < 3000:  # 3-second requirement
           return {
               "llm": "gpt-4o-mini",
               "tts": "tts-1-hd", 
               "stt": "whisper-1"
           }
       else:  # Quality over speed
           return {
               "llm": "gpt-4o",
               "tts": "tts-1-hd",
               "stt": "whisper-large"
           }
   ```

2. **Implement streaming and caching:**
   ```python
   import asyncio
   from typing import AsyncIterator

   class StreamingProcessor:
       def __init__(self):
           self.response_cache = {}
       
       async def stream_llm_response(self, prompt: str) -> AsyncIterator[str]:
           # Check cache first
           cache_key = hash(prompt)
           if cache_key in self.response_cache:
               cached_response = self.response_cache[cache_key]
               for chunk in cached_response.split():
                   yield chunk
                   await asyncio.sleep(0.01)  # Simulate streaming
               return
           
           # Stream from LLM
           full_response = ""
           async for chunk in llm_provider.stream_generate(prompt):
               full_response += chunk
               yield chunk
           
           # Cache complete response
           self.response_cache[cache_key] = full_response
       
       async def parallel_processing(self, audio_data: bytes):
           """Process audio and prepare response in parallel."""
           # Start STT
           stt_task = asyncio.create_task(stt_provider.transcribe(audio_data))
           
           # While STT is running, prepare common responses
           common_responses_task = asyncio.create_task(
               self.preload_common_responses()
           )
           
           # Wait for STT
           transcription = await stt_task
           
           # Start LLM processing
           llm_task = asyncio.create_task(llm_provider.generate(transcription))
           
           # Wait for both to complete
           response, _ = await asyncio.gather(llm_task, common_responses_task)
           
           return response
   ```

### Issue: High Memory Usage

**Symptoms:**
- Out of memory errors
- System slowdown
- Memory leaks

**Diagnostic Steps:**

1. **Monitor memory usage:**
   ```python
   import psutil
   import gc
   import sys

   def memory_usage_debug():
       process = psutil.Process()
       memory_info = process.memory_info()
       
       print(f"RSS Memory: {memory_info.rss / 1024 / 1024:.1f} MB")
       print(f"VMS Memory: {memory_info.vms / 1024 / 1024:.1f} MB")
       print(f"Memory Percent: {process.memory_percent():.1f}%")
       
       # Python object counts
       print(f"Python objects: {len(gc.get_objects())}")
       
       # Garbage collection stats
       gc_stats = gc.get_stats()
       print(f"GC Stats: {gc_stats}")

   # Monitor during conversation
   async def monitor_memory_during_conversation():
       while True:
           memory_usage_debug()
           await asyncio.sleep(10)  # Check every 10 seconds
   ```

**Solutions:**

1. **Implement memory management:**
   ```python
   import gc
   import weakref

   class MemoryManager:
       def __init__(self, max_memory_mb=1000):
           self.max_memory_mb = max_memory_mb
           self.cached_objects = weakref.WeakValueDictionary()
       
       def check_memory_usage(self):
           process = psutil.Process()
           memory_mb = process.memory_info().rss / 1024 / 1024
           
           if memory_mb > self.max_memory_mb:
               self.cleanup_memory()
           
           return memory_mb
       
       def cleanup_memory(self):
           # Clear caches
           self.cached_objects.clear()
           
           # Force garbage collection
           gc.collect()
           
           # Clear large objects
           if hasattr(self, 'conversation_history'):
               # Keep only recent history
               self.conversation_history = self.conversation_history[-10:]
       
       def cache_with_limit(self, key, obj, max_size=100):
           if len(self.cached_objects) >= max_size:
               # Remove oldest entries
               keys_to_remove = list(self.cached_objects.keys())[:10]
               for k in keys_to_remove:
                   self.cached_objects.pop(k, None)
           
           self.cached_objects[key] = obj
   ```

---

## Deployment Issues

### Issue: Docker Container Fails to Start

**Symptoms:**
```bash
Error: container failed to start
ImportError in container logs
Permission denied errors
```

**Diagnostic Steps:**

1. **Check Docker logs:**
   ```bash
   # Get container logs
   docker logs <container_id>
   
   # Follow logs in real-time
   docker logs -f <container_id>
   
   # Check if container is running
   docker ps -a
   ```

2. **Test Docker image locally:**
   ```bash
   # Build image
   docker build -t voice-agent .
   
   # Run interactively
   docker run -it voice-agent /bin/bash
   
   # Test Python imports
   python -c "from src.sdk.python_sdk import VoiceAgentSDK"
   ```

**Solutions:**

1. **Fix Dockerfile issues:**
   ```dockerfile
   # Dockerfile
   FROM python:3.9-slim

   # Install system dependencies
   RUN apt-get update && apt-get install -y \
       build-essential \
       portaudio19-dev \
       && rm -rf /var/lib/apt/lists/*

   # Set working directory
   WORKDIR /app

   # Copy requirements first (for better caching)
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt

   # Copy application code
   COPY . .

   # Set Python path
   ENV PYTHONPATH=/app

   # Create non-root user
   RUN useradd -m appuser && chown -R appuser:appuser /app
   USER appuser

   # Health check
   HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
       CMD python -c "from src.sdk.python_sdk import VoiceAgentSDK; print('OK')"

   CMD ["python", "examples/basic_voice_bot.py"]
   ```

2. **Use multi-stage builds for optimization:**
   ```dockerfile
   # Multi-stage Dockerfile
   FROM python:3.9-slim as builder

   # Install build dependencies
   RUN apt-get update && apt-get install -y build-essential

   # Install Python dependencies
   COPY requirements.txt .
   RUN pip install --user --no-cache-dir -r requirements.txt

   # Production stage
   FROM python:3.9-slim

   # Install runtime dependencies
   RUN apt-get update && apt-get install -y \
       portaudio19-dev \
       && rm -rf /var/lib/apt/lists/*

   # Copy Python packages from builder
   COPY --from=builder /root/.local /root/.local

   # Copy application
   WORKDIR /app
   COPY . .

   ENV PYTHONPATH=/app
   ENV PATH=/root/.local/bin:$PATH

   CMD ["python", "examples/basic_voice_bot.py"]
   ```

### Issue: Kubernetes Deployment Fails

**Symptoms:**
```bash
Pod stuck in Pending state
ImagePullBackOff errors
CrashLoopBackOff errors
```

**Diagnostic Steps:**

1. **Check pod status:**
   ```bash
   # Get pod status
   kubectl get pods -l app=voice-agent
   
   # Describe pod for detailed info
   kubectl describe pod <pod-name>
   
   # Check logs
   kubectl logs <pod-name>
   kubectl logs <pod-name> --previous  # Previous container logs
   ```

2. **Check resource requirements:**
   ```bash
   # Check node resources
   kubectl top nodes
   kubectl describe nodes
   
   # Check pod resource requests
   kubectl describe pod <pod-name> | grep -A 10 Requests
   ```

**Solutions:**

1. **Fix Kubernetes manifests:**
   ```yaml
   # deployment.yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: voice-agent
   spec:
     replicas: 2
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
           env:
           - name: OPENAI_API_KEY
             valueFrom:
               secretKeyRef:
                 name: api-secrets
                 key: openai-key
           resources:
             requests:
               memory: "512Mi"
               cpu: "250m"
             limits:
               memory: "1Gi"
               cpu: "500m"
           livenessProbe:
             httpGet:
               path: /health
               port: 8000
             initialDelaySeconds: 30
             periodSeconds: 10
           readinessProbe:
             httpGet:
               path: /ready
               port: 8000
             initialDelaySeconds: 5
             periodSeconds: 5
   ```

2. **Create secrets for API keys:**
   ```bash
   # Create secret
   kubectl create secret generic api-secrets \
     --from-literal=openai-key=$OPENAI_API_KEY \
     --from-literal=anthropic-key=$ANTHROPIC_API_KEY
   
   # Verify secret
   kubectl get secrets api-secrets -o yaml
   ```

---

## Development Environment Issues

### Issue: IDE/Editor Integration Problems

**Symptoms:**
- Import errors in IDE
- No auto-completion
- Type checking not working

**Solutions:**

1. **Configure VS Code:**
   ```json
   // .vscode/settings.json
   {
       "python.defaultInterpreterPath": "./venv/bin/python",
       "python.linting.enabled": true,
       "python.linting.pylintEnabled": true,
       "python.formatting.provider": "black",
       "python.analysis.extraPaths": ["./src"],
       "python.analysis.autoImportCompletions": true
   }
   ```

2. **Configure PyCharm:**
   ```
   File → Settings → Project → Python Interpreter
   - Select your virtual environment
   
   File → Settings → Project → Project Structure
   - Mark 'src' folder as Sources Root
   ```

### Issue: Testing Environment Setup

**Symptoms:**
- Tests fail in CI but work locally
- Import errors in tests
- Mock objects not working

**Solutions:**

1. **Configure pytest:**
   ```ini
   # pytest.ini
   [tool:pytest]
   testpaths = tests
   python_files = test_*.py
   python_classes = Test*
   python_functions = test_*
   addopts = -v --tb=short --strict-markers
   markers =
       slow: marks tests as slow
       integration: marks tests as integration tests
       unit: marks tests as unit tests
   ```

2. **Set up test fixtures:**
   ```python
   # conftest.py
   import pytest
   import asyncio
   from unittest.mock import AsyncMock, Mock

   @pytest.fixture
   def event_loop():
       """Create event loop for async tests."""
       loop = asyncio.new_event_loop()
       yield loop
       loop.close()

   @pytest.fixture
   def mock_stt_provider():
       """Mock STT provider."""
       mock = AsyncMock()
       mock.transcribe.return_value = "Hello, how can I help you?"
       return mock

   @pytest.fixture
   def mock_llm_provider():
       """Mock LLM provider."""
       mock = AsyncMock()
       mock.generate.return_value = "I'm here to help!"
       return mock

   @pytest.fixture
   def voice_agent_config():
       """Test configuration."""
       return {
           "stt_provider": "mock",
           "llm_provider": "mock",
           "tts_provider": "mock"
       }
   ```

---

## Debugging Tools

### Log Analysis Scripts

```bash
# analyze_logs.sh
#!/bin/bash

LOG_FILE=${1:-voice_agent.log}

echo "=== Error Summary ==="
grep "ERROR" $LOG_FILE | cut -d'-' -f4- | sort | uniq -c | sort -nr

echo -e "\n=== Performance Issues ==="
grep "PERFORMANCE_METRIC" $LOG_FILE | grep -E "(latency|duration)" | \
    awk '{print $NF}' | sort -n | tail -10

echo -e "\n=== API Failures ==="
grep "API call failed" $LOG_FILE | cut -d':' -f3- | sort | uniq -c

echo -e "\n=== Recent Errors (last 10) ==="
grep "ERROR" $LOG_FILE | tail -10
```

### Health Check Endpoint

```python
# health_check.py
from fastapi import FastAPI
from typing import Dict, Any
import asyncio

app = FastAPI()

@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Comprehensive health check endpoint."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {}
    }
    
    # Check STT provider
    try:
        # Test STT with dummy data
        await test_stt_provider()
        health_status["components"]["stt"] = {"status": "healthy"}
    except Exception as e:
        health_status["components"]["stt"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "degraded"
    
    # Check LLM provider
    try:
        await test_llm_provider()
        health_status["components"]["llm"] = {"status": "healthy"}
    except Exception as e:
        health_status["components"]["llm"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "degraded"
    
    # Check TTS provider
    try:
        await test_tts_provider()
        health_status["components"]["tts"] = {"status": "healthy"}
    except Exception as e:
        health_status["components"]["tts"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "degraded"
    
    return health_status

async def test_stt_provider():
    # Implement actual STT test
    pass

async def test_llm_provider():
    # Implement actual LLM test
    pass

async def test_tts_provider():
    # Implement actual TTS test
    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

This troubleshooting guide provides comprehensive solutions for the most common issues you'll encounter when building voice agents. Keep it handy during development and deployment!