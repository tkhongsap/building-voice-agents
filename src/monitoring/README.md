# Structured Logging System

A comprehensive structured logging system with correlation ID tracking for the Voice Agents platform. This system provides JSON-formatted logs, automatic correlation tracking across the voice processing pipeline (STT → LLM → TTS), and integration with FastAPI and LiveKit.

## 🚀 Features

- **JSON Structured Logging**: All logs are formatted as JSON with consistent fields
- **Correlation ID Tracking**: Automatic correlation across all components and requests
- **Pipeline Correlation**: Track requests through STT → LLM → TTS pipeline stages
- **Performance Monitoring**: Built-in timing and metrics collection
- **FastAPI Integration**: Middleware for automatic request correlation
- **LiveKit Integration**: Session and participant tracking
- **Error Handling**: Comprehensive error logging with stack traces
- **Configurable Output**: Console, file, syslog, or multiple outputs
- **Production Ready**: Optimized for high-throughput voice processing

## 📋 Quick Start

### Basic Setup

```python
from src.monitoring.logging_config import setup_logging, LogFormat, LogOutput

# Setup JSON logging to console
setup_logging(
    level="INFO",
    format_type=LogFormat.JSON,
    output=LogOutput.CONSOLE,
    correlation_tracking=True
)

# Get a structured logger
from src.monitoring.logging_config import get_logger
logger = get_logger("my.module", component="voice_processing")

# Log with structured data
logger.info("Processing started", extra_data={
    "user_id": "123",
    "audio_duration": 5.2,
    "language": "en"
})
```

### Environment Configuration

Set these environment variables for automatic configuration:

```bash
export LOG_LEVEL=INFO
export LOG_FORMAT=json
export LOG_OUTPUT=console
export LOG_CORRELATION_TRACKING=true
export LOG_PERFORMANCE_TRACKING=true
```

## 🏗️ Architecture

### Core Components

1. **StructuredLogger**: Enhanced logger with structured data support
2. **CorrelationIdManager**: Manages correlation IDs across async operations
3. **StructuredJSONFormatter**: JSON formatter with correlation context
4. **LoggingMiddleware**: FastAPI and LiveKit integration
5. **PipelineTracker**: Tracks correlation across voice pipeline stages

### Log Record Structure

```json
{
  "timestamp": "2024-01-01T12:00:00.000Z",
  "level": "INFO",
  "message": "STT processing completed",
  "correlation_id": "550e8400-e29b-41d4-a716-446655440000",
  "request_id": "req_a1b2c3d4",
  "session_id": "sess_f5e6d7c8b9a0",
  "user_id": "user_123",
  "component": "stt",
  "component_type": "stt",
  "module": "openai_stt",
  "function": "transcribe_audio",
  "line_number": 145,
  "file_path": "/app/src/components/stt/openai_stt.py",
  "process_id": 1234,
  "thread_id": 5678,
  "extra_data": {
    "audio_size_bytes": 48000,
    "confidence": 0.95,
    "language": "en",
    "duration_ms": 150
  },
  "performance_metrics": {
    "duration_ms": 150,
    "throughput_bytes_per_second": 320000,
    "success": true
  }
}
```

## 🔧 Configuration

### Programmatic Configuration

```python
from src.monitoring.logging_config import LoggingConfig, LoggingManager, LogFormat, LogOutput

config = LoggingConfig(
    level="DEBUG",
    format_type=LogFormat.JSON,
    output=LogOutput.BOTH,  # Console and file
    log_file="logs/voice_agents.log",
    max_file_size=10 * 1024 * 1024,  # 10MB
    backup_count=5,
    correlation_tracking=True,
    performance_tracking=True,
    console_colors=True
)

manager = LoggingManager()
manager.configure(config)
```

### File-based Configuration

```json
{
  "level": "INFO",
  "format_type": "json",
  "output": "file",
  "log_file": "logs/voice_agents.log",
  "max_file_size": 10485760,
  "backup_count": 5,
  "correlation_tracking": true,
  "performance_tracking": true,
  "filter_modules": ["voice_agents.stt", "voice_agents.llm", "voice_agents.tts"]
}
```

## 🔗 Integration Examples

### FastAPI Integration

```python
from fastapi import FastAPI
from src.monitoring.logging_middleware import add_correlation_middleware

app = FastAPI()

# Add correlation middleware
add_correlation_middleware(
    app,
    correlation_header="X-Correlation-ID",
    log_requests=True,
    log_responses=True,
    exclude_paths=["/health", "/metrics"]
)

@app.get("/api/status")
async def get_status():
    # All logs within this request will have the same correlation ID
    logger = get_logger("api.status")
    logger.info("Status endpoint called")
    return {"status": "ok"}
```

### Voice Pipeline Integration

```python
from src.pipeline.voice_pipeline_with_logging import VoicePipelineWithLogging
from src.monitoring.logging_middleware import correlation_context

# Create pipeline with automatic correlation tracking
pipeline = VoicePipelineWithLogging(
    stt_provider=stt_provider,
    llm_provider=llm_provider,
    tts_provider=tts_provider,
    session_id="session_123"
)

# Process audio with full correlation tracking
async with correlation_context(
    user_id="user_456",
    session_id="session_123"
):
    result, metrics = await pipeline.process_audio(
        audio_data=audio_bytes,
        conversation_context={"previous_topic": "weather"}
    )
```

### Component Integration

```python
# STT Component
class MySTTProvider(BaseSTTProvider):
    async def transcribe_audio(self, audio_data: bytes) -> STTResult:
        # Use the enhanced logging method
        return await self.transcribe_audio_with_logging(
            audio_data, 
            pipeline_id=self._current_pipeline_id
        )

# LLM Component  
class MyLLMProvider(BaseLLMProvider):
    async def generate_response(self, messages) -> LLMResponse:
        # Use the enhanced logging method
        return await self.chat_with_logging(
            user_message,
            pipeline_id=self._current_pipeline_id
        )

# TTS Component
class MyTTSProvider(BaseTTSProvider):
    async def synthesize_speech(self, text: str) -> TTSResult:
        # Use the enhanced logging method
        return await self.speak_with_logging(
            text,
            pipeline_id=self._current_pipeline_id
        )
```

## 🎯 Advanced Usage

### Decorators for Automatic Logging

```python
from src.monitoring.structured_logging import timed_operation, log_with_correlation

@log_with_correlation(component="audio_processor")
@timed_operation(operation_name="process_audio", log_args=True)
async def process_audio_file(file_path: str) -> dict:
    # Automatic timing, correlation tracking, and argument logging
    # Process audio file
    return {"status": "processed", "duration": 5.2}
```

### Custom Performance Metrics

```python
logger = get_logger("voice.stt", component="whisper")

# Log operation lifecycle
logger.log_operation_start("transcribe", {"audio_size": 48000})

# ... perform transcription ...

logger.log_operation_end("transcribe", duration=0.15, success=True, {
    "text_length": 25,
    "confidence": 0.95
})

# Log performance metrics
logger.log_performance_metrics("transcribe", {
    "duration_ms": 150,
    "audio_size_bytes": 48000,
    "text_length": 25,
    "confidence": 0.95,
    "throughput_bytes_per_second": 320000
})
```

### Error Handling with Context

```python
try:
    result = await risky_operation()
except Exception as e:
    logger.exception("Operation failed", extra_data={
        "operation": "risky_operation",
        "input_params": params,
        "error_code": "STT_001",
        "recovery_action": "retry_with_fallback"
    })
    raise
```

## 📊 Monitoring and Observability

### Log Analysis Queries

Query examples for common monitoring scenarios:

```bash
# Find all errors for a specific correlation ID
jq 'select(.correlation_id == "550e8400-e29b-41d4-a716-446655440000" and .level == "ERROR")' voice_agents.log

# Performance metrics for STT operations
jq 'select(.component_type == "stt" and .performance_metrics != null) | .performance_metrics' voice_agents.log

# Pipeline stage durations
jq 'select(.phase == "stage_end") | {stage: .stage, duration_ms: .duration_ms}' voice_agents.log

# Error rates by component
jq -s 'group_by(.component) | map({component: .[0].component, error_count: length})' voice_agents.log
```

### Metrics Collection

The system automatically collects these metrics:

- **Operation Timing**: Duration of all operations
- **Throughput**: Bytes/characters per second
- **Success Rates**: Success/failure ratios
- **Pipeline Stages**: Individual stage performance
- **Resource Usage**: Memory and CPU when available
- **Audio Quality**: Confidence scores and quality metrics

### Integration with Monitoring Systems

#### Prometheus Integration

```python
# Export metrics in Prometheus format
from src.monitoring.performance_monitor import global_performance_monitor

metrics = global_performance_monitor.export_metrics(format_type="prometheus")
# Expose via /metrics endpoint
```

#### Grafana Dashboards

Use the structured logs to create Grafana dashboards:

1. **Voice Pipeline Performance**: Track STT→LLM→TTS latencies
2. **Error Monitoring**: Error rates by component and correlation
3. **User Experience**: Response times and success rates
4. **Resource Utilization**: CPU, memory, and throughput metrics

## 🧪 Testing

### Running Tests

```bash
# Run the comprehensive test suite
python -m pytest src/monitoring/test_structured_logging.py -v

# Run specific test categories
python -m pytest src/monitoring/test_structured_logging.py::TestStructuredLogger -v
python -m pytest src/monitoring/test_structured_logging.py::TestCorrelationIdManager -v
```

### Performance Testing

```python
# Test logging performance impact
python src/monitoring/test_structured_logging.py::TestIntegration::test_performance_impact
```

## 📚 Examples

See comprehensive examples in `examples_structured_logging.py`:

1. **Basic Structured Logging**: Setup and basic usage
2. **Correlation Tracking**: Cross-component correlation
3. **Middleware Integration**: FastAPI and LiveKit integration
4. **Voice Pipeline**: End-to-end pipeline correlation
5. **Performance Monitoring**: Metrics and timing
6. **Error Handling**: Error recovery with structured logging

Run all examples:

```bash
python src/monitoring/examples_structured_logging.py
```

## 🔒 Security Considerations

### Data Sanitization

The logging system automatically handles sensitive data:

- User IDs are included but audio content is not logged
- Request/response bodies are logged only if explicitly enabled
- Error messages are sanitized to prevent information leakage

### Configuration Security

```python
# Example: Exclude sensitive modules from logging
config = LoggingConfig(
    exclude_modules=["auth", "secrets", "credentials"],
    filter_modules=["voice_agents.stt", "voice_agents.llm", "voice_agents.tts"]
)
```

## 🚀 Production Deployment

### Recommended Configuration

```python
# Production configuration
production_config = LoggingConfig(
    level="INFO",  # INFO level for production
    format_type=LogFormat.JSON,
    output=LogOutput.FILE,
    log_file="/var/log/voice_agents/app.log",
    max_file_size=50 * 1024 * 1024,  # 50MB
    backup_count=10,
    correlation_tracking=True,
    performance_tracking=True,
    include_traceback=True,  # Include tracebacks for debugging
    exclude_modules=["urllib3", "requests.packages.urllib3"]  # Reduce noise
)
```

### Log Rotation

The system uses Python's `RotatingFileHandler` for automatic log rotation:

- Files are rotated when they reach `max_file_size`
- `backup_count` determines how many old files to keep
- Filenames follow the pattern: `app.log`, `app.log.1`, `app.log.2`, etc.

### Monitoring Alerts

Set up alerts based on log patterns:

```bash
# High error rate
tail -f /var/log/voice_agents/app.log | jq -r 'select(.level == "ERROR")' | while read line; do
    # Send alert if error rate exceeds threshold
done

# Long response times
tail -f /var/log/voice_agents/app.log | jq -r 'select(.performance_metrics.duration_ms > 5000)' | while read line; do
    # Alert on slow operations
done
```

## 🤝 Contributing

To extend the logging system:

1. Add new log fields to `StructuredLogRecord`
2. Update the JSON formatter in `StructuredJSONFormatter`
3. Add integration points in component base classes
4. Write comprehensive tests for new functionality

## 📄 License

This structured logging system is part of the Voice Agents platform.