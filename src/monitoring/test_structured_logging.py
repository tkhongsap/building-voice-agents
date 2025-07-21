"""
Comprehensive Test Suite for Structured Logging System

This module provides comprehensive tests for the structured logging system
including correlation ID tracking, JSON formatting, and pipeline integration.
"""

import pytest
import json
import asyncio
import time
import uuid
import logging
import tempfile
import os
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path
from typing import Dict, Any, List

from .structured_logging import (
    StructuredLogger,
    StructuredJSONFormatter,
    CorrelationIdManager,
    StructuredLogRecord,
    LogLevel,
    ComponentType,
    log_with_correlation,
    timed_operation,
    correlation_id_var,
    request_id_var,
    session_id_var,
    user_id_var,
    component_var
)
from .logging_config import (
    LoggingConfig,
    LoggingManager,
    StructuredLoggingSetup,
    LogFormat,
    LogOutput,
    setup_logging,
    get_logger
)
from .logging_middleware import (
    CorrelationMiddleware,
    LiveKitSessionMiddleware,
    PipelineCorrelationTracker,
    correlation_context
)


class TestStructuredLogRecord:
    """Test StructuredLogRecord functionality."""
    
    def test_create_basic_record(self):
        """Test creating a basic log record."""
        record = StructuredLogRecord(
            timestamp="2024-01-01T00:00:00Z",
            level="INFO",
            message="Test message"
        )
        
        assert record.timestamp == "2024-01-01T00:00:00Z"
        assert record.level == "INFO"
        assert record.message == "Test message"
        assert record.correlation_id is None
        assert record.extra_data == {}
    
    def test_record_with_all_fields(self):
        """Test creating a record with all fields."""
        extra_data = {"key": "value", "number": 42}
        error_details = {"exception_type": "ValueError", "message": "Invalid input"}
        
        record = StructuredLogRecord(
            timestamp="2024-01-01T00:00:00Z",
            level="ERROR",
            message="Test error message",
            correlation_id="corr-123",
            request_id="req-456",
            session_id="sess-789",
            user_id="user-abc",
            component="test_component",
            component_type="api",
            module="test_module",
            function="test_function",
            line_number=42,
            file_path="/path/to/file.py",
            process_id=1234,
            thread_id=5678,
            extra_data=extra_data,
            error_details=error_details
        )
        
        assert record.correlation_id == "corr-123"
        assert record.request_id == "req-456"
        assert record.session_id == "sess-789"
        assert record.user_id == "user-abc"
        assert record.component == "test_component"
        assert record.component_type == "api"
        assert record.extra_data == extra_data
        assert record.error_details == error_details
    
    def test_to_dict(self):
        """Test converting record to dictionary."""
        record = StructuredLogRecord(
            timestamp="2024-01-01T00:00:00Z",
            level="INFO",
            message="Test message",
            correlation_id="corr-123",
            extra_data={"key": "value"}
        )
        
        data = record.to_dict()
        
        assert data["timestamp"] == "2024-01-01T00:00:00Z"
        assert data["level"] == "INFO"
        assert data["message"] == "Test message"
        assert data["correlation_id"] == "corr-123"
        assert data["extra_data"] == {"key": "value"}
        assert "request_id" not in data  # None values should be excluded
    
    def test_to_json(self):
        """Test converting record to JSON."""
        record = StructuredLogRecord(
            timestamp="2024-01-01T00:00:00Z",
            level="INFO",
            message="Test message",
            correlation_id="corr-123"
        )
        
        json_str = record.to_json()
        parsed = json.loads(json_str)
        
        assert parsed["timestamp"] == "2024-01-01T00:00:00Z"
        assert parsed["level"] == "INFO"
        assert parsed["message"] == "Test message"
        assert parsed["correlation_id"] == "corr-123"


class TestStructuredJSONFormatter:
    """Test StructuredJSONFormatter functionality."""
    
    def test_basic_formatting(self):
        """Test basic JSON formatting."""
        formatter = StructuredJSONFormatter()
        
        # Create a mock log record
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.created = 1704067200.0  # 2024-01-01 00:00:00 UTC
        record.module = "test_module"
        record.funcName = "test_function"
        record.thread = 12345
        
        formatted = formatter.format(record)
        parsed = json.loads(formatted)
        
        assert parsed["level"] == "INFO"
        assert parsed["message"] == "Test message"
        assert parsed["module"] == "test_module"
        assert parsed["function"] == "test_function"
        assert parsed["line_number"] == 42
        assert "timestamp" in parsed
    
    def test_formatting_with_context(self):
        """Test formatting with correlation context."""
        formatter = StructuredJSONFormatter()
        
        # Set context variables
        correlation_id_var.set("corr-123")
        request_id_var.set("req-456")
        session_id_var.set("sess-789")
        component_var.set("test_component")
        
        try:
            record = logging.LogRecord(
                name="test.logger",
                level=logging.INFO,
                pathname="/path/to/file.py",
                lineno=42,
                msg="Test message",
                args=(),
                exc_info=None
            )
            record.created = 1704067200.0
            record.module = "test_module"
            record.funcName = "test_function"
            record.thread = 12345
            
            formatted = formatter.format(record)
            parsed = json.loads(formatted)
            
            assert parsed["correlation_id"] == "corr-123"
            assert parsed["request_id"] == "req-456"
            assert parsed["session_id"] == "sess-789"
            assert parsed["component"] == "test_component"
            
        finally:
            # Clean up context
            correlation_id_var.set(None)
            request_id_var.set(None)
            session_id_var.set(None)
            component_var.set(None)
    
    def test_formatting_with_exception(self):
        """Test formatting with exception information."""
        formatter = StructuredJSONFormatter(include_traceback=True)
        
        try:
            raise ValueError("Test error")
        except ValueError:
            record = logging.LogRecord(
                name="test.logger",
                level=logging.ERROR,
                pathname="/path/to/file.py",
                lineno=42,
                msg="Error occurred",
                args=(),
                exc_info=True
            )
            record.created = 1704067200.0
            record.module = "test_module"
            record.funcName = "test_function"
            record.thread = 12345
            
            formatted = formatter.format(record)
            parsed = json.loads(formatted)
            
            assert parsed["level"] == "ERROR"
            assert "error_details" in parsed
            assert parsed["error_details"]["exception_type"] == "ValueError"
            assert parsed["error_details"]["exception_message"] == "Test error"
            assert "traceback" in parsed["error_details"]


class TestCorrelationIdManager:
    """Test CorrelationIdManager functionality."""
    
    def test_generate_correlation_id(self):
        """Test generating correlation IDs."""
        corr_id1 = CorrelationIdManager.generate_correlation_id()
        corr_id2 = CorrelationIdManager.generate_correlation_id()
        
        assert corr_id1 != corr_id2
        assert len(corr_id1) == 36  # UUID format
        assert len(corr_id2) == 36
    
    def test_set_and_get_correlation_id(self):
        """Test setting and getting correlation ID."""
        test_id = "test-correlation-id"
        
        # Initially should be None
        assert CorrelationIdManager.get_correlation_id() is None
        
        # Set and retrieve
        set_id = CorrelationIdManager.set_correlation_id(test_id)
        assert set_id == test_id
        assert CorrelationIdManager.get_correlation_id() == test_id
        
        # Clear
        CorrelationIdManager.clear_correlation_id()
        assert CorrelationIdManager.get_correlation_id() is None
    
    def test_generate_request_id(self):
        """Test generating request IDs."""
        req_id = CorrelationIdManager.generate_request_id()
        assert req_id.startswith("req_")
        assert len(req_id) == 12  # "req_" + 8 hex chars
    
    def test_generate_session_id(self):
        """Test generating session IDs."""
        sess_id = CorrelationIdManager.generate_session_id()
        assert sess_id.startswith("sess_")
        assert len(sess_id) == 17  # "sess_" + 12 hex chars
    
    def test_get_all_context(self):
        """Test getting all context variables."""
        # Set various context variables
        CorrelationIdManager.set_correlation_id("corr-123")
        CorrelationIdManager.set_request_id("req-456")
        CorrelationIdManager.set_session_id("sess-789")
        CorrelationIdManager.set_user_id("user-abc")
        CorrelationIdManager.set_component_context("test_component")
        
        try:
            context = CorrelationIdManager.get_all_context()
            
            assert context["correlation_id"] == "corr-123"
            assert context["request_id"] == "req-456"
            assert context["session_id"] == "sess-789"
            assert context["user_id"] == "user-abc"
            assert context["component"] == "test_component"
            
        finally:
            # Clean up
            CorrelationIdManager.clear_correlation_id()
            correlation_id_var.set(None)
            request_id_var.set(None)
            session_id_var.set(None)
            user_id_var.set(None)
            component_var.set(None)


class TestStructuredLogger:
    """Test StructuredLogger functionality."""
    
    def test_create_logger(self):
        """Test creating a structured logger."""
        logger = StructuredLogger("test.logger", "test_component")
        
        assert logger.component == "test_component"
        assert logger.logger.name == "test.logger"
    
    def test_log_operation_lifecycle(self):
        """Test logging operation start and end."""
        logger = StructuredLogger("test.logger", "test_component")
        
        # Mock the underlying logger
        with patch.object(logger.logger, 'log') as mock_log:
            # Test operation start
            logger.log_operation_start("test_operation", {"input": "data"})
            
            # Verify start log
            assert mock_log.called
            args, kwargs = mock_log.call_args
            assert args[0] == logging.INFO
            assert "Starting operation: test_operation" in args[1]
            assert "extra" in kwargs
            
            # Test operation end
            mock_log.reset_mock()
            logger.log_operation_end("test_operation", 0.5, True, {"output": "result"})
            
            # Verify end log
            assert mock_log.called
            args, kwargs = mock_log.call_args
            assert args[0] == logging.INFO
            assert "Completed operation: test_operation" in args[1]
    
    def test_log_performance_metrics(self):
        """Test logging performance metrics."""
        logger = StructuredLogger("test.logger", "test_component")
        
        with patch.object(logger.logger, 'log') as mock_log:
            metrics = {
                "duration_ms": 500,
                "throughput": 100,
                "success_rate": 0.95
            }
            
            logger.log_performance_metrics("test_operation", metrics)
            
            assert mock_log.called
            args, kwargs = mock_log.call_args
            assert "Performance metrics for test_operation" in args[1]
            assert "extra" in kwargs
            assert kwargs["extra"]["performance_metrics"] == metrics


class TestLoggingConfiguration:
    """Test logging configuration functionality."""
    
    def test_default_config(self):
        """Test default logging configuration."""
        config = LoggingConfig()
        
        assert config.level == "INFO"
        assert config.format_type == LogFormat.JSON
        assert config.output == LogOutput.CONSOLE
        assert config.correlation_tracking is True
        assert config.performance_tracking is True
    
    def test_custom_config(self):
        """Test custom logging configuration."""
        config = LoggingConfig(
            level="DEBUG",
            format_type=LogFormat.PLAIN,
            output=LogOutput.FILE,
            log_file="/tmp/test.log",
            correlation_tracking=False
        )
        
        assert config.level == "DEBUG"
        assert config.format_type == LogFormat.PLAIN
        assert config.output == LogOutput.FILE
        assert config.log_file == "/tmp/test.log"
        assert config.correlation_tracking is False
    
    def test_logging_manager_configuration(self):
        """Test logging manager configuration."""
        manager = LoggingManager()
        
        # Test initial state
        assert not manager.is_configured()
        assert manager.get_configuration() is None
        
        # Configure with defaults
        config = manager.configure()
        
        assert manager.is_configured()
        assert manager.get_configuration() == config
        assert config.level == "INFO"


class TestLoggingMiddleware:
    """Test logging middleware functionality."""
    
    @pytest.mark.asyncio
    async def test_correlation_context(self):
        """Test correlation context manager."""
        # Initially context should be empty
        initial_context = CorrelationIdManager.get_all_context()
        assert all(value is None for value in initial_context.values())
        
        # Use correlation context
        async with correlation_context(
            correlation_id="corr-123",
            request_id="req-456",
            session_id="sess-789",
            user_id="user-abc",
            component="test_component"
        ) as context:
            # Inside context, values should be set
            assert context["correlation_id"] == "corr-123"
            assert context["request_id"] == "req-456"
            assert context["session_id"] == "sess-789"
            assert context["user_id"] == "user-abc"
            assert context["component"] == "test_component"
            
            # Verify context variables are set
            assert CorrelationIdManager.get_correlation_id() == "corr-123"
        
        # After context, values should be restored
        final_context = CorrelationIdManager.get_all_context()
        assert all(value is None for value in final_context.values())
    
    @pytest.mark.asyncio
    async def test_livekit_session_middleware(self):
        """Test LiveKit session middleware."""
        middleware = LiveKitSessionMiddleware()
        
        # Start session
        session_id = await middleware.session_start(
            "test_room",
            "test_participant",
            {"test": "metadata"}
        )
        
        assert session_id.startswith("sess_")
        assert session_id in middleware.active_sessions
        
        session_info = middleware.get_session_info(session_id)
        assert session_info is not None
        assert session_info["room_name"] == "test_room"
        assert session_info["participant_identity"] == "test_participant"
        
        # Log pipeline stage
        await middleware.log_pipeline_stage(
            session_id,
            "stt",
            "openai_stt",
            0.5,
            True,
            {"confidence": 0.95}
        )
        
        session_info = middleware.get_session_info(session_id)
        assert len(session_info["pipeline_stages"]) == 1
        assert session_info["pipeline_stages"][0]["stage"] == "stt"
        
        # End session
        await middleware.session_end(session_id, "completed")
        
        assert session_id not in middleware.active_sessions
    
    @pytest.mark.asyncio
    async def test_pipeline_correlation_tracker(self):
        """Test pipeline correlation tracker."""
        tracker = PipelineCorrelationTracker()
        
        # Start pipeline
        pipeline_id = await tracker.start_pipeline(
            "sess_12345",
            "Test user input"
        )
        
        assert pipeline_id.startswith("pipe_")
        assert pipeline_id in tracker.active_pipelines
        
        # Log stage start
        await tracker.log_stage_start(
            pipeline_id,
            "stt",
            "openai_stt",
            {"audio_size": 1024}
        )
        
        pipeline_info = tracker.get_pipeline_info(pipeline_id)
        assert "stt" in pipeline_info["stages"]
        
        # Log stage end
        await tracker.log_stage_end(
            pipeline_id,
            "stt",
            True,
            {"text": "Hello world"}
        )
        
        pipeline_info = tracker.get_pipeline_info(pipeline_id)
        assert pipeline_info["stages"]["stt"]["success"] is True
        
        # End pipeline
        await tracker.end_pipeline(pipeline_id, "Final output")
        
        assert pipeline_id not in tracker.active_pipelines


class TestDecorators:
    """Test logging decorators."""
    
    def test_log_with_correlation_sync(self):
        """Test log_with_correlation decorator for sync functions."""
        @log_with_correlation(correlation_id="test-corr-id")
        def test_function():
            return CorrelationIdManager.get_correlation_id()
        
        # Initial state
        assert CorrelationIdManager.get_correlation_id() is None
        
        # Call decorated function
        result = test_function()
        assert result == "test-corr-id"
        
        # After function, context should be restored
        assert CorrelationIdManager.get_correlation_id() is None
    
    @pytest.mark.asyncio
    async def test_log_with_correlation_async(self):
        """Test log_with_correlation decorator for async functions."""
        @log_with_correlation(correlation_id="test-corr-id")
        async def test_async_function():
            return CorrelationIdManager.get_correlation_id()
        
        # Initial state
        assert CorrelationIdManager.get_correlation_id() is None
        
        # Call decorated function
        result = await test_async_function()
        assert result == "test-corr-id"
        
        # After function, context should be restored
        assert CorrelationIdManager.get_correlation_id() is None
    
    @pytest.mark.asyncio
    async def test_timed_operation_decorator(self):
        """Test timed_operation decorator."""
        # Mock the logger to capture calls
        with patch('src.monitoring.structured_logging.StructuredLogger') as mock_logger_class:
            mock_logger = Mock()
            mock_logger_class.return_value = mock_logger
            
            @timed_operation(operation_name="test_op", component="test_comp")
            async def test_operation():
                await asyncio.sleep(0.1)  # Simulate work
                return "result"
            
            result = await test_operation()
            
            assert result == "result"
            
            # Verify logger was called
            assert mock_logger.log_operation_start.called
            assert mock_logger.log_operation_end.called
            assert mock_logger.log_performance_metrics.called


class TestIntegration:
    """Integration tests for the complete logging system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_logging(self):
        """Test end-to-end logging workflow."""
        # Setup temporary log file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
            log_file = f.name
        
        try:
            # Configure logging
            config = LoggingConfig(
                level="DEBUG",
                format_type=LogFormat.JSON,
                output=LogOutput.FILE,
                log_file=log_file,
                correlation_tracking=True
            )
            
            manager = LoggingManager()
            manager.configure(config)
            
            # Set correlation context
            correlation_id = CorrelationIdManager.generate_correlation_id()
            session_id = CorrelationIdManager.generate_session_id()
            
            manager.set_correlation_context(
                correlation_id=correlation_id,
                session_id=session_id,
                user_id="test_user",
                component="integration_test"
            )
            
            # Create logger and log messages
            logger = manager.get_logger("integration.test")
            
            logger.info("Starting integration test")
            logger.debug("Debug information", extra_data={"test": "data"})
            logger.warning("Warning message")
            
            # Simulate error
            try:
                raise ValueError("Test error")
            except ValueError:
                logger.exception("Error occurred during test")
            
            # Wait for logs to be written
            await asyncio.sleep(0.1)
            
            # Verify log file contents
            with open(log_file, 'r') as f:
                log_lines = f.readlines()
            
            assert len(log_lines) >= 4  # At least 4 log entries
            
            # Parse and verify log entries
            for line in log_lines:
                log_entry = json.loads(line.strip())
                
                assert "timestamp" in log_entry
                assert "level" in log_entry
                assert "message" in log_entry
                assert log_entry["correlation_id"] == correlation_id
                assert log_entry["session_id"] == session_id
                assert log_entry["user_id"] == "test_user"
                assert log_entry["component"] == "integration_test"
            
            # Check specific log entries
            error_logs = [
                json.loads(line.strip()) for line in log_lines
                if json.loads(line.strip())["level"] == "ERROR"
            ]
            
            assert len(error_logs) >= 1
            error_log = error_logs[0]
            assert "error_details" in error_log
            assert error_log["error_details"]["exception_type"] == "ValueError"
            
        finally:
            # Cleanup
            manager.clear_correlation_context()
            if os.path.exists(log_file):
                os.unlink(log_file)
    
    def test_performance_impact(self):
        """Test that structured logging doesn't significantly impact performance."""
        # Setup simple console logging
        config = LoggingConfig(
            level="INFO",
            format_type=LogFormat.JSON,
            output=LogOutput.CONSOLE
        )
        
        manager = LoggingManager()
        manager.configure(config)
        logger = manager.get_logger("performance.test")
        
        # Measure performance
        start_time = time.time()
        
        for i in range(1000):
            logger.info(f"Test message {i}", extra_data={"iteration": i})
        
        duration = time.time() - start_time
        
        # Should complete 1000 log messages in reasonable time
        assert duration < 5.0  # 5 seconds should be plenty
        
        # Calculate messages per second
        messages_per_second = 1000 / duration
        assert messages_per_second > 100  # Should handle at least 100 msgs/sec


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])