"""
Unit tests for conversation inspector.

Tests the conversation inspection, debugging, and analysis capabilities.
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path
import tempfile

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from monitoring.conversation_inspector import (
    ConversationInspector,
    ConversationEvent,
    ConversationTurn,
    ConversationSession,
    EventType,
    InspectorIntegration
)


class TestConversationEvent:
    """Test ConversationEvent class."""
    
    def test_event_creation(self):
        """Test creating conversation events."""
        event = ConversationEvent(
            event_type=EventType.STT_START,
            timestamp=datetime.now(),
            data={"text": "test"},
            metadata={"session": "123"}
        )
        
        assert event.event_type == EventType.STT_START
        assert event.data["text"] == "test"
        assert event.metadata["session"] == "123"
    
    def test_event_to_dict(self):
        """Test converting event to dictionary."""
        timestamp = datetime.now()
        event = ConversationEvent(
            event_type=EventType.LLM_RESPONSE_COMPLETE,
            timestamp=timestamp,
            duration_ms=1500.0,
            data={"response": "Hello"}
        )
        
        event_dict = event.to_dict()
        assert event_dict["event_type"] == "llm_response_complete"
        assert event_dict["timestamp"] == timestamp.isoformat()
        assert event_dict["duration_ms"] == 1500.0
        assert event_dict["data"]["response"] == "Hello"


class TestConversationTurn:
    """Test ConversationTurn class."""
    
    def test_turn_creation(self):
        """Test creating conversation turns."""
        turn = ConversationTurn(
            turn_id="turn_0",
            speaker="user",
            start_time=datetime.now()
        )
        
        assert turn.turn_id == "turn_0"
        assert turn.speaker == "user"
        assert turn.duration is None  # Not ended yet
        assert len(turn.events) == 0
    
    def test_turn_duration(self):
        """Test calculating turn duration."""
        start = datetime.now()
        end = start + timedelta(seconds=2.5)
        
        turn = ConversationTurn(
            turn_id="turn_0",
            speaker="agent",
            start_time=start,
            end_time=end
        )
        
        assert turn.duration.total_seconds() == 2.5
    
    def test_add_events(self):
        """Test adding events to a turn."""
        turn = ConversationTurn(
            turn_id="turn_0",
            speaker="user",
            start_time=datetime.now()
        )
        
        event1 = ConversationEvent(EventType.STT_START, datetime.now())
        event2 = ConversationEvent(EventType.STT_FINAL, datetime.now())
        
        turn.add_event(event1)
        turn.add_event(event2)
        
        assert len(turn.events) == 2
        assert turn.events[0].event_type == EventType.STT_START
        assert turn.events[1].event_type == EventType.STT_FINAL
    
    def test_latency_breakdown(self):
        """Test getting latency breakdown for a turn."""
        turn = ConversationTurn(
            turn_id="turn_0",
            speaker="agent",
            start_time=datetime.now()
        )
        
        # Add STT events
        stt_start = datetime.now()
        stt_end = stt_start + timedelta(milliseconds=500)
        turn.add_event(ConversationEvent(EventType.STT_START, stt_start))
        turn.add_event(ConversationEvent(EventType.STT_FINAL, stt_end))
        
        # Add LLM events
        llm_start = stt_end + timedelta(milliseconds=100)
        llm_end = llm_start + timedelta(milliseconds=1000)
        turn.add_event(ConversationEvent(EventType.LLM_PROMPT_SENT, llm_start))
        turn.add_event(ConversationEvent(EventType.LLM_RESPONSE_COMPLETE, llm_end))
        
        # Add TTS events
        tts_start = llm_end + timedelta(milliseconds=50)
        tts_end = tts_start + timedelta(milliseconds=800)
        turn.add_event(ConversationEvent(EventType.TTS_START, tts_start))
        turn.add_event(ConversationEvent(EventType.TTS_COMPLETE, tts_end))
        
        breakdown = turn.get_latency_breakdown()
        
        assert "stt_latency" in breakdown
        assert "llm_latency" in breakdown
        assert "tts_latency" in breakdown
        assert "total_latency" in breakdown
        
        # Check approximate values (allowing for small timing differences)
        assert 490 < breakdown["stt_latency"] < 510
        assert 990 < breakdown["llm_latency"] < 1010
        assert 790 < breakdown["tts_latency"] < 810


class TestConversationSession:
    """Test ConversationSession class."""
    
    def test_session_creation(self):
        """Test creating conversation sessions."""
        session = ConversationSession(
            session_id="session_123",
            start_time=datetime.now()
        )
        
        assert session.session_id == "session_123"
        assert session.duration is None
        assert session.turn_count == 0
        assert len(session.user_turns) == 0
        assert len(session.agent_turns) == 0
    
    def test_turn_filtering(self):
        """Test filtering turns by speaker."""
        session = ConversationSession(
            session_id="session_123",
            start_time=datetime.now()
        )
        
        # Add turns
        user_turn1 = ConversationTurn("turn_0", "user", datetime.now())
        agent_turn1 = ConversationTurn("turn_1", "agent", datetime.now())
        user_turn2 = ConversationTurn("turn_2", "user", datetime.now())
        
        session.turns.extend([user_turn1, agent_turn1, user_turn2])
        
        assert session.turn_count == 3
        assert len(session.user_turns) == 2
        assert len(session.agent_turns) == 1
        assert session.user_turns[0].turn_id == "turn_0"
        assert session.agent_turns[0].turn_id == "turn_1"
    
    def test_average_latencies(self):
        """Test calculating average latencies across session."""
        session = ConversationSession(
            session_id="session_123",
            start_time=datetime.now()
        )
        
        # Create turns with mock latency data
        for i in range(3):
            turn = ConversationTurn(f"turn_{i}", "agent", datetime.now())
            
            # Add events to create latency data
            base_time = datetime.now()
            turn.add_event(ConversationEvent(EventType.STT_START, base_time))
            turn.add_event(ConversationEvent(EventType.STT_FINAL, base_time + timedelta(milliseconds=500 + i*100)))
            
            session.turns.append(turn)
        
        averages = session.get_average_latencies()
        
        assert "avg_stt_latency" in averages
        assert "p95_stt_latency" in averages
        assert "max_stt_latency" in averages
        
        # Check values (should be 500, 600, 700 ms)
        assert 550 < averages["avg_stt_latency"] < 650  # ~600ms average


class TestConversationInspector:
    """Test ConversationInspector class."""
    
    @pytest.mark.asyncio
    async def test_inspector_creation(self):
        """Test creating conversation inspector."""
        inspector = ConversationInspector({
            "buffer_size": 5000,
            "enable_audio_analysis": False
        })
        
        assert inspector.buffer_size == 5000
        assert inspector.enable_audio_analysis is False
        assert not inspector.is_monitoring
        assert inspector.current_session is None
    
    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self):
        """Test starting and stopping monitoring."""
        inspector = ConversationInspector()
        
        # Start monitoring
        await inspector.start_monitoring()
        assert inspector.is_monitoring
        assert inspector.current_session is not None
        assert inspector.current_session.start_time is not None
        
        session_id = inspector.current_session.session_id
        
        # Stop monitoring
        await inspector.stop_monitoring()
        assert not inspector.is_monitoring
        assert inspector.current_session.end_time is not None
    
    @pytest.mark.asyncio
    async def test_log_event(self):
        """Test logging events."""
        inspector = ConversationInspector()
        await inspector.start_monitoring()
        
        # Log an event
        event = inspector.log_event(
            EventType.STT_START,
            {"language": "en"},
            {"component": "whisper"}
        )
        
        assert event.event_type == EventType.STT_START
        assert event.data["language"] == "en"
        assert event.metadata["component"] == "whisper"
        
        # Check event is in buffers
        assert len(inspector.event_buffer) == 1
        assert len(inspector.current_session.events) == 1
    
    @pytest.mark.asyncio
    async def test_turn_management(self):
        """Test turn start and end."""
        inspector = ConversationInspector()
        await inspector.start_monitoring()
        
        # Start user turn
        turn = inspector.start_turn("user")
        assert turn.speaker == "user"
        assert inspector.current_turn == turn
        assert len(inspector.current_session.turns) == 1
        
        # Log some events
        inspector.log_stt_event("start")
        inspector.log_stt_event("final", text="Hello", confidence=0.95)
        
        # End turn
        inspector.end_turn("Hello")
        assert inspector.current_turn is None
        assert turn.end_time is not None
        assert turn.text == "Hello"
        assert len(turn.events) > 0
    
    @pytest.mark.asyncio
    async def test_event_callbacks(self):
        """Test event callbacks."""
        inspector = ConversationInspector()
        
        callback_called = False
        event_data = None
        
        def test_callback(event):
            nonlocal callback_called, event_data
            callback_called = True
            event_data = event.data
        
        # Add callback
        inspector.add_event_callback(EventType.STT_FINAL, test_callback)
        
        # Log event
        inspector.log_event(EventType.STT_FINAL, {"text": "test"})
        
        assert callback_called
        assert event_data["text"] == "test"
    
    @pytest.mark.asyncio
    async def test_stt_event_logging(self):
        """Test STT event logging."""
        inspector = ConversationInspector()
        
        inspector.log_stt_event("start")
        inspector.log_stt_event("partial", text="Hello")
        inspector.log_stt_event("final", text="Hello world", confidence=0.92, language="en")
        
        assert len(inspector.event_buffer) == 3
        
        # Check final event
        final_event = inspector.event_buffer[-1]
        assert final_event.event_type == EventType.STT_FINAL
        assert final_event.data["text"] == "Hello world"
        assert final_event.data["confidence"] == 0.92
        assert final_event.data["language"] == "en"
    
    @pytest.mark.asyncio
    async def test_llm_event_logging(self):
        """Test LLM event logging."""
        inspector = ConversationInspector()
        
        inspector.log_llm_event("prompt", prompt="Test prompt", model="gpt-4")
        inspector.log_llm_event("complete", response="Test response", tokens=50)
        
        assert len(inspector.event_buffer) == 2
        
        # Check events
        prompt_event = inspector.event_buffer[0]
        assert prompt_event.event_type == EventType.LLM_PROMPT_SENT
        assert prompt_event.data["model"] == "gpt-4"
        
        response_event = inspector.event_buffer[1]
        assert response_event.event_type == EventType.LLM_RESPONSE_COMPLETE
        assert response_event.data["tokens"] == 50
    
    @pytest.mark.asyncio
    async def test_error_logging(self):
        """Test error logging."""
        inspector = ConversationInspector()
        await inspector.start_monitoring()
        
        # Start a turn
        inspector.start_turn("agent")
        
        # Log an error
        error = ValueError("Test error")
        inspector.log_error("stt", error, {"retry_count": 2})
        
        assert inspector.error_counts["stt"] == 1
        assert len(inspector.event_buffer) > 0
        
        # Check error event
        error_event = next(e for e in inspector.event_buffer if e.event_type == EventType.ERROR)
        assert error_event.data["component"] == "stt"
        assert error_event.data["error_type"] == "ValueError"
        assert error_event.data["error_message"] == "Test error"
        assert error_event.data["context"]["retry_count"] == 2
        
        # Check turn errors
        assert len(inspector.current_turn.errors) == 1
    
    @pytest.mark.asyncio
    async def test_get_conversation_insights(self):
        """Test getting conversation insights."""
        inspector = ConversationInspector()
        await inspector.start_monitoring()
        
        # Create some conversation data
        inspector.start_turn("user")
        inspector.log_stt_event("final", text="Hello")
        inspector.end_turn("Hello")
        
        inspector.start_turn("agent")
        inspector.log_llm_event("complete", response="Hi there")
        inspector.end_turn("Hi there")
        
        # Log an error
        inspector.log_error("tts", Exception("TTS error"))
        
        # Get insights
        insights = inspector.get_conversation_insights()
        
        assert insights["turn_count"] == 2
        assert insights["user_turns"] == 1
        assert insights["agent_turns"] == 1
        assert insights["error_counts"]["tts"] == 1
    
    @pytest.mark.asyncio
    async def test_anomaly_detection(self):
        """Test anomaly detection."""
        inspector = ConversationInspector()
        await inspector.start_monitoring()
        
        # Create high error rate
        for i in range(10):
            inspector.log_error("stt", Exception(f"Error {i}"))
        
        # Create high latency turn
        turn = inspector.start_turn("agent")
        base_time = datetime.now()
        turn.add_event(ConversationEvent(EventType.LLM_PROMPT_SENT, base_time))
        turn.add_event(ConversationEvent(EventType.LLM_RESPONSE_COMPLETE, base_time + timedelta(seconds=10)))
        inspector.current_session.turns.append(turn)
        
        # Detect anomalies
        anomalies = inspector.detect_anomalies()
        
        assert len(anomalies) > 0
        
        # Check for high error rate anomaly
        error_anomaly = next((a for a in anomalies if a["type"] == "high_error_rate"), None)
        assert error_anomaly is not None
        assert error_anomaly["component"] == "stt"
        assert error_anomaly["error_count"] == 10
    
    @pytest.mark.asyncio
    async def test_turn_timeline(self):
        """Test getting turn timeline."""
        inspector = ConversationInspector()
        await inspector.start_monitoring()
        
        # Create a turn with events
        turn = inspector.start_turn("user")
        
        # Add events with small delays
        inspector.log_vad_event("speech_start")
        await asyncio.sleep(0.01)
        inspector.log_stt_event("start")
        await asyncio.sleep(0.01)
        inspector.log_stt_event("final", text="Test")
        
        # Get timeline
        timeline = inspector.get_turn_timeline()
        
        assert len(timeline) >= 3
        assert timeline[0]["event"] == "turn_start"
        assert timeline[0]["time_ms"] == 0  # First event at time 0
        
        # Check events are in chronological order
        for i in range(1, len(timeline)):
            assert timeline[i]["time_ms"] >= timeline[i-1]["time_ms"]
    
    @pytest.mark.asyncio
    async def test_export_session(self):
        """Test exporting session data."""
        inspector = ConversationInspector()
        await inspector.start_monitoring()
        
        # Create some data
        inspector.start_turn("user")
        inspector.log_stt_event("final", text="Test export")
        inspector.end_turn("Test export")
        
        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            inspector.export_session(temp_path)
            
            # Check file was created and contains data
            with open(temp_path, 'r') as f:
                data = json.load(f)
            
            assert "session" in data
            assert "turns" in data
            assert "events" in data
            assert len(data["turns"]) == 1
            assert data["turns"][0]["text"] == "Test export"
            
        finally:
            # Clean up
            Path(temp_path).unlink()
    
    @pytest.mark.asyncio
    async def test_replay_conversation(self):
        """Test replaying conversation."""
        inspector = ConversationInspector()
        
        # Create mock session data
        session_data = {
            "events": [
                {
                    "timestamp": datetime.now().isoformat(),
                    "event_type": "turn_start",
                    "data": {"speaker": "user"}
                },
                {
                    "timestamp": (datetime.now() + timedelta(seconds=0.5)).isoformat(),
                    "event_type": "stt_final",
                    "data": {"text": "Hello"}
                },
                {
                    "timestamp": (datetime.now() + timedelta(seconds=1)).isoformat(),
                    "event_type": "turn_start",
                    "data": {"speaker": "agent"}
                },
                {
                    "timestamp": (datetime.now() + timedelta(seconds=1.5)).isoformat(),
                    "event_type": "llm_response_complete",
                    "data": {"response": "Hi there"}
                }
            ]
        }
        
        # Test replay (with very high speed to make test fast)
        with patch('time.sleep') as mock_sleep:
            inspector.replay_conversation(session_data, speed=1000.0)
            
            # Should have called sleep for timing
            assert mock_sleep.call_count > 0


class TestInspectorIntegration:
    """Test inspector integration helpers."""
    
    @pytest.mark.asyncio
    async def test_create_agent_callbacks(self):
        """Test creating agent callbacks."""
        inspector = ConversationInspector()
        await inspector.start_monitoring()
        
        # Create callbacks
        callbacks = InspectorIntegration.create_agent_callbacks(inspector)
        
        # Test user speech callbacks
        await callbacks["on_user_speech_start"]()
        assert inspector.current_turn is not None
        assert inspector.current_turn.speaker == "user"
        
        await callbacks["on_user_speech_end"]("Hello", 0.95)
        assert inspector.current_turn is None
        assert len(inspector.current_session.turns) == 1
        assert inspector.current_session.turns[0].text == "Hello"
        
        # Test agent speech callbacks
        await callbacks["on_agent_speech_start"]()
        assert inspector.current_turn is not None
        assert inspector.current_turn.speaker == "agent"
        
        await callbacks["on_agent_speech_end"]("Hi there")
        assert inspector.current_turn is None
        assert len(inspector.current_session.turns) == 2
        
        # Test error callback
        error = ValueError("Test error")
        await callbacks["on_error"]("stt", error)
        assert inspector.error_counts["stt"] == 1


class TestAnalysisMethods:
    """Test analysis methods."""
    
    @pytest.mark.asyncio
    async def test_analyze_turns(self):
        """Test turn analysis."""
        inspector = ConversationInspector()
        await inspector.start_monitoring()
        
        # Create turns with different characteristics
        for i in range(3):
            turn = ConversationTurn(f"user_{i}", "user", datetime.now())
            turn.text = "Hello" * (i + 1)  # Varying lengths
            turn.audio_duration_ms = 1000 + i * 500
            inspector.current_session.turns.append(turn)
        
        for i in range(2):
            turn = ConversationTurn(f"agent_{i}", "agent", datetime.now())
            turn.text = "Response" * (i + 1)
            turn.audio_duration_ms = 2000 + i * 1000
            inspector.current_session.turns.append(turn)
        
        # Analyze
        analysis = inspector._analyze_turns()
        
        assert analysis["user_statistics"]["total_turns"] == 3
        assert analysis["user_statistics"]["avg_turn_length"] > 0
        assert analysis["user_statistics"]["total_speaking_time_ms"] == 4500  # 1000 + 1500 + 2000
        
        assert analysis["agent_statistics"]["total_turns"] == 2
        assert analysis["agent_statistics"]["total_speaking_time_ms"] == 5000  # 2000 + 3000
    
    @pytest.mark.asyncio
    async def test_generate_recommendations(self):
        """Test recommendation generation."""
        inspector = ConversationInspector()
        await inspector.start_monitoring()
        
        # Create session with high latencies
        turn = ConversationTurn("turn_0", "agent", datetime.now())
        base_time = datetime.now()
        
        # High STT latency
        turn.add_event(ConversationEvent(EventType.STT_START, base_time))
        turn.add_event(ConversationEvent(EventType.STT_FINAL, base_time + timedelta(seconds=2)))
        
        # High LLM latency
        turn.add_event(ConversationEvent(EventType.LLM_PROMPT_SENT, base_time))
        turn.add_event(ConversationEvent(EventType.LLM_RESPONSE_COMPLETE, base_time + timedelta(seconds=4)))
        
        inspector.current_session.turns.append(turn)
        
        # Add errors
        inspector.log_error("tts", Exception("TTS error"))
        
        # Generate recommendations
        recommendations = inspector._generate_recommendations()
        
        assert len(recommendations) > 0
        assert any("STT" in rec for rec in recommendations)
        assert any("LLM" in rec for rec in recommendations)
        assert any("error" in rec.lower() for rec in recommendations)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])