"""
Conversation Inspector

Advanced debugging tools for inspecting and analyzing voice agent conversations
in real-time. Provides deep insights into conversation flow, timing, component
performance, and potential issues.

Features:
- Real-time conversation monitoring
- Component performance tracking
- Turn-by-turn analysis
- Audio quality metrics
- LLM prompt/response inspection
- Error detection and diagnosis
- Conversation replay capability
- Export for offline analysis

Usage:
    from monitoring.conversation_inspector import ConversationInspector
    
    inspector = ConversationInspector()
    agent.add_inspector(inspector)
    
    # Start monitoring
    await inspector.start_monitoring()
    
    # Get insights
    insights = inspector.get_conversation_insights()
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import numpy as np
from pathlib import Path


class EventType(Enum):
    """Types of conversation events."""
    # Audio events
    AUDIO_INPUT_START = "audio_input_start"
    AUDIO_INPUT_END = "audio_input_end"
    AUDIO_OUTPUT_START = "audio_output_start"
    AUDIO_OUTPUT_END = "audio_output_end"
    
    # STT events
    STT_START = "stt_start"
    STT_PARTIAL = "stt_partial"
    STT_FINAL = "stt_final"
    STT_ERROR = "stt_error"
    
    # LLM events
    LLM_PROMPT_SENT = "llm_prompt_sent"
    LLM_TOKEN_RECEIVED = "llm_token_received"
    LLM_RESPONSE_COMPLETE = "llm_response_complete"
    LLM_ERROR = "llm_error"
    
    # TTS events
    TTS_START = "tts_start"
    TTS_CHUNK_GENERATED = "tts_chunk_generated"
    TTS_COMPLETE = "tts_complete"
    TTS_ERROR = "tts_error"
    
    # VAD events
    VAD_SPEECH_START = "vad_speech_start"
    VAD_SPEECH_END = "vad_speech_end"
    
    # Turn management
    TURN_START = "turn_start"
    TURN_END = "turn_end"
    INTERRUPTION_DETECTED = "interruption_detected"
    
    # System events
    PIPELINE_FLUSH = "pipeline_flush"
    ERROR = "error"
    WARNING = "warning"


@dataclass
class ConversationEvent:
    """A single event in a conversation."""
    event_type: EventType
    timestamp: datetime
    duration_ms: Optional[float] = None
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
            "data": self.data,
            "metadata": self.metadata
        }


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""
    turn_id: str
    speaker: str  # "user" or "agent"
    start_time: datetime
    end_time: Optional[datetime] = None
    text: Optional[str] = None
    audio_duration_ms: Optional[float] = None
    events: List[ConversationEvent] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def duration(self) -> Optional[timedelta]:
        """Get turn duration."""
        if self.end_time:
            return self.end_time - self.start_time
        return None
    
    def add_event(self, event: ConversationEvent) -> None:
        """Add an event to this turn."""
        self.events.append(event)
    
    def get_latency_breakdown(self) -> Dict[str, float]:
        """Get latency breakdown for this turn."""
        breakdown = {}
        
        # Calculate component latencies
        stt_events = [e for e in self.events if e.event_type in [EventType.STT_START, EventType.STT_FINAL]]
        if len(stt_events) >= 2:
            breakdown["stt_latency"] = (stt_events[-1].timestamp - stt_events[0].timestamp).total_seconds() * 1000
        
        llm_events = [e for e in self.events if e.event_type in [EventType.LLM_PROMPT_SENT, EventType.LLM_RESPONSE_COMPLETE]]
        if len(llm_events) >= 2:
            breakdown["llm_latency"] = (llm_events[-1].timestamp - llm_events[0].timestamp).total_seconds() * 1000
        
        tts_events = [e for e in self.events if e.event_type in [EventType.TTS_START, EventType.TTS_COMPLETE]]
        if len(tts_events) >= 2:
            breakdown["tts_latency"] = (tts_events[-1].timestamp - tts_events[0].timestamp).total_seconds() * 1000
        
        # Total latency
        if breakdown:
            breakdown["total_latency"] = sum(breakdown.values())
        
        return breakdown


@dataclass
class ConversationSession:
    """A complete conversation session."""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    turns: List[ConversationTurn] = field(default_factory=list)
    events: List[ConversationEvent] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> Optional[timedelta]:
        """Get session duration."""
        if self.end_time:
            return self.end_time - self.start_time
        return None
    
    @property
    def turn_count(self) -> int:
        """Get number of turns."""
        return len(self.turns)
    
    @property
    def user_turns(self) -> List[ConversationTurn]:
        """Get user turns."""
        return [t for t in self.turns if t.speaker == "user"]
    
    @property
    def agent_turns(self) -> List[ConversationTurn]:
        """Get agent turns."""
        return [t for t in self.turns if t.speaker == "agent"]
    
    def get_average_latencies(self) -> Dict[str, float]:
        """Get average latencies across all turns."""
        all_latencies = defaultdict(list)
        
        for turn in self.turns:
            breakdown = turn.get_latency_breakdown()
            for component, latency in breakdown.items():
                all_latencies[component].append(latency)
        
        averages = {}
        for component, latencies in all_latencies.items():
            if latencies:
                averages[f"avg_{component}"] = np.mean(latencies)
                averages[f"p95_{component}"] = np.percentile(latencies, 95)
                averages[f"max_{component}"] = max(latencies)
        
        return averages


class ConversationInspector:
    """
    Advanced conversation inspection and debugging tool.
    
    Provides real-time monitoring, analysis, and debugging capabilities
    for voice agent conversations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Current session
        self.current_session: Optional[ConversationSession] = None
        self.current_turn: Optional[ConversationTurn] = None
        
        # Event tracking
        self.event_buffer: List[ConversationEvent] = []
        self.event_callbacks: Dict[EventType, List[Callable]] = defaultdict(list)
        
        # Performance tracking
        self.component_timings: Dict[str, List[float]] = defaultdict(list)
        self.error_counts: Dict[str, int] = defaultdict(int)
        
        # Configuration
        self.buffer_size = config.get("buffer_size", 10000)
        self.export_format = config.get("export_format", "json")
        self.enable_audio_analysis = config.get("enable_audio_analysis", True)
        self.enable_prompt_logging = config.get("enable_prompt_logging", True)
        
        # Real-time monitoring
        self.is_monitoring = False
        self.monitor_callbacks: List[Callable] = []
    
    async def start_monitoring(self) -> None:
        """Start real-time conversation monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        
        # Start new session
        self.current_session = ConversationSession(
            session_id=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            start_time=datetime.now()
        )
        
        print(f"🔍 Conversation Inspector started - Session: {self.current_session.session_id}")
    
    async def stop_monitoring(self) -> None:
        """Stop conversation monitoring."""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        
        if self.current_session:
            self.current_session.end_time = datetime.now()
            
            # Generate final report
            await self.generate_session_report()
        
        print("🔍 Conversation Inspector stopped")
    
    def log_event(
        self,
        event_type: EventType,
        data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ConversationEvent:
        """Log a conversation event."""
        event = ConversationEvent(
            event_type=event_type,
            timestamp=datetime.now(),
            data=data or {},
            metadata=metadata or {}
        )
        
        # Add to buffers
        self.event_buffer.append(event)
        if len(self.event_buffer) > self.buffer_size:
            self.event_buffer.pop(0)
        
        if self.current_session:
            self.current_session.events.append(event)
        
        if self.current_turn:
            self.current_turn.add_event(event)
        
        # Trigger callbacks
        for callback in self.event_callbacks[event_type]:
            try:
                callback(event)
            except Exception as e:
                print(f"❌ Event callback error: {e}")
        
        # Trigger monitor callbacks
        for callback in self.monitor_callbacks:
            try:
                callback(event)
            except Exception as e:
                print(f"❌ Monitor callback error: {e}")
        
        return event
    
    def start_turn(self, speaker: str) -> ConversationTurn:
        """Start a new conversation turn."""
        turn_id = f"turn_{len(self.current_session.turns) if self.current_session else 0}"
        
        self.current_turn = ConversationTurn(
            turn_id=turn_id,
            speaker=speaker,
            start_time=datetime.now()
        )
        
        if self.current_session:
            self.current_session.turns.append(self.current_turn)
        
        self.log_event(EventType.TURN_START, {"speaker": speaker, "turn_id": turn_id})
        
        return self.current_turn
    
    def end_turn(self, text: Optional[str] = None) -> None:
        """End the current turn."""
        if not self.current_turn:
            return
        
        self.current_turn.end_time = datetime.now()
        self.current_turn.text = text
        
        # Calculate turn metrics
        if self.current_turn.duration:
            duration_ms = self.current_turn.duration.total_seconds() * 1000
            self.current_turn.metrics["duration_ms"] = duration_ms
        
        # Get latency breakdown
        latencies = self.current_turn.get_latency_breakdown()
        self.current_turn.metrics.update(latencies)
        
        self.log_event(EventType.TURN_END, {
            "speaker": self.current_turn.speaker,
            "turn_id": self.current_turn.turn_id,
            "duration_ms": self.current_turn.metrics.get("duration_ms"),
            "text": text
        })
        
        self.current_turn = None
    
    def log_stt_event(
        self,
        event_type: str,
        text: Optional[str] = None,
        confidence: Optional[float] = None,
        language: Optional[str] = None
    ) -> None:
        """Log STT-related event."""
        event_map = {
            "start": EventType.STT_START,
            "partial": EventType.STT_PARTIAL,
            "final": EventType.STT_FINAL,
            "error": EventType.STT_ERROR
        }
        
        if event_type not in event_map:
            return
        
        data = {}
        if text:
            data["text"] = text
        if confidence is not None:
            data["confidence"] = confidence
        if language:
            data["language"] = language
        
        self.log_event(event_map[event_type], data)
    
    def log_llm_event(
        self,
        event_type: str,
        prompt: Optional[str] = None,
        response: Optional[str] = None,
        tokens: Optional[int] = None,
        model: Optional[str] = None
    ) -> None:
        """Log LLM-related event."""
        event_map = {
            "prompt": EventType.LLM_PROMPT_SENT,
            "token": EventType.LLM_TOKEN_RECEIVED,
            "complete": EventType.LLM_RESPONSE_COMPLETE,
            "error": EventType.LLM_ERROR
        }
        
        if event_type not in event_map:
            return
        
        data = {}
        if prompt and self.enable_prompt_logging:
            data["prompt"] = prompt
        if response:
            data["response"] = response
        if tokens is not None:
            data["tokens"] = tokens
        if model:
            data["model"] = model
        
        self.log_event(event_map[event_type], data)
    
    def log_tts_event(
        self,
        event_type: str,
        text: Optional[str] = None,
        voice: Optional[str] = None,
        duration_ms: Optional[float] = None
    ) -> None:
        """Log TTS-related event."""
        event_map = {
            "start": EventType.TTS_START,
            "chunk": EventType.TTS_CHUNK_GENERATED,
            "complete": EventType.TTS_COMPLETE,
            "error": EventType.TTS_ERROR
        }
        
        if event_type not in event_map:
            return
        
        data = {}
        if text:
            data["text"] = text
        if voice:
            data["voice"] = voice
        if duration_ms is not None:
            data["duration_ms"] = duration_ms
        
        self.log_event(event_map[event_type], data)
    
    def log_vad_event(self, event_type: str, energy: Optional[float] = None) -> None:
        """Log VAD-related event."""
        event_map = {
            "speech_start": EventType.VAD_SPEECH_START,
            "speech_end": EventType.VAD_SPEECH_END
        }
        
        if event_type not in event_map:
            return
        
        data = {}
        if energy is not None:
            data["energy"] = energy
        
        self.log_event(event_map[event_type], data)
    
    def log_error(self, component: str, error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """Log an error event."""
        self.error_counts[component] += 1
        
        error_data = {
            "component": component,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {}
        }
        
        self.log_event(EventType.ERROR, error_data)
        
        if self.current_turn:
            self.current_turn.errors.append(error_data)
    
    def add_event_callback(self, event_type: EventType, callback: Callable) -> None:
        """Add callback for specific event type."""
        self.event_callbacks[event_type].append(callback)
    
    def add_monitor_callback(self, callback: Callable) -> None:
        """Add callback for all events."""
        self.monitor_callbacks.append(callback)
    
    def get_conversation_insights(self) -> Dict[str, Any]:
        """Get real-time conversation insights."""
        if not self.current_session:
            return {}
        
        insights = {
            "session_id": self.current_session.session_id,
            "duration": str(self.current_session.duration) if self.current_session.duration else "Ongoing",
            "turn_count": self.current_session.turn_count,
            "user_turns": len(self.current_session.user_turns),
            "agent_turns": len(self.current_session.agent_turns),
            "total_events": len(self.current_session.events),
            "error_counts": dict(self.error_counts)
        }
        
        # Add latency insights
        avg_latencies = self.current_session.get_average_latencies()
        insights.update(avg_latencies)
        
        # Add turn insights
        if self.current_session.turns:
            turn_durations = [
                t.duration.total_seconds() * 1000
                for t in self.current_session.turns
                if t.duration
            ]
            
            if turn_durations:
                insights["avg_turn_duration_ms"] = np.mean(turn_durations)
                insights["max_turn_duration_ms"] = max(turn_durations)
        
        return insights
    
    def get_turn_timeline(self, turn_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get timeline of events for a turn."""
        if not self.current_session:
            return []
        
        if turn_id:
            # Find specific turn
            turn = next((t for t in self.current_session.turns if t.turn_id == turn_id), None)
            if not turn:
                return []
            events = turn.events
        else:
            # Get current turn
            if not self.current_turn:
                return []
            events = self.current_turn.events
        
        timeline = []
        start_time = events[0].timestamp if events else datetime.now()
        
        for event in events:
            relative_time = (event.timestamp - start_time).total_seconds() * 1000
            timeline.append({
                "time_ms": relative_time,
                "event": event.event_type.value,
                "data": event.data
            })
        
        return timeline
    
    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect anomalies in the conversation."""
        anomalies = []
        
        if not self.current_session:
            return anomalies
        
        # Check for high error rates
        for component, count in self.error_counts.items():
            if count > 5:
                anomalies.append({
                    "type": "high_error_rate",
                    "component": component,
                    "error_count": count,
                    "severity": "high"
                })
        
        # Check for high latencies
        avg_latencies = self.current_session.get_average_latencies()
        latency_thresholds = {
            "avg_stt_latency": 2000,  # 2s
            "avg_llm_latency": 5000,  # 5s
            "avg_tts_latency": 3000,  # 3s
            "avg_total_latency": 10000  # 10s
        }
        
        for metric, threshold in latency_thresholds.items():
            if metric in avg_latencies and avg_latencies[metric] > threshold:
                anomalies.append({
                    "type": "high_latency",
                    "metric": metric,
                    "value": avg_latencies[metric],
                    "threshold": threshold,
                    "severity": "medium"
                })
        
        # Check for interruptions
        interruption_count = sum(
            1 for e in self.current_session.events
            if e.event_type == EventType.INTERRUPTION_DETECTED
        )
        
        if interruption_count > 3:
            anomalies.append({
                "type": "frequent_interruptions",
                "count": interruption_count,
                "severity": "low"
            })
        
        return anomalies
    
    async def generate_session_report(self) -> Dict[str, Any]:
        """Generate comprehensive session report."""
        if not self.current_session:
            return {}
        
        report = {
            "session_summary": {
                "session_id": self.current_session.session_id,
                "start_time": self.current_session.start_time.isoformat(),
                "end_time": self.current_session.end_time.isoformat() if self.current_session.end_time else None,
                "duration": str(self.current_session.duration) if self.current_session.duration else None,
                "total_turns": self.current_session.turn_count,
                "total_events": len(self.current_session.events)
            },
            "performance_metrics": self.current_session.get_average_latencies(),
            "turn_analysis": self._analyze_turns(),
            "error_analysis": self._analyze_errors(),
            "anomalies": self.detect_anomalies(),
            "recommendations": self._generate_recommendations()
        }
        
        # Print summary
        print("\n" + "="*60)
        print("📊 CONVERSATION INSPECTION REPORT")
        print("="*60)
        print(f"Session: {report['session_summary']['session_id']}")
        print(f"Duration: {report['session_summary']['duration']}")
        print(f"Turns: {report['session_summary']['total_turns']}")
        print(f"Events: {report['session_summary']['total_events']}")
        
        if report['performance_metrics']:
            print("\nPerformance Metrics:")
            for metric, value in report['performance_metrics'].items():
                print(f"  {metric}: {value:.2f}ms")
        
        if report['anomalies']:
            print(f"\n⚠️ Detected {len(report['anomalies'])} anomalies")
        
        print("="*60)
        
        return report
    
    def _analyze_turns(self) -> Dict[str, Any]:
        """Analyze conversation turns."""
        if not self.current_session:
            return {}
        
        user_turns = self.current_session.user_turns
        agent_turns = self.current_session.agent_turns
        
        analysis = {
            "user_statistics": {
                "total_turns": len(user_turns),
                "avg_turn_length": np.mean([len(t.text or "") for t in user_turns]) if user_turns else 0,
                "total_speaking_time_ms": sum([t.audio_duration_ms or 0 for t in user_turns])
            },
            "agent_statistics": {
                "total_turns": len(agent_turns),
                "avg_turn_length": np.mean([len(t.text or "") for t in agent_turns]) if agent_turns else 0,
                "total_speaking_time_ms": sum([t.audio_duration_ms or 0 for t in agent_turns])
            },
            "turn_patterns": self._analyze_turn_patterns()
        }
        
        return analysis
    
    def _analyze_turn_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in turn-taking."""
        if not self.current_session:
            return {}
        
        patterns = {
            "avg_turn_transition_time_ms": 0,
            "interruption_rate": 0,
            "turn_success_rate": 0
        }
        
        # Calculate turn transition times
        transition_times = []
        for i in range(1, len(self.current_session.turns)):
            prev_turn = self.current_session.turns[i-1]
            curr_turn = self.current_session.turns[i]
            
            if prev_turn.end_time and curr_turn.start_time:
                transition_time = (curr_turn.start_time - prev_turn.end_time).total_seconds() * 1000
                transition_times.append(transition_time)
        
        if transition_times:
            patterns["avg_turn_transition_time_ms"] = np.mean(transition_times)
        
        # Calculate interruption rate
        total_turns = len(self.current_session.turns)
        interruptions = sum(
            1 for e in self.current_session.events
            if e.event_type == EventType.INTERRUPTION_DETECTED
        )
        
        if total_turns > 0:
            patterns["interruption_rate"] = interruptions / total_turns
        
        # Calculate turn success rate (turns without errors)
        successful_turns = sum(1 for t in self.current_session.turns if not t.errors)
        if total_turns > 0:
            patterns["turn_success_rate"] = successful_turns / total_turns
        
        return patterns
    
    def _analyze_errors(self) -> Dict[str, Any]:
        """Analyze errors in the conversation."""
        if not self.current_session:
            return {}
        
        error_analysis = {
            "total_errors": sum(self.error_counts.values()),
            "errors_by_component": dict(self.error_counts),
            "error_timeline": [],
            "most_common_errors": []
        }
        
        # Build error timeline
        error_events = [e for e in self.current_session.events if e.event_type == EventType.ERROR]
        
        for event in error_events:
            error_analysis["error_timeline"].append({
                "timestamp": event.timestamp.isoformat(),
                "component": event.data.get("component"),
                "error_type": event.data.get("error_type"),
                "message": event.data.get("error_message")
            })
        
        # Find most common error types
        error_types = defaultdict(int)
        for event in error_events:
            error_type = event.data.get("error_type", "Unknown")
            error_types[error_type] += 1
        
        error_analysis["most_common_errors"] = sorted(
            error_types.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return error_analysis
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        if not self.current_session:
            return recommendations
        
        # Check performance metrics
        avg_latencies = self.current_session.get_average_latencies()
        
        if avg_latencies.get("avg_stt_latency", 0) > 1500:
            recommendations.append("Consider optimizing STT settings or using a faster STT provider")
        
        if avg_latencies.get("avg_llm_latency", 0) > 3000:
            recommendations.append("LLM response time is high - consider using a faster model or reducing prompt complexity")
        
        if avg_latencies.get("avg_tts_latency", 0) > 2000:
            recommendations.append("TTS generation is slow - consider using streaming TTS or a faster voice")
        
        # Check error rates
        if self.error_counts:
            recommendations.append(f"Address errors in components: {', '.join(self.error_counts.keys())}")
        
        # Check interruption patterns
        interruption_count = sum(
            1 for e in self.current_session.events
            if e.event_type == EventType.INTERRUPTION_DETECTED
        )
        
        if interruption_count > 5:
            recommendations.append("High interruption rate detected - consider adjusting VAD sensitivity or turn detection thresholds")
        
        return recommendations
    
    def export_session(self, file_path: str, format: Optional[str] = None) -> None:
        """Export session data for offline analysis."""
        if not self.current_session:
            return
        
        format = format or self.export_format
        
        if format == "json":
            data = {
                "session": {
                    "session_id": self.current_session.session_id,
                    "start_time": self.current_session.start_time.isoformat(),
                    "end_time": self.current_session.end_time.isoformat() if self.current_session.end_time else None,
                    "metadata": self.current_session.metadata
                },
                "turns": [
                    {
                        "turn_id": turn.turn_id,
                        "speaker": turn.speaker,
                        "start_time": turn.start_time.isoformat(),
                        "end_time": turn.end_time.isoformat() if turn.end_time else None,
                        "text": turn.text,
                        "metrics": turn.metrics,
                        "errors": turn.errors,
                        "events": [e.to_dict() for e in turn.events]
                    }
                    for turn in self.current_session.turns
                ],
                "events": [e.to_dict() for e in self.current_session.events],
                "insights": self.get_conversation_insights(),
                "report": asyncio.run(self.generate_session_report())
            }
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
        
        print(f"📁 Session exported to: {file_path}")
    
    def replay_conversation(
        self,
        session_data: Dict[str, Any],
        speed: float = 1.0
    ) -> None:
        """Replay a conversation from exported data."""
        print(f"▶️ Replaying conversation at {speed}x speed")
        
        events = session_data.get("events", [])
        if not events:
            print("No events to replay")
            return
        
        start_time = datetime.fromisoformat(events[0]["timestamp"])
        
        for event in events:
            event_time = datetime.fromisoformat(event["timestamp"])
            relative_time = (event_time - start_time).total_seconds()
            
            # Wait for the appropriate time
            time.sleep(relative_time / speed)
            
            # Display event
            event_type = event["event_type"]
            data = event.get("data", {})
            
            if event_type == "turn_start":
                print(f"\n🔄 {data.get('speaker', 'Unknown')} turn started")
            elif event_type == "stt_final":
                print(f"🎤 User: {data.get('text', '')}")
            elif event_type == "llm_response_complete":
                print(f"🤖 Agent: {data.get('response', '')}")
            elif event_type == "error":
                print(f"❌ Error in {data.get('component', 'Unknown')}: {data.get('error_message', '')}")
        
        print("\n✅ Replay complete")


# Integration helpers for voice agents
class InspectorIntegration:
    """Helper class for integrating the inspector with voice agents."""
    
    @staticmethod
    def create_agent_callbacks(inspector: ConversationInspector) -> Dict[str, Callable]:
        """Create callbacks for agent integration."""
        
        async def on_user_speech_start():
            inspector.start_turn("user")
            inspector.log_vad_event("speech_start")
        
        async def on_user_speech_end(text: str, confidence: float = None):
            inspector.log_stt_event("final", text=text, confidence=confidence)
            inspector.end_turn(text)
        
        async def on_agent_speech_start():
            inspector.start_turn("agent")
        
        async def on_agent_speech_end(text: str):
            inspector.end_turn(text)
        
        async def on_stt_partial(text: str):
            inspector.log_stt_event("partial", text=text)
        
        async def on_llm_prompt(prompt: str, model: str = None):
            inspector.log_llm_event("prompt", prompt=prompt, model=model)
        
        async def on_llm_response(response: str, tokens: int = None):
            inspector.log_llm_event("complete", response=response, tokens=tokens)
        
        async def on_tts_start(text: str, voice: str = None):
            inspector.log_tts_event("start", text=text, voice=voice)
        
        async def on_tts_complete(duration_ms: float = None):
            inspector.log_tts_event("complete", duration_ms=duration_ms)
        
        async def on_error(component: str, error: Exception):
            inspector.log_error(component, error)
        
        return {
            "on_user_speech_start": on_user_speech_start,
            "on_user_speech_end": on_user_speech_end,
            "on_agent_speech_start": on_agent_speech_start,
            "on_agent_speech_end": on_agent_speech_end,
            "on_stt_partial": on_stt_partial,
            "on_llm_prompt": on_llm_prompt,
            "on_llm_response": on_llm_response,
            "on_tts_start": on_tts_start,
            "on_tts_complete": on_tts_complete,
            "on_error": on_error
        }


# Example usage
async def demo_conversation_inspector():
    """Demonstrate conversation inspector capabilities."""
    print("🔍 Conversation Inspector Demo")
    print("="*50)
    
    # Create inspector
    inspector = ConversationInspector({
        "enable_audio_analysis": True,
        "enable_prompt_logging": True
    })
    
    # Start monitoring
    await inspector.start_monitoring()
    
    # Simulate a conversation
    # Turn 1: User speaks
    inspector.start_turn("user")
    inspector.log_vad_event("speech_start")
    await asyncio.sleep(0.1)
    inspector.log_stt_event("start")
    await asyncio.sleep(0.5)
    inspector.log_stt_event("partial", text="Hello, I need")
    await asyncio.sleep(0.3)
    inspector.log_stt_event("final", text="Hello, I need help with my account", confidence=0.95)
    inspector.log_vad_event("speech_end")
    inspector.end_turn("Hello, I need help with my account")
    
    # Turn 2: Agent responds
    inspector.start_turn("agent")
    inspector.log_llm_event("prompt", prompt="User: Hello, I need help with my account", model="gpt-4")
    await asyncio.sleep(1.0)  # Simulate LLM thinking
    inspector.log_llm_event("complete", response="I'd be happy to help you with your account. What specific issue are you experiencing?", tokens=25)
    inspector.log_tts_event("start", text="I'd be happy to help you with your account.", voice="nova")
    await asyncio.sleep(0.8)
    inspector.log_tts_event("complete", duration_ms=800)
    inspector.end_turn("I'd be happy to help you with your account. What specific issue are you experiencing?")
    
    # Simulate an error
    inspector.log_error("stt", Exception("Connection timeout"), {"attempt": 1})
    
    # Get insights
    insights = inspector.get_conversation_insights()
    print("\n📊 Real-time Insights:")
    for key, value in insights.items():
        print(f"  {key}: {value}")
    
    # Detect anomalies
    anomalies = inspector.detect_anomalies()
    if anomalies:
        print(f"\n⚠️ Detected {len(anomalies)} anomalies:")
        for anomaly in anomalies:
            print(f"  - {anomaly['type']}: {anomaly.get('severity', 'unknown')} severity")
    
    # Stop monitoring
    await inspector.stop_monitoring()
    
    # Export session
    inspector.export_session("conversation_debug.json")
    
    print("\n✅ Inspector demo completed!")


if __name__ == "__main__":
    asyncio.run(demo_conversation_inspector())