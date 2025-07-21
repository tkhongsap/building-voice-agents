"""
Session Playback and Analysis Tools

This module provides comprehensive playback and analysis capabilities for recorded
voice agent sessions, including audio playback, transcript synchronization,
quality analysis, and performance insights.
"""

import asyncio
import json
import time
import wave
import gzip
from typing import Dict, List, Optional, Any, Union, Callable, AsyncIterator, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
import logging
import threading
import io

from .structured_logging import StructuredLogger
from .session_recording import SessionMetadata, AudioChunk, TranscriptSegment, QualityMetrics, SessionStatus, AudioFormat
from .transcript_processor import TranscriptExporter, TranscriptFormat


class PlaybackState(Enum):
    """Playback state enumeration."""
    STOPPED = "stopped"
    PLAYING = "playing"
    PAUSED = "paused"
    SEEKING = "seeking"
    BUFFERING = "buffering"
    ERROR = "error"


class AnalysisType(Enum):
    """Types of session analysis."""
    QUALITY = "quality"
    CONVERSATION_FLOW = "conversation_flow"
    SPEAKER_ANALYTICS = "speaker_analytics"
    PERFORMANCE = "performance"
    SENTIMENT = "sentiment"
    KEYWORD_ANALYSIS = "keyword_analysis"
    INTERRUPTION_ANALYSIS = "interruption_analysis"


@dataclass
class PlaybackPosition:
    """Current playback position."""
    timestamp: float
    audio_position: Optional[float] = None
    transcript_segment_id: Optional[str] = None
    percentage: float = 0.0


@dataclass
class ConversationFlowMetrics:
    """Conversation flow analysis metrics."""
    turn_taking_efficiency: float
    interruption_rate: float
    silence_ratio: float
    speaker_balance: Dict[str, float]
    response_latency_avg: float
    response_latency_p95: float
    conversation_momentum: List[float]
    engagement_score: float


@dataclass
class SpeakerAnalytics:
    """Speaker-specific analytics."""
    speaker_id: str
    total_speaking_time: float
    word_count: int
    average_speaking_speed: float  # words per minute
    confidence_score: float
    sentiment_distribution: Dict[str, float]
    interruption_count: int
    turn_count: int
    vocabulary_richness: float


class SessionPlaybackManager:
    """
    Comprehensive session playback and analysis manager.
    
    Provides audio playback, transcript synchronization, and real-time
    analysis of recorded voice agent sessions.
    """
    
    def __init__(
        self,
        storage_path: str = "./recordings",
        enable_real_time_analysis: bool = True,
        cache_enabled: bool = True,
        logger: Optional[StructuredLogger] = None
    ):
        """
        Initialize session playback manager.
        
        Args:
            storage_path: Path to recorded sessions
            enable_real_time_analysis: Enable real-time analysis during playback
            cache_enabled: Enable analysis result caching
            logger: Optional logger instance
        """
        self.storage_path = Path(storage_path)
        self.enable_real_time_analysis = enable_real_time_analysis
        self.cache_enabled = cache_enabled
        self.logger = logger or StructuredLogger(__name__, "session_playback")
        
        # Active playback sessions
        self.active_players: Dict[str, 'SessionPlayer'] = {}
        
        # Analysis cache
        self.analysis_cache: Dict[str, Dict[str, Any]] = {}
        
        # Playback callbacks
        self.callbacks: Dict[str, List[Callable]] = {
            "on_playback_start": [],
            "on_playback_stop": [],
            "on_playback_pause": [],
            "on_playback_position": [],
            "on_analysis_update": []
        }
        
        self.logger.info(
            "Session playback manager initialized",
            extra_data={
                "storage_path": str(self.storage_path),
                "real_time_analysis": self.enable_real_time_analysis,
                "cache_enabled": self.cache_enabled
            }
        )
    
    async def load_session(self, session_id: str) -> Optional[SessionMetadata]:
        """
        Load session metadata from storage.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session metadata if found
        """
        metadata_file = self.storage_path / session_id / f"{session_id}_metadata.json"
        
        if not metadata_file.exists():
            # Try direct path
            metadata_file = self.storage_path / f"{session_id}_metadata.json"
        
        if not metadata_file.exists():
            self.logger.warning(f"Session metadata not found: {session_id}")
            return None
        
        try:
            with open(metadata_file, 'r') as f:
                metadata_dict = json.load(f)
            
            # Convert back to SessionMetadata
            metadata = self._dict_to_session_metadata(metadata_dict)
            
            self.logger.info(
                f"Session loaded: {session_id}",
                extra_data={
                    "session_id": session_id,
                    "duration": metadata.duration,
                    "status": metadata.status.value if metadata.status else None
                }
            )
            
            return metadata
            
        except Exception as e:
            self.logger.exception(
                f"Error loading session metadata: {session_id}",
                extra_data={"error": str(e)}
            )
            return None
    
    async def create_player(
        self,
        session_id: str,
        enable_audio: bool = True,
        enable_transcript: bool = True,
        playback_speed: float = 1.0
    ) -> Optional['SessionPlayer']:
        """
        Create a session player for playback.
        
        Args:
            session_id: Session identifier
            enable_audio: Enable audio playback
            enable_transcript: Enable transcript playback
            playback_speed: Playback speed multiplier
            
        Returns:
            SessionPlayer instance if successful
        """
        if session_id in self.active_players:
            self.logger.warning(f"Player already exists for session: {session_id}")
            return self.active_players[session_id]
        
        # Load session metadata
        metadata = await self.load_session(session_id)
        if not metadata:
            return None
        
        # Create player
        player = SessionPlayer(
            session_id=session_id,
            metadata=metadata,
            storage_path=self.storage_path,
            enable_audio=enable_audio,
            enable_transcript=enable_transcript,
            playback_speed=playback_speed,
            real_time_analysis=self.enable_real_time_analysis,
            manager=self,
            logger=self.logger
        )
        
        # Initialize player
        await player.initialize()
        
        self.active_players[session_id] = player
        
        self.logger.info(
            f"Session player created: {session_id}",
            extra_data={
                "session_id": session_id,
                "enable_audio": enable_audio,
                "enable_transcript": enable_transcript,
                "playback_speed": playback_speed
            }
        )
        
        return player
    
    async def analyze_session(
        self,
        session_id: str,
        analysis_types: List[AnalysisType],
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of a session.
        
        Args:
            session_id: Session identifier
            analysis_types: Types of analysis to perform
            force_refresh: Force refresh of cached results
            
        Returns:
            Analysis results dictionary
        """
        # Check cache first
        cache_key = f"{session_id}_{hash(tuple(analysis_types))}"
        if self.cache_enabled and not force_refresh and cache_key in self.analysis_cache:
            self.logger.info(f"Returning cached analysis for session: {session_id}")
            return self.analysis_cache[cache_key]
        
        # Load session data
        metadata = await self.load_session(session_id)
        if not metadata:
            raise ValueError(f"Session not found: {session_id}")
        
        audio_data = await self._load_audio_data(session_id)
        transcript_data = await self._load_transcript_data(session_id)
        quality_data = await self._load_quality_data(session_id)
        
        # Perform analysis
        analyzer = SessionAnalyzer(
            session_id=session_id,
            metadata=metadata,
            audio_data=audio_data,
            transcript_data=transcript_data,
            quality_data=quality_data,
            logger=self.logger
        )
        
        results = {}
        
        for analysis_type in analysis_types:
            try:
                if analysis_type == AnalysisType.QUALITY:
                    results["quality"] = await analyzer.analyze_quality()
                elif analysis_type == AnalysisType.CONVERSATION_FLOW:
                    results["conversation_flow"] = await analyzer.analyze_conversation_flow()
                elif analysis_type == AnalysisType.SPEAKER_ANALYTICS:
                    results["speaker_analytics"] = await analyzer.analyze_speakers()
                elif analysis_type == AnalysisType.PERFORMANCE:
                    results["performance"] = await analyzer.analyze_performance()
                elif analysis_type == AnalysisType.SENTIMENT:
                    results["sentiment"] = await analyzer.analyze_sentiment()
                elif analysis_type == AnalysisType.KEYWORD_ANALYSIS:
                    results["keywords"] = await analyzer.analyze_keywords()
                elif analysis_type == AnalysisType.INTERRUPTION_ANALYSIS:
                    results["interruptions"] = await analyzer.analyze_interruptions()
                    
            except Exception as e:
                self.logger.exception(
                    f"Error in {analysis_type.value} analysis for session {session_id}",
                    extra_data={"error": str(e)}
                )
                results[analysis_type.value] = {"error": str(e)}
        
        # Cache results
        if self.cache_enabled:
            self.analysis_cache[cache_key] = results
        
        self.logger.info(
            f"Session analysis completed: {session_id}",
            extra_data={
                "session_id": session_id,
                "analysis_types": [t.value for t in analysis_types],
                "results_keys": list(results.keys())
            }
        )
        
        return results
    
    async def export_analysis_report(
        self,
        session_id: str,
        analysis_results: Dict[str, Any],
        format: str = "json",
        output_path: Optional[str] = None
    ) -> str:
        """
        Export analysis report in specified format.
        
        Args:
            session_id: Session identifier
            analysis_results: Analysis results to export
            format: Export format (json, html, pdf)
            output_path: Output file path
            
        Returns:
            Path to exported file
        """
        exporter = AnalysisReportExporter(
            session_id=session_id,
            analysis_results=analysis_results,
            logger=self.logger
        )
        
        return await exporter.export(format, output_path)
    
    def add_callback(self, event: str, callback: Callable):
        """Add a callback for playback events."""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
    
    def remove_callback(self, event: str, callback: Callable):
        """Remove a callback for playback events."""
        if event in self.callbacks and callback in self.callbacks[event]:
            self.callbacks[event].remove(callback)
    
    async def _trigger_callbacks(self, event: str, *args):
        """Trigger callbacks for an event."""
        for callback in self.callbacks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(*args)
                else:
                    callback(*args)
            except Exception as e:
                self.logger.exception(
                    f"Error in playback callback: {event}",
                    extra_data={"error": str(e), "event": event}
                )
    
    async def _load_audio_data(self, session_id: str) -> Optional[List[AudioChunk]]:
        """Load audio data for a session."""
        audio_file = self._find_session_file(session_id, "audio")
        if not audio_file:
            return None
        
        try:
            # This is a simplified implementation
            # In reality, you would parse the audio file and create AudioChunk objects
            return []
        except Exception as e:
            self.logger.exception(f"Error loading audio data for session {session_id}")
            return None
    
    async def _load_transcript_data(self, session_id: str) -> Optional[List[TranscriptSegment]]:
        """Load transcript data for a session."""
        transcript_file = self._find_session_file(session_id, "transcript.json")
        if not transcript_file:
            return None
        
        try:
            # Handle compressed files
            if transcript_file.suffix == '.gz':
                with gzip.open(transcript_file, 'rt') as f:
                    data = json.load(f)
            else:
                with open(transcript_file, 'r') as f:
                    data = json.load(f)
            
            segments = []
            for segment_data in data.get("segments", []):
                segment = TranscriptSegment(**segment_data)
                segments.append(segment)
            
            return segments
            
        except Exception as e:
            self.logger.exception(f"Error loading transcript data for session {session_id}")
            return None
    
    async def _load_quality_data(self, session_id: str) -> Optional[QualityMetrics]:
        """Load quality metrics for a session."""
        quality_file = self._find_session_file(session_id, "quality.json")
        if not quality_file:
            return None
        
        try:
            # Handle compressed files
            if quality_file.suffix == '.gz':
                with gzip.open(quality_file, 'rt') as f:
                    data = json.load(f)
            else:
                with open(quality_file, 'r') as f:
                    data = json.load(f)
            
            return QualityMetrics(**data)
            
        except Exception as e:
            self.logger.exception(f"Error loading quality data for session {session_id}")
            return None
    
    def _find_session_file(self, session_id: str, file_suffix: str) -> Optional[Path]:
        """Find a session file with given suffix."""
        # Try session directory first
        session_dir = self.storage_path / session_id
        if session_dir.exists():
            file_path = session_dir / f"{session_id}_{file_suffix}"
            if file_path.exists():
                return file_path
            # Try compressed version
            file_path_gz = session_dir / f"{session_id}_{file_suffix}.gz"
            if file_path_gz.exists():
                return file_path_gz
        
        # Try direct path
        file_path = self.storage_path / f"{session_id}_{file_suffix}"
        if file_path.exists():
            return file_path
        
        # Try compressed version
        file_path_gz = self.storage_path / f"{session_id}_{file_suffix}.gz"
        if file_path_gz.exists():
            return file_path_gz
        
        return None
    
    def _dict_to_session_metadata(self, data: Dict[str, Any]) -> SessionMetadata:
        """Convert dictionary to SessionMetadata object."""
        # This is a simplified conversion
        # In reality, you would handle all the enum conversions properly
        from .session_recording import SessionStatus, PrivacyLevel, AudioFormat
        
        metadata = SessionMetadata(
            session_id=data["session_id"],
            user_id=data.get("user_id"),
            agent_id=data.get("agent_id"),
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
            duration=data.get("duration"),
            status=SessionStatus(data.get("status", "completed")),
            privacy_level=PrivacyLevel(data.get("privacy_level", "full")),
            tags=data.get("tags", []),
            custom_metadata=data.get("custom_metadata", {}),
            audio_format=AudioFormat(data.get("audio_format", "wav")),
            sample_rate=data.get("sample_rate", 44100),
            channels=data.get("channels", 1),
            bit_depth=data.get("bit_depth", 16),
            storage_path=data.get("storage_path"),
            compressed=data.get("compressed", False),
            encrypted=data.get("encrypted", False),
            retention_until=datetime.fromisoformat(data["retention_until"]) if data.get("retention_until") else None
        )
        
        # Handle quality metrics if present
        if "quality_metrics" in data and data["quality_metrics"]:
            metadata.quality_metrics = QualityMetrics(**data["quality_metrics"])
        
        return metadata


class SessionPlayer:
    """
    Individual session player for audio and transcript playback.
    
    Provides synchronized playback of audio and transcript with
    real-time analysis capabilities.
    """
    
    def __init__(
        self,
        session_id: str,
        metadata: SessionMetadata,
        storage_path: Path,
        enable_audio: bool,
        enable_transcript: bool,
        playback_speed: float,
        real_time_analysis: bool,
        manager: SessionPlaybackManager,
        logger: StructuredLogger
    ):
        self.session_id = session_id
        self.metadata = metadata
        self.storage_path = storage_path
        self.enable_audio = enable_audio
        self.enable_transcript = enable_transcript
        self.playback_speed = playback_speed
        self.real_time_analysis = real_time_analysis
        self.manager = manager
        self.logger = logger
        
        # Playback state
        self.state = PlaybackState.STOPPED
        self.position = PlaybackPosition(timestamp=0.0)
        self.start_time = 0.0
        self.pause_time = 0.0
        
        # Data
        self.audio_data: Optional[bytes] = None
        self.transcript_segments: List[TranscriptSegment] = []
        self.quality_metrics: Optional[QualityMetrics] = None
        
        # Playback control
        self.playback_task: Optional[asyncio.Task] = None
        self.analysis_task: Optional[asyncio.Task] = None
        self.stop_event = asyncio.Event()
        
        # Audio playback (placeholder - would use audio library)
        self.audio_player = None
    
    async def initialize(self):
        """Initialize the session player."""
        # Load audio data if enabled
        if self.enable_audio:
            await self._load_audio_data()
        
        # Load transcript data if enabled
        if self.enable_transcript:
            await self._load_transcript_data()
        
        # Load quality metrics
        await self._load_quality_metrics()
        
        self.logger.info(
            f"Session player initialized: {self.session_id}",
            extra_data={
                "session_id": self.session_id,
                "has_audio": self.audio_data is not None,
                "transcript_segments": len(self.transcript_segments),
                "has_quality": self.quality_metrics is not None
            }
        )
    
    async def play(self, start_position: float = 0.0):
        """Start playback from specified position."""
        if self.state == PlaybackState.PLAYING:
            return
        
        self.position.timestamp = start_position
        self.start_time = time.time() - start_position
        self.state = PlaybackState.PLAYING
        self.stop_event.clear()
        
        # Start playback task
        self.playback_task = asyncio.create_task(self._playback_loop())
        
        # Start analysis task if enabled
        if self.real_time_analysis:
            self.analysis_task = asyncio.create_task(self._analysis_loop())
        
        # Trigger callbacks
        await self.manager._trigger_callbacks("on_playback_start", self.session_id, start_position)
        
        self.logger.info(
            f"Playback started: {self.session_id}",
            extra_data={"session_id": self.session_id, "start_position": start_position}
        )
    
    async def pause(self):
        """Pause playback."""
        if self.state != PlaybackState.PLAYING:
            return
        
        self.state = PlaybackState.PAUSED
        self.pause_time = time.time()
        
        # Trigger callbacks
        await self.manager._trigger_callbacks("on_playback_pause", self.session_id, self.position.timestamp)
        
        self.logger.info(f"Playback paused: {self.session_id}")
    
    async def resume(self):
        """Resume playback."""
        if self.state != PlaybackState.PAUSED:
            return
        
        # Adjust start time for pause duration
        pause_duration = time.time() - self.pause_time
        self.start_time += pause_duration
        
        self.state = PlaybackState.PLAYING
        
        self.logger.info(f"Playback resumed: {self.session_id}")
    
    async def stop(self):
        """Stop playback."""
        self.state = PlaybackState.STOPPED
        self.stop_event.set()
        
        # Cancel tasks
        if self.playback_task:
            self.playback_task.cancel()
            try:
                await self.playback_task
            except asyncio.CancelledError:
                pass
        
        if self.analysis_task:
            self.analysis_task.cancel()
            try:
                await self.analysis_task
            except asyncio.CancelledError:
                pass
        
        # Trigger callbacks
        await self.manager._trigger_callbacks("on_playback_stop", self.session_id)
        
        self.logger.info(f"Playback stopped: {self.session_id}")
    
    async def seek(self, position: float):
        """Seek to specific position."""
        if position < 0 or (self.metadata.duration and position > self.metadata.duration):
            raise ValueError("Invalid seek position")
        
        was_playing = self.state == PlaybackState.PLAYING
        
        if was_playing:
            await self.pause()
        
        self.position.timestamp = position
        self.start_time = time.time() - position
        
        if was_playing:
            await self.resume()
        
        self.logger.info(
            f"Seeked to position: {position}",
            extra_data={"session_id": self.session_id, "position": position}
        )
    
    def get_current_position(self) -> PlaybackPosition:
        """Get current playback position."""
        if self.state == PlaybackState.PLAYING:
            current_time = time.time()
            self.position.timestamp = (current_time - self.start_time) * self.playback_speed
            
            # Calculate percentage
            if self.metadata.duration:
                self.position.percentage = min(100.0, (self.position.timestamp / self.metadata.duration) * 100)
            
            # Find current transcript segment
            self.position.transcript_segment_id = self._find_current_transcript_segment()
        
        return self.position
    
    async def _playback_loop(self):
        """Main playback loop."""
        try:
            while not self.stop_event.is_set() and self.state == PlaybackState.PLAYING:
                current_position = self.get_current_position()
                
                # Check if playback is complete
                if self.metadata.duration and current_position.timestamp >= self.metadata.duration:
                    await self.stop()
                    break
                
                # Trigger position callbacks
                await self.manager._trigger_callbacks("on_playback_position", self.session_id, current_position)
                
                await asyncio.sleep(0.1)  # Update every 100ms
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.exception(f"Error in playback loop: {e}")
            self.state = PlaybackState.ERROR
    
    async def _analysis_loop(self):
        """Real-time analysis loop during playback."""
        try:
            while not self.stop_event.is_set() and self.state == PlaybackState.PLAYING:
                current_position = self.get_current_position()
                
                # Perform real-time analysis
                analysis_data = await self._get_realtime_analysis(current_position.timestamp)
                
                if analysis_data:
                    await self.manager._trigger_callbacks("on_analysis_update", self.session_id, analysis_data)
                
                await asyncio.sleep(1.0)  # Analyze every second
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.exception(f"Error in analysis loop: {e}")
    
    async def _load_audio_data(self):
        """Load audio data for playback."""
        audio_file = self.manager._find_session_file(self.session_id, f"audio.{self.metadata.audio_format.value}")
        if not audio_file:
            self.logger.warning(f"Audio file not found for session: {self.session_id}")
            return
        
        try:
            # Handle compressed files
            if audio_file.suffix == '.gz':
                with gzip.open(audio_file, 'rb') as f:
                    self.audio_data = f.read()
            else:
                with open(audio_file, 'rb') as f:
                    self.audio_data = f.read()
            
            self.logger.info(f"Audio data loaded: {len(self.audio_data)} bytes")
            
        except Exception as e:
            self.logger.exception(f"Error loading audio data: {e}")
    
    async def _load_transcript_data(self):
        """Load transcript data for playback."""
        transcript_file = self.manager._find_session_file(self.session_id, "transcript.json")
        if not transcript_file:
            self.logger.warning(f"Transcript file not found for session: {self.session_id}")
            return
        
        try:
            # Handle compressed files
            if transcript_file.suffix == '.gz':
                with gzip.open(transcript_file, 'rt') as f:
                    data = json.load(f)
            else:
                with open(transcript_file, 'r') as f:
                    data = json.load(f)
            
            for segment_data in data.get("segments", []):
                segment = TranscriptSegment(**segment_data)
                self.transcript_segments.append(segment)
            
            self.logger.info(f"Transcript loaded: {len(self.transcript_segments)} segments")
            
        except Exception as e:
            self.logger.exception(f"Error loading transcript data: {e}")
    
    async def _load_quality_metrics(self):
        """Load quality metrics for analysis."""
        quality_file = self.manager._find_session_file(self.session_id, "quality.json")
        if not quality_file:
            return
        
        try:
            # Handle compressed files
            if quality_file.suffix == '.gz':
                with gzip.open(quality_file, 'rt') as f:
                    data = json.load(f)
            else:
                with open(quality_file, 'r') as f:
                    data = json.load(f)
            
            self.quality_metrics = QualityMetrics(**data)
            
        except Exception as e:
            self.logger.exception(f"Error loading quality metrics: {e}")
    
    def _find_current_transcript_segment(self) -> Optional[str]:
        """Find the transcript segment at current position."""
        for segment in self.transcript_segments:
            if segment.timestamp <= self.position.timestamp <= (segment.end_timestamp or segment.timestamp + 5):
                return segment.id
        return None
    
    async def _get_realtime_analysis(self, timestamp: float) -> Optional[Dict[str, Any]]:
        """Get real-time analysis data for current position."""
        # This would perform real-time analysis based on current position
        # For now, return placeholder data
        
        current_segment = None
        for segment in self.transcript_segments:
            if segment.timestamp <= timestamp <= (segment.end_timestamp or segment.timestamp + 5):
                current_segment = segment
                break
        
        if current_segment:
            return {
                "timestamp": timestamp,
                "current_segment": current_segment.to_dict(),
                "speaker": current_segment.speaker_id,
                "confidence": current_segment.confidence
            }
        
        return None


class SessionAnalyzer:
    """
    Comprehensive session analyzer for recorded voice sessions.
    
    Provides detailed analysis of conversation quality, speaker behavior,
    performance metrics, and other insights.
    """
    
    def __init__(
        self,
        session_id: str,
        metadata: SessionMetadata,
        audio_data: Optional[List[AudioChunk]],
        transcript_data: Optional[List[TranscriptSegment]],
        quality_data: Optional[QualityMetrics],
        logger: StructuredLogger
    ):
        self.session_id = session_id
        self.metadata = metadata
        self.audio_data = audio_data or []
        self.transcript_data = transcript_data or []
        self.quality_data = quality_data
        self.logger = logger
    
    async def analyze_quality(self) -> Dict[str, Any]:
        """Analyze session quality metrics."""
        quality_analysis = {
            "overall_score": 0.0,
            "audio_quality": {},
            "transcript_quality": {},
            "conversation_quality": {}
        }
        
        # Audio quality analysis
        if self.quality_data:
            quality_analysis["audio_quality"] = {
                "score": self.quality_data.audio_quality_score or 0.0,
                "background_noise": self.quality_data.background_noise_level or 0.0,
                "volume_variance": self.quality_data.volume_variance or 0.0
            }
        
        # Transcript quality analysis
        if self.transcript_data:
            confidences = [s.confidence for s in self.transcript_data if s.confidence]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            quality_analysis["transcript_quality"] = {
                "avg_confidence": avg_confidence,
                "accuracy_score": self.quality_data.transcript_accuracy_score if self.quality_data else avg_confidence,
                "total_segments": len(self.transcript_data),
                "low_confidence_segments": len([s for s in self.transcript_data if s.confidence and s.confidence < 0.7])
            }
        
        # Conversation quality analysis
        if self.quality_data:
            quality_analysis["conversation_quality"] = {
                "interruption_count": self.quality_data.interruption_count,
                "silence_ratio": self.quality_data.silence_duration / (self.quality_data.silence_duration + self.quality_data.speech_duration) if (self.quality_data.silence_duration + self.quality_data.speech_duration) > 0 else 0,
                "speech_ratio": self.quality_data.speech_duration / (self.quality_data.silence_duration + self.quality_data.speech_duration) if (self.quality_data.silence_duration + self.quality_data.speech_duration) > 0 else 0,
                "speaker_switches": self.quality_data.speaker_switch_count
            }
        
        # Calculate overall score
        scores = []
        if quality_analysis["audio_quality"].get("score"):
            scores.append(quality_analysis["audio_quality"]["score"])
        if quality_analysis["transcript_quality"].get("avg_confidence"):
            scores.append(quality_analysis["transcript_quality"]["avg_confidence"])
        
        if scores:
            quality_analysis["overall_score"] = sum(scores) / len(scores)
        
        return quality_analysis
    
    async def analyze_conversation_flow(self) -> ConversationFlowMetrics:
        """Analyze conversation flow patterns."""
        if not self.transcript_data:
            return ConversationFlowMetrics(
                turn_taking_efficiency=0.0,
                interruption_rate=0.0,
                silence_ratio=0.0,
                speaker_balance={},
                response_latency_avg=0.0,
                response_latency_p95=0.0,
                conversation_momentum=[],
                engagement_score=0.0
            )
        
        # Calculate turn-taking efficiency
        turn_switches = 0
        response_latencies = []
        current_speaker = None
        
        for i, segment in enumerate(self.transcript_data):
            if segment.speaker_id != current_speaker:
                turn_switches += 1
                current_speaker = segment.speaker_id
                
                # Calculate response latency
                if i > 0:
                    latency = segment.timestamp - (self.transcript_data[i-1].end_timestamp or self.transcript_data[i-1].timestamp)
                    response_latencies.append(latency)
        
        # Speaker balance
        speaker_time = {}
        for segment in self.transcript_data:
            if segment.speaker_id:
                duration = (segment.end_timestamp or segment.timestamp + 2) - segment.timestamp
                speaker_time[segment.speaker_id] = speaker_time.get(segment.speaker_id, 0) + duration
        
        total_time = sum(speaker_time.values())
        speaker_balance = {speaker: time / total_time for speaker, time in speaker_time.items()} if total_time > 0 else {}
        
        # Calculate metrics
        turn_taking_efficiency = turn_switches / len(self.transcript_data) if self.transcript_data else 0
        interruption_rate = self.quality_data.interruption_count / len(self.transcript_data) if self.quality_data and self.transcript_data else 0
        silence_ratio = self.quality_data.silence_duration / (self.quality_data.silence_duration + self.quality_data.speech_duration) if self.quality_data and (self.quality_data.silence_duration + self.quality_data.speech_duration) > 0 else 0
        
        response_latency_avg = sum(response_latencies) / len(response_latencies) if response_latencies else 0
        response_latency_p95 = sorted(response_latencies)[int(0.95 * len(response_latencies))] if response_latencies else 0
        
        # Conversation momentum (simplified)
        conversation_momentum = [len(segment.text.split()) for segment in self.transcript_data]
        
        # Engagement score (simplified)
        engagement_score = min(1.0, turn_taking_efficiency * 2) * (1 - silence_ratio) * (1 - interruption_rate)
        
        return ConversationFlowMetrics(
            turn_taking_efficiency=turn_taking_efficiency,
            interruption_rate=interruption_rate,
            silence_ratio=silence_ratio,
            speaker_balance=speaker_balance,
            response_latency_avg=response_latency_avg,
            response_latency_p95=response_latency_p95,
            conversation_momentum=conversation_momentum,
            engagement_score=engagement_score
        )
    
    async def analyze_speakers(self) -> List[SpeakerAnalytics]:
        """Analyze individual speaker behavior."""
        if not self.transcript_data:
            return []
        
        speaker_data = {}
        
        # Collect speaker data
        for segment in self.transcript_data:
            if not segment.speaker_id:
                continue
            
            if segment.speaker_id not in speaker_data:
                speaker_data[segment.speaker_id] = {
                    "segments": [],
                    "word_count": 0,
                    "speaking_time": 0.0,
                    "confidences": [],
                    "turns": 0
                }
            
            data = speaker_data[segment.speaker_id]
            data["segments"].append(segment)
            data["word_count"] += len(segment.text.split())
            data["speaking_time"] += (segment.end_timestamp or segment.timestamp + 2) - segment.timestamp
            
            if segment.confidence:
                data["confidences"].append(segment.confidence)
        
        # Calculate speaker analytics
        speaker_analytics = []
        
        for speaker_id, data in speaker_data.items():
            # Calculate speaking speed (words per minute)
            speaking_speed = (data["word_count"] / data["speaking_time"]) * 60 if data["speaking_time"] > 0 else 0
            
            # Calculate confidence score
            confidence_score = sum(data["confidences"]) / len(data["confidences"]) if data["confidences"] else 0
            
            # Calculate vocabulary richness (unique words / total words)
            all_words = []
            for segment in data["segments"]:
                all_words.extend(segment.text.lower().split())
            vocabulary_richness = len(set(all_words)) / len(all_words) if all_words else 0
            
            analytics = SpeakerAnalytics(
                speaker_id=speaker_id,
                total_speaking_time=data["speaking_time"],
                word_count=data["word_count"],
                average_speaking_speed=speaking_speed,
                confidence_score=confidence_score,
                sentiment_distribution={},  # Would be calculated if sentiment analysis is enabled
                interruption_count=0,  # Would be calculated from interruption analysis
                turn_count=len(data["segments"]),
                vocabulary_richness=vocabulary_richness
            )
            
            speaker_analytics.append(analytics)
        
        return speaker_analytics
    
    async def analyze_performance(self) -> Dict[str, Any]:
        """Analyze system performance metrics."""
        performance_analysis = {}
        
        if self.quality_data:
            performance_analysis = {
                "latency": {
                    "avg": self.quality_data.latency_avg or 0,
                    "p95": self.quality_data.latency_p95 or 0
                },
                "quality_scores": {
                    "audio": self.quality_data.audio_quality_score or 0,
                    "transcript": self.quality_data.transcript_accuracy_score or 0
                },
                "session_metrics": {
                    "duration": self.metadata.duration or 0,
                    "interruptions": self.quality_data.interruption_count,
                    "speaker_switches": self.quality_data.speaker_switch_count
                }
            }
        
        return performance_analysis
    
    async def analyze_sentiment(self) -> Dict[str, Any]:
        """Analyze sentiment throughout the session."""
        # Placeholder implementation
        return {
            "overall_sentiment": "neutral",
            "sentiment_timeline": [],
            "speaker_sentiments": {}
        }
    
    async def analyze_keywords(self) -> Dict[str, Any]:
        """Analyze keywords and topics."""
        if not self.transcript_data:
            return {"keywords": [], "topics": []}
        
        # Simple keyword extraction
        all_text = " ".join(segment.text for segment in self.transcript_data)
        words = all_text.lower().split()
        
        # Filter common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]
        
        # Count frequency
        keyword_freq = {}
        for keyword in keywords:
            keyword_freq[keyword] = keyword_freq.get(keyword, 0) + 1
        
        # Get top keywords
        top_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)[:20]
        
        return {
            "keywords": top_keywords,
            "total_words": len(words),
            "unique_words": len(set(words)),
            "vocabulary_richness": len(set(words)) / len(words) if words else 0
        }
    
    async def analyze_interruptions(self) -> Dict[str, Any]:
        """Analyze interruption patterns."""
        # Placeholder implementation
        return {
            "total_interruptions": self.quality_data.interruption_count if self.quality_data else 0,
            "interruption_rate": 0.0,
            "interruption_timeline": [],
            "speaker_interruptions": {}
        }


class AnalysisReportExporter:
    """Export analysis reports in various formats."""
    
    def __init__(
        self,
        session_id: str,
        analysis_results: Dict[str, Any],
        logger: StructuredLogger
    ):
        self.session_id = session_id
        self.analysis_results = analysis_results
        self.logger = logger
    
    async def export(self, format: str, output_path: Optional[str] = None) -> str:
        """Export analysis report in specified format."""
        if format == "json":
            return await self._export_json(output_path)
        elif format == "html":
            return await self._export_html(output_path)
        elif format == "pdf":
            return await self._export_pdf(output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    async def _export_json(self, output_path: Optional[str]) -> str:
        """Export as JSON format."""
        data = {
            "session_id": self.session_id,
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "analysis_results": self.analysis_results
        }
        
        content = json.dumps(data, indent=2, default=str)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(content)
            return output_path
        
        return content
    
    async def _export_html(self, output_path: Optional[str]) -> str:
        """Export as HTML format."""
        # Placeholder HTML template
        html_content = f"""
        <html>
        <head>
            <title>Session Analysis Report - {self.session_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f5f5f5; padding: 20px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; border: 1px solid #ddd; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Session Analysis Report</h1>
                <p>Session ID: {self.session_id}</p>
                <p>Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
            </div>
            
            <div class="section">
                <h2>Analysis Results</h2>
                <pre>{json.dumps(self.analysis_results, indent=2, default=str)}</pre>
            </div>
        </body>
        </html>
        """
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(html_content)
            return output_path
        
        return html_content
    
    async def _export_pdf(self, output_path: Optional[str]) -> str:
        """Export as PDF format."""
        # Placeholder - would use a PDF library like reportlab
        # For now, just create a text file
        text_content = f"""
Session Analysis Report
======================

Session ID: {self.session_id}
Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}

Analysis Results:
{json.dumps(self.analysis_results, indent=2, default=str)}
        """
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(text_content)
            return output_path
        
        return text_content