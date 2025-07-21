"""
Session Recording and Transcript Management System

This module provides comprehensive session recording capabilities for voice agents,
including audio recording, real-time transcript generation, session metadata tracking,
and quality analytics for production voice AI deployments.
"""

import asyncio
import json
import time
import uuid
import wave
import io
import threading
from typing import Dict, List, Optional, Any, Union, Callable, AsyncIterator
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
import logging
import hashlib
import gzip
import base64

from .structured_logging import StructuredLogger, CorrelationIdManager


class SessionStatus(Enum):
    """Status of a recording session."""
    PENDING = "pending"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"


class AudioFormat(Enum):
    """Supported audio recording formats."""
    WAV = "wav"
    MP3 = "mp3"
    OPUS = "opus"
    FLAC = "flac"


class TranscriptFormat(Enum):
    """Supported transcript formats."""
    JSON = "json"
    TEXT = "text"
    SRT = "srt"
    VTT = "vtt"
    DOCX = "docx"


class PrivacyLevel(Enum):
    """Privacy levels for session recording."""
    FULL = "full"  # Record everything
    AUDIO_ONLY = "audio_only"  # No transcript recording
    TRANSCRIPT_ONLY = "transcript_only"  # No audio recording
    METADATA_ONLY = "metadata_only"  # Only session metadata
    DISABLED = "disabled"  # No recording


@dataclass
class AudioChunk:
    """Represents a chunk of audio data."""
    timestamp: float
    data: bytes
    sample_rate: int
    channels: int
    bit_depth: int
    format: AudioFormat
    speaker_id: Optional[str] = None
    volume_level: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "data_size": len(self.data),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "bit_depth": self.bit_depth,
            "format": self.format.value,
            "speaker_id": self.speaker_id,
            "volume_level": self.volume_level
        }


@dataclass
class TranscriptSegment:
    """Represents a segment of transcript."""
    id: str
    timestamp: float
    end_timestamp: Optional[float]
    text: str
    speaker_id: Optional[str] = None
    confidence: Optional[float] = None
    language: Optional[str] = None
    is_final: bool = False
    words: Optional[List[Dict[str, Any]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class QualityMetrics:
    """Quality metrics for a recording session."""
    audio_quality_score: Optional[float] = None
    transcript_accuracy_score: Optional[float] = None
    latency_p95: Optional[float] = None
    latency_avg: Optional[float] = None
    interruption_count: int = 0
    silence_duration: float = 0.0
    speech_duration: float = 0.0
    speaker_switch_count: int = 0
    volume_variance: Optional[float] = None
    background_noise_level: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class SessionMetadata:
    """Metadata for a recording session."""
    session_id: str
    user_id: Optional[str]
    agent_id: Optional[str]
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    status: SessionStatus = SessionStatus.PENDING
    privacy_level: PrivacyLevel = PrivacyLevel.FULL
    tags: List[str] = field(default_factory=list)
    custom_metadata: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Optional[QualityMetrics] = None
    
    # Technical metadata
    audio_format: AudioFormat = AudioFormat.WAV
    sample_rate: int = 44100
    channels: int = 1
    bit_depth: int = 16
    
    # Storage metadata
    storage_path: Optional[str] = None
    compressed: bool = False
    encrypted: bool = False
    retention_until: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        data["start_time"] = self.start_time.isoformat()
        if self.end_time:
            data["end_time"] = self.end_time.isoformat()
        if self.retention_until:
            data["retention_until"] = self.retention_until.isoformat()
        # Convert enums to values
        data["status"] = self.status.value
        data["privacy_level"] = self.privacy_level.value
        data["audio_format"] = self.audio_format.value
        return data


class SessionRecordingManager:
    """
    Comprehensive session recording manager for voice agents.
    
    Handles audio recording, transcript generation, metadata tracking,
    and quality analytics for voice agent sessions.
    """
    
    def __init__(
        self,
        storage_path: str = "./recordings",
        enable_compression: bool = True,
        enable_encryption: bool = False,
        default_retention_days: int = 30,
        max_session_duration: Optional[float] = 3600,  # 1 hour
        quality_monitoring: bool = True,
        real_time_transcription: bool = True
    ):
        """
        Initialize session recording manager.
        
        Args:
            storage_path: Base path for storing recordings
            enable_compression: Whether to compress audio data
            enable_encryption: Whether to encrypt stored data
            default_retention_days: Default retention period in days
            max_session_duration: Maximum session duration in seconds
            quality_monitoring: Whether to enable quality monitoring
            real_time_transcription: Whether to enable real-time transcription
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.enable_compression = enable_compression
        self.enable_encryption = enable_encryption
        self.default_retention_days = default_retention_days
        self.max_session_duration = max_session_duration
        self.quality_monitoring = quality_monitoring
        self.real_time_transcription = real_time_transcription
        
        # Active sessions
        self.active_sessions: Dict[str, 'RecordingSession'] = {}
        
        # Session callbacks
        self.session_callbacks: Dict[str, List[Callable]] = {
            "on_session_start": [],
            "on_session_end": [],
            "on_audio_chunk": [],
            "on_transcript_segment": [],
            "on_quality_update": []
        }
        
        # Logger
        self.logger = StructuredLogger(__name__, "session_recording")
        
        # Background tasks
        self.cleanup_task: Optional[asyncio.Task] = None
        self.quality_monitor_task: Optional[asyncio.Task] = None
        
        self.logger.info(
            "Session recording manager initialized",
            extra_data={
                "storage_path": str(self.storage_path),
                "compression": self.enable_compression,
                "encryption": self.enable_encryption,
                "retention_days": self.default_retention_days,
                "quality_monitoring": self.quality_monitoring,
                "real_time_transcription": self.real_time_transcription
            }
        )
    
    async def start_session(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        privacy_level: PrivacyLevel = PrivacyLevel.FULL,
        audio_format: AudioFormat = AudioFormat.WAV,
        sample_rate: int = 44100,
        channels: int = 1,
        bit_depth: int = 16,
        tags: Optional[List[str]] = None,
        custom_metadata: Optional[Dict[str, Any]] = None,
        retention_days: Optional[int] = None
    ) -> str:
        """
        Start a new recording session.
        
        Args:
            user_id: User identifier
            agent_id: Agent identifier
            privacy_level: Privacy level for recording
            audio_format: Audio format for recording
            sample_rate: Audio sample rate
            channels: Number of audio channels
            bit_depth: Audio bit depth
            tags: Optional tags for the session
            custom_metadata: Custom metadata for the session
            retention_days: Custom retention period
            
        Returns:
            Session ID
        """
        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        
        # Calculate retention date
        retention_days = retention_days or self.default_retention_days
        retention_until = datetime.now(timezone.utc) + timedelta(days=retention_days)
        
        # Create session metadata
        metadata = SessionMetadata(
            session_id=session_id,
            user_id=user_id,
            agent_id=agent_id,
            start_time=datetime.now(timezone.utc),
            privacy_level=privacy_level,
            tags=tags or [],
            custom_metadata=custom_metadata or {},
            audio_format=audio_format,
            sample_rate=sample_rate,
            channels=channels,
            bit_depth=bit_depth,
            retention_until=retention_until,
            status=SessionStatus.ACTIVE
        )
        
        # Create recording session
        session = RecordingSession(
            metadata=metadata,
            storage_path=self.storage_path,
            enable_compression=self.enable_compression,
            enable_encryption=self.enable_encryption,
            quality_monitoring=self.quality_monitoring,
            real_time_transcription=self.real_time_transcription,
            logger=self.logger
        )
        
        # Initialize session
        await session.initialize()
        
        # Store active session
        self.active_sessions[session_id] = session
        
        # Set correlation context
        CorrelationIdManager.set_session_id(session_id)
        if user_id:
            CorrelationIdManager.set_user_id(user_id)
        
        # Trigger callbacks
        await self._trigger_callbacks("on_session_start", session_id, metadata)
        
        self.logger.info(
            f"Recording session started: {session_id}",
            extra_data={
                "session_id": session_id,
                "user_id": user_id,
                "agent_id": agent_id,
                "privacy_level": privacy_level.value,
                "audio_format": audio_format.value,
                "sample_rate": sample_rate,
                "retention_until": retention_until.isoformat()
            }
        )
        
        return session_id
    
    async def record_audio_chunk(
        self,
        session_id: str,
        audio_data: bytes,
        timestamp: Optional[float] = None,
        speaker_id: Optional[str] = None
    ):
        """
        Record an audio chunk for a session.
        
        Args:
            session_id: Session identifier
            audio_data: Audio data bytes
            timestamp: Timestamp for the audio chunk
            speaker_id: Optional speaker identifier
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found or not active")
        
        session = self.active_sessions[session_id]
        
        if session.metadata.privacy_level in [PrivacyLevel.TRANSCRIPT_ONLY, PrivacyLevel.METADATA_ONLY, PrivacyLevel.DISABLED]:
            return  # Audio recording disabled for this privacy level
        
        timestamp = timestamp or time.time()
        
        # Create audio chunk
        chunk = AudioChunk(
            timestamp=timestamp,
            data=audio_data,
            sample_rate=session.metadata.sample_rate,
            channels=session.metadata.channels,
            bit_depth=session.metadata.bit_depth,
            format=session.metadata.audio_format,
            speaker_id=speaker_id,
            volume_level=self._calculate_volume_level(audio_data) if self.quality_monitoring else None
        )
        
        # Record the chunk
        await session.record_audio_chunk(chunk)
        
        # Trigger callbacks
        await self._trigger_callbacks("on_audio_chunk", session_id, chunk)
    
    async def add_transcript_segment(
        self,
        session_id: str,
        text: str,
        timestamp: Optional[float] = None,
        end_timestamp: Optional[float] = None,
        speaker_id: Optional[str] = None,
        confidence: Optional[float] = None,
        language: Optional[str] = None,
        is_final: bool = False,
        words: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Add a transcript segment to a session.
        
        Args:
            session_id: Session identifier
            text: Transcript text
            timestamp: Start timestamp
            end_timestamp: End timestamp
            speaker_id: Speaker identifier
            confidence: Confidence score
            language: Language code
            is_final: Whether this is the final transcript
            words: Word-level details
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found or not active")
        
        session = self.active_sessions[session_id]
        
        if session.metadata.privacy_level in [PrivacyLevel.AUDIO_ONLY, PrivacyLevel.METADATA_ONLY, PrivacyLevel.DISABLED]:
            return  # Transcript recording disabled for this privacy level
        
        timestamp = timestamp or time.time()
        segment_id = f"seg_{uuid.uuid4().hex[:8]}"
        
        # Create transcript segment
        segment = TranscriptSegment(
            id=segment_id,
            timestamp=timestamp,
            end_timestamp=end_timestamp,
            text=text,
            speaker_id=speaker_id,
            confidence=confidence,
            language=language,
            is_final=is_final,
            words=words
        )
        
        # Add the segment
        await session.add_transcript_segment(segment)
        
        # Trigger callbacks
        await self._trigger_callbacks("on_transcript_segment", session_id, segment)
    
    async def end_session(
        self,
        session_id: str,
        final_quality_score: Optional[float] = None
    ) -> SessionMetadata:
        """
        End a recording session.
        
        Args:
            session_id: Session identifier
            final_quality_score: Final quality score for the session
            
        Returns:
            Final session metadata
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found or not active")
        
        session = self.active_sessions[session_id]
        
        # Finalize the session
        metadata = await session.finalize(final_quality_score)
        
        # Remove from active sessions
        del self.active_sessions[session_id]
        
        # Trigger callbacks
        await self._trigger_callbacks("on_session_end", session_id, metadata)
        
        self.logger.info(
            f"Recording session ended: {session_id}",
            extra_data={
                "session_id": session_id,
                "duration": metadata.duration,
                "status": metadata.status.value,
                "quality_score": final_quality_score,
                "storage_path": metadata.storage_path
            }
        )
        
        return metadata
    
    async def pause_session(self, session_id: str):
        """Pause a recording session."""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found or not active")
        
        session = self.active_sessions[session_id]
        await session.pause()
        
        self.logger.info(f"Recording session paused: {session_id}")
    
    async def resume_session(self, session_id: str):
        """Resume a paused recording session."""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found or not active")
        
        session = self.active_sessions[session_id]
        await session.resume()
        
        self.logger.info(f"Recording session resumed: {session_id}")
    
    def get_session_status(self, session_id: str) -> Optional[SessionStatus]:
        """Get the status of a session."""
        if session_id in self.active_sessions:
            return self.active_sessions[session_id].metadata.status
        
        # Check if session exists in storage
        session_file = self.storage_path / f"{session_id}_metadata.json"
        if session_file.exists():
            try:
                with open(session_file, 'r') as f:
                    metadata = json.load(f)
                    return SessionStatus(metadata.get("status", "completed"))
            except Exception:
                pass
        
        return None
    
    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs."""
        return list(self.active_sessions.keys())
    
    def add_callback(self, event: str, callback: Callable):
        """Add a callback for session events."""
        if event in self.session_callbacks:
            self.session_callbacks[event].append(callback)
    
    def remove_callback(self, event: str, callback: Callable):
        """Remove a callback for session events."""
        if event in self.session_callbacks and callback in self.session_callbacks[event]:
            self.session_callbacks[event].remove(callback)
    
    async def _trigger_callbacks(self, event: str, *args):
        """Trigger callbacks for an event."""
        for callback in self.session_callbacks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(*args)
                else:
                    callback(*args)
            except Exception as e:
                self.logger.exception(
                    f"Error in session callback: {event}",
                    extra_data={"error": str(e), "event": event}
                )
    
    def _calculate_volume_level(self, audio_data: bytes) -> float:
        """Calculate volume level from audio data."""
        try:
            # Simple RMS calculation for volume level
            import struct
            values = struct.unpack(f"{len(audio_data)//2}h", audio_data)
            rms = (sum(v**2 for v in values) / len(values)) ** 0.5
            return min(rms / 32768.0, 1.0)  # Normalize to 0-1
        except Exception:
            return 0.0
    
    async def start_background_tasks(self):
        """Start background maintenance tasks."""
        if self.cleanup_task is None:
            self.cleanup_task = asyncio.create_task(self._cleanup_expired_sessions())
        
        if self.quality_monitoring and self.quality_monitor_task is None:
            self.quality_monitor_task = asyncio.create_task(self._monitor_session_quality())
    
    async def stop_background_tasks(self):
        """Stop background maintenance tasks."""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
            self.cleanup_task = None
        
        if self.quality_monitor_task:
            self.quality_monitor_task.cancel()
            try:
                await self.quality_monitor_task
            except asyncio.CancelledError:
                pass
            self.quality_monitor_task = None
    
    async def _cleanup_expired_sessions(self):
        """Background task to clean up expired sessions."""
        while True:
            try:
                await asyncio.sleep(3600)  # Check every hour
                
                current_time = datetime.now(timezone.utc)
                expired_sessions = []
                
                # Check stored sessions for expiration
                for session_file in self.storage_path.glob("*_metadata.json"):
                    try:
                        with open(session_file, 'r') as f:
                            metadata = json.load(f)
                            
                        retention_until = metadata.get("retention_until")
                        if retention_until:
                            retention_date = datetime.fromisoformat(retention_until.replace('Z', '+00:00'))
                            if current_time > retention_date:
                                expired_sessions.append(metadata["session_id"])
                    except Exception:
                        continue
                
                # Clean up expired sessions
                for session_id in expired_sessions:
                    await self._cleanup_session_files(session_id)
                
                if expired_sessions:
                    self.logger.info(
                        f"Cleaned up {len(expired_sessions)} expired sessions",
                        extra_data={"expired_sessions": expired_sessions}
                    )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.exception(
                    "Error in session cleanup task",
                    extra_data={"error": str(e)}
                )
    
    async def _monitor_session_quality(self):
        """Background task to monitor session quality."""
        while True:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                
                for session_id, session in self.active_sessions.items():
                    if session.quality_monitoring:
                        quality_metrics = await session.get_current_quality_metrics()
                        if quality_metrics:
                            await self._trigger_callbacks("on_quality_update", session_id, quality_metrics)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.exception(
                    "Error in quality monitoring task",
                    extra_data={"error": str(e)}
                )
    
    async def _cleanup_session_files(self, session_id: str):
        """Clean up all files for a session."""
        try:
            # Remove all session-related files
            for pattern in [
                f"{session_id}_metadata.json",
                f"{session_id}_audio.*",
                f"{session_id}_transcript.*",
                f"{session_id}_quality.*"
            ]:
                for file_path in self.storage_path.glob(pattern):
                    file_path.unlink()
            
            self.logger.info(f"Cleaned up files for session: {session_id}")
            
        except Exception as e:
            self.logger.exception(
                f"Error cleaning up session files: {session_id}",
                extra_data={"error": str(e)}
            )


class RecordingSession:
    """
    Individual recording session handler.
    
    Manages audio recording, transcription, and quality metrics
    for a single voice agent session.
    """
    
    def __init__(
        self,
        metadata: SessionMetadata,
        storage_path: Path,
        enable_compression: bool = True,
        enable_encryption: bool = False,
        quality_monitoring: bool = True,
        real_time_transcription: bool = True,
        logger: Optional[StructuredLogger] = None
    ):
        self.metadata = metadata
        self.storage_path = storage_path
        self.enable_compression = enable_compression
        self.enable_encryption = enable_encryption
        self.quality_monitoring = quality_monitoring
        self.real_time_transcription = real_time_transcription
        self.logger = logger or StructuredLogger(__name__, "recording_session")
        
        # Session data
        self.audio_chunks: List[AudioChunk] = []
        self.transcript_segments: List[TranscriptSegment] = []
        self.quality_metrics = QualityMetrics()
        
        # Recording state
        self.is_initialized = False
        self.is_paused = False
        self.start_time = time.time()
        self.pause_duration = 0.0
        
        # File handles
        self.audio_file: Optional[io.BytesIO] = None
        self.transcript_file: Optional[io.StringIO] = None
        
        # Threading lock for thread-safe operations
        self.lock = threading.Lock()
    
    async def initialize(self):
        """Initialize the recording session."""
        if self.is_initialized:
            return
        
        # Create storage directory
        session_dir = self.storage_path / self.metadata.session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Set storage path in metadata
        self.metadata.storage_path = str(session_dir)
        
        # Initialize audio recording if enabled
        if self.metadata.privacy_level in [PrivacyLevel.FULL, PrivacyLevel.AUDIO_ONLY]:
            self.audio_file = io.BytesIO()
        
        # Initialize transcript recording if enabled
        if self.metadata.privacy_level in [PrivacyLevel.FULL, PrivacyLevel.TRANSCRIPT_ONLY]:
            self.transcript_file = io.StringIO()
        
        self.is_initialized = True
        
        # Save initial metadata
        await self._save_metadata()
        
        self.logger.info(
            f"Recording session initialized: {self.metadata.session_id}",
            extra_data={
                "session_id": self.metadata.session_id,
                "storage_path": self.metadata.storage_path,
                "privacy_level": self.metadata.privacy_level.value
            }
        )
    
    async def record_audio_chunk(self, chunk: AudioChunk):
        """Record an audio chunk."""
        if not self.is_initialized or self.is_paused:
            return
        
        with self.lock:
            self.audio_chunks.append(chunk)
            
            # Write to audio buffer if available
            if self.audio_file:
                self.audio_file.write(chunk.data)
            
            # Update quality metrics
            if self.quality_monitoring and chunk.volume_level is not None:
                self._update_quality_metrics_audio(chunk)
    
    async def add_transcript_segment(self, segment: TranscriptSegment):
        """Add a transcript segment."""
        if not self.is_initialized or self.is_paused:
            return
        
        with self.lock:
            self.transcript_segments.append(segment)
            
            # Write to transcript buffer if available
            if self.transcript_file:
                self.transcript_file.write(f"[{segment.timestamp}] {segment.speaker_id or 'Speaker'}: {segment.text}\n")
            
            # Update quality metrics
            if self.quality_monitoring:
                self._update_quality_metrics_transcript(segment)
    
    async def pause(self):
        """Pause the recording session."""
        self.is_paused = True
        self.metadata.status = SessionStatus.PAUSED
        await self._save_metadata()
    
    async def resume(self):
        """Resume the recording session."""
        if self.is_paused:
            self.is_paused = False
            self.metadata.status = SessionStatus.ACTIVE
            await self._save_metadata()
    
    async def finalize(self, final_quality_score: Optional[float] = None) -> SessionMetadata:
        """Finalize the recording session."""
        self.metadata.end_time = datetime.now(timezone.utc)
        self.metadata.duration = time.time() - self.start_time - self.pause_duration
        self.metadata.status = SessionStatus.COMPLETED
        
        # Finalize quality metrics
        if self.quality_monitoring:
            self._finalize_quality_metrics(final_quality_score)
            self.metadata.quality_metrics = self.quality_metrics
        
        # Save final audio file
        if self.audio_file and self.audio_chunks:
            await self._save_audio_file()
        
        # Save final transcript file
        if self.transcript_file and self.transcript_segments:
            await self._save_transcript_file()
        
        # Save final metadata
        await self._save_metadata()
        
        # Save quality metrics
        if self.quality_monitoring:
            await self._save_quality_metrics()
        
        self.logger.info(
            f"Recording session finalized: {self.metadata.session_id}",
            extra_data={
                "session_id": self.metadata.session_id,
                "duration": self.metadata.duration,
                "audio_chunks": len(self.audio_chunks),
                "transcript_segments": len(self.transcript_segments),
                "quality_score": final_quality_score
            }
        )
        
        return self.metadata
    
    async def get_current_quality_metrics(self) -> Optional[QualityMetrics]:
        """Get current quality metrics for the session."""
        if not self.quality_monitoring:
            return None
        
        return self.quality_metrics
    
    def _update_quality_metrics_audio(self, chunk: AudioChunk):
        """Update quality metrics based on audio chunk."""
        if chunk.volume_level is not None:
            # Update speech/silence duration based on volume
            duration = len(chunk.data) / (chunk.sample_rate * chunk.channels * (chunk.bit_depth // 8))
            
            if chunk.volume_level > 0.1:  # Speech threshold
                self.quality_metrics.speech_duration += duration
            else:
                self.quality_metrics.silence_duration += duration
    
    def _update_quality_metrics_transcript(self, segment: TranscriptSegment):
        """Update quality metrics based on transcript segment."""
        # Update transcript accuracy if confidence is available
        if segment.confidence is not None:
            if self.quality_metrics.transcript_accuracy_score is None:
                self.quality_metrics.transcript_accuracy_score = segment.confidence
            else:
                # Running average
                total_segments = len(self.transcript_segments)
                self.quality_metrics.transcript_accuracy_score = (
                    (self.quality_metrics.transcript_accuracy_score * (total_segments - 1) + segment.confidence) / total_segments
                )
    
    def _finalize_quality_metrics(self, final_quality_score: Optional[float] = None):
        """Finalize quality metrics for the session."""
        # Set final quality score
        if final_quality_score is not None:
            self.quality_metrics.audio_quality_score = final_quality_score
        
        # Calculate latency metrics from audio chunks
        if self.audio_chunks:
            latencies = []
            for i in range(1, len(self.audio_chunks)):
                latency = self.audio_chunks[i].timestamp - self.audio_chunks[i-1].timestamp
                latencies.append(latency)
            
            if latencies:
                self.quality_metrics.latency_avg = sum(latencies) / len(latencies)
                self.quality_metrics.latency_p95 = sorted(latencies)[int(0.95 * len(latencies))]
        
        # Count speaker switches
        current_speaker = None
        for segment in self.transcript_segments:
            if segment.speaker_id and segment.speaker_id != current_speaker:
                self.quality_metrics.speaker_switch_count += 1
                current_speaker = segment.speaker_id
    
    async def _save_metadata(self):
        """Save session metadata to file."""
        metadata_file = Path(self.metadata.storage_path) / f"{self.metadata.session_id}_metadata.json"
        
        try:
            with open(metadata_file, 'w') as f:
                json.dump(self.metadata.to_dict(), f, indent=2)
        except Exception as e:
            self.logger.exception(
                f"Error saving metadata for session {self.metadata.session_id}",
                extra_data={"error": str(e)}
            )
    
    async def _save_audio_file(self):
        """Save audio data to file."""
        if not self.audio_file or not self.audio_chunks:
            return
        
        audio_file_path = Path(self.metadata.storage_path) / f"{self.metadata.session_id}_audio.{self.metadata.audio_format.value}"
        
        try:
            # Reset buffer position
            self.audio_file.seek(0)
            audio_data = self.audio_file.read()
            
            if self.metadata.audio_format == AudioFormat.WAV:
                # Create WAV file
                with wave.open(str(audio_file_path), 'wb') as wav_file:
                    wav_file.setnchannels(self.metadata.channels)
                    wav_file.setsampwidth(self.metadata.bit_depth // 8)
                    wav_file.setframerate(self.metadata.sample_rate)
                    wav_file.writeframes(audio_data)
            else:
                # Save raw audio data
                with open(audio_file_path, 'wb') as f:
                    f.write(audio_data)
            
            # Compress if enabled
            if self.enable_compression:
                await self._compress_file(audio_file_path)
            
        except Exception as e:
            self.logger.exception(
                f"Error saving audio file for session {self.metadata.session_id}",
                extra_data={"error": str(e)}
            )
    
    async def _save_transcript_file(self):
        """Save transcript data to file."""
        if not self.transcript_file or not self.transcript_segments:
            return
        
        # Save as JSON
        transcript_file_path = Path(self.metadata.storage_path) / f"{self.metadata.session_id}_transcript.json"
        
        try:
            transcript_data = {
                "session_id": self.metadata.session_id,
                "segments": [segment.to_dict() for segment in self.transcript_segments]
            }
            
            with open(transcript_file_path, 'w') as f:
                json.dump(transcript_data, f, indent=2)
            
            # Also save as text
            text_file_path = Path(self.metadata.storage_path) / f"{self.metadata.session_id}_transcript.txt"
            self.transcript_file.seek(0)
            
            with open(text_file_path, 'w') as f:
                f.write(self.transcript_file.read())
            
            # Compress if enabled
            if self.enable_compression:
                await self._compress_file(transcript_file_path)
                await self._compress_file(text_file_path)
            
        except Exception as e:
            self.logger.exception(
                f"Error saving transcript file for session {self.metadata.session_id}",
                extra_data={"error": str(e)}
            )
    
    async def _save_quality_metrics(self):
        """Save quality metrics to file."""
        if not self.quality_monitoring or not self.quality_metrics:
            return
        
        quality_file_path = Path(self.metadata.storage_path) / f"{self.metadata.session_id}_quality.json"
        
        try:
            with open(quality_file_path, 'w') as f:
                json.dump(self.quality_metrics.to_dict(), f, indent=2)
        except Exception as e:
            self.logger.exception(
                f"Error saving quality metrics for session {self.metadata.session_id}",
                extra_data={"error": str(e)}
            )
    
    async def _compress_file(self, file_path: Path):
        """Compress a file using gzip."""
        try:
            with open(file_path, 'rb') as f:
                data = f.read()
            
            compressed_path = file_path.with_suffix(file_path.suffix + '.gz')
            
            with gzip.open(compressed_path, 'wb') as f:
                f.write(data)
            
            # Remove original file
            file_path.unlink()
            
        except Exception as e:
            self.logger.exception(
                f"Error compressing file {file_path}",
                extra_data={"error": str(e)}
            )