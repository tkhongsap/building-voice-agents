"""
Session Recording Integration with Voice Pipeline

This module demonstrates how to integrate the session recording system
with the existing voice pipeline and LiveKit infrastructure for production
voice agent deployments.
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timezone
import logging

from .session_recording import SessionRecordingManager, PrivacyLevel, AudioFormat
from .transcript_processor import TranscriptProcessor, TranscriptSegment
from .privacy_manager import PrivacyManager, ConsentType
from .export_manager import ExportManager, ExportFormat, ReportType
from .session_playback import SessionPlaybackManager
from .structured_logging import StructuredLogger, CorrelationIdManager

# Import voice pipeline components
from ..pipeline.voice_pipeline_with_logging import VoicePipelineWithLogging
from ..components.stt.base_stt import STTResult
from ..components.llm.base_llm import LLMResponse
from ..components.tts.base_tts import TTSResult


@dataclass
class RecordingConfig:
    """Configuration for session recording."""
    enable_recording: bool = True
    enable_transcripts: bool = True
    enable_quality_monitoring: bool = True
    privacy_level: PrivacyLevel = PrivacyLevel.FULL
    audio_format: AudioFormat = AudioFormat.WAV
    sample_rate: int = 44100
    retention_days: int = 30
    require_consent: bool = True
    anonymize_exports: bool = False


class VoicePipelineWithRecording(VoicePipelineWithLogging):
    """
    Enhanced voice pipeline with integrated session recording.
    
    Extends the existing voice pipeline to include comprehensive
    session recording, transcript processing, and privacy management.
    """
    
    def __init__(
        self,
        stt_provider,
        llm_provider,
        tts_provider,
        vad_provider=None,
        config=None,
        session_id=None,
        recording_config: Optional[RecordingConfig] = None,
        storage_path: str = "./recordings"
    ):
        """
        Initialize enhanced voice pipeline with recording.
        
        Args:
            stt_provider: Speech-to-text provider
            llm_provider: Large language model provider
            tts_provider: Text-to-speech provider
            vad_provider: Voice activity detection provider
            config: Pipeline configuration
            session_id: Session identifier
            recording_config: Recording configuration
            storage_path: Storage path for recordings
        """
        super().__init__(stt_provider, llm_provider, tts_provider, vad_provider, config, session_id)
        
        self.recording_config = recording_config or RecordingConfig()
        self.storage_path = storage_path
        
        # Initialize recording components
        self.recording_manager = SessionRecordingManager(
            storage_path=storage_path,
            default_retention_days=self.recording_config.retention_days,
            quality_monitoring=self.recording_config.enable_quality_monitoring
        )
        
        self.transcript_processor = TranscriptProcessor(
            enable_real_time=True,
            enable_speaker_diarization=True,
            enable_keyword_extraction=True
        )
        
        self.privacy_manager = PrivacyManager(
            storage_path=storage_path,
            default_retention_days=self.recording_config.retention_days
        )
        
        self.export_manager = ExportManager(
            storage_path=storage_path,
            export_path=f"{storage_path}/exports"
        )
        
        # Recording state
        self.recording_session_id: Optional[str] = None
        self.transcript_session_id: Optional[str] = None
        self.current_user_id: Optional[str] = None
        
        self.logger.info(
            "Voice pipeline with recording initialized",
            extra_data={
                "recording_enabled": self.recording_config.enable_recording,
                "privacy_level": self.recording_config.privacy_level.value,
                "storage_path": storage_path
            }
        )
    
    async def initialize(self):
        """Initialize the enhanced pipeline."""
        await super().initialize()
        
        # Start background tasks for recording components
        await self.recording_manager.start_background_tasks()
        await self.privacy_manager.start_background_tasks()
        await self.export_manager.start_export_workers()
    
    async def cleanup(self):
        """Cleanup the enhanced pipeline."""
        # End any active recording sessions
        if self.recording_session_id:
            await self.end_recording_session()
        
        # Stop background tasks
        await self.recording_manager.stop_background_tasks()
        await self.privacy_manager.stop_background_tasks()
        await self.export_manager.stop_export_workers()
        
        await super().cleanup()
    
    async def start_recording_session(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        require_consent: bool = None,
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start a new recording session.
        
        Args:
            user_id: User identifier
            agent_id: Agent identifier
            require_consent: Whether to require user consent
            custom_metadata: Custom metadata for the session
            
        Returns:
            Recording session ID
        """
        require_consent = require_consent if require_consent is not None else self.recording_config.require_consent
        
        # Check consent if required
        if require_consent and user_id:
            has_consent = await self.privacy_manager.check_consent(user_id, ConsentType.RECORDING)
            if not has_consent:
                # Request consent
                consent_results = await self.privacy_manager.request_consent(
                    user_id=user_id,
                    consent_types=[ConsentType.RECORDING, ConsentType.TRANSCRIPT, ConsentType.ANALYTICS]
                )
                
                if not consent_results.get(ConsentType.RECORDING.value, False):
                    raise PermissionError("User consent required for recording")
        
        # Start recording session
        if self.recording_config.enable_recording:
            self.recording_session_id = await self.recording_manager.start_session(
                user_id=user_id,
                agent_id=agent_id,
                privacy_level=self.recording_config.privacy_level,
                audio_format=self.recording_config.audio_format,
                sample_rate=self.recording_config.sample_rate,
                custom_metadata=custom_metadata or {},
                retention_days=self.recording_config.retention_days
            )
        
        # Start transcript processing session
        if self.recording_config.enable_transcripts:
            transcript_session = await self.transcript_processor.start_session(
                session_id=self.recording_session_id or f"transcript_{int(time.time())}",
                user_id=user_id,
                language="en"  # Could be configurable
            )
            self.transcript_session_id = transcript_session.session_id
        
        self.current_user_id = user_id
        
        self.logger.info(
            f"Recording session started: {self.recording_session_id}",
            extra_data={
                "recording_session_id": self.recording_session_id,
                "transcript_session_id": self.transcript_session_id,
                "user_id": user_id,
                "agent_id": agent_id
            }
        )
        
        return self.recording_session_id or self.transcript_session_id
    
    async def end_recording_session(self, final_quality_score: Optional[float] = None) -> Dict[str, Any]:
        """
        End the current recording session.
        
        Args:
            final_quality_score: Final quality score for the session
            
        Returns:
            Session summary and metadata
        """
        results = {}
        
        # End recording session
        if self.recording_session_id:
            try:
                metadata = await self.recording_manager.end_session(
                    self.recording_session_id, 
                    final_quality_score
                )
                
                # Apply retention policy
                await self.privacy_manager.apply_retention_policy(
                    self.recording_session_id, 
                    metadata
                )
                
                results["recording"] = {
                    "session_id": self.recording_session_id,
                    "duration": metadata.duration,
                    "status": metadata.status.value,
                    "quality_score": final_quality_score
                }
                
            except Exception as e:
                self.logger.exception("Error ending recording session", extra_data={"error": str(e)})
                results["recording"] = {"error": str(e)}
        
        # End transcript session
        if self.transcript_session_id:
            try:
                summary = await self.transcript_processor.end_session(self.transcript_session_id)
                results["transcript"] = summary
                
            except Exception as e:
                self.logger.exception("Error ending transcript session", extra_data={"error": str(e)})
                results["transcript"] = {"error": str(e)}
        
        # Reset session IDs
        session_id = self.recording_session_id or self.transcript_session_id
        self.recording_session_id = None
        self.transcript_session_id = None
        self.current_user_id = None
        
        self.logger.info(
            f"Recording session ended: {session_id}",
            extra_data={
                "session_id": session_id,
                "results": results
            }
        )
        
        return results
    
    async def process_audio_with_recording(
        self,
        audio_data: bytes,
        user_id: Optional[str] = None,
        conversation_context: Optional[Dict[str, Any]] = None,
        speaker_id: Optional[str] = None
    ) -> tuple[TTSResult, Any]:
        """
        Process audio through the pipeline with recording.
        
        Args:
            audio_data: Input audio data
            user_id: User identifier
            conversation_context: Conversation context
            speaker_id: Speaker identifier for the audio
            
        Returns:
            Tuple of (TTS result, pipeline metrics)
        """
        # Process through the base pipeline
        tts_result, metrics = await self.process_audio(audio_data, user_id, conversation_context)
        
        # Record audio chunk if recording is active
        if self.recording_session_id and self.recording_config.enable_recording:
            try:
                await self.recording_manager.record_audio_chunk(
                    session_id=self.recording_session_id,
                    audio_data=audio_data,
                    timestamp=time.time(),
                    speaker_id=speaker_id or "user"
                )
            except Exception as e:
                self.logger.exception("Error recording audio chunk", extra_data={"error": str(e)})
        
        return tts_result, metrics
    
    async def add_transcript_from_stt(
        self,
        stt_result: STTResult,
        speaker_id: Optional[str] = None
    ):
        """
        Add transcript from STT result.
        
        Args:
            stt_result: Speech-to-text result
            speaker_id: Speaker identifier
        """
        if not self.recording_config.enable_transcripts:
            return
        
        # Create transcript segment
        segment = TranscriptSegment(
            id=f"seg_{int(time.time() * 1000)}",
            timestamp=time.time(),
            text=stt_result.text,
            speaker_id=speaker_id or "user",
            confidence=getattr(stt_result, 'confidence', None),
            language=getattr(stt_result, 'language', None),
            is_final=getattr(stt_result, 'is_final', True)
        )
        
        # Add to recording session
        if self.recording_session_id:
            try:
                await self.recording_manager.add_transcript_segment(
                    session_id=self.recording_session_id,
                    text=segment.text,
                    timestamp=segment.timestamp,
                    speaker_id=segment.speaker_id,
                    confidence=segment.confidence,
                    language=segment.language,
                    is_final=segment.is_final
                )
            except Exception as e:
                self.logger.exception("Error adding transcript to recording", extra_data={"error": str(e)})
        
        # Process with transcript processor
        if self.transcript_session_id:
            try:
                await self.transcript_processor.process_segment(self.transcript_session_id, segment)
            except Exception as e:
                self.logger.exception("Error processing transcript segment", extra_data={"error": str(e)})
    
    async def add_transcript_from_llm(
        self,
        llm_response: LLMResponse,
        speaker_id: Optional[str] = None
    ):
        """
        Add transcript from LLM response.
        
        Args:
            llm_response: Large language model response
            speaker_id: Speaker identifier
        """
        if not self.recording_config.enable_transcripts:
            return
        
        # Create transcript segment for LLM response
        segment = TranscriptSegment(
            id=f"seg_{int(time.time() * 1000)}",
            timestamp=time.time(),
            text=llm_response.content,
            speaker_id=speaker_id or "agent",
            confidence=1.0,  # LLM responses are considered final
            is_final=True
        )
        
        # Add to recording session
        if self.recording_session_id:
            try:
                await self.recording_manager.add_transcript_segment(
                    session_id=self.recording_session_id,
                    text=segment.text,
                    timestamp=segment.timestamp,
                    speaker_id=segment.speaker_id,
                    confidence=segment.confidence,
                    is_final=segment.is_final
                )
            except Exception as e:
                self.logger.exception("Error adding LLM transcript to recording", extra_data={"error": str(e)})
        
        # Process with transcript processor
        if self.transcript_session_id:
            try:
                await self.transcript_processor.process_segment(self.transcript_session_id, segment)
            except Exception as e:
                self.logger.exception("Error processing LLM transcript segment", extra_data={"error": str(e)})
    
    async def export_current_session(
        self,
        format: ExportFormat = ExportFormat.JSON,
        include_audio: bool = False
    ) -> Optional[str]:
        """
        Export the current recording session.
        
        Args:
            format: Export format
            include_audio: Whether to include audio data
            
        Returns:
            Path to exported file
        """
        if not self.recording_session_id:
            raise ValueError("No active recording session")
        
        try:
            result = await self.export_manager.export_single_session(
                session_id=self.recording_session_id,
                format=format,
                include_audio=include_audio,
                include_transcripts=True
            )
            
            if result.success:
                return result.output_path
            else:
                raise RuntimeError(f"Export failed: {result.error_message}")
                
        except Exception as e:
            self.logger.exception("Error exporting session", extra_data={"error": str(e)})
            raise
    
    async def get_session_playback_manager(self) -> SessionPlaybackManager:
        """
        Get session playback manager for recorded sessions.
        
        Returns:
            Session playback manager
        """
        return SessionPlaybackManager(storage_path=self.storage_path)
    
    def get_recording_status(self) -> Dict[str, Any]:
        """
        Get current recording status.
        
        Returns:
            Recording status information
        """
        return {
            "recording_active": self.recording_session_id is not None,
            "transcript_active": self.transcript_session_id is not None,
            "recording_session_id": self.recording_session_id,
            "transcript_session_id": self.transcript_session_id,
            "user_id": self.current_user_id,
            "config": {
                "enable_recording": self.recording_config.enable_recording,
                "enable_transcripts": self.recording_config.enable_transcripts,
                "privacy_level": self.recording_config.privacy_level.value,
                "retention_days": self.recording_config.retention_days
            }
        }


class LiveKitVoiceAgentWithRecording:
    """
    LiveKit voice agent with integrated session recording.
    
    Provides a complete voice agent implementation with built-in
    session recording, transcript processing, and analytics.
    """
    
    def __init__(
        self,
        room_name: str,
        user_token: str,
        recording_config: Optional[RecordingConfig] = None,
        storage_path: str = "./recordings"
    ):
        """
        Initialize LiveKit voice agent with recording.
        
        Args:
            room_name: LiveKit room name
            user_token: User authentication token
            recording_config: Recording configuration
            storage_path: Storage path for recordings
        """
        self.room_name = room_name
        self.user_token = user_token
        self.recording_config = recording_config or RecordingConfig()
        self.storage_path = storage_path
        
        # Initialize pipeline (would be configured with actual providers)
        self.pipeline: Optional[VoicePipelineWithRecording] = None
        
        # Agent state
        self.is_connected = False
        self.current_user_id: Optional[str] = None
        self.session_start_time: Optional[datetime] = None
        
        self.logger = StructuredLogger(__name__, "livekit_agent")
    
    async def connect(self):
        """Connect to LiveKit room and start recording."""
        try:
            # In a real implementation, this would connect to LiveKit
            # For now, we'll simulate the connection
            
            self.is_connected = True
            self.session_start_time = datetime.now(timezone.utc)
            
            # Extract user ID from token (simplified)
            self.current_user_id = f"user_{int(time.time())}"
            
            # Start recording session
            if self.pipeline:
                await self.pipeline.start_recording_session(
                    user_id=self.current_user_id,
                    agent_id="livekit_agent",
                    custom_metadata={
                        "room_name": self.room_name,
                        "platform": "livekit",
                        "agent_version": "1.0.0"
                    }
                )
            
            self.logger.info(
                f"Connected to LiveKit room: {self.room_name}",
                extra_data={
                    "room_name": self.room_name,
                    "user_id": self.current_user_id,
                    "recording_enabled": self.recording_config.enable_recording
                }
            )
            
        except Exception as e:
            self.logger.exception("Error connecting to LiveKit", extra_data={"error": str(e)})
            raise
    
    async def disconnect(self):
        """Disconnect from LiveKit room and end recording."""
        try:
            if self.pipeline:
                # Calculate session quality score (simplified)
                session_duration = (datetime.now(timezone.utc) - self.session_start_time).total_seconds() if self.session_start_time else 0
                quality_score = min(1.0, max(0.0, 0.8 + (session_duration / 3600) * 0.1))  # Simplified scoring
                
                # End recording session
                results = await self.pipeline.end_recording_session(final_quality_score=quality_score)
                
                self.logger.info(
                    "Recording session ended",
                    extra_data={
                        "session_duration": session_duration,
                        "quality_score": quality_score,
                        "results": results
                    }
                )
            
            self.is_connected = False
            self.current_user_id = None
            self.session_start_time = None
            
            self.logger.info(f"Disconnected from LiveKit room: {self.room_name}")
            
        except Exception as e:
            self.logger.exception("Error disconnecting from LiveKit", extra_data={"error": str(e)})
    
    async def handle_audio_frame(self, audio_data: bytes):
        """
        Handle incoming audio frame from LiveKit.
        
        Args:
            audio_data: Audio frame data
        """
        if not self.pipeline or not self.is_connected:
            return
        
        try:
            # Process audio through pipeline with recording
            tts_result, metrics = await self.pipeline.process_audio_with_recording(
                audio_data=audio_data,
                user_id=self.current_user_id,
                speaker_id="user"
            )
            
            # Send TTS result back through LiveKit (simplified)
            await self._send_audio_to_livekit(tts_result.audio_data)
            
        except Exception as e:
            self.logger.exception("Error handling audio frame", extra_data={"error": str(e)})
    
    async def handle_text_message(self, message: str):
        """
        Handle incoming text message.
        
        Args:
            message: Text message from user
        """
        if not self.pipeline:
            return
        
        try:
            # Add text as transcript
            if self.pipeline.recording_session_id:
                await self.pipeline.recording_manager.add_transcript_segment(
                    session_id=self.pipeline.recording_session_id,
                    text=message,
                    timestamp=time.time(),
                    speaker_id="user",
                    confidence=1.0,
                    is_final=True
                )
        
        except Exception as e:
            self.logger.exception("Error handling text message", extra_data={"error": str(e)})
    
    async def get_session_analytics(self) -> Dict[str, Any]:
        """
        Get analytics for the current session.
        
        Returns:
            Session analytics data
        """
        if not self.pipeline or not self.pipeline.recording_session_id:
            return {"error": "No active recording session"}
        
        try:
            # Get basic session info
            status = self.pipeline.get_recording_status()
            
            # Get session playback manager for analysis
            playback_manager = await self.pipeline.get_session_playback_manager()
            
            # Perform basic analysis
            analysis_results = await playback_manager.analyze_session(
                session_id=self.pipeline.recording_session_id,
                analysis_types=[
                    AnalysisType.QUALITY,
                    AnalysisType.CONVERSATION_FLOW,
                    AnalysisType.PERFORMANCE
                ]
            )
            
            return {
                "session_status": status,
                "analytics": analysis_results,
                "session_duration": (datetime.now(timezone.utc) - self.session_start_time).total_seconds() if self.session_start_time else 0
            }
            
        except Exception as e:
            self.logger.exception("Error getting session analytics", extra_data={"error": str(e)})
            return {"error": str(e)}
    
    async def export_session(
        self,
        format: ExportFormat = ExportFormat.HTML,
        include_audio: bool = False
    ) -> Optional[str]:
        """
        Export the current session.
        
        Args:
            format: Export format
            include_audio: Whether to include audio
            
        Returns:
            Path to exported file
        """
        if not self.pipeline:
            raise ValueError("Pipeline not initialized")
        
        return await self.pipeline.export_current_session(format, include_audio)
    
    async def _send_audio_to_livekit(self, audio_data: bytes):
        """Send audio data back to LiveKit room."""
        # In a real implementation, this would send audio through LiveKit
        # For now, we'll just log it
        self.logger.debug(f"Sending audio to LiveKit: {len(audio_data)} bytes")


# Example usage and integration patterns
async def example_voice_agent_with_recording():
    """Example of using the voice agent with recording."""
    
    # Configure recording
    recording_config = RecordingConfig(
        enable_recording=True,
        enable_transcripts=True,
        privacy_level=PrivacyLevel.FULL,
        retention_days=30,
        require_consent=True
    )
    
    # Initialize LiveKit agent with recording
    agent = LiveKitVoiceAgentWithRecording(
        room_name="voice_session_room",
        user_token="user_token_here",
        recording_config=recording_config,
        storage_path="./voice_recordings"
    )
    
    try:
        # Connect and start recording
        await agent.connect()
        
        # Simulate some conversation
        await agent.handle_text_message("Hello, I need help with my account")
        
        # Simulate audio processing
        fake_audio = b"simulated_audio_data" * 100
        await agent.handle_audio_frame(fake_audio)
        
        # Get session analytics
        analytics = await agent.get_session_analytics()
        print(f"Session analytics: {analytics}")
        
        # Export session
        export_path = await agent.export_session(ExportFormat.HTML, include_audio=False)
        print(f"Session exported to: {export_path}")
        
    finally:
        # Disconnect and end recording
        await agent.disconnect()


async def example_standalone_recording():
    """Example of using recording system standalone."""
    
    # Initialize recording components
    recording_manager = SessionRecordingManager(
        storage_path="./recordings",
        default_retention_days=30
    )
    
    transcript_processor = TranscriptProcessor(
        enable_real_time=True,
        enable_speaker_diarization=True
    )
    
    privacy_manager = PrivacyManager(
        storage_path="./recordings",
        consent_storage_path="./consent"
    )
    
    try:
        # Start background tasks
        await recording_manager.start_background_tasks()
        await privacy_manager.start_background_tasks()
        
        # Request user consent
        consent_results = await privacy_manager.request_consent(
            user_id="example_user",
            consent_types=[ConsentType.RECORDING, ConsentType.TRANSCRIPT]
        )
        
        if consent_results[ConsentType.RECORDING.value]:
            # Start recording session
            session_id = await recording_manager.start_session(
                user_id="example_user",
                tags=["example", "demo"]
            )
            
            # Start transcript session
            transcript_session = await transcript_processor.start_session(
                session_id=session_id,
                user_id="example_user"
            )
            
            # Simulate conversation
            conversation = [
                {"speaker": "user", "text": "Hello, I need help", "timestamp": time.time()},
                {"speaker": "agent", "text": "How can I assist you?", "timestamp": time.time() + 1},
                {"speaker": "user", "text": "I want to check my order status", "timestamp": time.time() + 3}
            ]
            
            for turn in conversation:
                # Record audio (simulated)
                audio_data = f"audio_for_{turn['text'][:10]}".encode()
                await recording_manager.record_audio_chunk(
                    session_id=session_id,
                    audio_data=audio_data,
                    timestamp=turn["timestamp"],
                    speaker_id=turn["speaker"]
                )
                
                # Add transcript
                await recording_manager.add_transcript_segment(
                    session_id=session_id,
                    text=turn["text"],
                    timestamp=turn["timestamp"],
                    speaker_id=turn["speaker"],
                    confidence=0.95
                )
            
            # End sessions
            metadata = await recording_manager.end_session(session_id, final_quality_score=0.9)
            summary = await transcript_processor.end_session(session_id)
            
            print(f"Recording completed: {metadata.session_id}")
            print(f"Duration: {metadata.duration:.2f}s")
            print(f"Transcript summary: {summary}")
            
            # Apply retention policy
            await privacy_manager.apply_retention_policy(session_id, metadata)
            
    finally:
        # Cleanup
        await recording_manager.stop_background_tasks()
        await privacy_manager.stop_background_tasks()


if __name__ == "__main__":
    # Run examples
    print("Running voice agent with recording example...")
    asyncio.run(example_voice_agent_with_recording())
    
    print("\nRunning standalone recording example...")
    asyncio.run(example_standalone_recording())
    
    print("\nExamples completed successfully!")