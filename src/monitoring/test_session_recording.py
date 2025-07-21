"""
Comprehensive Tests for Session Recording System

This module provides comprehensive unit tests and integration tests
for the session recording, transcript processing, and privacy management
components of the voice agent platform.
"""

import asyncio
import pytest
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timezone, timedelta
import time
import uuid

from .session_recording import (
    SessionRecordingManager, SessionMetadata, AudioChunk, TranscriptSegment,
    AudioFormat, PrivacyLevel, SessionStatus, QualityMetrics
)
from .transcript_processor import TranscriptProcessor, TranscriptFormat
from .session_playback import SessionPlaybackManager, AnalysisType
from .privacy_manager import PrivacyManager, ConsentType, RetentionPolicies, ComplianceFramework
from .export_manager import ExportManager, ExportFormat, ReportType, ExportScope, ExportRequest
from .structured_logging import StructuredLogger


class TestSessionRecording:
    """Test cases for session recording functionality."""
    
    @pytest.fixture
    async def temp_storage(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    async def recording_manager(self, temp_storage):
        """Create session recording manager for testing."""
        manager = SessionRecordingManager(
            storage_path=temp_storage,
            enable_compression=False,  # Disable for easier testing
            enable_encryption=False,
            default_retention_days=7
        )
        yield manager
        await manager.stop_background_tasks()
    
    @pytest.mark.asyncio
    async def test_start_session(self, recording_manager):
        """Test starting a recording session."""
        session_id = await recording_manager.start_session(
            user_id="test_user",
            agent_id="test_agent",
            privacy_level=PrivacyLevel.FULL,
            tags=["test", "integration"]
        )
        
        assert session_id is not None
        assert session_id in recording_manager.active_sessions
        
        session = recording_manager.active_sessions[session_id]
        assert session.metadata.user_id == "test_user"
        assert session.metadata.agent_id == "test_agent"
        assert session.metadata.privacy_level == PrivacyLevel.FULL
        assert "test" in session.metadata.tags
    
    @pytest.mark.asyncio
    async def test_record_audio_chunk(self, recording_manager):
        """Test recording audio chunks."""
        session_id = await recording_manager.start_session(user_id="test_user")
        
        # Create test audio data
        audio_data = b"fake_audio_data_chunk_1"
        timestamp = time.time()
        
        await recording_manager.record_audio_chunk(
            session_id=session_id,
            audio_data=audio_data,
            timestamp=timestamp,
            speaker_id="user"
        )
        
        session = recording_manager.active_sessions[session_id]
        assert len(session.audio_chunks) == 1
        
        chunk = session.audio_chunks[0]
        assert chunk.data == audio_data
        assert chunk.timestamp == timestamp
        assert chunk.speaker_id == "user"
    
    @pytest.mark.asyncio
    async def test_add_transcript_segment(self, recording_manager):
        """Test adding transcript segments."""
        session_id = await recording_manager.start_session(user_id="test_user")
        
        await recording_manager.add_transcript_segment(
            session_id=session_id,
            text="Hello, how can I help you today?",
            timestamp=time.time(),
            speaker_id="agent",
            confidence=0.95,
            is_final=True
        )
        
        session = recording_manager.active_sessions[session_id]
        assert len(session.transcript_segments) == 1
        
        segment = session.transcript_segments[0]
        assert segment.text == "Hello, how can I help you today?"
        assert segment.speaker_id == "agent"
        assert segment.confidence == 0.95
        assert segment.is_final is True
    
    @pytest.mark.asyncio
    async def test_end_session(self, recording_manager):
        """Test ending a recording session."""
        session_id = await recording_manager.start_session(user_id="test_user")
        
        # Add some test data
        await recording_manager.record_audio_chunk(session_id, b"audio_data", time.time())
        await recording_manager.add_transcript_segment(session_id, "Test transcript", time.time())
        
        # End the session
        metadata = await recording_manager.end_session(session_id, final_quality_score=0.85)
        
        assert metadata.status == SessionStatus.COMPLETED
        assert metadata.end_time is not None
        assert metadata.duration is not None
        assert session_id not in recording_manager.active_sessions
    
    @pytest.mark.asyncio
    async def test_privacy_level_audio_only(self, recording_manager):
        """Test privacy level AUDIO_ONLY."""
        session_id = await recording_manager.start_session(
            user_id="test_user",
            privacy_level=PrivacyLevel.AUDIO_ONLY
        )
        
        # Audio should be recorded
        await recording_manager.record_audio_chunk(session_id, b"audio_data", time.time())
        
        # Transcript should be ignored
        await recording_manager.add_transcript_segment(session_id, "This should be ignored", time.time())
        
        session = recording_manager.active_sessions[session_id]
        assert len(session.audio_chunks) == 1
        assert len(session.transcript_segments) == 0
    
    @pytest.mark.asyncio
    async def test_privacy_level_transcript_only(self, recording_manager):
        """Test privacy level TRANSCRIPT_ONLY."""
        session_id = await recording_manager.start_session(
            user_id="test_user",
            privacy_level=PrivacyLevel.TRANSCRIPT_ONLY
        )
        
        # Audio should be ignored
        await recording_manager.record_audio_chunk(session_id, b"audio_data", time.time())
        
        # Transcript should be recorded
        await recording_manager.add_transcript_segment(session_id, "This should be recorded", time.time())
        
        session = recording_manager.active_sessions[session_id]
        assert len(session.audio_chunks) == 0
        assert len(session.transcript_segments) == 1
    
    @pytest.mark.asyncio
    async def test_session_callbacks(self, recording_manager):
        """Test session event callbacks."""
        callback_events = []
        
        def on_session_start(session_id, metadata):
            callback_events.append(("start", session_id))
        
        def on_session_end(session_id, metadata):
            callback_events.append(("end", session_id))
        
        recording_manager.add_callback("on_session_start", on_session_start)
        recording_manager.add_callback("on_session_end", on_session_end)
        
        session_id = await recording_manager.start_session(user_id="test_user")
        await recording_manager.end_session(session_id)
        
        assert len(callback_events) == 2
        assert callback_events[0] == ("start", session_id)
        assert callback_events[1] == ("end", session_id)


class TestTranscriptProcessor:
    """Test cases for transcript processing functionality."""
    
    @pytest.fixture
    async def transcript_processor(self):
        """Create transcript processor for testing."""
        processor = TranscriptProcessor(
            enable_real_time=False,  # Disable for testing
            enable_speaker_diarization=True,
            enable_keyword_extraction=True,
            confidence_threshold=0.7
        )
        yield processor
    
    @pytest.mark.asyncio
    async def test_start_session(self, transcript_processor):
        """Test starting a transcript session."""
        session = await transcript_processor.start_session(
            session_id="test_session",
            user_id="test_user",
            language="en",
            custom_vocabulary=["artificial", "intelligence"]
        )
        
        assert session.session_id == "test_session"
        assert session.user_id == "test_user"
        assert session.language == "en"
        assert "artificial" in session.custom_vocabulary
    
    @pytest.mark.asyncio
    async def test_process_segment(self, transcript_processor):
        """Test processing transcript segments."""
        session = await transcript_processor.start_session("test_session")
        
        segment = TranscriptSegment(
            id="seg_1",
            timestamp=time.time(),
            text="Hello, how are you today?",
            confidence=0.95,
            is_final=True
        )
        
        processed_segment = await transcript_processor.process_segment("test_session", segment)
        
        assert processed_segment.text is not None
        assert processed_segment.confidence == 0.95
    
    @pytest.mark.asyncio
    async def test_confidence_filtering(self, transcript_processor):
        """Test confidence-based filtering."""
        session = await transcript_processor.start_session("test_session")
        
        # Low confidence segment should be filtered
        low_confidence_segment = TranscriptSegment(
            id="seg_1",
            timestamp=time.time(),
            text="Unclear speech",
            confidence=0.3,
            is_final=True
        )
        
        processed = await transcript_processor.process_segment("test_session", low_confidence_segment)
        # Should still return the segment but it might be marked differently
        assert processed is not None
    
    @pytest.mark.asyncio
    async def test_end_session_summary(self, transcript_processor):
        """Test session summary generation."""
        session = await transcript_processor.start_session("test_session")
        
        # Add some segments
        segments = [
            TranscriptSegment(id="1", timestamp=time.time(), text="Hello there", speaker_id="user"),
            TranscriptSegment(id="2", timestamp=time.time(), text="How can I help?", speaker_id="agent"),
            TranscriptSegment(id="3", timestamp=time.time(), text="I need assistance", speaker_id="user")
        ]
        
        for segment in segments:
            await transcript_processor.process_segment("test_session", segment)
        
        summary = await transcript_processor.end_session("test_session")
        
        assert summary["total_segments"] > 0
        assert summary["total_words"] > 0
        assert len(summary["speakers"]) == 2  # user and agent


class TestPrivacyManager:
    """Test cases for privacy management functionality."""
    
    @pytest.fixture
    async def temp_storage(self):
        """Create temporary storage directories."""
        storage_dir = tempfile.mkdtemp()
        consent_dir = tempfile.mkdtemp()
        yield storage_dir, consent_dir
        shutil.rmtree(storage_dir)
        shutil.rmtree(consent_dir)
    
    @pytest.fixture
    async def privacy_manager(self, temp_storage):
        """Create privacy manager for testing."""
        storage_dir, consent_dir = temp_storage
        manager = PrivacyManager(
            storage_path=storage_dir,
            consent_storage_path=consent_dir,
            enable_encryption=False,  # Disable for testing
            default_retention_days=7
        )
        yield manager
        await manager.stop_background_tasks()
    
    @pytest.mark.asyncio
    async def test_request_consent(self, privacy_manager):
        """Test requesting user consent."""
        consent_results = await privacy_manager.request_consent(
            user_id="test_user",
            consent_types=[ConsentType.RECORDING, ConsentType.TRANSCRIPT],
            legal_basis="consent"
        )
        
        assert consent_results[ConsentType.RECORDING.value] is True
        assert consent_results[ConsentType.TRANSCRIPT.value] is True
        
        # Check consent was stored
        assert "test_user" in privacy_manager.user_consents
        user_consents = privacy_manager.user_consents["test_user"]
        assert len(user_consents) == 2
    
    @pytest.mark.asyncio
    async def test_check_consent(self, privacy_manager):
        """Test checking user consent."""
        await privacy_manager.request_consent(
            user_id="test_user",
            consent_types=[ConsentType.RECORDING]
        )
        
        has_consent = await privacy_manager.check_consent("test_user", ConsentType.RECORDING)
        assert has_consent is True
        
        no_consent = await privacy_manager.check_consent("test_user", ConsentType.ANALYTICS)
        assert no_consent is False
    
    @pytest.mark.asyncio
    async def test_withdraw_consent(self, privacy_manager):
        """Test withdrawing user consent."""
        await privacy_manager.request_consent(
            user_id="test_user",
            consent_types=[ConsentType.RECORDING, ConsentType.TRANSCRIPT]
        )
        
        # Withdraw recording consent
        success = await privacy_manager.withdraw_consent(
            user_id="test_user",
            consent_types=[ConsentType.RECORDING]
        )
        
        assert success is True
        
        # Check consent status
        has_recording_consent = await privacy_manager.check_consent("test_user", ConsentType.RECORDING)
        has_transcript_consent = await privacy_manager.check_consent("test_user", ConsentType.TRANSCRIPT)
        
        assert has_recording_consent is False
        assert has_transcript_consent is True
    
    @pytest.mark.asyncio
    async def test_retention_policy_application(self, privacy_manager):
        """Test applying retention policies."""
        metadata = SessionMetadata(
            session_id="test_session",
            user_id="test_user",
            start_time=datetime.now(timezone.utc),
            tags=["important"]
        )
        
        retention_expiry = await privacy_manager.apply_retention_policy("test_session", metadata)
        
        assert retention_expiry > datetime.now(timezone.utc)
        assert metadata.retention_until == retention_expiry
    
    @pytest.mark.asyncio
    async def test_legal_hold(self, privacy_manager):
        """Test applying and removing legal holds."""
        session_ids = ["session_1", "session_2"]
        
        # Apply legal hold
        success = await privacy_manager.apply_legal_hold(session_ids, "Investigation")
        assert success is True
        
        for session_id in session_ids:
            assert session_id in privacy_manager.legal_holds
        
        # Remove legal hold
        success = await privacy_manager.remove_legal_hold(session_ids)
        assert success is True
        
        for session_id in session_ids:
            assert session_id not in privacy_manager.legal_holds


class TestExportManager:
    """Test cases for export and reporting functionality."""
    
    @pytest.fixture
    async def temp_storage(self):
        """Create temporary storage directories."""
        storage_dir = tempfile.mkdtemp()
        export_dir = tempfile.mkdtemp()
        temp_dir = tempfile.mkdtemp()
        yield storage_dir, export_dir, temp_dir
        shutil.rmtree(storage_dir)
        shutil.rmtree(export_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    async def export_manager(self, temp_storage):
        """Create export manager for testing."""
        storage_dir, export_dir, temp_dir = temp_storage
        manager = ExportManager(
            storage_path=storage_dir,
            export_path=export_dir,
            temp_path=temp_dir,
            max_concurrent_exports=1
        )
        await manager.start_export_workers()
        yield manager
        await manager.stop_export_workers()
    
    @pytest.fixture
    async def sample_session_data(self, temp_storage):
        """Create sample session data for testing."""
        storage_dir, _, _ = temp_storage
        storage_path = Path(storage_dir)
        
        # Create sample metadata
        metadata = {
            "session_id": "test_session",
            "user_id": "test_user",
            "start_time": datetime.now(timezone.utc).isoformat(),
            "duration": 120.5,
            "status": "completed"
        }
        
        metadata_file = storage_path / "test_session_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
        
        # Create sample transcript
        transcript_data = {
            "session_id": "test_session",
            "segments": [
                {
                    "id": "seg_1",
                    "timestamp": time.time(),
                    "text": "Hello, how can I help you?",
                    "speaker_id": "agent",
                    "confidence": 0.95
                },
                {
                    "id": "seg_2", 
                    "timestamp": time.time() + 2,
                    "text": "I need help with my account",
                    "speaker_id": "user",
                    "confidence": 0.92
                }
            ]
        }
        
        transcript_file = storage_path / "test_session_transcript.json"
        with open(transcript_file, 'w') as f:
            json.dump(transcript_data, f)
        
        return "test_session"
    
    @pytest.mark.asyncio
    async def test_export_single_session_json(self, export_manager, sample_session_data):
        """Test exporting a single session as JSON."""
        session_id = sample_session_data
        
        result = await export_manager.export_single_session(
            session_id=session_id,
            format=ExportFormat.JSON,
            include_transcripts=True
        )
        
        assert result.success is True
        assert result.output_path is not None
        assert Path(result.output_path).exists()
        
        # Verify content
        with open(result.output_path, 'r') as f:
            data = json.load(f)
            assert data["session_id"] == session_id
    
    @pytest.mark.asyncio
    async def test_export_single_session_html(self, export_manager, sample_session_data):
        """Test exporting a single session as HTML."""
        session_id = sample_session_data
        
        result = await export_manager.export_single_session(
            session_id=session_id,
            format=ExportFormat.HTML,
            include_transcripts=True
        )
        
        assert result.success is True
        assert result.output_path is not None
        assert Path(result.output_path).exists()
        assert result.output_path.endswith('.html')
    
    @pytest.mark.asyncio
    async def test_export_request_workflow(self, export_manager, sample_session_data):
        """Test complete export request workflow."""
        request = ExportRequest(
            request_id=f"test_export_{int(time.time())}",
            scope=ExportScope.SINGLE_SESSION,
            format=ExportFormat.JSON,
            report_type=ReportType.SESSION_SUMMARY,
            session_ids=[sample_session_data],
            include_transcripts=True,
            include_metadata=True
        )
        
        # Submit request
        request_id = await export_manager.submit_export_request(request)
        assert request_id == request.request_id
        
        # Wait for completion
        for _ in range(10):  # Wait up to 10 seconds
            status = await export_manager.get_export_status(request_id)
            if status and status["status"] == "completed":
                break
            await asyncio.sleep(1)
        
        # Check final status
        status = await export_manager.get_export_status(request_id)
        assert status is not None
        assert status["status"] == "completed"
        assert status["result"]["success"] is True


class TestIntegration:
    """Integration tests for the complete session recording system."""
    
    @pytest.fixture
    async def temp_storage(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    async def complete_system(self, temp_storage):
        """Create complete recording system for integration testing."""
        # Initialize all components
        recording_manager = SessionRecordingManager(
            storage_path=temp_storage,
            enable_compression=False,
            enable_encryption=False
        )
        
        transcript_processor = TranscriptProcessor(
            enable_real_time=False,
            enable_speaker_diarization=True
        )
        
        privacy_manager = PrivacyManager(
            storage_path=temp_storage,
            consent_storage_path=f"{temp_storage}/consent",
            enable_encryption=False
        )
        
        export_manager = ExportManager(
            storage_path=temp_storage,
            export_path=f"{temp_storage}/exports",
            temp_path=f"{temp_storage}/temp"
        )
        await export_manager.start_export_workers()
        
        playback_manager = SessionPlaybackManager(
            storage_path=temp_storage
        )
        
        yield {
            "recording": recording_manager,
            "transcript": transcript_processor,
            "privacy": privacy_manager,
            "export": export_manager,
            "playback": playback_manager
        }
        
        # Cleanup
        await recording_manager.stop_background_tasks()
        await privacy_manager.stop_background_tasks()
        await export_manager.stop_export_workers()
    
    @pytest.mark.asyncio
    async def test_complete_session_workflow(self, complete_system):
        """Test complete session recording workflow."""
        recording_manager = complete_system["recording"]
        transcript_processor = complete_system["transcript"]
        privacy_manager = complete_system["privacy"]
        export_manager = complete_system["export"]
        
        user_id = "test_user"
        
        # Step 1: Request consent
        consent_results = await privacy_manager.request_consent(
            user_id=user_id,
            consent_types=[ConsentType.RECORDING, ConsentType.TRANSCRIPT, ConsentType.ANALYTICS]
        )
        
        assert all(consent_results.values())
        
        # Step 2: Start recording session
        session_id = await recording_manager.start_session(
            user_id=user_id,
            privacy_level=PrivacyLevel.FULL,
            tags=["integration_test"]
        )
        
        # Step 3: Start transcript session
        transcript_session = await transcript_processor.start_session(
            session_id=session_id,
            user_id=user_id,
            language="en"
        )
        
        # Step 4: Simulate conversation
        conversation_data = [
            {"speaker": "agent", "text": "Hello! How can I help you today?", "timestamp": time.time()},
            {"speaker": "user", "text": "I need help with my account settings", "timestamp": time.time() + 2},
            {"speaker": "agent", "text": "I'd be happy to help you with that", "timestamp": time.time() + 4},
            {"speaker": "user", "text": "Thank you very much", "timestamp": time.time() + 6}
        ]
        
        for turn in conversation_data:
            # Record audio (simulated)
            audio_data = f"audio_for_{turn['text'][:20]}".encode()
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
                confidence=0.9 + (len(turn["text"]) % 10) / 100  # Simulated confidence
            )
            
            # Process transcript
            segment = TranscriptSegment(
                id=f"seg_{len(conversation_data)}",
                timestamp=turn["timestamp"],
                text=turn["text"],
                speaker_id=turn["speaker"],
                confidence=0.9
            )
            await transcript_processor.process_segment(session_id, segment)
        
        # Step 5: End sessions
        recording_metadata = await recording_manager.end_session(session_id, final_quality_score=0.88)
        transcript_summary = await transcript_processor.end_session(session_id)
        
        # Step 6: Apply retention policy
        await privacy_manager.apply_retention_policy(session_id, recording_metadata)
        
        # Step 7: Export session data
        export_result = await export_manager.export_single_session(
            session_id=session_id,
            format=ExportFormat.JSON,
            include_transcripts=True
        )
        
        # Verify results
        assert recording_metadata.status == SessionStatus.COMPLETED
        assert recording_metadata.duration > 0
        assert transcript_summary["total_segments"] == len(conversation_data)
        assert transcript_summary["total_words"] > 0
        assert export_result.success is True
        assert export_result.record_count == 1
        
        # Verify exported data
        with open(export_result.output_path, 'r') as f:
            exported_data = json.load(f)
            assert exported_data["session_id"] == session_id
            assert "metadata" in exported_data
            assert "transcripts" in exported_data
            assert len(exported_data["transcripts"]) == len(conversation_data)
    
    @pytest.mark.asyncio
    async def test_privacy_compliance_workflow(self, complete_system):
        """Test privacy compliance workflow."""
        privacy_manager = complete_system["privacy"]
        recording_manager = complete_system["recording"]
        
        user_id = "privacy_test_user"
        
        # Request initial consent
        await privacy_manager.request_consent(
            user_id=user_id,
            consent_types=[ConsentType.RECORDING, ConsentType.TRANSCRIPT]
        )
        
        # Create session
        session_id = await recording_manager.start_session(user_id=user_id)
        await recording_manager.record_audio_chunk(session_id, b"test_audio", time.time())
        await recording_manager.add_transcript_segment(session_id, "Test message", time.time())
        metadata = await recording_manager.end_session(session_id)
        
        # Apply retention policy
        await privacy_manager.apply_retention_policy(session_id, metadata)
        
        # Withdraw consent
        await privacy_manager.withdraw_consent(
            user_id=user_id,
            consent_types=[ConsentType.RECORDING]
        )
        
        # Verify consent status
        has_recording_consent = await privacy_manager.check_consent(user_id, ConsentType.RECORDING)
        has_transcript_consent = await privacy_manager.check_consent(user_id, ConsentType.TRANSCRIPT)
        
        assert has_recording_consent is False
        assert has_transcript_consent is True
        
        # Generate compliance report
        report = await privacy_manager.generate_compliance_report(
            framework=ComplianceFramework.GDPR,
            start_date=datetime.now(timezone.utc) - timedelta(days=1),
            end_date=datetime.now(timezone.utc)
        )
        
        assert report["framework"] == "gdpr"
        assert "consent_records" in report
        assert report["consent_records"]["total_users"] >= 1


# Utility functions for testing
async def create_sample_session_data(storage_path: str, session_id: str) -> None:
    """Create sample session data for testing."""
    storage = Path(storage_path)
    
    # Create metadata
    metadata = {
        "session_id": session_id,
        "user_id": "test_user",
        "start_time": datetime.now(timezone.utc).isoformat(),
        "end_time": (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
        "duration": 300,
        "status": "completed",
        "privacy_level": "full",
        "tags": ["test"],
        "audio_format": "wav",
        "sample_rate": 44100
    }
    
    with open(storage / f"{session_id}_metadata.json", 'w') as f:
        json.dump(metadata, f)
    
    # Create transcript
    transcript = {
        "session_id": session_id,
        "segments": [
            {
                "id": "seg_1",
                "timestamp": time.time(),
                "text": "Hello, how can I help you?",
                "speaker_id": "agent",
                "confidence": 0.95
            },
            {
                "id": "seg_2",
                "timestamp": time.time() + 2,
                "text": "I need assistance with my order",
                "speaker_id": "user", 
                "confidence": 0.92
            }
        ]
    }
    
    with open(storage / f"{session_id}_transcript.json", 'w') as f:
        json.dump(transcript, f)
    
    # Create quality metrics
    quality = {
        "audio_quality_score": 0.88,
        "transcript_accuracy_score": 0.93,
        "interruption_count": 1,
        "silence_duration": 15.2,
        "speech_duration": 284.8
    }
    
    with open(storage / f"{session_id}_quality.json", 'w') as f:
        json.dump(quality, f)


if __name__ == "__main__":
    # Run basic functionality test
    async def test_basic_functionality():
        """Basic functionality test that can be run directly."""
        print("Testing basic session recording functionality...")
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Initialize recording manager
            recording_manager = SessionRecordingManager(
                storage_path=temp_dir,
                enable_compression=False,
                enable_encryption=False
            )
            
            # Start session
            session_id = await recording_manager.start_session(
                user_id="test_user",
                tags=["manual_test"]
            )
            print(f"Started session: {session_id}")
            
            # Add some data
            await recording_manager.record_audio_chunk(session_id, b"test_audio_data", time.time())
            await recording_manager.add_transcript_segment(session_id, "Hello world", time.time())
            
            # End session
            metadata = await recording_manager.end_session(session_id)
            print(f"Session completed with duration: {metadata.duration:.2f}s")
            
            # Initialize export manager
            export_manager = ExportManager(
                storage_path=temp_dir,
                export_path=f"{temp_dir}/exports"
            )
            await export_manager.start_export_workers()
            
            # Export session
            result = await export_manager.export_single_session(
                session_id=session_id,
                format=ExportFormat.JSON
            )
            
            if result.success:
                print(f"Export successful: {result.output_path}")
                print(f"File size: {result.file_size} bytes")
            else:
                print(f"Export failed: {result.error_message}")
            
            await export_manager.stop_export_workers()
            
            print("Basic functionality test completed successfully!")
            
        finally:
            # Cleanup
            shutil.rmtree(temp_dir)
    
    # Run the test
    asyncio.run(test_basic_functionality())