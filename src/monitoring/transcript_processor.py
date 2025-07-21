"""
Real-time Transcript Generation and Processing System

This module provides comprehensive transcript generation, processing, and storage
capabilities for voice agent sessions, including real-time transcription,
post-processing, and multiple output formats.
"""

import asyncio
import json
import time
import uuid
import re
from typing import Dict, List, Optional, Any, Union, Callable, AsyncIterator
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
import logging
from io import StringIO

from .structured_logging import StructuredLogger
from .session_recording import TranscriptSegment, TranscriptFormat


class TranscriptProcessor:
    """
    Advanced transcript processor with real-time generation and post-processing.
    
    Handles transcript generation from STT providers, real-time processing,
    speaker diarization, sentiment analysis, and export to multiple formats.
    """
    
    def __init__(
        self,
        enable_real_time: bool = True,
        enable_speaker_diarization: bool = True,
        enable_sentiment_analysis: bool = False,
        enable_keyword_extraction: bool = True,
        enable_summary_generation: bool = False,
        confidence_threshold: float = 0.7,
        merge_similar_segments: bool = True,
        segment_merge_threshold: float = 2.0,  # seconds
        logger: Optional[StructuredLogger] = None
    ):
        """
        Initialize transcript processor.
        
        Args:
            enable_real_time: Enable real-time transcript processing
            enable_speaker_diarization: Enable speaker identification
            enable_sentiment_analysis: Enable sentiment analysis
            enable_keyword_extraction: Enable keyword extraction
            enable_summary_generation: Enable summary generation
            confidence_threshold: Minimum confidence for segments
            merge_similar_segments: Whether to merge similar segments
            segment_merge_threshold: Time threshold for merging segments
            logger: Optional logger instance
        """
        self.enable_real_time = enable_real_time
        self.enable_speaker_diarization = enable_speaker_diarization
        self.enable_sentiment_analysis = enable_sentiment_analysis
        self.enable_keyword_extraction = enable_keyword_extraction
        self.enable_summary_generation = enable_summary_generation
        self.confidence_threshold = confidence_threshold
        self.merge_similar_segments = merge_similar_segments
        self.segment_merge_threshold = segment_merge_threshold
        
        self.logger = logger or StructuredLogger(__name__, "transcript_processor")
        
        # Processing state
        self.active_sessions: Dict[str, 'TranscriptSession'] = {}
        self.processors: Dict[str, Callable] = {}
        
        # Real-time callbacks
        self.callbacks: Dict[str, List[Callable]] = {
            "on_segment_processed": [],
            "on_session_summary": [],
            "on_speaker_identified": [],
            "on_keyword_extracted": [],
            "on_sentiment_analyzed": []
        }
        
        self.logger.info(
            "Transcript processor initialized",
            extra_data={
                "real_time": self.enable_real_time,
                "speaker_diarization": self.enable_speaker_diarization,
                "sentiment_analysis": self.enable_sentiment_analysis,
                "keyword_extraction": self.enable_keyword_extraction,
                "summary_generation": self.enable_summary_generation,
                "confidence_threshold": self.confidence_threshold
            }
        )
    
    async def start_session(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        language: str = "en",
        custom_vocabulary: Optional[List[str]] = None,
        speaker_profiles: Optional[Dict[str, Any]] = None
    ) -> 'TranscriptSession':
        """
        Start a new transcript processing session.
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            language: Language code for processing
            custom_vocabulary: Custom vocabulary for recognition
            speaker_profiles: Known speaker profiles
            
        Returns:
            TranscriptSession instance
        """
        if session_id in self.active_sessions:
            raise ValueError(f"Session {session_id} already active")
        
        session = TranscriptSession(
            session_id=session_id,
            user_id=user_id,
            language=language,
            custom_vocabulary=custom_vocabulary or [],
            speaker_profiles=speaker_profiles or {},
            processor=self,
            logger=self.logger
        )
        
        await session.initialize()
        self.active_sessions[session_id] = session
        
        self.logger.info(
            f"Transcript session started: {session_id}",
            extra_data={
                "session_id": session_id,
                "user_id": user_id,
                "language": language,
                "custom_vocabulary_size": len(custom_vocabulary or []),
                "speaker_profiles": len(speaker_profiles or {})
            }
        )
        
        return session
    
    async def process_segment(
        self,
        session_id: str,
        segment: TranscriptSegment
    ) -> TranscriptSegment:
        """
        Process a transcript segment.
        
        Args:
            session_id: Session identifier
            segment: Transcript segment to process
            
        Returns:
            Processed transcript segment
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        processed_segment = await session.process_segment(segment)
        
        # Trigger callbacks
        await self._trigger_callbacks("on_segment_processed", session_id, processed_segment)
        
        return processed_segment
    
    async def end_session(self, session_id: str) -> Dict[str, Any]:
        """
        End a transcript processing session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session summary and analytics
        """
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        session = self.active_sessions[session_id]
        summary = await session.finalize()
        
        # Remove from active sessions
        del self.active_sessions[session_id]
        
        # Trigger callbacks
        await self._trigger_callbacks("on_session_summary", session_id, summary)
        
        self.logger.info(
            f"Transcript session ended: {session_id}",
            extra_data={
                "session_id": session_id,
                "total_segments": summary.get("total_segments", 0),
                "total_words": summary.get("total_words", 0),
                "duration": summary.get("duration", 0),
                "speakers": summary.get("speakers", [])
            }
        )
        
        return summary
    
    async def export_transcript(
        self,
        session_id: str,
        format: TranscriptFormat,
        output_path: Optional[str] = None,
        include_metadata: bool = True,
        include_timestamps: bool = True,
        include_confidence: bool = False
    ) -> str:
        """
        Export transcript in specified format.
        
        Args:
            session_id: Session identifier
            format: Output format
            output_path: Output file path
            include_metadata: Whether to include metadata
            include_timestamps: Whether to include timestamps
            include_confidence: Whether to include confidence scores
            
        Returns:
            Path to exported file or content string
        """
        if session_id in self.active_sessions:
            # Active session
            session = self.active_sessions[session_id]
            segments = session.segments
        else:
            # Load from storage if available
            segments = await self._load_segments_from_storage(session_id)
        
        if not segments:
            raise ValueError(f"No transcript data found for session {session_id}")
        
        exporter = TranscriptExporter(
            segments=segments,
            session_id=session_id,
            logger=self.logger
        )
        
        return await exporter.export(
            format=format,
            output_path=output_path,
            include_metadata=include_metadata,
            include_timestamps=include_timestamps,
            include_confidence=include_confidence
        )
    
    def add_callback(self, event: str, callback: Callable):
        """Add a callback for transcript events."""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
    
    def remove_callback(self, event: str, callback: Callable):
        """Remove a callback for transcript events."""
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
                    f"Error in transcript callback: {event}",
                    extra_data={"error": str(e), "event": event}
                )
    
    async def _load_segments_from_storage(self, session_id: str) -> List[TranscriptSegment]:
        """Load transcript segments from storage."""
        # This would load from the session recording storage
        # For now, return empty list
        return []


class TranscriptSession:
    """
    Individual transcript processing session.
    
    Handles real-time transcript processing, speaker diarization,
    and analytics for a single voice session.
    """
    
    def __init__(
        self,
        session_id: str,
        user_id: Optional[str],
        language: str,
        custom_vocabulary: List[str],
        speaker_profiles: Dict[str, Any],
        processor: TranscriptProcessor,
        logger: StructuredLogger
    ):
        self.session_id = session_id
        self.user_id = user_id
        self.language = language
        self.custom_vocabulary = custom_vocabulary
        self.speaker_profiles = speaker_profiles
        self.processor = processor
        self.logger = logger
        
        # Session data
        self.segments: List[TranscriptSegment] = []
        self.speakers: Dict[str, Dict[str, Any]] = {}
        self.keywords: Dict[str, int] = {}
        self.sentiment_history: List[Dict[str, Any]] = []
        
        # Processing state
        self.is_initialized = False
        self.start_time = time.time()
        self.last_segment_time = 0.0
        
        # Real-time processing
        self.pending_segments: List[TranscriptSegment] = []
        self.processing_task: Optional[asyncio.Task] = None
    
    async def initialize(self):
        """Initialize the transcript session."""
        if self.is_initialized:
            return
        
        # Start real-time processing if enabled
        if self.processor.enable_real_time:
            self.processing_task = asyncio.create_task(self._real_time_processor())
        
        self.is_initialized = True
        
        self.logger.info(
            f"Transcript session initialized: {self.session_id}",
            extra_data={
                "session_id": self.session_id,
                "language": self.language,
                "real_time": self.processor.enable_real_time
            }
        )
    
    async def process_segment(self, segment: TranscriptSegment) -> TranscriptSegment:
        """
        Process a transcript segment.
        
        Args:
            segment: Raw transcript segment
            
        Returns:
            Processed transcript segment
        """
        # Apply confidence filtering
        if segment.confidence and segment.confidence < self.processor.confidence_threshold:
            self.logger.debug(
                f"Segment filtered due to low confidence: {segment.confidence}",
                extra_data={
                    "session_id": self.session_id,
                    "segment_id": segment.id,
                    "confidence": segment.confidence,
                    "threshold": self.processor.confidence_threshold
                }
            )
            return segment
        
        # Clean and normalize text
        processed_segment = await self._clean_text(segment)
        
        # Speaker diarization
        if self.processor.enable_speaker_diarization:
            processed_segment = await self._identify_speaker(processed_segment)
        
        # Keyword extraction
        if self.processor.enable_keyword_extraction:
            await self._extract_keywords(processed_segment)
        
        # Sentiment analysis
        if self.processor.enable_sentiment_analysis:
            await self._analyze_sentiment(processed_segment)
        
        # Merge with previous segments if configured
        if self.processor.merge_similar_segments:
            processed_segment = await self._maybe_merge_segment(processed_segment)
        
        # Add to segments list
        if processed_segment:
            self.segments.append(processed_segment)
            self.last_segment_time = processed_segment.timestamp
        
        return processed_segment
    
    async def finalize(self) -> Dict[str, Any]:
        """
        Finalize the transcript session.
        
        Returns:
            Session summary and analytics
        """
        # Stop real-time processing
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        # Generate session summary
        summary = await self._generate_session_summary()
        
        self.logger.info(
            f"Transcript session finalized: {self.session_id}",
            extra_data={
                "session_id": self.session_id,
                "summary": summary
            }
        )
        
        return summary
    
    async def _clean_text(self, segment: TranscriptSegment) -> TranscriptSegment:
        """Clean and normalize transcript text."""
        text = segment.text
        
        # Remove filler words and normalize
        filler_words = ['um', 'uh', 'ah', 'er', 'like', 'you know']
        words = text.split()
        cleaned_words = [word for word in words if word.lower() not in filler_words]
        
        # Custom vocabulary replacements
        for vocab_word in self.custom_vocabulary:
            text = re.sub(
                r'\b' + re.escape(vocab_word.lower()) + r'\b',
                vocab_word,
                text,
                flags=re.IGNORECASE
            )
        
        # Basic text normalization
        text = re.sub(r'\s+', ' ', text).strip()
        text = text.capitalize() if text else text
        
        # Create new segment with cleaned text
        cleaned_segment = TranscriptSegment(
            id=segment.id,
            timestamp=segment.timestamp,
            end_timestamp=segment.end_timestamp,
            text=text,
            speaker_id=segment.speaker_id,
            confidence=segment.confidence,
            language=segment.language,
            is_final=segment.is_final,
            words=segment.words
        )
        
        return cleaned_segment
    
    async def _identify_speaker(self, segment: TranscriptSegment) -> TranscriptSegment:
        """Identify speaker for the segment."""
        # Simple speaker identification based on voice characteristics
        # In a real implementation, this would use ML models
        
        if segment.speaker_id:
            # Speaker already identified
            return segment
        
        # Placeholder speaker identification
        # This would use audio features, voice patterns, etc.
        estimated_speaker = "Speaker_1"  # Default speaker
        
        # Check if we have speaker profiles
        if self.speaker_profiles:
            # Match against known speaker profiles
            # This is a simplified version
            for speaker_id, profile in self.speaker_profiles.items():
                # In reality, this would compare voice features
                estimated_speaker = speaker_id
                break
        
        # Update speaker tracking
        if estimated_speaker not in self.speakers:
            self.speakers[estimated_speaker] = {
                "first_seen": segment.timestamp,
                "segment_count": 0,
                "total_words": 0,
                "avg_confidence": 0.0
            }
        
        speaker_data = self.speakers[estimated_speaker]
        speaker_data["segment_count"] += 1
        speaker_data["total_words"] += len(segment.text.split())
        
        if segment.confidence:
            speaker_data["avg_confidence"] = (
                (speaker_data["avg_confidence"] * (speaker_data["segment_count"] - 1) + segment.confidence) /
                speaker_data["segment_count"]
            )
        
        # Create new segment with speaker ID
        speaker_segment = TranscriptSegment(
            id=segment.id,
            timestamp=segment.timestamp,
            end_timestamp=segment.end_timestamp,
            text=segment.text,
            speaker_id=estimated_speaker,
            confidence=segment.confidence,
            language=segment.language,
            is_final=segment.is_final,
            words=segment.words
        )
        
        return speaker_segment
    
    async def _extract_keywords(self, segment: TranscriptSegment):
        """Extract keywords from the segment."""
        # Simple keyword extraction
        # In a real implementation, this would use NLP libraries
        
        words = segment.text.lower().split()
        
        # Filter out common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you',
            'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }
        
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]
        
        # Update keyword frequency
        for keyword in keywords:
            self.keywords[keyword] = self.keywords.get(keyword, 0) + 1
    
    async def _analyze_sentiment(self, segment: TranscriptSegment):
        """Analyze sentiment of the segment."""
        # Simple sentiment analysis
        # In a real implementation, this would use ML models
        
        text = segment.text.lower()
        
        # Simple positive/negative word lists
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'happy', 'pleased']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'sad', 'angry', 'frustrated', 'disappointed']
        
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        if positive_count > negative_count:
            sentiment = "positive"
            score = 0.6 + (positive_count - negative_count) * 0.1
        elif negative_count > positive_count:
            sentiment = "negative"
            score = 0.4 - (negative_count - positive_count) * 0.1
        else:
            sentiment = "neutral"
            score = 0.5
        
        # Clamp score between 0 and 1
        score = max(0.0, min(1.0, score))
        
        sentiment_data = {
            "timestamp": segment.timestamp,
            "sentiment": sentiment,
            "score": score,
            "text": segment.text
        }
        
        self.sentiment_history.append(sentiment_data)
    
    async def _maybe_merge_segment(self, segment: TranscriptSegment) -> Optional[TranscriptSegment]:
        """Maybe merge segment with previous segment if similar."""
        if not self.segments:
            return segment
        
        last_segment = self.segments[-1]
        
        # Check if segments should be merged
        time_diff = segment.timestamp - (last_segment.end_timestamp or last_segment.timestamp)
        same_speaker = segment.speaker_id == last_segment.speaker_id
        
        if time_diff <= self.processor.segment_merge_threshold and same_speaker:
            # Merge segments
            merged_text = f"{last_segment.text} {segment.text}".strip()
            
            merged_segment = TranscriptSegment(
                id=last_segment.id,  # Keep original ID
                timestamp=last_segment.timestamp,
                end_timestamp=segment.end_timestamp or segment.timestamp,
                text=merged_text,
                speaker_id=segment.speaker_id,
                confidence=min(last_segment.confidence or 1.0, segment.confidence or 1.0),
                language=segment.language,
                is_final=segment.is_final,
                words=(last_segment.words or []) + (segment.words or [])
            )
            
            # Replace last segment with merged segment
            self.segments[-1] = merged_segment
            
            return None  # Don't add as new segment
        
        return segment
    
    async def _generate_session_summary(self) -> Dict[str, Any]:
        """Generate comprehensive session summary."""
        duration = time.time() - self.start_time
        
        # Basic statistics
        total_segments = len(self.segments)
        total_words = sum(len(segment.text.split()) for segment in self.segments)
        avg_confidence = sum(segment.confidence or 0 for segment in self.segments) / total_segments if total_segments > 0 else 0
        
        # Speaker statistics
        speaker_stats = {}
        for speaker_id, data in self.speakers.items():
            speaker_stats[speaker_id] = {
                "segment_count": data["segment_count"],
                "word_count": data["total_words"],
                "avg_confidence": data["avg_confidence"],
                "speaking_time_percentage": (data["segment_count"] / total_segments * 100) if total_segments > 0 else 0
            }
        
        # Top keywords
        top_keywords = sorted(self.keywords.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Sentiment summary
        sentiment_summary = {}
        if self.sentiment_history:
            sentiments = [s["sentiment"] for s in self.sentiment_history]
            sentiment_summary = {
                "positive_count": sentiments.count("positive"),
                "negative_count": sentiments.count("negative"),
                "neutral_count": sentiments.count("neutral"),
                "avg_score": sum(s["score"] for s in self.sentiment_history) / len(self.sentiment_history)
            }
        
        summary = {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "language": self.language,
            "duration": duration,
            "total_segments": total_segments,
            "total_words": total_words,
            "avg_confidence": avg_confidence,
            "speakers": list(self.speakers.keys()),
            "speaker_stats": speaker_stats,
            "top_keywords": top_keywords,
            "sentiment_summary": sentiment_summary,
            "start_time": self.start_time,
            "end_time": time.time()
        }
        
        return summary
    
    async def _real_time_processor(self):
        """Real-time processing task for pending segments."""
        try:
            while True:
                if self.pending_segments:
                    # Process pending segments
                    segments_to_process = self.pending_segments.copy()
                    self.pending_segments.clear()
                    
                    for segment in segments_to_process:
                        await self.process_segment(segment)
                
                await asyncio.sleep(0.1)  # Process every 100ms
                
        except asyncio.CancelledError:
            # Process any remaining segments before exiting
            for segment in self.pending_segments:
                await self.process_segment(segment)


class TranscriptExporter:
    """
    Export transcript data to various formats.
    
    Supports JSON, plain text, SRT, VTT, and DOCX formats
    with configurable metadata inclusion.
    """
    
    def __init__(
        self,
        segments: List[TranscriptSegment],
        session_id: str,
        logger: Optional[StructuredLogger] = None
    ):
        self.segments = segments
        self.session_id = session_id
        self.logger = logger or StructuredLogger(__name__, "transcript_exporter")
    
    async def export(
        self,
        format: TranscriptFormat,
        output_path: Optional[str] = None,
        include_metadata: bool = True,
        include_timestamps: bool = True,
        include_confidence: bool = False
    ) -> str:
        """
        Export transcript in specified format.
        
        Args:
            format: Output format
            output_path: Output file path
            include_metadata: Whether to include metadata
            include_timestamps: Whether to include timestamps
            include_confidence: Whether to include confidence scores
            
        Returns:
            Path to exported file or content string
        """
        if format == TranscriptFormat.JSON:
            return await self._export_json(output_path, include_metadata, include_confidence)
        elif format == TranscriptFormat.TEXT:
            return await self._export_text(output_path, include_timestamps, include_confidence)
        elif format == TranscriptFormat.SRT:
            return await self._export_srt(output_path)
        elif format == TranscriptFormat.VTT:
            return await self._export_vtt(output_path)
        elif format == TranscriptFormat.DOCX:
            return await self._export_docx(output_path, include_timestamps, include_confidence)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    async def _export_json(self, output_path: Optional[str], include_metadata: bool, include_confidence: bool) -> str:
        """Export as JSON format."""
        data = {
            "session_id": self.session_id,
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "segments": []
        }
        
        for segment in self.segments:
            segment_data = segment.to_dict()
            
            if not include_confidence:
                segment_data.pop("confidence", None)
            
            if not include_metadata:
                # Remove metadata fields
                for field in ["words", "language"]:
                    segment_data.pop(field, None)
            
            data["segments"].append(segment_data)
        
        content = json.dumps(data, indent=2, ensure_ascii=False)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return output_path
        
        return content
    
    async def _export_text(self, output_path: Optional[str], include_timestamps: bool, include_confidence: bool) -> str:
        """Export as plain text format."""
        lines = []
        
        for segment in self.segments:
            line_parts = []
            
            if include_timestamps:
                timestamp_str = datetime.fromtimestamp(segment.timestamp).strftime("%H:%M:%S")
                line_parts.append(f"[{timestamp_str}]")
            
            if segment.speaker_id:
                line_parts.append(f"{segment.speaker_id}:")
            
            line_parts.append(segment.text)
            
            if include_confidence and segment.confidence:
                line_parts.append(f"(confidence: {segment.confidence:.2f})")
            
            lines.append(" ".join(line_parts))
        
        content = "\n".join(lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return output_path
        
        return content
    
    async def _export_srt(self, output_path: Optional[str]) -> str:
        """Export as SRT subtitle format."""
        lines = []
        
        for i, segment in enumerate(self.segments, 1):
            start_time = self._format_time_srt(segment.timestamp)
            end_time = self._format_time_srt(segment.end_timestamp or segment.timestamp + 2)
            
            lines.append(str(i))
            lines.append(f"{start_time} --> {end_time}")
            
            if segment.speaker_id:
                lines.append(f"{segment.speaker_id}: {segment.text}")
            else:
                lines.append(segment.text)
            
            lines.append("")  # Empty line between subtitles
        
        content = "\n".join(lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return output_path
        
        return content
    
    async def _export_vtt(self, output_path: Optional[str]) -> str:
        """Export as WebVTT format."""
        lines = ["WEBVTT", ""]
        
        for segment in self.segments:
            start_time = self._format_time_vtt(segment.timestamp)
            end_time = self._format_time_vtt(segment.end_timestamp or segment.timestamp + 2)
            
            lines.append(f"{start_time} --> {end_time}")
            
            if segment.speaker_id:
                lines.append(f"<v {segment.speaker_id}>{segment.text}")
            else:
                lines.append(segment.text)
            
            lines.append("")  # Empty line between cues
        
        content = "\n".join(lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return output_path
        
        return content
    
    async def _export_docx(self, output_path: Optional[str], include_timestamps: bool, include_confidence: bool) -> str:
        """Export as DOCX format."""
        # Note: This is a placeholder implementation
        # In a real implementation, you would use python-docx library
        
        content_lines = []
        content_lines.append(f"Transcript for Session: {self.session_id}")
        content_lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        content_lines.append("")
        
        for segment in self.segments:
            line_parts = []
            
            if include_timestamps:
                timestamp_str = datetime.fromtimestamp(segment.timestamp).strftime("%H:%M:%S")
                line_parts.append(f"[{timestamp_str}]")
            
            if segment.speaker_id:
                line_parts.append(f"{segment.speaker_id}:")
            
            line_parts.append(segment.text)
            
            if include_confidence and segment.confidence:
                line_parts.append(f"(confidence: {segment.confidence:.2f})")
            
            content_lines.append(" ".join(line_parts))
        
        content = "\n".join(content_lines)
        
        # For now, save as text file with .docx extension
        # In reality, this would create a proper DOCX file
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return output_path
        
        return content
    
    def _format_time_srt(self, timestamp: float) -> str:
        """Format timestamp for SRT format."""
        hours = int(timestamp // 3600)
        minutes = int((timestamp % 3600) // 60)
        seconds = int(timestamp % 60)
        milliseconds = int((timestamp % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
    
    def _format_time_vtt(self, timestamp: float) -> str:
        """Format timestamp for VTT format."""
        hours = int(timestamp // 3600)
        minutes = int((timestamp % 3600) // 60)
        seconds = timestamp % 60
        
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


# Utility functions for transcript processing
async def create_transcript_from_stt_stream(
    stt_stream: AsyncIterator[Dict[str, Any]],
    session_id: str,
    processor: TranscriptProcessor
) -> List[TranscriptSegment]:
    """
    Create transcript segments from STT stream.
    
    Args:
        stt_stream: Async iterator of STT results
        session_id: Session identifier
        processor: Transcript processor instance
        
    Returns:
        List of processed transcript segments
    """
    segments = []
    
    async for stt_result in stt_stream:
        # Convert STT result to TranscriptSegment
        segment = TranscriptSegment(
            id=f"seg_{uuid.uuid4().hex[:8]}",
            timestamp=stt_result.get("timestamp", time.time()),
            end_timestamp=stt_result.get("end_timestamp"),
            text=stt_result.get("text", ""),
            speaker_id=stt_result.get("speaker_id"),
            confidence=stt_result.get("confidence"),
            language=stt_result.get("language"),
            is_final=stt_result.get("is_final", False),
            words=stt_result.get("words")
        )
        
        # Process segment
        processed_segment = await processor.process_segment(session_id, segment)
        segments.append(processed_segment)
    
    return segments