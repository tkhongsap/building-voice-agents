"""
Export and Reporting Manager for Session Recordings

This module provides comprehensive export and reporting capabilities for
voice agent session recordings, including multiple format support,
batch processing, and automated report generation.
"""

import asyncio
import json
import csv
import time
import zipfile
import shutil
from typing import Dict, List, Optional, Any, Union, Callable, AsyncIterator
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
import logging
import io
import base64

from .structured_logging import StructuredLogger
from .session_recording import SessionMetadata, TranscriptSegment, QualityMetrics, SessionStatus, PrivacyLevel
from .transcript_processor import TranscriptFormat, TranscriptExporter
from .session_playback import SessionAnalyzer, AnalysisType


class ExportFormat(Enum):
    """Supported export formats."""
    JSON = "json"
    CSV = "csv"
    XML = "xml"
    EXCEL = "excel"
    PDF = "pdf"
    HTML = "html"
    ZIP = "zip"
    AUDIO_WAV = "audio_wav"
    AUDIO_MP3 = "audio_mp3"


class ReportType(Enum):
    """Types of reports that can be generated."""
    SESSION_SUMMARY = "session_summary"
    QUALITY_REPORT = "quality_report"
    TRANSCRIPT_REPORT = "transcript_report"
    ANALYTICS_REPORT = "analytics_report"
    COMPLIANCE_REPORT = "compliance_report"
    PERFORMANCE_REPORT = "performance_report"
    CUSTOM_REPORT = "custom_report"


class ExportScope(Enum):
    """Scope of data to export."""
    SINGLE_SESSION = "single_session"
    MULTIPLE_SESSIONS = "multiple_sessions"
    DATE_RANGE = "date_range"
    USER_SESSIONS = "user_sessions"
    AGENT_SESSIONS = "agent_sessions"
    ALL_SESSIONS = "all_sessions"


@dataclass
class ExportRequest:
    """Export request configuration."""
    request_id: str
    scope: ExportScope
    format: ExportFormat
    report_type: ReportType
    
    # Scope parameters
    session_ids: Optional[List[str]] = None
    user_ids: Optional[List[str]] = None
    agent_ids: Optional[List[str]] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    # Export options
    include_audio: bool = False
    include_transcripts: bool = True
    include_metadata: bool = True
    include_quality_metrics: bool = True
    include_analytics: bool = False
    
    # Privacy options
    anonymize_data: bool = False
    privacy_level: PrivacyLevel = PrivacyLevel.FULL
    
    # Output options
    output_path: Optional[str] = None
    compress_output: bool = False
    encrypt_output: bool = False
    
    # Request metadata
    requested_by: Optional[str] = None
    created_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["created_date"] = self.created_date.isoformat()
        if self.start_date:
            data["start_date"] = self.start_date.isoformat()
        if self.end_date:
            data["end_date"] = self.end_date.isoformat()
        data["scope"] = self.scope.value
        data["format"] = self.format.value
        data["report_type"] = self.report_type.value
        data["privacy_level"] = self.privacy_level.value
        return data


@dataclass
class ExportResult:
    """Result of an export operation."""
    request_id: str
    success: bool
    output_path: Optional[str] = None
    file_size: Optional[int] = None
    record_count: int = 0
    duration: float = 0.0
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    completed_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["completed_date"] = self.completed_date.isoformat()
        return data


class ExportManager:
    """
    Comprehensive export and reporting manager.
    
    Handles export requests, format conversion, batch processing,
    and automated report generation for voice agent sessions.
    """
    
    def __init__(
        self,
        storage_path: str = "./recordings",
        export_path: str = "./exports",
        temp_path: str = "./temp",
        max_concurrent_exports: int = 3,
        default_batch_size: int = 100,
        enable_compression: bool = True,
        logger: Optional[StructuredLogger] = None
    ):
        """
        Initialize export manager.
        
        Args:
            storage_path: Path to session recordings
            export_path: Path for exported files
            temp_path: Path for temporary files
            max_concurrent_exports: Maximum concurrent export operations
            default_batch_size: Default batch size for processing
            enable_compression: Enable output compression
            logger: Optional logger instance
        """
        self.storage_path = Path(storage_path)
        self.export_path = Path(export_path)
        self.temp_path = Path(temp_path)
        
        # Create directories
        self.export_path.mkdir(parents=True, exist_ok=True)
        self.temp_path.mkdir(parents=True, exist_ok=True)
        
        self.max_concurrent_exports = max_concurrent_exports
        self.default_batch_size = default_batch_size
        self.enable_compression = enable_compression
        
        # Active exports
        self.active_exports: Dict[str, asyncio.Task] = {}
        self.export_history: List[ExportResult] = []
        
        # Export queue
        self.export_queue: asyncio.Queue = asyncio.Queue()
        self.export_workers: List[asyncio.Task] = []
        
        # Format handlers
        self.format_handlers: Dict[ExportFormat, Callable] = {
            ExportFormat.JSON: self._export_json,
            ExportFormat.CSV: self._export_csv,
            ExportFormat.XML: self._export_xml,
            ExportFormat.HTML: self._export_html,
            ExportFormat.PDF: self._export_pdf,
            ExportFormat.ZIP: self._export_zip,
            ExportFormat.AUDIO_WAV: self._export_audio_wav,
            ExportFormat.AUDIO_MP3: self._export_audio_mp3
        }
        
        # Report generators
        self.report_generators: Dict[ReportType, Callable] = {
            ReportType.SESSION_SUMMARY: self._generate_session_summary_report,
            ReportType.QUALITY_REPORT: self._generate_quality_report,
            ReportType.TRANSCRIPT_REPORT: self._generate_transcript_report,
            ReportType.ANALYTICS_REPORT: self._generate_analytics_report,
            ReportType.COMPLIANCE_REPORT: self._generate_compliance_report,
            ReportType.PERFORMANCE_REPORT: self._generate_performance_report,
            ReportType.CUSTOM_REPORT: self._generate_custom_report
        }
        
        # Export callbacks
        self.callbacks: Dict[str, List[Callable]] = {
            "on_export_start": [],
            "on_export_progress": [],
            "on_export_complete": [],
            "on_export_error": []
        }
        
        self.logger = logger or StructuredLogger(__name__, "export_manager")
        
        self.logger.info(
            "Export manager initialized",
            extra_data={
                "storage_path": str(self.storage_path),
                "export_path": str(self.export_path),
                "max_concurrent_exports": self.max_concurrent_exports,
                "compression_enabled": self.enable_compression
            }
        )
    
    async def submit_export_request(self, request: ExportRequest) -> str:
        """
        Submit an export request for processing.
        
        Args:
            request: Export request configuration
            
        Returns:
            Request ID for tracking
        """
        # Validate request
        await self._validate_export_request(request)
        
        # Queue the request
        await self.export_queue.put(request)
        
        self.logger.info(
            f"Export request submitted: {request.request_id}",
            extra_data={
                "request_id": request.request_id,
                "scope": request.scope.value,
                "format": request.format.value,
                "report_type": request.report_type.value
            }
        )
        
        return request.request_id
    
    async def get_export_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of an export request.
        
        Args:
            request_id: Request identifier
            
        Returns:
            Export status information
        """
        # Check active exports
        if request_id in self.active_exports:
            task = self.active_exports[request_id]
            return {
                "request_id": request_id,
                "status": "running" if not task.done() else "completed",
                "progress": "processing"  # Could be enhanced with actual progress
            }
        
        # Check export history
        for result in self.export_history:
            if result.request_id == request_id:
                return {
                    "request_id": request_id,
                    "status": "completed" if result.success else "failed",
                    "result": result.to_dict()
                }
        
        return None
    
    async def cancel_export(self, request_id: str) -> bool:
        """
        Cancel an active export request.
        
        Args:
            request_id: Request identifier
            
        Returns:
            True if successfully cancelled
        """
        if request_id in self.active_exports:
            task = self.active_exports[request_id]
            task.cancel()
            
            try:
                await task
            except asyncio.CancelledError:
                pass
            
            del self.active_exports[request_id]
            
            self.logger.info(f"Export cancelled: {request_id}")
            return True
        
        return False
    
    async def export_single_session(
        self,
        session_id: str,
        format: ExportFormat = ExportFormat.JSON,
        include_audio: bool = False,
        include_transcripts: bool = True,
        output_path: Optional[str] = None
    ) -> ExportResult:
        """
        Export a single session with simplified interface.
        
        Args:
            session_id: Session identifier
            format: Export format
            include_audio: Include audio data
            include_transcripts: Include transcript data
            output_path: Optional output path
            
        Returns:
            Export result
        """
        request = ExportRequest(
            request_id=f"single_{session_id}_{int(time.time())}",
            scope=ExportScope.SINGLE_SESSION,
            format=format,
            report_type=ReportType.SESSION_SUMMARY,
            session_ids=[session_id],
            include_audio=include_audio,
            include_transcripts=include_transcripts,
            output_path=output_path
        )
        
        return await self._process_export_request(request)
    
    async def export_sessions_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        format: ExportFormat = ExportFormat.ZIP,
        report_type: ReportType = ReportType.SESSION_SUMMARY,
        output_path: Optional[str] = None
    ) -> ExportResult:
        """
        Export sessions within a date range.
        
        Args:
            start_date: Start date
            end_date: End date
            format: Export format
            report_type: Type of report to generate
            output_path: Optional output path
            
        Returns:
            Export result
        """
        request = ExportRequest(
            request_id=f"daterange_{int(start_date.timestamp())}_{int(end_date.timestamp())}",
            scope=ExportScope.DATE_RANGE,
            format=format,
            report_type=report_type,
            start_date=start_date,
            end_date=end_date,
            output_path=output_path
        )
        
        return await self._process_export_request(request)
    
    async def generate_automated_reports(
        self,
        report_types: List[ReportType],
        schedule_interval: timedelta = timedelta(days=1)
    ):
        """
        Generate automated reports on a schedule.
        
        Args:
            report_types: Types of reports to generate
            schedule_interval: Interval between reports
        """
        while True:
            try:
                current_time = datetime.now(timezone.utc)
                end_date = current_time
                start_date = current_time - schedule_interval
                
                for report_type in report_types:
                    request = ExportRequest(
                        request_id=f"auto_{report_type.value}_{int(current_time.timestamp())}",
                        scope=ExportScope.DATE_RANGE,
                        format=ExportFormat.HTML,
                        report_type=report_type,
                        start_date=start_date,
                        end_date=end_date,
                        output_path=str(self.export_path / f"automated_{report_type.value}_{current_time.strftime('%Y%m%d_%H%M%S')}.html")
                    )
                    
                    await self.submit_export_request(request)
                
                # Wait for next interval
                await asyncio.sleep(schedule_interval.total_seconds())
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.exception("Error in automated report generation", extra_data={"error": str(e)})
                await asyncio.sleep(3600)  # Wait 1 hour before retrying
    
    async def start_export_workers(self):
        """Start background export worker tasks."""
        self.export_workers = [
            asyncio.create_task(self._export_worker(i))
            for i in range(self.max_concurrent_exports)
        ]
        
        self.logger.info(f"Started {len(self.export_workers)} export workers")
    
    async def stop_export_workers(self):
        """Stop background export worker tasks."""
        for worker in self.export_workers:
            worker.cancel()
        
        await asyncio.gather(*self.export_workers, return_exceptions=True)
        self.export_workers.clear()
        
        self.logger.info("Stopped export workers")
    
    def add_callback(self, event: str, callback: Callable):
        """Add a callback for export events."""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
    
    def remove_callback(self, event: str, callback: Callable):
        """Remove a callback for export events."""
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
                    f"Error in export callback: {event}",
                    extra_data={"error": str(e), "event": event}
                )
    
    async def _export_worker(self, worker_id: int):
        """Background worker to process export requests."""
        self.logger.info(f"Export worker {worker_id} started")
        
        try:
            while True:
                # Get next request from queue
                request = await self.export_queue.get()
                
                try:
                    # Track active export
                    task = asyncio.create_task(self._process_export_request(request))
                    self.active_exports[request.request_id] = task
                    
                    # Process the request
                    result = await task
                    
                    # Store result
                    self.export_history.append(result)
                    
                    # Cleanup
                    del self.active_exports[request.request_id]
                    
                except Exception as e:
                    # Handle export error
                    result = ExportResult(
                        request_id=request.request_id,
                        success=False,
                        error_message=str(e)
                    )
                    self.export_history.append(result)
                    
                    if request.request_id in self.active_exports:
                        del self.active_exports[request.request_id]
                    
                    await self._trigger_callbacks("on_export_error", request.request_id, str(e))
                    
                    self.logger.exception(
                        f"Export failed: {request.request_id}",
                        extra_data={"error": str(e)}
                    )
                
                finally:
                    self.export_queue.task_done()
                    
        except asyncio.CancelledError:
            self.logger.info(f"Export worker {worker_id} cancelled")
        except Exception as e:
            self.logger.exception(f"Export worker {worker_id} error", extra_data={"error": str(e)})
    
    async def _process_export_request(self, request: ExportRequest) -> ExportResult:
        """Process an export request."""
        start_time = time.time()
        
        # Trigger start callback
        await self._trigger_callbacks("on_export_start", request.request_id)
        
        try:
            # Collect data based on scope
            session_data = await self._collect_session_data(request)
            
            # Generate report
            report_data = await self._generate_report(request, session_data)
            
            # Export in requested format
            output_path = await self._export_data(request, report_data)
            
            # Calculate file size
            file_size = Path(output_path).stat().st_size if output_path and Path(output_path).exists() else None
            
            # Create result
            result = ExportResult(
                request_id=request.request_id,
                success=True,
                output_path=output_path,
                file_size=file_size,
                record_count=len(session_data),
                duration=time.time() - start_time
            )
            
            # Trigger completion callback
            await self._trigger_callbacks("on_export_complete", request.request_id, result)
            
            self.logger.info(
                f"Export completed: {request.request_id}",
                extra_data={
                    "request_id": request.request_id,
                    "output_path": output_path,
                    "file_size": file_size,
                    "record_count": len(session_data),
                    "duration": result.duration
                }
            )
            
            return result
            
        except Exception as e:
            result = ExportResult(
                request_id=request.request_id,
                success=False,
                error_message=str(e),
                duration=time.time() - start_time
            )
            
            self.logger.exception(
                f"Export failed: {request.request_id}",
                extra_data={"error": str(e)}
            )
            
            return result
    
    async def _validate_export_request(self, request: ExportRequest):
        """Validate an export request."""
        if request.scope == ExportScope.SINGLE_SESSION and not request.session_ids:
            raise ValueError("Single session scope requires session_ids")
        
        if request.scope == ExportScope.DATE_RANGE and (not request.start_date or not request.end_date):
            raise ValueError("Date range scope requires start_date and end_date")
        
        if request.format not in self.format_handlers:
            raise ValueError(f"Unsupported export format: {request.format}")
        
        if request.report_type not in self.report_generators:
            raise ValueError(f"Unsupported report type: {request.report_type}")
    
    async def _collect_session_data(self, request: ExportRequest) -> List[Dict[str, Any]]:
        """Collect session data based on request scope."""
        session_data = []
        
        if request.scope == ExportScope.SINGLE_SESSION:
            for session_id in request.session_ids or []:
                data = await self._load_session_data(session_id, request)
                if data:
                    session_data.append(data)
        
        elif request.scope == ExportScope.DATE_RANGE:
            # Find sessions in date range
            for metadata_file in self.storage_path.glob("*_metadata.json"):
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    start_time = datetime.fromisoformat(metadata["start_time"])
                    
                    if request.start_date <= start_time <= request.end_date:
                        data = await self._load_session_data(metadata["session_id"], request)
                        if data:
                            session_data.append(data)
                            
                except Exception as e:
                    self.logger.warning(f"Error loading session metadata from {metadata_file}: {e}")
        
        elif request.scope == ExportScope.USER_SESSIONS:
            # Find sessions for specific users
            for user_id in request.user_ids or []:
                for metadata_file in self.storage_path.glob("*_metadata.json"):
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        
                        if metadata.get("user_id") == user_id:
                            data = await self._load_session_data(metadata["session_id"], request)
                            if data:
                                session_data.append(data)
                                
                    except Exception as e:
                        self.logger.warning(f"Error loading session metadata from {metadata_file}: {e}")
        
        return session_data
    
    async def _load_session_data(self, session_id: str, request: ExportRequest) -> Optional[Dict[str, Any]]:
        """Load data for a single session."""
        try:
            session_data = {"session_id": session_id}
            
            # Load metadata
            if request.include_metadata:
                metadata = await self._load_session_metadata(session_id)
                if metadata:
                    session_data["metadata"] = metadata
                else:
                    return None  # Skip if no metadata
            
            # Load transcripts
            if request.include_transcripts:
                transcripts = await self._load_session_transcripts(session_id)
                session_data["transcripts"] = transcripts
            
            # Load quality metrics
            if request.include_quality_metrics:
                quality = await self._load_session_quality(session_id)
                session_data["quality_metrics"] = quality
            
            # Load audio (if requested and allowed by privacy level)
            if request.include_audio and request.privacy_level in [PrivacyLevel.FULL, PrivacyLevel.AUDIO_ONLY]:
                audio = await self._load_session_audio(session_id)
                session_data["audio"] = audio
            
            # Apply anonymization if requested
            if request.anonymize_data:
                session_data = await self._anonymize_session_data(session_data)
            
            return session_data
            
        except Exception as e:
            self.logger.exception(f"Error loading session data: {session_id}", extra_data={"error": str(e)})
            return None
    
    async def _generate_report(self, request: ExportRequest, session_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate report based on request type."""
        generator = self.report_generators.get(request.report_type)
        if not generator:
            raise ValueError(f"No generator for report type: {request.report_type}")
        
        return await generator(request, session_data)
    
    async def _export_data(self, request: ExportRequest, report_data: Dict[str, Any]) -> Optional[str]:
        """Export data in the requested format."""
        handler = self.format_handlers.get(request.format)
        if not handler:
            raise ValueError(f"No handler for format: {request.format}")
        
        return await handler(request, report_data)
    
    # Format handlers
    async def _export_json(self, request: ExportRequest, data: Dict[str, Any]) -> str:
        """Export data as JSON."""
        output_path = request.output_path or str(self.export_path / f"{request.request_id}.json")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str, ensure_ascii=False)
        
        return output_path
    
    async def _export_csv(self, request: ExportRequest, data: Dict[str, Any]) -> str:
        """Export data as CSV."""
        output_path = request.output_path or str(self.export_path / f"{request.request_id}.csv")
        
        # Flatten data for CSV format
        rows = []
        
        if "sessions" in data:
            for session in data["sessions"]:
                row = {}
                
                # Add metadata fields
                if "metadata" in session:
                    metadata = session["metadata"]
                    row.update({
                        "session_id": metadata.get("session_id"),
                        "user_id": metadata.get("user_id"),
                        "agent_id": metadata.get("agent_id"),
                        "start_time": metadata.get("start_time"),
                        "duration": metadata.get("duration"),
                        "status": metadata.get("status")
                    })
                
                # Add quality metrics
                if "quality_metrics" in session and session["quality_metrics"]:
                    quality = session["quality_metrics"]
                    row.update({
                        "audio_quality_score": quality.get("audio_quality_score"),
                        "transcript_accuracy_score": quality.get("transcript_accuracy_score"),
                        "interruption_count": quality.get("interruption_count")
                    })
                
                rows.append(row)
        
        # Write CSV
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
        
        return output_path
    
    async def _export_html(self, request: ExportRequest, data: Dict[str, Any]) -> str:
        """Export data as HTML report."""
        output_path = request.output_path or str(self.export_path / f"{request.request_id}.html")
        
        # Generate HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{data.get('title', 'Session Report')}</title>
            <meta charset="UTF-8">
            <style>
                body {{ 
                    font-family: Arial, sans-serif; 
                    margin: 40px; 
                    background-color: #f5f5f5;
                }}
                .header {{ 
                    background-color: #ffffff; 
                    padding: 30px; 
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin-bottom: 20px;
                }}
                .section {{ 
                    background-color: #ffffff;
                    margin: 20px 0; 
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .metric {{ 
                    display: inline-block; 
                    margin: 10px; 
                    padding: 15px; 
                    border: 1px solid #ddd; 
                    border-radius: 4px;
                    background-color: #f9f9f9;
                }}
                .summary-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 10px 0;
                }}
                .summary-table th, .summary-table td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                .summary-table th {{
                    background-color: #f2f2f2;
                }}
                .quality-score {{
                    font-size: 1.2em;
                    font-weight: bold;
                    color: #2e7d32;
                }}
                .warning {{ color: #f57c00; }}
                .error {{ color: #d32f2f; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{data.get('title', 'Session Report')}</h1>
                <p><strong>Generated:</strong> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
                <p><strong>Report Type:</strong> {request.report_type.value.replace('_', ' ').title()}</p>
                <p><strong>Total Sessions:</strong> {data.get('total_sessions', 0)}</p>
            </div>
        """
        
        # Add summary section
        if "summary" in data:
            html_content += f"""
            <div class="section">
                <h2>Summary</h2>
                <div class="metric">
                    <h3>Average Quality Score</h3>
                    <div class="quality-score">{data['summary'].get('avg_quality_score', 'N/A')}</div>
                </div>
                <div class="metric">
                    <h3>Total Duration</h3>
                    <div>{data['summary'].get('total_duration', 'N/A')} minutes</div>
                </div>
                <div class="metric">
                    <h3>Total Recordings</h3>
                    <div>{data['summary'].get('total_sessions', 0)}</div>
                </div>
            </div>
            """
        
        # Add sessions table
        if "sessions" in data and data["sessions"]:
            html_content += """
            <div class="section">
                <h2>Session Details</h2>
                <table class="summary-table">
                    <thead>
                        <tr>
                            <th>Session ID</th>
                            <th>Start Time</th>
                            <th>Duration</th>
                            <th>Quality Score</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
            """
            
            for session in data["sessions"]:
                metadata = session.get("metadata", {})
                quality = session.get("quality_metrics", {})
                
                html_content += f"""
                        <tr>
                            <td>{metadata.get('session_id', 'N/A')}</td>
                            <td>{metadata.get('start_time', 'N/A')}</td>
                            <td>{metadata.get('duration', 'N/A')}</td>
                            <td>{quality.get('audio_quality_score', 'N/A')}</td>
                            <td>{metadata.get('status', 'N/A')}</td>
                        </tr>
                """
            
            html_content += """
                    </tbody>
                </table>
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return output_path
    
    async def _export_pdf(self, request: ExportRequest, data: Dict[str, Any]) -> str:
        """Export data as PDF (placeholder implementation)."""
        # For a real implementation, you would use a library like reportlab
        output_path = request.output_path or str(self.export_path / f"{request.request_id}.pdf")
        
        # For now, create a text file with PDF extension
        content = f"PDF Report\n\nGenerated: {datetime.now(timezone.utc)}\n\n"
        content += json.dumps(data, indent=2, default=str)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return output_path
    
    async def _export_xml(self, request: ExportRequest, data: Dict[str, Any]) -> str:
        """Export data as XML."""
        output_path = request.output_path or str(self.export_path / f"{request.request_id}.xml")
        
        # Simple XML generation
        xml_content = '<?xml version="1.0" encoding="UTF-8"?>\n<report>\n'
        
        def dict_to_xml(d, indent=1):
            xml = ""
            for key, value in d.items():
                spaces = "  " * indent
                if isinstance(value, dict):
                    xml += f"{spaces}<{key}>\n{dict_to_xml(value, indent+1)}{spaces}</{key}>\n"
                elif isinstance(value, list):
                    xml += f"{spaces}<{key}>\n"
                    for item in value:
                        if isinstance(item, dict):
                            xml += f"{spaces}  <item>\n{dict_to_xml(item, indent+2)}{spaces}  </item>\n"
                        else:
                            xml += f"{spaces}  <item>{item}</item>\n"
                    xml += f"{spaces}</{key}>\n"
                else:
                    xml += f"{spaces}<{key}>{value}</{key}>\n"
            return xml
        
        xml_content += dict_to_xml(data)
        xml_content += "</report>"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(xml_content)
        
        return output_path
    
    async def _export_zip(self, request: ExportRequest, data: Dict[str, Any]) -> str:
        """Export data as ZIP archive."""
        output_path = request.output_path or str(self.export_path / f"{request.request_id}.zip")
        
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add JSON report
            json_content = json.dumps(data, indent=2, default=str)
            zip_file.writestr(f"{request.request_id}_report.json", json_content)
            
            # Add individual session files if available
            if "sessions" in data:
                for i, session in enumerate(data["sessions"]):
                    session_content = json.dumps(session, indent=2, default=str)
                    session_id = session.get("session_id", f"session_{i}")
                    zip_file.writestr(f"sessions/{session_id}.json", session_content)
                    
                    # Add transcripts if available
                    if "transcripts" in session and session["transcripts"]:
                        transcript_content = "\n".join([
                            f"[{t.get('timestamp', 'N/A')}] {t.get('speaker_id', 'Speaker')}: {t.get('text', '')}"
                            for t in session["transcripts"]
                        ])
                        zip_file.writestr(f"transcripts/{session_id}.txt", transcript_content)
        
        return output_path
    
    async def _export_audio_wav(self, request: ExportRequest, data: Dict[str, Any]) -> str:
        """Export audio data as WAV files."""
        # This would export actual audio files
        # For now, return a placeholder
        output_path = request.output_path or str(self.export_path / f"{request.request_id}_audio.zip")
        return output_path
    
    async def _export_audio_mp3(self, request: ExportRequest, data: Dict[str, Any]) -> str:
        """Export audio data as MP3 files."""
        # This would export actual audio files converted to MP3
        # For now, return a placeholder
        output_path = request.output_path or str(self.export_path / f"{request.request_id}_audio.zip")
        return output_path
    
    # Report generators
    async def _generate_session_summary_report(self, request: ExportRequest, session_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate session summary report."""
        total_sessions = len(session_data)
        total_duration = 0
        quality_scores = []
        
        for session in session_data:
            metadata = session.get("metadata", {})
            quality = session.get("quality_metrics", {})
            
            if metadata.get("duration"):
                total_duration += metadata["duration"]
            
            if quality.get("audio_quality_score"):
                quality_scores.append(quality["audio_quality_score"])
        
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        return {
            "title": "Session Summary Report",
            "total_sessions": total_sessions,
            "sessions": session_data,
            "summary": {
                "total_sessions": total_sessions,
                "total_duration": round(total_duration / 60, 2),  # Convert to minutes
                "avg_quality_score": round(avg_quality, 2),
                "quality_distribution": {
                    "excellent": len([s for s in quality_scores if s >= 0.9]),
                    "good": len([s for s in quality_scores if 0.7 <= s < 0.9]),
                    "fair": len([s for s in quality_scores if 0.5 <= s < 0.7]),
                    "poor": len([s for s in quality_scores if s < 0.5])
                }
            }
        }
    
    async def _generate_quality_report(self, request: ExportRequest, session_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate quality analysis report."""
        # Implement quality-specific analysis
        return {
            "title": "Quality Analysis Report",
            "sessions": session_data,
            "quality_analysis": {}  # Would contain detailed quality metrics
        }
    
    async def _generate_transcript_report(self, request: ExportRequest, session_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate transcript analysis report."""
        return {
            "title": "Transcript Analysis Report",
            "sessions": session_data,
            "transcript_analysis": {}  # Would contain transcript-specific analysis
        }
    
    async def _generate_analytics_report(self, request: ExportRequest, session_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate analytics report."""
        return {
            "title": "Analytics Report", 
            "sessions": session_data,
            "analytics": {}  # Would contain detailed analytics
        }
    
    async def _generate_compliance_report(self, request: ExportRequest, session_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate compliance report."""
        return {
            "title": "Compliance Report",
            "sessions": session_data,
            "compliance_status": {}  # Would contain compliance analysis
        }
    
    async def _generate_performance_report(self, request: ExportRequest, session_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate performance report."""
        return {
            "title": "Performance Report",
            "sessions": session_data,
            "performance_metrics": {}  # Would contain performance analysis
        }
    
    async def _generate_custom_report(self, request: ExportRequest, session_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate custom report."""
        return {
            "title": "Custom Report",
            "sessions": session_data,
            "custom_data": {}  # Would contain custom analysis
        }
    
    # Data loading helpers
    async def _load_session_metadata(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load session metadata."""
        metadata_file = self.storage_path / f"{session_id}_metadata.json"
        if not metadata_file.exists():
            metadata_file = self.storage_path / session_id / f"{session_id}_metadata.json"
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        
        return None
    
    async def _load_session_transcripts(self, session_id: str) -> List[Dict[str, Any]]:
        """Load session transcripts."""
        transcript_file = self.storage_path / f"{session_id}_transcript.json"
        if not transcript_file.exists():
            transcript_file = self.storage_path / session_id / f"{session_id}_transcript.json"
        
        if transcript_file.exists():
            try:
                with open(transcript_file, 'r') as f:
                    data = json.load(f)
                    return data.get("segments", [])
            except Exception:
                pass
        
        return []
    
    async def _load_session_quality(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load session quality metrics."""
        quality_file = self.storage_path / f"{session_id}_quality.json"
        if not quality_file.exists():
            quality_file = self.storage_path / session_id / f"{session_id}_quality.json"
        
        if quality_file.exists():
            try:
                with open(quality_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        
        return None
    
    async def _load_session_audio(self, session_id: str) -> Optional[str]:
        """Load session audio (return base64 encoded for JSON export)."""
        # Find audio file
        for ext in ['wav', 'mp3', 'opus']:
            audio_file = self.storage_path / f"{session_id}_audio.{ext}"
            if not audio_file.exists():
                audio_file = self.storage_path / session_id / f"{session_id}_audio.{ext}"
            
            if audio_file.exists():
                try:
                    with open(audio_file, 'rb') as f:
                        audio_data = f.read()
                        return base64.b64encode(audio_data).decode('utf-8')
                except Exception:
                    pass
        
        return None
    
    async def _anonymize_session_data(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Anonymize session data for privacy."""
        # Simple anonymization - remove/hash PII
        if "metadata" in session_data:
            metadata = session_data["metadata"]
            if "user_id" in metadata:
                metadata["user_id"] = f"user_{hashlib.sha256(metadata['user_id'].encode()).hexdigest()[:8]}"
        
        return session_data