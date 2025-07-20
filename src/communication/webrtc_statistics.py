"""
WebRTC Statistics Collection and Reporting

This module provides comprehensive WebRTC statistics collection, analysis,
and reporting capabilities for monitoring connection quality and performance.
"""

import asyncio
import logging
import time
import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Dict, Any, List, Callable, Tuple, Set
from collections import defaultdict, deque
import statistics
from datetime import datetime, timedelta

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

from monitoring.performance_monitor import monitor_performance, global_performance_monitor

logger = logging.getLogger(__name__)


class StatType(Enum):
    """WebRTC statistics types."""
    INBOUND_RTP = "inbound-rtp"
    OUTBOUND_RTP = "outbound-rtp"
    REMOTE_INBOUND_RTP = "remote-inbound-rtp"
    REMOTE_OUTBOUND_RTP = "remote-outbound-rtp"
    MEDIA_SOURCE = "media-source"
    CANDIDATE_PAIR = "candidate-pair"
    LOCAL_CANDIDATE = "local-candidate"
    REMOTE_CANDIDATE = "remote-candidate"
    CERTIFICATE = "certificate"
    CODEC = "codec"
    TRANSPORT = "transport"
    PEER_CONNECTION = "peer-connection"


class ReportType(Enum):
    """Types of statistics reports."""
    REAL_TIME = "real_time"          # Current snapshot
    SUMMARY = "summary"               # Aggregated summary
    DETAILED = "detailed"             # Full detailed stats
    HISTORICAL = "historical"         # Time-series data
    DIAGNOSTIC = "diagnostic"         # Diagnostic information
    PERFORMANCE = "performance"       # Performance metrics


@dataclass
class RTCStatsSnapshot:
    """Single snapshot of WebRTC statistics."""
    timestamp: float
    stats_id: str
    type: str
    
    # Common fields
    ssrc: Optional[int] = None
    media_type: Optional[str] = None  # audio/video
    
    # RTP stream stats
    packets_sent: Optional[int] = None
    packets_received: Optional[int] = None
    bytes_sent: Optional[int] = None
    bytes_received: Optional[int] = None
    packets_lost: Optional[int] = None
    jitter: Optional[float] = None
    round_trip_time: Optional[float] = None
    
    # Audio specific
    audio_level: Optional[float] = None
    total_audio_energy: Optional[float] = None
    echo_return_loss: Optional[float] = None
    echo_return_loss_enhancement: Optional[float] = None
    
    # Video specific
    frames_encoded: Optional[int] = None
    frames_decoded: Optional[int] = None
    frames_dropped: Optional[int] = None
    frame_width: Optional[int] = None
    frame_height: Optional[int] = None
    frames_per_second: Optional[float] = None
    
    # Codec info
    codec_id: Optional[str] = None
    mime_type: Optional[str] = None
    clock_rate: Optional[int] = None
    channels: Optional[int] = None
    
    # Network info
    available_outgoing_bitrate: Optional[float] = None
    available_incoming_bitrate: Optional[float] = None
    current_round_trip_time: Optional[float] = None
    total_round_trip_time: Optional[float] = None
    
    # Quality metrics
    quality_limitation_reason: Optional[str] = None
    quality_limitation_durations: Dict[str, float] = field(default_factory=dict)
    
    # Custom metrics
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class ConnectionStats:
    """Aggregated connection statistics."""
    # Connection info
    connection_id: str
    local_candidate_type: Optional[str] = None
    remote_candidate_type: Optional[str] = None
    network_type: Optional[str] = None
    
    # Performance metrics
    avg_bitrate_bps: float = 0.0
    avg_packet_loss_percent: float = 0.0
    avg_jitter_ms: float = 0.0
    avg_rtt_ms: float = 0.0
    
    # Totals
    total_bytes_sent: int = 0
    total_bytes_received: int = 0
    total_packets_sent: int = 0
    total_packets_received: int = 0
    total_packets_lost: int = 0
    
    # Connection quality
    connection_quality_score: float = 0.0  # 0-100
    stability_score: float = 0.0           # 0-100
    
    # Time metrics
    connection_duration_seconds: float = 0.0
    time_to_first_byte_ms: Optional[float] = None
    
    # Issues detected
    quality_issues: List[str] = field(default_factory=list)
    reconnection_count: int = 0


@dataclass
class MediaStreamStats:
    """Statistics for a media stream."""
    stream_id: str
    media_type: str  # audio/video
    direction: str   # inbound/outbound
    
    # Current metrics
    current_bitrate_bps: float = 0.0
    current_packet_loss_percent: float = 0.0
    current_jitter_ms: float = 0.0
    
    # Aggregated metrics
    avg_bitrate_bps: float = 0.0
    max_bitrate_bps: float = 0.0
    min_bitrate_bps: float = 0.0
    
    # Audio specific
    avg_audio_level: Optional[float] = None
    silence_percent: Optional[float] = None
    
    # Video specific
    avg_framerate: Optional[float] = None
    resolution_changes: int = 0
    current_resolution: Optional[Tuple[int, int]] = None
    
    # Quality
    quality_score: float = 0.0  # 0-100
    mos_score: Optional[float] = None  # 1-5


class WebRTCStatsCollector:
    """Collects and analyzes WebRTC statistics."""
    
    def __init__(self, collection_interval_ms: int = 1000):
        self.collection_interval_ms = collection_interval_ms
        self.is_collecting = False
        
        # Statistics storage
        self.snapshots: deque = deque(maxlen=3600)  # 1 hour of snapshots
        self.current_stats: Dict[str, RTCStatsSnapshot] = {}
        self.historical_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=3600))
        
        # Aggregated stats
        self.connection_stats = ConnectionStats(connection_id=str(time.time()))
        self.media_streams: Dict[str, MediaStreamStats] = {}
        
        # Analysis results
        self.quality_trends: Dict[str, str] = {}
        self.anomalies: List[Dict[str, Any]] = []
        self.recommendations: List[str] = []
        
        # Event callbacks
        self.on_stats_updated_callbacks: List[Callable] = []
        self.on_quality_degraded_callbacks: List[Callable] = []
        self.on_anomaly_detected_callbacks: List[Callable] = []
        
        # Monitoring
        self.monitor = global_performance_monitor.register_component(
            "webrtc_stats_collector", "communication"
        )
        
        # Background tasks
        self._collection_task: Optional[asyncio.Task] = None
        self._analysis_task: Optional[asyncio.Task] = None
    
    async def start_collection(self):
        """Start statistics collection."""
        if self.is_collecting:
            return
        
        self.is_collecting = True
        self._collection_task = asyncio.create_task(self._collection_loop())
        self._analysis_task = asyncio.create_task(self._analysis_loop())
        
        logger.info("Started WebRTC statistics collection")
    
    async def stop_collection(self):
        """Stop statistics collection."""
        self.is_collecting = False
        
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
        
        if self._analysis_task:
            self._analysis_task.cancel()
            try:
                await self._analysis_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped WebRTC statistics collection")
    
    @monitor_performance(component="webrtc_stats", operation="collect_stats")
    async def collect_stats(self, rtc_stats: Dict[str, Any]):
        """Process raw WebRTC statistics."""
        try:
            timestamp = time.time()
            
            # Parse and store individual stats
            for stat_id, stat_data in rtc_stats.items():
                stat_type = stat_data.get('type', '')
                
                # Create snapshot
                snapshot = self._create_snapshot(timestamp, stat_id, stat_data)
                
                # Store current stats
                self.current_stats[stat_id] = snapshot
                
                # Add to historical data
                self.historical_data[stat_id].append(snapshot)
                
                # Process by type
                if stat_type == StatType.INBOUND_RTP.value:
                    await self._process_inbound_rtp(snapshot)
                elif stat_type == StatType.OUTBOUND_RTP.value:
                    await self._process_outbound_rtp(snapshot)
                elif stat_type == StatType.CANDIDATE_PAIR.value:
                    await self._process_candidate_pair(snapshot)
            
            # Store complete snapshot
            self.snapshots.append({
                'timestamp': timestamp,
                'stats': dict(self.current_stats)
            })
            
            # Update aggregated stats
            await self._update_aggregated_stats()
            
            # Trigger callbacks
            await self._trigger_stats_updated()
            
        except Exception as e:
            logger.error(f"Error collecting stats: {e}")
    
    def _create_snapshot(self, timestamp: float, stat_id: str, stat_data: Dict[str, Any]) -> RTCStatsSnapshot:
        """Create stats snapshot from raw data."""
        snapshot = RTCStatsSnapshot(
            timestamp=timestamp,
            stats_id=stat_id,
            type=stat_data.get('type', ''),
            ssrc=stat_data.get('ssrc'),
            media_type=stat_data.get('mediaType')
        )
        
        # Map common fields
        field_mapping = {
            'packetsSent': 'packets_sent',
            'packetsReceived': 'packets_received',
            'bytesSent': 'bytes_sent',
            'bytesReceived': 'bytes_received',
            'packetsLost': 'packets_lost',
            'jitter': 'jitter',
            'roundTripTime': 'round_trip_time',
            'audioLevel': 'audio_level',
            'totalAudioEnergy': 'total_audio_energy',
            'framesEncoded': 'frames_encoded',
            'framesDecoded': 'frames_decoded',
            'framesDropped': 'frames_dropped',
            'frameWidth': 'frame_width',
            'frameHeight': 'frame_height',
            'framesPerSecond': 'frames_per_second'
        }
        
        for rtc_field, snapshot_field in field_mapping.items():
            if rtc_field in stat_data:
                setattr(snapshot, snapshot_field, stat_data[rtc_field])
        
        # Store any additional fields in custom metrics
        known_fields = set(field_mapping.keys()) | {'type', 'id', 'timestamp', 'ssrc', 'mediaType'}
        for key, value in stat_data.items():
            if key not in known_fields:
                snapshot.custom_metrics[key] = value
        
        return snapshot
    
    async def _process_inbound_rtp(self, snapshot: RTCStatsSnapshot):
        """Process inbound RTP statistics."""
        if not snapshot.media_type:
            return
        
        # Get or create media stream stats
        stream_id = f"inbound_{snapshot.media_type}_{snapshot.ssrc}"
        if stream_id not in self.media_streams:
            self.media_streams[stream_id] = MediaStreamStats(
                stream_id=stream_id,
                media_type=snapshot.media_type,
                direction="inbound"
            )
        
        stream_stats = self.media_streams[stream_id]
        
        # Calculate current metrics
        history = self.historical_data.get(snapshot.stats_id, [])
        if len(history) >= 2:
            prev_snapshot = history[-2]
            time_delta = snapshot.timestamp - prev_snapshot.timestamp
            
            if time_delta > 0:
                # Calculate bitrate
                bytes_delta = (snapshot.bytes_received or 0) - (prev_snapshot.bytes_received or 0)
                stream_stats.current_bitrate_bps = (bytes_delta * 8) / time_delta
                
                # Calculate packet loss
                packets_delta = (snapshot.packets_received or 0) - (prev_snapshot.packets_received or 0)
                lost_delta = (snapshot.packets_lost or 0) - (prev_snapshot.packets_lost or 0)
                
                if packets_delta + lost_delta > 0:
                    stream_stats.current_packet_loss_percent = (lost_delta / (packets_delta + lost_delta)) * 100
                
                # Update jitter
                if snapshot.jitter:
                    stream_stats.current_jitter_ms = snapshot.jitter * 1000
    
    async def _process_outbound_rtp(self, snapshot: RTCStatsSnapshot):
        """Process outbound RTP statistics."""
        if not snapshot.media_type:
            return
        
        # Get or create media stream stats
        stream_id = f"outbound_{snapshot.media_type}_{snapshot.ssrc}"
        if stream_id not in self.media_streams:
            self.media_streams[stream_id] = MediaStreamStats(
                stream_id=stream_id,
                media_type=snapshot.media_type,
                direction="outbound"
            )
        
        stream_stats = self.media_streams[stream_id]
        
        # Calculate current metrics
        history = self.historical_data.get(snapshot.stats_id, [])
        if len(history) >= 2:
            prev_snapshot = history[-2]
            time_delta = snapshot.timestamp - prev_snapshot.timestamp
            
            if time_delta > 0:
                # Calculate bitrate
                bytes_delta = (snapshot.bytes_sent or 0) - (prev_snapshot.bytes_sent or 0)
                stream_stats.current_bitrate_bps = (bytes_delta * 8) / time_delta
    
    async def _process_candidate_pair(self, snapshot: RTCStatsSnapshot):
        """Process candidate pair statistics."""
        # Update connection stats
        if snapshot.current_round_trip_time:
            self.connection_stats.avg_rtt_ms = snapshot.current_round_trip_time * 1000
        
        # Extract candidate types from custom metrics
        if 'localCandidateType' in snapshot.custom_metrics:
            self.connection_stats.local_candidate_type = snapshot.custom_metrics['localCandidateType']
        
        if 'remoteCandidateType' in snapshot.custom_metrics:
            self.connection_stats.remote_candidate_type = snapshot.custom_metrics['remoteCandidateType']
    
    async def _update_aggregated_stats(self):
        """Update aggregated statistics."""
        # Update connection stats
        total_sent = 0
        total_received = 0
        total_lost = 0
        
        for stat in self.current_stats.values():
            if stat.bytes_sent:
                total_sent += stat.bytes_sent
            if stat.bytes_received:
                total_received += stat.bytes_received
            if stat.packets_lost:
                total_lost += stat.packets_lost
        
        self.connection_stats.total_bytes_sent = total_sent
        self.connection_stats.total_bytes_received = total_received
        self.connection_stats.total_packets_lost = total_lost
        
        # Calculate average metrics for media streams
        for stream_stats in self.media_streams.values():
            history = []
            for snapshot in self.snapshots[-60:]:  # Last minute
                for stat in snapshot['stats'].values():
                    if (stat.media_type == stream_stats.media_type and
                        stat.type.endswith(stream_stats.direction)):
                        history.append(stat)
            
            if history:
                # Calculate averages
                bitrates = []
                for i in range(1, len(history)):
                    time_delta = history[i].timestamp - history[i-1].timestamp
                    if time_delta > 0:
                        if stream_stats.direction == "inbound":
                            bytes_delta = (history[i].bytes_received or 0) - (history[i-1].bytes_received or 0)
                        else:
                            bytes_delta = (history[i].bytes_sent or 0) - (history[i-1].bytes_sent or 0)
                        
                        bitrate = (bytes_delta * 8) / time_delta
                        bitrates.append(bitrate)
                
                if bitrates:
                    stream_stats.avg_bitrate_bps = statistics.mean(bitrates)
                    stream_stats.max_bitrate_bps = max(bitrates)
                    stream_stats.min_bitrate_bps = min(bitrates)
    
    async def _collection_loop(self):
        """Main collection loop."""
        while self.is_collecting:
            try:
                # In real implementation, this would fetch stats from RTCPeerConnection
                # For now, we'll wait for stats to be pushed via collect_stats()
                
                await asyncio.sleep(self.collection_interval_ms / 1000)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in collection loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _analysis_loop(self):
        """Statistics analysis loop."""
        while self.is_collecting:
            try:
                # Perform periodic analysis
                await self._analyze_quality_trends()
                await self._detect_anomalies()
                await self._generate_recommendations()
                
                # Sleep for analysis interval (less frequent than collection)
                await asyncio.sleep(5.0)  # Analyze every 5 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in analysis loop: {e}")
                await asyncio.sleep(5.0)
    
    async def _analyze_quality_trends(self):
        """Analyze quality trends from statistics."""
        for stream_id, stream_stats in self.media_streams.items():
            # Simple trend analysis
            if stream_stats.current_bitrate_bps > stream_stats.avg_bitrate_bps * 1.2:
                self.quality_trends[stream_id] = "improving"
            elif stream_stats.current_bitrate_bps < stream_stats.avg_bitrate_bps * 0.8:
                self.quality_trends[stream_id] = "degrading"
                
                # Trigger quality degraded callback
                await self._trigger_quality_degraded(stream_id, stream_stats)
            else:
                self.quality_trends[stream_id] = "stable"
    
    async def _detect_anomalies(self):
        """Detect anomalies in statistics."""
        anomalies = []
        
        # Check for sudden bitrate drops
        for stream_id, stream_stats in self.media_streams.items():
            if (stream_stats.avg_bitrate_bps > 0 and
                stream_stats.current_bitrate_bps < stream_stats.avg_bitrate_bps * 0.5):
                anomalies.append({
                    "type": "bitrate_drop",
                    "stream_id": stream_id,
                    "severity": "high",
                    "message": f"Bitrate dropped to {stream_stats.current_bitrate_bps:.0f} bps"
                })
        
        # Check for high packet loss
        for stream_stats in self.media_streams.values():
            if stream_stats.current_packet_loss_percent > 5.0:
                anomalies.append({
                    "type": "high_packet_loss",
                    "stream_id": stream_stats.stream_id,
                    "severity": "high",
                    "message": f"Packet loss at {stream_stats.current_packet_loss_percent:.1f}%"
                })
        
        # Store and trigger callbacks for new anomalies
        for anomaly in anomalies:
            if anomaly not in self.anomalies:
                self.anomalies.append(anomaly)
                await self._trigger_anomaly_detected(anomaly)
    
    async def _generate_recommendations(self):
        """Generate recommendations based on statistics."""
        self.recommendations.clear()
        
        # Check audio quality
        audio_streams = [s for s in self.media_streams.values() if s.media_type == "audio"]
        for stream in audio_streams:
            if stream.current_packet_loss_percent > 2.0:
                self.recommendations.append(
                    "Consider reducing audio bitrate or switching to a more resilient codec"
                )
            
            if stream.current_jitter_ms and stream.current_jitter_ms > 30:
                self.recommendations.append(
                    "High jitter detected - enable jitter buffer or increase buffer size"
                )
        
        # Check video quality
        video_streams = [s for s in self.media_streams.values() if s.media_type == "video"]
        for stream in video_streams:
            if stream.current_bitrate_bps < 200000:  # Less than 200 kbps
                self.recommendations.append(
                    "Video bitrate is low - consider disabling video or reducing resolution"
                )
        
        # Check connection quality
        if self.connection_stats.avg_rtt_ms > 150:
            self.recommendations.append(
                "High latency detected - consider using a closer media server"
            )
    
    # Report generation methods
    async def generate_report(self, report_type: ReportType = ReportType.SUMMARY) -> Dict[str, Any]:
        """Generate statistics report."""
        if report_type == ReportType.REAL_TIME:
            return self._generate_realtime_report()
        elif report_type == ReportType.SUMMARY:
            return self._generate_summary_report()
        elif report_type == ReportType.DETAILED:
            return self._generate_detailed_report()
        elif report_type == ReportType.HISTORICAL:
            return self._generate_historical_report()
        elif report_type == ReportType.DIAGNOSTIC:
            return self._generate_diagnostic_report()
        elif report_type == ReportType.PERFORMANCE:
            return self._generate_performance_report()
        else:
            return {}
    
    def _generate_realtime_report(self) -> Dict[str, Any]:
        """Generate real-time statistics report."""
        return {
            "timestamp": time.time(),
            "connection": {
                "id": self.connection_stats.connection_id,
                "local_candidate": self.connection_stats.local_candidate_type,
                "remote_candidate": self.connection_stats.remote_candidate_type,
                "rtt_ms": self.connection_stats.avg_rtt_ms
            },
            "streams": {
                stream_id: {
                    "type": stream.media_type,
                    "direction": stream.direction,
                    "bitrate_bps": stream.current_bitrate_bps,
                    "packet_loss_percent": stream.current_packet_loss_percent,
                    "jitter_ms": stream.current_jitter_ms
                }
                for stream_id, stream in self.media_streams.items()
            },
            "quality_trends": self.quality_trends
        }
    
    def _generate_summary_report(self) -> Dict[str, Any]:
        """Generate summary statistics report."""
        return {
            "report_time": datetime.now().isoformat(),
            "connection_summary": {
                "total_bytes_sent": self.connection_stats.total_bytes_sent,
                "total_bytes_received": self.connection_stats.total_bytes_received,
                "total_packets_lost": self.connection_stats.total_packets_lost,
                "avg_rtt_ms": self.connection_stats.avg_rtt_ms,
                "connection_type": f"{self.connection_stats.local_candidate_type} -> {self.connection_stats.remote_candidate_type}"
            },
            "media_summary": {
                stream_id: {
                    "media_type": stream.media_type,
                    "direction": stream.direction,
                    "avg_bitrate_bps": stream.avg_bitrate_bps,
                    "max_bitrate_bps": stream.max_bitrate_bps,
                    "min_bitrate_bps": stream.min_bitrate_bps,
                    "quality_score": stream.quality_score
                }
                for stream_id, stream in self.media_streams.items()
            },
            "issues_detected": len(self.anomalies),
            "recommendations": self.recommendations
        }
    
    def _generate_detailed_report(self) -> Dict[str, Any]:
        """Generate detailed statistics report."""
        return {
            "report_time": datetime.now().isoformat(),
            "full_snapshot": {
                stat_id: asdict(stat)
                for stat_id, stat in self.current_stats.items()
            },
            "connection_stats": asdict(self.connection_stats),
            "media_streams": {
                stream_id: asdict(stream)
                for stream_id, stream in self.media_streams.items()
            },
            "anomalies": self.anomalies,
            "quality_trends": self.quality_trends,
            "recommendations": self.recommendations
        }
    
    def _generate_historical_report(self) -> Dict[str, Any]:
        """Generate historical statistics report."""
        # Get time series data for the last hour
        time_series = defaultdict(list)
        
        for snapshot in self.snapshots:
            timestamp = snapshot['timestamp']
            
            # Aggregate metrics at each timestamp
            total_bitrate = 0
            total_packet_loss = 0
            stream_count = 0
            
            for stat in snapshot['stats'].values():
                if stat.type == StatType.INBOUND_RTP.value:
                    # Calculate metrics from historical data
                    pass  # Implementation depends on specific needs
            
            time_series['timestamps'].append(timestamp)
            time_series['avg_bitrate'].append(total_bitrate / max(stream_count, 1))
            time_series['avg_packet_loss'].append(total_packet_loss / max(stream_count, 1))
        
        return {
            "report_time": datetime.now().isoformat(),
            "time_range": {
                "start": time_series['timestamps'][0] if time_series['timestamps'] else None,
                "end": time_series['timestamps'][-1] if time_series['timestamps'] else None,
                "duration_seconds": len(time_series['timestamps'])
            },
            "time_series": dict(time_series),
            "statistics": {
                "samples": len(self.snapshots),
                "collection_interval_ms": self.collection_interval_ms
            }
        }
    
    def _generate_diagnostic_report(self) -> Dict[str, Any]:
        """Generate diagnostic report for troubleshooting."""
        return {
            "report_time": datetime.now().isoformat(),
            "connection_diagnostics": {
                "connection_type": f"{self.connection_stats.local_candidate_type} -> {self.connection_stats.remote_candidate_type}",
                "network_path": self._diagnose_network_path(),
                "quality_issues": self.connection_stats.quality_issues,
                "reconnections": self.connection_stats.reconnection_count
            },
            "stream_diagnostics": {
                stream_id: {
                    "status": self._diagnose_stream_health(stream),
                    "issues": self._diagnose_stream_issues(stream),
                    "trend": self.quality_trends.get(stream_id, "unknown")
                }
                for stream_id, stream in self.media_streams.items()
            },
            "detected_problems": self.anomalies,
            "recommended_actions": self.recommendations
        }
    
    def _generate_performance_report(self) -> Dict[str, Any]:
        """Generate performance metrics report."""
        # Calculate performance scores
        audio_performance = self._calculate_audio_performance()
        video_performance = self._calculate_video_performance()
        network_performance = self._calculate_network_performance()
        
        return {
            "report_time": datetime.now().isoformat(),
            "overall_score": (audio_performance + video_performance + network_performance) / 3,
            "audio_performance": audio_performance,
            "video_performance": video_performance,
            "network_performance": network_performance,
            "detailed_scores": {
                "audio": {
                    "quality": self._calculate_audio_quality_score(),
                    "stability": self._calculate_audio_stability_score(),
                    "latency": self._calculate_audio_latency_score()
                },
                "video": {
                    "quality": self._calculate_video_quality_score(),
                    "stability": self._calculate_video_stability_score(),
                    "framerate": self._calculate_video_framerate_score()
                },
                "network": {
                    "bandwidth": self._calculate_bandwidth_score(),
                    "latency": self._calculate_latency_score(),
                    "stability": self._calculate_network_stability_score()
                }
            }
        }
    
    # Diagnostic helper methods
    def _diagnose_network_path(self) -> str:
        """Diagnose network path quality."""
        if self.connection_stats.local_candidate_type == "relay":
            return "Using TURN relay - possible firewall/NAT issues"
        elif self.connection_stats.local_candidate_type == "srflx":
            return "Using STUN - moderate NAT traversal"
        elif self.connection_stats.local_candidate_type == "host":
            return "Direct connection - optimal path"
        return "Unknown network path"
    
    def _diagnose_stream_health(self, stream: MediaStreamStats) -> str:
        """Diagnose stream health status."""
        if stream.current_packet_loss_percent > 5:
            return "critical"
        elif stream.current_packet_loss_percent > 2:
            return "degraded"
        elif stream.current_bitrate_bps < stream.avg_bitrate_bps * 0.7:
            return "unstable"
        return "healthy"
    
    def _diagnose_stream_issues(self, stream: MediaStreamStats) -> List[str]:
        """Diagnose specific stream issues."""
        issues = []
        
        if stream.current_packet_loss_percent > 2:
            issues.append("High packet loss")
        
        if stream.current_jitter_ms and stream.current_jitter_ms > 30:
            issues.append("High jitter")
        
        if stream.current_bitrate_bps < stream.min_bitrate_bps * 1.1:
            issues.append("Low bitrate")
        
        return issues
    
    # Performance calculation methods
    def _calculate_audio_performance(self) -> float:
        """Calculate audio performance score (0-100)."""
        audio_streams = [s for s in self.media_streams.values() if s.media_type == "audio"]
        if not audio_streams:
            return 100.0
        
        scores = []
        for stream in audio_streams:
            score = 100.0
            
            # Deduct for packet loss
            score -= min(stream.current_packet_loss_percent * 10, 50)
            
            # Deduct for jitter
            if stream.current_jitter_ms:
                score -= min(stream.current_jitter_ms / 10, 20)
            
            # Deduct for low bitrate
            if stream.avg_bitrate_bps > 0:
                bitrate_ratio = stream.current_bitrate_bps / stream.avg_bitrate_bps
                if bitrate_ratio < 0.8:
                    score -= (1 - bitrate_ratio) * 30
            
            scores.append(max(0, score))
        
        return statistics.mean(scores) if scores else 100.0
    
    def _calculate_video_performance(self) -> float:
        """Calculate video performance score (0-100)."""
        video_streams = [s for s in self.media_streams.values() if s.media_type == "video"]
        if not video_streams:
            return 100.0
        
        scores = []
        for stream in video_streams:
            score = 100.0
            
            # Similar scoring logic for video
            scores.append(max(0, score))
        
        return statistics.mean(scores) if scores else 100.0
    
    def _calculate_network_performance(self) -> float:
        """Calculate network performance score (0-100)."""
        score = 100.0
        
        # Deduct for RTT
        if self.connection_stats.avg_rtt_ms > 0:
            if self.connection_stats.avg_rtt_ms > 200:
                score -= min((self.connection_stats.avg_rtt_ms - 200) / 10, 30)
        
        # Deduct for packet loss
        if self.connection_stats.total_packets_lost > 0:
            loss_rate = self.connection_stats.total_packets_lost / max(
                self.connection_stats.total_packets_sent + self.connection_stats.total_packets_received, 1
            )
            score -= min(loss_rate * 1000, 40)
        
        return max(0, score)
    
    def _calculate_audio_quality_score(self) -> float:
        """Calculate audio quality sub-score."""
        # Implementation based on audio-specific metrics
        return 85.0  # Placeholder
    
    def _calculate_audio_stability_score(self) -> float:
        """Calculate audio stability sub-score."""
        return 90.0  # Placeholder
    
    def _calculate_audio_latency_score(self) -> float:
        """Calculate audio latency sub-score."""
        return 88.0  # Placeholder
    
    def _calculate_video_quality_score(self) -> float:
        """Calculate video quality sub-score."""
        return 82.0  # Placeholder
    
    def _calculate_video_stability_score(self) -> float:
        """Calculate video stability sub-score."""
        return 85.0  # Placeholder
    
    def _calculate_video_framerate_score(self) -> float:
        """Calculate video framerate sub-score."""
        return 90.0  # Placeholder
    
    def _calculate_bandwidth_score(self) -> float:
        """Calculate bandwidth utilization score."""
        return 87.0  # Placeholder
    
    def _calculate_latency_score(self) -> float:
        """Calculate network latency score."""
        if self.connection_stats.avg_rtt_ms == 0:
            return 100.0
        
        if self.connection_stats.avg_rtt_ms < 50:
            return 100.0
        elif self.connection_stats.avg_rtt_ms < 150:
            return 90.0
        elif self.connection_stats.avg_rtt_ms < 300:
            return 70.0
        else:
            return max(50.0 - (self.connection_stats.avg_rtt_ms - 300) / 10, 0)
    
    def _calculate_network_stability_score(self) -> float:
        """Calculate network stability score."""
        return 88.0  # Placeholder
    
    # Event callbacks
    async def _trigger_stats_updated(self):
        """Trigger stats updated callbacks."""
        for callback in self.on_stats_updated_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(self.current_stats)
                else:
                    callback(self.current_stats)
            except Exception as e:
                logger.error(f"Error in stats updated callback: {e}")
    
    async def _trigger_quality_degraded(self, stream_id: str, stream_stats: MediaStreamStats):
        """Trigger quality degraded callbacks."""
        for callback in self.on_quality_degraded_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(stream_id, stream_stats)
                else:
                    callback(stream_id, stream_stats)
            except Exception as e:
                logger.error(f"Error in quality degraded callback: {e}")
    
    async def _trigger_anomaly_detected(self, anomaly: Dict[str, Any]):
        """Trigger anomaly detected callbacks."""
        for callback in self.on_anomaly_detected_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(anomaly)
                else:
                    callback(anomaly)
            except Exception as e:
                logger.error(f"Error in anomaly detected callback: {e}")
    
    # Callback registration
    def on_stats_updated(self, callback: Callable):
        """Register callback for stats update events."""
        self.on_stats_updated_callbacks.append(callback)
    
    def on_quality_degraded(self, callback: Callable):
        """Register callback for quality degradation events."""
        self.on_quality_degraded_callbacks.append(callback)
    
    def on_anomaly_detected(self, callback: Callable):
        """Register callback for anomaly detection events."""
        self.on_anomaly_detected_callbacks.append(callback)
    
    # Export methods
    def export_to_json(self, filepath: str, report_type: ReportType = ReportType.DETAILED):
        """Export statistics to JSON file."""
        try:
            report = asyncio.run(self.generate_report(report_type))
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Exported statistics to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export statistics: {e}")
    
    def export_to_csv(self, filepath: str):
        """Export time series data to CSV."""
        if not PANDAS_AVAILABLE:
            logger.error("pandas not available for CSV export")
            return
        
        try:
            # Create DataFrame from snapshots
            data = []
            for snapshot in self.snapshots:
                row = {'timestamp': snapshot['timestamp']}
                
                # Extract key metrics
                for stat in snapshot['stats'].values():
                    if stat.type == StatType.INBOUND_RTP.value and stat.media_type == "audio":
                        row['audio_bytes_received'] = stat.bytes_received
                        row['audio_packets_lost'] = stat.packets_lost
                        row['audio_jitter'] = stat.jitter
                
                data.append(row)
            
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)
            logger.info(f"Exported time series to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export CSV: {e}")
    
    # Cleanup
    async def cleanup(self):
        """Clean up resources."""
        await self.stop_collection()
        logger.info("WebRTC statistics collector cleaned up")


# Convenience functions
def create_stats_collector(**kwargs) -> WebRTCStatsCollector:
    """Create WebRTC statistics collector."""
    return WebRTCStatsCollector(**kwargs)


# Global stats collector
_global_stats_collector: Optional[WebRTCStatsCollector] = None


def get_global_stats_collector() -> Optional[WebRTCStatsCollector]:
    """Get global statistics collector."""
    return _global_stats_collector


def set_global_stats_collector(collector: WebRTCStatsCollector):
    """Set global statistics collector."""
    global _global_stats_collector
    _global_stats_collector = collector