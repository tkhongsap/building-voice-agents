"""
Media Quality Monitoring and Adaptive Bitrate Control

This module provides comprehensive media quality monitoring, automatic bitrate 
adaptation, and quality-of-service metrics for WebRTC communications.
"""

import asyncio
import logging
import time
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List, Callable, Tuple, Deque
from collections import deque, defaultdict
import json

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

from monitoring.performance_monitor import monitor_performance, global_performance_monitor

logger = logging.getLogger(__name__)


class MediaType(Enum):
    """Media stream types."""
    AUDIO = "audio"
    VIDEO = "video"
    SCREEN = "screen"


class QualityLevel(Enum):
    """Media quality levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNUSABLE = "unusable"


class AdaptationStrategy(Enum):
    """Bitrate adaptation strategies."""
    CONSERVATIVE = "conservative"  # Slow, cautious changes
    MODERATE = "moderate"         # Balanced approach
    AGGRESSIVE = "aggressive"     # Quick, responsive changes
    CUSTOM = "custom"            # User-defined strategy


@dataclass
class MediaQualityMetrics:
    """Real-time media quality metrics."""
    # Basic metrics
    bitrate_bps: Optional[float] = None
    packet_loss_percent: Optional[float] = None
    jitter_ms: Optional[float] = None
    latency_ms: Optional[float] = None
    
    # Audio-specific metrics
    audio_level: Optional[float] = None  # 0.0 to 1.0
    audio_energy: Optional[float] = None
    silence_ratio: Optional[float] = None
    
    # Video-specific metrics  
    framerate: Optional[float] = None
    resolution: Optional[Tuple[int, int]] = None
    keyframe_interval: Optional[float] = None
    
    # Quality indicators
    mos_score: Optional[float] = None  # Mean Opinion Score (1-5)
    quality_level: QualityLevel = QualityLevel.GOOD
    
    # Network metrics
    rtt_ms: Optional[float] = None  # Round-trip time
    bandwidth_mbps: Optional[float] = None
    congestion_level: Optional[float] = None  # 0.0 to 1.0
    
    # Timestamps
    timestamp: float = field(default_factory=time.time)
    measurement_duration_ms: Optional[float] = None


@dataclass
class BitrateConfig:
    """Bitrate configuration and limits."""
    # Audio bitrate limits (bps)
    min_audio_bitrate: int = 6000      # 6 kbps minimum
    max_audio_bitrate: int = 128000    # 128 kbps maximum
    default_audio_bitrate: int = 32000  # 32 kbps default
    
    # Video bitrate limits (bps)
    min_video_bitrate: int = 50000      # 50 kbps minimum
    max_video_bitrate: int = 4000000    # 4 Mbps maximum
    default_video_bitrate: int = 800000  # 800 kbps default
    
    # Adaptation parameters
    increase_factor: float = 1.2  # 20% increase
    decrease_factor: float = 0.8  # 20% decrease
    probe_factor: float = 1.5    # 50% probe increase
    
    # Thresholds
    packet_loss_threshold: float = 2.0   # 2% loss threshold
    jitter_threshold_ms: float = 30.0    # 30ms jitter threshold
    latency_threshold_ms: float = 150.0  # 150ms latency threshold
    
    # Adaptation timing
    adaptation_interval_ms: int = 1000    # Check every second
    stable_duration_ms: int = 5000       # 5 seconds stable before increase
    emergency_duration_ms: int = 500      # 500ms for emergency reduction


@dataclass
class QualityHistory:
    """Historical quality metrics for trend analysis."""
    metrics: Deque[MediaQualityMetrics] = field(default_factory=lambda: deque(maxlen=300))  # 5 minutes
    bitrate_changes: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=100))
    quality_events: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=100))
    
    # Statistics
    avg_bitrate: float = 0.0
    avg_packet_loss: float = 0.0
    avg_jitter: float = 0.0
    avg_latency: float = 0.0
    
    # Trends
    bitrate_trend: str = "stable"  # increasing, decreasing, stable
    quality_trend: str = "stable"
    
    def add_metrics(self, metrics: MediaQualityMetrics):
        """Add new metrics to history."""
        self.metrics.append(metrics)
        self._update_statistics()
    
    def _update_statistics(self):
        """Update running statistics."""
        if not self.metrics:
            return
        
        recent_metrics = list(self.metrics)[-60:]  # Last minute
        
        # Calculate averages
        bitrates = [m.bitrate_bps for m in recent_metrics if m.bitrate_bps]
        if bitrates:
            self.avg_bitrate = statistics.mean(bitrates)
        
        packet_losses = [m.packet_loss_percent for m in recent_metrics if m.packet_loss_percent is not None]
        if packet_losses:
            self.avg_packet_loss = statistics.mean(packet_losses)
        
        jitters = [m.jitter_ms for m in recent_metrics if m.jitter_ms]
        if jitters:
            self.avg_jitter = statistics.mean(jitters)
        
        latencies = [m.latency_ms for m in recent_metrics if m.latency_ms]
        if latencies:
            self.avg_latency = statistics.mean(latencies)


class MediaQualityMonitor:
    """Monitors media quality and provides adaptive bitrate control."""
    
    def __init__(self, 
                 media_type: MediaType = MediaType.AUDIO,
                 bitrate_config: BitrateConfig = None,
                 adaptation_strategy: AdaptationStrategy = AdaptationStrategy.MODERATE):
        self.media_type = media_type
        self.bitrate_config = bitrate_config or BitrateConfig()
        self.adaptation_strategy = adaptation_strategy
        
        # Current state
        self.current_bitrate = self._get_default_bitrate()
        self.target_bitrate = self.current_bitrate
        self.is_monitoring = False
        
        # Quality tracking
        self.current_metrics = MediaQualityMetrics()
        self.history = QualityHistory()
        
        # Adaptation state
        self.last_adaptation_time = 0.0
        self.stable_quality_start = 0.0
        self.consecutive_losses = 0
        self.probe_state = "idle"  # idle, probing, backing_off
        
        # Event callbacks
        self.on_quality_changed_callbacks: List[Callable] = []
        self.on_bitrate_changed_callbacks: List[Callable] = []
        self.on_quality_alert_callbacks: List[Callable] = []
        
        # Monitoring
        self.monitor = global_performance_monitor.register_component(
            f"media_quality_{media_type.value}", "communication"
        )
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._adaptation_task: Optional[asyncio.Task] = None
    
    def _get_default_bitrate(self) -> int:
        """Get default bitrate for media type."""
        if self.media_type == MediaType.AUDIO:
            return self.bitrate_config.default_audio_bitrate
        else:
            return self.bitrate_config.default_video_bitrate
    
    def _get_min_bitrate(self) -> int:
        """Get minimum bitrate for media type."""
        if self.media_type == MediaType.AUDIO:
            return self.bitrate_config.min_audio_bitrate
        else:
            return self.bitrate_config.min_video_bitrate
    
    def _get_max_bitrate(self) -> int:
        """Get maximum bitrate for media type."""
        if self.media_type == MediaType.AUDIO:
            return self.bitrate_config.max_audio_bitrate
        else:
            return self.bitrate_config.max_video_bitrate
    
    async def start_monitoring(self):
        """Start quality monitoring and adaptation."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        
        # Start monitoring tasks
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self._adaptation_task = asyncio.create_task(self._adaptation_loop())
        
        logger.info(f"Started {self.media_type.value} quality monitoring")
    
    async def stop_monitoring(self):
        """Stop quality monitoring."""
        self.is_monitoring = False
        
        # Cancel tasks
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        if self._adaptation_task:
            self._adaptation_task.cancel()
            try:
                await self._adaptation_task
            except asyncio.CancelledError:
                pass
        
        logger.info(f"Stopped {self.media_type.value} quality monitoring")
    
    @monitor_performance(component="media_quality", operation="update_metrics")
    async def update_metrics(self, metrics: Dict[str, Any]):
        """Update current quality metrics."""
        try:
            # Update basic metrics
            self.current_metrics.bitrate_bps = metrics.get('bitrate', self.current_metrics.bitrate_bps)
            self.current_metrics.packet_loss_percent = metrics.get('packet_loss', self.current_metrics.packet_loss_percent)
            self.current_metrics.jitter_ms = metrics.get('jitter', self.current_metrics.jitter_ms)
            self.current_metrics.latency_ms = metrics.get('latency', self.current_metrics.latency_ms)
            
            # Update audio metrics
            if self.media_type == MediaType.AUDIO:
                self.current_metrics.audio_level = metrics.get('audio_level', self.current_metrics.audio_level)
                self.current_metrics.audio_energy = metrics.get('audio_energy', self.current_metrics.audio_energy)
                self.current_metrics.silence_ratio = metrics.get('silence_ratio', self.current_metrics.silence_ratio)
            
            # Update video metrics
            elif self.media_type == MediaType.VIDEO:
                self.current_metrics.framerate = metrics.get('framerate', self.current_metrics.framerate)
                self.current_metrics.resolution = metrics.get('resolution', self.current_metrics.resolution)
                self.current_metrics.keyframe_interval = metrics.get('keyframe_interval', self.current_metrics.keyframe_interval)
            
            # Update network metrics
            self.current_metrics.rtt_ms = metrics.get('rtt', self.current_metrics.rtt_ms)
            self.current_metrics.bandwidth_mbps = metrics.get('bandwidth', self.current_metrics.bandwidth_mbps)
            self.current_metrics.congestion_level = metrics.get('congestion', self.current_metrics.congestion_level)
            
            # Calculate quality level
            self.current_metrics.quality_level = self._calculate_quality_level()
            
            # Calculate MOS score
            self.current_metrics.mos_score = self._calculate_mos_score()
            
            # Update timestamp
            self.current_metrics.timestamp = time.time()
            
            # Add to history
            self.history.add_metrics(MediaQualityMetrics(**self.current_metrics.__dict__))
            
            # Trigger quality changed callback
            await self._trigger_quality_changed()
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    def _calculate_quality_level(self) -> QualityLevel:
        """Calculate overall quality level from metrics."""
        # Simple quality calculation based on key metrics
        if self.current_metrics.packet_loss_percent is None:
            return QualityLevel.GOOD
        
        packet_loss = self.current_metrics.packet_loss_percent
        jitter = self.current_metrics.jitter_ms or 0
        latency = self.current_metrics.latency_ms or 0
        
        # Quality scoring
        if packet_loss < 0.5 and jitter < 20 and latency < 100:
            return QualityLevel.EXCELLENT
        elif packet_loss < 1.0 and jitter < 30 and latency < 150:
            return QualityLevel.GOOD
        elif packet_loss < 2.0 and jitter < 50 and latency < 250:
            return QualityLevel.FAIR
        elif packet_loss < 5.0 and jitter < 100 and latency < 400:
            return QualityLevel.POOR
        else:
            return QualityLevel.UNUSABLE
    
    def _calculate_mos_score(self) -> float:
        """Calculate Mean Opinion Score (1-5) from network metrics."""
        # E-model based MOS calculation (simplified)
        
        # Base score
        mos = 4.5
        
        # Deduct for packet loss
        if self.current_metrics.packet_loss_percent:
            loss_factor = min(self.current_metrics.packet_loss_percent * 0.3, 2.0)
            mos -= loss_factor
        
        # Deduct for latency
        if self.current_metrics.latency_ms:
            if self.current_metrics.latency_ms > 150:
                latency_factor = min((self.current_metrics.latency_ms - 150) * 0.01, 1.5)
                mos -= latency_factor
        
        # Deduct for jitter
        if self.current_metrics.jitter_ms:
            if self.current_metrics.jitter_ms > 30:
                jitter_factor = min((self.current_metrics.jitter_ms - 30) * 0.02, 1.0)
                mos -= jitter_factor
        
        # Ensure MOS is within valid range
        return max(1.0, min(5.0, mos))
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Analyze quality trends
                self._analyze_quality_trends()
                
                # Check for quality alerts
                await self._check_quality_alerts()
                
                # Sleep for monitoring interval
                await asyncio.sleep(1.0)  # Check every second
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(1.0)
    
    def _analyze_quality_trends(self):
        """Analyze quality trends from history."""
        if len(self.history.metrics) < 10:
            return
        
        recent_metrics = list(self.history.metrics)[-30:]  # Last 30 seconds
        
        # Analyze bitrate trend
        bitrates = [m.bitrate_bps for m in recent_metrics if m.bitrate_bps]
        if len(bitrates) >= 10:
            # Simple trend detection
            first_half_avg = statistics.mean(bitrates[:len(bitrates)//2])
            second_half_avg = statistics.mean(bitrates[len(bitrates)//2:])
            
            if second_half_avg > first_half_avg * 1.1:
                self.history.bitrate_trend = "increasing"
            elif second_half_avg < first_half_avg * 0.9:
                self.history.bitrate_trend = "decreasing"
            else:
                self.history.bitrate_trend = "stable"
        
        # Analyze quality trend
        quality_scores = [m.mos_score for m in recent_metrics if m.mos_score]
        if len(quality_scores) >= 10:
            first_half_avg = statistics.mean(quality_scores[:len(quality_scores)//2])
            second_half_avg = statistics.mean(quality_scores[len(quality_scores)//2:])
            
            if second_half_avg > first_half_avg * 1.05:
                self.history.quality_trend = "improving"
            elif second_half_avg < first_half_avg * 0.95:
                self.history.quality_trend = "degrading"
            else:
                self.history.quality_trend = "stable"
    
    async def _check_quality_alerts(self):
        """Check for quality issues that need alerts."""
        alerts = []
        
        # Check packet loss
        if (self.current_metrics.packet_loss_percent and 
            self.current_metrics.packet_loss_percent > 5.0):
            alerts.append({
                "type": "high_packet_loss",
                "severity": "critical",
                "value": self.current_metrics.packet_loss_percent,
                "message": f"High packet loss detected: {self.current_metrics.packet_loss_percent:.1f}%"
            })
        
        # Check latency
        if (self.current_metrics.latency_ms and 
            self.current_metrics.latency_ms > 300):
            alerts.append({
                "type": "high_latency",
                "severity": "warning",
                "value": self.current_metrics.latency_ms,
                "message": f"High latency detected: {self.current_metrics.latency_ms:.0f}ms"
            })
        
        # Check quality level
        if self.current_metrics.quality_level == QualityLevel.POOR:
            alerts.append({
                "type": "poor_quality",
                "severity": "warning",
                "message": "Overall media quality is poor"
            })
        elif self.current_metrics.quality_level == QualityLevel.UNUSABLE:
            alerts.append({
                "type": "unusable_quality",
                "severity": "critical",
                "message": "Media quality is unusable"
            })
        
        # Trigger alerts
        for alert in alerts:
            await self._trigger_quality_alert(alert)
    
    async def _adaptation_loop(self):
        """Bitrate adaptation loop."""
        while self.is_monitoring:
            try:
                current_time = time.time()
                
                # Check if enough time has passed since last adaptation
                if (current_time - self.last_adaptation_time < 
                    self.bitrate_config.adaptation_interval_ms / 1000):
                    await asyncio.sleep(0.1)
                    continue
                
                # Perform adaptation
                await self._adapt_bitrate()
                
                self.last_adaptation_time = current_time
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in adaptation loop: {e}")
                await asyncio.sleep(1.0)
    
    @monitor_performance(component="media_quality", operation="adapt_bitrate")
    async def _adapt_bitrate(self):
        """Adapt bitrate based on current conditions."""
        if not self.current_metrics.packet_loss_percent:
            return
        
        previous_bitrate = self.current_bitrate
        
        # Get adaptation decision
        decision = self._get_adaptation_decision()
        
        if decision == "increase":
            await self._increase_bitrate()
        elif decision == "decrease":
            await self._decrease_bitrate()
        elif decision == "emergency_decrease":
            await self._emergency_decrease_bitrate()
        
        # Log bitrate change if it occurred
        if self.current_bitrate != previous_bitrate:
            change_event = {
                "timestamp": time.time(),
                "previous_bitrate": previous_bitrate,
                "new_bitrate": self.current_bitrate,
                "reason": decision,
                "packet_loss": self.current_metrics.packet_loss_percent,
                "quality_level": self.current_metrics.quality_level.value
            }
            
            self.history.bitrate_changes.append(change_event)
            await self._trigger_bitrate_changed(previous_bitrate, self.current_bitrate, decision)
    
    def _get_adaptation_decision(self) -> str:
        """Determine adaptation decision based on current metrics."""
        # Emergency conditions
        if (self.current_metrics.packet_loss_percent > 10.0 or
            self.current_metrics.quality_level == QualityLevel.UNUSABLE):
            return "emergency_decrease"
        
        # Poor conditions
        if (self.current_metrics.packet_loss_percent > self.bitrate_config.packet_loss_threshold or
            (self.current_metrics.jitter_ms and 
             self.current_metrics.jitter_ms > self.bitrate_config.jitter_threshold_ms) or
            self.current_metrics.quality_level == QualityLevel.POOR):
            
            self.stable_quality_start = 0  # Reset stable timer
            return "decrease"
        
        # Check if quality has been stable
        current_time = time.time()
        
        if self.stable_quality_start == 0:
            self.stable_quality_start = current_time
            return "maintain"
        
        # Good conditions for sufficient time
        if (current_time - self.stable_quality_start > 
            self.bitrate_config.stable_duration_ms / 1000):
            
            # Only increase if below max and quality is good/excellent
            if (self.current_bitrate < self._get_max_bitrate() and
                self.current_metrics.quality_level in [QualityLevel.EXCELLENT, QualityLevel.GOOD]):
                return "increase"
        
        return "maintain"
    
    async def _increase_bitrate(self):
        """Increase bitrate."""
        if self.adaptation_strategy == AdaptationStrategy.CONSERVATIVE:
            factor = 1.1  # 10% increase
        elif self.adaptation_strategy == AdaptationStrategy.AGGRESSIVE:
            factor = self.bitrate_config.probe_factor  # Probe increase
        else:
            factor = self.bitrate_config.increase_factor
        
        new_bitrate = int(self.current_bitrate * factor)
        new_bitrate = min(new_bitrate, self._get_max_bitrate())
        
        if new_bitrate > self.current_bitrate:
            self.current_bitrate = new_bitrate
            self.target_bitrate = new_bitrate
            logger.info(f"Increased {self.media_type.value} bitrate to {new_bitrate} bps")
    
    async def _decrease_bitrate(self):
        """Decrease bitrate."""
        if self.adaptation_strategy == AdaptationStrategy.CONSERVATIVE:
            factor = 0.9  # 10% decrease
        elif self.adaptation_strategy == AdaptationStrategy.AGGRESSIVE:
            factor = 0.7  # 30% decrease
        else:
            factor = self.bitrate_config.decrease_factor
        
        new_bitrate = int(self.current_bitrate * factor)
        new_bitrate = max(new_bitrate, self._get_min_bitrate())
        
        if new_bitrate < self.current_bitrate:
            self.current_bitrate = new_bitrate
            self.target_bitrate = new_bitrate
            self.stable_quality_start = 0  # Reset stable timer
            logger.info(f"Decreased {self.media_type.value} bitrate to {new_bitrate} bps")
    
    async def _emergency_decrease_bitrate(self):
        """Emergency bitrate reduction."""
        # Drop to 50% or minimum
        new_bitrate = max(int(self.current_bitrate * 0.5), self._get_min_bitrate())
        
        if new_bitrate < self.current_bitrate:
            self.current_bitrate = new_bitrate
            self.target_bitrate = new_bitrate
            self.stable_quality_start = 0
            logger.warning(f"Emergency: Reduced {self.media_type.value} bitrate to {new_bitrate} bps")
    
    # Manual control methods
    async def set_target_bitrate(self, bitrate: int):
        """Manually set target bitrate."""
        bitrate = max(self._get_min_bitrate(), min(bitrate, self._get_max_bitrate()))
        
        previous_bitrate = self.current_bitrate
        self.current_bitrate = bitrate
        self.target_bitrate = bitrate
        
        logger.info(f"Manually set {self.media_type.value} bitrate to {bitrate} bps")
        
        if bitrate != previous_bitrate:
            await self._trigger_bitrate_changed(previous_bitrate, bitrate, "manual")
    
    def set_adaptation_strategy(self, strategy: AdaptationStrategy):
        """Change adaptation strategy."""
        self.adaptation_strategy = strategy
        logger.info(f"Changed adaptation strategy to {strategy.value}")
    
    # Event callbacks
    async def _trigger_quality_changed(self):
        """Trigger quality changed callbacks."""
        for callback in self.on_quality_changed_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(self.current_metrics)
                else:
                    callback(self.current_metrics)
            except Exception as e:
                logger.error(f"Error in quality changed callback: {e}")
    
    async def _trigger_bitrate_changed(self, old_bitrate: int, new_bitrate: int, reason: str):
        """Trigger bitrate changed callbacks."""
        for callback in self.on_bitrate_changed_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(old_bitrate, new_bitrate, reason)
                else:
                    callback(old_bitrate, new_bitrate, reason)
            except Exception as e:
                logger.error(f"Error in bitrate changed callback: {e}")
    
    async def _trigger_quality_alert(self, alert: Dict[str, Any]):
        """Trigger quality alert callbacks."""
        for callback in self.on_quality_alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                logger.error(f"Error in quality alert callback: {e}")
    
    # Callback registration
    def on_quality_changed(self, callback: Callable):
        """Register callback for quality change events."""
        self.on_quality_changed_callbacks.append(callback)
    
    def on_bitrate_changed(self, callback: Callable):
        """Register callback for bitrate change events."""
        self.on_bitrate_changed_callbacks.append(callback)
    
    def on_quality_alert(self, callback: Callable):
        """Register callback for quality alert events."""
        self.on_quality_alert_callbacks.append(callback)
    
    # Status methods
    def get_current_metrics(self) -> MediaQualityMetrics:
        """Get current quality metrics."""
        return self.current_metrics
    
    def get_current_bitrate(self) -> int:
        """Get current bitrate."""
        return self.current_bitrate
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """Get quality summary."""
        return {
            "media_type": self.media_type.value,
            "current_bitrate": self.current_bitrate,
            "target_bitrate": self.target_bitrate,
            "quality_level": self.current_metrics.quality_level.value,
            "mos_score": self.current_metrics.mos_score,
            "packet_loss": self.current_metrics.packet_loss_percent,
            "latency": self.current_metrics.latency_ms,
            "jitter": self.current_metrics.jitter_ms,
            "adaptation_strategy": self.adaptation_strategy.value,
            "quality_trend": self.history.quality_trend,
            "bitrate_trend": self.history.bitrate_trend
        }
    
    def get_quality_history(self) -> List[MediaQualityMetrics]:
        """Get quality metrics history."""
        return list(self.history.metrics)
    
    def get_adaptation_history(self) -> List[Dict[str, Any]]:
        """Get bitrate adaptation history."""
        return list(self.history.bitrate_changes)
    
    async def cleanup(self):
        """Clean up resources."""
        await self.stop_monitoring()
        logger.info(f"{self.media_type.value} quality monitor cleaned up")


# Convenience functions
def create_audio_quality_monitor(**kwargs) -> MediaQualityMonitor:
    """Create audio quality monitor."""
    return MediaQualityMonitor(MediaType.AUDIO, **kwargs)


def create_video_quality_monitor(**kwargs) -> MediaQualityMonitor:
    """Create video quality monitor."""
    return MediaQualityMonitor(MediaType.VIDEO, **kwargs)


# Global monitors
_global_audio_monitor: Optional[MediaQualityMonitor] = None
_global_video_monitor: Optional[MediaQualityMonitor] = None


def get_audio_quality_monitor() -> Optional[MediaQualityMonitor]:
    """Get global audio quality monitor."""
    return _global_audio_monitor


def set_audio_quality_monitor(monitor: MediaQualityMonitor):
    """Set global audio quality monitor."""
    global _global_audio_monitor
    _global_audio_monitor = monitor


def get_video_quality_monitor() -> Optional[MediaQualityMonitor]:
    """Get global video quality monitor."""
    return _global_video_monitor


def set_video_quality_monitor(monitor: MediaQualityMonitor):
    """Set global video quality monitor."""
    global _global_video_monitor
    _global_video_monitor = monitor