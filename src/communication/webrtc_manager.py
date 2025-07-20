"""
WebRTC Connection Manager for LiveKit Voice Agents Platform

This module provides WebRTC connection management, peer-to-peer communication,
and media handling using LiveKit infrastructure.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List, Callable, Union
from contextlib import asynccontextmanager

try:
    from livekit import (
        Room, RoomOptions, VideoQuality, AudioSource, AudioFrame,
        TrackSource, RemoteAudioTrack, RemoteVideoTrack, RtcConfiguration,
        IceServer, ConnectOptions, LocalAudioTrack, LocalVideoTrack
    )
    LIVEKIT_AVAILABLE = True
except ImportError:
    LIVEKIT_AVAILABLE = False
    # Mock classes for development without LiveKit
    class Room: pass
    class RoomOptions: pass
    class VideoQuality: pass
    class AudioSource: pass
    class AudioFrame: pass
    class TrackSource: pass
    class RemoteAudioTrack: pass
    class RemoteVideoTrack: pass
    class RtcConfiguration: pass
    class IceServer: pass
    class ConnectOptions: pass
    class LocalAudioTrack: pass
    class LocalVideoTrack: pass

from monitoring.performance_monitor import monitor_performance, global_performance_monitor

# Import DTMF detection capabilities
try:
    from .dtmf_detector import DTMFDetector, DTMFConfig, DTMFDetection, create_dtmf_detector
    DTMF_AVAILABLE = True
except ImportError:
    DTMF_AVAILABLE = False
    DTMFDetector = None
    DTMFConfig = None
    DTMFDetection = None

# Import security management capabilities
try:
    from .security_manager import SecurityManager, SecurityConfig, SecurityLevel, SecurityMetrics
    SECURITY_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False
    SecurityManager = None
    SecurityConfig = None
    SecurityLevel = None
    SecurityMetrics = None

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """WebRTC connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


class NetworkQuality(Enum):
    """Network quality levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNKNOWN = "unknown"


class CodecType(Enum):
    """Supported audio/video codecs."""
    OPUS = "opus"
    G711_PCMU = "PCMU"
    G711_PCMA = "PCMA"
    H264 = "H264"
    VP8 = "VP8"
    VP9 = "VP9"


class CodecPreference(Enum):
    """Codec preference based on quality and efficiency."""
    HIGH_QUALITY = "high_quality"  # Prioritize audio quality
    LOW_BANDWIDTH = "low_bandwidth"  # Prioritize bandwidth efficiency
    LOW_LATENCY = "low_latency"  # Prioritize low latency
    COMPATIBILITY = "compatibility"  # Prioritize broad compatibility


# Codec quality rankings and characteristics
CODEC_CHARACTERISTICS = {
    CodecType.OPUS: {
        "quality_score": 95,
        "bandwidth_efficiency": 90,
        "latency_ms": 20,
        "cpu_usage": 60,
        "sample_rates": [8000, 16000, 24000, 48000],
        "bitrates": {"min": 6000, "max": 510000, "default": 64000}
    },
    CodecType.G711_PCMU: {
        "quality_score": 70,
        "bandwidth_efficiency": 40,
        "latency_ms": 10,
        "cpu_usage": 20,
        "sample_rates": [8000],
        "bitrates": {"min": 64000, "max": 64000, "default": 64000}
    },
    CodecType.G711_PCMA: {
        "quality_score": 70,
        "bandwidth_efficiency": 40,
        "latency_ms": 10,
        "cpu_usage": 20,
        "sample_rates": [8000],
        "bitrates": {"min": 64000, "max": 64000, "default": 64000}
    }
}

# Codec preference rankings
CODEC_PREFERENCES = {
    CodecPreference.HIGH_QUALITY: [CodecType.OPUS, CodecType.G711_PCMU, CodecType.G711_PCMA],
    CodecPreference.LOW_BANDWIDTH: [CodecType.OPUS, CodecType.G711_PCMU, CodecType.G711_PCMA],
    CodecPreference.LOW_LATENCY: [CodecType.G711_PCMU, CodecType.G711_PCMA, CodecType.OPUS],
    CodecPreference.COMPATIBILITY: [CodecType.G711_PCMU, CodecType.OPUS, CodecType.G711_PCMA]
}


@dataclass
class ConnectionConfig:
    """WebRTC connection configuration."""
    room_url: str
    token: str
    participant_name: Optional[str] = None
    
    # ICE configuration
    ice_servers: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]}
    ])
    
    # Connection settings
    auto_subscribe: bool = True
    adaptive_stream: bool = True
    dynacast: bool = True
    
    # Audio settings
    audio_enabled: bool = True
    echo_cancellation: bool = True
    noise_suppression: bool = True
    auto_gain_control: bool = True
    
    # DTMF settings
    dtmf_detection_enabled: bool = True
    dtmf_sample_rate: int = 8000
    
    # Security settings
    security_enabled: bool = True
    security_level: SecurityLevel = SecurityLevel.ENHANCED if SECURITY_AVAILABLE else None
    require_encryption: bool = True
    
    # Video settings
    video_enabled: bool = False
    video_quality: str = "high"
    
    # Codec and quality adaptation settings
    codec_preference: CodecPreference = CodecPreference.HIGH_QUALITY
    adaptive_bitrate: bool = True
    auto_codec_selection: bool = True
    min_bitrate_kbps: int = 16  # Minimum bitrate
    max_bitrate_kbps: int = 128  # Maximum bitrate
    target_bitrate_kbps: int = 64  # Target bitrate
    
    # Quality adaptation thresholds
    quality_adaptation_enabled: bool = True
    packet_loss_threshold_percent: float = 2.0  # Switch to lower quality
    latency_threshold_ms: float = 200.0  # Switch to lower latency codec
    bandwidth_threshold_kbps: float = 50.0  # Switch to lower bitrate
    
    # Timeouts and retry
    connection_timeout: float = 30.0
    reconnect_attempts: int = 5
    reconnect_delay: float = 2.0
    
    # Advanced reconnection settings
    max_reconnect_delay: float = 30.0  # Maximum delay between attempts
    connection_health_check_interval: float = 10.0  # Health check frequency
    persistent_reconnection: bool = True  # Keep trying indefinitely after initial attempts
    fast_reconnect_window_seconds: float = 30.0  # Use fast reconnect within this window
    backup_ice_servers: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun3.l.google.com:19302"]}
    ])


@dataclass
class ConnectionMetrics:
    """WebRTC connection metrics."""
    connected_at: Optional[float] = None
    last_ping: Optional[float] = None
    latency_ms: Optional[float] = None
    jitter_ms: Optional[float] = None
    packet_loss_percent: Optional[float] = None
    bitrate_kbps: Optional[float] = None
    network_quality: NetworkQuality = NetworkQuality.UNKNOWN
    
    # Audio metrics
    audio_level: Optional[float] = None
    audio_packets_sent: int = 0
    audio_packets_received: int = 0
    audio_bytes_sent: int = 0
    audio_bytes_received: int = 0
    
    # Connection stability
    reconnect_count: int = 0
    total_downtime_ms: float = 0.0
    connection_state: ConnectionState = ConnectionState.DISCONNECTED
    
    # Advanced reconnection metrics
    last_disconnect_time: Optional[float] = None
    last_reconnect_attempt_time: Optional[float] = None
    successful_reconnects: int = 0
    failed_reconnects: int = 0
    current_reconnect_strategy: str = "standard"
    connection_stability_score: float = 1.0  # 0-1 based on recent stability
    
    # Codec and quality metrics
    current_audio_codec: Optional[CodecType] = None
    current_bitrate_kbps: Optional[float] = None
    target_bitrate_kbps: Optional[float] = None
    codec_switches: int = 0
    quality_adaptations: int = 0
    
    # Bandwidth estimation
    estimated_bandwidth_kbps: Optional[float] = None
    bandwidth_utilization_percent: Optional[float] = None


class WebRTCManager:
    """Manages WebRTC connections and media streams for voice agents."""
    
    def __init__(self, config: ConnectionConfig):
        self.config = config
        self.room: Optional[Room] = None
        self.connection_state = ConnectionState.DISCONNECTED
        self.metrics = ConnectionMetrics()
        
        # Event callbacks
        self.on_connected_callbacks: List[Callable] = []
        self.on_disconnected_callbacks: List[Callable] = []
        self.on_participant_connected_callbacks: List[Callable] = []
        self.on_participant_disconnected_callbacks: List[Callable] = []
        self.on_track_subscribed_callbacks: List[Callable] = []
        self.on_track_unsubscribed_callbacks: List[Callable] = []
        self.on_audio_frame_callbacks: List[Callable] = []
        self.on_network_quality_changed_callbacks: List[Callable] = []
        self.on_dtmf_detected_callbacks: List[Callable] = []
        
        # Audio processing
        self.audio_source: Optional[AudioSource] = None
        self.local_audio_track: Optional[LocalAudioTrack] = None
        
        # DTMF processing
        self.dtmf_detector: Optional[DTMFDetector] = None
        if config.dtmf_detection_enabled and DTMF_AVAILABLE:
            dtmf_config = DTMFConfig(sample_rate=config.dtmf_sample_rate)
            self.dtmf_detector = DTMFDetector(dtmf_config)
            self._setup_dtmf_callbacks()
        
        # Security management
        self.security_manager: Optional[SecurityManager] = None
        if config.security_enabled and SECURITY_AVAILABLE and config.security_level:
            security_config = SecurityConfig(security_level=config.security_level)
            self.security_manager = SecurityManager(security_config)
            self._setup_security_callbacks()
        
        # Monitoring
        self.monitor = global_performance_monitor.register_component(
            "webrtc_manager", "communication"
        )
        
        # Connection management
        self._reconnect_task: Optional[asyncio.Task] = None
        self._metrics_task: Optional[asyncio.Task] = None
        self._is_monitoring = False
        
        # Quality adaptation state
        self._last_adaptation_time = 0.0
        self._adaptation_cooldown_seconds = 5.0  # Wait 5 seconds between adaptations
        self._current_codec_preference = config.codec_preference
        
        # Advanced reconnection state
        self._health_check_task: Optional[asyncio.Task] = None
        self._persistent_reconnect_task: Optional[asyncio.Task] = None
        self._connection_history: List[Dict[str, Any]] = []
        self._ice_server_index = 0  # For cycling through backup servers
        self._fast_reconnect_deadline: Optional[float] = None
    
    @monitor_performance(component="webrtc_manager", operation="connect")
    async def connect(self) -> bool:
        """Connect to LiveKit room."""
        if not LIVEKIT_AVAILABLE:
            logger.warning("LiveKit not available, using mock connection")
            self.connection_state = ConnectionState.CONNECTED
            self.metrics.connected_at = time.time()
            await self._trigger_connected()
            return True
        
        try:
            self.connection_state = ConnectionState.CONNECTING
            logger.info(f"Connecting to room: {self.config.room_url}")
            
            # Create room with options
            room_options = RoomOptions(
                auto_subscribe=self.config.auto_subscribe,
                adaptive_stream=self.config.adaptive_stream,
                dynacast=self.config.dynacast
            )
            
            self.room = Room(room_options)
            
            # Set up event handlers
            self._setup_event_handlers()
            
            # Create connection options
            connect_options = ConnectOptions(
                auto_subscribe=self.config.auto_subscribe,
                rtc_config=self._create_rtc_config()
            )
            
            # Connect to room
            await asyncio.wait_for(
                self.room.connect(
                    self.config.room_url,
                    self.config.token,
                    options=connect_options
                ),
                timeout=self.config.connection_timeout
            )
            
            self.connection_state = ConnectionState.CONNECTED
            self.metrics.connected_at = time.time()
            
            # Start monitoring
            await self._start_monitoring()
            
            # Start health checking
            await self._start_health_checking()
            
            # Set up audio if enabled
            if self.config.audio_enabled:
                await self._setup_audio()
            
            # Initialize security if enabled
            if self.security_manager:
                await self._setup_security()
            
            logger.info("Successfully connected to LiveKit room")
            await self._trigger_connected()
            
            return True
            
        except asyncio.TimeoutError:
            logger.error("Connection timeout")
            self.connection_state = ConnectionState.FAILED
            return False
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self.connection_state = ConnectionState.FAILED
            return False
    
    def _create_rtc_config(self) -> 'RtcConfiguration':
        """Create RTC configuration with ICE servers."""
        if not LIVEKIT_AVAILABLE:
            return None
        
        ice_servers = []
        for server_config in self.config.ice_servers:
            ice_server = IceServer(
                urls=server_config["urls"],
                username=server_config.get("username"),
                credential=server_config.get("credential")
            )
            ice_servers.append(ice_server)
        
        return RtcConfiguration(ice_servers=ice_servers)
    
    def _setup_event_handlers(self):
        """Set up LiveKit room event handlers."""
        if not self.room:
            return
        
        @self.room.on("connected")
        def on_connected():
            asyncio.create_task(self._on_room_connected())
        
        @self.room.on("disconnected")
        def on_disconnected():
            asyncio.create_task(self._on_room_disconnected())
        
        @self.room.on("participant_connected")
        def on_participant_connected(participant):
            asyncio.create_task(self._on_participant_connected(participant))
        
        @self.room.on("participant_disconnected")
        def on_participant_disconnected(participant):
            asyncio.create_task(self._on_participant_disconnected(participant))
        
        @self.room.on("track_subscribed")
        def on_track_subscribed(track, publication, participant):
            asyncio.create_task(self._on_track_subscribed(track, publication, participant))
        
        @self.room.on("track_unsubscribed")
        def on_track_unsubscribed(track, publication, participant):
            asyncio.create_task(self._on_track_unsubscribed(track, publication, participant))
    
    async def _setup_audio(self):
        """Set up local audio track."""
        if not LIVEKIT_AVAILABLE:
            logger.info("Mock audio setup completed")
            return
        
        try:
            # Create audio source
            self.audio_source = AudioSource(
                sample_rate=16000,
                num_channels=1
            )
            
            # Create local audio track
            self.local_audio_track = LocalAudioTrack.create_audio_track(
                "microphone",
                self.audio_source
            )
            
            # Publish the track
            await self.room.local_participant.publish_track(
                self.local_audio_track,
                TrackSource.SOURCE_MICROPHONE
            )
            
            logger.info("Audio track published successfully")
            
        except Exception as e:
            logger.error(f"Failed to set up audio: {e}")
    
    async def _setup_security(self):
        """Set up security for the connection."""
        if not self.security_manager or not SECURITY_AVAILABLE:
            logger.info("Mock security setup completed")
            return
        
        try:
            # Initialize security manager
            await self.security_manager.initialize()
            
            # Establish secure connection
            success = await self.security_manager.establish_secure_connection()
            
            if success:
                logger.info("Security established successfully")
            else:
                logger.error("Failed to establish security")
                if self.config.require_encryption:
                    raise RuntimeError("Security establishment required but failed")
            
        except Exception as e:
            logger.error(f"Failed to set up security: {e}")
            if self.config.require_encryption:
                raise
    
    def _setup_security_callbacks(self):
        """Set up security manager callbacks."""
        if not self.security_manager:
            return
        
        # Register security event callbacks
        self.security_manager.on_security_established(self._on_security_established)
        self.security_manager.on_security_violation(self._on_security_violation)
        self.security_manager.on_certificate_expiry(self._on_certificate_expiry)
    
    async def _on_security_established(self, security_metrics):
        """Handle security establishment event."""
        logger.info(f"Security established: {security_metrics.cipher_suite}")
        # Could trigger application callbacks here
    
    async def _on_security_violation(self, violation_type: str, details: Dict[str, Any]):
        """Handle security violation event."""
        logger.warning(f"Security violation detected: {violation_type} - {details}")
        # Could trigger application callbacks or take corrective action
    
    async def _on_certificate_expiry(self, time_to_expiry: float):
        """Handle certificate expiry warning."""
        days_to_expiry = time_to_expiry / (24 * 3600)
        logger.warning(f"Certificate expires in {days_to_expiry:.1f} days")
        # Could trigger application callbacks for certificate renewal
    
    async def _start_monitoring(self):
        """Start connection and metrics monitoring."""
        if self._is_monitoring:
            return
        
        self._is_monitoring = True
        self._metrics_task = asyncio.create_task(self._metrics_loop())
    
    async def _start_health_checking(self):
        """Start connection health checking."""
        if not self._health_check_task or self._health_check_task.done():
            self._health_check_task = asyncio.create_task(self._health_check_loop())
    
    async def _stop_monitoring(self):
        """Stop monitoring tasks."""
        self._is_monitoring = False
        
        if self._metrics_task:
            self._metrics_task.cancel()
            try:
                await self._metrics_task
            except asyncio.CancelledError:
                pass
        
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        if self._reconnect_task:
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass
        
        if self._persistent_reconnect_task:
            self._persistent_reconnect_task.cancel()
            try:
                await self._persistent_reconnect_task
            except asyncio.CancelledError:
                pass
    
    async def _metrics_loop(self):
        """Continuously collect connection metrics."""
        while self._is_monitoring:
            try:
                await self._collect_metrics()
                await asyncio.sleep(1.0)  # Collect metrics every second
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(1.0)
    
    async def _collect_metrics(self):
        """Collect current connection metrics."""
        if not self.room or not LIVEKIT_AVAILABLE:
            return
        
        try:
            # Get connection stats (this would be real LiveKit API calls)
            stats = await self._get_connection_stats()
            
            if stats:
                self.metrics.latency_ms = stats.get("latency_ms")
                self.metrics.jitter_ms = stats.get("jitter_ms")
                self.metrics.packet_loss_percent = stats.get("packet_loss_percent")
                self.metrics.bitrate_kbps = stats.get("bitrate_kbps")
                
                # Update network quality
                old_quality = self.metrics.network_quality
                self.metrics.network_quality = self._calculate_network_quality(stats)
                
                if old_quality != self.metrics.network_quality:
                    await self._trigger_network_quality_changed(
                        old_quality, self.metrics.network_quality
                    )
            
            # Perform quality adaptation if enabled
            if self.config.quality_adaptation_enabled:
                await self._adapt_quality()
            
            # Record metrics for monitoring
            if self.metrics.latency_ms:
                global_performance_monitor.record_metric(
                    "latency_ms", self.metrics.latency_ms, component="webrtc_manager"
                )
            
            if self.metrics.packet_loss_percent:
                global_performance_monitor.record_metric(
                    "packet_loss_percent", self.metrics.packet_loss_percent,
                    component="webrtc_manager"
                )
            
            if self.metrics.current_bitrate_kbps:
                global_performance_monitor.record_metric(
                    "bitrate_kbps", self.metrics.current_bitrate_kbps,
                    component="webrtc_manager"
                )
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
    
    async def _health_check_loop(self):
        """Continuously monitor connection health."""
        while self.connection_state == ConnectionState.CONNECTED:
            try:
                await asyncio.sleep(self.config.connection_health_check_interval)
                
                # Perform health check
                is_healthy = await self._perform_health_check()
                
                if not is_healthy:
                    logger.warning("Connection health check failed")
                    await self._handle_unhealthy_connection()
                else:
                    # Update stability score
                    self._update_connection_stability_score(True)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(5.0)  # Wait before retry
    
    async def _perform_health_check(self) -> bool:
        """Perform a comprehensive connection health check."""
        if not LIVEKIT_AVAILABLE:
            return True  # Mock success for development
        
        try:
            # Check if room connection is still valid
            if not self.room or self.connection_state != ConnectionState.CONNECTED:
                return False
            
            # Check recent metrics for signs of connection issues
            current_time = time.time()
            
            # Check for recent audio activity
            if (self.metrics.last_ping and 
                current_time - self.metrics.last_ping > 30.0):  # No ping in 30 seconds
                logger.debug("Health check: No recent ping activity")
                return False
            
            # Check packet loss trends
            if (self.metrics.packet_loss_percent and 
                self.metrics.packet_loss_percent > 10.0):  # >10% packet loss
                logger.debug(f"Health check: High packet loss: {self.metrics.packet_loss_percent}%")
                return False
            
            # Check latency trends
            if (self.metrics.latency_ms and 
                self.metrics.latency_ms > 1000.0):  # >1 second latency
                logger.debug(f"Health check: High latency: {self.metrics.latency_ms}ms")
                return False
            
            # In real implementation, could ping the server or check WebRTC stats
            return True
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return False
    
    async def _handle_unhealthy_connection(self):
        """Handle an unhealthy connection."""
        self._update_connection_stability_score(False)
        
        # If connection appears severely unhealthy, trigger reconnection
        if self.metrics.connection_stability_score < 0.3:
            logger.info("Connection stability below threshold, triggering reconnection")
            await self._handle_reconnection()
    
    def _update_connection_stability_score(self, is_healthy: bool):
        """Update connection stability score based on health checks."""
        # Simple exponential moving average
        alpha = 0.1  # Learning rate
        new_score = 1.0 if is_healthy else 0.0
        
        self.metrics.connection_stability_score = (
            alpha * new_score + (1 - alpha) * self.metrics.connection_stability_score
        )
    
    async def _get_connection_stats(self) -> Optional[Dict[str, Any]]:
        """Get connection statistics from LiveKit."""
        # This would be real LiveKit API calls
        # For now, return mock data
        if not LIVEKIT_AVAILABLE:
            return {
                "latency_ms": 50.0,
                "jitter_ms": 2.0,
                "packet_loss_percent": 0.1,
                "bitrate_kbps": 64.0
            }
        
        # Real implementation would call LiveKit stats API
        return None
    
    def _calculate_network_quality(self, stats: Dict[str, Any]) -> NetworkQuality:
        """Calculate network quality based on connection stats."""
        latency = stats.get("latency_ms", 0)
        packet_loss = stats.get("packet_loss_percent", 0)
        jitter = stats.get("jitter_ms", 0)
        
        # Simple quality calculation
        if latency < 50 and packet_loss < 0.5 and jitter < 5:
            return NetworkQuality.EXCELLENT
        elif latency < 100 and packet_loss < 1.0 and jitter < 10:
            return NetworkQuality.GOOD
        elif latency < 200 and packet_loss < 3.0 and jitter < 20:
            return NetworkQuality.FAIR
        else:
            return NetworkQuality.POOR
    
    @monitor_performance(component="webrtc_manager", operation="disconnect")
    async def disconnect(self):
        """Disconnect from LiveKit room."""
        try:
            await self._stop_monitoring()
            
            if self.room and LIVEKIT_AVAILABLE:
                await self.room.disconnect()
            
            self.connection_state = ConnectionState.DISCONNECTED
            
            # Clean up resources
            self.room = None
            self.audio_source = None
            self.local_audio_track = None
            
            logger.info("Disconnected from LiveKit room")
            await self._trigger_disconnected()
            
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
    
    # Quality adaptation methods
    async def _adapt_quality(self):
        """Adapt audio quality based on network conditions."""
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self._last_adaptation_time < self._adaptation_cooldown_seconds:
            return
        
        try:
            # Check if adaptation is needed
            should_adapt, new_settings = self._evaluate_quality_adaptation()
            
            if should_adapt:
                await self._apply_quality_adaptation(new_settings)
                self._last_adaptation_time = current_time
                self.metrics.quality_adaptations += 1
                
        except Exception as e:
            logger.error(f"Error in quality adaptation: {e}")
    
    def _evaluate_quality_adaptation(self) -> tuple[bool, Dict[str, Any]]:
        """Evaluate if quality adaptation is needed."""
        # Get current network conditions
        packet_loss = self.metrics.packet_loss_percent or 0
        latency = self.metrics.latency_ms or 0
        estimated_bandwidth = self.metrics.estimated_bandwidth_kbps or 1000
        
        # Current settings
        current_bitrate = self.metrics.current_bitrate_kbps or self.config.target_bitrate_kbps
        current_codec = self.metrics.current_audio_codec
        
        new_settings = {}
        should_adapt = False
        
        # Check packet loss threshold
        if packet_loss > self.config.packet_loss_threshold_percent:
            # High packet loss - reduce bitrate or switch codec
            if current_bitrate > self.config.min_bitrate_kbps:
                new_bitrate = max(
                    current_bitrate * 0.8,  # Reduce by 20%
                    self.config.min_bitrate_kbps
                )
                new_settings["bitrate_kbps"] = new_bitrate
                should_adapt = True
                logger.info(f"Reducing bitrate due to packet loss: {packet_loss}%")
        
        # Check latency threshold
        elif latency > self.config.latency_threshold_ms:
            # High latency - switch to lower latency codec
            if current_codec != CodecType.G711_PCMU:
                new_settings["codec"] = self._select_optimal_codec(CodecPreference.LOW_LATENCY)
                should_adapt = True
                logger.info(f"Switching to low-latency codec due to latency: {latency}ms")
        
        # Check bandwidth threshold
        elif estimated_bandwidth < self.config.bandwidth_threshold_kbps:
            # Low bandwidth - optimize for bandwidth efficiency
            if current_bitrate > estimated_bandwidth * 0.8:  # Use 80% of available bandwidth
                new_bitrate = max(
                    estimated_bandwidth * 0.8,
                    self.config.min_bitrate_kbps
                )
                new_settings["bitrate_kbps"] = new_bitrate
                should_adapt = True
                logger.info(f"Reducing bitrate due to bandwidth: {estimated_bandwidth} kbps")
        
        # Check for quality improvement opportunities
        else:
            # Good conditions - can we improve quality?
            if packet_loss < 0.5 and latency < 100 and estimated_bandwidth > current_bitrate * 1.5:
                if current_bitrate < self.config.max_bitrate_kbps:
                    new_bitrate = min(
                        current_bitrate * 1.2,  # Increase by 20%
                        self.config.max_bitrate_kbps,
                        estimated_bandwidth * 0.8
                    )
                    new_settings["bitrate_kbps"] = new_bitrate
                    should_adapt = True
                    logger.info(f"Increasing bitrate due to good conditions: {estimated_bandwidth} kbps available")
        
        return should_adapt, new_settings
    
    async def _apply_quality_adaptation(self, settings: Dict[str, Any]):
        """Apply quality adaptation settings."""
        if not LIVEKIT_AVAILABLE:
            logger.info(f"Mock quality adaptation applied: {settings}")
            
            # Update metrics for mock
            if "bitrate_kbps" in settings:
                self.metrics.current_bitrate_kbps = settings["bitrate_kbps"]
                self.metrics.target_bitrate_kbps = settings["bitrate_kbps"]
            
            if "codec" in settings:
                self.metrics.current_audio_codec = settings["codec"]
                self.metrics.codec_switches += 1
            
            return
        
        try:
            # Apply bitrate changes
            if "bitrate_kbps" in settings and self.local_audio_track:
                new_bitrate = int(settings["bitrate_kbps"] * 1000)  # Convert to bps
                # In real implementation, this would call LiveKit APIs
                # await self.local_audio_track.set_bitrate(new_bitrate)
                
                self.metrics.current_bitrate_kbps = settings["bitrate_kbps"]
                self.metrics.target_bitrate_kbps = settings["bitrate_kbps"]
                logger.info(f"Applied bitrate adaptation: {settings['bitrate_kbps']} kbps")
            
            # Apply codec changes
            if "codec" in settings:
                new_codec = settings["codec"]
                # In real implementation, this would require renegotiation
                # await self._renegotiate_codec(new_codec)
                
                self.metrics.current_audio_codec = new_codec
                self.metrics.codec_switches += 1
                logger.info(f"Applied codec adaptation: {new_codec.value}")
        
        except Exception as e:
            logger.error(f"Error applying quality adaptation: {e}")
    
    def _select_optimal_codec(self, preference: CodecPreference) -> CodecType:
        """Select optimal codec based on preference and network conditions."""
        preferred_codecs = CODEC_PREFERENCES.get(preference, [CodecType.OPUS])
        
        # For now, return the first preferred codec
        # In a real implementation, this would check codec availability
        return preferred_codecs[0] if preferred_codecs else CodecType.OPUS
    
    def _estimate_bandwidth(self) -> float:
        """Estimate available bandwidth based on connection statistics."""
        # Simple bandwidth estimation based on current metrics
        if self.metrics.bitrate_kbps and self.metrics.packet_loss_percent is not None:
            # Adjust based on packet loss
            base_bandwidth = self.metrics.bitrate_kbps
            loss_factor = 1.0 - (self.metrics.packet_loss_percent / 100.0)
            estimated = base_bandwidth / max(loss_factor, 0.1)  # Prevent division by zero
            
            self.metrics.estimated_bandwidth_kbps = estimated
            return estimated
        
        # Default estimate
        return 100.0  # 100 kbps default
    
    async def send_audio_frame(self, audio_data: bytes, sample_rate: int = 16000):
        """Send audio frame to the room."""
        if not self.audio_source or not LIVEKIT_AVAILABLE:
            return
        
        try:
            # Apply additional encryption if security manager is enabled
            processed_audio_data = audio_data
            if self.security_manager:
                processed_audio_data = await self.security_manager.encrypt_audio_data(audio_data)
            
            # Create audio frame
            audio_frame = AudioFrame(
                data=processed_audio_data,
                sample_rate=sample_rate,
                num_channels=1,
                samples_per_channel=len(processed_audio_data) // 2  # 16-bit audio
            )
            
            # Capture to audio source
            await self.audio_source.capture_frame(audio_frame)
            
            # Update metrics
            self.metrics.audio_packets_sent += 1
            self.metrics.audio_bytes_sent += len(processed_audio_data)
            
        except Exception as e:
            logger.error(f"Error sending audio frame: {e}")
    
    async def _handle_reconnection(self):
        """Handle automatic reconnection with advanced strategies."""
        if self._reconnect_task and not self._reconnect_task.done():
            return  # Already reconnecting
        
        # Record disconnection time
        self.metrics.last_disconnect_time = time.time()
        
        # Determine reconnection strategy
        strategy = self._determine_reconnection_strategy()
        self.metrics.current_reconnect_strategy = strategy
        
        logger.info(f"Starting reconnection with strategy: {strategy}")
        self._reconnect_task = asyncio.create_task(self._reconnect_loop(strategy))
    
    def _determine_reconnection_strategy(self) -> str:
        """Determine the best reconnection strategy based on connection history."""
        current_time = time.time()
        
        # Check if we're in fast reconnect window
        if (self.metrics.last_disconnect_time and 
            current_time - self.metrics.last_disconnect_time < self.config.fast_reconnect_window_seconds):
            return "fast"
        
        # Check connection stability score
        if self.metrics.connection_stability_score < 0.5:
            return "adaptive"  # Use different ICE servers, longer delays
        
        # Check recent reconnection history
        recent_failures = sum(1 for event in self._connection_history[-5:] 
                            if event.get("type") == "reconnect_failed")
        
        if recent_failures >= 3:
            return "robust"  # Try alternative strategies
        
        return "standard"
    
    async def _reconnect_loop(self, strategy: str):
        """Enhanced reconnection loop with different strategies."""
        if strategy == "fast":
            await self._fast_reconnect()
        elif strategy == "adaptive":
            await self._adaptive_reconnect()
        elif strategy == "robust":
            await self._robust_reconnect()
        else:
            await self._standard_reconnect()
        
        # If all strategies failed, start persistent reconnection if enabled
        if (self.connection_state == ConnectionState.FAILED and 
            self.config.persistent_reconnection):
            self._persistent_reconnect_task = asyncio.create_task(self._persistent_reconnect())
    
    async def _fast_reconnect(self):
        """Fast reconnection strategy for temporary disconnections."""
        logger.info("Using fast reconnection strategy")
        
        for attempt in range(3):  # Only 3 quick attempts
            try:
                self.connection_state = ConnectionState.RECONNECTING
                self.metrics.last_reconnect_attempt_time = time.time()
                
                # Minimal delay for fast reconnect
                if attempt > 0:
                    await asyncio.sleep(min(1.0 * attempt, 5.0))
                
                success = await self.connect()
                
                if success:
                    self.metrics.successful_reconnects += 1
                    self._record_connection_event("reconnect_success", {"strategy": "fast", "attempt": attempt + 1})
                    logger.info(f"Fast reconnection successful on attempt {attempt + 1}")
                    return
                
            except Exception as e:
                logger.error(f"Fast reconnection attempt {attempt + 1} failed: {e}")
                self.metrics.failed_reconnects += 1
        
        # Fast reconnect failed, escalate to standard
        await self._standard_reconnect()
    
    async def _standard_reconnect(self):
        """Standard reconnection strategy with exponential backoff."""
        logger.info("Using standard reconnection strategy")
        
        for attempt in range(self.config.reconnect_attempts):
            try:
                self.connection_state = ConnectionState.RECONNECTING
                self.metrics.last_reconnect_attempt_time = time.time()
                
                # Exponential backoff with jitter
                if attempt > 0:
                    delay = min(
                        self.config.reconnect_delay * (2 ** attempt) + (attempt * 0.5),  # Add jitter
                        self.config.max_reconnect_delay
                    )
                    await asyncio.sleep(delay)
                
                success = await self.connect()
                
                if success:
                    self.metrics.successful_reconnects += 1
                    self._record_connection_event("reconnect_success", {"strategy": "standard", "attempt": attempt + 1})
                    logger.info(f"Standard reconnection successful on attempt {attempt + 1}")
                    return
                
            except Exception as e:
                logger.error(f"Standard reconnection attempt {attempt + 1} failed: {e}")
                self.metrics.failed_reconnects += 1
        
        self.connection_state = ConnectionState.FAILED
        self._record_connection_event("reconnect_failed", {"strategy": "standard"})
        logger.error("Standard reconnection attempts exhausted")
    
    async def _adaptive_reconnect(self):
        """Adaptive reconnection with alternative ICE servers."""
        logger.info("Using adaptive reconnection strategy")
        
        # Try with backup ICE servers
        original_ice_servers = self.config.ice_servers.copy()
        
        for attempt in range(self.config.reconnect_attempts):
            try:
                self.connection_state = ConnectionState.RECONNECTING
                self.metrics.last_reconnect_attempt_time = time.time()
                
                # Cycle through different ICE server configurations
                if attempt > 0:
                    self._rotate_ice_servers()
                    delay = min(
                        self.config.reconnect_delay * (1.5 ** attempt),
                        self.config.max_reconnect_delay
                    )
                    await asyncio.sleep(delay)
                
                success = await self.connect()
                
                if success:
                    self.metrics.successful_reconnects += 1
                    self._record_connection_event("reconnect_success", {"strategy": "adaptive", "attempt": attempt + 1})
                    logger.info(f"Adaptive reconnection successful on attempt {attempt + 1}")
                    return
                
            except Exception as e:
                logger.error(f"Adaptive reconnection attempt {attempt + 1} failed: {e}")
                self.metrics.failed_reconnects += 1
        
        # Restore original ICE servers
        self.config.ice_servers = original_ice_servers
        
        self.connection_state = ConnectionState.FAILED
        self._record_connection_event("reconnect_failed", {"strategy": "adaptive"})
        logger.error("Adaptive reconnection attempts exhausted")
    
    async def _robust_reconnect(self):
        """Robust reconnection with multiple fallback strategies."""
        logger.info("Using robust reconnection strategy")
        
        strategies = [
            ("fast_retry", self._fast_reconnect),
            ("adaptive_ice", self._adaptive_reconnect),
            ("standard_backoff", self._standard_reconnect)
        ]
        
        for strategy_name, strategy_func in strategies:
            logger.info(f"Trying robust strategy: {strategy_name}")
            
            try:
                await strategy_func()
                
                if self.connection_state == ConnectionState.CONNECTED:
                    self._record_connection_event("reconnect_success", {"strategy": f"robust_{strategy_name}"})
                    return
                
            except Exception as e:
                logger.error(f"Robust strategy {strategy_name} failed: {e}")
            
            # Wait between strategy attempts
            await asyncio.sleep(5.0)
        
        self.connection_state = ConnectionState.FAILED
        self._record_connection_event("reconnect_failed", {"strategy": "robust"})
        logger.error("All robust reconnection strategies failed")
    
    async def _persistent_reconnect(self):
        """Persistent reconnection that continues indefinitely."""
        logger.info("Starting persistent reconnection")
        
        attempt = 0
        while self.connection_state == ConnectionState.FAILED:
            try:
                attempt += 1
                self.connection_state = ConnectionState.RECONNECTING
                self.metrics.last_reconnect_attempt_time = time.time()
                
                # Use longer delays for persistent reconnection
                delay = min(30.0 + (attempt % 5) * 10.0, 300.0)  # 30s to 5min cycle
                await asyncio.sleep(delay)
                
                # Rotate through different strategies
                if attempt % 6 == 0:
                    strategy = "robust"
                elif attempt % 3 == 0:
                    strategy = "adaptive"
                else:
                    strategy = "standard"
                
                logger.info(f"Persistent reconnection attempt {attempt} using {strategy} strategy")
                
                if strategy == "robust":
                    await self._robust_reconnect()
                elif strategy == "adaptive":
                    await self._adaptive_reconnect()
                else:
                    await self._standard_reconnect()
                
                if self.connection_state == ConnectionState.CONNECTED:
                    self.metrics.successful_reconnects += 1
                    self._record_connection_event("persistent_reconnect_success", {"attempt": attempt})
                    logger.info(f"Persistent reconnection successful after {attempt} attempts")
                    return
                
            except asyncio.CancelledError:
                logger.info("Persistent reconnection cancelled")
                break
            except Exception as e:
                logger.error(f"Persistent reconnection attempt {attempt} failed: {e}")
                self.metrics.failed_reconnects += 1
    
    def _rotate_ice_servers(self):
        """Rotate to backup ICE servers."""
        if self.config.backup_ice_servers:
            self._ice_server_index = (self._ice_server_index + 1) % len(self.config.backup_ice_servers)
            backup_server = self.config.backup_ice_servers[self._ice_server_index]
            
            # Replace one of the existing ICE servers with backup
            if self.config.ice_servers:
                self.config.ice_servers[0] = backup_server
                logger.info(f"Rotated to backup ICE server: {backup_server['urls']}")
    
    def _record_connection_event(self, event_type: str, metadata: Dict[str, Any] = None):
        """Record connection events for analysis."""
        event = {
            "type": event_type,
            "timestamp": time.time(),
            "connection_state": self.connection_state.value,
            "metadata": metadata or {}
        }
        
        self._connection_history.append(event)
        
        # Keep only recent history (last 100 events)
        if len(self._connection_history) > 100:
            self._connection_history = self._connection_history[-100:]
    
    # Event handler methods
    async def _on_room_connected(self):
        """Handle room connected event."""
        logger.info("Room connected")
        await self._trigger_connected()
    
    async def _on_room_disconnected(self):
        """Handle room disconnected event."""
        logger.info("Room disconnected")
        if self.connection_state == ConnectionState.CONNECTED:
            # Unexpected disconnection, try to reconnect
            await self._handle_reconnection()
        await self._trigger_disconnected()
    
    async def _on_participant_connected(self, participant):
        """Handle participant connected event."""
        logger.info(f"Participant connected: {participant.identity}")
        await self._trigger_participant_connected(participant)
    
    async def _on_participant_disconnected(self, participant):
        """Handle participant disconnected event."""
        logger.info(f"Participant disconnected: {participant.identity}")
        await self._trigger_participant_disconnected(participant)
    
    async def _on_track_subscribed(self, track, publication, participant):
        """Handle track subscribed event."""
        logger.info(f"Track subscribed: {track.kind} from {participant.identity}")
        
        if isinstance(track, RemoteAudioTrack):
            # Set up audio frame handler
            @track.on("frame_received")
            def on_audio_frame(frame):
                asyncio.create_task(self._on_audio_frame_received(frame, participant))
        
        await self._trigger_track_subscribed(track, publication, participant)
    
    async def _on_track_unsubscribed(self, track, publication, participant):
        """Handle track unsubscribed event."""
        logger.info(f"Track unsubscribed: {track.kind} from {participant.identity}")
        await self._trigger_track_unsubscribed(track, publication, participant)
    
    async def _on_audio_frame_received(self, frame, participant):
        """Handle received audio frame."""
        # Update metrics
        self.metrics.audio_packets_received += 1
        self.metrics.audio_bytes_received += len(frame.data)
        
        # Process DTMF detection if enabled
        if self.dtmf_detector:
            try:
                detections = await self.dtmf_detector.process_audio_frame(frame.data)
                for detection in detections:
                    await self._trigger_dtmf_detected(detection, participant)
            except Exception as e:
                logger.error(f"Error in DTMF detection: {e}")
        
        # Trigger callbacks
        await self._trigger_audio_frame(frame, participant)
    
    # Event callback triggers
    async def _trigger_connected(self):
        """Trigger connected callbacks."""
        for callback in self.on_connected_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                logger.error(f"Error in connected callback: {e}")
    
    async def _trigger_disconnected(self):
        """Trigger disconnected callbacks."""
        for callback in self.on_disconnected_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                logger.error(f"Error in disconnected callback: {e}")
    
    async def _trigger_participant_connected(self, participant):
        """Trigger participant connected callbacks."""
        for callback in self.on_participant_connected_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(participant)
                else:
                    callback(participant)
            except Exception as e:
                logger.error(f"Error in participant connected callback: {e}")
    
    async def _trigger_participant_disconnected(self, participant):
        """Trigger participant disconnected callbacks."""
        for callback in self.on_participant_disconnected_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(participant)
                else:
                    callback(participant)
            except Exception as e:
                logger.error(f"Error in participant disconnected callback: {e}")
    
    async def _trigger_track_subscribed(self, track, publication, participant):
        """Trigger track subscribed callbacks."""
        for callback in self.on_track_subscribed_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(track, publication, participant)
                else:
                    callback(track, publication, participant)
            except Exception as e:
                logger.error(f"Error in track subscribed callback: {e}")
    
    async def _trigger_track_unsubscribed(self, track, publication, participant):
        """Trigger track unsubscribed callbacks."""
        for callback in self.on_track_unsubscribed_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(track, publication, participant)
                else:
                    callback(track, publication, participant)
            except Exception as e:
                logger.error(f"Error in track unsubscribed callback: {e}")
    
    async def _trigger_audio_frame(self, frame, participant):
        """Trigger audio frame callbacks."""
        for callback in self.on_audio_frame_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(frame, participant)
                else:
                    callback(frame, participant)
            except Exception as e:
                logger.error(f"Error in audio frame callback: {e}")
    
    async def _trigger_network_quality_changed(self, old_quality, new_quality):
        """Trigger network quality changed callbacks."""
        for callback in self.on_network_quality_changed_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(old_quality, new_quality)
                else:
                    callback(old_quality, new_quality)
            except Exception as e:
                logger.error(f"Error in network quality changed callback: {e}")
    
    # Public callback registration methods
    def on_connected(self, callback: Callable):
        """Register callback for connection events."""
        self.on_connected_callbacks.append(callback)
    
    def on_disconnected(self, callback: Callable):
        """Register callback for disconnection events."""
        self.on_disconnected_callbacks.append(callback)
    
    def on_participant_connected(self, callback: Callable):
        """Register callback for participant connection events."""
        self.on_participant_connected_callbacks.append(callback)
    
    def on_participant_disconnected(self, callback: Callable):
        """Register callback for participant disconnection events."""
        self.on_participant_disconnected_callbacks.append(callback)
    
    def on_track_subscribed(self, callback: Callable):
        """Register callback for track subscription events."""
        self.on_track_subscribed_callbacks.append(callback)
    
    def on_track_unsubscribed(self, callback: Callable):
        """Register callback for track unsubscription events."""
        self.on_track_unsubscribed_callbacks.append(callback)
    
    def on_audio_frame(self, callback: Callable):
        """Register callback for audio frame events."""
        self.on_audio_frame_callbacks.append(callback)
    
    def on_network_quality_changed(self, callback: Callable):
        """Register callback for network quality change events."""
        self.on_network_quality_changed_callbacks.append(callback)
    
    def on_dtmf_detected(self, callback: Callable):
        """Register callback for DTMF tone detection events."""
        self.on_dtmf_detected_callbacks.append(callback)
    
    # DTMF-related methods
    def _setup_dtmf_callbacks(self):
        """Set up DTMF detector callbacks."""
        if self.dtmf_detector:
            self.dtmf_detector.on_tone_detected(self._on_dtmf_tone_detected)
            self.dtmf_detector.on_sequence_detected(self._on_dtmf_sequence_detected)
    
    async def _on_dtmf_tone_detected(self, detection):
        """Handle DTMF tone detection from detector."""
        logger.info(f"DTMF tone detected: {detection.character} (confidence: {detection.confidence:.2f})")
    
    async def _on_dtmf_sequence_detected(self, sequence):
        """Handle DTMF sequence detection from detector."""
        logger.info(f"DTMF sequence detected: {sequence}")
    
    async def _trigger_dtmf_detected(self, detection, participant):
        """Trigger DTMF detected callbacks."""
        for callback in self.on_dtmf_detected_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(detection, participant)
                else:
                    callback(detection, participant)
            except Exception as e:
                logger.error(f"Error in DTMF detected callback: {e}")
    
    def get_dtmf_sequence(self) -> str:
        """Get current DTMF sequence."""
        if self.dtmf_detector:
            return self.dtmf_detector.get_current_sequence()
        return ""
    
    def clear_dtmf_sequence(self):
        """Clear current DTMF sequence."""
        if self.dtmf_detector:
            self.dtmf_detector.clear_sequence()
    
    # Status and information methods
    def get_connection_state(self) -> ConnectionState:
        """Get current connection state."""
        return self.connection_state
    
    def get_metrics(self) -> ConnectionMetrics:
        """Get current connection metrics."""
        return self.metrics
    
    def is_connected(self) -> bool:
        """Check if currently connected."""
        return self.connection_state == ConnectionState.CONNECTED
    
    def get_participants(self) -> List[Any]:
        """Get list of current participants."""
        if not self.room or not LIVEKIT_AVAILABLE:
            return []
        
        return list(self.room.participants.values())
    
    def get_local_participant(self) -> Optional[Any]:
        """Get local participant."""
        if not self.room or not LIVEKIT_AVAILABLE:
            return None
        
        return self.room.local_participant
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get current security status."""
        if not self.security_manager:
            return {
                "enabled": False,
                "status": "disabled"
            }
        
        return {
            "enabled": True,
            **self.security_manager.get_security_status()
        }


# Convenience functions
async def create_webrtc_manager(
    room_url: str,
    token: str,
    participant_name: str = None,
    **kwargs
) -> WebRTCManager:
    """Create and configure a WebRTC manager."""
    config = ConnectionConfig(
        room_url=room_url,
        token=token,
        participant_name=participant_name,
        **kwargs
    )
    
    return WebRTCManager(config)


# Global WebRTC manager for easy access
_global_webrtc_manager: Optional[WebRTCManager] = None


def get_global_webrtc_manager() -> Optional[WebRTCManager]:
    """Get the global WebRTC manager instance."""
    return _global_webrtc_manager


def set_global_webrtc_manager(manager: WebRTCManager):
    """Set the global WebRTC manager instance."""
    global _global_webrtc_manager
    _global_webrtc_manager = manager