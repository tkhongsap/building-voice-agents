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
    
    # Video settings
    video_enabled: bool = False
    video_quality: str = "high"
    
    # Timeouts and retry
    connection_timeout: float = 30.0
    reconnect_attempts: int = 5
    reconnect_delay: float = 2.0


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
        
        # Audio processing
        self.audio_source: Optional[AudioSource] = None
        self.local_audio_track: Optional[LocalAudioTrack] = None
        
        # Monitoring
        self.monitor = global_performance_monitor.register_component(
            "webrtc_manager", "communication"
        )
        
        # Connection management
        self._reconnect_task: Optional[asyncio.Task] = None
        self._metrics_task: Optional[asyncio.Task] = None
        self._is_monitoring = False
    
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
            
            # Set up audio if enabled
            if self.config.audio_enabled:
                await self._setup_audio()
            
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
    
    async def _start_monitoring(self):
        """Start connection and metrics monitoring."""
        if self._is_monitoring:
            return
        
        self._is_monitoring = True
        self._metrics_task = asyncio.create_task(self._metrics_loop())
    
    async def _stop_monitoring(self):
        """Stop monitoring tasks."""
        self._is_monitoring = False
        
        if self._metrics_task:
            self._metrics_task.cancel()
            try:
                await self._metrics_task
            except asyncio.CancelledError:
                pass
        
        if self._reconnect_task:
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
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
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
    
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
    
    async def send_audio_frame(self, audio_data: bytes, sample_rate: int = 16000):
        """Send audio frame to the room."""
        if not self.audio_source or not LIVEKIT_AVAILABLE:
            return
        
        try:
            # Create audio frame
            audio_frame = AudioFrame(
                data=audio_data,
                sample_rate=sample_rate,
                num_channels=1,
                samples_per_channel=len(audio_data) // 2  # 16-bit audio
            )
            
            # Capture to audio source
            await self.audio_source.capture_frame(audio_frame)
            
            # Update metrics
            self.metrics.audio_packets_sent += 1
            self.metrics.audio_bytes_sent += len(audio_data)
            
        except Exception as e:
            logger.error(f"Error sending audio frame: {e}")
    
    async def _handle_reconnection(self):
        """Handle automatic reconnection."""
        if self._reconnect_task and not self._reconnect_task.done():
            return  # Already reconnecting
        
        self._reconnect_task = asyncio.create_task(self._reconnect_loop())
    
    async def _reconnect_loop(self):
        """Reconnection loop with exponential backoff."""
        for attempt in range(self.config.reconnect_attempts):
            try:
                self.connection_state = ConnectionState.RECONNECTING
                logger.info(f"Reconnection attempt {attempt + 1}/{self.config.reconnect_attempts}")
                
                # Wait before reconnecting
                await asyncio.sleep(self.config.reconnect_delay * (2 ** attempt))
                
                # Attempt to reconnect
                success = await self.connect()
                
                if success:
                    self.metrics.reconnect_count += 1
                    logger.info("Reconnection successful")
                    return
                
            except Exception as e:
                logger.error(f"Reconnection attempt {attempt + 1} failed: {e}")
        
        # All reconnection attempts failed
        self.connection_state = ConnectionState.FAILED
        logger.error("All reconnection attempts failed")
    
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