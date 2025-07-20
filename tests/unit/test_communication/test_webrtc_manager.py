"""
Unit tests for WebRTC Manager.

Tests WebRTC connection management, quality adaptation, reconnection logic,
DTMF detection integration, and security features.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from communication.webrtc_manager import (
    WebRTCManager, 
    WebRTCConfig,
    ConnectionState,
    ConnectionMetrics,
    QualityPreset,
    ReconnectionStrategy
)


class TestWebRTCConfig:
    """Test WebRTC configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = WebRTCConfig()
        
        assert config.server_url == ""
        assert config.api_key == ""
        assert config.api_secret == ""
        assert config.room_name == ""
        assert config.participant_name == ""
        assert config.enable_audio == True
        assert config.enable_video == False
        assert config.auto_subscribe == True
        assert config.reconnect_attempts == 5
        assert config.reconnect_delay == 2.0
        assert config.max_reconnect_delay == 30.0
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = WebRTCConfig(
            server_url="wss://test.livekit.cloud",
            api_key="test_key",
            room_name="test_room",
            enable_video=True,
            reconnect_attempts=3
        )
        
        assert config.server_url == "wss://test.livekit.cloud"
        assert config.api_key == "test_key" 
        assert config.room_name == "test_room"
        assert config.enable_video == True
        assert config.reconnect_attempts == 3


class TestWebRTCManager:
    """Test WebRTC Manager functionality."""
    
    @pytest.fixture
    def webrtc_manager(self, mock_webrtc_config):
        """Create WebRTC manager for testing."""
        with patch('communication.webrtc_manager.LIVEKIT_AVAILABLE', False):
            manager = WebRTCManager(mock_webrtc_config)
            return manager
    
    @pytest.fixture
    def webrtc_manager_with_mocks(self, mock_webrtc_config):
        """Create WebRTC manager with mocked dependencies."""
        with patch('communication.webrtc_manager.LIVEKIT_AVAILABLE', True), \
             patch('communication.webrtc_manager.Room') as mock_room_class, \
             patch('communication.webrtc_manager.DTMF_AVAILABLE', True), \
             patch('communication.webrtc_manager.SECURITY_AVAILABLE', True):
            
            mock_room = Mock()
            mock_room.connect = AsyncMock(return_value=True)
            mock_room.disconnect = AsyncMock()
            mock_room.participants = {}
            mock_room.local_participant = Mock()
            mock_room_class.return_value = mock_room
            
            manager = WebRTCManager(mock_webrtc_config)
            manager.room = mock_room
            
            return manager
    
    def test_initialization(self, webrtc_manager):
        """Test WebRTC manager initialization."""
        assert webrtc_manager.connection_state == ConnectionState.DISCONNECTED
        assert webrtc_manager.room is None
        assert webrtc_manager.metrics is not None
        assert webrtc_manager.current_quality_preset == QualityPreset.BALANCED
        assert webrtc_manager._reconnect_task is None
    
    @pytest.mark.asyncio
    async def test_connect_success_mock(self, webrtc_manager):
        """Test successful connection with mocked LiveKit."""
        with patch('communication.webrtc_manager.LIVEKIT_AVAILABLE', False):
            result = await webrtc_manager.connect()
            
            # Should succeed with mock implementation
            assert result == True
            assert webrtc_manager.connection_state == ConnectionState.CONNECTED
            assert webrtc_manager.metrics.connect_time is not None
    
    @pytest.mark.asyncio
    async def test_connect_with_livekit_mock(self, webrtc_manager_with_mocks):
        """Test connection with mocked LiveKit."""
        result = await webrtc_manager_with_mocks.connect()
        
        assert result == True
        assert webrtc_manager_with_mocks.connection_state == ConnectionState.CONNECTED
        assert webrtc_manager_with_mocks.room is not None
    
    @pytest.mark.asyncio
    async def test_disconnect(self, webrtc_manager_with_mocks):
        """Test disconnection."""
        # First connect
        await webrtc_manager_with_mocks.connect()
        assert webrtc_manager_with_mocks.connection_state == ConnectionState.CONNECTED
        
        # Then disconnect
        await webrtc_manager_with_mocks.disconnect()
        assert webrtc_manager_with_mocks.connection_state == ConnectionState.DISCONNECTED
        assert webrtc_manager_with_mocks.metrics.disconnect_time is not None
    
    @pytest.mark.asyncio 
    async def test_quality_adaptation(self, webrtc_manager):
        """Test quality adaptation functionality."""
        # Test setting quality preset
        webrtc_manager.set_quality_preset(QualityPreset.HIGH_QUALITY)
        assert webrtc_manager.current_quality_preset == QualityPreset.HIGH_QUALITY
        
        # Test adaptive quality change
        await webrtc_manager._adapt_quality()
        # Should not crash and should update metrics
        assert webrtc_manager.metrics is not None
    
    @pytest.mark.asyncio
    async def test_reconnection_logic(self, webrtc_manager):
        """Test reconnection logic."""
        webrtc_manager.connection_state = ConnectionState.DISCONNECTED
        webrtc_manager.metrics.failed_connections = 2
        
        # Test strategy determination
        strategy = webrtc_manager._determine_reconnection_strategy()
        assert strategy in ["fast", "standard", "adaptive", "robust"]
        
        # Test fast reconnect
        with patch.object(webrtc_manager, 'connect', new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = True
            await webrtc_manager._fast_reconnect()
            mock_connect.assert_called()
    
    @pytest.mark.asyncio
    async def test_audio_frame_processing(self, webrtc_manager, mock_audio_frame):
        """Test audio frame processing."""
        # Mock DTMF detector
        with patch.object(webrtc_manager, 'dtmf_detector') as mock_dtmf:
            mock_dtmf.process_audio_frame = AsyncMock(return_value=[])
            
            await webrtc_manager._process_audio_frame(mock_audio_frame, Mock())
            
            # Should have processed the frame
            mock_dtmf.process_audio_frame.assert_called_once()
    
    def test_metrics_tracking(self, webrtc_manager):
        """Test metrics tracking."""
        metrics = webrtc_manager.get_metrics()
        
        assert isinstance(metrics, ConnectionMetrics)
        assert hasattr(metrics, 'connection_attempts')
        assert hasattr(metrics, 'successful_connections')
        assert hasattr(metrics, 'failed_connections')
        assert hasattr(metrics, 'total_reconnects')
    
    def test_callback_registration(self, webrtc_manager):
        """Test event callback registration."""
        callback_called = False
        
        def test_callback():
            nonlocal callback_called
            callback_called = True
        
        webrtc_manager.on_connected(test_callback)
        assert len(webrtc_manager.on_connected_callbacks) == 1
        
        # Test async callback
        async def async_callback():
            nonlocal callback_called  
            callback_called = True
        
        webrtc_manager.on_disconnected(async_callback)
        assert len(webrtc_manager.on_disconnected_callbacks) == 1
    
    @pytest.mark.asyncio
    async def test_security_integration(self, webrtc_manager):
        """Test security manager integration."""
        # Mock security manager
        mock_security = Mock()
        mock_security.initialize = AsyncMock(return_value=True)
        mock_security.establish_secure_connection = AsyncMock(return_value=True)
        
        webrtc_manager.security_manager = mock_security
        
        # Test security setup
        result = await webrtc_manager._setup_security()
        assert result == True
    
    @pytest.mark.asyncio
    async def test_health_check(self, webrtc_manager):
        """Test connection health check."""
        # Mock connected state
        webrtc_manager.connection_state = ConnectionState.CONNECTED
        webrtc_manager.room = Mock()
        
        # Test health check
        is_healthy = await webrtc_manager._perform_health_check()
        assert isinstance(is_healthy, bool)
    
    @pytest.mark.asyncio
    async def test_cleanup(self, webrtc_manager_with_mocks):
        """Test cleanup functionality."""
        # Connect first
        await webrtc_manager_with_mocks.connect()
        
        # Start some background tasks
        webrtc_manager_with_mocks._is_monitoring = True
        
        # Test cleanup
        await webrtc_manager_with_mocks.cleanup()
        
        assert webrtc_manager_with_mocks._is_monitoring == False
        webrtc_manager_with_mocks.room.disconnect.assert_called()
    
    def test_connection_state_management(self, webrtc_manager):
        """Test connection state management."""
        # Test initial state
        assert webrtc_manager.get_connection_state() == ConnectionState.DISCONNECTED
        assert webrtc_manager.is_connected() == False
        
        # Test state change
        webrtc_manager.connection_state = ConnectionState.CONNECTED
        assert webrtc_manager.is_connected() == True


class TestQualityAdaptation:
    """Test quality adaptation functionality."""
    
    @pytest.fixture
    def quality_manager(self, mock_webrtc_config):
        """Create WebRTC manager for quality testing."""
        with patch('communication.webrtc_manager.LIVEKIT_AVAILABLE', False):
            manager = WebRTCManager(mock_webrtc_config)
            return manager
    
    def test_quality_presets(self, quality_manager):
        """Test different quality presets."""
        presets = [
            QualityPreset.HIGH_QUALITY,
            QualityPreset.BALANCED, 
            QualityPreset.LOW_BANDWIDTH,
            QualityPreset.LOW_LATENCY
        ]
        
        for preset in presets:
            quality_manager.set_quality_preset(preset)
            assert quality_manager.current_quality_preset == preset
    
    @pytest.mark.asyncio
    async def test_bitrate_adjustment(self, quality_manager):
        """Test bitrate adjustment logic."""
        # Mock current bitrate
        quality_manager.target_bitrate = 64000
        
        # Test increase
        await quality_manager._increase_bitrate()
        assert quality_manager.target_bitrate > 64000
        
        # Test decrease  
        quality_manager.target_bitrate = 64000
        await quality_manager._decrease_bitrate()
        assert quality_manager.target_bitrate < 64000
    
    @pytest.mark.asyncio
    async def test_network_condition_adaptation(self, quality_manager):
        """Test adaptation to network conditions."""
        # Simulate poor network conditions
        quality_manager.metrics.packet_loss_percent = 5.0
        quality_manager.metrics.latency_ms = 250.0
        
        await quality_manager._adapt_quality()
        
        # Should have adapted to conditions
        assert quality_manager.metrics is not None


class TestDTMFIntegration:
    """Test DTMF integration with WebRTC."""
    
    @pytest.fixture
    def webrtc_with_dtmf(self, mock_webrtc_config):
        """Create WebRTC manager with DTMF support."""
        with patch('communication.webrtc_manager.DTMF_AVAILABLE', True):
            manager = WebRTCManager(mock_webrtc_config)
            
            # Mock DTMF detector
            mock_detector = Mock()
            mock_detector.process_audio_frame = AsyncMock(return_value=[])
            mock_detector.get_current_sequence = Mock(return_value="123")
            mock_detector.clear_sequence = Mock()
            manager.dtmf_detector = mock_detector
            
            return manager
    
    def test_dtmf_setup(self, webrtc_with_dtmf):
        """Test DTMF detector setup."""
        assert webrtc_with_dtmf.dtmf_detector is not None
    
    def test_dtmf_sequence_access(self, webrtc_with_dtmf):
        """Test DTMF sequence access methods."""
        sequence = webrtc_with_dtmf.get_dtmf_sequence()
        assert sequence == "123"
        
        webrtc_with_dtmf.clear_dtmf_sequence()
        webrtc_with_dtmf.dtmf_detector.clear_sequence.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_dtmf_callback_registration(self, webrtc_with_dtmf):
        """Test DTMF event callback registration."""
        callback_called = False
        detected_tone = None
        
        def dtmf_callback(detection, participant):
            nonlocal callback_called, detected_tone
            callback_called = True
            detected_tone = detection
        
        webrtc_with_dtmf.on_dtmf_detected(dtmf_callback)
        
        # Simulate DTMF detection
        mock_detection = Mock()
        mock_detection.character = "1"
        mock_participant = Mock()
        
        await webrtc_with_dtmf._trigger_dtmf_detected(mock_detection, mock_participant)
        
        assert callback_called == True
        assert detected_tone == mock_detection


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.fixture
    def error_manager(self, mock_webrtc_config):
        """Create WebRTC manager for error testing."""
        with patch('communication.webrtc_manager.LIVEKIT_AVAILABLE', False):
            manager = WebRTCManager(mock_webrtc_config)
            return manager
    
    @pytest.mark.asyncio
    async def test_connection_failure_handling(self, error_manager):
        """Test handling of connection failures.""" 
        # Mock connection failure
        with patch.object(error_manager, '_connect_to_room', side_effect=Exception("Connection failed")):
            result = await error_manager.connect()
            
            assert result == False
            assert error_manager.connection_state == ConnectionState.FAILED
            assert error_manager.metrics.failed_connections > 0
    
    @pytest.mark.asyncio
    async def test_reconnection_failure_handling(self, error_manager):
        """Test handling of reconnection failures."""
        error_manager.connection_state = ConnectionState.DISCONNECTED
        
        # Mock reconnection failure
        with patch.object(error_manager, 'connect', side_effect=Exception("Reconnect failed")):
            await error_manager._standard_reconnect()
            
            # Should have attempted reconnection
            assert error_manager.metrics.failed_reconnects > 0
    
    @pytest.mark.asyncio
    async def test_audio_processing_error_handling(self, error_manager):
        """Test audio processing error handling."""
        mock_frame = Mock()
        mock_frame.data = b"invalid_audio_data"
        mock_participant = Mock()
        
        # Should not crash on invalid audio data
        await error_manager._process_audio_frame(mock_frame, mock_participant)
        
        # Should continue processing despite errors
        assert True  # Test passes if no exception
    
    def test_invalid_configuration_handling(self):
        """Test handling of invalid configuration."""
        # Test with empty configuration
        config = WebRTCConfig()
        
        with patch('communication.webrtc_manager.LIVEKIT_AVAILABLE', False):
            manager = WebRTCManager(config)
            
            # Should initialize despite empty config
            assert manager is not None
            assert manager.connection_state == ConnectionState.DISCONNECTED


class TestPerformanceMetrics:
    """Test performance monitoring and metrics."""
    
    @pytest.fixture
    def metrics_manager(self, mock_webrtc_config):
        """Create WebRTC manager for metrics testing."""
        with patch('communication.webrtc_manager.LIVEKIT_AVAILABLE', False):
            manager = WebRTCManager(mock_webrtc_config)
            return manager
    
    def test_metrics_initialization(self, metrics_manager):
        """Test metrics initialization."""
        metrics = metrics_manager.metrics
        
        assert metrics.connection_attempts == 0
        assert metrics.successful_connections == 0
        assert metrics.failed_connections == 0
        assert metrics.total_reconnects == 0
        assert metrics.connect_time is None
        assert metrics.disconnect_time is None
    
    @pytest.mark.asyncio
    async def test_metrics_tracking_during_connection(self, metrics_manager):
        """Test metrics tracking during connection lifecycle."""
        initial_attempts = metrics_manager.metrics.connection_attempts
        
        # Attempt connection
        await metrics_manager.connect()
        
        # Should have tracked the attempt
        assert metrics_manager.metrics.connection_attempts > initial_attempts
        assert metrics_manager.metrics.successful_connections > 0
    
    def test_connection_quality_metrics(self, metrics_manager):
        """Test connection quality metrics."""
        # Simulate quality data
        metrics_manager.metrics.packet_loss_percent = 2.5
        metrics_manager.metrics.latency_ms = 150.0
        metrics_manager.metrics.jitter_ms = 25.0
        metrics_manager.metrics.bitrate_bps = 128000.0
        
        metrics = metrics_manager.get_metrics()
        
        assert metrics.packet_loss_percent == 2.5
        assert metrics.latency_ms == 150.0
        assert metrics.jitter_ms == 25.0
        assert metrics.bitrate_bps == 128000.0
    
    def test_performance_monitoring_integration(self, metrics_manager):
        """Test integration with performance monitoring system."""
        # Should have monitor component registered
        assert metrics_manager.monitor is not None
        
        # Test metric recording
        metrics_manager.monitor.record_metric("test_metric", 123.45)
        # Should not crash


# Integration test markers
pytestmark = pytest.mark.unit