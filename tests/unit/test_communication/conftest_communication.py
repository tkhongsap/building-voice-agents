"""
Test fixtures specific to Task 2.0 Communication components.

This module provides mock objects, test data, and fixtures for testing
WebRTC, telephony, security, and networking components.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from typing import Dict, Any, List, Optional
import numpy as np

# ============================================================================
# WEBRTC MOCK FIXTURES
# ============================================================================

@pytest.fixture
def mock_webrtc_config():
    """Create a mock WebRTC configuration."""
    config = Mock()
    config.server_url = "wss://test.livekit.cloud"
    config.api_key = "test_api_key"
    config.api_secret = "test_api_secret"
    config.room_name = "test_room"
    config.participant_name = "test_participant"
    config.enable_audio = True
    config.enable_video = False
    config.auto_subscribe = True
    config.reconnect_attempts = 3
    config.reconnect_delay = 2.0
    config.connection_timeout = 30.0
    config.enable_encryption = True
    config.security_level = Mock(value="enhanced")
    return config


@pytest.fixture
def mock_livekit_room():
    """Create a mock LiveKit room."""
    room = Mock()
    room.participants = {}
    room.local_participant = Mock()
    room.local_participant.identity = "test_participant"
    room.local_participant.tracks = {}
    room.connect = AsyncMock(return_value=True)
    room.disconnect = AsyncMock()
    room.publish_track = AsyncMock()
    room.unpublish_track = AsyncMock()
    room.get_stats = AsyncMock(return_value={})
    return room


@pytest.fixture
def mock_audio_frame():
    """Create a mock audio frame."""
    frame = Mock()
    frame.data = b'\x00\x01' * 480  # 480 samples
    frame.sample_rate = 16000
    frame.channels = 1
    frame.timestamp = time.time()
    return frame


# ============================================================================
# TELEPHONY MOCK FIXTURES
# ============================================================================

@pytest.fixture
def mock_sip_config():
    """Create a mock SIP configuration."""
    config = Mock()
    config.sip_server = "sip.test.com"
    config.sip_port = 5060
    config.username = "test_user"
    config.password = "test_pass"
    config.display_name = "Test User"
    config.transport = "UDP"
    config.audio_codec = "PCMU"
    config.audio_sample_rate = 8000
    config.register_expires = 3600
    config.auto_register = True
    return config


@pytest.fixture
def mock_twilio_config():
    """Create a mock Twilio configuration."""
    config = Mock()
    config.account_sid = "test_account_sid"
    config.auth_token = "test_auth_token"
    config.phone_number = "+1234567890"
    config.webhook_url = "https://test.example.com/webhook"
    config.voice = "alice"
    config.language = "en-US"
    config.record_calls = False
    return config


@pytest.fixture
def mock_sip_call():
    """Create a mock SIP call."""
    call = Mock()
    call.call_id = "test_call_123"
    call.remote_uri = "sip:caller@test.com"
    call.state = "connected"
    call.audio_codec = "PCMU"
    call.remote_ip = "192.168.1.100"
    call.remote_port = 5060
    call.send_dtmf = AsyncMock()
    call.hangup = AsyncMock()
    return call


# ============================================================================
# DTMF MOCK FIXTURES  
# ============================================================================

@pytest.fixture
def mock_dtmf_config():
    """Create a mock DTMF configuration."""
    config = Mock()
    config.sample_rate = 8000
    config.frame_size = 160
    config.min_tone_duration_ms = 40
    config.max_tone_duration_ms = 200
    config.inter_digit_delay_ms = 40
    config.frequency_tolerance_hz = 20.0
    config.amplitude_threshold = 0.1
    config.snr_threshold_db = 6.0
    config.use_goertzel_filter = True
    config.enable_twist_detection = True
    config.max_twist_db = 6.0
    return config


@pytest.fixture
def mock_dtmf_audio_data():
    """Create mock audio data containing DTMF tones."""
    # Generate sine wave for DTMF tone '1' (697 Hz + 1209 Hz)
    if not hasattr(mock_dtmf_audio_data, '_cache'):
        sample_rate = 8000
        duration = 0.1  # 100ms
        samples = int(sample_rate * duration)
        t = np.linspace(0, duration, samples, False)
        
        # DTMF '1' frequencies
        tone_697 = np.sin(2 * np.pi * 697 * t)
        tone_1209 = np.sin(2 * np.pi * 1209 * t)
        combined = (tone_697 + tone_1209) / 2
        
        # Convert to 16-bit PCM
        audio_int16 = (combined * 32767).astype(np.int16)
        mock_dtmf_audio_data._cache = audio_int16.tobytes()
    
    return mock_dtmf_audio_data._cache


# ============================================================================
# SECURITY MOCK FIXTURES
# ============================================================================

@pytest.fixture
def mock_security_config():
    """Create a mock security configuration."""
    config = Mock()
    config.security_level = Mock(value="enhanced")
    config.certificate_type = Mock(value="self_signed")
    config.certificate_validity_days = 365
    config.key_size = 2048
    config.preferred_encryption = Mock(value="aes_256_gcm")
    config.require_secure_transport = True
    config.enforce_srtp = True
    config.dtls_timeout_ms = 5000
    config.verify_peer_certificate = True
    config.key_rotation_interval_hours = 24
    return config


@pytest.fixture
def mock_certificate():
    """Create a mock certificate."""
    cert = Mock()
    cert.fingerprint = Mock(return_value=b'mock_fingerprint')
    cert.public_bytes = Mock(return_value=b'mock_cert_pem')
    cert.not_valid_after = Mock()
    return cert


# ============================================================================
# PLATFORM COMPATIBILITY FIXTURES
# ============================================================================

@pytest.fixture
def mock_platform_info():
    """Create mock platform information."""
    info = Mock()
    info.platform_type = Mock(value="desktop")
    info.os_name = "Linux"
    info.os_version = "5.4.0"
    info.python_version = "3.9.0"
    info.has_audio_access = True
    info.has_video_access = True
    info.has_microphone = True
    info.has_camera = True
    info.has_speakers = True
    info.has_gui = True
    info.supports_webrtc = True
    info.supports_websockets = True
    info.supports_tcp = True
    return info


@pytest.fixture
def mock_audio_devices():
    """Create mock audio device list."""
    return [
        {
            "device_id": "0",
            "label": "Default Microphone",
            "kind": "audioinput",
            "group_id": "default",
            "sample_rate": 44100,
            "channels": 1
        },
        {
            "device_id": "1", 
            "label": "Default Speakers",
            "kind": "audiooutput",
            "group_id": "default",
            "sample_rate": 44100,
            "channels": 2
        }
    ]


# ============================================================================
# QUALITY MONITORING FIXTURES
# ============================================================================

@pytest.fixture
def mock_quality_metrics():
    """Create mock quality metrics."""
    metrics = Mock()
    metrics.bitrate_bps = 128000.0
    metrics.packet_loss_percent = 1.5
    metrics.jitter_ms = 20.0
    metrics.latency_ms = 150.0
    metrics.audio_level = 0.75
    metrics.quality_level = Mock(value="good")
    metrics.mos_score = 4.2
    metrics.timestamp = time.time()
    return metrics


@pytest.fixture
def mock_bitrate_config():
    """Create mock bitrate configuration."""
    config = Mock()
    config.min_audio_bitrate = 6000
    config.max_audio_bitrate = 128000
    config.default_audio_bitrate = 32000
    config.increase_factor = 1.2
    config.decrease_factor = 0.8
    config.packet_loss_threshold = 2.0
    config.adaptation_interval_ms = 1000
    return config


# ============================================================================
# STATISTICS FIXTURES
# ============================================================================

@pytest.fixture
def mock_rtc_stats():
    """Create mock WebRTC statistics."""
    return {
        "inbound-rtp": {
            "type": "inbound-rtp",
            "ssrc": 123456789,
            "mediaType": "audio",
            "packetsReceived": 1000,
            "bytesReceived": 128000,
            "packetsLost": 10,
            "jitter": 0.02,
            "roundTripTime": 0.15
        },
        "outbound-rtp": {
            "type": "outbound-rtp", 
            "ssrc": 987654321,
            "mediaType": "audio",
            "packetsSent": 950,
            "bytesSent": 121600,
            "roundTripTime": 0.15
        },
        "candidate-pair": {
            "type": "candidate-pair",
            "currentRoundTripTime": 0.15,
            "localCandidateType": "host",
            "remoteCandidateType": "host"
        }
    }


# ============================================================================
# NETWORK FIXTURES
# ============================================================================

@pytest.fixture
def mock_region_info():
    """Create mock region information."""
    region = Mock()
    region.region_id = "us-east-1"
    region.region_name = "US East"
    region.endpoint_url = "wss://us-east-1.livekit.cloud"
    region.geographic_location = "North America"
    region.average_latency_ms = 50.0
    region.current_connections = 10
    region.max_connections = 1000
    region.status = Mock(value="available")
    region.health_score = 95.0
    region.weight = 1.0
    region.priority = 1
    return region


@pytest.fixture
def mock_connection_request():
    """Create mock connection request."""
    request = Mock()
    request.client_id = "test_client_123"
    request.client_ip = "192.168.1.100"
    request.client_location = "us"
    request.required_features = ["audio", "dtmf"]
    request.performance_requirements = {"max_latency": 200}
    return request


@pytest.fixture
def mock_diagnostic_result():
    """Create mock diagnostic test result."""
    result = Mock()
    result.test_type = Mock(value="connectivity")
    result.success = True
    result.duration_ms = 150.0
    result.result_data = {"successful_connections": 3, "total_tests": 3}
    result.error_message = None
    result.timestamp = time.time()
    return result


# ============================================================================
# CONNECTION POOL FIXTURES
# ============================================================================

@pytest.fixture
def mock_pool_config():
    """Create mock connection pool configuration."""
    config = Mock()
    config.min_connections = 2
    config.max_connections = 10
    config.initial_connections = 3
    config.max_idle_time_seconds = 300
    config.max_lifetime_seconds = 3600
    config.strategy = Mock(value="fifo")
    config.enable_preemptive_creation = True
    config.enable_resource_monitoring = True
    config.cleanup_interval_seconds = 60
    return config


@pytest.fixture
def mock_connection_factory():
    """Create mock connection factory."""
    async def factory():
        conn = Mock()
        conn.connect = AsyncMock(return_value=True)
        conn.close = AsyncMock()
        conn.send = AsyncMock()
        conn.receive = AsyncMock()
        return conn
    
    return factory


# ============================================================================
# HELPER FIXTURES
# ============================================================================

@pytest.fixture
def mock_performance_monitor():
    """Create mock performance monitor for communication tests."""
    monitor = Mock()
    monitor.register_component = Mock(return_value=Mock(
        monitor_operation=Mock(return_value=Mock(
            __aenter__=AsyncMock(),
            __aexit__=AsyncMock()
        ))
    ))
    monitor.record_metric = Mock()
    return monitor


@pytest.fixture
def mock_async_context_manager():
    """Create a mock async context manager."""
    class MockAsyncContextManager:
        async def __aenter__(self):
            return self
        
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
    
    return MockAsyncContextManager()


@pytest.fixture
def communication_test_data():
    """Provide common test data for communication tests."""
    return {
        "audio_data": b'\x00\x01' * 1600,  # 1600 samples
        "text_message": "Test message for communication",
        "phone_number": "+1234567890",
        "sip_uri": "sip:test@example.com",
        "dtmf_sequence": "1234*#",
        "test_ip": "192.168.1.100",
        "test_domain": "test.example.com"
    }


# ============================================================================
# MOCK EXTERNAL DEPENDENCIES
# ============================================================================

@pytest.fixture
def mock_livekit_imports():
    """Mock LiveKit imports."""
    with patch.dict('sys.modules', {
        'livekit': Mock(),
        'livekit.Room': Mock(),
        'livekit.RoomOptions': Mock(),
        'livekit.ConnectOptions': Mock(),
    }):
        yield


@pytest.fixture  
def mock_telephony_imports():
    """Mock telephony library imports."""
    with patch.dict('sys.modules', {
        'sip': Mock(),
        'twilio.rest': Mock(),
        'twilio.twiml': Mock(),
    }):
        yield


@pytest.fixture
def mock_crypto_imports():
    """Mock cryptography imports."""
    with patch.dict('sys.modules', {
        'cryptography': Mock(),
        'cryptography.hazmat.primitives': Mock(),
        'cryptography.x509': Mock(),
    }):
        yield


@pytest.fixture
def mock_network_imports():
    """Mock network library imports."""
    with patch.dict('sys.modules', {
        'requests': Mock(),
        'ping3': Mock(),
        'speedtest': Mock(),
        'psutil': Mock(),
    }):
        yield