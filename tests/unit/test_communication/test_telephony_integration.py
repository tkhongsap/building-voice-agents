"""
Unit tests for Telephony Integration.

Tests SIP and Twilio integration, call management, DTMF handling,
and telephony provider abstractions.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from communication.telephony_integration import (
    TelephonyManager,
    SIPConfig,
    TwilioConfig,
    CallInfo,
    CallState,
    CallDirection,
    TelephonyProvider
)


class TestSIPConfig:
    """Test SIP configuration."""
    
    def test_default_sip_config(self):
        """Test default SIP configuration."""
        config = SIPConfig(sip_server="sip.test.com")
        
        assert config.sip_server == "sip.test.com"
        assert config.sip_port == 5060
        assert config.transport == "UDP"
        assert config.audio_codec == "PCMU"
        assert config.audio_sample_rate == 8000
        assert config.register_expires == 3600
        assert config.auto_register == True
    
    def test_custom_sip_config(self):
        """Test custom SIP configuration."""
        config = SIPConfig(
            sip_server="custom.sip.server",
            sip_port=5061,
            username="testuser",
            password="testpass",
            transport="TCP",
            audio_codec="PCMA"
        )
        
        assert config.sip_server == "custom.sip.server"
        assert config.sip_port == 5061
        assert config.username == "testuser"
        assert config.password == "testpass"
        assert config.transport == "TCP"
        assert config.audio_codec == "PCMA"


class TestTwilioConfig:
    """Test Twilio configuration."""
    
    def test_twilio_config(self):
        """Test Twilio configuration."""
        config = TwilioConfig(
            account_sid="test_sid",
            auth_token="test_token",
            phone_number="+1234567890"
        )
        
        assert config.account_sid == "test_sid"
        assert config.auth_token == "test_token"
        assert config.phone_number == "+1234567890"
        assert config.voice == "alice"
        assert config.language == "en-US"
        assert config.record_calls == False


class TestCallInfo:
    """Test call information tracking."""
    
    def test_call_info_creation(self):
        """Test call info creation."""
        call = CallInfo(
            call_id="test_call_123",
            direction=CallDirection.INBOUND,
            caller_id="+1234567890",
            callee_id="+0987654321",
            state=CallState.RINGING
        )
        
        assert call.call_id == "test_call_123"
        assert call.direction == CallDirection.INBOUND
        assert call.caller_id == "+1234567890"
        assert call.callee_id == "+0987654321"
        assert call.state == CallState.RINGING
        assert call.start_time is None
        assert call.duration == 0.0
    
    def test_call_info_with_metrics(self):
        """Test call info with performance metrics."""
        call = CallInfo(
            call_id="test_call_123",
            direction=CallDirection.OUTBOUND,
            caller_id="+1111111111",
            callee_id="+2222222222",
            state=CallState.CONNECTED,
            start_time=time.time(),
            connect_time=time.time(),
            audio_codec="PCMU",
            packet_loss=1.5,
            jitter_ms=20.0,
            latency_ms=150.0
        )
        
        assert call.audio_codec == "PCMU"
        assert call.packet_loss == 1.5
        assert call.jitter_ms == 20.0
        assert call.latency_ms == 150.0


class TestTelephonyManager:
    """Test telephony manager functionality."""
    
    @pytest.fixture
    def sip_manager(self, mock_sip_config):
        """Create SIP telephony manager for testing."""
        with patch('communication.telephony_integration.SIP_AVAILABLE', False):
            manager = TelephonyManager(TelephonyProvider.SIP)
            manager.configure_sip(mock_sip_config)
            return manager
    
    @pytest.fixture
    def twilio_manager(self, mock_twilio_config):
        """Create Twilio telephony manager for testing."""
        with patch('communication.telephony_integration.TWILIO_AVAILABLE', False):
            manager = TelephonyManager(TelephonyProvider.TWILIO)
            manager.configure_twilio(mock_twilio_config)
            return manager
    
    @pytest.fixture
    def sip_manager_with_mocks(self, mock_sip_config):
        """Create SIP manager with mocked dependencies."""
        with patch('communication.telephony_integration.SIP_AVAILABLE', True), \
             patch('communication.telephony_integration.SIPUser') as mock_sip_user_class:
            
            mock_sip_user = Mock()
            mock_sip_user.register = AsyncMock(return_value=True)
            mock_sip_user.make_call = AsyncMock()
            mock_sip_user.refresh_registration = AsyncMock()
            mock_sip_user.unregister = AsyncMock()
            mock_sip_user_class.return_value = mock_sip_user
            
            manager = TelephonyManager(TelephonyProvider.SIP)
            manager.configure_sip(mock_sip_config)
            manager.sip_user = mock_sip_user
            
            return manager
    
    def test_sip_manager_initialization(self, sip_manager):
        """Test SIP manager initialization."""
        assert sip_manager.provider == TelephonyProvider.SIP
        assert sip_manager.sip_config is not None
        assert sip_manager.is_registered == False
        assert len(sip_manager.active_calls) == 0
        assert len(sip_manager.call_history) == 0
    
    def test_twilio_manager_initialization(self, twilio_manager):
        """Test Twilio manager initialization."""
        assert twilio_manager.provider == TelephonyProvider.TWILIO
        assert twilio_manager.twilio_config is not None
        assert twilio_manager.is_registered == False
        assert len(twilio_manager.active_calls) == 0
    
    @pytest.mark.asyncio
    async def test_sip_registration_mock(self, sip_manager):
        """Test SIP registration with mock."""
        result = await sip_manager.register()
        
        # Should succeed with mock implementation
        assert result == True
        assert sip_manager.is_registered == True
        assert sip_manager.registration_time is not None
    
    @pytest.mark.asyncio
    async def test_sip_registration_with_sip_mocks(self, sip_manager_with_mocks):
        """Test SIP registration with mocked SIP library."""
        result = await sip_manager_with_mocks.register()
        
        assert result == True
        assert sip_manager_with_mocks.is_registered == True
        sip_manager_with_mocks.sip_user.register.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_twilio_registration_mock(self, twilio_manager):
        """Test Twilio registration with mock."""
        result = await twilio_manager.register()
        
        # Should succeed with mock implementation
        assert result == True
        assert twilio_manager.is_registered == True
    
    @pytest.mark.asyncio
    async def test_make_sip_call(self, sip_manager):
        """Test making SIP call."""
        destination = "sip:test@example.com"
        caller_id = "+1234567890"
        
        call_id = await sip_manager.make_call(destination, caller_id)
        
        assert call_id is not None
        assert call_id in sip_manager.active_calls
        
        call_info = sip_manager.active_calls[call_id]
        assert call_info.direction == CallDirection.OUTBOUND
        assert call_info.callee_id == destination
        assert call_info.caller_id == caller_id
        assert call_info.state == CallState.CONNECTING
    
    @pytest.mark.asyncio
    async def test_make_twilio_call(self, twilio_manager):
        """Test making Twilio call."""
        destination = "+1987654321"
        
        call_id = await twilio_manager.make_call(destination)
        
        assert call_id is not None
        assert call_id in twilio_manager.active_calls
        
        call_info = twilio_manager.active_calls[call_id]
        assert call_info.direction == CallDirection.OUTBOUND
        assert call_info.callee_id == destination
    
    @pytest.mark.asyncio
    async def test_hangup_call(self, sip_manager):
        """Test hanging up a call."""
        # First make a call
        call_id = await sip_manager.make_call("sip:test@example.com")
        assert call_id in sip_manager.active_calls
        
        # Then hang up
        result = await sip_manager.hangup_call(call_id)
        assert result == True
        
        # Call should be cleaned up (moved to history or removed)
        # Depending on implementation
    
    @pytest.mark.asyncio
    async def test_send_dtmf(self, sip_manager):
        """Test sending DTMF tones."""
        # Make a call first
        call_id = await sip_manager.make_call("sip:test@example.com")
        
        # Send DTMF
        result = await sip_manager.send_dtmf(call_id, "123*#")
        assert result == True  # Should succeed with mock
    
    @pytest.mark.asyncio
    async def test_send_dtmf_invalid_call(self, sip_manager):
        """Test sending DTMF to invalid call."""
        result = await sip_manager.send_dtmf("invalid_call_id", "123")
        assert result == False
    
    def test_call_info_retrieval(self, sip_manager):
        """Test call information retrieval."""
        # Initially no calls
        active_calls = sip_manager.get_active_calls()
        assert len(active_calls) == 0
        
        call_history = sip_manager.get_call_history()
        assert len(call_history) == 0
        
        # Test getting specific call info
        call_info = sip_manager.get_call_info("nonexistent_call")
        assert call_info is None
    
    def test_registration_status(self, sip_manager):
        """Test registration status information."""
        status = sip_manager.get_registration_status()
        
        assert "is_registered" in status
        assert "registration_time" in status
        assert "provider" in status
        assert "uptime_seconds" in status
        
        assert status["is_registered"] == False
        assert status["provider"] == TelephonyProvider.SIP.value


class TestCallManagement:
    """Test call lifecycle management."""
    
    @pytest.fixture
    def call_manager(self, mock_sip_config):
        """Create telephony manager for call testing."""
        with patch('communication.telephony_integration.SIP_AVAILABLE', False):
            manager = TelephonyManager(TelephonyProvider.SIP)
            manager.configure_sip(mock_sip_config)
            return manager
    
    @pytest.mark.asyncio
    async def test_incoming_call_handling(self, call_manager):
        """Test handling incoming call."""
        mock_sip_call = Mock()
        mock_sip_call.call_id = "incoming_call_123"
        mock_sip_call.remote_uri = "sip:caller@example.com"
        
        await call_manager._handle_incoming_call(mock_sip_call)
        
        # Should have created call info
        assert len(call_manager.active_calls) > 0
        
        # Find the call
        call_info = None
        for call in call_manager.active_calls.values():
            if call.provider_call_id == "incoming_call_123":
                call_info = call
                break
        
        assert call_info is not None
        assert call_info.direction == CallDirection.INBOUND
        assert call_info.state == CallState.RINGING
    
    @pytest.mark.asyncio
    async def test_call_connected_handling(self, call_manager):
        """Test handling call connection."""
        # First create an incoming call
        mock_sip_call = Mock()
        mock_sip_call.call_id = "test_call_123"
        mock_sip_call.remote_uri = "sip:caller@example.com"
        mock_sip_call.audio_codec = "PCMU"
        mock_sip_call.remote_ip = "192.168.1.100"
        mock_sip_call.remote_port = 5060
        
        await call_manager._handle_incoming_call(mock_sip_call)
        
        # Then handle connection
        await call_manager._handle_call_connected(mock_sip_call)
        
        # Find the call and check state
        call_info = None
        for call in call_manager.active_calls.values():
            if call.provider_call_id == "test_call_123":
                call_info = call
                break
        
        assert call_info is not None
        assert call_info.state == CallState.CONNECTED
        assert call_info.connect_time is not None
        assert call_info.audio_codec == "PCMU"
    
    @pytest.mark.asyncio
    async def test_call_disconnected_handling(self, call_manager):
        """Test handling call disconnection."""
        # Create and connect a call
        mock_sip_call = Mock()
        mock_sip_call.call_id = "test_call_123"
        mock_sip_call.remote_uri = "sip:caller@example.com"
        
        await call_manager._handle_incoming_call(mock_sip_call)
        await call_manager._handle_call_connected(mock_sip_call)
        
        # Then disconnect
        await call_manager._handle_call_disconnected(mock_sip_call)
        
        # Call should be moved to history
        assert len(call_manager.call_history) > 0
        
        # Find the call in history
        call_info = None
        for call in call_manager.call_history:
            if call.provider_call_id == "test_call_123":
                call_info = call
                break
        
        assert call_info is not None
        assert call_info.state == CallState.DISCONNECTED
        assert call_info.end_time is not None
    
    @pytest.mark.asyncio
    async def test_dtmf_received_handling(self, call_manager):
        """Test handling received DTMF tones."""
        # Create a call
        mock_sip_call = Mock()
        mock_sip_call.call_id = "test_call_123"
        mock_sip_call.remote_uri = "sip:caller@example.com"
        
        await call_manager._handle_incoming_call(mock_sip_call)
        
        # Test DTMF reception
        dtmf_received = False
        received_digit = None
        
        def dtmf_callback(call_info, digit):
            nonlocal dtmf_received, received_digit
            dtmf_received = True
            received_digit = digit
        
        call_manager.on_dtmf_received(dtmf_callback)
        
        # Simulate DTMF reception
        await call_manager._handle_dtmf_received(mock_sip_call, "1")
        
        assert dtmf_received == True
        assert received_digit == "1"


class TestEventCallbacks:
    """Test event callback system."""
    
    @pytest.fixture
    def callback_manager(self, mock_sip_config):
        """Create telephony manager for callback testing."""
        with patch('communication.telephony_integration.SIP_AVAILABLE', False):
            manager = TelephonyManager(TelephonyProvider.SIP)
            manager.configure_sip(mock_sip_config)
            return manager
    
    def test_callback_registration(self, callback_manager):
        """Test event callback registration."""
        def test_callback(call_info):
            pass
        
        # Test all callback types
        callback_manager.on_incoming_call(test_callback)
        callback_manager.on_call_connected(test_callback)
        callback_manager.on_call_disconnected(test_callback)
        callback_manager.on_call_failed(test_callback)
        
        assert len(callback_manager.on_incoming_call_callbacks) == 1
        assert len(callback_manager.on_call_connected_callbacks) == 1
        assert len(callback_manager.on_call_disconnected_callbacks) == 1
        assert len(callback_manager.on_call_failed_callbacks) == 1
    
    @pytest.mark.asyncio
    async def test_incoming_call_callback(self, callback_manager):
        """Test incoming call callback execution."""
        callback_called = False
        received_call_info = None
        
        def incoming_callback(call_info):
            nonlocal callback_called, received_call_info
            callback_called = True
            received_call_info = call_info
        
        callback_manager.on_incoming_call(incoming_callback)
        
        # Create mock call info
        mock_call = CallInfo(
            call_id="test_call",
            direction=CallDirection.INBOUND,
            caller_id="+1234567890",
            callee_id="+0987654321",
            state=CallState.RINGING
        )
        
        await callback_manager._trigger_incoming_call(mock_call)
        
        assert callback_called == True
        assert received_call_info == mock_call
    
    @pytest.mark.asyncio
    async def test_async_callback_support(self, callback_manager):
        """Test async callback support."""
        callback_called = False
        
        async def async_callback(call_info):
            nonlocal callback_called
            await asyncio.sleep(0.01)  # Simulate async work
            callback_called = True
        
        callback_manager.on_call_connected(async_callback)
        
        mock_call = CallInfo(
            call_id="test_call",
            direction=CallDirection.OUTBOUND,
            caller_id="+1111111111",
            callee_id="+2222222222",
            state=CallState.CONNECTED
        )
        
        await callback_manager._trigger_call_connected(mock_call)
        
        assert callback_called == True


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.fixture
    def error_manager(self, mock_sip_config):
        """Create telephony manager for error testing."""
        with patch('communication.telephony_integration.SIP_AVAILABLE', False):
            manager = TelephonyManager(TelephonyProvider.SIP)
            manager.configure_sip(mock_sip_config)
            return manager
    
    @pytest.mark.asyncio
    async def test_registration_failure_handling(self, error_manager):
        """Test registration failure handling."""
        # Mock registration failure
        with patch.object(error_manager, '_register_sip', side_effect=Exception("Registration failed")):
            result = await error_manager.register()
            
            assert result == False
            assert error_manager.is_registered == False
    
    @pytest.mark.asyncio
    async def test_call_failure_handling(self, error_manager):
        """Test call failure handling."""
        # Mock call failure
        with patch.object(error_manager, '_make_sip_call', side_effect=Exception("Call failed")):
            call_id = await error_manager.make_call("sip:test@example.com")
            
            assert call_id is None
    
    @pytest.mark.asyncio
    async def test_hangup_invalid_call(self, error_manager):
        """Test hanging up invalid call."""
        result = await error_manager.hangup_call("nonexistent_call")
        assert result == False
    
    def test_invalid_provider_handling(self):
        """Test handling of invalid telephony provider."""
        with patch('communication.telephony_integration.SIP_AVAILABLE', False):
            manager = TelephonyManager(TelephonyProvider.SIP)
            
            # Should initialize without crashing
            assert manager.provider == TelephonyProvider.SIP
    
    @pytest.mark.asyncio
    async def test_callback_error_handling(self, error_manager):
        """Test error handling in callbacks."""
        def failing_callback(call_info):
            raise Exception("Callback failed")
        
        error_manager.on_incoming_call(failing_callback)
        
        mock_call = CallInfo(
            call_id="test_call",
            direction=CallDirection.INBOUND,
            caller_id="+1234567890",
            callee_id="+0987654321",
            state=CallState.RINGING
        )
        
        # Should not crash despite callback failure
        await error_manager._trigger_incoming_call(mock_call)
        
        # Test continues if no exception raised


class TestCleanup:
    """Test cleanup and resource management."""
    
    @pytest.fixture
    def cleanup_manager(self, mock_sip_config):
        """Create telephony manager for cleanup testing."""
        with patch('communication.telephony_integration.SIP_AVAILABLE', False):
            manager = TelephonyManager(TelephonyProvider.SIP)
            manager.configure_sip(mock_sip_config)
            return manager
    
    @pytest.mark.asyncio
    async def test_cleanup_with_active_calls(self, cleanup_manager):
        """Test cleanup with active calls."""
        # Create some active calls
        call_id1 = await cleanup_manager.make_call("sip:test1@example.com")
        call_id2 = await cleanup_manager.make_call("sip:test2@example.com")
        
        assert len(cleanup_manager.active_calls) == 2
        
        # Cleanup should handle active calls
        await cleanup_manager.cleanup()
        
        # Should have attempted to clean up
        assert cleanup_manager.is_registered == False
    
    @pytest.mark.asyncio
    async def test_cleanup_background_tasks(self, cleanup_manager):
        """Test cleanup of background tasks."""
        # Simulate background task
        cleanup_manager._keepalive_task = Mock()
        cleanup_manager._keepalive_task.cancel = Mock()
        
        await cleanup_manager.cleanup()
        
        cleanup_manager._keepalive_task.cancel.assert_called_once()


# Integration test markers
pytestmark = pytest.mark.unit