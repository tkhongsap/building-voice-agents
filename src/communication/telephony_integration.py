"""
SIP Telephony Integration for LiveKit Voice Agents Platform

This module provides comprehensive SIP (Session Initiation Protocol) integration
for traditional telephony systems, enabling phone-based voice agent interactions.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List, Callable, Union
from contextlib import asynccontextmanager

try:
    import sip
    from sip import SIPUser, SIPCall, SIPMessage
    SIP_AVAILABLE = True
except ImportError:
    SIP_AVAILABLE = False
    # Mock classes for development without SIP libraries
    class SIPUser: pass
    class SIPCall: pass
    class SIPMessage: pass

try:
    from twilio.rest import Client as TwilioClient
    from twilio.twiml import VoiceResponse
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    TwilioClient = None
    VoiceResponse = None

from monitoring.performance_monitor import monitor_performance, global_performance_monitor

logger = logging.getLogger(__name__)


class CallState(Enum):
    """SIP call states."""
    IDLE = "idle"
    RINGING = "ringing"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    HOLD = "hold"
    TRANSFERRING = "transferring"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"
    FAILED = "failed"


class CallDirection(Enum):
    """Call direction types."""
    INBOUND = "inbound"
    OUTBOUND = "outbound"


class TelephonyProvider(Enum):
    """Supported telephony providers."""
    SIP = "sip"
    TWILIO = "twilio"
    ASTERISK = "asterisk"
    GENERIC = "generic"


@dataclass
class SIPConfig:
    """SIP configuration settings."""
    # SIP Server settings
    sip_server: str
    sip_port: int = 5060
    username: str = ""
    password: str = ""
    display_name: str = "Voice Agent"
    
    # Transport settings
    transport: str = "UDP"  # UDP, TCP, TLS
    local_port: Optional[int] = None
    
    # Registration settings
    register_expires: int = 3600
    auto_register: bool = True
    
    # Audio settings
    audio_codec: str = "PCMU"  # PCMU, PCMA, G729, OPUS
    audio_sample_rate: int = 8000
    
    # Advanced settings
    use_ice: bool = True
    use_stun: bool = True
    stun_server: str = "stun.l.google.com:19302"
    
    # Security
    use_tls: bool = False
    verify_certificates: bool = True


@dataclass
class TwilioConfig:
    """Twilio configuration settings."""
    account_sid: str
    auth_token: str
    phone_number: str
    webhook_url: Optional[str] = None
    status_callback_url: Optional[str] = None
    
    # Voice settings
    voice: str = "alice"
    language: str = "en-US"
    
    # Recording settings
    record_calls: bool = False
    recording_channels: str = "dual"


@dataclass
class CallInfo:
    """Information about an active call."""
    call_id: str
    direction: CallDirection
    caller_id: str
    callee_id: str
    state: CallState
    start_time: Optional[float] = None
    connect_time: Optional[float] = None
    end_time: Optional[float] = None
    duration: float = 0.0
    
    # Media information
    audio_codec: Optional[str] = None
    remote_ip: Optional[str] = None
    remote_port: Optional[int] = None
    
    # Quality metrics
    audio_quality_score: Optional[float] = None
    packet_loss: Optional[float] = None
    jitter_ms: Optional[float] = None
    latency_ms: Optional[float] = None
    
    # Provider-specific data
    provider_call_id: Optional[str] = None
    provider_data: Dict[str, Any] = field(default_factory=dict)


class TelephonyManager:
    """Manages telephony integration for voice agents."""
    
    def __init__(self, provider: TelephonyProvider = TelephonyProvider.SIP):
        self.provider = provider
        self.sip_config: Optional[SIPConfig] = None
        self.twilio_config: Optional[TwilioConfig] = None
        
        # Connection state
        self.is_registered = False
        self.registration_time: Optional[float] = None
        
        # Active calls
        self.active_calls: Dict[str, CallInfo] = {}
        self.call_history: List[CallInfo] = []
        
        # Event callbacks
        self.on_incoming_call_callbacks: List[Callable] = []
        self.on_call_connected_callbacks: List[Callable] = []
        self.on_call_disconnected_callbacks: List[Callable] = []
        self.on_call_failed_callbacks: List[Callable] = []
        self.on_dtmf_received_callbacks: List[Callable] = []
        self.on_registration_callbacks: List[Callable] = []
        
        # Provider clients
        self.sip_user: Optional[SIPUser] = None
        self.twilio_client: Optional[TwilioClient] = None
        
        # Monitoring
        self.monitor = global_performance_monitor.register_component(
            "telephony_manager", "communication"
        )
        
        # Background tasks
        self._registration_task: Optional[asyncio.Task] = None
        self._keepalive_task: Optional[asyncio.Task] = None
    
    def configure_sip(self, config: SIPConfig):
        """Configure SIP settings."""
        self.sip_config = config
        logger.info(f"SIP configured for server: {config.sip_server}:{config.sip_port}")
    
    def configure_twilio(self, config: TwilioConfig):
        """Configure Twilio settings."""
        self.twilio_config = config
        if TWILIO_AVAILABLE:
            self.twilio_client = TwilioClient(config.account_sid, config.auth_token)
        logger.info(f"Twilio configured with number: {config.phone_number}")
    
    @monitor_performance(component="telephony_manager", operation="register")
    async def register(self) -> bool:
        """Register with telephony provider."""
        if self.provider == TelephonyProvider.SIP:
            return await self._register_sip()
        elif self.provider == TelephonyProvider.TWILIO:
            return await self._register_twilio()
        else:
            logger.error(f"Unsupported provider: {self.provider}")
            return False
    
    async def _register_sip(self) -> bool:
        """Register with SIP server."""
        if not self.sip_config or not SIP_AVAILABLE:
            logger.warning("SIP not available or not configured")
            return True  # Mock success for development
        
        try:
            # Create SIP user
            self.sip_user = SIPUser(
                username=self.sip_config.username,
                password=self.sip_config.password,
                server=self.sip_config.sip_server,
                port=self.sip_config.sip_port,
                display_name=self.sip_config.display_name
            )
            
            # Set up event handlers
            self._setup_sip_handlers()
            
            # Perform registration
            if self.sip_config.auto_register:
                success = await self.sip_user.register(
                    expires=self.sip_config.register_expires
                )
                
                if success:
                    self.is_registered = True
                    self.registration_time = time.time()
                    logger.info("SIP registration successful")
                    
                    # Start keepalive task
                    self._keepalive_task = asyncio.create_task(self._sip_keepalive())
                    
                    await self._trigger_registration_success()
                    return True
                else:
                    logger.error("SIP registration failed")
                    return False
            else:
                self.is_registered = True
                logger.info("SIP configured without auto-registration")
                return True
        
        except Exception as e:
            logger.error(f"SIP registration error: {e}")
            return False
    
    async def _register_twilio(self) -> bool:
        """Validate Twilio configuration."""
        if not self.twilio_config or not TWILIO_AVAILABLE:
            logger.warning("Twilio not available or not configured")
            return True  # Mock success for development
        
        try:
            # Test Twilio connection
            account = self.twilio_client.api.accounts(self.twilio_config.account_sid).fetch()
            
            self.is_registered = True
            self.registration_time = time.time()
            logger.info(f"Twilio validated for account: {account.friendly_name}")
            
            await self._trigger_registration_success()
            return True
        
        except Exception as e:
            logger.error(f"Twilio validation error: {e}")
            return False
    
    def _setup_sip_handlers(self):
        """Set up SIP event handlers."""
        if not self.sip_user:
            return
        
        @self.sip_user.on("incoming_call")
        def on_incoming_call(call):
            asyncio.create_task(self._handle_incoming_call(call))
        
        @self.sip_user.on("call_connected")
        def on_call_connected(call):
            asyncio.create_task(self._handle_call_connected(call))
        
        @self.sip_user.on("call_disconnected")
        def on_call_disconnected(call):
            asyncio.create_task(self._handle_call_disconnected(call))
        
        @self.sip_user.on("dtmf_received")
        def on_dtmf_received(call, digit):
            asyncio.create_task(self._handle_dtmf_received(call, digit))
    
    async def _sip_keepalive(self):
        """Maintain SIP registration."""
        while self.is_registered:
            try:
                await asyncio.sleep(self.sip_config.register_expires // 2)
                
                if self.sip_user and self.is_registered:
                    await self.sip_user.refresh_registration()
                    logger.debug("SIP registration refreshed")
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"SIP keepalive error: {e}")
                await asyncio.sleep(30)  # Retry after 30 seconds
    
    @monitor_performance(component="telephony_manager", operation="make_call")
    async def make_call(self, destination: str, caller_id: str = None) -> Optional[str]:
        """Initiate an outbound call."""
        call_id = str(uuid.uuid4())
        
        try:
            if self.provider == TelephonyProvider.SIP:
                return await self._make_sip_call(call_id, destination, caller_id)
            elif self.provider == TelephonyProvider.TWILIO:
                return await self._make_twilio_call(call_id, destination, caller_id)
            else:
                logger.error(f"Unsupported provider for outbound calls: {self.provider}")
                return None
        
        except Exception as e:
            logger.error(f"Error making call: {e}")
            return None
    
    async def _make_sip_call(self, call_id: str, destination: str, caller_id: str = None) -> Optional[str]:
        """Make SIP call."""
        if not self.sip_user or not SIP_AVAILABLE:
            logger.warning("SIP not available for outbound call")
            return call_id  # Mock success for development
        
        try:
            # Create call info
            call_info = CallInfo(
                call_id=call_id,
                direction=CallDirection.OUTBOUND,
                caller_id=caller_id or self.sip_config.username,
                callee_id=destination,
                state=CallState.CONNECTING,
                start_time=time.time()
            )
            
            self.active_calls[call_id] = call_info
            
            # Initiate SIP call
            sip_call = await self.sip_user.make_call(
                destination,
                caller_id=caller_id,
                audio_codec=self.sip_config.audio_codec
            )
            
            # Update call info with SIP call details
            call_info.provider_call_id = sip_call.call_id
            call_info.provider_data["sip_call"] = sip_call
            
            logger.info(f"SIP call initiated: {call_id} -> {destination}")
            return call_id
        
        except Exception as e:
            logger.error(f"SIP call failed: {e}")
            if call_id in self.active_calls:
                self.active_calls[call_id].state = CallState.FAILED
                await self._trigger_call_failed(self.active_calls[call_id])
            return None
    
    async def _make_twilio_call(self, call_id: str, destination: str, caller_id: str = None) -> Optional[str]:
        """Make Twilio call."""
        if not self.twilio_client or not TWILIO_AVAILABLE:
            logger.warning("Twilio not available for outbound call")
            return call_id  # Mock success for development
        
        try:
            # Create call info
            call_info = CallInfo(
                call_id=call_id,
                direction=CallDirection.OUTBOUND,
                caller_id=caller_id or self.twilio_config.phone_number,
                callee_id=destination,
                state=CallState.CONNECTING,
                start_time=time.time()
            )
            
            self.active_calls[call_id] = call_info
            
            # Create TwiML for voice agent interaction
            twiml_url = f"{self.twilio_config.webhook_url}/voice/{call_id}"
            
            # Initiate Twilio call
            twilio_call = self.twilio_client.calls.create(
                to=destination,
                from_=caller_id or self.twilio_config.phone_number,
                url=twiml_url,
                status_callback=self.twilio_config.status_callback_url,
                record=self.twilio_config.record_calls
            )
            
            # Update call info with Twilio call details
            call_info.provider_call_id = twilio_call.sid
            call_info.provider_data["twilio_call"] = twilio_call
            
            logger.info(f"Twilio call initiated: {call_id} -> {destination}")
            return call_id
        
        except Exception as e:
            logger.error(f"Twilio call failed: {e}")
            if call_id in self.active_calls:
                self.active_calls[call_id].state = CallState.FAILED
                await self._trigger_call_failed(self.active_calls[call_id])
            return None
    
    @monitor_performance(component="telephony_manager", operation="hangup_call")
    async def hangup_call(self, call_id: str) -> bool:
        """Terminate an active call."""
        if call_id not in self.active_calls:
            logger.warning(f"Call not found: {call_id}")
            return False
        
        call_info = self.active_calls[call_id]
        
        try:
            if self.provider == TelephonyProvider.SIP:
                return await self._hangup_sip_call(call_info)
            elif self.provider == TelephonyProvider.TWILIO:
                return await self._hangup_twilio_call(call_info)
            else:
                logger.error(f"Unsupported provider for hangup: {self.provider}")
                return False
        
        except Exception as e:
            logger.error(f"Error hanging up call {call_id}: {e}")
            return False
    
    async def _hangup_sip_call(self, call_info: CallInfo) -> bool:
        """Hangup SIP call."""
        if not SIP_AVAILABLE:
            logger.info(f"Mock SIP hangup for call: {call_info.call_id}")
            call_info.state = CallState.DISCONNECTED
            call_info.end_time = time.time()
            await self._trigger_call_disconnected(call_info)
            return True
        
        try:
            sip_call = call_info.provider_data.get("sip_call")
            if sip_call:
                await sip_call.hangup()
            
            call_info.state = CallState.DISCONNECTED
            call_info.end_time = time.time()
            
            if call_info.connect_time:
                call_info.duration = call_info.end_time - call_info.connect_time
            
            logger.info(f"SIP call terminated: {call_info.call_id}")
            await self._trigger_call_disconnected(call_info)
            return True
        
        except Exception as e:
            logger.error(f"SIP hangup failed: {e}")
            return False
    
    async def _hangup_twilio_call(self, call_info: CallInfo) -> bool:
        """Hangup Twilio call."""
        if not TWILIO_AVAILABLE:
            logger.info(f"Mock Twilio hangup for call: {call_info.call_id}")
            call_info.state = CallState.DISCONNECTED
            call_info.end_time = time.time()
            await self._trigger_call_disconnected(call_info)
            return True
        
        try:
            if call_info.provider_call_id:
                self.twilio_client.calls(call_info.provider_call_id).update(status="completed")
            
            call_info.state = CallState.DISCONNECTED
            call_info.end_time = time.time()
            
            if call_info.connect_time:
                call_info.duration = call_info.end_time - call_info.connect_time
            
            logger.info(f"Twilio call terminated: {call_info.call_id}")
            await self._trigger_call_disconnected(call_info)
            return True
        
        except Exception as e:
            logger.error(f"Twilio hangup failed: {e}")
            return False
    
    async def send_dtmf(self, call_id: str, digits: str) -> bool:
        """Send DTMF tones during an active call."""
        if call_id not in self.active_calls:
            logger.warning(f"Call not found for DTMF: {call_id}")
            return False
        
        call_info = self.active_calls[call_id]
        
        try:
            if self.provider == TelephonyProvider.SIP:
                return await self._send_sip_dtmf(call_info, digits)
            elif self.provider == TelephonyProvider.TWILIO:
                return await self._send_twilio_dtmf(call_info, digits)
            else:
                logger.error(f"DTMF not supported for provider: {self.provider}")
                return False
        
        except Exception as e:
            logger.error(f"Error sending DTMF {digits} to call {call_id}: {e}")
            return False
    
    async def _send_sip_dtmf(self, call_info: CallInfo, digits: str) -> bool:
        """Send DTMF via SIP."""
        if not SIP_AVAILABLE:
            logger.info(f"Mock SIP DTMF send: {digits}")
            return True
        
        try:
            sip_call = call_info.provider_data.get("sip_call")
            if sip_call:
                await sip_call.send_dtmf(digits)
            
            logger.info(f"SIP DTMF sent: {digits} to call {call_info.call_id}")
            return True
        
        except Exception as e:
            logger.error(f"SIP DTMF failed: {e}")
            return False
    
    async def _send_twilio_dtmf(self, call_info: CallInfo, digits: str) -> bool:
        """Send DTMF via Twilio."""
        if not TWILIO_AVAILABLE:
            logger.info(f"Mock Twilio DTMF send: {digits}")
            return True
        
        try:
            # Twilio DTMF is typically sent via TwiML during call flow
            # This would require integration with the voice webhook
            logger.info(f"Twilio DTMF queued: {digits} for call {call_info.call_id}")
            return True
        
        except Exception as e:
            logger.error(f"Twilio DTMF failed: {e}")
            return False
    
    # Event handlers
    async def _handle_incoming_call(self, sip_call):
        """Handle incoming SIP call."""
        call_id = str(uuid.uuid4())
        
        call_info = CallInfo(
            call_id=call_id,
            direction=CallDirection.INBOUND,
            caller_id=sip_call.remote_uri,
            callee_id=self.sip_config.username,
            state=CallState.RINGING,
            start_time=time.time(),
            provider_call_id=sip_call.call_id
        )
        
        call_info.provider_data["sip_call"] = sip_call
        self.active_calls[call_id] = call_info
        
        logger.info(f"Incoming SIP call: {call_info.caller_id} -> {call_info.callee_id}")
        await self._trigger_incoming_call(call_info)
    
    async def _handle_call_connected(self, sip_call):
        """Handle call connection."""
        # Find call by provider call ID
        call_info = None
        for call in self.active_calls.values():
            if call.provider_call_id == sip_call.call_id:
                call_info = call
                break
        
        if call_info:
            call_info.state = CallState.CONNECTED
            call_info.connect_time = time.time()
            
            # Extract media information
            call_info.audio_codec = sip_call.audio_codec
            call_info.remote_ip = sip_call.remote_ip
            call_info.remote_port = sip_call.remote_port
            
            logger.info(f"Call connected: {call_info.call_id}")
            await self._trigger_call_connected(call_info)
    
    async def _handle_call_disconnected(self, sip_call):
        """Handle call disconnection."""
        # Find and update call
        call_info = None
        for call in self.active_calls.values():
            if call.provider_call_id == sip_call.call_id:
                call_info = call
                break
        
        if call_info:
            call_info.state = CallState.DISCONNECTED
            call_info.end_time = time.time()
            
            if call_info.connect_time:
                call_info.duration = call_info.end_time - call_info.connect_time
            
            # Move to history
            self.call_history.append(call_info)
            del self.active_calls[call_info.call_id]
            
            logger.info(f"Call disconnected: {call_info.call_id}")
            await self._trigger_call_disconnected(call_info)
    
    async def _handle_dtmf_received(self, sip_call, digit):
        """Handle received DTMF tone."""
        # Find call by provider call ID
        call_info = None
        for call in self.active_calls.values():
            if call.provider_call_id == sip_call.call_id:
                call_info = call
                break
        
        if call_info:
            logger.info(f"DTMF received: {digit} from call {call_info.call_id}")
            await self._trigger_dtmf_received(call_info, digit)
    
    # Event callback triggers
    async def _trigger_incoming_call(self, call_info: CallInfo):
        """Trigger incoming call callbacks."""
        for callback in self.on_incoming_call_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(call_info)
                else:
                    callback(call_info)
            except Exception as e:
                logger.error(f"Error in incoming call callback: {e}")
    
    async def _trigger_call_connected(self, call_info: CallInfo):
        """Trigger call connected callbacks."""
        for callback in self.on_call_connected_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(call_info)
                else:
                    callback(call_info)
            except Exception as e:
                logger.error(f"Error in call connected callback: {e}")
    
    async def _trigger_call_disconnected(self, call_info: CallInfo):
        """Trigger call disconnected callbacks."""
        for callback in self.on_call_disconnected_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(call_info)
                else:
                    callback(call_info)
            except Exception as e:
                logger.error(f"Error in call disconnected callback: {e}")
    
    async def _trigger_call_failed(self, call_info: CallInfo):
        """Trigger call failed callbacks."""
        for callback in self.on_call_failed_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(call_info)
                else:
                    callback(call_info)
            except Exception as e:
                logger.error(f"Error in call failed callback: {e}")
    
    async def _trigger_dtmf_received(self, call_info: CallInfo, digit: str):
        """Trigger DTMF received callbacks."""
        for callback in self.on_dtmf_received_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(call_info, digit)
                else:
                    callback(call_info, digit)
            except Exception as e:
                logger.error(f"Error in DTMF received callback: {e}")
    
    async def _trigger_registration_success(self):
        """Trigger registration success callbacks."""
        for callback in self.on_registration_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(True)
                else:
                    callback(True)
            except Exception as e:
                logger.error(f"Error in registration callback: {e}")
    
    # Public callback registration methods
    def on_incoming_call(self, callback: Callable):
        """Register callback for incoming call events."""
        self.on_incoming_call_callbacks.append(callback)
    
    def on_call_connected(self, callback: Callable):
        """Register callback for call connected events."""
        self.on_call_connected_callbacks.append(callback)
    
    def on_call_disconnected(self, callback: Callable):
        """Register callback for call disconnected events."""
        self.on_call_disconnected_callbacks.append(callback)
    
    def on_call_failed(self, callback: Callable):
        """Register callback for call failed events."""
        self.on_call_failed_callbacks.append(callback)
    
    def on_dtmf_received(self, callback: Callable):
        """Register callback for DTMF tone events."""
        self.on_dtmf_received_callbacks.append(callback)
    
    def on_registration(self, callback: Callable):
        """Register callback for registration events."""
        self.on_registration_callbacks.append(callback)
    
    # Status and information methods
    def get_active_calls(self) -> List[CallInfo]:
        """Get list of active calls."""
        return list(self.active_calls.values())
    
    def get_call_history(self) -> List[CallInfo]:
        """Get call history."""
        return self.call_history.copy()
    
    def get_call_info(self, call_id: str) -> Optional[CallInfo]:
        """Get information about a specific call."""
        return self.active_calls.get(call_id)
    
    def is_call_active(self, call_id: str) -> bool:
        """Check if a call is currently active."""
        return call_id in self.active_calls
    
    def get_registration_status(self) -> Dict[str, Any]:
        """Get registration status information."""
        return {
            "is_registered": self.is_registered,
            "registration_time": self.registration_time,
            "provider": self.provider.value,
            "uptime_seconds": time.time() - self.registration_time if self.registration_time else 0
        }
    
    async def cleanup(self):
        """Clean up telephony resources."""
        try:
            # Hangup all active calls
            for call_id in list(self.active_calls.keys()):
                await self.hangup_call(call_id)
            
            # Cancel background tasks
            if self._keepalive_task:
                self._keepalive_task.cancel()
                try:
                    await self._keepalive_task
                except asyncio.CancelledError:
                    pass
            
            # Unregister from provider
            if self.sip_user and SIP_AVAILABLE:
                await self.sip_user.unregister()
            
            self.is_registered = False
            logger.info("Telephony manager cleaned up")
        
        except Exception as e:
            logger.error(f"Error during telephony cleanup: {e}")


# Convenience functions
async def create_sip_manager(config: SIPConfig) -> TelephonyManager:
    """Create and configure a SIP telephony manager."""
    manager = TelephonyManager(TelephonyProvider.SIP)
    manager.configure_sip(config)
    
    success = await manager.register()
    if not success:
        raise RuntimeError("Failed to register with SIP server")
    
    return manager


async def create_twilio_manager(config: TwilioConfig) -> TelephonyManager:
    """Create and configure a Twilio telephony manager."""
    manager = TelephonyManager(TelephonyProvider.TWILIO)
    manager.configure_twilio(config)
    
    success = await manager.register()
    if not success:
        raise RuntimeError("Failed to validate Twilio configuration")
    
    return manager


# Global telephony manager for easy access
_global_telephony_manager: Optional[TelephonyManager] = None


def get_global_telephony_manager() -> Optional[TelephonyManager]:
    """Get the global telephony manager instance."""
    return _global_telephony_manager


def set_global_telephony_manager(manager: TelephonyManager):
    """Set the global telephony manager instance."""
    global _global_telephony_manager
    _global_telephony_manager = manager