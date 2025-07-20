"""
Cross-Platform Compatibility Layer for WebRTC Voice Agents

This module provides platform-specific adapters and compatibility layers
for running voice agents across web, mobile, and desktop environments.
"""

import asyncio
import logging
import platform
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, List, Callable, Union
import json

# Platform detection
try:
    # Check if running in web environment (Pyodide)
    import js
    import pyodide
    WEB_AVAILABLE = True
except ImportError:
    WEB_AVAILABLE = False
    js = None
    pyodide = None

try:
    # Mobile-specific imports (Kivy for mobile development)
    from kivy.app import App
    from kivy.utils import platform as kivy_platform
    MOBILE_AVAILABLE = True
except ImportError:
    MOBILE_AVAILABLE = False
    App = None
    kivy_platform = None

try:
    # Desktop-specific imports
    import tkinter as tk
    from tkinter import messagebox
    DESKTOP_UI_AVAILABLE = True
except ImportError:
    DESKTOP_UI_AVAILABLE = False
    tk = None
    messagebox = None

try:
    # Audio system access
    import sounddevice as sd
    import numpy as np
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    sd = None
    np = None

from monitoring.performance_monitor import monitor_performance, global_performance_monitor

logger = logging.getLogger(__name__)


class PlatformType(Enum):
    """Supported platform types."""
    WEB = "web"
    MOBILE = "mobile" 
    DESKTOP = "desktop"
    SERVER = "server"
    UNKNOWN = "unknown"


class MobileOS(Enum):
    """Mobile operating systems."""
    ANDROID = "android"
    IOS = "ios"
    UNKNOWN = "unknown"


class DesktopOS(Enum):
    """Desktop operating systems."""
    WINDOWS = "windows"
    MACOS = "macos"
    LINUX = "linux"
    UNKNOWN = "unknown"


@dataclass
class PlatformInfo:
    """Platform detection and capability information."""
    platform_type: PlatformType
    os_name: str
    os_version: str
    python_version: str
    
    # Capabilities
    has_audio_access: bool = False
    has_video_access: bool = False
    has_microphone: bool = False
    has_camera: bool = False
    has_speakers: bool = False
    
    # UI capabilities
    has_gui: bool = False
    has_notifications: bool = False
    has_file_system: bool = False
    
    # Network capabilities
    supports_webrtc: bool = False
    supports_websockets: bool = False
    supports_tcp: bool = False
    
    # Performance characteristics
    cpu_cores: Optional[int] = None
    memory_mb: Optional[int] = None
    is_low_power: bool = False


class PlatformAdapter(ABC):
    """Abstract base class for platform-specific adapters."""
    
    def __init__(self):
        self.platform_info: Optional[PlatformInfo] = None
        self.monitor = global_performance_monitor.register_component(
            "platform_adapter", "communication"
        )
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize platform-specific components."""
        pass
    
    @abstractmethod
    async def get_audio_devices(self) -> List[Dict[str, Any]]:
        """Get available audio input/output devices."""
        pass
    
    @abstractmethod
    async def request_microphone_permission(self) -> bool:
        """Request microphone access permission."""
        pass
    
    @abstractmethod
    async def request_camera_permission(self) -> bool:
        """Request camera access permission."""
        pass
    
    @abstractmethod
    async def show_notification(self, title: str, message: str) -> bool:
        """Show platform-specific notification."""
        pass
    
    @abstractmethod
    async def get_system_info(self) -> Dict[str, Any]:
        """Get platform-specific system information."""
        pass
    
    @abstractmethod
    async def cleanup(self):
        """Clean up platform-specific resources."""
        pass


class WebAdapter(PlatformAdapter):
    """Platform adapter for web environments (browser/Pyodide)."""
    
    def __init__(self):
        super().__init__()
        self.media_stream = None
        self.audio_context = None
        
    async def initialize(self) -> bool:
        """Initialize web platform components."""
        if not WEB_AVAILABLE:
            logger.warning("Web platform not available")
            return False
        
        try:
            # Create platform info
            self.platform_info = PlatformInfo(
                platform_type=PlatformType.WEB,
                os_name="Browser",
                os_version=str(js.navigator.userAgent),
                python_version=sys.version,
                has_audio_access=True,
                has_video_access=True,
                has_microphone=await self._check_microphone_available(),
                has_camera=await self._check_camera_available(),
                has_speakers=True,
                has_gui=True,
                has_notifications=True,
                has_file_system=False,  # Limited in browser
                supports_webrtc=True,
                supports_websockets=True,
                supports_tcp=False  # Not directly available in browser
            )
            
            logger.info("Web platform adapter initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize web adapter: {e}")
            return False
    
    async def _check_microphone_available(self) -> bool:
        """Check if microphone is available in browser."""
        try:
            # Check if getUserMedia is available
            if hasattr(js.navigator, 'mediaDevices'):
                return True
            return False
        except Exception:
            return False
    
    async def _check_camera_available(self) -> bool:
        """Check if camera is available in browser."""
        try:
            # Similar to microphone check
            if hasattr(js.navigator, 'mediaDevices'):
                return True
            return False
        except Exception:
            return False
    
    async def get_audio_devices(self) -> List[Dict[str, Any]]:
        """Get available audio devices via Web API."""
        if not WEB_AVAILABLE:
            return []
        
        try:
            # In a real implementation, this would call:
            # devices = await js.navigator.mediaDevices.enumerateDevices()
            # For now, return mock devices
            return [
                {
                    "device_id": "default",
                    "label": "Default Microphone",
                    "kind": "audioinput",
                    "group_id": "default_group"
                },
                {
                    "device_id": "default_output",
                    "label": "Default Speakers",
                    "kind": "audiooutput", 
                    "group_id": "default_group"
                }
            ]
        
        except Exception as e:
            logger.error(f"Error getting audio devices: {e}")
            return []
    
    async def request_microphone_permission(self) -> bool:
        """Request microphone permission via browser API."""
        if not WEB_AVAILABLE:
            return False
        
        try:
            # In a real implementation:
            # stream = await js.navigator.mediaDevices.getUserMedia({"audio": True})
            # self.media_stream = stream
            logger.info("Mock microphone permission granted")
            return True
        
        except Exception as e:
            logger.error(f"Microphone permission denied: {e}")
            return False
    
    async def request_camera_permission(self) -> bool:
        """Request camera permission via browser API."""
        if not WEB_AVAILABLE:
            return False
        
        try:
            # In a real implementation:
            # stream = await js.navigator.mediaDevices.getUserMedia({"video": True})
            logger.info("Mock camera permission granted")
            return True
        
        except Exception as e:
            logger.error(f"Camera permission denied: {e}")
            return False
    
    async def show_notification(self, title: str, message: str) -> bool:
        """Show browser notification."""
        if not WEB_AVAILABLE:
            return False
        
        try:
            # In a real implementation:
            # notification = js.Notification.new(title, {"body": message})
            logger.info(f"Mock notification: {title} - {message}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to show notification: {e}")
            return False
    
    async def get_system_info(self) -> Dict[str, Any]:
        """Get browser/system information."""
        if not WEB_AVAILABLE:
            return {}
        
        try:
            return {
                "user_agent": str(js.navigator.userAgent),
                "language": str(js.navigator.language),
                "online": bool(js.navigator.onLine),
                "platform": str(js.navigator.platform),
                "memory": getattr(js.navigator, 'deviceMemory', None),
                "connection": {
                    "effective_type": getattr(js.navigator.connection, 'effectiveType', None) if hasattr(js.navigator, 'connection') else None,
                    "downlink": getattr(js.navigator.connection, 'downlink', None) if hasattr(js.navigator, 'connection') else None
                }
            }
        
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            return {}
    
    async def cleanup(self):
        """Clean up web resources."""
        try:
            if self.media_stream:
                # In real implementation: self.media_stream.getTracks().forEach(track => track.stop())
                self.media_stream = None
            
            if self.audio_context:
                # In real implementation: await self.audio_context.close()
                self.audio_context = None
            
            logger.info("Web adapter cleaned up")
        
        except Exception as e:
            logger.error(f"Error during web cleanup: {e}")


class MobileAdapter(PlatformAdapter):
    """Platform adapter for mobile environments (Android/iOS)."""
    
    def __init__(self):
        super().__init__()
        self.mobile_os = MobileOS.UNKNOWN
        
    async def initialize(self) -> bool:
        """Initialize mobile platform components."""
        if not MOBILE_AVAILABLE:
            logger.warning("Mobile platform not available")
            return False
        
        try:
            # Detect mobile OS
            if kivy_platform == 'android':
                self.mobile_os = MobileOS.ANDROID
            elif kivy_platform == 'ios':
                self.mobile_os = MobileOS.IOS
            
            # Create platform info
            self.platform_info = PlatformInfo(
                platform_type=PlatformType.MOBILE,
                os_name=kivy_platform or "Mobile",
                os_version=self._get_mobile_version(),
                python_version=sys.version,
                has_audio_access=True,
                has_video_access=True,
                has_microphone=await self._check_mobile_microphone(),
                has_camera=await self._check_mobile_camera(),
                has_speakers=True,
                has_gui=True,
                has_notifications=True,
                has_file_system=True,  # Limited access
                supports_webrtc=True,
                supports_websockets=True,
                supports_tcp=True,
                is_low_power=True  # Mobile devices are generally low power
            )
            
            logger.info(f"Mobile platform adapter initialized for {self.mobile_os.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize mobile adapter: {e}")
            return False
    
    def _get_mobile_version(self) -> str:
        """Get mobile OS version."""
        try:
            if self.mobile_os == MobileOS.ANDROID:
                # In real implementation, would use Android API
                return "Android 10+"
            elif self.mobile_os == MobileOS.IOS:
                # In real implementation, would use iOS API
                return "iOS 14+"
            return "Unknown"
        except Exception:
            return "Unknown"
    
    async def _check_mobile_microphone(self) -> bool:
        """Check microphone availability on mobile."""
        try:
            # In real implementation, would check permissions via platform API
            return True
        except Exception:
            return False
    
    async def _check_mobile_camera(self) -> bool:
        """Check camera availability on mobile."""
        try:
            # In real implementation, would check permissions via platform API
            return True
        except Exception:
            return False
    
    async def get_audio_devices(self) -> List[Dict[str, Any]]:
        """Get mobile audio devices."""
        try:
            devices = [
                {
                    "device_id": "builtin_mic",
                    "label": "Built-in Microphone",
                    "kind": "audioinput",
                    "group_id": "builtin"
                },
                {
                    "device_id": "builtin_speaker",
                    "label": "Built-in Speaker",
                    "kind": "audiooutput",
                    "group_id": "builtin"
                }
            ]
            
            # Add Bluetooth devices if available
            if await self._check_bluetooth_available():
                devices.extend([
                    {
                        "device_id": "bluetooth_headset",
                        "label": "Bluetooth Headset",
                        "kind": "audioinput",
                        "group_id": "bluetooth"
                    },
                    {
                        "device_id": "bluetooth_speaker",
                        "label": "Bluetooth Speaker", 
                        "kind": "audiooutput",
                        "group_id": "bluetooth"
                    }
                ])
            
            return devices
            
        except Exception as e:
            logger.error(f"Error getting mobile audio devices: {e}")
            return []
    
    async def _check_bluetooth_available(self) -> bool:
        """Check if Bluetooth devices are available."""
        # In real implementation, would check via platform API
        return True
    
    async def request_microphone_permission(self) -> bool:
        """Request microphone permission on mobile."""
        try:
            if self.mobile_os == MobileOS.ANDROID:
                # In real implementation: request via Android API
                logger.info("Mock Android microphone permission granted")
                return True
            elif self.mobile_os == MobileOS.IOS:
                # In real implementation: request via iOS API
                logger.info("Mock iOS microphone permission granted")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Mobile microphone permission error: {e}")
            return False
    
    async def request_camera_permission(self) -> bool:
        """Request camera permission on mobile."""
        try:
            if self.mobile_os == MobileOS.ANDROID:
                # In real implementation: request via Android API
                logger.info("Mock Android camera permission granted")
                return True
            elif self.mobile_os == MobileOS.IOS:
                # In real implementation: request via iOS API
                logger.info("Mock iOS camera permission granted")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Mobile camera permission error: {e}")
            return False
    
    async def show_notification(self, title: str, message: str) -> bool:
        """Show mobile notification."""
        try:
            # In real implementation: use platform notification API
            logger.info(f"Mock mobile notification: {title} - {message}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to show mobile notification: {e}")
            return False
    
    async def get_system_info(self) -> Dict[str, Any]:
        """Get mobile system information."""
        try:
            return {
                "mobile_os": self.mobile_os.value,
                "device_model": self._get_device_model(),
                "battery_level": await self._get_battery_level(),
                "network_type": await self._get_network_type(),
                "orientation": await self._get_screen_orientation(),
                "low_power_mode": await self._is_low_power_mode()
            }
            
        except Exception as e:
            logger.error(f"Error getting mobile system info: {e}")
            return {}
    
    def _get_device_model(self) -> str:
        """Get mobile device model."""
        # In real implementation, would use platform API
        return "Mock Mobile Device"
    
    async def _get_battery_level(self) -> Optional[float]:
        """Get battery level (0.0 to 1.0)."""
        # In real implementation, would use platform API
        return 0.85  # Mock 85%
    
    async def _get_network_type(self) -> str:
        """Get current network type."""
        # In real implementation, would check network API
        return "wifi"  # Mock WiFi
    
    async def _get_screen_orientation(self) -> str:
        """Get screen orientation."""
        # In real implementation, would check orientation API
        return "portrait"  # Mock portrait
    
    async def _is_low_power_mode(self) -> bool:
        """Check if device is in low power mode."""
        # In real implementation, would check power API
        return False
    
    async def cleanup(self):
        """Clean up mobile resources."""
        try:
            # Clean up mobile-specific resources
            logger.info("Mobile adapter cleaned up")
        
        except Exception as e:
            logger.error(f"Error during mobile cleanup: {e}")


class DesktopAdapter(PlatformAdapter):
    """Platform adapter for desktop environments (Windows/macOS/Linux)."""
    
    def __init__(self):
        super().__init__()
        self.desktop_os = DesktopOS.UNKNOWN
        
    async def initialize(self) -> bool:
        """Initialize desktop platform components.""" 
        try:
            # Detect desktop OS
            system = platform.system().lower()
            if system == "windows":
                self.desktop_os = DesktopOS.WINDOWS
            elif system == "darwin":
                self.desktop_os = DesktopOS.MACOS
            elif system == "linux":
                self.desktop_os = DesktopOS.LINUX
            
            # Create platform info
            self.platform_info = PlatformInfo(
                platform_type=PlatformType.DESKTOP,
                os_name=platform.system(),
                os_version=platform.release(),
                python_version=sys.version,
                has_audio_access=AUDIO_AVAILABLE,
                has_video_access=True,
                has_microphone=await self._check_desktop_microphone(),
                has_camera=await self._check_desktop_camera(),
                has_speakers=await self._check_desktop_speakers(),
                has_gui=DESKTOP_UI_AVAILABLE,
                has_notifications=True,
                has_file_system=True,
                supports_webrtc=True,
                supports_websockets=True,
                supports_tcp=True,
                cpu_cores=self._get_cpu_cores(),
                memory_mb=self._get_memory_mb()
            )
            
            logger.info(f"Desktop platform adapter initialized for {self.desktop_os.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize desktop adapter: {e}")
            return False
    
    def _get_cpu_cores(self) -> Optional[int]:
        """Get number of CPU cores."""
        try:
            import os
            return os.cpu_count()
        except Exception:
            return None
    
    def _get_memory_mb(self) -> Optional[int]:
        """Get total system memory in MB."""
        try:
            import psutil
            return int(psutil.virtual_memory().total / (1024 * 1024))
        except ImportError:
            # psutil not available
            return None
        except Exception:
            return None
    
    async def _check_desktop_microphone(self) -> bool:
        """Check microphone availability on desktop."""
        if not AUDIO_AVAILABLE:
            return False
        
        try:
            # Check if any input devices are available
            devices = sd.query_devices()
            input_devices = [d for d in devices if d['max_input_channels'] > 0]
            return len(input_devices) > 0
        except Exception:
            return False
    
    async def _check_desktop_camera(self) -> bool:
        """Check camera availability on desktop."""
        try:
            # In real implementation, would check camera via OpenCV or similar
            return True
        except Exception:
            return False
    
    async def _check_desktop_speakers(self) -> bool:
        """Check speaker availability on desktop."""
        if not AUDIO_AVAILABLE:
            return False
        
        try:
            # Check if any output devices are available
            devices = sd.query_devices()
            output_devices = [d for d in devices if d['max_output_channels'] > 0]
            return len(output_devices) > 0
        except Exception:
            return False
    
    async def get_audio_devices(self) -> List[Dict[str, Any]]:
        """Get desktop audio devices."""
        if not AUDIO_AVAILABLE:
            return []
        
        try:
            devices = []
            device_list = sd.query_devices()
            
            for i, device in enumerate(device_list):
                if device['max_input_channels'] > 0:
                    devices.append({
                        "device_id": str(i),
                        "label": device['name'],
                        "kind": "audioinput",
                        "group_id": device.get('hostapi', 'default'),
                        "sample_rate": device['default_samplerate'],
                        "channels": device['max_input_channels']
                    })
                
                if device['max_output_channels'] > 0:
                    devices.append({
                        "device_id": str(i),
                        "label": device['name'],
                        "kind": "audiooutput",
                        "group_id": device.get('hostapi', 'default'),
                        "sample_rate": device['default_samplerate'],
                        "channels": device['max_output_channels']
                    })
            
            return devices
            
        except Exception as e:
            logger.error(f"Error getting desktop audio devices: {e}")
            return []
    
    async def request_microphone_permission(self) -> bool:
        """Request microphone permission on desktop."""
        try:
            # On desktop, usually no explicit permission needed
            # But we can test access
            if AUDIO_AVAILABLE:
                # Test recording briefly
                test_duration = 0.1  # 100ms test
                sample_rate = 44100
                
                audio_data = sd.rec(
                    int(test_duration * sample_rate),
                    samplerate=sample_rate,
                    channels=1,
                    blocking=True
                )
                
                logger.info("Desktop microphone access confirmed")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Desktop microphone access error: {e}")
            return False
    
    async def request_camera_permission(self) -> bool:
        """Request camera permission on desktop."""
        try:
            # In real implementation, would test camera access
            logger.info("Mock desktop camera permission granted")
            return True
            
        except Exception as e:
            logger.error(f"Desktop camera permission error: {e}")
            return False
    
    async def show_notification(self, title: str, message: str) -> bool:
        """Show desktop notification."""
        try:
            if self.desktop_os == DesktopOS.WINDOWS:
                # In real implementation: use Windows toast notifications
                logger.info(f"Mock Windows notification: {title} - {message}")
                return True
            elif self.desktop_os == DesktopOS.MACOS:
                # In real implementation: use macOS notification center
                logger.info(f"Mock macOS notification: {title} - {message}")
                return True
            elif self.desktop_os == DesktopOS.LINUX:
                # In real implementation: use libnotify
                logger.info(f"Mock Linux notification: {title} - {message}")
                return True
            
            # Fallback to GUI dialog if available
            if DESKTOP_UI_AVAILABLE:
                messagebox.showinfo(title, message)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to show desktop notification: {e}")
            return False
    
    async def get_system_info(self) -> Dict[str, Any]:
        """Get desktop system information."""
        try:
            info = {
                "desktop_os": self.desktop_os.value,
                "architecture": platform.architecture()[0],
                "processor": platform.processor(),
                "hostname": platform.node(),
                "python_implementation": platform.python_implementation()
            }
            
            # Add additional info if psutil is available
            try:
                import psutil
                info.update({
                    "cpu_percent": psutil.cpu_percent(interval=1),
                    "memory_percent": psutil.virtual_memory().percent,
                    "disk_usage": psutil.disk_usage('/').percent,
                    "boot_time": psutil.boot_time()
                })
            except ImportError:
                pass
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting desktop system info: {e}")
            return {}
    
    async def cleanup(self):
        """Clean up desktop resources."""
        try:
            # Clean up desktop-specific resources
            logger.info("Desktop adapter cleaned up")
            
        except Exception as e:
            logger.error(f"Error during desktop cleanup: {e}")


class PlatformCompatibilityManager:
    """Main manager for cross-platform compatibility."""
    
    def __init__(self):
        self.current_platform: Optional[PlatformType] = None
        self.adapter: Optional[PlatformAdapter] = None
        self.platform_info: Optional[PlatformInfo] = None
        
        # Monitoring
        self.monitor = global_performance_monitor.register_component(
            "platform_compatibility", "communication"
        )
    
    @monitor_performance(component="platform_compatibility", operation="detect_platform")
    async def detect_and_initialize_platform(self) -> bool:
        """Detect current platform and initialize appropriate adapter."""
        try:
            # Detect platform
            self.current_platform = await self._detect_platform()
            
            # Create appropriate adapter
            if self.current_platform == PlatformType.WEB:
                self.adapter = WebAdapter()
            elif self.current_platform == PlatformType.MOBILE:
                self.adapter = MobileAdapter()
            elif self.current_platform == PlatformType.DESKTOP:
                self.adapter = DesktopAdapter()
            else:
                logger.error(f"Unsupported platform: {self.current_platform}")
                return False
            
            # Initialize adapter
            success = await self.adapter.initialize()
            
            if success:
                self.platform_info = self.adapter.platform_info
                logger.info(f"Platform compatibility initialized for {self.current_platform.value}")
                return True
            else:
                logger.error(f"Failed to initialize {self.current_platform.value} adapter")
                return False
                
        except Exception as e:
            logger.error(f"Platform detection/initialization failed: {e}")
            return False
    
    async def _detect_platform(self) -> PlatformType:
        """Detect the current platform type."""
        # Check for web environment first
        if WEB_AVAILABLE:
            return PlatformType.WEB
        
        # Check for mobile environment
        if MOBILE_AVAILABLE:
            return PlatformType.MOBILE
        
        # Check if running on desktop
        system = platform.system().lower()
        if system in ['windows', 'darwin', 'linux']:
            return PlatformType.DESKTOP
        
        # Check if running on server (no GUI)
        if not DESKTOP_UI_AVAILABLE and not MOBILE_AVAILABLE:
            return PlatformType.SERVER
        
        return PlatformType.UNKNOWN
    
    # Unified API methods that delegate to platform adapter
    
    async def get_audio_devices(self) -> List[Dict[str, Any]]:
        """Get available audio devices (platform-agnostic)."""
        if not self.adapter:
            return []
        return await self.adapter.get_audio_devices()
    
    async def request_permissions(self, microphone: bool = True, camera: bool = False) -> Dict[str, bool]:
        """Request media permissions (platform-agnostic).""" 
        if not self.adapter:
            return {"microphone": False, "camera": False}
        
        results = {}
        
        if microphone:
            results["microphone"] = await self.adapter.request_microphone_permission()
        
        if camera:
            results["camera"] = await self.adapter.request_camera_permission()
        
        return results
    
    async def show_notification(self, title: str, message: str) -> bool:
        """Show notification (platform-agnostic)."""
        if not self.adapter:
            return False
        return await self.adapter.show_notification(title, message)
    
    async def get_system_info(self) -> Dict[str, Any]:
        """Get system information (platform-agnostic)."""
        if not self.adapter:
            return {}
        
        base_info = {
            "platform_type": self.current_platform.value if self.current_platform else "unknown",
            "platform_info": self.platform_info.__dict__ if self.platform_info else {}
        }
        
        adapter_info = await self.adapter.get_system_info()
        return {**base_info, **adapter_info}
    
    def get_platform_capabilities(self) -> Dict[str, bool]:
        """Get platform capability summary."""
        if not self.platform_info:
            return {}
        
        return {
            "audio_access": self.platform_info.has_audio_access,
            "video_access": self.platform_info.has_video_access,
            "microphone": self.platform_info.has_microphone,
            "camera": self.platform_info.has_camera,
            "speakers": self.platform_info.has_speakers,
            "gui": self.platform_info.has_gui,
            "notifications": self.platform_info.has_notifications,
            "file_system": self.platform_info.has_file_system,
            "webrtc": self.platform_info.supports_webrtc,
            "websockets": self.platform_info.supports_websockets,
            "tcp": self.platform_info.supports_tcp
        }
    
    def is_mobile(self) -> bool:
        """Check if running on mobile platform."""
        return self.current_platform == PlatformType.MOBILE
    
    def is_web(self) -> bool:
        """Check if running in web environment."""
        return self.current_platform == PlatformType.WEB
    
    def is_desktop(self) -> bool:
        """Check if running on desktop."""
        return self.current_platform == PlatformType.DESKTOP
    
    def is_low_power(self) -> bool:
        """Check if platform is low power (mobile/embedded)."""
        return self.platform_info.is_low_power if self.platform_info else False
    
    async def cleanup(self):
        """Clean up platform resources."""
        try:
            if self.adapter:
                await self.adapter.cleanup()
            
            logger.info("Platform compatibility manager cleaned up")
            
        except Exception as e:
            logger.error(f"Error during platform cleanup: {e}")


# Convenience functions
async def create_platform_manager() -> PlatformCompatibilityManager:
    """Create and initialize a platform compatibility manager."""
    manager = PlatformCompatibilityManager()
    
    success = await manager.detect_and_initialize_platform()
    if not success:
        raise RuntimeError("Failed to initialize platform compatibility")
    
    return manager


# Global platform manager for easy access
_global_platform_manager: Optional[PlatformCompatibilityManager] = None


def get_global_platform_manager() -> Optional[PlatformCompatibilityManager]:
    """Get the global platform manager instance."""
    return _global_platform_manager


def set_global_platform_manager(manager: PlatformCompatibilityManager):
    """Set the global platform manager instance."""
    global _global_platform_manager
    _global_platform_manager = manager


# Platform detection utilities
def detect_platform_type() -> PlatformType:
    """Quick platform type detection (synchronous)."""
    if WEB_AVAILABLE:
        return PlatformType.WEB
    
    if MOBILE_AVAILABLE:
        return PlatformType.MOBILE
    
    system = platform.system().lower()
    if system in ['windows', 'darwin', 'linux']:
        return PlatformType.DESKTOP
    
    return PlatformType.UNKNOWN


def get_platform_requirements() -> Dict[str, List[str]]:
    """Get platform-specific dependency requirements."""
    return {
        "web": [
            "pyodide",  # For web deployment
            "js",       # JavaScript interop
        ],
        "mobile": [
            "kivy",           # Mobile UI framework
            "plyer",          # Platform-specific APIs
            "android",        # Android-specific (if targeting Android)
        ],
        "desktop": [
            "sounddevice",    # Audio device access
            "numpy",          # Audio processing
            "tkinter",        # GUI (usually built-in)
            "psutil",         # System monitoring
        ],
        "common": [
            "asyncio",        # Async support
            "logging",        # Logging
        ]
    }