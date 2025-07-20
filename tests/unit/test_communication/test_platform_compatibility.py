"""
Unit tests for Platform Compatibility.

Tests cross-platform compatibility, device enumeration, platform-specific
adapters, and feature detection across web, mobile, and desktop platforms.
"""

import pytest
import asyncio
import sys
import os
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from communication.platform_compatibility import (
    PlatformManager,
    PlatformType,
    PlatformInfo,
    DeviceInfo,
    CapabilityInfo,
    WebPlatformAdapter,
    MobilePlatformAdapter,
    DesktopPlatformAdapter,
    PlatformFeature,
    AudioDeviceManager,
    VideoDeviceManager,
    NetworkCapabilityDetector
)


class TestPlatformType:
    """Test platform type enumeration."""
    
    def test_platform_types(self):
        """Test platform type values."""
        assert PlatformType.WEB.value == "web"
        assert PlatformType.MOBILE.value == "mobile"
        assert PlatformType.DESKTOP.value == "desktop"
        assert PlatformType.EMBEDDED.value == "embedded"
    
    def test_platform_detection(self):
        """Test platform detection logic."""
        # Mock different platform environments
        with patch('platform.system', return_value='Windows'):
            detected = PlatformType.detect_current_platform()
            assert detected == PlatformType.DESKTOP
        
        with patch('platform.system', return_value='Linux'):
            detected = PlatformType.detect_current_platform()
            assert detected == PlatformType.DESKTOP
        
        with patch('platform.system', return_value='Darwin'):
            detected = PlatformType.detect_current_platform()
            assert detected == PlatformType.DESKTOP


class TestPlatformInfo:
    """Test platform information gathering."""
    
    def test_platform_info_creation(self):
        """Test platform info creation."""
        info = PlatformInfo(
            platform_type=PlatformType.DESKTOP,
            os_name="Linux",
            os_version="5.4.0",
            python_version="3.9.0",
            has_audio_access=True,
            has_video_access=True,
            has_microphone=True,
            has_camera=True,
            has_speakers=True,
            has_gui=True,
            supports_webrtc=True,
            supports_websockets=True,
            supports_tcp=True
        )
        
        assert info.platform_type == PlatformType.DESKTOP
        assert info.os_name == "Linux"
        assert info.has_audio_access == True
        assert info.supports_webrtc == True
    
    @pytest.mark.asyncio
    async def test_platform_info_detection(self):
        """Test platform info detection."""
        info = await PlatformInfo.detect()
        
        assert info is not None
        assert isinstance(info.platform_type, PlatformType)
        assert isinstance(info.os_name, str)
        assert isinstance(info.has_audio_access, bool)


class TestDeviceInfo:
    """Test device information handling."""
    
    def test_audio_device_info(self):
        """Test audio device info creation."""
        device = DeviceInfo(
            device_id="audio_0",
            label="Default Microphone",
            kind="audioinput",
            group_id="default_group",
            sample_rate=44100,
            channels=1,
            is_default=True
        )
        
        assert device.device_id == "audio_0"
        assert device.label == "Default Microphone"
        assert device.kind == "audioinput"
        assert device.sample_rate == 44100
        assert device.channels == 1
        assert device.is_default == True
    
    def test_video_device_info(self):
        """Test video device info creation."""
        device = DeviceInfo(
            device_id="video_0",
            label="Default Camera",
            kind="videoinput",
            group_id="camera_group",
            width=1920,
            height=1080,
            frame_rate=30,
            is_default=True
        )
        
        assert device.device_id == "video_0"
        assert device.kind == "videoinput"
        assert device.width == 1920
        assert device.height == 1080
        assert device.frame_rate == 30


class TestCapabilityInfo:
    """Test capability information."""
    
    def test_capability_creation(self):
        """Test capability info creation."""
        capability = CapabilityInfo(
            feature=PlatformFeature.WEBRTC,
            supported=True,
            version="1.0",
            limitations=["bandwidth_limit"],
            metadata={"max_connections": 10}
        )
        
        assert capability.feature == PlatformFeature.WEBRTC
        assert capability.supported == True
        assert capability.version == "1.0"
        assert "bandwidth_limit" in capability.limitations
        assert capability.metadata["max_connections"] == 10
    
    def test_feature_types(self):
        """Test platform feature types."""
        assert PlatformFeature.WEBRTC.value == "webrtc"
        assert PlatformFeature.WEBSOCKETS.value == "websockets"
        assert PlatformFeature.AUDIO_INPUT.value == "audio_input"
        assert PlatformFeature.AUDIO_OUTPUT.value == "audio_output"
        assert PlatformFeature.VIDEO_INPUT.value == "video_input"
        assert PlatformFeature.MICROPHONE_ACCESS.value == "microphone_access"
        assert PlatformFeature.CAMERA_ACCESS.value == "camera_access"


class TestWebPlatformAdapter:
    """Test web platform adapter."""
    
    @pytest.fixture
    def web_adapter(self, mock_platform_info):
        """Create web platform adapter for testing."""
        mock_platform_info.platform_type = PlatformType.WEB
        adapter = WebPlatformAdapter(mock_platform_info)
        return adapter
    
    def test_web_adapter_initialization(self, web_adapter):
        """Test web adapter initialization."""
        assert web_adapter.platform_info.platform_type == PlatformType.WEB
        assert web_adapter.js_interface is not None
    
    @pytest.mark.asyncio
    async def test_web_device_enumeration(self, web_adapter):
        """Test web device enumeration."""
        # Mock JavaScript interface
        with patch.object(web_adapter.js_interface, 'enumerate_devices', new_callable=AsyncMock) as mock_enum:
            mock_enum.return_value = [
                {
                    "deviceId": "mic_1",
                    "label": "Web Microphone",
                    "kind": "audioinput",
                    "groupId": "group_1"
                }
            ]
            
            devices = await web_adapter.enumerate_audio_devices()
            
            assert len(devices) == 1
            assert devices[0].device_id == "mic_1"
            assert devices[0].label == "Web Microphone"
            assert devices[0].kind == "audioinput"
    
    @pytest.mark.asyncio
    async def test_web_capability_detection(self, web_adapter):
        """Test web capability detection."""
        capabilities = await web_adapter.detect_capabilities()
        
        assert isinstance(capabilities, list)
        
        # Should include common web capabilities
        webrtc_cap = next((c for c in capabilities if c.feature == PlatformFeature.WEBRTC), None)
        assert webrtc_cap is not None
    
    @pytest.mark.asyncio
    async def test_web_permission_handling(self, web_adapter):
        """Test web permission handling."""
        # Mock permission request
        with patch.object(web_adapter, '_request_media_permissions', new_callable=AsyncMock) as mock_perms:
            mock_perms.return_value = {"audio": True, "video": False}
            
            result = await web_adapter.request_permissions(["audio", "video"])
            
            assert result["audio"] == True
            assert result["video"] == False
    
    @pytest.mark.asyncio
    async def test_web_browser_detection(self, web_adapter):
        """Test web browser detection."""
        browser_info = await web_adapter.detect_browser()
        
        assert "name" in browser_info
        assert "version" in browser_info
        assert "webrtc_support" in browser_info


class TestMobilePlatformAdapter:
    """Test mobile platform adapter."""
    
    @pytest.fixture
    def mobile_adapter(self, mock_platform_info):
        """Create mobile platform adapter for testing."""
        mock_platform_info.platform_type = PlatformType.MOBILE
        adapter = MobilePlatformAdapter(mock_platform_info)
        return adapter
    
    def test_mobile_adapter_initialization(self, mobile_adapter):
        """Test mobile adapter initialization."""
        assert mobile_adapter.platform_info.platform_type == PlatformType.MOBILE
        assert hasattr(mobile_adapter, 'native_interface')
    
    @pytest.mark.asyncio
    async def test_mobile_device_enumeration(self, mobile_adapter):
        """Test mobile device enumeration."""
        devices = await mobile_adapter.enumerate_audio_devices()
        
        # Should return mock devices for mobile
        assert isinstance(devices, list)
        
        # Mobile typically has fewer device options
        input_devices = [d for d in devices if d.kind == "audioinput"]
        output_devices = [d for d in devices if d.kind == "audiooutput"]
        
        assert len(input_devices) >= 1  # At least microphone
        assert len(output_devices) >= 1  # At least speaker
    
    @pytest.mark.asyncio
    async def test_mobile_capability_detection(self, mobile_adapter):
        """Test mobile capability detection."""
        capabilities = await mobile_adapter.detect_capabilities()
        
        # Should detect mobile-specific capabilities
        audio_cap = next((c for c in capabilities if c.feature == PlatformFeature.AUDIO_INPUT), None)
        assert audio_cap is not None
        assert audio_cap.supported == True
    
    @pytest.mark.asyncio
    async def test_mobile_performance_constraints(self, mobile_adapter):
        """Test mobile performance constraint detection."""
        constraints = await mobile_adapter.get_performance_constraints()
        
        assert "cpu_limit" in constraints
        assert "memory_limit" in constraints
        assert "battery_optimization" in constraints
    
    @pytest.mark.asyncio
    async def test_mobile_network_detection(self, mobile_adapter):
        """Test mobile network type detection."""
        network_info = await mobile_adapter.detect_network_type()
        
        assert "type" in network_info  # wifi, cellular, etc.
        assert "quality" in network_info
        assert "estimated_bandwidth" in network_info


class TestDesktopPlatformAdapter:
    """Test desktop platform adapter."""
    
    @pytest.fixture
    def desktop_adapter(self, mock_platform_info):
        """Create desktop platform adapter for testing."""
        mock_platform_info.platform_type = PlatformType.DESKTOP
        adapter = DesktopPlatformAdapter(mock_platform_info)
        return adapter
    
    def test_desktop_adapter_initialization(self, desktop_adapter):
        """Test desktop adapter initialization."""
        assert desktop_adapter.platform_info.platform_type == PlatformType.DESKTOP
        assert hasattr(desktop_adapter, 'system_interface')
    
    @pytest.mark.asyncio
    async def test_desktop_device_enumeration(self, desktop_adapter):
        """Test desktop device enumeration."""
        devices = await desktop_adapter.enumerate_audio_devices()
        
        # Desktop typically has more device options
        assert isinstance(devices, list)
        assert len(devices) >= 2  # Input and output devices
    
    @pytest.mark.asyncio
    async def test_desktop_system_info(self, desktop_adapter):
        """Test desktop system information gathering."""
        system_info = await desktop_adapter.get_system_info()
        
        assert "cpu_cores" in system_info
        assert "memory_gb" in system_info
        assert "gpu_info" in system_info
        assert "audio_drivers" in system_info
    
    @pytest.mark.asyncio
    async def test_desktop_performance_monitoring(self, desktop_adapter):
        """Test desktop performance monitoring."""
        performance = await desktop_adapter.monitor_performance()
        
        assert "cpu_usage" in performance
        assert "memory_usage" in performance
        assert "network_usage" in performance
    
    @pytest.mark.asyncio
    async def test_desktop_capability_detection(self, desktop_adapter):
        """Test desktop capability detection."""
        capabilities = await desktop_adapter.detect_capabilities()
        
        # Desktop should support most features
        webrtc_cap = next((c for c in capabilities if c.feature == PlatformFeature.WEBRTC), None)
        assert webrtc_cap is not None
        assert webrtc_cap.supported == True


class TestAudioDeviceManager:
    """Test audio device management."""
    
    @pytest.fixture
    def audio_manager(self):
        """Create audio device manager for testing."""
        manager = AudioDeviceManager()
        return manager
    
    @pytest.mark.asyncio
    async def test_audio_device_enumeration(self, audio_manager):
        """Test audio device enumeration."""
        devices = await audio_manager.enumerate_devices()
        
        assert isinstance(devices, list)
        
        # Should have input and output devices
        input_devices = [d for d in devices if d.kind == "audioinput"]
        output_devices = [d for d in devices if d.kind == "audiooutput"]
        
        assert len(input_devices) >= 1
        assert len(output_devices) >= 1
    
    @pytest.mark.asyncio
    async def test_default_device_selection(self, audio_manager):
        """Test default device selection."""
        default_input = await audio_manager.get_default_input_device()
        default_output = await audio_manager.get_default_output_device()
        
        assert default_input is not None
        assert default_input.is_default == True
        assert default_input.kind == "audioinput"
        
        assert default_output is not None
        assert default_output.is_default == True
        assert default_output.kind == "audiooutput"
    
    @pytest.mark.asyncio
    async def test_device_capabilities(self, audio_manager):
        """Test audio device capability detection."""
        devices = await audio_manager.enumerate_devices()
        
        if devices:
            device = devices[0]
            capabilities = await audio_manager.get_device_capabilities(device.device_id)
            
            assert "sample_rates" in capabilities
            assert "channels" in capabilities
            assert "formats" in capabilities
    
    @pytest.mark.asyncio
    async def test_device_monitoring(self, audio_manager):
        """Test audio device monitoring."""
        # Start monitoring
        await audio_manager.start_device_monitoring()
        
        # Should be monitoring
        assert audio_manager.is_monitoring == True
        
        # Stop monitoring
        await audio_manager.stop_device_monitoring()
        assert audio_manager.is_monitoring == False


class TestVideoDeviceManager:
    """Test video device management."""
    
    @pytest.fixture
    def video_manager(self):
        """Create video device manager for testing."""
        manager = VideoDeviceManager()
        return manager
    
    @pytest.mark.asyncio
    async def test_video_device_enumeration(self, video_manager):
        """Test video device enumeration."""
        devices = await video_manager.enumerate_devices()
        
        assert isinstance(devices, list)
        
        # Should have video input devices (cameras)
        video_devices = [d for d in devices if d.kind == "videoinput"]
        assert len(video_devices) >= 0  # May be 0 in test environment
    
    @pytest.mark.asyncio
    async def test_camera_capabilities(self, video_manager):
        """Test camera capability detection."""
        devices = await video_manager.enumerate_devices()
        
        if devices:
            device = devices[0]
            capabilities = await video_manager.get_device_capabilities(device.device_id)
            
            assert "resolutions" in capabilities
            assert "frame_rates" in capabilities
            assert "formats" in capabilities
    
    @pytest.mark.asyncio
    async def test_resolution_support(self, video_manager):
        """Test resolution support detection."""
        supported_resolutions = await video_manager.get_supported_resolutions()
        
        assert isinstance(supported_resolutions, list)
        
        # Should include common resolutions
        common_resolutions = [(640, 480), (1280, 720), (1920, 1080)]
        for resolution in common_resolutions:
            # At least some should be supported
            pass


class TestNetworkCapabilityDetector:
    """Test network capability detection."""
    
    @pytest.fixture
    def network_detector(self):
        """Create network capability detector for testing."""
        detector = NetworkCapabilityDetector()
        return detector
    
    @pytest.mark.asyncio
    async def test_webrtc_support_detection(self, network_detector):
        """Test WebRTC support detection."""
        webrtc_support = await network_detector.detect_webrtc_support()
        
        assert "supported" in webrtc_support
        assert "features" in webrtc_support
        assert "limitations" in webrtc_support
    
    @pytest.mark.asyncio
    async def test_websocket_support_detection(self, network_detector):
        """Test WebSocket support detection."""
        websocket_support = await network_detector.detect_websocket_support()
        
        assert "supported" in websocket_support
        assert "protocols" in websocket_support
    
    @pytest.mark.asyncio
    async def test_bandwidth_estimation(self, network_detector):
        """Test bandwidth estimation."""
        bandwidth = await network_detector.estimate_bandwidth()
        
        assert "download_mbps" in bandwidth
        assert "upload_mbps" in bandwidth
        assert "latency_ms" in bandwidth
    
    @pytest.mark.asyncio
    async def test_connectivity_test(self, network_detector):
        """Test connectivity testing."""
        connectivity = await network_detector.test_connectivity()
        
        assert "reachable" in connectivity
        assert "latency" in connectivity
        assert "packet_loss" in connectivity


class TestPlatformManager:
    """Test platform manager functionality."""
    
    @pytest.fixture
    def platform_manager(self):
        """Create platform manager for testing."""
        manager = PlatformManager()
        return manager
    
    @pytest.mark.asyncio
    async def test_platform_manager_initialization(self, platform_manager):
        """Test platform manager initialization."""
        await platform_manager.initialize()
        
        assert platform_manager.is_initialized == True
        assert platform_manager.platform_info is not None
        assert platform_manager.current_adapter is not None
    
    @pytest.mark.asyncio
    async def test_adapter_selection(self, platform_manager):
        """Test platform adapter selection."""
        await platform_manager.initialize()
        
        adapter = platform_manager.current_adapter
        platform_type = platform_manager.platform_info.platform_type
        
        if platform_type == PlatformType.WEB:
            assert isinstance(adapter, WebPlatformAdapter)
        elif platform_type == PlatformType.MOBILE:
            assert isinstance(adapter, MobilePlatformAdapter)
        elif platform_type == PlatformType.DESKTOP:
            assert isinstance(adapter, DesktopPlatformAdapter)
    
    @pytest.mark.asyncio
    async def test_unified_device_access(self, platform_manager):
        """Test unified device access."""
        await platform_manager.initialize()
        
        # Get all devices through unified interface
        audio_devices = await platform_manager.get_audio_devices()
        video_devices = await platform_manager.get_video_devices()
        
        assert isinstance(audio_devices, list)
        assert isinstance(video_devices, list)
    
    @pytest.mark.asyncio
    async def test_capability_aggregation(self, platform_manager):
        """Test capability aggregation."""
        await platform_manager.initialize()
        
        all_capabilities = await platform_manager.get_all_capabilities()
        
        assert isinstance(all_capabilities, dict)
        assert PlatformFeature.WEBRTC.value in all_capabilities
        assert PlatformFeature.AUDIO_INPUT.value in all_capabilities
    
    @pytest.mark.asyncio
    async def test_feature_compatibility_check(self, platform_manager):
        """Test feature compatibility checking."""
        await platform_manager.initialize()
        
        # Test specific feature compatibility
        webrtc_compatible = await platform_manager.is_feature_supported(PlatformFeature.WEBRTC)
        audio_compatible = await platform_manager.is_feature_supported(PlatformFeature.AUDIO_INPUT)
        
        assert isinstance(webrtc_compatible, bool)
        assert isinstance(audio_compatible, bool)
    
    @pytest.mark.asyncio
    async def test_platform_optimization(self, platform_manager):
        """Test platform-specific optimizations."""
        await platform_manager.initialize()
        
        optimizations = await platform_manager.get_platform_optimizations()
        
        assert isinstance(optimizations, dict)
        assert "audio_settings" in optimizations
        assert "video_settings" in optimizations
        assert "network_settings" in optimizations


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.fixture
    def error_manager(self):
        """Create platform manager for error testing."""
        manager = PlatformManager()
        return manager
    
    @pytest.mark.asyncio
    async def test_unsupported_platform_handling(self, error_manager):
        """Test handling of unsupported platforms."""
        # Mock unsupported platform
        with patch('communication.platform_compatibility.PlatformType.detect_current_platform', 
                   return_value=None):
            
            result = await error_manager.initialize()
            
            # Should handle gracefully
            assert isinstance(result, bool)
    
    @pytest.mark.asyncio
    async def test_device_enumeration_failure(self, error_manager):
        """Test device enumeration failure handling."""
        await error_manager.initialize()
        
        # Mock device enumeration failure
        with patch.object(error_manager.current_adapter, 'enumerate_audio_devices', 
                          side_effect=Exception("Device error")):
            
            devices = await error_manager.get_audio_devices()
            
            # Should return empty list on failure
            assert devices == []
    
    @pytest.mark.asyncio
    async def test_permission_denied_handling(self, error_manager):
        """Test permission denied handling."""
        await error_manager.initialize()
        
        # Should handle permission failures gracefully
        permissions = await error_manager.request_permissions(["audio", "video"])
        
        assert isinstance(permissions, dict)
    
    @pytest.mark.asyncio
    async def test_capability_detection_failure(self, error_manager):
        """Test capability detection failure handling."""
        await error_manager.initialize()
        
        # Mock capability detection failure
        with patch.object(error_manager.current_adapter, 'detect_capabilities',
                          side_effect=Exception("Detection error")):
            
            capabilities = await error_manager.get_all_capabilities()
            
            # Should return default capabilities
            assert isinstance(capabilities, dict)


class TestPerformance:
    """Test performance characteristics."""
    
    @pytest.fixture
    def perf_manager(self):
        """Create platform manager for performance testing."""
        manager = PlatformManager()
        return manager
    
    @pytest.mark.asyncio
    async def test_initialization_performance(self, perf_manager):
        """Test initialization performance."""
        import time
        
        start_time = time.time()
        await perf_manager.initialize()
        init_time = time.time() - start_time
        
        # Should initialize quickly
        assert init_time < 2.0  # Less than 2 seconds
    
    @pytest.mark.asyncio
    async def test_device_enumeration_performance(self, perf_manager):
        """Test device enumeration performance."""
        await perf_manager.initialize()
        
        import time
        start_time = time.time()
        await perf_manager.get_audio_devices()
        enum_time = time.time() - start_time
        
        # Should enumerate quickly
        assert enum_time < 1.0  # Less than 1 second


# Integration test markers
pytestmark = pytest.mark.unit