"""
Unit tests for Network Diagnostics.

Tests comprehensive network diagnostics, connectivity tests, issue detection,
troubleshooting recommendations, and network performance analysis.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import sys
import os
from typing import Dict, List, Any, Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from communication.network_diagnostics import (
    NetworkDiagnostics,
    DiagnosticsConfig,
    ConnectivityTest,
    TestType,
    TestResult,
    TestSeverity,
    NetworkIssue,
    IssueType,
    IssueSeverity,
    TroubleshootingRecommendation,
    NetworkPerformanceAnalyzer,
    BandwidthTester,
    LatencyAnalyzer,
    PacketLossDetector,
    JitterMeasurer,
    DiagnosticReport,
    NetworkConditionDetector
)


class TestDiagnosticsConfig:
    """Test network diagnostics configuration."""
    
    def test_default_config(self):
        """Test default diagnostics configuration."""
        config = DiagnosticsConfig()
        
        assert config.enable_connectivity_tests == True
        assert config.enable_bandwidth_tests == True
        assert config.enable_latency_tests == True
        assert config.enable_packet_loss_tests == True
        assert config.enable_jitter_tests == True
        assert config.test_timeout_seconds == 30
        assert config.test_interval_seconds == 300
        assert config.max_test_attempts == 3
        assert config.enable_continuous_monitoring == False
        assert config.enable_issue_detection == True
        assert config.enable_recommendations == True
    
    def test_custom_config(self):
        """Test custom diagnostics configuration."""
        config = DiagnosticsConfig(
            test_timeout_seconds=60,
            test_interval_seconds=600,
            max_test_attempts=5,
            enable_continuous_monitoring=True,
            enable_bandwidth_tests=False
        )
        
        assert config.test_timeout_seconds == 60
        assert config.test_interval_seconds == 600
        assert config.max_test_attempts == 5
        assert config.enable_continuous_monitoring == True
        assert config.enable_bandwidth_tests == False


class TestTestType:
    """Test diagnostic test type enumeration."""
    
    def test_test_types(self):
        """Test diagnostic test type values."""
        assert TestType.CONNECTIVITY.value == "connectivity"
        assert TestType.BANDWIDTH.value == "bandwidth"
        assert TestType.LATENCY.value == "latency"
        assert TestType.PACKET_LOSS.value == "packet_loss"
        assert TestType.JITTER.value == "jitter"
        assert TestType.DNS_RESOLUTION.value == "dns_resolution"
        assert TestType.FIREWALL.value == "firewall"
        assert TestType.NAT_TRAVERSAL.value == "nat_traversal"
    
    def test_test_priorities(self):
        """Test test type priorities."""
        assert TestType.CONNECTIVITY.get_priority() > TestType.BANDWIDTH.get_priority()
        assert TestType.LATENCY.get_priority() > TestType.JITTER.get_priority()


class TestTestSeverity:
    """Test test severity levels."""
    
    def test_severity_values(self):
        """Test test severity values."""
        assert TestSeverity.INFO.value == "info"
        assert TestSeverity.WARNING.value == "warning"
        assert TestSeverity.ERROR.value == "error"
        assert TestSeverity.CRITICAL.value == "critical"
    
    def test_severity_ordering(self):
        """Test severity level ordering."""
        assert TestSeverity.INFO.get_numeric_value() < TestSeverity.WARNING.get_numeric_value()
        assert TestSeverity.WARNING.get_numeric_value() < TestSeverity.ERROR.get_numeric_value()
        assert TestSeverity.ERROR.get_numeric_value() < TestSeverity.CRITICAL.get_numeric_value()


class TestConnectivityTest:
    """Test connectivity test implementation."""
    
    def test_test_creation(self):
        """Test connectivity test creation."""
        test = ConnectivityTest(
            test_type=TestType.CONNECTIVITY,
            target_host="test.example.com",
            target_port=443,
            protocol="TCP",
            timeout_seconds=10
        )
        
        assert test.test_type == TestType.CONNECTIVITY
        assert test.target_host == "test.example.com"
        assert test.target_port == 443
        assert test.protocol == "TCP"
        assert test.timeout_seconds == 10
    
    @pytest.mark.asyncio
    async def test_tcp_connectivity(self):
        """Test TCP connectivity testing."""
        test = ConnectivityTest(
            test_type=TestType.CONNECTIVITY,
            target_host="google.com",
            target_port=80,
            protocol="TCP"
        )
        
        # Mock successful connection
        with patch('socket.create_connection') as mock_connect:
            mock_connect.return_value.close = Mock()
            
            result = await test.execute()
            
            assert result.success == True
            assert result.test_type == TestType.CONNECTIVITY
    
    @pytest.mark.asyncio
    async def test_udp_connectivity(self):
        """Test UDP connectivity testing."""
        test = ConnectivityTest(
            test_type=TestType.CONNECTIVITY,
            target_host="8.8.8.8",
            target_port=53,
            protocol="UDP"
        )
        
        # Mock UDP socket
        with patch('socket.socket') as mock_socket:
            mock_udp = Mock()
            mock_udp.sendto = Mock()
            mock_udp.recvfrom = Mock(return_value=(b'response', ('8.8.8.8', 53)))
            mock_socket.return_value = mock_udp
            
            result = await test.execute()
            
            assert result.success == True
    
    @pytest.mark.asyncio
    async def test_connection_failure(self):
        """Test connection failure handling."""
        test = ConnectivityTest(
            test_type=TestType.CONNECTIVITY,
            target_host="nonexistent.invalid",
            target_port=12345,
            protocol="TCP"
        )
        
        # Mock connection failure
        with patch('socket.create_connection', side_effect=ConnectionRefusedError("Connection refused")):
            result = await test.execute()
            
            assert result.success == False
            assert result.error_message is not None
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test connection timeout handling."""
        test = ConnectivityTest(
            test_type=TestType.CONNECTIVITY,
            target_host="test.example.com",
            target_port=443,
            timeout_seconds=0.001  # Very short timeout
        )
        
        # Mock timeout
        with patch('socket.create_connection', side_effect=asyncio.TimeoutError("Timeout")):
            result = await test.execute()
            
            assert result.success == False
            assert "timeout" in result.error_message.lower()


class TestTestResult:
    """Test test result handling."""
    
    def test_result_creation(self):
        """Test test result creation."""
        result = TestResult(
            test_type=TestType.LATENCY,
            success=True,
            duration_ms=150.5,
            result_data={"average_latency": 25.3, "min_latency": 20.1, "max_latency": 30.8},
            error_message=None,
            severity=TestSeverity.INFO,
            timestamp=time.time()
        )
        
        assert result.test_type == TestType.LATENCY
        assert result.success == True
        assert result.duration_ms == 150.5
        assert result.result_data["average_latency"] == 25.3
        assert result.severity == TestSeverity.INFO
    
    def test_result_analysis(self):
        """Test test result analysis."""
        # Good latency result
        good_result = TestResult(
            test_type=TestType.LATENCY,
            success=True,
            result_data={"average_latency": 25.0}
        )
        
        # Poor latency result
        poor_result = TestResult(
            test_type=TestType.LATENCY,
            success=True,
            result_data={"average_latency": 300.0}
        )
        
        assert good_result.analyze_quality() == "excellent"
        assert poor_result.analyze_quality() == "poor"
    
    def test_result_comparison(self):
        """Test test result comparison."""
        baseline_result = TestResult(
            test_type=TestType.BANDWIDTH,
            result_data={"download_mbps": 100.0, "upload_mbps": 50.0}
        )
        
        current_result = TestResult(
            test_type=TestType.BANDWIDTH,
            result_data={"download_mbps": 80.0, "upload_mbps": 40.0}
        )
        
        comparison = current_result.compare_with(baseline_result)
        
        assert comparison["download_degradation"] == 20.0  # 20% decrease
        assert comparison["upload_degradation"] == 20.0


class TestBandwidthTester:
    """Test bandwidth testing functionality."""
    
    @pytest.fixture
    def bandwidth_tester(self):
        """Create bandwidth tester for testing."""
        config = DiagnosticsConfig()
        tester = BandwidthTester(config)
        return tester
    
    @pytest.mark.asyncio
    async def test_download_speed_test(self, bandwidth_tester):
        """Test download speed measurement."""
        # Mock HTTP response
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = Mock()
            mock_response.content.read = AsyncMock(return_value=b'x' * 1024 * 1024)  # 1MB
            mock_get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_get.return_value.__aexit__ = AsyncMock(return_value=None)
            
            result = await bandwidth_tester.test_download_speed()
            
            assert result.success == True
            assert "download_mbps" in result.result_data
            assert result.result_data["download_mbps"] > 0
    
    @pytest.mark.asyncio
    async def test_upload_speed_test(self, bandwidth_tester):
        """Test upload speed measurement."""
        # Mock HTTP response
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = Mock()
            mock_response.status = 200
            mock_post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_post.return_value.__aexit__ = AsyncMock(return_value=None)
            
            result = await bandwidth_tester.test_upload_speed()
            
            assert result.success == True
            assert "upload_mbps" in result.result_data
            assert result.result_data["upload_mbps"] > 0
    
    @pytest.mark.asyncio
    async def test_bidirectional_speed_test(self, bandwidth_tester):
        """Test bidirectional speed measurement."""
        # Mock both download and upload
        with patch.object(bandwidth_tester, 'test_download_speed') as mock_download, \
             patch.object(bandwidth_tester, 'test_upload_speed') as mock_upload:
            
            mock_download.return_value = TestResult(
                test_type=TestType.BANDWIDTH,
                success=True,
                result_data={"download_mbps": 100.0}
            )
            
            mock_upload.return_value = TestResult(
                test_type=TestType.BANDWIDTH,
                success=True,
                result_data={"upload_mbps": 50.0}
            )
            
            result = await bandwidth_tester.test_bidirectional_speed()
            
            assert result.success == True
            assert result.result_data["download_mbps"] == 100.0
            assert result.result_data["upload_mbps"] == 50.0


class TestLatencyAnalyzer:
    """Test latency analysis functionality."""
    
    @pytest.fixture
    def latency_analyzer(self):
        """Create latency analyzer for testing."""
        config = DiagnosticsConfig()
        analyzer = LatencyAnalyzer(config)
        return analyzer
    
    @pytest.mark.asyncio
    async def test_ping_latency_measurement(self, latency_analyzer):
        """Test ping latency measurement."""
        # Mock ping response
        with patch('ping3.ping') as mock_ping:
            mock_ping.return_value = 0.025  # 25ms
            
            result = await latency_analyzer.measure_ping_latency("8.8.8.8")
            
            assert result.success == True
            assert result.result_data["latency_ms"] == 25.0
    
    @pytest.mark.asyncio
    async def test_tcp_latency_measurement(self, latency_analyzer):
        """Test TCP connection latency measurement."""
        # Mock TCP connection timing
        with patch('socket.create_connection') as mock_connect:
            mock_socket = Mock()
            mock_socket.close = Mock()
            mock_connect.return_value = mock_socket
            
            # Mock timing
            with patch('time.perf_counter', side_effect=[0.0, 0.025]):  # 25ms
                result = await latency_analyzer.measure_tcp_latency("test.com", 80)
                
                assert result.success == True
                assert result.result_data["tcp_latency_ms"] == 25.0
    
    @pytest.mark.asyncio
    async def test_multi_sample_latency(self, latency_analyzer):
        """Test multi-sample latency measurement."""
        # Mock multiple ping responses
        with patch('ping3.ping', side_effect=[0.020, 0.025, 0.030, 0.022, 0.028]):
            result = await latency_analyzer.measure_latency_statistics("8.8.8.8", samples=5)
            
            assert result.success == True
            assert "average_latency" in result.result_data
            assert "min_latency" in result.result_data
            assert "max_latency" in result.result_data
            assert "jitter_ms" in result.result_data
    
    @pytest.mark.asyncio
    async def test_latency_trending(self, latency_analyzer):
        """Test latency trending analysis."""
        # Add historical latency measurements
        measurements = [20.0, 22.0, 25.0, 28.0, 32.0]  # Increasing trend
        
        for latency in measurements:
            latency_analyzer.add_measurement(latency, time.time())
        
        trend = latency_analyzer.analyze_latency_trend()
        
        assert trend["direction"] == "increasing"
        assert trend["magnitude"] > 0


class TestPacketLossDetector:
    """Test packet loss detection."""
    
    @pytest.fixture
    def packet_loss_detector(self):
        """Create packet loss detector for testing."""
        config = DiagnosticsConfig()
        detector = PacketLossDetector(config)
        return detector
    
    @pytest.mark.asyncio
    async def test_icmp_packet_loss(self, packet_loss_detector):
        """Test ICMP packet loss detection."""
        # Mock ping responses (some failures)
        ping_responses = [0.020, None, 0.025, 0.030, None]  # 2 failures out of 5
        
        with patch('ping3.ping', side_effect=ping_responses):
            result = await packet_loss_detector.detect_icmp_packet_loss("8.8.8.8", count=5)
            
            assert result.success == True
            assert result.result_data["packet_loss_percent"] == 40.0  # 2/5 * 100
            assert result.result_data["packets_sent"] == 5
            assert result.result_data["packets_received"] == 3
    
    @pytest.mark.asyncio
    async def test_udp_packet_loss(self, packet_loss_detector):
        """Test UDP packet loss detection."""
        # Mock UDP socket behavior
        with patch('socket.socket') as mock_socket:
            mock_udp = Mock()
            
            # Simulate some packets being lost (no response)
            def mock_recvfrom(size):
                if mock_recvfrom.call_count % 3 == 0:  # Every 3rd packet lost
                    raise socket.timeout("Timeout")
                return (b'response', ('8.8.8.8', 53))
            
            mock_recvfrom.call_count = 0
            mock_udp.recvfrom = mock_recvfrom
            mock_udp.sendto = Mock()
            mock_socket.return_value = mock_udp
            
            result = await packet_loss_detector.detect_udp_packet_loss("8.8.8.8", 53, count=9)
            
            assert result.success == True
            # Should detect approximately 33% packet loss
            assert 30 <= result.result_data["packet_loss_percent"] <= 40
    
    @pytest.mark.asyncio
    async def test_packet_loss_severity_assessment(self, packet_loss_detector):
        """Test packet loss severity assessment."""
        # Low packet loss
        low_loss_result = TestResult(
            test_type=TestType.PACKET_LOSS,
            success=True,
            result_data={"packet_loss_percent": 0.5}
        )
        
        # High packet loss
        high_loss_result = TestResult(
            test_type=TestType.PACKET_LOSS,
            success=True,
            result_data={"packet_loss_percent": 15.0}
        )
        
        low_severity = packet_loss_detector.assess_packet_loss_severity(low_loss_result)
        high_severity = packet_loss_detector.assess_packet_loss_severity(high_loss_result)
        
        assert low_severity == TestSeverity.INFO
        assert high_severity == TestSeverity.ERROR


class TestJitterMeasurer:
    """Test jitter measurement."""
    
    @pytest.fixture
    def jitter_measurer(self):
        """Create jitter measurer for testing."""
        config = DiagnosticsConfig()
        measurer = JitterMeasurer(config)
        return measurer
    
    @pytest.mark.asyncio
    async def test_latency_jitter_calculation(self, jitter_measurer):
        """Test jitter calculation from latency measurements."""
        # Mock latency measurements with variation
        latencies = [20.0, 25.0, 18.0, 30.0, 22.0, 35.0, 19.0, 28.0]
        
        with patch('ping3.ping', side_effect=[l/1000 for l in latencies]):
            result = await jitter_measurer.measure_jitter("8.8.8.8", samples=8)
            
            assert result.success == True
            assert "jitter_ms" in result.result_data
            assert "max_jitter_ms" in result.result_data
            assert result.result_data["jitter_ms"] > 0
    
    @pytest.mark.asyncio
    async def test_rtp_jitter_simulation(self, jitter_measurer):
        """Test RTP-style jitter measurement."""
        # Simulate RTP packet timing
        packet_times = [0.0, 0.020, 0.042, 0.065, 0.085, 0.108]  # Varying intervals
        
        jitter = jitter_measurer.calculate_rtp_jitter(packet_times)
        
        assert jitter >= 0.0
        assert isinstance(jitter, float)
    
    def test_jitter_quality_assessment(self, jitter_measurer):
        """Test jitter quality assessment."""
        # Low jitter (good)
        low_jitter = 5.0
        
        # High jitter (poor)
        high_jitter = 50.0
        
        low_quality = jitter_measurer.assess_jitter_quality(low_jitter)
        high_quality = jitter_measurer.assess_jitter_quality(high_jitter)
        
        assert low_quality in ["excellent", "good"]
        assert high_quality in ["poor", "unacceptable"]


class TestNetworkIssue:
    """Test network issue detection and representation."""
    
    def test_issue_creation(self):
        """Test network issue creation."""
        issue = NetworkIssue(
            issue_type=IssueType.HIGH_LATENCY,
            severity=IssueSeverity.WARNING,
            description="High latency detected to primary server",
            affected_component="connectivity",
            detected_value=250.0,
            threshold_value=100.0,
            timestamp=time.time()
        )
        
        assert issue.issue_type == IssueType.HIGH_LATENCY
        assert issue.severity == IssueSeverity.WARNING
        assert issue.description == "High latency detected to primary server"
        assert issue.detected_value == 250.0
        assert issue.threshold_value == 100.0
    
    def test_issue_types(self):
        """Test network issue types."""
        assert IssueType.CONNECTIVITY_FAILURE.value == "connectivity_failure"
        assert IssueType.HIGH_LATENCY.value == "high_latency"
        assert IssueType.PACKET_LOSS.value == "packet_loss"
        assert IssueType.LOW_BANDWIDTH.value == "low_bandwidth"
        assert IssueType.HIGH_JITTER.value == "high_jitter"
        assert IssueType.DNS_RESOLUTION_FAILURE.value == "dns_resolution_failure"
        assert IssueType.FIREWALL_BLOCKING.value == "firewall_blocking"
    
    def test_issue_severity_levels(self):
        """Test issue severity levels."""
        assert IssueSeverity.LOW.value == "low"
        assert IssueSeverity.MEDIUM.value == "medium"
        assert IssueSeverity.HIGH.value == "high"
        assert IssueSeverity.CRITICAL.value == "critical"


class TestTroubleshootingRecommendation:
    """Test troubleshooting recommendations."""
    
    def test_recommendation_creation(self):
        """Test troubleshooting recommendation creation."""
        recommendation = TroubleshootingRecommendation(
            issue_type=IssueType.HIGH_LATENCY,
            title="Optimize network route",
            description="Consider using a different server region",
            steps=[
                "Check current server region",
                "Test latency to other regions",
                "Switch to lower latency region"
            ],
            priority=2,
            estimated_impact="High",
            difficulty="Easy"
        )
        
        assert recommendation.issue_type == IssueType.HIGH_LATENCY
        assert recommendation.title == "Optimize network route"
        assert len(recommendation.steps) == 3
        assert recommendation.priority == 2
        assert recommendation.estimated_impact == "High"
    
    def test_recommendation_applicability(self):
        """Test recommendation applicability check."""
        latency_recommendation = TroubleshootingRecommendation(
            issue_type=IssueType.HIGH_LATENCY,
            title="Reduce latency"
        )
        
        bandwidth_recommendation = TroubleshootingRecommendation(
            issue_type=IssueType.LOW_BANDWIDTH,
            title="Increase bandwidth"
        )
        
        latency_issue = NetworkIssue(issue_type=IssueType.HIGH_LATENCY)
        bandwidth_issue = NetworkIssue(issue_type=IssueType.LOW_BANDWIDTH)
        
        assert latency_recommendation.is_applicable_to(latency_issue) == True
        assert latency_recommendation.is_applicable_to(bandwidth_issue) == False
        assert bandwidth_recommendation.is_applicable_to(bandwidth_issue) == True


class TestNetworkConditionDetector:
    """Test network condition detection."""
    
    @pytest.fixture
    def condition_detector(self):
        """Create network condition detector for testing."""
        config = DiagnosticsConfig()
        detector = NetworkConditionDetector(config)
        return detector
    
    @pytest.mark.asyncio
    async def test_overall_condition_assessment(self, condition_detector):
        """Test overall network condition assessment."""
        # Mock test results
        test_results = [
            TestResult(TestType.CONNECTIVITY, success=True),
            TestResult(TestType.LATENCY, success=True, result_data={"average_latency": 25.0}),
            TestResult(TestType.BANDWIDTH, success=True, result_data={"download_mbps": 100.0}),
            TestResult(TestType.PACKET_LOSS, success=True, result_data={"packet_loss_percent": 0.5})
        ]
        
        condition = await condition_detector.assess_network_condition(test_results)
        
        assert condition["overall_quality"] in ["excellent", "good", "fair", "poor"]
        assert "latency_quality" in condition
        assert "bandwidth_quality" in condition
        assert "reliability_quality" in condition
    
    @pytest.mark.asyncio
    async def test_bottleneck_detection(self, condition_detector):
        """Test network bottleneck detection."""
        # Results indicating bandwidth bottleneck
        bottleneck_results = [
            TestResult(TestType.CONNECTIVITY, success=True),
            TestResult(TestType.LATENCY, success=True, result_data={"average_latency": 20.0}),  # Good
            TestResult(TestType.BANDWIDTH, success=True, result_data={"download_mbps": 5.0}),   # Poor
            TestResult(TestType.PACKET_LOSS, success=True, result_data={"packet_loss_percent": 0.1})  # Good
        ]
        
        bottlenecks = await condition_detector.detect_bottlenecks(bottleneck_results)
        
        assert len(bottlenecks) > 0
        assert any(b["type"] == "bandwidth" for b in bottlenecks)
    
    def test_connection_type_detection(self, condition_detector):
        """Test connection type detection based on characteristics."""
        # Cellular-like characteristics
        cellular_results = {
            "latency": 80.0,
            "bandwidth": 20.0,
            "jitter": 15.0,
            "packet_loss": 1.0
        }
        
        # Fiber-like characteristics
        fiber_results = {
            "latency": 5.0,
            "bandwidth": 500.0,
            "jitter": 2.0,
            "packet_loss": 0.1
        }
        
        cellular_type = condition_detector.detect_connection_type(cellular_results)
        fiber_type = condition_detector.detect_connection_type(fiber_results)
        
        assert cellular_type in ["cellular", "wireless"]
        assert fiber_type in ["fiber", "ethernet", "high_speed"]


class TestNetworkDiagnostics:
    """Test main network diagnostics functionality."""
    
    @pytest.fixture
    def network_diagnostics(self):
        """Create network diagnostics for testing."""
        config = DiagnosticsConfig()
        diagnostics = NetworkDiagnostics(config)
        return diagnostics
    
    def test_diagnostics_initialization(self, network_diagnostics):
        """Test network diagnostics initialization."""
        assert network_diagnostics.config is not None
        assert network_diagnostics.bandwidth_tester is not None
        assert network_diagnostics.latency_analyzer is not None
        assert network_diagnostics.packet_loss_detector is not None
        assert network_diagnostics.jitter_measurer is not None
        assert len(network_diagnostics.test_results) == 0
    
    @pytest.mark.asyncio
    async def test_comprehensive_diagnostics(self, network_diagnostics):
        """Test comprehensive network diagnostics execution."""
        # Mock all component tests
        with patch.object(network_diagnostics.bandwidth_tester, 'test_bidirectional_speed') as mock_bandwidth, \
             patch.object(network_diagnostics.latency_analyzer, 'measure_latency_statistics') as mock_latency, \
             patch.object(network_diagnostics.packet_loss_detector, 'detect_icmp_packet_loss') as mock_packet_loss, \
             patch.object(network_diagnostics.jitter_measurer, 'measure_jitter') as mock_jitter:
            
            # Setup mock returns
            mock_bandwidth.return_value = TestResult(TestType.BANDWIDTH, success=True, result_data={"download_mbps": 100.0})
            mock_latency.return_value = TestResult(TestType.LATENCY, success=True, result_data={"average_latency": 25.0})
            mock_packet_loss.return_value = TestResult(TestType.PACKET_LOSS, success=True, result_data={"packet_loss_percent": 0.5})
            mock_jitter.return_value = TestResult(TestType.JITTER, success=True, result_data={"jitter_ms": 5.0})
            
            results = await network_diagnostics.run_comprehensive_diagnostics()
            
            assert len(results) >= 4
            assert any(r.test_type == TestType.BANDWIDTH for r in results)
            assert any(r.test_type == TestType.LATENCY for r in results)
            assert any(r.test_type == TestType.PACKET_LOSS for r in results)
            assert any(r.test_type == TestType.JITTER for r in results)
    
    @pytest.mark.asyncio
    async def test_targeted_diagnostics(self, network_diagnostics):
        """Test targeted diagnostics for specific issues."""
        # Test latency-focused diagnostics
        with patch.object(network_diagnostics.latency_analyzer, 'measure_latency_statistics') as mock_latency:
            mock_latency.return_value = TestResult(TestType.LATENCY, success=True)
            
            results = await network_diagnostics.run_targeted_diagnostics([TestType.LATENCY])
            
            assert len(results) == 1
            assert results[0].test_type == TestType.LATENCY
    
    @pytest.mark.asyncio
    async def test_issue_detection(self, network_diagnostics):
        """Test network issue detection from test results."""
        # Create results indicating issues
        problem_results = [
            TestResult(TestType.LATENCY, success=True, result_data={"average_latency": 300.0}),  # High latency
            TestResult(TestType.PACKET_LOSS, success=True, result_data={"packet_loss_percent": 5.0}),  # High packet loss
            TestResult(TestType.BANDWIDTH, success=True, result_data={"download_mbps": 1.0})  # Low bandwidth
        ]
        
        issues = await network_diagnostics.detect_issues(problem_results)
        
        assert len(issues) >= 2  # Should detect multiple issues
        issue_types = [issue.issue_type for issue in issues]
        assert IssueType.HIGH_LATENCY in issue_types
        assert IssueType.PACKET_LOSS in issue_types or IssueType.LOW_BANDWIDTH in issue_types
    
    @pytest.mark.asyncio
    async def test_recommendation_generation(self, network_diagnostics):
        """Test troubleshooting recommendation generation."""
        # Create issues
        issues = [
            NetworkIssue(
                issue_type=IssueType.HIGH_LATENCY,
                severity=IssueSeverity.MEDIUM,
                detected_value=200.0
            ),
            NetworkIssue(
                issue_type=IssueType.PACKET_LOSS,
                severity=IssueSeverity.HIGH,
                detected_value=3.0
            )
        ]
        
        recommendations = await network_diagnostics.generate_recommendations(issues)
        
        assert len(recommendations) >= len(issues)
        rec_types = [rec.issue_type for rec in recommendations]
        assert IssueType.HIGH_LATENCY in rec_types
        assert IssueType.PACKET_LOSS in rec_types
    
    @pytest.mark.asyncio
    async def test_diagnostic_report_generation(self, network_diagnostics):
        """Test comprehensive diagnostic report generation."""
        # Mock comprehensive diagnostics
        with patch.object(network_diagnostics, 'run_comprehensive_diagnostics') as mock_diagnostics, \
             patch.object(network_diagnostics, 'detect_issues') as mock_issues, \
             patch.object(network_diagnostics, 'generate_recommendations') as mock_recommendations:
            
            # Setup mocks
            mock_diagnostics.return_value = [
                TestResult(TestType.CONNECTIVITY, success=True),
                TestResult(TestType.LATENCY, success=True, result_data={"average_latency": 50.0})
            ]
            
            mock_issues.return_value = [
                NetworkIssue(issue_type=IssueType.HIGH_LATENCY, severity=IssueSeverity.MEDIUM)
            ]
            
            mock_recommendations.return_value = [
                TroubleshootingRecommendation(issue_type=IssueType.HIGH_LATENCY, title="Optimize routing")
            ]
            
            report = await network_diagnostics.generate_diagnostic_report()
            
            assert report is not None
            assert "test_results" in report
            assert "detected_issues" in report
            assert "recommendations" in report
            assert "summary" in report


class TestDiagnosticReport:
    """Test diagnostic report structure."""
    
    def test_report_creation(self):
        """Test diagnostic report creation."""
        report = DiagnosticReport(
            timestamp=time.time(),
            test_results=[
                TestResult(TestType.CONNECTIVITY, success=True),
                TestResult(TestType.LATENCY, success=True, result_data={"average_latency": 25.0})
            ],
            detected_issues=[
                NetworkIssue(issue_type=IssueType.HIGH_LATENCY, severity=IssueSeverity.LOW)
            ],
            recommendations=[
                TroubleshootingRecommendation(issue_type=IssueType.HIGH_LATENCY, title="Optimize connection")
            ],
            overall_assessment="Good network performance with minor latency issues"
        )
        
        assert len(report.test_results) == 2
        assert len(report.detected_issues) == 1
        assert len(report.recommendations) == 1
        assert report.overall_assessment is not None
    
    def test_report_summary_generation(self):
        """Test report summary generation."""
        report = DiagnosticReport(
            test_results=[
                TestResult(TestType.CONNECTIVITY, success=True),
                TestResult(TestType.BANDWIDTH, success=True, result_data={"download_mbps": 100.0}),
                TestResult(TestType.LATENCY, success=False)
            ],
            detected_issues=[],
            recommendations=[]
        )
        
        summary = report.generate_summary()
        
        assert "total_tests" in summary
        assert "successful_tests" in summary
        assert "failed_tests" in summary
        assert summary["total_tests"] == 3
        assert summary["successful_tests"] == 2
        assert summary["failed_tests"] == 1


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.fixture
    def error_diagnostics(self):
        """Create network diagnostics for error testing."""
        config = DiagnosticsConfig()
        diagnostics = NetworkDiagnostics(config)
        return diagnostics
    
    @pytest.mark.asyncio
    async def test_test_timeout_handling(self, error_diagnostics):
        """Test handling of test timeouts."""
        # Mock test that times out
        with patch.object(error_diagnostics.latency_analyzer, 'measure_latency_statistics',
                          side_effect=asyncio.TimeoutError("Test timeout")):
            
            results = await error_diagnostics.run_targeted_diagnostics([TestType.LATENCY])
            
            # Should handle timeout gracefully
            assert len(results) == 1
            assert results[0].success == False
            assert "timeout" in results[0].error_message.lower()
    
    @pytest.mark.asyncio
    async def test_network_unreachable_handling(self, error_diagnostics):
        """Test handling of network unreachable errors."""
        # Mock connectivity test failure
        unreachable_test = ConnectivityTest(
            test_type=TestType.CONNECTIVITY,
            target_host="unreachable.invalid",
            target_port=80
        )
        
        with patch('socket.create_connection', side_effect=OSError("Network unreachable")):
            result = await unreachable_test.execute()
            
            assert result.success == False
            assert result.error_message is not None
    
    @pytest.mark.asyncio
    async def test_partial_test_failure_handling(self, error_diagnostics):
        """Test handling when some tests fail."""
        # Mock mixed success/failure results
        with patch.object(error_diagnostics.bandwidth_tester, 'test_bidirectional_speed',
                          side_effect=Exception("Bandwidth test failed")), \
             patch.object(error_diagnostics.latency_analyzer, 'measure_latency_statistics') as mock_latency:
            
            mock_latency.return_value = TestResult(TestType.LATENCY, success=True)
            
            results = await error_diagnostics.run_comprehensive_diagnostics()
            
            # Should have some successful and some failed results
            successful_results = [r for r in results if r.success]
            failed_results = [r for r in results if not r.success]
            
            assert len(successful_results) > 0
            assert len(failed_results) > 0


class TestPerformance:
    """Test performance characteristics."""
    
    @pytest.fixture
    def perf_diagnostics(self):
        """Create network diagnostics for performance testing."""
        config = DiagnosticsConfig(test_timeout_seconds=5)  # Shorter timeout for tests
        diagnostics = NetworkDiagnostics(config)
        return diagnostics
    
    @pytest.mark.asyncio
    async def test_diagnostic_execution_performance(self, perf_diagnostics):
        """Test diagnostic execution performance."""
        import time
        
        # Mock fast test responses
        with patch.object(perf_diagnostics.latency_analyzer, 'measure_latency_statistics') as mock_latency:
            mock_latency.return_value = TestResult(TestType.LATENCY, success=True)
            
            start_time = time.time()
            await perf_diagnostics.run_targeted_diagnostics([TestType.LATENCY])
            execution_time = time.time() - start_time
            
            # Should execute quickly
            assert execution_time < 1.0  # Less than 1 second
    
    @pytest.mark.asyncio
    async def test_concurrent_test_execution(self, perf_diagnostics):
        """Test concurrent test execution performance."""
        import time
        
        # Mock multiple test types
        with patch.object(perf_diagnostics.latency_analyzer, 'measure_latency_statistics') as mock_latency, \
             patch.object(perf_diagnostics.bandwidth_tester, 'test_bidirectional_speed') as mock_bandwidth, \
             patch.object(perf_diagnostics.packet_loss_detector, 'detect_icmp_packet_loss') as mock_packet_loss:
            
            # Setup mock delays
            async def slow_latency():
                await asyncio.sleep(0.1)
                return TestResult(TestType.LATENCY, success=True)
            
            async def slow_bandwidth():
                await asyncio.sleep(0.1)
                return TestResult(TestType.BANDWIDTH, success=True)
            
            async def slow_packet_loss():
                await asyncio.sleep(0.1)
                return TestResult(TestType.PACKET_LOSS, success=True)
            
            mock_latency.side_effect = slow_latency
            mock_bandwidth.side_effect = slow_bandwidth
            mock_packet_loss.side_effect = slow_packet_loss
            
            start_time = time.time()
            results = await perf_diagnostics.run_targeted_diagnostics([
                TestType.LATENCY,
                TestType.BANDWIDTH,
                TestType.PACKET_LOSS
            ])
            execution_time = time.time() - start_time
            
            # Should execute concurrently (less than sum of individual times)
            assert execution_time < 0.25  # Much less than 0.3 seconds (3 * 0.1)
            assert len(results) == 3


# Integration test markers
pytestmark = pytest.mark.unit