"""
Unit tests for WebRTC Statistics.

Tests WebRTC statistics collection, performance analysis, diagnostic reports,
and real-time statistics monitoring.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import sys
import os
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from communication.webrtc_statistics import (
    WebRTCStatisticsCollector,
    StatisticsConfig,
    RTCStatReport,
    StatisticType,
    ConnectionStatistics,
    MediaStatistics,
    NetworkStatistics,
    PerformanceAnalyzer,
    StatisticsTrend,
    DiagnosticReport,
    StatisticsEvent,
    StatisticsEventType
)


class TestStatisticsConfig:
    """Test statistics configuration."""
    
    def test_default_config(self):
        """Test default statistics configuration."""
        config = StatisticsConfig()
        
        assert config.collection_interval_ms == 1000
        assert config.history_window_size == 60
        assert config.enable_performance_analysis == True
        assert config.enable_trend_analysis == True
        assert config.enable_diagnostic_reports == True
        assert config.report_generation_interval_ms == 30000
        assert config.enable_real_time_alerts == True
        assert config.alert_thresholds is not None
    
    def test_custom_config(self):
        """Test custom statistics configuration."""
        config = StatisticsConfig(
            collection_interval_ms=500,
            history_window_size=120,
            enable_performance_analysis=False,
            report_generation_interval_ms=60000
        )
        
        assert config.collection_interval_ms == 500
        assert config.history_window_size == 120
        assert config.enable_performance_analysis == False
        assert config.report_generation_interval_ms == 60000


class TestStatisticType:
    """Test statistic type enumeration."""
    
    def test_statistic_types(self):
        """Test statistic type values."""
        assert StatisticType.INBOUND_RTP.value == "inbound-rtp"
        assert StatisticType.OUTBOUND_RTP.value == "outbound-rtp"
        assert StatisticType.CANDIDATE_PAIR.value == "candidate-pair"
        assert StatisticType.LOCAL_CANDIDATE.value == "local-candidate"
        assert StatisticType.REMOTE_CANDIDATE.value == "remote-candidate"
        assert StatisticType.CERTIFICATE.value == "certificate"
        assert StatisticType.CODEC.value == "codec"
        assert StatisticType.TRANSPORT.value == "transport"


class TestRTCStatReport:
    """Test RTC statistics report."""
    
    def test_stat_report_creation(self):
        """Test RTC stat report creation."""
        report = RTCStatReport(
            stat_type=StatisticType.INBOUND_RTP,
            ssrc=123456789,
            media_type="audio",
            packets_received=1000,
            bytes_received=128000,
            packets_lost=10,
            jitter=0.025,
            round_trip_time=0.120,
            timestamp=time.time()
        )
        
        assert report.stat_type == StatisticType.INBOUND_RTP
        assert report.ssrc == 123456789
        assert report.media_type == "audio"
        assert report.packets_received == 1000
        assert report.bytes_received == 128000
        assert report.packets_lost == 10
        assert report.jitter == 0.025
        assert report.round_trip_time == 0.120
    
    def test_packet_loss_calculation(self):
        """Test packet loss percentage calculation."""
        report = RTCStatReport(
            packets_received=950,
            packets_lost=50
        )
        
        loss_percent = report.calculate_packet_loss_percent()
        assert loss_percent == 5.0  # 50/(950+50) * 100
    
    def test_bitrate_calculation(self):
        """Test bitrate calculation."""
        current_time = time.time()
        
        report1 = RTCStatReport(
            bytes_received=64000,
            timestamp=current_time
        )
        
        report2 = RTCStatReport(
            bytes_received=128000,
            timestamp=current_time + 1.0  # 1 second later
        )
        
        bitrate = report2.calculate_bitrate(report1)
        assert bitrate == 512000.0  # (128000-64000) * 8 bits/byte


class TestConnectionStatistics:
    """Test connection statistics."""
    
    def test_connection_stats_creation(self):
        """Test connection statistics creation."""
        stats = ConnectionStatistics(
            connection_state="connected",
            ice_connection_state="connected",
            dtls_state="connected",
            selected_candidate_pair_id="pair_1",
            local_candidate_type="host",
            remote_candidate_type="host",
            current_round_trip_time=0.150,
            available_outgoing_bitrate=256000,
            available_incoming_bitrate=512000
        )
        
        assert stats.connection_state == "connected"
        assert stats.ice_connection_state == "connected"
        assert stats.dtls_state == "connected"
        assert stats.current_round_trip_time == 0.150
        assert stats.available_outgoing_bitrate == 256000
    
    def test_connection_quality_assessment(self):
        """Test connection quality assessment."""
        good_stats = ConnectionStatistics(
            connection_state="connected",
            current_round_trip_time=0.080,
            available_outgoing_bitrate=512000
        )
        
        poor_stats = ConnectionStatistics(
            connection_state="connecting",
            current_round_trip_time=0.300,
            available_outgoing_bitrate=32000
        )
        
        assert good_stats.assess_quality() == "good"
        assert poor_stats.assess_quality() == "poor"


class TestMediaStatistics:
    """Test media statistics."""
    
    def test_media_stats_creation(self):
        """Test media statistics creation."""
        stats = MediaStatistics(
            media_type="audio",
            codec_name="opus",
            clock_rate=48000,
            bitrate_bps=64000,
            packets_sent=500,
            packets_received=480,
            bytes_sent=64000,
            bytes_received=61440,
            audio_level=0.75,
            total_audio_energy=1234.5,
            total_samples_duration=10.0
        )
        
        assert stats.media_type == "audio"
        assert stats.codec_name == "opus"
        assert stats.clock_rate == 48000
        assert stats.bitrate_bps == 64000
        assert stats.audio_level == 0.75
    
    def test_media_quality_metrics(self):
        """Test media quality metrics calculation."""
        stats = MediaStatistics(
            packets_sent=1000,
            packets_received=950,
            bytes_sent=128000,
            bytes_received=121600,
            audio_level=0.8
        )
        
        packet_loss = stats.calculate_packet_loss()
        assert packet_loss == 5.0  # (1000-950)/1000 * 100
        
        efficiency = stats.calculate_transmission_efficiency()
        assert efficiency == 95.0  # 121600/128000 * 100


class TestNetworkStatistics:
    """Test network statistics."""
    
    def test_network_stats_creation(self):
        """Test network statistics creation."""
        stats = NetworkStatistics(
            network_type="wifi",
            transport_type="udp",
            local_ip="192.168.1.100",
            remote_ip="203.0.113.1",
            total_round_trip_time=1.5,
            current_round_trip_time=0.120,
            available_outgoing_bitrate=1000000,
            available_incoming_bitrate=2000000,
            bytes_sent=1024000,
            bytes_received=2048000,
            packets_sent=1000,
            packets_received=2000
        )
        
        assert stats.network_type == "wifi"
        assert stats.transport_type == "udp"
        assert stats.local_ip == "192.168.1.100"
        assert stats.remote_ip == "203.0.113.1"
        assert stats.current_round_trip_time == 0.120
    
    def test_network_performance_calculation(self):
        """Test network performance calculation."""
        stats = NetworkStatistics(
            available_outgoing_bitrate=1000000,
            available_incoming_bitrate=2000000,
            current_round_trip_time=0.050,
            packets_sent=1000,
            packets_received=1000
        )
        
        performance = stats.calculate_network_performance()
        
        assert performance["latency_category"] == "excellent"  # < 100ms
        assert performance["bandwidth_category"] == "high"
        assert performance["reliability_score"] > 0.9


class TestPerformanceAnalyzer:
    """Test performance analyzer."""
    
    @pytest.fixture
    def performance_analyzer(self):
        """Create performance analyzer for testing."""
        config = StatisticsConfig()
        analyzer = PerformanceAnalyzer(config)
        return analyzer
    
    def test_analyzer_initialization(self, performance_analyzer):
        """Test performance analyzer initialization."""
        assert performance_analyzer.config is not None
        assert len(performance_analyzer.performance_history) == 0
        assert performance_analyzer.baseline_metrics is None
    
    def test_baseline_establishment(self, performance_analyzer):
        """Test performance baseline establishment."""
        # Add multiple performance samples
        for i in range(10):
            stats = ConnectionStatistics(
                current_round_trip_time=0.100 + i * 0.01,
                available_outgoing_bitrate=500000 + i * 10000
            )
            performance_analyzer.add_performance_sample(stats)
        
        baseline = performance_analyzer.establish_baseline()
        
        assert baseline is not None
        assert "average_rtt" in baseline
        assert "average_bitrate" in baseline
        assert baseline["sample_count"] == 10
    
    def test_performance_deviation_detection(self, performance_analyzer):
        """Test performance deviation detection."""
        # Establish baseline with good performance
        for i in range(5):
            good_stats = ConnectionStatistics(
                current_round_trip_time=0.080,
                available_outgoing_bitrate=1000000
            )
            performance_analyzer.add_performance_sample(good_stats)
        
        performance_analyzer.establish_baseline()
        
        # Add degraded performance sample
        bad_stats = ConnectionStatistics(
            current_round_trip_time=0.300,  # Much higher latency
            available_outgoing_bitrate=100000  # Much lower bitrate
        )
        
        deviation = performance_analyzer.detect_performance_deviation(bad_stats)
        
        assert deviation is not None
        assert deviation["latency_deviation"] > 1.0  # Significant deviation
        assert deviation["bitrate_deviation"] > 1.0
    
    def test_trend_analysis(self, performance_analyzer):
        """Test performance trend analysis."""
        # Add degrading performance samples
        for i in range(10):
            stats = ConnectionStatistics(
                current_round_trip_time=0.050 + i * 0.020,  # Increasing latency
                available_outgoing_bitrate=1000000 - i * 50000  # Decreasing bitrate
            )
            performance_analyzer.add_performance_sample(stats)
        
        trend = performance_analyzer.analyze_performance_trend()
        
        assert trend is not None
        assert trend.direction == "degrading"
        assert trend.confidence > 0.7
    
    def test_performance_score_calculation(self, performance_analyzer):
        """Test performance score calculation."""
        excellent_stats = ConnectionStatistics(
            current_round_trip_time=0.030,
            available_outgoing_bitrate=2000000
        )
        
        poor_stats = ConnectionStatistics(
            current_round_trip_time=0.400,
            available_outgoing_bitrate=50000
        )
        
        excellent_score = performance_analyzer.calculate_performance_score(excellent_stats)
        poor_score = performance_analyzer.calculate_performance_score(poor_stats)
        
        assert excellent_score > poor_score
        assert 0.0 <= excellent_score <= 1.0
        assert 0.0 <= poor_score <= 1.0


class TestWebRTCStatisticsCollector:
    """Test WebRTC statistics collector."""
    
    @pytest.fixture
    def statistics_collector(self, mock_rtc_stats):
        """Create statistics collector for testing."""
        config = StatisticsConfig()
        collector = WebRTCStatisticsCollector(config)
        return collector
    
    def test_collector_initialization(self, statistics_collector):
        """Test statistics collector initialization."""
        assert statistics_collector.config is not None
        assert statistics_collector.is_collecting == False
        assert statistics_collector.performance_analyzer is not None
        assert len(statistics_collector.statistics_history) == 0
    
    @pytest.mark.asyncio
    async def test_collection_lifecycle(self, statistics_collector):
        """Test statistics collection lifecycle."""
        # Mock WebRTC connection
        mock_connection = Mock()
        mock_connection.get_stats = AsyncMock(return_value={
            "inbound-rtp": {
                "type": "inbound-rtp",
                "packetsReceived": 1000,
                "bytesReceived": 128000
            }
        })
        
        # Start collection
        await statistics_collector.start_collection(mock_connection)
        assert statistics_collector.is_collecting == True
        
        # Stop collection
        await statistics_collector.stop_collection()
        assert statistics_collector.is_collecting == False
    
    @pytest.mark.asyncio
    async def test_statistics_parsing(self, statistics_collector, mock_rtc_stats):
        """Test raw statistics parsing."""
        parsed_stats = await statistics_collector.parse_raw_statistics(mock_rtc_stats)
        
        assert parsed_stats is not None
        assert "connection_stats" in parsed_stats
        assert "media_stats" in parsed_stats
        assert "network_stats" in parsed_stats
    
    @pytest.mark.asyncio
    async def test_real_time_analysis(self, statistics_collector):
        """Test real-time statistics analysis."""
        await statistics_collector.start_collection(Mock())
        
        # Mock statistics update
        mock_stats = {
            "inbound-rtp": {
                "type": "inbound-rtp",
                "packetsReceived": 1000,
                "bytesReceived": 128000,
                "packetsLost": 50
            }
        }
        
        await statistics_collector._process_statistics_update(mock_stats)
        
        # Should have processed and stored statistics
        assert len(statistics_collector.statistics_history) > 0
        
        await statistics_collector.stop_collection()
    
    def test_callback_registration(self, statistics_collector):
        """Test statistics event callback registration."""
        events_received = []
        
        def stats_callback(event):
            events_received.append(event)
        
        statistics_collector.on_statistics_updated(stats_callback)
        statistics_collector.on_performance_alert(stats_callback)
        
        assert len(statistics_collector.statistics_updated_callbacks) == 1
        assert len(statistics_collector.performance_alert_callbacks) == 1
    
    @pytest.mark.asyncio
    async def test_diagnostic_report_generation(self, statistics_collector):
        """Test diagnostic report generation."""
        # Add some statistics history
        for i in range(5):
            stats = {
                "connection_stats": ConnectionStatistics(
                    current_round_trip_time=0.100 + i * 0.01
                ),
                "timestamp": time.time() + i
            }
            statistics_collector.statistics_history.append(stats)
        
        report = await statistics_collector.generate_diagnostic_report()
        
        assert report is not None
        assert "summary" in report
        assert "performance_analysis" in report
        assert "recommendations" in report
        assert "trend_analysis" in report
    
    def test_statistics_export(self, statistics_collector):
        """Test statistics data export."""
        # Add test data
        test_stats = {
            "connection_stats": ConnectionStatistics(current_round_trip_time=0.120),
            "timestamp": time.time()
        }
        statistics_collector.statistics_history.append(test_stats)
        
        # Export as JSON
        json_export = statistics_collector.export_statistics("json")
        assert json_export is not None
        assert isinstance(json_export, str)
        
        # Export as CSV
        csv_export = statistics_collector.export_statistics("csv")
        assert csv_export is not None
        assert isinstance(csv_export, str)


class TestStatisticsTrend:
    """Test statistics trend analysis."""
    
    def test_trend_creation(self):
        """Test statistics trend creation."""
        trend = StatisticsTrend(
            metric_name="latency",
            direction="increasing",
            magnitude=0.5,
            confidence=0.8,
            time_period_seconds=60,
            start_value=0.100,
            end_value=0.150
        )
        
        assert trend.metric_name == "latency"
        assert trend.direction == "increasing"
        assert trend.magnitude == 0.5
        assert trend.confidence == 0.8
        assert trend.time_period_seconds == 60
    
    def test_trend_significance(self):
        """Test trend significance assessment."""
        significant_trend = StatisticsTrend(
            magnitude=0.8,
            confidence=0.9
        )
        
        insignificant_trend = StatisticsTrend(
            magnitude=0.1,
            confidence=0.3
        )
        
        assert significant_trend.is_significant() == True
        assert insignificant_trend.is_significant() == False


class TestDiagnosticReport:
    """Test diagnostic report generation."""
    
    def test_report_creation(self):
        """Test diagnostic report creation."""
        report = DiagnosticReport(
            timestamp=time.time(),
            connection_summary="Connected via UDP",
            performance_score=0.85,
            identified_issues=["High packet loss detected"],
            recommendations=["Reduce video quality"],
            trend_analysis={"latency": "stable"},
            raw_statistics={"packets_sent": 1000}
        )
        
        assert report.performance_score == 0.85
        assert len(report.identified_issues) == 1
        assert len(report.recommendations) == 1
        assert "latency" in report.trend_analysis
    
    def test_report_severity_assessment(self):
        """Test report severity assessment."""
        critical_report = DiagnosticReport(
            performance_score=0.2,
            identified_issues=["Connection failed", "High packet loss"]
        )
        
        good_report = DiagnosticReport(
            performance_score=0.9,
            identified_issues=[]
        )
        
        assert critical_report.assess_severity() == "critical"
        assert good_report.assess_severity() == "good"


class TestStatisticsEvent:
    """Test statistics event handling."""
    
    def test_statistics_event_creation(self):
        """Test statistics event creation."""
        event = StatisticsEvent(
            event_type=StatisticsEventType.PERFORMANCE_DEGRADED,
            metric_name="packet_loss",
            current_value=5.0,
            threshold_value=2.0,
            severity=Mock(value="high"),
            message="Packet loss exceeded threshold",
            timestamp=time.time()
        )
        
        assert event.event_type == StatisticsEventType.PERFORMANCE_DEGRADED
        assert event.metric_name == "packet_loss"
        assert event.current_value == 5.0
        assert event.threshold_value == 2.0
        assert event.severity.value == "high"
    
    def test_event_types(self):
        """Test statistics event types."""
        assert StatisticsEventType.STATISTICS_UPDATED.value == "statistics_updated"
        assert StatisticsEventType.PERFORMANCE_IMPROVED.value == "performance_improved"
        assert StatisticsEventType.PERFORMANCE_DEGRADED.value == "performance_degraded"
        assert StatisticsEventType.CONNECTION_QUALITY_CHANGED.value == "connection_quality_changed"
        assert StatisticsEventType.THRESHOLD_EXCEEDED.value == "threshold_exceeded"


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.fixture
    def error_collector(self):
        """Create statistics collector for error testing."""
        config = StatisticsConfig()
        collector = WebRTCStatisticsCollector(config)
        return collector
    
    @pytest.mark.asyncio
    async def test_invalid_statistics_handling(self, error_collector):
        """Test handling of invalid statistics data."""
        # Test with None data
        parsed = await error_collector.parse_raw_statistics(None)
        assert parsed is not None  # Should return empty/default structure
        
        # Test with malformed data
        malformed_stats = {"invalid": "structure"}
        parsed = await error_collector.parse_raw_statistics(malformed_stats)
        assert parsed is not None
    
    @pytest.mark.asyncio
    async def test_collection_error_recovery(self, error_collector):
        """Test collection error recovery."""
        # Mock connection that fails stats collection
        mock_connection = Mock()
        mock_connection.get_stats = AsyncMock(side_effect=Exception("Stats error"))
        
        await error_collector.start_collection(mock_connection)
        
        # Should handle error gracefully and continue
        await asyncio.sleep(0.1)
        assert error_collector.is_collecting == True
        
        await error_collector.stop_collection()
    
    @pytest.mark.asyncio
    async def test_callback_error_handling(self, error_collector):
        """Test error handling in statistics callbacks."""
        def failing_callback(event):
            raise Exception("Callback failed")
        
        error_collector.on_statistics_updated(failing_callback)
        
        # Should not crash when callback fails
        stats_event = StatisticsEvent(
            event_type=StatisticsEventType.STATISTICS_UPDATED,
            metric_name="test"
        )
        
        await error_collector._trigger_statistics_updated(stats_event)
        # Test continues if no exception raised


class TestPerformance:
    """Test performance characteristics."""
    
    @pytest.fixture
    def perf_collector(self):
        """Create statistics collector for performance testing."""
        config = StatisticsConfig(collection_interval_ms=100)  # Fast collection
        collector = WebRTCStatisticsCollector(config)
        return collector
    
    @pytest.mark.asyncio
    async def test_collection_performance(self, perf_collector):
        """Test statistics collection performance."""
        mock_connection = Mock()
        mock_connection.get_stats = AsyncMock(return_value={
            "inbound-rtp": {"type": "inbound-rtp", "packetsReceived": 1000}
        })
        
        await perf_collector.start_collection(mock_connection)
        
        # Let it collect for a short time
        await asyncio.sleep(0.3)
        
        # Should have collected multiple samples efficiently
        assert len(perf_collector.statistics_history) > 0
        
        await perf_collector.stop_collection()
    
    @pytest.mark.asyncio
    async def test_analysis_performance(self, perf_collector):
        """Test statistics analysis performance."""
        import time
        
        # Large statistics object
        large_stats = {
            f"stat_{i}": {
                "type": f"type_{i}",
                "value": i * 1000
            } for i in range(100)
        }
        
        start_time = time.time()
        await perf_collector.parse_raw_statistics(large_stats)
        analysis_time = time.time() - start_time
        
        # Should analyze quickly
        assert analysis_time < 0.1  # Less than 100ms


# Integration test markers
pytestmark = pytest.mark.unit