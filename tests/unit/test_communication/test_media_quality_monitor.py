"""
Unit tests for Media Quality Monitor.

Tests media quality monitoring, adaptive bitrate control, MOS score calculation,
quality trend analysis, and performance optimization.
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

from communication.media_quality_monitor import (
    MediaQualityMonitor,
    QualityConfig,
    QualityLevel,
    QualityMetrics,
    BitrateController,
    MOSCalculator,
    QualityTrendAnalyzer,
    AdaptationStrategy,
    QualityEvent,
    QualityEventType,
    NetworkCondition,
    QualityThresholds
)


class TestQualityConfig:
    """Test quality monitoring configuration."""
    
    def test_default_config(self):
        """Test default quality configuration."""
        config = QualityConfig()
        
        assert config.monitoring_interval_ms == 1000
        assert config.adaptation_interval_ms == 5000
        assert config.quality_window_size == 10
        assert config.min_audio_bitrate == 6000
        assert config.max_audio_bitrate == 128000
        assert config.default_audio_bitrate == 32000
        assert config.packet_loss_threshold == 2.0
        assert config.latency_threshold_ms == 200.0
        assert config.jitter_threshold_ms == 50.0
        assert config.enable_adaptation == True
        assert config.enable_trend_analysis == True
    
    def test_custom_config(self):
        """Test custom quality configuration."""
        config = QualityConfig(
            monitoring_interval_ms=500,
            min_audio_bitrate=8000,
            max_audio_bitrate=64000,
            packet_loss_threshold=1.0,
            latency_threshold_ms=150.0
        )
        
        assert config.monitoring_interval_ms == 500
        assert config.min_audio_bitrate == 8000
        assert config.max_audio_bitrate == 64000
        assert config.packet_loss_threshold == 1.0
        assert config.latency_threshold_ms == 150.0


class TestQualityLevel:
    """Test quality level enumeration."""
    
    def test_quality_levels(self):
        """Test quality level values."""
        assert QualityLevel.POOR.value == "poor"
        assert QualityLevel.FAIR.value == "fair"
        assert QualityLevel.GOOD.value == "good"
        assert QualityLevel.EXCELLENT.value == "excellent"
    
    def test_quality_level_ordering(self):
        """Test quality level numeric ordering."""
        assert QualityLevel.POOR.get_numeric_value() < QualityLevel.FAIR.get_numeric_value()
        assert QualityLevel.FAIR.get_numeric_value() < QualityLevel.GOOD.get_numeric_value()
        assert QualityLevel.GOOD.get_numeric_value() < QualityLevel.EXCELLENT.get_numeric_value()
    
    def test_quality_from_mos(self):
        """Test quality level determination from MOS score."""
        assert QualityLevel.from_mos_score(1.5) == QualityLevel.POOR
        assert QualityLevel.from_mos_score(2.5) == QualityLevel.FAIR
        assert QualityLevel.from_mos_score(3.5) == QualityLevel.GOOD
        assert QualityLevel.from_mos_score(4.5) == QualityLevel.EXCELLENT


class TestQualityMetrics:
    """Test quality metrics handling."""
    
    def test_metrics_creation(self):
        """Test quality metrics creation."""
        metrics = QualityMetrics(
            bitrate_bps=64000.0,
            packet_loss_percent=1.5,
            jitter_ms=25.0,
            latency_ms=120.0,
            audio_level=0.8,
            quality_level=QualityLevel.GOOD,
            mos_score=3.8,
            timestamp=time.time()
        )
        
        assert metrics.bitrate_bps == 64000.0
        assert metrics.packet_loss_percent == 1.5
        assert metrics.jitter_ms == 25.0
        assert metrics.latency_ms == 120.0
        assert metrics.audio_level == 0.8
        assert metrics.quality_level == QualityLevel.GOOD
        assert metrics.mos_score == 3.8
    
    def test_metrics_comparison(self):
        """Test metrics comparison and validation."""
        good_metrics = QualityMetrics(
            packet_loss_percent=0.5,
            latency_ms=100.0,
            jitter_ms=15.0,
            mos_score=4.0
        )
        
        poor_metrics = QualityMetrics(
            packet_loss_percent=5.0,
            latency_ms=300.0,
            jitter_ms=80.0,
            mos_score=2.0
        )
        
        assert good_metrics.is_better_than(poor_metrics) == True
        assert poor_metrics.is_better_than(good_metrics) == False
    
    def test_metrics_validation(self):
        """Test metrics validation."""
        valid_metrics = QualityMetrics(
            packet_loss_percent=1.0,
            latency_ms=150.0,
            jitter_ms=30.0
        )
        
        invalid_metrics = QualityMetrics(
            packet_loss_percent=10.0,  # High packet loss
            latency_ms=500.0,          # High latency
            jitter_ms=100.0            # High jitter
        )
        
        thresholds = QualityThresholds()
        assert valid_metrics.meets_thresholds(thresholds) == True
        assert invalid_metrics.meets_thresholds(thresholds) == False


class TestQualityThresholds:
    """Test quality thresholds."""
    
    def test_default_thresholds(self):
        """Test default quality thresholds."""
        thresholds = QualityThresholds()
        
        assert thresholds.max_packet_loss_percent == 2.0
        assert thresholds.max_latency_ms == 200.0
        assert thresholds.max_jitter_ms == 50.0
        assert thresholds.min_mos_score == 3.0
        assert thresholds.min_audio_level == 0.1
    
    def test_threshold_validation(self):
        """Test threshold validation logic."""
        thresholds = QualityThresholds(
            max_packet_loss_percent=1.0,
            max_latency_ms=150.0,
            max_jitter_ms=30.0
        )
        
        # Good metrics
        assert thresholds.validate_packet_loss(0.5) == True
        assert thresholds.validate_latency(100.0) == True
        assert thresholds.validate_jitter(20.0) == True
        
        # Bad metrics
        assert thresholds.validate_packet_loss(2.0) == False
        assert thresholds.validate_latency(200.0) == False
        assert thresholds.validate_jitter(50.0) == False


class TestMOSCalculator:
    """Test MOS (Mean Opinion Score) calculator."""
    
    @pytest.fixture
    def mos_calculator(self):
        """Create MOS calculator for testing."""
        return MOSCalculator()
    
    def test_mos_calculation_good_quality(self, mos_calculator):
        """Test MOS calculation for good quality metrics."""
        metrics = QualityMetrics(
            packet_loss_percent=0.5,
            latency_ms=80.0,
            jitter_ms=10.0
        )
        
        mos_score = mos_calculator.calculate_mos(metrics)
        
        # Should be high quality score
        assert mos_score >= 4.0
        assert mos_score <= 5.0
    
    def test_mos_calculation_poor_quality(self, mos_calculator):
        """Test MOS calculation for poor quality metrics."""
        metrics = QualityMetrics(
            packet_loss_percent=8.0,
            latency_ms=400.0,
            jitter_ms=100.0
        )
        
        mos_score = mos_calculator.calculate_mos(metrics)
        
        # Should be low quality score
        assert mos_score >= 1.0
        assert mos_score <= 2.5
    
    def test_mos_calculation_with_audio_level(self, mos_calculator):
        """Test MOS calculation including audio level."""
        metrics = QualityMetrics(
            packet_loss_percent=1.0,
            latency_ms=120.0,
            jitter_ms=25.0,
            audio_level=0.9  # Good audio level
        )
        
        mos_score = mos_calculator.calculate_mos(metrics)
        
        # Should account for good audio level
        assert mos_score >= 3.5
    
    def test_mos_calculation_edge_cases(self, mos_calculator):
        """Test MOS calculation edge cases."""
        # Perfect metrics
        perfect_metrics = QualityMetrics(
            packet_loss_percent=0.0,
            latency_ms=10.0,
            jitter_ms=1.0,
            audio_level=1.0
        )
        
        perfect_mos = mos_calculator.calculate_mos(perfect_metrics)
        assert perfect_mos == 5.0
        
        # Worst metrics
        worst_metrics = QualityMetrics(
            packet_loss_percent=50.0,
            latency_ms=1000.0,
            jitter_ms=200.0,
            audio_level=0.0
        )
        
        worst_mos = mos_calculator.calculate_mos(worst_metrics)
        assert worst_mos == 1.0


class TestBitrateController:
    """Test bitrate controller."""
    
    @pytest.fixture
    def bitrate_controller(self, mock_quality_metrics):
        """Create bitrate controller for testing."""
        config = QualityConfig()
        controller = BitrateController(config)
        return controller
    
    def test_controller_initialization(self, bitrate_controller):
        """Test bitrate controller initialization."""
        assert bitrate_controller.config is not None
        assert bitrate_controller.current_bitrate == bitrate_controller.config.default_audio_bitrate
        assert bitrate_controller.target_bitrate == bitrate_controller.config.default_audio_bitrate
    
    @pytest.mark.asyncio
    async def test_bitrate_increase(self, bitrate_controller):
        """Test bitrate increase logic."""
        initial_bitrate = bitrate_controller.current_bitrate
        
        # Good quality metrics should increase bitrate
        good_metrics = QualityMetrics(
            packet_loss_percent=0.1,
            latency_ms=50.0,
            jitter_ms=5.0
        )
        
        await bitrate_controller.adapt_bitrate(good_metrics)
        
        assert bitrate_controller.target_bitrate >= initial_bitrate
    
    @pytest.mark.asyncio
    async def test_bitrate_decrease(self, bitrate_controller):
        """Test bitrate decrease logic."""
        # Set higher initial bitrate
        bitrate_controller.current_bitrate = 64000
        bitrate_controller.target_bitrate = 64000
        
        # Poor quality metrics should decrease bitrate
        poor_metrics = QualityMetrics(
            packet_loss_percent=5.0,
            latency_ms=300.0,
            jitter_ms=80.0
        )
        
        await bitrate_controller.adapt_bitrate(poor_metrics)
        
        assert bitrate_controller.target_bitrate < 64000
    
    @pytest.mark.asyncio
    async def test_bitrate_limits(self, bitrate_controller):
        """Test bitrate limiting."""
        # Test minimum limit
        very_poor_metrics = QualityMetrics(
            packet_loss_percent=20.0,
            latency_ms=1000.0,
            jitter_ms=200.0
        )
        
        await bitrate_controller.adapt_bitrate(very_poor_metrics)
        assert bitrate_controller.target_bitrate >= bitrate_controller.config.min_audio_bitrate
        
        # Test maximum limit
        perfect_metrics = QualityMetrics(
            packet_loss_percent=0.0,
            latency_ms=10.0,
            jitter_ms=1.0
        )
        
        # Apply multiple times to try to exceed max
        for _ in range(10):
            await bitrate_controller.adapt_bitrate(perfect_metrics)
        
        assert bitrate_controller.target_bitrate <= bitrate_controller.config.max_audio_bitrate
    
    def test_adaptation_strategy_selection(self, bitrate_controller):
        """Test adaptation strategy selection."""
        # Test different network conditions
        cellular_condition = NetworkCondition.CELLULAR
        wifi_condition = NetworkCondition.WIFI
        ethernet_condition = NetworkCondition.ETHERNET
        
        cellular_strategy = bitrate_controller.get_adaptation_strategy(cellular_condition)
        wifi_strategy = bitrate_controller.get_adaptation_strategy(wifi_condition)
        ethernet_strategy = bitrate_controller.get_adaptation_strategy(ethernet_condition)
        
        # Cellular should be more conservative
        assert cellular_strategy.aggressiveness < wifi_strategy.aggressiveness
        assert wifi_strategy.aggressiveness < ethernet_strategy.aggressiveness


class TestQualityTrendAnalyzer:
    """Test quality trend analyzer."""
    
    @pytest.fixture
    def trend_analyzer(self):
        """Create quality trend analyzer for testing."""
        config = QualityConfig()
        analyzer = QualityTrendAnalyzer(config)
        return analyzer
    
    def test_analyzer_initialization(self, trend_analyzer):
        """Test trend analyzer initialization."""
        assert trend_analyzer.config is not None
        assert len(trend_analyzer.quality_history) == 0
        assert trend_analyzer.current_trend is None
    
    def test_quality_history_management(self, trend_analyzer):
        """Test quality history management."""
        # Add metrics to history
        for i in range(15):
            metrics = QualityMetrics(
                packet_loss_percent=float(i),
                latency_ms=100.0 + i * 10,
                timestamp=time.time() + i
            )
            trend_analyzer.add_quality_sample(metrics)
        
        # Should maintain window size
        assert len(trend_analyzer.quality_history) == trend_analyzer.config.quality_window_size
    
    def test_trend_detection_improving(self, trend_analyzer):
        """Test improving quality trend detection."""
        # Add improving quality samples
        for i in range(10):
            metrics = QualityMetrics(
                packet_loss_percent=5.0 - i * 0.5,  # Decreasing packet loss
                latency_ms=200.0 - i * 10,          # Decreasing latency
                timestamp=time.time() + i
            )
            trend_analyzer.add_quality_sample(metrics)
        
        trend = trend_analyzer.analyze_trend()
        assert trend.direction == "improving"
        assert trend.confidence > 0.7
    
    def test_trend_detection_degrading(self, trend_analyzer):
        """Test degrading quality trend detection."""
        # Add degrading quality samples
        for i in range(10):
            metrics = QualityMetrics(
                packet_loss_percent=0.5 + i * 0.3,  # Increasing packet loss
                latency_ms=80.0 + i * 15,           # Increasing latency
                timestamp=time.time() + i
            )
            trend_analyzer.add_quality_sample(metrics)
        
        trend = trend_analyzer.analyze_trend()
        assert trend.direction == "degrading"
        assert trend.confidence > 0.7
    
    def test_trend_detection_stable(self, trend_analyzer):
        """Test stable quality trend detection."""
        # Add stable quality samples
        for i in range(10):
            metrics = QualityMetrics(
                packet_loss_percent=1.0 + (i % 2) * 0.1,  # Small variations
                latency_ms=120.0 + (i % 3) * 5,           # Small variations
                timestamp=time.time() + i
            )
            trend_analyzer.add_quality_sample(metrics)
        
        trend = trend_analyzer.analyze_trend()
        assert trend.direction == "stable"
    
    def test_prediction_generation(self, trend_analyzer):
        """Test quality prediction generation."""
        # Add trend data
        for i in range(10):
            metrics = QualityMetrics(
                packet_loss_percent=1.0 + i * 0.2,
                latency_ms=100.0 + i * 10,
                timestamp=time.time() + i
            )
            trend_analyzer.add_quality_sample(metrics)
        
        # Generate prediction
        prediction = trend_analyzer.predict_future_quality(5)  # 5 seconds ahead
        
        assert prediction is not None
        assert hasattr(prediction, 'predicted_mos')
        assert hasattr(prediction, 'confidence')
        assert 0.0 <= prediction.confidence <= 1.0


class TestMediaQualityMonitor:
    """Test media quality monitor main functionality."""
    
    @pytest.fixture
    def quality_monitor(self, mock_quality_metrics):
        """Create media quality monitor for testing."""
        config = QualityConfig()
        monitor = MediaQualityMonitor(config)
        return monitor
    
    def test_monitor_initialization(self, quality_monitor):
        """Test quality monitor initialization."""
        assert quality_monitor.config is not None
        assert quality_monitor.is_monitoring == False
        assert quality_monitor.bitrate_controller is not None
        assert quality_monitor.mos_calculator is not None
        assert quality_monitor.trend_analyzer is not None
    
    @pytest.mark.asyncio
    async def test_monitoring_lifecycle(self, quality_monitor):
        """Test monitoring start/stop lifecycle."""
        # Start monitoring
        await quality_monitor.start_monitoring()
        assert quality_monitor.is_monitoring == True
        
        # Should have started background task
        assert quality_monitor._monitoring_task is not None
        
        # Stop monitoring
        await quality_monitor.stop_monitoring()
        assert quality_monitor.is_monitoring == False
    
    @pytest.mark.asyncio
    async def test_quality_measurement(self, quality_monitor):
        """Test quality measurement collection."""
        # Mock RTC stats
        mock_stats = {
            "inbound-rtp": {
                "packetsReceived": 1000,
                "packetsLost": 10,
                "jitter": 0.025,
                "bytesReceived": 128000
            },
            "candidate-pair": {
                "currentRoundTripTime": 0.120
            }
        }
        
        metrics = await quality_monitor.measure_quality(mock_stats)
        
        assert metrics is not None
        assert metrics.packet_loss_percent == 1.0  # 10/1000 * 100
        assert metrics.jitter_ms == 25.0  # 0.025 * 1000
        assert metrics.latency_ms == 120.0  # 0.120 * 1000
    
    @pytest.mark.asyncio
    async def test_adaptive_quality_control(self, quality_monitor):
        """Test adaptive quality control."""
        await quality_monitor.start_monitoring()
        
        # Simulate poor quality
        poor_metrics = QualityMetrics(
            packet_loss_percent=5.0,
            latency_ms=300.0,
            jitter_ms=80.0
        )
        
        # Process quality update
        await quality_monitor._process_quality_update(poor_metrics)
        
        # Should have adapted bitrate downward
        assert quality_monitor.bitrate_controller.target_bitrate < quality_monitor.config.default_audio_bitrate
        
        await quality_monitor.stop_monitoring()
    
    def test_callback_registration(self, quality_monitor):
        """Test quality event callback registration."""
        events_received = []
        
        def quality_callback(event):
            events_received.append(event)
        
        quality_monitor.on_quality_changed(quality_callback)
        quality_monitor.on_bitrate_changed(quality_callback)
        
        assert len(quality_monitor.quality_changed_callbacks) == 1
        assert len(quality_monitor.bitrate_changed_callbacks) == 1
    
    @pytest.mark.asyncio
    async def test_quality_event_triggering(self, quality_monitor):
        """Test quality event triggering."""
        events_received = []
        
        def event_handler(event):
            events_received.append(event)
        
        quality_monitor.on_quality_changed(event_handler)
        
        # Trigger quality change event
        quality_event = QualityEvent(
            event_type=QualityEventType.QUALITY_DEGRADED,
            metrics=QualityMetrics(packet_loss_percent=5.0),
            timestamp=time.time()
        )
        
        await quality_monitor._trigger_quality_changed(quality_event)
        
        assert len(events_received) == 1
        assert events_received[0] == quality_event
    
    def test_quality_report_generation(self, quality_monitor):
        """Test quality report generation."""
        # Add some quality history
        for i in range(5):
            metrics = QualityMetrics(
                packet_loss_percent=float(i),
                latency_ms=100.0 + i * 10,
                timestamp=time.time() + i
            )
            quality_monitor.trend_analyzer.add_quality_sample(metrics)
        
        report = quality_monitor.generate_quality_report()
        
        assert "current_quality" in report
        assert "trend_analysis" in report
        assert "bitrate_info" in report
        assert "recommendations" in report
    
    @pytest.mark.asyncio
    async def test_quality_optimization(self, quality_monitor):
        """Test automatic quality optimization."""
        await quality_monitor.start_monitoring()
        
        # Simulate network improvement
        good_metrics = QualityMetrics(
            packet_loss_percent=0.2,
            latency_ms=60.0,
            jitter_ms=8.0
        )
        
        await quality_monitor._process_quality_update(good_metrics)
        
        # Should have optimized settings
        optimizations = quality_monitor.get_current_optimizations()
        assert optimizations is not None
        assert "bitrate_adjustment" in optimizations
        
        await quality_monitor.stop_monitoring()


class TestQualityEvent:
    """Test quality event handling."""
    
    def test_quality_event_creation(self):
        """Test quality event creation."""
        metrics = QualityMetrics(packet_loss_percent=3.0)
        
        event = QualityEvent(
            event_type=QualityEventType.QUALITY_DEGRADED,
            metrics=metrics,
            previous_metrics=None,
            severity=Mock(value="medium"),
            message="Quality has degraded",
            timestamp=time.time()
        )
        
        assert event.event_type == QualityEventType.QUALITY_DEGRADED
        assert event.metrics == metrics
        assert event.severity.value == "medium"
        assert event.message == "Quality has degraded"
    
    def test_event_types(self):
        """Test quality event types."""
        assert QualityEventType.QUALITY_IMPROVED.value == "quality_improved"
        assert QualityEventType.QUALITY_DEGRADED.value == "quality_degraded"
        assert QualityEventType.BITRATE_CHANGED.value == "bitrate_changed"
        assert QualityEventType.NETWORK_CONDITION_CHANGED.value == "network_condition_changed"


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.fixture
    def error_monitor(self):
        """Create quality monitor for error testing."""
        config = QualityConfig()
        monitor = MediaQualityMonitor(config)
        return monitor
    
    @pytest.mark.asyncio
    async def test_invalid_stats_handling(self, error_monitor):
        """Test handling of invalid RTC stats."""
        # Test with None stats
        metrics = await error_monitor.measure_quality(None)
        assert metrics is not None  # Should return default metrics
        
        # Test with empty stats
        metrics = await error_monitor.measure_quality({})
        assert metrics is not None
        
        # Test with malformed stats
        bad_stats = {"invalid": "data"}
        metrics = await error_monitor.measure_quality(bad_stats)
        assert metrics is not None
    
    @pytest.mark.asyncio
    async def test_monitoring_error_recovery(self, error_monitor):
        """Test monitoring error recovery."""
        await error_monitor.start_monitoring()
        
        # Simulate monitoring error
        with patch.object(error_monitor, '_collect_quality_metrics', side_effect=Exception("Collection error")):
            # Should handle error gracefully and continue monitoring
            await asyncio.sleep(0.1)
            
            assert error_monitor.is_monitoring == True
        
        await error_monitor.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_callback_error_handling(self, error_monitor):
        """Test error handling in quality callbacks."""
        def failing_callback(event):
            raise Exception("Callback failed")
        
        error_monitor.on_quality_changed(failing_callback)
        
        # Should not crash when callback fails
        quality_event = QualityEvent(
            event_type=QualityEventType.QUALITY_IMPROVED,
            metrics=QualityMetrics(packet_loss_percent=0.5)
        )
        
        await error_monitor._trigger_quality_changed(quality_event)
        # Test continues if no exception raised


class TestPerformance:
    """Test performance characteristics."""
    
    @pytest.fixture
    def perf_monitor(self):
        """Create quality monitor for performance testing."""
        config = QualityConfig(monitoring_interval_ms=100)  # Fast monitoring
        monitor = MediaQualityMonitor(config)
        return monitor
    
    @pytest.mark.asyncio
    async def test_monitoring_performance(self, perf_monitor):
        """Test monitoring performance overhead."""
        await perf_monitor.start_monitoring()
        
        # Let it monitor for a short time
        await asyncio.sleep(0.5)
        
        # Should have collected multiple samples efficiently
        history_length = len(perf_monitor.trend_analyzer.quality_history)
        assert history_length > 0
        
        await perf_monitor.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_calculation_performance(self, perf_monitor):
        """Test calculation performance."""
        import time
        
        # Large stats object
        large_stats = {
            "inbound-rtp": {
                "packetsReceived": 10000,
                "packetsLost": 50,
                "jitter": 0.030,
                "bytesReceived": 1280000
            },
            "candidate-pair": {
                "currentRoundTripTime": 0.150
            }
        }
        
        start_time = time.time()
        metrics = await perf_monitor.measure_quality(large_stats)
        calc_time = time.time() - start_time
        
        # Should calculate quickly
        assert calc_time < 0.01  # Less than 10ms
        assert metrics is not None


# Integration test markers
pytestmark = pytest.mark.unit