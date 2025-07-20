"""
Unit tests for Multi-Region Manager.

Tests multi-region deployment, optimal server selection, load balancing,
health monitoring, and latency optimization across global regions.
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

from communication.multi_region_manager import (
    MultiRegionManager,
    RegionConfig,
    RegionInfo,
    RegionStatus,
    LoadBalancingStrategy,
    ServerSelectionCriteria,
    RegionHealthMonitor,
    LatencyOptimizer,
    RegionEvent,
    RegionEventType,
    ConnectionRequest,
    RegionPerformance
)


class TestRegionConfig:
    """Test region configuration."""
    
    def test_default_config(self):
        """Test default region configuration."""
        config = RegionConfig()
        
        assert config.health_check_interval_ms == 30000
        assert config.latency_measurement_interval_ms == 10000
        assert config.max_connection_attempts == 3
        assert config.connection_timeout_ms == 5000
        assert config.fallback_strategy == LoadBalancingStrategy.ROUND_ROBIN
        assert config.enable_automatic_failover == True
        assert config.enable_latency_optimization == True
        assert config.region_weight_adjustment_factor == 0.1
    
    def test_custom_config(self):
        """Test custom region configuration."""
        config = RegionConfig(
            health_check_interval_ms=15000,
            max_connection_attempts=5,
            fallback_strategy=LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN,
            enable_automatic_failover=False
        )
        
        assert config.health_check_interval_ms == 15000
        assert config.max_connection_attempts == 5
        assert config.fallback_strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN
        assert config.enable_automatic_failover == False


class TestRegionInfo:
    """Test region information handling."""
    
    def test_region_info_creation(self):
        """Test region info creation."""
        region = RegionInfo(
            region_id="us-east-1",
            region_name="US East",
            endpoint_url="wss://us-east-1.livekit.cloud",
            geographic_location="North America",
            average_latency_ms=45.0,
            current_connections=25,
            max_connections=1000,
            status=RegionStatus.AVAILABLE,
            health_score=95.0,
            weight=1.0,
            priority=1
        )
        
        assert region.region_id == "us-east-1"
        assert region.region_name == "US East"
        assert region.endpoint_url == "wss://us-east-1.livekit.cloud"
        assert region.geographic_location == "North America"
        assert region.average_latency_ms == 45.0
        assert region.current_connections == 25
        assert region.max_connections == 1000
        assert region.status == RegionStatus.AVAILABLE
        assert region.health_score == 95.0
    
    def test_region_capacity_check(self):
        """Test region capacity checking."""
        full_region = RegionInfo(
            region_id="test",
            current_connections=1000,
            max_connections=1000
        )
        
        available_region = RegionInfo(
            region_id="test",
            current_connections=500,
            max_connections=1000
        )
        
        assert full_region.has_capacity() == False
        assert available_region.has_capacity() == True
        assert available_region.get_capacity_percentage() == 50.0
    
    def test_region_suitability_scoring(self):
        """Test region suitability scoring."""
        excellent_region = RegionInfo(
            region_id="excellent",
            average_latency_ms=20.0,
            health_score=98.0,
            current_connections=100,
            max_connections=1000
        )
        
        poor_region = RegionInfo(
            region_id="poor",
            average_latency_ms=200.0,
            health_score=60.0,
            current_connections=950,
            max_connections=1000
        )
        
        excellent_score = excellent_region.calculate_suitability_score()
        poor_score = poor_region.calculate_suitability_score()
        
        assert excellent_score > poor_score
        assert 0.0 <= excellent_score <= 1.0
        assert 0.0 <= poor_score <= 1.0


class TestRegionStatus:
    """Test region status enumeration."""
    
    def test_region_statuses(self):
        """Test region status values."""
        assert RegionStatus.AVAILABLE.value == "available"
        assert RegionStatus.UNAVAILABLE.value == "unavailable"
        assert RegionStatus.DEGRADED.value == "degraded"
        assert RegionStatus.MAINTENANCE.value == "maintenance"
        assert RegionStatus.OVERLOADED.value == "overloaded"
    
    def test_status_priorities(self):
        """Test region status priorities."""
        assert RegionStatus.AVAILABLE.get_priority() > RegionStatus.DEGRADED.get_priority()
        assert RegionStatus.DEGRADED.get_priority() > RegionStatus.OVERLOADED.get_priority()
        assert RegionStatus.OVERLOADED.get_priority() > RegionStatus.UNAVAILABLE.get_priority()


class TestLoadBalancingStrategy:
    """Test load balancing strategies."""
    
    def test_strategy_types(self):
        """Test load balancing strategy types."""
        assert LoadBalancingStrategy.ROUND_ROBIN.value == "round_robin"
        assert LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN.value == "weighted_round_robin"
        assert LoadBalancingStrategy.LEAST_CONNECTIONS.value == "least_connections"
        assert LoadBalancingStrategy.LATENCY_BASED.value == "latency_based"
        assert LoadBalancingStrategy.GEOGRAPHIC.value == "geographic"
    
    def test_strategy_selection_logic(self):
        """Test strategy selection logic."""
        regions = [
            RegionInfo("us-east-1", current_connections=100, average_latency_ms=50.0, weight=1.0),
            RegionInfo("us-west-1", current_connections=200, average_latency_ms=30.0, weight=1.5),
            RegionInfo("eu-west-1", current_connections=150, average_latency_ms=80.0, weight=1.0)
        ]
        
        # Test least connections strategy
        least_conn_region = LoadBalancingStrategy.LEAST_CONNECTIONS.select_region(regions)
        assert least_conn_region.region_id == "us-east-1"  # Lowest connections
        
        # Test latency-based strategy
        latency_region = LoadBalancingStrategy.LATENCY_BASED.select_region(regions)
        assert latency_region.region_id == "us-west-1"  # Lowest latency


class TestServerSelectionCriteria:
    """Test server selection criteria."""
    
    def test_criteria_creation(self):
        """Test server selection criteria creation."""
        criteria = ServerSelectionCriteria(
            max_latency_ms=100.0,
            min_health_score=80.0,
            preferred_geographic_regions=["North America", "Europe"],
            required_features=["audio", "video"],
            performance_requirements={"min_bandwidth": 100000}
        )
        
        assert criteria.max_latency_ms == 100.0
        assert criteria.min_health_score == 80.0
        assert "North America" in criteria.preferred_geographic_regions
        assert "audio" in criteria.required_features
        assert criteria.performance_requirements["min_bandwidth"] == 100000
    
    def test_region_matching(self):
        """Test region matching against criteria."""
        criteria = ServerSelectionCriteria(
            max_latency_ms=100.0,
            min_health_score=85.0
        )
        
        good_region = RegionInfo(
            region_id="good",
            average_latency_ms=60.0,
            health_score=90.0
        )
        
        poor_region = RegionInfo(
            region_id="poor",
            average_latency_ms=150.0,
            health_score=70.0
        )
        
        assert criteria.matches_region(good_region) == True
        assert criteria.matches_region(poor_region) == False


class TestConnectionRequest:
    """Test connection request handling."""
    
    def test_request_creation(self):
        """Test connection request creation."""
        request = ConnectionRequest(
            client_id="client_123",
            client_ip="192.168.1.100",
            client_location="us",
            required_features=["audio", "dtmf"],
            performance_requirements={"max_latency": 150},
            preferred_regions=["us-east-1", "us-west-1"]
        )
        
        assert request.client_id == "client_123"
        assert request.client_ip == "192.168.1.100"
        assert request.client_location == "us"
        assert "audio" in request.required_features
        assert request.performance_requirements["max_latency"] == 150
        assert "us-east-1" in request.preferred_regions
    
    def test_request_priority_calculation(self):
        """Test connection request priority calculation."""
        high_priority_request = ConnectionRequest(
            client_id="vip_client",
            performance_requirements={"max_latency": 50},
            priority_level=1
        )
        
        low_priority_request = ConnectionRequest(
            client_id="regular_client",
            performance_requirements={"max_latency": 200},
            priority_level=5
        )
        
        assert high_priority_request.calculate_priority() > low_priority_request.calculate_priority()


class TestRegionPerformance:
    """Test region performance tracking."""
    
    def test_performance_creation(self):
        """Test region performance creation."""
        performance = RegionPerformance(
            region_id="us-east-1",
            average_latency_ms=55.0,
            packet_loss_percent=0.5,
            jitter_ms=10.0,
            connection_success_rate=0.98,
            current_load_percent=60.0,
            throughput_mbps=500.0,
            error_rate=0.02
        )
        
        assert performance.region_id == "us-east-1"
        assert performance.average_latency_ms == 55.0
        assert performance.packet_loss_percent == 0.5
        assert performance.connection_success_rate == 0.98
        assert performance.current_load_percent == 60.0
    
    def test_performance_score_calculation(self):
        """Test performance score calculation."""
        excellent_performance = RegionPerformance(
            region_id="excellent",
            average_latency_ms=20.0,
            packet_loss_percent=0.1,
            connection_success_rate=0.99,
            current_load_percent=30.0
        )
        
        poor_performance = RegionPerformance(
            region_id="poor",
            average_latency_ms=200.0,
            packet_loss_percent=5.0,
            connection_success_rate=0.80,
            current_load_percent=95.0
        )
        
        excellent_score = excellent_performance.calculate_overall_score()
        poor_score = poor_performance.calculate_overall_score()
        
        assert excellent_score > poor_score
        assert 0.0 <= excellent_score <= 1.0
        assert 0.0 <= poor_score <= 1.0


class TestRegionHealthMonitor:
    """Test region health monitoring."""
    
    @pytest.fixture
    def health_monitor(self, mock_region_info):
        """Create region health monitor for testing."""
        config = RegionConfig()
        monitor = RegionHealthMonitor(config)
        return monitor
    
    def test_monitor_initialization(self, health_monitor):
        """Test health monitor initialization."""
        assert health_monitor.config is not None
        assert health_monitor.is_monitoring == False
        assert len(health_monitor.monitored_regions) == 0
    
    @pytest.mark.asyncio
    async def test_region_registration(self, health_monitor):
        """Test region registration for monitoring."""
        region = RegionInfo(
            region_id="test-region",
            endpoint_url="wss://test.example.com"
        )
        
        await health_monitor.register_region(region)
        
        assert "test-region" in health_monitor.monitored_regions
        assert health_monitor.monitored_regions["test-region"] == region
    
    @pytest.mark.asyncio
    async def test_health_check_execution(self, health_monitor):
        """Test health check execution."""
        region = RegionInfo(
            region_id="test-region",
            endpoint_url="wss://test.example.com"
        )
        
        await health_monitor.register_region(region)
        
        # Mock health check
        with patch.object(health_monitor, '_perform_region_health_check', new_callable=AsyncMock) as mock_check:
            mock_check.return_value = {
                "status": RegionStatus.AVAILABLE,
                "health_score": 95.0,
                "response_time_ms": 45.0
            }
            
            result = await health_monitor.check_region_health("test-region")
            
            assert result["status"] == RegionStatus.AVAILABLE
            assert result["health_score"] == 95.0
            mock_check.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_monitoring_lifecycle(self, health_monitor):
        """Test health monitoring lifecycle."""
        # Add test region
        region = RegionInfo(region_id="test", endpoint_url="wss://test.com")
        await health_monitor.register_region(region)
        
        # Start monitoring
        await health_monitor.start_monitoring()
        assert health_monitor.is_monitoring == True
        
        # Stop monitoring
        await health_monitor.stop_monitoring()
        assert health_monitor.is_monitoring == False
    
    @pytest.mark.asyncio
    async def test_health_degradation_detection(self, health_monitor):
        """Test health degradation detection."""
        region = RegionInfo(
            region_id="degrading-region",
            health_score=95.0
        )
        
        await health_monitor.register_region(region)
        
        # Simulate health degradation
        degraded_health = {
            "status": RegionStatus.DEGRADED,
            "health_score": 60.0,
            "response_time_ms": 200.0
        }
        
        with patch.object(health_monitor, '_perform_region_health_check', new_callable=AsyncMock) as mock_check:
            mock_check.return_value = degraded_health
            
            await health_monitor.check_region_health("degrading-region")
            
            # Should detect degradation
            updated_region = health_monitor.monitored_regions["degrading-region"]
            assert updated_region.status == RegionStatus.DEGRADED
            assert updated_region.health_score == 60.0


class TestLatencyOptimizer:
    """Test latency optimization."""
    
    @pytest.fixture
    def latency_optimizer(self):
        """Create latency optimizer for testing."""
        config = RegionConfig()
        optimizer = LatencyOptimizer(config)
        return optimizer
    
    def test_optimizer_initialization(self, latency_optimizer):
        """Test latency optimizer initialization."""
        assert latency_optimizer.config is not None
        assert len(latency_optimizer.latency_measurements) == 0
    
    @pytest.mark.asyncio
    async def test_latency_measurement(self, latency_optimizer):
        """Test latency measurement to regions."""
        regions = [
            RegionInfo("us-east-1", endpoint_url="wss://us-east-1.example.com"),
            RegionInfo("us-west-1", endpoint_url="wss://us-west-1.example.com")
        ]
        
        # Mock latency measurement
        with patch.object(latency_optimizer, '_measure_region_latency', new_callable=AsyncMock) as mock_measure:
            mock_measure.side_effect = [45.0, 80.0]  # Different latencies
            
            measurements = await latency_optimizer.measure_latencies(regions)
            
            assert len(measurements) == 2
            assert measurements["us-east-1"] == 45.0
            assert measurements["us-west-1"] == 80.0
    
    @pytest.mark.asyncio
    async def test_optimal_region_selection(self, latency_optimizer):
        """Test optimal region selection based on latency."""
        regions = [
            RegionInfo("high-latency", average_latency_ms=150.0),
            RegionInfo("low-latency", average_latency_ms=30.0),
            RegionInfo("medium-latency", average_latency_ms=80.0)
        ]
        
        optimal_region = await latency_optimizer.select_optimal_region(regions)
        
        assert optimal_region.region_id == "low-latency"
    
    @pytest.mark.asyncio
    async def test_latency_based_routing(self, latency_optimizer):
        """Test latency-based routing optimization."""
        # Add latency measurements
        latency_optimizer.latency_measurements = {
            "us-east-1": 45.0,
            "us-west-1": 80.0,
            "eu-west-1": 120.0
        }
        
        client_request = ConnectionRequest(
            client_location="us-east",
            performance_requirements={"max_latency": 100}
        )
        
        suitable_regions = await latency_optimizer.get_suitable_regions(client_request)
        
        # Should only include regions meeting latency requirement
        assert len(suitable_regions) == 2  # us-east-1 and us-west-1
        assert all(region.region_id in ["us-east-1", "us-west-1"] for region in suitable_regions)


class TestMultiRegionManager:
    """Test multi-region manager functionality."""
    
    @pytest.fixture
    def region_manager(self, mock_region_info):
        """Create multi-region manager for testing."""
        config = RegionConfig()
        manager = MultiRegionManager(config)
        return manager
    
    def test_manager_initialization(self, region_manager):
        """Test multi-region manager initialization."""
        assert region_manager.config is not None
        assert len(region_manager.regions) == 0
        assert region_manager.health_monitor is not None
        assert region_manager.latency_optimizer is not None
        assert region_manager.current_strategy == region_manager.config.fallback_strategy
    
    @pytest.mark.asyncio
    async def test_region_management(self, region_manager):
        """Test region addition and removal."""
        region = RegionInfo(
            region_id="test-region",
            endpoint_url="wss://test.example.com"
        )
        
        # Add region
        await region_manager.add_region(region)
        assert "test-region" in region_manager.regions
        
        # Remove region
        await region_manager.remove_region("test-region")
        assert "test-region" not in region_manager.regions
    
    @pytest.mark.asyncio
    async def test_connection_request_handling(self, region_manager):
        """Test connection request handling."""
        # Add test regions
        regions = [
            RegionInfo("us-east-1", health_score=90.0, average_latency_ms=50.0),
            RegionInfo("us-west-1", health_score=85.0, average_latency_ms=80.0),
            RegionInfo("eu-west-1", health_score=95.0, average_latency_ms=120.0)
        ]
        
        for region in regions:
            await region_manager.add_region(region)
        
        # Create connection request
        request = ConnectionRequest(
            client_id="test_client",
            client_location="us",
            performance_requirements={"max_latency": 100}
        )
        
        # Select region for request
        selected_region = await region_manager.select_region_for_request(request)
        
        assert selected_region is not None
        assert selected_region.region_id in ["us-east-1", "us-west-1"]  # Should meet latency requirement
    
    @pytest.mark.asyncio
    async def test_load_balancing_strategies(self, region_manager):
        """Test different load balancing strategies."""
        # Add regions with different loads
        regions = [
            RegionInfo("low-load", current_connections=100, max_connections=1000),
            RegionInfo("high-load", current_connections=800, max_connections=1000),
            RegionInfo("medium-load", current_connections=500, max_connections=1000)
        ]
        
        for region in regions:
            await region_manager.add_region(region)
        
        # Test least connections strategy
        region_manager.set_load_balancing_strategy(LoadBalancingStrategy.LEAST_CONNECTIONS)
        request = ConnectionRequest(client_id="test")
        
        selected_region = await region_manager.select_region_for_request(request)
        assert selected_region.region_id == "low-load"
    
    @pytest.mark.asyncio
    async def test_automatic_failover(self, region_manager):
        """Test automatic failover functionality."""
        # Add regions
        primary_region = RegionInfo("primary", status=RegionStatus.AVAILABLE)
        backup_region = RegionInfo("backup", status=RegionStatus.AVAILABLE)
        
        await region_manager.add_region(primary_region)
        await region_manager.add_region(backup_region)
        
        # Simulate primary region failure
        region_manager.regions["primary"].status = RegionStatus.UNAVAILABLE
        
        request = ConnectionRequest(client_id="test")
        selected_region = await region_manager.select_region_for_request(request)
        
        # Should select backup region
        assert selected_region.region_id == "backup"
    
    def test_callback_registration(self, region_manager):
        """Test region event callback registration."""
        events_received = []
        
        def region_callback(event):
            events_received.append(event)
        
        region_manager.on_region_status_changed(region_callback)
        region_manager.on_region_selected(region_callback)
        
        assert len(region_manager.region_status_changed_callbacks) == 1
        assert len(region_manager.region_selected_callbacks) == 1
    
    @pytest.mark.asyncio
    async def test_region_performance_monitoring(self, region_manager):
        """Test region performance monitoring."""
        await region_manager.start_monitoring()
        
        # Add test region
        region = RegionInfo("performance-test", endpoint_url="wss://test.com")
        await region_manager.add_region(region)
        
        # Get performance report
        performance_report = await region_manager.get_performance_report()
        
        assert performance_report is not None
        assert "regions" in performance_report
        assert "overall_health" in performance_report
        
        await region_manager.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_geographic_optimization(self, region_manager):
        """Test geographic-based optimization."""
        # Add regions in different locations
        regions = [
            RegionInfo("us-east", geographic_location="North America"),
            RegionInfo("eu-west", geographic_location="Europe"),
            RegionInfo("ap-south", geographic_location="Asia Pacific")
        ]
        
        for region in regions:
            await region_manager.add_region(region)
        
        # Request from US client
        us_request = ConnectionRequest(
            client_location="us",
            preferred_geographic_regions=["North America"]
        )
        
        selected_region = await region_manager.select_region_for_request(us_request)
        assert selected_region.region_id == "us-east"


class TestRegionEvent:
    """Test region event handling."""
    
    def test_region_event_creation(self):
        """Test region event creation."""
        event = RegionEvent(
            event_type=RegionEventType.REGION_STATUS_CHANGED,
            region_id="us-east-1",
            previous_status=RegionStatus.AVAILABLE,
            current_status=RegionStatus.DEGRADED,
            message="Region health degraded",
            timestamp=time.time()
        )
        
        assert event.event_type == RegionEventType.REGION_STATUS_CHANGED
        assert event.region_id == "us-east-1"
        assert event.previous_status == RegionStatus.AVAILABLE
        assert event.current_status == RegionStatus.DEGRADED
    
    def test_event_types(self):
        """Test region event types."""
        assert RegionEventType.REGION_ADDED.value == "region_added"
        assert RegionEventType.REGION_REMOVED.value == "region_removed"
        assert RegionEventType.REGION_STATUS_CHANGED.value == "region_status_changed"
        assert RegionEventType.REGION_SELECTED.value == "region_selected"
        assert RegionEventType.FAILOVER_TRIGGERED.value == "failover_triggered"


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.fixture
    def error_manager(self):
        """Create multi-region manager for error testing."""
        config = RegionConfig()
        manager = MultiRegionManager(config)
        return manager
    
    @pytest.mark.asyncio
    async def test_no_available_regions_handling(self, error_manager):
        """Test handling when no regions are available."""
        request = ConnectionRequest(client_id="test")
        
        # No regions added
        selected_region = await error_manager.select_region_for_request(request)
        assert selected_region is None
    
    @pytest.mark.asyncio
    async def test_all_regions_unavailable_handling(self, error_manager):
        """Test handling when all regions are unavailable."""
        # Add unavailable regions
        unavailable_region = RegionInfo("unavailable", status=RegionStatus.UNAVAILABLE)
        await error_manager.add_region(unavailable_region)
        
        request = ConnectionRequest(client_id="test")
        selected_region = await error_manager.select_region_for_request(request)
        
        assert selected_region is None
    
    @pytest.mark.asyncio
    async def test_health_check_failure_handling(self, error_manager):
        """Test health check failure handling."""
        region = RegionInfo("test", endpoint_url="wss://invalid.example.com")
        await error_manager.add_region(region)
        
        # Mock health check failure
        with patch.object(error_manager.health_monitor, '_perform_region_health_check',
                          side_effect=Exception("Health check failed")):
            
            # Should handle gracefully
            result = await error_manager.health_monitor.check_region_health("test")
            assert result is not None
    
    @pytest.mark.asyncio
    async def test_callback_error_handling(self, error_manager):
        """Test error handling in region callbacks."""
        def failing_callback(event):
            raise Exception("Callback failed")
        
        error_manager.on_region_status_changed(failing_callback)
        
        # Should not crash when callback fails
        region_event = RegionEvent(
            event_type=RegionEventType.REGION_STATUS_CHANGED,
            region_id="test"
        )
        
        await error_manager._trigger_region_status_changed(region_event)
        # Test continues if no exception raised


class TestPerformance:
    """Test performance characteristics."""
    
    @pytest.fixture
    def perf_manager(self):
        """Create multi-region manager for performance testing."""
        config = RegionConfig()
        manager = MultiRegionManager(config)
        return manager
    
    @pytest.mark.asyncio
    async def test_region_selection_performance(self, perf_manager):
        """Test region selection performance with many regions."""
        import time
        
        # Add many regions
        for i in range(50):
            region = RegionInfo(
                region_id=f"region_{i}",
                health_score=80.0 + (i % 20),
                average_latency_ms=50.0 + (i % 30)
            )
            await perf_manager.add_region(region)
        
        request = ConnectionRequest(client_id="perf_test")
        
        start_time = time.time()
        selected_region = await perf_manager.select_region_for_request(request)
        selection_time = time.time() - start_time
        
        # Should select quickly even with many regions
        assert selection_time < 0.1  # Less than 100ms
        assert selected_region is not None
    
    @pytest.mark.asyncio
    async def test_health_monitoring_performance(self, perf_manager):
        """Test health monitoring performance."""
        await perf_manager.start_monitoring()
        
        # Add multiple regions
        for i in range(10):
            region = RegionInfo(f"region_{i}", endpoint_url=f"wss://region{i}.com")
            await perf_manager.add_region(region)
        
        # Let monitoring run briefly
        await asyncio.sleep(0.2)
        
        # Should handle multiple regions efficiently
        assert len(perf_manager.regions) == 10
        
        await perf_manager.stop_monitoring()


# Integration test markers
pytestmark = pytest.mark.unit