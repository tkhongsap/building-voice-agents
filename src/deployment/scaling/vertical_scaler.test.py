"""
Unit tests for Vertical Auto-Scaling

This module contains comprehensive tests for the vertical scaling system,
including resource analysis, recommendations, and Kubernetes integration.
"""

import asyncio
import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock

from .vertical_scaler import (
    VerticalAutoScaler,
    ResourceAnalyzer,
    KubernetesVerticalScaler,
    ResourceMetrics,
    ResourceRecommendation,
    ResourceConstraints,
    ResourceType,
    ScalingDirection,
    create_vertical_scaler,
    create_resource_constraints
)
from ...monitoring.performance_monitor import PerformanceMonitor


class TestResourceMetrics:
    """Test cases for ResourceMetrics dataclass."""
    
    def test_metrics_creation(self):
        """Test resource metrics creation with defaults."""
        metrics = ResourceMetrics("worker-1")
        
        assert metrics.worker_id == "worker-1"
        assert metrics.cpu_usage == 0.0
        assert metrics.cpu_limit == 0.0
        assert metrics.memory_usage == 0.0
        assert metrics.memory_limit == 0.0
        assert metrics.cpu_utilization == 0.0
        assert metrics.memory_utilization == 0.0
        assert isinstance(metrics.timestamp, datetime)
    
    def test_metrics_with_values(self):
        """Test resource metrics creation with custom values."""
        metrics = ResourceMetrics(
            worker_id="worker-1",
            cpu_usage=1.5,
            cpu_limit=2.0,
            memory_usage=2 * 1024**3,  # 2GB
            memory_limit=4 * 1024**3,  # 4GB
            cpu_utilization=75.0,
            memory_utilization=50.0
        )
        
        assert metrics.worker_id == "worker-1"
        assert metrics.cpu_usage == 1.5
        assert metrics.cpu_limit == 2.0
        assert metrics.memory_usage == 2 * 1024**3
        assert metrics.memory_limit == 4 * 1024**3
        assert metrics.cpu_utilization == 75.0
        assert metrics.memory_utilization == 50.0


class TestResourceRecommendation:
    """Test cases for ResourceRecommendation dataclass."""
    
    def test_recommendation_creation(self):
        """Test resource recommendation creation."""
        recommendation = ResourceRecommendation(
            worker_id="worker-1",
            resource_type=ResourceType.CPU,
            current_value=2.0,
            recommended_value=3.0,
            direction=ScalingDirection.UP,
            confidence=0.8,
            reason="High CPU utilization"
        )
        
        assert recommendation.worker_id == "worker-1"
        assert recommendation.resource_type == ResourceType.CPU
        assert recommendation.current_value == 2.0
        assert recommendation.recommended_value == 3.0
        assert recommendation.direction == ScalingDirection.UP
        assert recommendation.confidence == 0.8
        assert recommendation.reason == "High CPU utilization"
        assert recommendation.potential_savings == 0.0
        assert recommendation.performance_impact == "low"


class TestResourceConstraints:
    """Test cases for ResourceConstraints dataclass."""
    
    def test_constraints_creation(self):
        """Test resource constraints creation with defaults."""
        constraints = ResourceConstraints()
        
        assert constraints.min_cpu == 0.1
        assert constraints.max_cpu == 16.0
        assert constraints.min_memory == 128 * 1024 * 1024  # 128MB
        assert constraints.max_memory == 32 * 1024 * 1024 * 1024  # 32GB
        assert constraints.cpu_step == 0.1
        assert constraints.memory_step == 128 * 1024 * 1024  # 128MB
    
    def test_constraints_with_custom_values(self):
        """Test resource constraints creation with custom values."""
        constraints = ResourceConstraints(
            min_cpu=0.2,
            max_cpu=8.0,
            min_memory=256 * 1024 * 1024,  # 256MB
            max_memory=16 * 1024 * 1024 * 1024,  # 16GB
            cpu_step=0.2,
            memory_step=256 * 1024 * 1024
        )
        
        assert constraints.min_cpu == 0.2
        assert constraints.max_cpu == 8.0
        assert constraints.min_memory == 256 * 1024 * 1024
        assert constraints.max_memory == 16 * 1024 * 1024 * 1024
        assert constraints.cpu_step == 0.2
        assert constraints.memory_step == 256 * 1024 * 1024


class TestResourceAnalyzer:
    """Test cases for ResourceAnalyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create a test resource analyzer."""
        return ResourceAnalyzer(analysis_window=60)  # 1 hour for testing
    
    def create_test_metrics(self, worker_id: str, cpu_util: float, memory_util: float) -> ResourceMetrics:
        """Create test metrics with specified utilizations."""
        return ResourceMetrics(
            worker_id=worker_id,
            cpu_usage=cpu_util / 100.0 * 2.0,  # Assuming 2.0 CPU limit
            cpu_limit=2.0,
            memory_usage=memory_util / 100.0 * 4 * 1024**3,  # Assuming 4GB limit
            memory_limit=4 * 1024**3,
            cpu_utilization=cpu_util,
            memory_utilization=memory_util
        )
    
    def test_analyzer_initialization(self, analyzer):
        """Test resource analyzer initialization."""
        assert analyzer.analysis_window == 60
        assert len(analyzer.metrics_history) == 0
        assert isinstance(analyzer.constraints, ResourceConstraints)
    
    def test_add_metrics(self, analyzer):
        """Test adding metrics to analyzer."""
        metrics = self.create_test_metrics("worker-1", 50.0, 60.0)
        analyzer.add_metrics(metrics)
        
        assert "worker-1" in analyzer.metrics_history
        assert len(analyzer.metrics_history["worker-1"]) == 1
        assert analyzer.metrics_history["worker-1"][0] == metrics
    
    def test_add_multiple_metrics(self, analyzer):
        """Test adding multiple metrics for same worker."""
        worker_id = "worker-1"
        
        for i in range(5):
            metrics = self.create_test_metrics(worker_id, 50.0 + i * 10, 60.0)
            analyzer.add_metrics(metrics)
        
        assert len(analyzer.metrics_history[worker_id]) == 5
    
    def test_analyze_cpu_usage_scale_up(self, analyzer):
        """Test CPU analysis recommending scale-up."""
        worker_id = "worker-1"
        
        # Add metrics with high CPU utilization
        for _ in range(15):
            metrics = self.create_test_metrics(worker_id, 85.0, 50.0)
            analyzer.add_metrics(metrics)
        
        recommendation = analyzer.analyze_cpu_usage(worker_id)
        
        assert recommendation is not None
        assert recommendation.resource_type == ResourceType.CPU
        assert recommendation.direction == ScalingDirection.UP
        assert recommendation.recommended_value > recommendation.current_value
        assert "High CPU utilization" in recommendation.reason
    
    def test_analyze_cpu_usage_scale_down(self, analyzer):
        """Test CPU analysis recommending scale-down."""
        worker_id = "worker-1"
        
        # Add metrics with low CPU utilization
        for _ in range(15):
            metrics = self.create_test_metrics(worker_id, 20.0, 50.0)
            analyzer.add_metrics(metrics)
        
        recommendation = analyzer.analyze_cpu_usage(worker_id)
        
        assert recommendation is not None
        assert recommendation.resource_type == ResourceType.CPU
        assert recommendation.direction == ScalingDirection.DOWN
        assert recommendation.recommended_value < recommendation.current_value
        assert "Low CPU utilization" in recommendation.reason
        assert recommendation.potential_savings > 0
    
    def test_analyze_cpu_usage_no_recommendation(self, analyzer):
        """Test CPU analysis with no recommendation needed."""
        worker_id = "worker-1"
        
        # Add metrics with moderate CPU utilization
        for _ in range(15):
            metrics = self.create_test_metrics(worker_id, 50.0, 50.0)
            analyzer.add_metrics(metrics)
        
        recommendation = analyzer.analyze_cpu_usage(worker_id)
        
        assert recommendation is None
    
    def test_analyze_cpu_usage_insufficient_data(self, analyzer):
        """Test CPU analysis with insufficient data."""
        worker_id = "worker-1"
        
        # Add only a few metrics
        for _ in range(3):
            metrics = self.create_test_metrics(worker_id, 80.0, 50.0)
            analyzer.add_metrics(metrics)
        
        recommendation = analyzer.analyze_cpu_usage(worker_id)
        
        assert recommendation is None
    
    def test_analyze_memory_usage_scale_up(self, analyzer):
        """Test memory analysis recommending scale-up."""
        worker_id = "worker-1"
        
        # Add metrics with high memory utilization
        for _ in range(15):
            metrics = self.create_test_metrics(worker_id, 50.0, 90.0)
            analyzer.add_metrics(metrics)
        
        recommendation = analyzer.analyze_memory_usage(worker_id)
        
        assert recommendation is not None
        assert recommendation.resource_type == ResourceType.MEMORY
        assert recommendation.direction == ScalingDirection.UP
        assert recommendation.recommended_value > recommendation.current_value
        assert "High memory utilization" in recommendation.reason
    
    def test_analyze_memory_usage_scale_down(self, analyzer):
        """Test memory analysis recommending scale-down."""
        worker_id = "worker-1"
        
        # Add metrics with low memory utilization
        for _ in range(15):
            metrics = self.create_test_metrics(worker_id, 50.0, 30.0)
            analyzer.add_metrics(metrics)
        
        recommendation = analyzer.analyze_memory_usage(worker_id)
        
        assert recommendation is not None
        assert recommendation.resource_type == ResourceType.MEMORY
        assert recommendation.direction == ScalingDirection.DOWN
        assert recommendation.recommended_value < recommendation.current_value
        assert "Low memory utilization" in recommendation.reason
        assert recommendation.potential_savings > 0
    
    def test_get_recommendations(self, analyzer):
        """Test getting all recommendations for a worker."""
        worker_id = "worker-1"
        
        # Add metrics that should trigger both CPU and memory recommendations
        for _ in range(15):
            metrics = self.create_test_metrics(worker_id, 85.0, 90.0)  # High utilization
            analyzer.add_metrics(metrics)
        
        recommendations = analyzer.get_recommendations(worker_id)
        
        # Should have both CPU and memory recommendations
        assert len(recommendations) == 2
        
        resource_types = {r.resource_type for r in recommendations}
        assert ResourceType.CPU in resource_types
        assert ResourceType.MEMORY in resource_types
        
        # Both should be scale-up recommendations
        for rec in recommendations:
            assert rec.direction == ScalingDirection.UP
    
    def test_percentile_calculation(self, analyzer):
        """Test percentile calculation method."""
        data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        
        p50 = analyzer._percentile(data, 50)
        p95 = analyzer._percentile(data, 95)
        p99 = analyzer._percentile(data, 99)
        
        assert p50 == 50
        assert p95 == 90
        assert p99 == 100
    
    def test_trend_calculation(self, analyzer):
        """Test trend calculation method."""
        # Increasing trend
        increasing_data = [1, 2, 3, 4, 5]
        trend_up = analyzer._calculate_trend(increasing_data)
        assert trend_up > 0
        
        # Decreasing trend
        decreasing_data = [5, 4, 3, 2, 1]
        trend_down = analyzer._calculate_trend(decreasing_data)
        assert trend_down < 0
        
        # Flat trend
        flat_data = [3, 3, 3, 3, 3]
        trend_flat = analyzer._calculate_trend(flat_data)
        assert abs(trend_flat) < 0.1


class TestKubernetesVerticalScaler:
    """Test cases for KubernetesVerticalScaler."""
    
    @pytest.fixture
    def mock_k8s_clients(self):
        """Create mock Kubernetes clients."""
        with patch('src.deployment.scaling.vertical_scaler.client') as mock_client:
            mock_apps_v1 = Mock()
            mock_core_v1 = Mock()
            mock_client.AppsV1Api.return_value = mock_apps_v1
            mock_client.CoreV1Api.return_value = mock_core_v1
            yield mock_apps_v1, mock_core_v1
    
    @pytest.fixture
    def k8s_scaler(self, mock_k8s_clients):
        """Create KubernetesVerticalScaler with mocked clients."""
        mock_apps_v1, mock_core_v1 = mock_k8s_clients
        
        with patch('src.deployment.scaling.vertical_scaler.K8S_AVAILABLE', True):
            with patch('src.deployment.scaling.vertical_scaler.k8s_config') as mock_config:
                mock_config.load_incluster_config.side_effect = Exception()
                mock_config.load_kube_config.return_value = None
                mock_config.ConfigException = Exception
                
                scaler = KubernetesVerticalScaler("test-namespace")
                scaler.k8s_apps_v1 = mock_apps_v1
                scaler.k8s_core_v1 = mock_core_v1
                return scaler
    
    def test_cpu_to_k8s_format(self, k8s_scaler):
        """Test CPU value conversion to Kubernetes format."""
        assert k8s_scaler._cpu_to_k8s_format(2.0) == "2.0"
        assert k8s_scaler._cpu_to_k8s_format(1.5) == "1.5"
        assert k8s_scaler._cpu_to_k8s_format(0.5) == "500m"
        assert k8s_scaler._cpu_to_k8s_format(0.1) == "100m"
    
    def test_memory_to_k8s_format(self, k8s_scaler):
        """Test memory value conversion to Kubernetes format."""
        assert k8s_scaler._memory_to_k8s_format(4 * 1024**3) == "4Gi"  # 4GB
        assert k8s_scaler._memory_to_k8s_format(512 * 1024**2) == "512Mi"  # 512MB
        assert k8s_scaler._memory_to_k8s_format(128 * 1024) == "128Ki"  # 128KB
    
    async def test_get_pod_resources(self, k8s_scaler, mock_k8s_clients):
        """Test getting pod resource configuration."""
        mock_apps_v1, mock_core_v1 = mock_k8s_clients
        
        # Mock pod response
        mock_pod = Mock()
        mock_container = Mock()
        mock_resources = Mock()
        
        mock_resources.limits = {"cpu": "2", "memory": "4Gi"}
        mock_resources.requests = {"cpu": "1", "memory": "2Gi"}
        mock_container.resources = mock_resources
        mock_pod.spec.containers = [mock_container]
        
        mock_core_v1.read_namespaced_pod.return_value = mock_pod
        
        resources = await k8s_scaler.get_pod_resources("test-pod")
        
        assert "limits" in resources
        assert "requests" in resources
        assert resources["limits"]["cpu"] == "2"
        assert resources["limits"]["memory"] == "4Gi"
        assert resources["requests"]["cpu"] == "1"
        assert resources["requests"]["memory"] == "2Gi"
    
    async def test_update_deployment_resources(self, k8s_scaler, mock_k8s_clients):
        """Test updating deployment resource configuration."""
        mock_apps_v1, mock_core_v1 = mock_k8s_clients
        
        # Mock deployment response
        mock_deployment = Mock()
        mock_container = Mock()
        mock_resources = Mock()
        
        mock_resources.limits = {}
        mock_resources.requests = {}
        mock_container.resources = mock_resources
        mock_deployment.spec.template.spec.containers = [mock_container]
        
        mock_apps_v1.read_namespaced_deployment.return_value = mock_deployment
        mock_apps_v1.patch_namespaced_deployment.return_value = None
        
        result = await k8s_scaler.update_deployment_resources(
            "test-deployment",
            cpu_limit=3.0,
            memory_limit=8 * 1024**3,
            cpu_request=2.0,
            memory_request=4 * 1024**3
        )
        
        assert result is True
        
        # Verify the patch was called
        mock_apps_v1.patch_namespaced_deployment.assert_called_once()
        
        # Check that resource values were set correctly
        assert mock_container.resources.limits["cpu"] == "3.0"
        assert mock_container.resources.limits["memory"] == "8Gi"
        assert mock_container.resources.requests["cpu"] == "2.0"
        assert mock_container.resources.requests["memory"] == "4Gi"


class TestVerticalAutoScaler:
    """Test cases for VerticalAutoScaler."""
    
    @pytest.fixture
    def mock_performance_monitor(self):
        """Create mock PerformanceMonitor."""
        monitor = Mock(spec=PerformanceMonitor)
        monitor.record_metric = AsyncMock()
        return monitor
    
    @pytest.fixture
    async def vertical_scaler(self, mock_performance_monitor):
        """Create test vertical auto-scaler instance."""
        config = {
            "check_interval": 1.0,  # Fast for testing
            "analysis_window": 60,  # 1 hour for testing
            "min_confidence": 0.6,
            "dry_run": True,  # Don't actually scale
            "enable_cpu_scaling": True,
            "enable_memory_scaling": True
        }
        
        scaler = VerticalAutoScaler(
            config=config,
            performance_monitor=mock_performance_monitor
        )
        
        # Mock Kubernetes scaler
        scaler.k8s_scaler = Mock()
        scaler.k8s_scaler.get_pod_resources = AsyncMock(return_value={})
        scaler.k8s_scaler.update_deployment_resources = AsyncMock(return_value=True)
        
        await scaler.start()
        yield scaler
        await scaler.stop()
    
    def test_vertical_scaler_initialization(self):
        """Test vertical scaler initialization."""
        config = {"test": "value"}
        scaler = VerticalAutoScaler(config)
        
        assert scaler.config == config
        assert scaler.running is False
        assert isinstance(scaler.analyzer, ResourceAnalyzer)
        assert isinstance(scaler.k8s_scaler, KubernetesVerticalScaler)
    
    async def test_vertical_scaler_start_stop(self, vertical_scaler):
        """Test vertical scaler start and stop."""
        assert vertical_scaler.running is True
        assert len(vertical_scaler._tasks) > 0
        
        await vertical_scaler.stop()
        assert vertical_scaler.running is False
    
    async def test_collect_worker_metrics(self, vertical_scaler):
        """Test collecting worker metrics."""
        with patch('random.random', return_value=0.5):  # Fixed randomness for testing
            metrics = await vertical_scaler.collect_worker_metrics("worker-1", "pod-1")
        
        assert metrics is not None
        assert metrics.worker_id == "worker-1"
        assert metrics.cpu_limit > 0
        assert metrics.memory_limit > 0
        assert metrics.cpu_utilization >= 0
        assert metrics.memory_utilization >= 0
    
    async def test_apply_recommendation_high_confidence(self, vertical_scaler):
        """Test applying recommendation with high confidence."""
        recommendation = ResourceRecommendation(
            worker_id="worker-1",
            resource_type=ResourceType.CPU,
            current_value=2.0,
            recommended_value=3.0,
            direction=ScalingDirection.UP,
            confidence=0.9,
            reason="High CPU utilization"
        )
        
        result = await vertical_scaler.apply_recommendation(recommendation)
        
        assert result is True
        assert recommendation in vertical_scaler.applied_recommendations
    
    async def test_apply_recommendation_low_confidence(self, vertical_scaler):
        """Test skipping recommendation with low confidence."""
        recommendation = ResourceRecommendation(
            worker_id="worker-1",
            resource_type=ResourceType.CPU,
            current_value=2.0,
            recommended_value=3.0,
            direction=ScalingDirection.UP,
            confidence=0.3,  # Below minimum confidence
            reason="Uncertain CPU pattern"
        )
        
        result = await vertical_scaler.apply_recommendation(recommendation)
        
        assert result is False
        assert recommendation not in vertical_scaler.applied_recommendations
    
    async def test_apply_cpu_recommendation(self, vertical_scaler):
        """Test applying CPU scaling recommendation."""
        recommendation = ResourceRecommendation(
            worker_id="worker-1",
            resource_type=ResourceType.CPU,
            current_value=2.0,
            recommended_value=3.0,
            direction=ScalingDirection.UP,
            confidence=0.8,
            reason="High CPU utilization"
        )
        
        await vertical_scaler.apply_recommendation(recommendation)
        
        # Verify Kubernetes scaler was called for CPU
        vertical_scaler.k8s_scaler.update_deployment_resources.assert_called_once_with(
            "voice-agent-worker-1",
            cpu_limit=3.0,
            cpu_request=2.4  # 80% of limit
        )
    
    async def test_apply_memory_recommendation(self, vertical_scaler):
        """Test applying memory scaling recommendation."""
        memory_4gb = 4 * 1024**3
        memory_6gb = 6 * 1024**3
        
        recommendation = ResourceRecommendation(
            worker_id="worker-1",
            resource_type=ResourceType.MEMORY,
            current_value=memory_4gb,
            recommended_value=memory_6gb,
            direction=ScalingDirection.UP,
            confidence=0.8,
            reason="High memory utilization"
        )
        
        await vertical_scaler.apply_recommendation(recommendation)
        
        # Verify Kubernetes scaler was called for memory
        vertical_scaler.k8s_scaler.update_deployment_resources.assert_called_once_with(
            "voice-agent-worker-1",
            memory_limit=memory_6gb,
            memory_request=memory_6gb * 0.8  # 80% of limit
        )
    
    async def test_get_scaling_status(self, vertical_scaler):
        """Test getting scaling status."""
        # Add some recommendations
        recommendation = ResourceRecommendation(
            worker_id="worker-1",
            resource_type=ResourceType.CPU,
            current_value=2.0,
            recommended_value=3.0,
            direction=ScalingDirection.UP,
            confidence=0.8,
            reason="High CPU utilization",
            potential_savings=0.05
        )
        
        vertical_scaler.recommendations_history.append(recommendation)
        vertical_scaler.applied_recommendations.append(recommendation)
        
        status = await vertical_scaler.get_scaling_status()
        
        assert status["enabled"] is True
        assert status["cpu_scaling_enabled"] is True
        assert status["memory_scaling_enabled"] is True
        assert status["dry_run"] is True
        assert status["min_confidence"] == 0.6
        assert status["total_applied_recommendations"] == 1
        assert status["total_potential_savings"] == 0.05
        assert len(status["recent_recommendations"]) == 1
        assert len(status["recent_applied"]) == 1
        
        # Check recommendation details
        rec_data = status["recent_recommendations"][0]
        assert rec_data["worker_id"] == "worker-1"
        assert rec_data["resource_type"] == "cpu"
        assert rec_data["direction"] == "up"
        assert rec_data["confidence"] == 0.8


class TestHelperFunctions:
    """Test helper functions."""
    
    def test_create_vertical_scaler(self):
        """Test vertical scaler factory function."""
        config = {"test": "config"}
        
        scaler = create_vertical_scaler(config)
        
        assert isinstance(scaler, VerticalAutoScaler)
        assert scaler.config == config
    
    def test_create_resource_constraints(self):
        """Test resource constraints creation helper."""
        constraints = create_resource_constraints(
            min_cpu=0.2,
            max_cpu=8.0,
            min_memory_gb=0.5,
            max_memory_gb=16.0
        )
        
        assert constraints.min_cpu == 0.2
        assert constraints.max_cpu == 8.0
        assert constraints.min_memory == 0.5 * 1024**3
        assert constraints.max_memory == 16.0 * 1024**3


if __name__ == "__main__":
    pytest.main([__file__])