"""
Unit tests for Horizontal Auto-Scaling

This module contains comprehensive tests for the auto-scaling system,
including scaling policies, decision making, and Kubernetes integration.
"""

import asyncio
import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock

from .auto_scaler import (
    HorizontalAutoScaler,
    ScalingDirection,
    ScalingTrigger,
    ScalingMetrics,
    ScalingPolicy,
    ScalingEvent,
    PredictiveScaler,
    KubernetesScaler,
    create_auto_scaler,
    create_scaling_policy
)
from ...monitoring.performance_monitor import PerformanceMonitor
from ...workers.worker_orchestration import WorkerOrchestrator


class TestScalingMetrics:
    """Test cases for ScalingMetrics dataclass."""
    
    def test_metrics_creation(self):
        """Test metrics creation with defaults."""
        metrics = ScalingMetrics()
        
        assert metrics.cpu_utilization == 0.0
        assert metrics.memory_utilization == 0.0
        assert metrics.queue_depth == 0
        assert metrics.average_response_time == 0.0
        assert metrics.error_rate == 0.0
        assert metrics.throughput == 0.0
        assert metrics.active_workers == 0
        assert metrics.idle_workers == 0
        assert metrics.failed_workers == 0
        assert metrics.pending_jobs == 0
        assert isinstance(metrics.timestamp, datetime)
    
    def test_metrics_with_values(self):
        """Test metrics creation with custom values."""
        metrics = ScalingMetrics(
            cpu_utilization=75.0,
            memory_utilization=60.0,
            queue_depth=15,
            average_response_time=250.0,
            active_workers=5,
            pending_jobs=8
        )
        
        assert metrics.cpu_utilization == 75.0
        assert metrics.memory_utilization == 60.0
        assert metrics.queue_depth == 15
        assert metrics.average_response_time == 250.0
        assert metrics.active_workers == 5
        assert metrics.pending_jobs == 8


class TestScalingPolicy:
    """Test cases for ScalingPolicy dataclass."""
    
    def test_policy_creation(self):
        """Test policy creation with required fields."""
        policy = ScalingPolicy(
            trigger=ScalingTrigger.CPU_UTILIZATION,
            scale_up_threshold=70.0,
            scale_down_threshold=30.0
        )
        
        assert policy.trigger == ScalingTrigger.CPU_UTILIZATION
        assert policy.scale_up_threshold == 70.0
        assert policy.scale_down_threshold == 30.0
        assert policy.scale_up_cooldown == 300
        assert policy.scale_down_cooldown == 600
        assert policy.min_instances == 1
        assert policy.max_instances == 100
        assert policy.enabled is True
        assert policy.weight == 1.0
    
    def test_policy_with_custom_values(self):
        """Test policy creation with custom values."""
        policy = ScalingPolicy(
            trigger=ScalingTrigger.MEMORY_UTILIZATION,
            scale_up_threshold=80.0,
            scale_down_threshold=40.0,
            scale_up_cooldown=240,
            scale_down_cooldown=480,
            min_instances=2,
            max_instances=20,
            scale_up_step=2,
            scale_down_step=1,
            enabled=False,
            weight=0.8
        )
        
        assert policy.trigger == ScalingTrigger.MEMORY_UTILIZATION
        assert policy.scale_up_threshold == 80.0
        assert policy.scale_down_threshold == 40.0
        assert policy.scale_up_cooldown == 240
        assert policy.scale_down_cooldown == 480
        assert policy.min_instances == 2
        assert policy.max_instances == 20
        assert policy.scale_up_step == 2
        assert policy.scale_down_step == 1
        assert policy.enabled is False
        assert policy.weight == 0.8


class TestPredictiveScaler:
    """Test cases for PredictiveScaler."""
    
    def test_predictive_scaler_initialization(self):
        """Test predictive scaler initialization."""
        scaler = PredictiveScaler(history_window=120)
        
        assert scaler.history_window == 120
        assert len(scaler.metrics_history) == 0
        assert len(scaler.patterns) == 0
    
    def test_add_metrics(self):
        """Test adding metrics to predictive scaler."""
        scaler = PredictiveScaler()
        metrics = ScalingMetrics(cpu_utilization=50.0, throughput=100.0)
        
        scaler.add_metrics(metrics)
        
        assert len(scaler.metrics_history) == 1
        assert scaler.metrics_history[0][1] == metrics
    
    def test_predict_demand_no_data(self):
        """Test prediction with no historical data."""
        scaler = PredictiveScaler()
        
        prediction = scaler.predict_demand(15)
        
        assert prediction == {'cpu': 0, 'memory': 0, 'queue': 0, 'throughput': 0}
    
    def test_predict_demand_with_data(self):
        """Test prediction with historical data."""
        scaler = PredictiveScaler()
        
        # Add some metrics
        for i in range(5):
            metrics = ScalingMetrics(
                cpu_utilization=50.0 + i * 5,
                memory_utilization=40.0 + i * 3,
                throughput=100.0 + i * 10
            )
            scaler.add_metrics(metrics)
        
        prediction = scaler.predict_demand(15)
        
        # Should return latest metrics as fallback
        assert prediction['cpu'] > 0
        assert prediction['memory'] > 0
        assert prediction['throughput'] > 0


class TestKubernetesScaler:
    """Test cases for KubernetesScaler."""
    
    @pytest.fixture
    def mock_k8s_client(self):
        """Create mock Kubernetes client."""
        with patch('src.deployment.scaling.auto_scaler.client') as mock_client:
            mock_apps_v1 = Mock()
            mock_client.AppsV1Api.return_value = mock_apps_v1
            yield mock_apps_v1
    
    @pytest.fixture
    def k8s_scaler(self, mock_k8s_client):
        """Create KubernetesScaler with mocked client."""
        with patch('src.deployment.scaling.auto_scaler.K8S_AVAILABLE', True):
            with patch('src.deployment.scaling.auto_scaler.k8s_config') as mock_config:
                mock_config.load_incluster_config.side_effect = Exception()
                mock_config.load_kube_config.return_value = None
                mock_config.ConfigException = Exception
                
                scaler = KubernetesScaler("test-namespace", "test-deployment")
                scaler.k8s_apps_v1 = mock_k8s_client
                return scaler
    
    async def test_get_current_replicas(self, k8s_scaler, mock_k8s_client):
        """Test getting current replica count."""
        # Mock deployment response
        mock_deployment = Mock()
        mock_deployment.spec.replicas = 3
        mock_k8s_client.read_namespaced_deployment.return_value = mock_deployment
        
        replicas = await k8s_scaler.get_current_replicas()
        
        assert replicas == 3
        mock_k8s_client.read_namespaced_deployment.assert_called_once_with(
            name="test-deployment",
            namespace="test-namespace"
        )
    
    async def test_get_current_replicas_error(self, k8s_scaler, mock_k8s_client):
        """Test getting current replicas with error."""
        from kubernetes.client.rest import ApiException
        mock_k8s_client.read_namespaced_deployment.side_effect = ApiException()
        
        replicas = await k8s_scaler.get_current_replicas()
        
        assert replicas == 0
    
    async def test_scale_deployment(self, k8s_scaler, mock_k8s_client):
        """Test scaling deployment."""
        mock_k8s_client.patch_namespaced_deployment.return_value = None
        
        result = await k8s_scaler.scale_deployment(5)
        
        assert result is True
        mock_k8s_client.patch_namespaced_deployment.assert_called_once_with(
            name="test-deployment",
            namespace="test-namespace",
            body={"spec": {"replicas": 5}}
        )
    
    async def test_scale_deployment_error(self, k8s_scaler, mock_k8s_client):
        """Test scaling deployment with error."""
        from kubernetes.client.rest import ApiException
        mock_k8s_client.patch_namespaced_deployment.side_effect = ApiException()
        
        result = await k8s_scaler.scale_deployment(5)
        
        assert result is False
    
    async def test_get_deployment_status(self, k8s_scaler, mock_k8s_client):
        """Test getting deployment status."""
        # Mock deployment response
        mock_deployment = Mock()
        mock_deployment.spec.replicas = 3
        mock_deployment.status.ready_replicas = 2
        mock_deployment.status.available_replicas = 2
        mock_deployment.status.unavailable_replicas = 1
        mock_deployment.status.updated_replicas = 3
        mock_k8s_client.read_namespaced_deployment.return_value = mock_deployment
        
        status = await k8s_scaler.get_deployment_status()
        
        assert status["desired_replicas"] == 3
        assert status["ready_replicas"] == 2
        assert status["available_replicas"] == 2
        assert status["unavailable_replicas"] == 1
        assert status["updated_replicas"] == 3


class TestHorizontalAutoScaler:
    """Test cases for HorizontalAutoScaler."""
    
    @pytest.fixture
    def mock_orchestrator(self):
        """Create mock WorkerOrchestrator."""
        orchestrator = Mock(spec=WorkerOrchestrator)
        orchestrator.list_workers = AsyncMock(return_value=[])
        orchestrator.list_jobs = AsyncMock(return_value=[])
        orchestrator.job_queue = Mock()
        orchestrator.job_queue.qsize.return_value = 0
        return orchestrator
    
    @pytest.fixture
    def mock_performance_monitor(self):
        """Create mock PerformanceMonitor."""
        monitor = Mock(spec=PerformanceMonitor)
        monitor.record_metric = AsyncMock()
        return monitor
    
    @pytest.fixture
    async def auto_scaler(self, mock_orchestrator, mock_performance_monitor):
        """Create test auto-scaler instance."""
        config = {
            "check_interval": 1.0,  # Fast for testing
            "metrics_window": 60.0,
            "dry_run": True,  # Don't actually scale
            "prediction_enabled": False  # Disable for simpler testing
        }
        
        scaler = HorizontalAutoScaler(
            config=config,
            orchestrator=mock_orchestrator,
            performance_monitor=mock_performance_monitor
        )
        
        # Mock Kubernetes scaler
        scaler.k8s_scaler = Mock()
        scaler.k8s_scaler.get_current_replicas = AsyncMock(return_value=3)
        scaler.k8s_scaler.scale_deployment = AsyncMock(return_value=True)
        scaler.k8s_scaler.get_deployment_status = AsyncMock(return_value={})
        
        await scaler.start()
        yield scaler
        await scaler.stop()
    
    def test_auto_scaler_initialization(self):
        """Test auto-scaler initialization."""
        config = {"test": "value"}
        scaler = HorizontalAutoScaler(config)
        
        assert scaler.config == config
        assert scaler.running is False
        assert len(scaler.policies) > 0
        assert ScalingTrigger.CPU_UTILIZATION in scaler.policies
    
    async def test_auto_scaler_start_stop(self, auto_scaler):
        """Test auto-scaler start and stop."""
        assert auto_scaler.running is True
        assert len(auto_scaler._tasks) > 0
        
        await auto_scaler.stop()
        assert auto_scaler.running is False
    
    async def test_collect_metrics(self, auto_scaler, mock_orchestrator):
        """Test metrics collection."""
        # Mock worker data
        mock_worker = Mock()
        mock_worker.status.value = "idle"
        mock_worker.metrics = {
            "cpu_utilization": 50.0,
            "memory_utilization": 60.0,
            "avg_response_time": 200.0
        }
        mock_orchestrator.list_workers.return_value = [mock_worker]
        
        mock_job = Mock()
        mock_job.status.value = "queued"
        mock_orchestrator.list_jobs.return_value = [mock_job]
        
        auto_scaler.orchestrator.job_queue.qsize.return_value = 5
        
        metrics = await auto_scaler.collect_metrics()
        
        assert metrics.active_workers == 1
        assert metrics.idle_workers == 1
        assert metrics.pending_jobs == 1
        assert metrics.queue_depth == 5
        assert metrics.cpu_utilization == 50.0
        assert metrics.memory_utilization == 60.0
        assert metrics.average_response_time == 200.0
    
    async def test_make_scaling_decision_scale_up(self, auto_scaler):
        """Test scaling decision for scale-up."""
        # Add metrics history with high CPU
        for _ in range(5):
            metrics = ScalingMetrics(cpu_utilization=80.0, queue_depth=15)
            auto_scaler.metrics_history.append(metrics)
        
        current_metrics = ScalingMetrics(cpu_utilization=85.0, queue_depth=20)
        direction, amount, reason = await auto_scaler.make_scaling_decision(current_metrics)
        
        assert direction == ScalingDirection.UP
        assert amount > 0
        assert "cpu_utilization" in reason or "queue_depth" in reason
    
    async def test_make_scaling_decision_scale_down(self, auto_scaler):
        """Test scaling decision for scale-down."""
        # Add metrics history with low resource usage
        for _ in range(5):
            metrics = ScalingMetrics(cpu_utilization=20.0, queue_depth=1)
            auto_scaler.metrics_history.append(metrics)
        
        # Set last scale down to allow scaling
        auto_scaler.last_scale_down = datetime.now() - timedelta(seconds=700)
        
        current_metrics = ScalingMetrics(cpu_utilization=15.0, queue_depth=0)
        direction, amount, reason = await auto_scaler.make_scaling_decision(current_metrics)
        
        assert direction == ScalingDirection.DOWN
        assert amount > 0
    
    async def test_make_scaling_decision_no_action(self, auto_scaler):
        """Test scaling decision with no action needed."""
        # Add metrics history with moderate resource usage
        for _ in range(5):
            metrics = ScalingMetrics(cpu_utilization=50.0, queue_depth=5)
            auto_scaler.metrics_history.append(metrics)
        
        current_metrics = ScalingMetrics(cpu_utilization=50.0, queue_depth=5)
        direction, amount, reason = await auto_scaler.make_scaling_decision(current_metrics)
        
        assert direction == ScalingDirection.NONE
    
    async def test_make_scaling_decision_cooldown(self, auto_scaler):
        """Test scaling decision during cooldown period."""
        # Add metrics history with high CPU
        for _ in range(5):
            metrics = ScalingMetrics(cpu_utilization=80.0)
            auto_scaler.metrics_history.append(metrics)
        
        # Set recent scale-up
        auto_scaler.last_scale_up = datetime.now() - timedelta(seconds=60)
        
        current_metrics = ScalingMetrics(cpu_utilization=85.0)
        direction, amount, reason = await auto_scaler.make_scaling_decision(current_metrics)
        
        assert direction == ScalingDirection.NONE
        assert "cooldown" in reason
    
    async def test_execute_scaling_up(self, auto_scaler):
        """Test executing scale-up action."""
        auto_scaler.k8s_scaler.get_current_replicas.return_value = 3
        
        result = await auto_scaler.execute_scaling(ScalingDirection.UP, 2, "Test scale up")
        
        assert result is True
        assert auto_scaler.current_replicas == 5
        assert len(auto_scaler.scaling_events) == 1
        
        event = auto_scaler.scaling_events[0]
        assert event.direction == ScalingDirection.UP
        assert event.old_count == 3
        assert event.new_count == 5
    
    async def test_execute_scaling_down(self, auto_scaler):
        """Test executing scale-down action."""
        auto_scaler.k8s_scaler.get_current_replicas.return_value = 5
        
        result = await auto_scaler.execute_scaling(ScalingDirection.DOWN, 2, "Test scale down")
        
        assert result is True
        assert auto_scaler.current_replicas == 3
        assert len(auto_scaler.scaling_events) == 1
        
        event = auto_scaler.scaling_events[0]
        assert event.direction == ScalingDirection.DOWN
        assert event.old_count == 5
        assert event.new_count == 3
    
    async def test_execute_scaling_constraints(self, auto_scaler):
        """Test scaling with min/max constraints."""
        auto_scaler.k8s_scaler.get_current_replicas.return_value = 2
        
        # Try to scale down below minimum
        result = await auto_scaler.execute_scaling(ScalingDirection.DOWN, 5, "Test min constraint")
        
        # Should scale to minimum (1 or 2 depending on policy)
        assert auto_scaler.current_replicas >= 1
    
    async def test_get_scaling_status(self, auto_scaler):
        """Test getting scaling status."""
        # Add a scaling event
        event = ScalingEvent(
            timestamp=datetime.now(),
            direction=ScalingDirection.UP,
            trigger=ScalingTrigger.CPU_UTILIZATION,
            old_count=3,
            new_count=5,
            reason="Test event",
            metrics=ScalingMetrics()
        )
        auto_scaler.scaling_events.append(event)
        
        status = await auto_scaler.get_scaling_status()
        
        assert "current_replicas" in status
        assert "deployment_status" in status
        assert "policies" in status
        assert "recent_events" in status
        assert len(status["recent_events"]) == 1
        assert status["recent_events"][0]["direction"] == "up"
        assert status["recent_events"][0]["old_count"] == 3
        assert status["recent_events"][0]["new_count"] == 5
    
    async def test_update_policy(self, auto_scaler):
        """Test updating scaling policy."""
        original_threshold = auto_scaler.policies[ScalingTrigger.CPU_UTILIZATION].scale_up_threshold
        
        await auto_scaler.update_policy(
            ScalingTrigger.CPU_UTILIZATION,
            scale_up_threshold=90.0,
            enabled=False
        )
        
        updated_policy = auto_scaler.policies[ScalingTrigger.CPU_UTILIZATION]
        assert updated_policy.scale_up_threshold == 90.0
        assert updated_policy.enabled is False
        assert updated_policy.scale_up_threshold != original_threshold


class TestHelperFunctions:
    """Test helper functions."""
    
    def test_create_auto_scaler(self):
        """Test auto-scaler factory function."""
        config = {"test": "config"}
        orchestrator = Mock()
        
        scaler = create_auto_scaler(config, orchestrator)
        
        assert isinstance(scaler, HorizontalAutoScaler)
        assert scaler.config == config
        assert scaler.orchestrator == orchestrator
    
    def test_create_scaling_policy(self):
        """Test scaling policy creation helper."""
        policy_config = create_scaling_policy(
            "cpu_utilization",
            scale_up_threshold=80.0,
            scale_down_threshold=30.0,
            min_instances=2,
            max_instances=20
        )
        
        assert "cpu_utilization" in policy_config
        cpu_policy = policy_config["cpu_utilization"]
        assert cpu_policy["scale_up_threshold"] == 80.0
        assert cpu_policy["scale_down_threshold"] == 30.0
        assert cpu_policy["min_instances"] == 2
        assert cpu_policy["max_instances"] == 20


class TestScalingEvent:
    """Test ScalingEvent dataclass."""
    
    def test_scaling_event_creation(self):
        """Test scaling event creation."""
        metrics = ScalingMetrics(cpu_utilization=75.0)
        
        event = ScalingEvent(
            timestamp=datetime.now(),
            direction=ScalingDirection.UP,
            trigger=ScalingTrigger.CPU_UTILIZATION,
            old_count=3,
            new_count=5,
            reason="High CPU utilization",
            metrics=metrics
        )
        
        assert event.direction == ScalingDirection.UP
        assert event.trigger == ScalingTrigger.CPU_UTILIZATION
        assert event.old_count == 3
        assert event.new_count == 5
        assert event.reason == "High CPU utilization"
        assert event.metrics == metrics
        assert isinstance(event.timestamp, datetime)


if __name__ == "__main__":
    pytest.main([__file__])