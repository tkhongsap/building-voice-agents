"""
Unit tests for Worker Load Balancer

This module contains comprehensive tests for the load balancing system,
including different algorithms, health monitoring, and routing rules.
"""

import asyncio
import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from .load_balancer import (
    WorkerLoadBalancer,
    LoadBalancingAlgorithm,
    HealthStatus,
    LoadBalancingMetrics,
    WorkerHealth,
    RoundRobinStrategy,
    LeastConnectionsStrategy,
    WeightedRoundRobinStrategy,
    LeastResponseTimeStrategy,
    ResourceBasedStrategy,
    create_load_balancer,
    geo_routing_rule,
    capability_routing_rule,
    priority_routing_rule
)
from .worker_orchestration import WorkerInfo, WorkerStatus
from ..monitoring.performance_monitor import PerformanceMonitor


class TestLoadBalancingStrategies:
    """Test cases for load balancing strategies."""
    
    def create_test_workers(self, count: int = 3) -> List[WorkerHealth]:
        """Create test worker health objects."""
        workers = []
        for i in range(count):
            worker = WorkerHealth(
                worker_id=f"worker-{i}",
                status=HealthStatus.HEALTHY,
                metrics=LoadBalancingMetrics(
                    active_connections=i,
                    average_response_time=100.0 + (i * 50),
                    cpu_utilization=20.0 + (i * 20),
                    memory_utilization=30.0 + (i * 15)
                ),
                weight=1.0 + (i * 0.5)
            )
            workers.append(worker)
        return workers
    
    async def test_round_robin_strategy(self):
        """Test round-robin load balancing strategy."""
        strategy = RoundRobinStrategy()
        workers = self.create_test_workers(3)
        
        # Test multiple selections
        selections = []
        for _ in range(6):
            selected = await strategy.select_worker(workers, {})
            selections.append(selected)
        
        # Should cycle through workers
        expected = ["worker-0", "worker-1", "worker-2", "worker-0", "worker-1", "worker-2"]
        assert selections == expected
    
    async def test_round_robin_with_unhealthy_workers(self):
        """Test round-robin with some unhealthy workers."""
        strategy = RoundRobinStrategy()
        workers = self.create_test_workers(3)
        
        # Mark one worker as unhealthy
        workers[1].status = HealthStatus.CRITICAL
        
        # Should only select healthy workers
        selections = []
        for _ in range(4):
            selected = await strategy.select_worker(workers, {})
            selections.append(selected)
        
        # Should only cycle through healthy workers
        expected = ["worker-0", "worker-2", "worker-0", "worker-2"]
        assert selections == expected
    
    async def test_least_connections_strategy(self):
        """Test least connections load balancing strategy."""
        strategy = LeastConnectionsStrategy()
        workers = self.create_test_workers(3)
        
        # worker-0 has 0 connections, worker-1 has 1, worker-2 has 2
        selected = await strategy.select_worker(workers, {})
        assert selected == "worker-0"
        
        # Update connections and test again
        workers[0].metrics.active_connections = 5
        selected = await strategy.select_worker(workers, {})
        assert selected == "worker-1"
    
    async def test_weighted_round_robin_strategy(self):
        """Test weighted round-robin load balancing strategy."""
        strategy = WeightedRoundRobinStrategy()
        workers = self.create_test_workers(2)
        
        # Set different weights
        workers[0].weight = 3.0
        workers[1].weight = 1.0
        
        # Test multiple selections
        selections = []
        for _ in range(8):
            selected = await strategy.select_worker(workers, {})
            selections.append(selected)
        
        # Worker-0 should be selected more often due to higher weight
        worker_0_count = selections.count("worker-0")
        worker_1_count = selections.count("worker-1")
        assert worker_0_count > worker_1_count
    
    async def test_least_response_time_strategy(self):
        """Test least response time load balancing strategy."""
        strategy = LeastResponseTimeStrategy()
        workers = self.create_test_workers(3)
        
        # worker-0 has 100ms, worker-1 has 150ms, worker-2 has 200ms
        selected = await strategy.select_worker(workers, {})
        assert selected == "worker-0"
        
        # Update response times
        workers[0].metrics.average_response_time = 300.0
        selected = await strategy.select_worker(workers, {})
        assert selected == "worker-1"
    
    async def test_resource_based_strategy(self):
        """Test resource-based load balancing strategy."""
        strategy = ResourceBasedStrategy()
        workers = self.create_test_workers(3)
        
        # worker-0 has lowest resource usage
        selected = await strategy.select_worker(workers, {})
        assert selected == "worker-0"
        
        # Update resource usage
        workers[0].metrics.cpu_utilization = 90.0
        workers[0].metrics.memory_utilization = 85.0
        selected = await strategy.select_worker(workers, {})
        assert selected == "worker-1"
    
    async def test_strategy_with_no_workers(self):
        """Test strategies with no available workers."""
        strategies = [
            RoundRobinStrategy(),
            LeastConnectionsStrategy(),
            WeightedRoundRobinStrategy(),
            LeastResponseTimeStrategy(),
            ResourceBasedStrategy()
        ]
        
        for strategy in strategies:
            selected = await strategy.select_worker([], {})
            assert selected is None
    
    async def test_strategy_with_all_unhealthy_workers(self):
        """Test strategies with all unhealthy workers."""
        workers = self.create_test_workers(3)
        for worker in workers:
            worker.status = HealthStatus.CRITICAL
        
        strategies = [
            RoundRobinStrategy(),
            LeastConnectionsStrategy(),
            WeightedRoundRobinStrategy(),
            LeastResponseTimeStrategy(),
            ResourceBasedStrategy()
        ]
        
        for strategy in strategies:
            selected = await strategy.select_worker(workers, {})
            assert selected is None


class TestWorkerLoadBalancer:
    """Test cases for WorkerLoadBalancer."""
    
    @pytest.fixture
    async def load_balancer(self):
        """Create a test load balancer instance."""
        config = {
            "algorithm": "least_connections",
            "health_check_interval": 1.0,
            "health_timeout": 5.0,
            "failure_threshold": 2
        }
        lb = WorkerLoadBalancer(config)
        await lb.start()
        yield lb
        await lb.stop()
    
    @pytest.fixture
    def mock_performance_monitor(self):
        """Create a mock performance monitor."""
        monitor = Mock(spec=PerformanceMonitor)
        monitor.record_metric = AsyncMock()
        return monitor
    
    def create_test_worker_info(self, worker_id: str) -> WorkerInfo:
        """Create a test WorkerInfo object."""
        return WorkerInfo(
            worker_id=worker_id,
            status=WorkerStatus.IDLE,
            capabilities={"voice_processing", "transcription"}
        )
    
    async def test_load_balancer_initialization(self):
        """Test load balancer initialization."""
        config = {"algorithm": "round_robin"}
        lb = WorkerLoadBalancer(config)
        
        assert lb.config == config
        assert lb.algorithm == LoadBalancingAlgorithm.ROUND_ROBIN
        assert lb.running is False
        assert len(lb.worker_health) == 0
    
    async def test_load_balancer_start_stop(self, load_balancer):
        """Test load balancer start and stop."""
        assert load_balancer.running is True
        assert len(load_balancer._tasks) > 0
        
        await load_balancer.stop()
        assert load_balancer.running is False
    
    async def test_worker_registration(self, load_balancer):
        """Test worker registration with load balancer."""
        worker_id = "test-worker-1"
        capabilities = {"voice_processing"}
        
        await load_balancer.register_worker(worker_id, capabilities, weight=2.0)
        
        assert worker_id in load_balancer.worker_health
        health = load_balancer.worker_health[worker_id]
        assert health.worker_id == worker_id
        assert health.weight == 2.0
    
    async def test_worker_unregistration(self, load_balancer):
        """Test worker unregistration from load balancer."""
        worker_id = "test-worker-1"
        capabilities = {"voice_processing"}
        
        await load_balancer.register_worker(worker_id, capabilities)
        assert worker_id in load_balancer.worker_health
        
        await load_balancer.unregister_worker(worker_id)
        assert worker_id not in load_balancer.worker_health
    
    async def test_worker_selection(self, load_balancer):
        """Test worker selection with load balancer."""
        # Register workers
        await load_balancer.register_worker("worker-1", {"voice"})
        await load_balancer.register_worker("worker-2", {"voice"})
        
        # Create WorkerInfo objects
        worker_infos = [
            self.create_test_worker_info("worker-1"),
            self.create_test_worker_info("worker-2")
        ]
        
        # Select worker
        selected = await load_balancer.select_worker(worker_infos, {})
        assert selected in ["worker-1", "worker-2"]
    
    async def test_worker_selection_with_unregistered_workers(self, load_balancer):
        """Test worker selection with unregistered workers."""
        # Create WorkerInfo objects without registering
        worker_infos = [
            self.create_test_worker_info("worker-1"),
            self.create_test_worker_info("worker-2")
        ]
        
        # Should still work (auto-registration)
        selected = await load_balancer.select_worker(worker_infos, {})
        assert selected in ["worker-1", "worker-2"]
        
        # Workers should now be registered
        assert "worker-1" in load_balancer.worker_health
        assert "worker-2" in load_balancer.worker_health
    
    async def test_worker_metrics_update(self, load_balancer):
        """Test updating worker metrics."""
        worker_id = "test-worker"
        await load_balancer.register_worker(worker_id, {"voice"})
        
        # Update metrics
        metrics = LoadBalancingMetrics(
            active_connections=5,
            average_response_time=200.0,
            cpu_utilization=60.0,
            memory_utilization=70.0
        )
        
        await load_balancer.update_worker_metrics(worker_id, metrics)
        
        health = load_balancer.worker_health[worker_id]
        assert health.metrics.active_connections == 5
        assert health.metrics.average_response_time == 200.0
        assert health.metrics.cpu_utilization == 60.0
        assert health.metrics.memory_utilization == 70.0
    
    async def test_job_completion_reporting(self, load_balancer):
        """Test reporting job completion."""
        worker_id = "test-worker"
        await load_balancer.register_worker(worker_id, {"voice"})
        
        # Report successful job
        await load_balancer.report_job_completion(worker_id, True, 150.0)
        
        health = load_balancer.worker_health[worker_id]
        assert health.total_requests == 1
        assert health.successful_requests == 1
        assert health.consecutive_failures == 0
        assert health.metrics.average_response_time == 150.0
        assert health.metrics.error_rate == 0.0
    
    async def test_job_failure_reporting(self, load_balancer):
        """Test reporting job failures."""
        worker_id = "test-worker"
        await load_balancer.register_worker(worker_id, {"voice"})
        
        # Report failed jobs
        await load_balancer.report_job_completion(worker_id, False, 0.0)
        await load_balancer.report_job_completion(worker_id, False, 0.0)
        
        health = load_balancer.worker_health[worker_id]
        assert health.total_requests == 2
        assert health.successful_requests == 0
        assert health.consecutive_failures == 2
        assert health.metrics.error_rate == 1.0
        assert health.status == HealthStatus.CRITICAL  # Exceeds failure threshold
    
    async def test_algorithm_change(self, load_balancer):
        """Test changing load balancing algorithm."""
        original_algorithm = load_balancer.algorithm
        
        await load_balancer.set_algorithm(LoadBalancingAlgorithm.ROUND_ROBIN)
        
        assert load_balancer.algorithm == LoadBalancingAlgorithm.ROUND_ROBIN
        assert load_balancer.algorithm != original_algorithm
        assert isinstance(load_balancer.current_strategy, RoundRobinStrategy)
    
    async def test_routing_rules(self, load_balancer):
        """Test custom routing rules."""
        # Add a routing rule that filters workers by ID
        async def test_rule(workers, requirements):
            target_worker = requirements.get("target_worker")
            if target_worker:
                return [w for w in workers if w.worker_id == target_worker]
            return workers
        
        await load_balancer.add_routing_rule(test_rule)
        
        # Register workers
        await load_balancer.register_worker("worker-1", {"voice"})
        await load_balancer.register_worker("worker-2", {"voice"})
        
        worker_infos = [
            self.create_test_worker_info("worker-1"),
            self.create_test_worker_info("worker-2")
        ]
        
        # Test routing rule
        selected = await load_balancer.select_worker(
            worker_infos, 
            {"target_worker": "worker-2"}
        )
        assert selected == "worker-2"
    
    async def test_health_status_updates(self, load_balancer):
        """Test health status updates based on metrics."""
        worker_id = "test-worker"
        await load_balancer.register_worker(worker_id, {"voice"})
        
        # Update with high CPU utilization
        metrics = LoadBalancingMetrics(cpu_utilization=95.0)
        await load_balancer.update_worker_metrics(worker_id, metrics)
        
        health = load_balancer.worker_health[worker_id]
        assert health.status == HealthStatus.WARNING
        
        # Update with normal metrics
        metrics = LoadBalancingMetrics(cpu_utilization=30.0)
        await load_balancer.update_worker_metrics(worker_id, metrics)
        
        health = load_balancer.worker_health[worker_id]
        assert health.status == HealthStatus.HEALTHY
    
    async def test_get_worker_health(self, load_balancer):
        """Test getting worker health information."""
        worker_id = "test-worker"
        await load_balancer.register_worker(worker_id, {"voice"})
        
        health = await load_balancer.get_worker_health(worker_id)
        assert health is not None
        assert health.worker_id == worker_id
        
        # Test non-existent worker
        health = await load_balancer.get_worker_health("nonexistent")
        assert health is None
    
    async def test_get_all_worker_health(self, load_balancer):
        """Test getting all worker health information."""
        await load_balancer.register_worker("worker-1", {"voice"})
        await load_balancer.register_worker("worker-2", {"voice"})
        
        all_health = await load_balancer.get_all_worker_health()
        assert len(all_health) == 2
        assert "worker-1" in all_health
        assert "worker-2" in all_health
    
    async def test_get_load_distribution(self, load_balancer):
        """Test getting load distribution."""
        await load_balancer.register_worker("worker-1", {"voice"})
        await load_balancer.register_worker("worker-2", {"voice"})
        
        # Report some jobs
        await load_balancer.report_job_completion("worker-1", True, 100.0)
        await load_balancer.report_job_completion("worker-1", True, 100.0)
        await load_balancer.report_job_completion("worker-2", True, 100.0)
        
        distribution = await load_balancer.get_load_distribution()
        assert "worker-1" in distribution
        assert "worker-2" in distribution
        assert abs(distribution["worker-1"] - 66.67) < 0.1  # 2/3 of requests
        assert abs(distribution["worker-2"] - 33.33) < 0.1  # 1/3 of requests


class TestRoutingRules:
    """Test cases for routing rules."""
    
    def create_test_workers(self) -> List[WorkerHealth]:
        """Create test worker health objects."""
        return [
            WorkerHealth("us-east-worker", HealthStatus.HEALTHY),
            WorkerHealth("us-west-worker", HealthStatus.HEALTHY),
            WorkerHealth("eu-worker", HealthStatus.HEALTHY)
        ]
    
    async def test_geo_routing_rule(self):
        """Test geographical routing rule."""
        workers = self.create_test_workers()
        
        # Test with preferred region
        filtered = await geo_routing_rule(workers, {"preferred_region": "us-east"})
        assert len(filtered) == 1
        assert filtered[0].worker_id == "us-east-worker"
        
        # Test without preferred region
        filtered = await geo_routing_rule(workers, {})
        assert len(filtered) == 3  # All workers returned
        
        # Test with non-existent region
        filtered = await geo_routing_rule(workers, {"preferred_region": "asia"})
        assert len(filtered) == 3  # Fallback to all workers
    
    async def test_capability_routing_rule(self):
        """Test capability-based routing rule."""
        workers = self.create_test_workers()
        
        # Test with required capabilities
        filtered = await capability_routing_rule(workers, {"capabilities": {"voice"}})
        assert len(filtered) == 3  # Currently returns all (no capability metadata)
        
        # Test without required capabilities
        filtered = await capability_routing_rule(workers, {})
        assert len(filtered) == 3
    
    async def test_priority_routing_rule(self):
        """Test priority-based routing rule."""
        workers = self.create_test_workers()
        
        # Set up worker health for testing
        workers[0].status = HealthStatus.HEALTHY
        workers[0].metrics.error_rate = 0.005
        workers[1].status = HealthStatus.WARNING
        workers[1].metrics.error_rate = 0.02
        workers[2].status = HealthStatus.HEALTHY
        workers[2].metrics.error_rate = 0.0
        
        # Test with high priority
        filtered = await priority_routing_rule(workers, {"priority": 8})
        assert len(filtered) == 2  # Only healthy workers with low error rate
        
        # Test with normal priority
        filtered = await priority_routing_rule(workers, {"priority": 2})
        assert len(filtered) == 3  # All workers
        
        # Test without priority
        filtered = await priority_routing_rule(workers, {})
        assert len(filtered) == 3  # All workers


class TestFactoryFunction:
    """Test factory function."""
    
    def test_create_load_balancer(self):
        """Test load balancer factory function."""
        config = {"algorithm": "round_robin"}
        lb = create_load_balancer(config)
        
        assert isinstance(lb, WorkerLoadBalancer)
        assert lb.config == config
        assert lb.algorithm == LoadBalancingAlgorithm.ROUND_ROBIN


class TestLoadBalancingMetrics:
    """Test LoadBalancingMetrics dataclass."""
    
    def test_metrics_creation(self):
        """Test metrics creation with defaults."""
        metrics = LoadBalancingMetrics()
        
        assert metrics.active_connections == 0
        assert metrics.average_response_time == 0.0
        assert metrics.cpu_utilization == 0.0
        assert metrics.memory_utilization == 0.0
        assert metrics.queue_length == 0
        assert metrics.error_rate == 0.0
        assert metrics.throughput == 0.0
        assert isinstance(metrics.last_updated, datetime)
    
    def test_metrics_with_values(self):
        """Test metrics creation with custom values."""
        metrics = LoadBalancingMetrics(
            active_connections=5,
            average_response_time=150.0,
            cpu_utilization=60.0,
            memory_utilization=70.0,
            queue_length=3,
            error_rate=0.05,
            throughput=100.0
        )
        
        assert metrics.active_connections == 5
        assert metrics.average_response_time == 150.0
        assert metrics.cpu_utilization == 60.0
        assert metrics.memory_utilization == 70.0
        assert metrics.queue_length == 3
        assert metrics.error_rate == 0.05
        assert metrics.throughput == 100.0


class TestWorkerHealth:
    """Test WorkerHealth dataclass."""
    
    def test_worker_health_creation(self):
        """Test worker health creation with defaults."""
        health = WorkerHealth("test-worker")
        
        assert health.worker_id == "test-worker"
        assert health.status == HealthStatus.UNKNOWN
        assert isinstance(health.metrics, LoadBalancingMetrics)
        assert health.weight == 1.0
        assert health.consecutive_failures == 0
        assert health.total_requests == 0
        assert health.successful_requests == 0
        assert isinstance(health.last_health_check, datetime)
    
    def test_worker_health_with_values(self):
        """Test worker health creation with custom values."""
        metrics = LoadBalancingMetrics(active_connections=3)
        health = WorkerHealth(
            "test-worker",
            status=HealthStatus.HEALTHY,
            metrics=metrics,
            weight=2.0
        )
        
        assert health.worker_id == "test-worker"
        assert health.status == HealthStatus.HEALTHY
        assert health.metrics == metrics
        assert health.weight == 2.0


if __name__ == "__main__":
    pytest.main([__file__])