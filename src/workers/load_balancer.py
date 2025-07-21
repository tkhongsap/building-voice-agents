"""
Automatic Load Balancing for Worker Instances

This module provides sophisticated load balancing algorithms for distributing
jobs across worker instances in the LiveKit Voice Agents Platform.

Features:
- Multiple load balancing algorithms (round-robin, least-connections, weighted)
- Health-based routing and automatic failover
- Worker discovery and registration
- Real-time load monitoring and adjustment
- Custom routing rules and preferences
- Integration with worker orchestration system
"""

import asyncio
import logging
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from datetime import datetime, timedelta
import statistics
import random
from abc import ABC, abstractmethod

# Local imports
from .worker_orchestration import WorkerInfo, JobInfo, WorkerStatus
from ..monitoring.performance_monitor import PerformanceMonitor, MetricType
from ..components.error_handling import ErrorHandler, ErrorSeverity

logger = logging.getLogger(__name__)


class LoadBalancingAlgorithm(Enum):
    """Available load balancing algorithms."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_RESPONSE_TIME = "least_response_time"
    RESOURCE_BASED = "resource_based"
    CUSTOM = "custom"


class HealthStatus(Enum):
    """Health status for workers."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class LoadBalancingMetrics:
    """Metrics for load balancing decisions."""
    active_connections: int = 0
    average_response_time: float = 0.0
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    queue_length: int = 0
    error_rate: float = 0.0
    throughput: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class WorkerHealth:
    """Health information for a worker."""
    worker_id: str
    status: HealthStatus = HealthStatus.UNKNOWN
    metrics: LoadBalancingMetrics = field(default_factory=LoadBalancingMetrics)
    weight: float = 1.0
    last_health_check: datetime = field(default_factory=datetime.now)
    consecutive_failures: int = 0
    total_requests: int = 0
    successful_requests: int = 0


class LoadBalancingStrategy(ABC):
    """Abstract base class for load balancing strategies."""
    
    @abstractmethod
    async def select_worker(
        self,
        available_workers: List[WorkerHealth],
        job_requirements: Dict[str, Any]
    ) -> Optional[str]:
        """Select the best worker for a job."""
        pass
    
    @abstractmethod
    def get_algorithm_name(self) -> str:
        """Get the name of the algorithm."""
        pass


class RoundRobinStrategy(LoadBalancingStrategy):
    """Round-robin load balancing strategy."""
    
    def __init__(self):
        self.current_index = 0
        self._lock = threading.Lock()
    
    async def select_worker(
        self,
        available_workers: List[WorkerHealth],
        job_requirements: Dict[str, Any]
    ) -> Optional[str]:
        if not available_workers:
            return None
        
        with self._lock:
            # Filter healthy workers
            healthy_workers = [w for w in available_workers if w.status == HealthStatus.HEALTHY]
            if not healthy_workers:
                # Fallback to warning status workers
                healthy_workers = [w for w in available_workers if w.status == HealthStatus.WARNING]
            
            if not healthy_workers:
                return None
            
            # Round-robin selection
            selected_worker = healthy_workers[self.current_index % len(healthy_workers)]
            self.current_index = (self.current_index + 1) % len(healthy_workers)
            
            return selected_worker.worker_id
    
    def get_algorithm_name(self) -> str:
        return "round_robin"


class LeastConnectionsStrategy(LoadBalancingStrategy):
    """Least connections load balancing strategy."""
    
    async def select_worker(
        self,
        available_workers: List[WorkerHealth],
        job_requirements: Dict[str, Any]
    ) -> Optional[str]:
        if not available_workers:
            return None
        
        # Filter healthy workers
        healthy_workers = [w for w in available_workers if w.status == HealthStatus.HEALTHY]
        if not healthy_workers:
            healthy_workers = [w for w in available_workers if w.status == HealthStatus.WARNING]
        
        if not healthy_workers:
            return None
        
        # Select worker with least active connections
        selected_worker = min(healthy_workers, key=lambda w: w.metrics.active_connections)
        return selected_worker.worker_id
    
    def get_algorithm_name(self) -> str:
        return "least_connections"


class WeightedRoundRobinStrategy(LoadBalancingStrategy):
    """Weighted round-robin load balancing strategy."""
    
    def __init__(self):
        self.worker_weights = {}
        self.current_weights = {}
        self._lock = threading.Lock()
    
    async def select_worker(
        self,
        available_workers: List[WorkerHealth],
        job_requirements: Dict[str, Any]
    ) -> Optional[str]:
        if not available_workers:
            return None
        
        with self._lock:
            # Filter healthy workers
            healthy_workers = [w for w in available_workers if w.status == HealthStatus.HEALTHY]
            if not healthy_workers:
                healthy_workers = [w for w in available_workers if w.status == HealthStatus.WARNING]
            
            if not healthy_workers:
                return None
            
            # Update weights
            for worker in healthy_workers:
                if worker.worker_id not in self.worker_weights:
                    self.worker_weights[worker.worker_id] = worker.weight
                    self.current_weights[worker.worker_id] = worker.weight
            
            # Select worker with highest current weight
            selected_worker_id = max(
                healthy_workers,
                key=lambda w: self.current_weights.get(w.worker_id, 0)
            ).worker_id
            
            # Adjust weights
            max_weight = max(self.current_weights.values())
            for worker_id in self.current_weights:
                if worker_id == selected_worker_id:
                    self.current_weights[worker_id] -= max_weight
                else:
                    self.current_weights[worker_id] += self.worker_weights[worker_id]
            
            return selected_worker_id
    
    def get_algorithm_name(self) -> str:
        return "weighted_round_robin"


class LeastResponseTimeStrategy(LoadBalancingStrategy):
    """Least response time load balancing strategy."""
    
    async def select_worker(
        self,
        available_workers: List[WorkerHealth],
        job_requirements: Dict[str, Any]
    ) -> Optional[str]:
        if not available_workers:
            return None
        
        # Filter healthy workers
        healthy_workers = [w for w in available_workers if w.status == HealthStatus.HEALTHY]
        if not healthy_workers:
            healthy_workers = [w for w in available_workers if w.status == HealthStatus.WARNING]
        
        if not healthy_workers:
            return None
        
        # Select worker with lowest average response time
        selected_worker = min(healthy_workers, key=lambda w: w.metrics.average_response_time)
        return selected_worker.worker_id
    
    def get_algorithm_name(self) -> str:
        return "least_response_time"


class ResourceBasedStrategy(LoadBalancingStrategy):
    """Resource-based load balancing strategy."""
    
    def __init__(self, cpu_weight: float = 0.4, memory_weight: float = 0.3, queue_weight: float = 0.3):
        self.cpu_weight = cpu_weight
        self.memory_weight = memory_weight
        self.queue_weight = queue_weight
    
    async def select_worker(
        self,
        available_workers: List[WorkerHealth],
        job_requirements: Dict[str, Any]
    ) -> Optional[str]:
        if not available_workers:
            return None
        
        # Filter healthy workers
        healthy_workers = [w for w in available_workers if w.status == HealthStatus.HEALTHY]
        if not healthy_workers:
            healthy_workers = [w for w in available_workers if w.status == HealthStatus.WARNING]
        
        if not healthy_workers:
            return None
        
        # Calculate resource score for each worker (lower is better)
        def calculate_score(worker: WorkerHealth) -> float:
            cpu_score = worker.metrics.cpu_utilization * self.cpu_weight
            memory_score = worker.metrics.memory_utilization * self.memory_weight
            queue_score = (worker.metrics.queue_length / 10.0) * self.queue_weight  # Normalize queue length
            
            return cpu_score + memory_score + queue_score
        
        # Select worker with lowest resource score
        selected_worker = min(healthy_workers, key=calculate_score)
        return selected_worker.worker_id
    
    def get_algorithm_name(self) -> str:
        return "resource_based"


class WorkerLoadBalancer:
    """
    Advanced load balancer for worker instances.
    
    Provides intelligent routing of jobs to workers based on various algorithms
    and real-time health and performance metrics.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        performance_monitor: Optional[PerformanceMonitor] = None,
        error_handler: Optional[ErrorHandler] = None
    ):
        self.config = config or {}
        self.performance_monitor = performance_monitor or PerformanceMonitor()
        self.error_handler = error_handler or ErrorHandler()
        
        # Worker health tracking
        self.worker_health: Dict[str, WorkerHealth] = {}
        self.health_check_interval = self.config.get("health_check_interval", 30.0)
        self.health_timeout = self.config.get("health_timeout", 10.0)
        self.failure_threshold = self.config.get("failure_threshold", 3)
        
        # Load balancing configuration
        self.algorithm = LoadBalancingAlgorithm(
            self.config.get("algorithm", LoadBalancingAlgorithm.LEAST_CONNECTIONS.value)
        )
        self.strategies = self._initialize_strategies()
        self.current_strategy = self.strategies[self.algorithm]
        
        # State management
        self.running = False
        self._tasks: List[asyncio.Task] = []
        self._lock = asyncio.Lock()
        
        # Routing rules
        self.routing_rules: List[Callable] = []
        
        logger.info(f"WorkerLoadBalancer initialized with algorithm: {self.algorithm.value}")
    
    def _initialize_strategies(self) -> Dict[LoadBalancingAlgorithm, LoadBalancingStrategy]:
        """Initialize load balancing strategies."""
        return {
            LoadBalancingAlgorithm.ROUND_ROBIN: RoundRobinStrategy(),
            LoadBalancingAlgorithm.LEAST_CONNECTIONS: LeastConnectionsStrategy(),
            LoadBalancingAlgorithm.WEIGHTED_ROUND_ROBIN: WeightedRoundRobinStrategy(),
            LoadBalancingAlgorithm.LEAST_RESPONSE_TIME: LeastResponseTimeStrategy(),
            LoadBalancingAlgorithm.RESOURCE_BASED: ResourceBasedStrategy()
        }
    
    async def start(self):
        """Start the load balancer."""
        if self.running:
            return
        
        self.running = True
        
        # Start background tasks
        self._tasks = [
            asyncio.create_task(self._health_monitor()),
            asyncio.create_task(self._metrics_collector())
        ]
        
        await self.performance_monitor.record_metric(
            "load_balancer_started", 1, MetricType.COUNTER
        )
        
        logger.info("WorkerLoadBalancer started")
    
    async def stop(self):
        """Stop the load balancer."""
        if not self.running:
            return
        
        self.running = False
        
        # Cancel background tasks
        for task in self._tasks:
            task.cancel()
        
        if self._tasks:
            await asyncio.wait(self._tasks, return_when=asyncio.ALL_COMPLETED)
        
        await self.performance_monitor.record_metric(
            "load_balancer_stopped", 1, MetricType.COUNTER
        )
        
        logger.info("WorkerLoadBalancer stopped")
    
    async def register_worker(
        self,
        worker_id: str,
        capabilities: Set[str],
        weight: float = 1.0,
        initial_metrics: Optional[LoadBalancingMetrics] = None
    ):
        """Register a worker with the load balancer."""
        async with self._lock:
            if worker_id in self.worker_health:
                logger.warning(f"Worker {worker_id} already registered")
                return
            
            worker_health = WorkerHealth(
                worker_id=worker_id,
                weight=weight,
                metrics=initial_metrics or LoadBalancingMetrics()
            )
            
            self.worker_health[worker_id] = worker_health
            
            await self.performance_monitor.record_metric(
                "worker_registered", 1, MetricType.COUNTER,
                labels={"worker_id": worker_id}
            )
            
            logger.info(f"Worker {worker_id} registered with load balancer")
    
    async def unregister_worker(self, worker_id: str):
        """Unregister a worker from the load balancer."""
        async with self._lock:
            if worker_id in self.worker_health:
                del self.worker_health[worker_id]
                
                await self.performance_monitor.record_metric(
                    "worker_unregistered", 1, MetricType.COUNTER,
                    labels={"worker_id": worker_id}
                )
                
                logger.info(f"Worker {worker_id} unregistered from load balancer")
    
    async def select_worker(
        self,
        available_workers: List[WorkerInfo],
        job_requirements: Dict[str, Any]
    ) -> Optional[str]:
        """Select the best worker for a job using the configured algorithm."""
        if not available_workers:
            return None
        
        # Convert WorkerInfo to WorkerHealth for load balancing
        worker_health_list = []
        async with self._lock:
            for worker_info in available_workers:
                health = self.worker_health.get(worker_info.worker_id)
                if health:
                    worker_health_list.append(health)
        
        if not worker_health_list:
            # Fallback: register workers on-demand and use round-robin
            for worker_info in available_workers:
                await self.register_worker(worker_info.worker_id, worker_info.capabilities)
                health = self.worker_health[worker_info.worker_id]
                worker_health_list.append(health)
        
        # Apply routing rules
        filtered_workers = await self._apply_routing_rules(worker_health_list, job_requirements)
        
        # Use load balancing strategy
        selected_worker_id = await self.current_strategy.select_worker(
            filtered_workers, job_requirements
        )
        
        if selected_worker_id:
            await self.performance_monitor.record_metric(
                "worker_selected", 1, MetricType.COUNTER,
                labels={
                    "worker_id": selected_worker_id,
                    "algorithm": self.algorithm.value
                }
            )
        
        return selected_worker_id
    
    async def update_worker_metrics(
        self,
        worker_id: str,
        metrics: LoadBalancingMetrics
    ):
        """Update metrics for a worker."""
        async with self._lock:
            if worker_id in self.worker_health:
                self.worker_health[worker_id].metrics = metrics
                self.worker_health[worker_id].last_health_check = datetime.now()
                
                # Update health status based on metrics
                await self._update_worker_health_status(worker_id)
    
    async def report_job_completion(
        self,
        worker_id: str,
        success: bool,
        response_time: float
    ):
        """Report job completion for load balancing metrics."""
        async with self._lock:
            if worker_id not in self.worker_health:
                return
            
            health = self.worker_health[worker_id]
            health.total_requests += 1
            
            if success:
                health.successful_requests += 1
                health.consecutive_failures = 0
                
                # Update average response time
                current_avg = health.metrics.average_response_time
                total_requests = health.total_requests
                health.metrics.average_response_time = (
                    (current_avg * (total_requests - 1) + response_time) / total_requests
                )
                
            else:
                health.consecutive_failures += 1
                
                # Mark as unhealthy if too many consecutive failures
                if health.consecutive_failures >= self.failure_threshold:
                    health.status = HealthStatus.CRITICAL
            
            # Update error rate
            health.metrics.error_rate = 1.0 - (health.successful_requests / health.total_requests)
            
            await self._update_worker_health_status(worker_id)
    
    async def set_algorithm(self, algorithm: LoadBalancingAlgorithm):
        """Change the load balancing algorithm."""
        if algorithm in self.strategies:
            self.algorithm = algorithm
            self.current_strategy = self.strategies[algorithm]
            
            await self.performance_monitor.record_metric(
                "algorithm_changed", 1, MetricType.COUNTER,
                labels={"new_algorithm": algorithm.value}
            )
            
            logger.info(f"Load balancing algorithm changed to: {algorithm.value}")
    
    async def add_routing_rule(self, rule_func: Callable):
        """Add a custom routing rule."""
        self.routing_rules.append(rule_func)
        logger.info("Custom routing rule added")
    
    async def get_worker_health(self, worker_id: str) -> Optional[WorkerHealth]:
        """Get health information for a worker."""
        return self.worker_health.get(worker_id)
    
    async def get_all_worker_health(self) -> Dict[str, WorkerHealth]:
        """Get health information for all workers."""
        return self.worker_health.copy()
    
    async def get_load_distribution(self) -> Dict[str, float]:
        """Get current load distribution across workers."""
        total_requests = sum(h.total_requests for h in self.worker_health.values())
        if total_requests == 0:
            return {}
        
        return {
            worker_id: (health.total_requests / total_requests) * 100
            for worker_id, health in self.worker_health.items()
        }
    
    async def _apply_routing_rules(
        self,
        workers: List[WorkerHealth],
        job_requirements: Dict[str, Any]
    ) -> List[WorkerHealth]:
        """Apply custom routing rules to filter workers."""
        filtered_workers = workers
        
        for rule in self.routing_rules:
            try:
                filtered_workers = await rule(filtered_workers, job_requirements)
            except Exception as e:
                logger.error(f"Error applying routing rule: {e}")
        
        return filtered_workers
    
    async def _update_worker_health_status(self, worker_id: str):
        """Update health status based on current metrics."""
        health = self.worker_health[worker_id]
        metrics = health.metrics
        
        # Determine health status based on metrics
        if health.consecutive_failures >= self.failure_threshold:
            health.status = HealthStatus.CRITICAL
        elif (metrics.cpu_utilization > 90 or 
              metrics.memory_utilization > 90 or 
              metrics.error_rate > 0.1):
            health.status = HealthStatus.WARNING
        elif (metrics.cpu_utilization > 80 or 
              metrics.memory_utilization > 80 or 
              metrics.error_rate > 0.05):
            health.status = HealthStatus.WARNING
        else:
            health.status = HealthStatus.HEALTHY
    
    async def _health_monitor(self):
        """Background task to monitor worker health."""
        while self.running:
            try:
                current_time = datetime.now()
                stale_threshold = current_time - timedelta(seconds=self.health_timeout * 3)
                
                async with self._lock:
                    for worker_id, health in self.worker_health.items():
                        # Mark workers as unknown if no recent health updates
                        if health.last_health_check < stale_threshold:
                            health.status = HealthStatus.UNKNOWN
                
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
                await asyncio.sleep(1.0)
    
    async def _metrics_collector(self):
        """Background task to collect load balancing metrics."""
        while self.running:
            try:
                # Collect aggregate metrics
                total_workers = len(self.worker_health)
                healthy_workers = len([h for h in self.worker_health.values() if h.status == HealthStatus.HEALTHY])
                
                await self.performance_monitor.record_metric(
                    "lb_total_workers", total_workers, MetricType.GAUGE
                )
                
                await self.performance_monitor.record_metric(
                    "lb_healthy_workers", healthy_workers, MetricType.GAUGE
                )
                
                if self.worker_health:
                    avg_response_time = statistics.mean(
                        h.metrics.average_response_time for h in self.worker_health.values()
                    )
                    avg_error_rate = statistics.mean(
                        h.metrics.error_rate for h in self.worker_health.values()
                    )
                    
                    await self.performance_monitor.record_metric(
                        "lb_avg_response_time", avg_response_time, MetricType.GAUGE
                    )
                    
                    await self.performance_monitor.record_metric(
                        "lb_avg_error_rate", avg_error_rate, MetricType.GAUGE
                    )
                
                await asyncio.sleep(60.0)
                
            except Exception as e:
                logger.error(f"Error in metrics collector: {e}")
                await asyncio.sleep(1.0)


# Factory function
def create_load_balancer(config: Optional[Dict[str, Any]] = None) -> WorkerLoadBalancer:
    """Create a new WorkerLoadBalancer instance."""
    return WorkerLoadBalancer(config)


# Pre-defined routing rules
async def geo_routing_rule(
    workers: List[WorkerHealth],
    job_requirements: Dict[str, Any]
) -> List[WorkerHealth]:
    """Route jobs based on geographical preferences."""
    preferred_region = job_requirements.get("preferred_region")
    if not preferred_region:
        return workers
    
    # Filter workers by region (would need region metadata)
    region_workers = [w for w in workers if w.worker_id.startswith(preferred_region)]
    return region_workers if region_workers else workers


async def capability_routing_rule(
    workers: List[WorkerHealth],
    job_requirements: Dict[str, Any]
) -> List[WorkerHealth]:
    """Route jobs based on required capabilities."""
    required_capabilities = job_requirements.get("capabilities", set())
    if not required_capabilities:
        return workers
    
    # This would need capability metadata in WorkerHealth
    # For now, return all workers
    return workers


async def priority_routing_rule(
    workers: List[WorkerHealth],
    job_requirements: Dict[str, Any]
) -> List[WorkerHealth]:
    """Route high-priority jobs to best workers."""
    priority = job_requirements.get("priority", 0)
    if priority < 5:  # Normal priority
        return workers
    
    # For high priority, only use healthy workers with low error rates
    high_quality_workers = [
        w for w in workers 
        if w.status == HealthStatus.HEALTHY and w.metrics.error_rate < 0.01
    ]
    return high_quality_workers if high_quality_workers else workers