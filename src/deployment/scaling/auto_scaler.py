"""
Horizontal Auto-Scaling Implementation

This module provides intelligent horizontal auto-scaling capabilities for
worker instances based on various demand metrics and configurable policies.

Features:
- Metrics-based scaling (CPU, memory, queue depth, latency)
- Integration with Kubernetes HPA and custom scaling logic
- Predictive scaling using historical data
- Custom scaling policies and triggers
- Cost-aware scaling decisions
- Multi-dimensional scaling criteria
"""

import asyncio
import logging
import time
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime, timedelta
import json
import math

# Optional dependencies
try:
    from kubernetes import client, config as k8s_config
    from kubernetes.client.rest import ApiException
    K8S_AVAILABLE = True
except ImportError:
    K8S_AVAILABLE = False
    client = None
    k8s_config = None
    ApiException = Exception

# Local imports
from ...monitoring.performance_monitor import PerformanceMonitor, MetricType
from ...components.error_handling import ErrorHandler, ErrorSeverity
from ..worker_orchestration import WorkerOrchestrator

logger = logging.getLogger(__name__)


class ScalingDirection(Enum):
    """Scaling direction options."""
    UP = "up"
    DOWN = "down"
    NONE = "none"


class ScalingTrigger(Enum):
    """Triggers for scaling actions."""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    QUEUE_DEPTH = "queue_depth"
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    CUSTOM = "custom"


@dataclass
class ScalingMetrics:
    """Metrics used for scaling decisions."""
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    queue_depth: int = 0
    average_response_time: float = 0.0
    error_rate: float = 0.0
    throughput: float = 0.0
    active_workers: int = 0
    idle_workers: int = 0
    failed_workers: int = 0
    pending_jobs: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ScalingPolicy:
    """Configuration for scaling behavior."""
    trigger: ScalingTrigger
    scale_up_threshold: float
    scale_down_threshold: float
    scale_up_cooldown: int = 300  # seconds
    scale_down_cooldown: int = 600  # seconds
    min_instances: int = 1
    max_instances: int = 100
    scale_up_step: int = 1
    scale_down_step: int = 1
    enabled: bool = True
    weight: float = 1.0  # For multi-criteria scaling


@dataclass
class ScalingEvent:
    """Record of a scaling event."""
    timestamp: datetime
    direction: ScalingDirection
    trigger: ScalingTrigger
    old_count: int
    new_count: int
    reason: str
    metrics: ScalingMetrics


class PredictiveScaler:
    """
    Predictive scaling based on historical patterns.
    """
    
    def __init__(self, history_window: int = 24 * 60):  # 24 hours in minutes
        self.history_window = history_window
        self.metrics_history: List[Tuple[datetime, ScalingMetrics]] = []
        self.patterns: Dict[str, List[float]] = {}
    
    def add_metrics(self, metrics: ScalingMetrics):
        """Add metrics to history."""
        self.metrics_history.append((datetime.now(), metrics))
        
        # Cleanup old data
        cutoff_time = datetime.now() - timedelta(minutes=self.history_window)
        self.metrics_history = [
            (ts, m) for ts, m in self.metrics_history 
            if ts > cutoff_time
        ]
        
        # Update patterns
        self._update_patterns()
    
    def _update_patterns(self):
        """Update patterns based on historical data."""
        if len(self.metrics_history) < 10:
            return
        
        # Extract hourly patterns
        hourly_data = {}
        for timestamp, metrics in self.metrics_history:
            hour = timestamp.hour
            if hour not in hourly_data:
                hourly_data[hour] = []
            hourly_data[hour].append({
                'cpu': metrics.cpu_utilization,
                'memory': metrics.memory_utilization,
                'queue': metrics.queue_depth,
                'throughput': metrics.throughput
            })
        
        # Calculate averages for each hour
        for hour, data_points in hourly_data.items():
            if len(data_points) >= 3:  # Minimum data points for pattern
                self.patterns[f"hour_{hour}"] = {
                    'cpu': statistics.mean([d['cpu'] for d in data_points]),
                    'memory': statistics.mean([d['memory'] for d in data_points]),
                    'queue': statistics.mean([d['queue'] for d in data_points]),
                    'throughput': statistics.mean([d['throughput'] for d in data_points])
                }
    
    def predict_demand(self, minutes_ahead: int = 15) -> Dict[str, float]:
        """Predict demand metrics for future time."""
        future_time = datetime.now() + timedelta(minutes=minutes_ahead)
        future_hour = future_time.hour
        
        pattern_key = f"hour_{future_hour}"
        if pattern_key in self.patterns:
            return self.patterns[pattern_key]
        
        # Fallback to current metrics if no pattern available
        if self.metrics_history:
            latest_metrics = self.metrics_history[-1][1]
            return {
                'cpu': latest_metrics.cpu_utilization,
                'memory': latest_metrics.memory_utilization,
                'queue': latest_metrics.queue_depth,
                'throughput': latest_metrics.throughput
            }
        
        return {'cpu': 0, 'memory': 0, 'queue': 0, 'throughput': 0}


class KubernetesScaler:
    """
    Kubernetes-specific scaling operations.
    """
    
    def __init__(self, namespace: str = "default", deployment_name: str = "voice-agents"):
        self.namespace = namespace
        self.deployment_name = deployment_name
        self.k8s_apps_v1 = None
        
        if K8S_AVAILABLE:
            try:
                k8s_config.load_incluster_config()
            except k8s_config.ConfigException:
                try:
                    k8s_config.load_kube_config()
                except k8s_config.ConfigException:
                    logger.warning("Could not load Kubernetes configuration")
                    return
            
            self.k8s_apps_v1 = client.AppsV1Api()
    
    async def get_current_replicas(self) -> int:
        """Get current number of replicas."""
        if not self.k8s_apps_v1:
            return 0
        
        try:
            deployment = self.k8s_apps_v1.read_namespaced_deployment(
                name=self.deployment_name,
                namespace=self.namespace
            )
            return deployment.spec.replicas or 0
        except ApiException as e:
            logger.error(f"Error getting current replicas: {e}")
            return 0
    
    async def scale_deployment(self, target_replicas: int) -> bool:
        """Scale deployment to target replicas."""
        if not self.k8s_apps_v1:
            logger.warning("Kubernetes client not available")
            return False
        
        try:
            # Patch the deployment
            body = {"spec": {"replicas": target_replicas}}
            self.k8s_apps_v1.patch_namespaced_deployment(
                name=self.deployment_name,
                namespace=self.namespace,
                body=body
            )
            
            logger.info(f"Scaled deployment {self.deployment_name} to {target_replicas} replicas")
            return True
            
        except ApiException as e:
            logger.error(f"Error scaling deployment: {e}")
            return False
    
    async def get_deployment_status(self) -> Dict[str, Any]:
        """Get detailed deployment status."""
        if not self.k8s_apps_v1:
            return {}
        
        try:
            deployment = self.k8s_apps_v1.read_namespaced_deployment(
                name=self.deployment_name,
                namespace=self.namespace
            )
            
            return {
                "desired_replicas": deployment.spec.replicas,
                "ready_replicas": deployment.status.ready_replicas or 0,
                "available_replicas": deployment.status.available_replicas or 0,
                "unavailable_replicas": deployment.status.unavailable_replicas or 0,
                "updated_replicas": deployment.status.updated_replicas or 0
            }
            
        except ApiException as e:
            logger.error(f"Error getting deployment status: {e}")
            return {}


class HorizontalAutoScaler:
    """
    Comprehensive horizontal auto-scaling system.
    
    Provides intelligent scaling based on multiple metrics, predictive analysis,
    and configurable policies with Kubernetes integration.
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        orchestrator: Optional[WorkerOrchestrator] = None,
        performance_monitor: Optional[PerformanceMonitor] = None,
        error_handler: Optional[ErrorHandler] = None
    ):
        self.config = config or {}
        self.orchestrator = orchestrator
        self.performance_monitor = performance_monitor or PerformanceMonitor()
        self.error_handler = error_handler or ErrorHandler()
        
        # Scaling configuration
        self.check_interval = self.config.get("check_interval", 60.0)  # seconds
        self.metrics_window = self.config.get("metrics_window", 300.0)  # seconds
        self.prediction_enabled = self.config.get("prediction_enabled", True)
        self.dry_run = self.config.get("dry_run", False)
        
        # Scaling policies
        self.policies: Dict[ScalingTrigger, ScalingPolicy] = self._load_policies()
        
        # State management
        self.running = False
        self.last_scale_up = datetime.min
        self.last_scale_down = datetime.min
        self.scaling_events: List[ScalingEvent] = []
        self.current_replicas = 0
        
        # Metrics tracking
        self.metrics_history: List[ScalingMetrics] = []
        self.predictive_scaler = PredictiveScaler()
        
        # Kubernetes integration
        self.k8s_scaler = KubernetesScaler(
            namespace=self.config.get("kubernetes", {}).get("namespace", "default"),
            deployment_name=self.config.get("kubernetes", {}).get("deployment_name", "voice-agents")
        )
        
        # Background tasks
        self._tasks: List[asyncio.Task] = []
        
        logger.info("HorizontalAutoScaler initialized")
    
    def _load_policies(self) -> Dict[ScalingTrigger, ScalingPolicy]:
        """Load scaling policies from configuration."""
        default_policies = {
            ScalingTrigger.CPU_UTILIZATION: ScalingPolicy(
                trigger=ScalingTrigger.CPU_UTILIZATION,
                scale_up_threshold=70.0,
                scale_down_threshold=30.0,
                scale_up_cooldown=300,
                scale_down_cooldown=600,
                min_instances=2,
                max_instances=50,
                weight=1.0
            ),
            ScalingTrigger.MEMORY_UTILIZATION: ScalingPolicy(
                trigger=ScalingTrigger.MEMORY_UTILIZATION,
                scale_up_threshold=75.0,
                scale_down_threshold=40.0,
                scale_up_cooldown=300,
                scale_down_cooldown=600,
                min_instances=2,
                max_instances=50,
                weight=0.8
            ),
            ScalingTrigger.QUEUE_DEPTH: ScalingPolicy(
                trigger=ScalingTrigger.QUEUE_DEPTH,
                scale_up_threshold=10.0,
                scale_down_threshold=2.0,
                scale_up_cooldown=180,
                scale_down_cooldown=600,
                min_instances=1,
                max_instances=100,
                weight=1.2
            ),
            ScalingTrigger.RESPONSE_TIME: ScalingPolicy(
                trigger=ScalingTrigger.RESPONSE_TIME,
                scale_up_threshold=2000.0,  # 2 seconds
                scale_down_threshold=500.0,  # 0.5 seconds
                scale_up_cooldown=240,
                scale_down_cooldown=600,
                min_instances=2,
                max_instances=50,
                weight=1.5
            )
        }
        
        # Override with configuration
        policies_config = self.config.get("policies", {})
        for trigger_name, policy_config in policies_config.items():
            try:
                trigger = ScalingTrigger(trigger_name)
                if trigger in default_policies:
                    policy = default_policies[trigger]
                    # Update policy with config values
                    for key, value in policy_config.items():
                        if hasattr(policy, key):
                            setattr(policy, key, value)
            except (ValueError, AttributeError) as e:
                logger.warning(f"Invalid policy configuration for {trigger_name}: {e}")
        
        return default_policies
    
    async def start(self):
        """Start the auto-scaler."""
        if self.running:
            return
        
        self.running = True
        
        # Get initial replica count
        self.current_replicas = await self.k8s_scaler.get_current_replicas()
        
        # Start background tasks
        self._tasks = [
            asyncio.create_task(self._metrics_collector()),
            asyncio.create_task(self._scaling_controller()),
            asyncio.create_task(self._predictive_controller())
        ]
        
        await self.performance_monitor.record_metric(
            "autoscaler_started", 1, MetricType.COUNTER
        )
        
        logger.info("HorizontalAutoScaler started")
    
    async def stop(self):
        """Stop the auto-scaler."""
        if not self.running:
            return
        
        self.running = False
        
        # Cancel background tasks
        for task in self._tasks:
            task.cancel()
        
        if self._tasks:
            await asyncio.wait(self._tasks, return_when=asyncio.ALL_COMPLETED)
        
        await self.performance_monitor.record_metric(
            "autoscaler_stopped", 1, MetricType.COUNTER
        )
        
        logger.info("HorizontalAutoScaler stopped")
    
    async def collect_metrics(self) -> ScalingMetrics:
        """Collect current scaling metrics."""
        metrics = ScalingMetrics()
        
        if self.orchestrator:
            # Get metrics from orchestrator
            workers = await self.orchestrator.list_workers()
            jobs = await self.orchestrator.list_jobs()
            
            metrics.active_workers = len([w for w in workers if w.status.value in ["idle", "busy"]])
            metrics.idle_workers = len([w for w in workers if w.status.value == "idle"])
            metrics.failed_workers = len([w for w in workers if w.status.value == "failed"])
            metrics.pending_jobs = len([j for j in jobs if j.status.value == "queued"])
            
            # Calculate average metrics from workers
            if workers:
                cpu_values = []
                memory_values = []
                response_times = []
                
                for worker in workers:
                    worker_metrics = worker.metrics
                    if worker_metrics:
                        cpu_values.append(worker_metrics.get("cpu_utilization", 0))
                        memory_values.append(worker_metrics.get("memory_utilization", 0))
                        response_times.append(worker_metrics.get("avg_response_time", 0))
                
                if cpu_values:
                    metrics.cpu_utilization = statistics.mean(cpu_values)
                if memory_values:
                    metrics.memory_utilization = statistics.mean(memory_values)
                if response_times:
                    metrics.average_response_time = statistics.mean(response_times)
        
        # Get queue depth from orchestrator
        if self.orchestrator and hasattr(self.orchestrator, 'job_queue'):
            metrics.queue_depth = self.orchestrator.job_queue.qsize()
        
        return metrics
    
    async def make_scaling_decision(self, metrics: ScalingMetrics) -> Tuple[ScalingDirection, int, str]:
        """Make scaling decision based on current metrics."""
        if not self.metrics_history:
            return ScalingDirection.NONE, 0, "Insufficient metrics history"
        
        # Calculate average metrics over the window
        window_start = datetime.now() - timedelta(seconds=self.metrics_window)
        recent_metrics = [m for m in self.metrics_history if m.timestamp > window_start]
        
        if len(recent_metrics) < 3:
            return ScalingDirection.NONE, 0, "Insufficient recent metrics"
        
        # Multi-criteria decision making
        scale_up_votes = 0
        scale_down_votes = 0
        total_weight = 0
        reasons = []
        
        for trigger, policy in self.policies.items():
            if not policy.enabled:
                continue
            
            total_weight += policy.weight
            metric_value = self._get_metric_value(metrics, trigger)
            
            if metric_value > policy.scale_up_threshold:
                scale_up_votes += policy.weight
                reasons.append(f"{trigger.value}={metric_value:.1f} > {policy.scale_up_threshold}")
            elif metric_value < policy.scale_down_threshold:
                scale_down_votes += policy.weight
                reasons.append(f"{trigger.value}={metric_value:.1f} < {policy.scale_down_threshold}")
        
        # Check cooldown periods
        now = datetime.now()
        scale_up_cooldown = max(p.scale_up_cooldown for p in self.policies.values())
        scale_down_cooldown = max(p.scale_down_cooldown for p in self.policies.values())
        
        can_scale_up = (now - self.last_scale_up).total_seconds() > scale_up_cooldown
        can_scale_down = (now - self.last_scale_down).total_seconds() > scale_down_cooldown
        
        # Make decision
        if scale_up_votes > scale_down_votes and can_scale_up:
            # Determine scale-up amount
            max_step = max(p.scale_up_step for p in self.policies.values())
            scale_amount = min(max_step, max(1, int(scale_up_votes / total_weight * max_step)))
            
            return ScalingDirection.UP, scale_amount, "; ".join(reasons)
        
        elif scale_down_votes > scale_up_votes and can_scale_down:
            # Determine scale-down amount
            max_step = max(p.scale_down_step for p in self.policies.values())
            scale_amount = min(max_step, max(1, int(scale_down_votes / total_weight * max_step)))
            
            return ScalingDirection.DOWN, scale_amount, "; ".join(reasons)
        
        else:
            if not can_scale_up and scale_up_votes > scale_down_votes:
                return ScalingDirection.NONE, 0, "Scale-up in cooldown period"
            elif not can_scale_down and scale_down_votes > scale_up_votes:
                return ScalingDirection.NONE, 0, "Scale-down in cooldown period"
            else:
                return ScalingDirection.NONE, 0, "No scaling threshold met"
    
    def _get_metric_value(self, metrics: ScalingMetrics, trigger: ScalingTrigger) -> float:
        """Get metric value for specific trigger."""
        if trigger == ScalingTrigger.CPU_UTILIZATION:
            return metrics.cpu_utilization
        elif trigger == ScalingTrigger.MEMORY_UTILIZATION:
            return metrics.memory_utilization
        elif trigger == ScalingTrigger.QUEUE_DEPTH:
            return float(metrics.queue_depth)
        elif trigger == ScalingTrigger.RESPONSE_TIME:
            return metrics.average_response_time
        elif trigger == ScalingTrigger.ERROR_RATE:
            return metrics.error_rate * 100  # Convert to percentage
        elif trigger == ScalingTrigger.THROUGHPUT:
            return metrics.throughput
        else:
            return 0.0
    
    async def execute_scaling(self, direction: ScalingDirection, amount: int, reason: str) -> bool:
        """Execute scaling action."""
        if direction == ScalingDirection.NONE:
            return False
        
        current_replicas = await self.k8s_scaler.get_current_replicas()
        
        if direction == ScalingDirection.UP:
            target_replicas = current_replicas + amount
        else:  # ScalingDirection.DOWN
            target_replicas = current_replicas - amount
        
        # Apply constraints
        min_replicas = min(p.min_instances for p in self.policies.values())
        max_replicas = min(p.max_instances for p in self.policies.values())
        target_replicas = max(min_replicas, min(max_replicas, target_replicas))
        
        if target_replicas == current_replicas:
            return False
        
        # Record scaling event
        event = ScalingEvent(
            timestamp=datetime.now(),
            direction=direction,
            trigger=ScalingTrigger.CPU_UTILIZATION,  # Primary trigger
            old_count=current_replicas,
            new_count=target_replicas,
            reason=reason,
            metrics=self.metrics_history[-1] if self.metrics_history else ScalingMetrics()
        )
        self.scaling_events.append(event)
        
        # Execute scaling (unless dry run)
        if not self.dry_run:
            success = await self.k8s_scaler.scale_deployment(target_replicas)
            if not success:
                return False
        else:
            logger.info(f"DRY RUN: Would scale from {current_replicas} to {target_replicas}")
            success = True
        
        # Update state
        if success:
            self.current_replicas = target_replicas
            
            if direction == ScalingDirection.UP:
                self.last_scale_up = datetime.now()
            else:
                self.last_scale_down = datetime.now()
            
            # Record metrics
            await self.performance_monitor.record_metric(
                "scaling_event", 1, MetricType.COUNTER,
                labels={"direction": direction.value, "reason": "demand"}
            )
            
            await self.performance_monitor.record_metric(
                "current_replicas", target_replicas, MetricType.GAUGE
            )
            
            logger.info(
                f"Scaled {direction.value} from {current_replicas} to {target_replicas} "
                f"replicas. Reason: {reason}"
            )
        
        return success
    
    async def _metrics_collector(self):
        """Background task to collect metrics."""
        while self.running:
            try:
                metrics = await self.collect_metrics()
                self.metrics_history.append(metrics)
                
                # Add to predictive scaler
                if self.prediction_enabled:
                    self.predictive_scaler.add_metrics(metrics)
                
                # Cleanup old metrics
                cutoff_time = datetime.now() - timedelta(seconds=self.metrics_window * 3)
                self.metrics_history = [
                    m for m in self.metrics_history 
                    if m.timestamp > cutoff_time
                ]
                
                # Record metrics
                await self.performance_monitor.record_metric(
                    "autoscaler_cpu_utilization", metrics.cpu_utilization, MetricType.GAUGE
                )
                await self.performance_monitor.record_metric(
                    "autoscaler_memory_utilization", metrics.memory_utilization, MetricType.GAUGE
                )
                await self.performance_monitor.record_metric(
                    "autoscaler_queue_depth", metrics.queue_depth, MetricType.GAUGE
                )
                
                await asyncio.sleep(30.0)  # Collect metrics every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in metrics collector: {e}")
                await asyncio.sleep(30.0)
    
    async def _scaling_controller(self):
        """Background task for scaling decisions."""
        while self.running:
            try:
                if self.metrics_history:
                    current_metrics = self.metrics_history[-1]
                    direction, amount, reason = await self.make_scaling_decision(current_metrics)
                    
                    if direction != ScalingDirection.NONE:
                        await self.execute_scaling(direction, amount, reason)
                
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in scaling controller: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _predictive_controller(self):
        """Background task for predictive scaling."""
        if not self.prediction_enabled:
            return
        
        while self.running:
            try:
                # Predict demand 15 minutes ahead
                predicted_metrics = self.predictive_scaler.predict_demand(15)
                
                # If predicted demand is significantly higher, pre-scale
                if predicted_metrics:
                    current_avg_cpu = statistics.mean([
                        m.cpu_utilization for m in self.metrics_history[-5:]
                    ]) if len(self.metrics_history) >= 5 else 0
                    
                    predicted_cpu = predicted_metrics.get('cpu', 0)
                    
                    # If predicted CPU is 30% higher than current, consider pre-scaling
                    if predicted_cpu > current_avg_cpu * 1.3 and predicted_cpu > 60:
                        logger.info(
                            f"Predictive scaling: Current CPU {current_avg_cpu:.1f}%, "
                            f"Predicted CPU {predicted_cpu:.1f}%"
                        )
                        
                        # Execute small predictive scale-up
                        await self.execute_scaling(
                            ScalingDirection.UP, 1, 
                            f"Predictive scaling: forecasted CPU {predicted_cpu:.1f}%"
                        )
                
                await asyncio.sleep(600.0)  # Check predictions every 10 minutes
                
            except Exception as e:
                logger.error(f"Error in predictive controller: {e}")
                await asyncio.sleep(600.0)
    
    async def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status."""
        current_replicas = await self.k8s_scaler.get_current_replicas()
        deployment_status = await self.k8s_scaler.get_deployment_status()
        
        recent_events = self.scaling_events[-10:]  # Last 10 events
        
        return {
            "current_replicas": current_replicas,
            "deployment_status": deployment_status,
            "policies": {t.value: {
                "scale_up_threshold": p.scale_up_threshold,
                "scale_down_threshold": p.scale_down_threshold,
                "enabled": p.enabled
            } for t, p in self.policies.items()},
            "recent_events": [
                {
                    "timestamp": e.timestamp.isoformat(),
                    "direction": e.direction.value,
                    "old_count": e.old_count,
                    "new_count": e.new_count,
                    "reason": e.reason
                } for e in recent_events
            ],
            "last_scale_up": self.last_scale_up.isoformat() if self.last_scale_up != datetime.min else None,
            "last_scale_down": self.last_scale_down.isoformat() if self.last_scale_down != datetime.min else None,
            "dry_run": self.dry_run,
            "prediction_enabled": self.prediction_enabled
        }
    
    async def update_policy(self, trigger: ScalingTrigger, **kwargs):
        """Update a scaling policy."""
        if trigger in self.policies:
            policy = self.policies[trigger]
            for key, value in kwargs.items():
                if hasattr(policy, key):
                    setattr(policy, key, value)
            
            logger.info(f"Updated policy for {trigger.value}")
        else:
            logger.warning(f"Policy for {trigger.value} not found")


# Factory function
def create_auto_scaler(
    config: Optional[Dict[str, Any]] = None,
    orchestrator: Optional[WorkerOrchestrator] = None
) -> HorizontalAutoScaler:
    """Create a new HorizontalAutoScaler instance."""
    return HorizontalAutoScaler(config, orchestrator)


# Configuration helpers
def create_scaling_policy(
    trigger: str,
    scale_up_threshold: float,
    scale_down_threshold: float,
    **kwargs
) -> Dict[str, Any]:
    """Helper to create scaling policy configuration."""
    policy_config = {
        "scale_up_threshold": scale_up_threshold,
        "scale_down_threshold": scale_down_threshold
    }
    policy_config.update(kwargs)
    return {trigger: policy_config}