"""
Vertical Scaling for Resource Optimization

This module provides intelligent vertical scaling capabilities for optimizing
resource allocation of worker instances based on usage patterns and performance metrics.

Features:
- Resource monitoring and recommendation engine
- Automatic resource limit adjustment
- Memory and CPU optimization algorithms
- Performance-based scaling decisions
- Integration with Kubernetes VPA (Vertical Pod Autoscaler)
- Cost optimization through right-sizing
"""

import asyncio
import logging
import time
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
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

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of resources that can be scaled."""
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"


class ScalingDirection(Enum):
    """Vertical scaling direction."""
    UP = "up"
    DOWN = "down"
    NONE = "none"


class ResourceUnit(Enum):
    """Resource units for Kubernetes."""
    CPU_MILLICORES = "m"
    CPU_CORES = ""
    MEMORY_BYTES = ""
    MEMORY_KB = "Ki"
    MEMORY_MB = "Mi"
    MEMORY_GB = "Gi"


@dataclass
class ResourceMetrics:
    """Resource usage metrics for a worker."""
    worker_id: str
    cpu_usage: float = 0.0          # Current CPU usage (cores)
    cpu_limit: float = 0.0          # Current CPU limit (cores)
    memory_usage: float = 0.0       # Current memory usage (bytes)
    memory_limit: float = 0.0       # Current memory limit (bytes)
    cpu_utilization: float = 0.0    # CPU usage percentage
    memory_utilization: float = 0.0 # Memory usage percentage
    peak_cpu: float = 0.0           # Peak CPU usage in window
    peak_memory: float = 0.0        # Peak memory usage in window
    avg_cpu: float = 0.0            # Average CPU usage in window
    avg_memory: float = 0.0         # Average memory usage in window
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ResourceRecommendation:
    """Resource scaling recommendation."""
    worker_id: str
    resource_type: ResourceType
    current_value: float
    recommended_value: float
    direction: ScalingDirection
    confidence: float               # Confidence in recommendation (0-1)
    reason: str
    potential_savings: float = 0.0  # Cost savings if applicable
    performance_impact: str = "low" # Expected performance impact


@dataclass
class ResourceConstraints:
    """Resource constraints for scaling."""
    min_cpu: float = 0.1            # Minimum CPU cores
    max_cpu: float = 16.0           # Maximum CPU cores
    min_memory: float = 128 * 1024 * 1024  # Minimum memory bytes (128MB)
    max_memory: float = 32 * 1024 * 1024 * 1024  # Maximum memory bytes (32GB)
    cpu_step: float = 0.1           # CPU scaling step
    memory_step: float = 128 * 1024 * 1024  # Memory scaling step (128MB)


class ResourceAnalyzer:
    """
    Analyzes resource usage patterns and provides scaling recommendations.
    """
    
    def __init__(self, analysis_window: int = 24 * 60):  # 24 hours in minutes
        self.analysis_window = analysis_window
        self.metrics_history: Dict[str, List[ResourceMetrics]] = {}
        self.constraints = ResourceConstraints()
    
    def add_metrics(self, metrics: ResourceMetrics):
        """Add resource metrics for analysis."""
        worker_id = metrics.worker_id
        
        if worker_id not in self.metrics_history:
            self.metrics_history[worker_id] = []
        
        self.metrics_history[worker_id].append(metrics)
        
        # Cleanup old metrics
        cutoff_time = datetime.now() - timedelta(minutes=self.analysis_window)
        self.metrics_history[worker_id] = [
            m for m in self.metrics_history[worker_id]
            if m.timestamp > cutoff_time
        ]
    
    def analyze_cpu_usage(self, worker_id: str) -> Optional[ResourceRecommendation]:
        """Analyze CPU usage and provide recommendation."""
        if worker_id not in self.metrics_history:
            return None
        
        metrics_list = self.metrics_history[worker_id]
        if len(metrics_list) < 10:  # Need sufficient data
            return None
        
        # Calculate statistics
        cpu_utilizations = [m.cpu_utilization for m in metrics_list]
        avg_utilization = statistics.mean(cpu_utilizations)
        max_utilization = max(cpu_utilizations)
        percentile_95 = self._percentile(cpu_utilizations, 95)
        percentile_99 = self._percentile(cpu_utilizations, 99)
        
        current_limit = metrics_list[-1].cpu_limit
        current_usage = metrics_list[-1].cpu_usage
        
        # Determine recommendation
        if avg_utilization > 80 or percentile_95 > 90:
            # Scale up CPU
            if percentile_99 > 95:
                # High utilization, significant increase
                recommended_cpu = current_limit * 1.5
            else:
                # Moderate increase
                recommended_cpu = current_limit * 1.2
            
            recommended_cpu = min(recommended_cpu, self.constraints.max_cpu)
            recommended_cpu = self._round_cpu(recommended_cpu)
            
            if recommended_cpu > current_limit:
                return ResourceRecommendation(
                    worker_id=worker_id,
                    resource_type=ResourceType.CPU,
                    current_value=current_limit,
                    recommended_value=recommended_cpu,
                    direction=ScalingDirection.UP,
                    confidence=min(0.9, avg_utilization / 100.0 + 0.1),
                    reason=f"High CPU utilization: avg={avg_utilization:.1f}%, p95={percentile_95:.1f}%",
                    performance_impact="medium"
                )
        
        elif avg_utilization < 30 and percentile_95 < 50:
            # Scale down CPU
            if percentile_99 < 40:
                # Very low utilization, significant decrease
                recommended_cpu = current_limit * 0.7
            else:
                # Moderate decrease
                recommended_cpu = current_limit * 0.8
            
            recommended_cpu = max(recommended_cpu, self.constraints.min_cpu)
            recommended_cpu = self._round_cpu(recommended_cpu)
            
            if recommended_cpu < current_limit:
                potential_savings = (current_limit - recommended_cpu) * 0.05  # $0.05 per CPU hour estimate
                
                return ResourceRecommendation(
                    worker_id=worker_id,
                    resource_type=ResourceType.CPU,
                    current_value=current_limit,
                    recommended_value=recommended_cpu,
                    direction=ScalingDirection.DOWN,
                    confidence=max(0.6, 1.0 - (avg_utilization / 100.0)),
                    reason=f"Low CPU utilization: avg={avg_utilization:.1f}%, p95={percentile_95:.1f}%",
                    potential_savings=potential_savings,
                    performance_impact="low"
                )
        
        return None
    
    def analyze_memory_usage(self, worker_id: str) -> Optional[ResourceRecommendation]:
        """Analyze memory usage and provide recommendation."""
        if worker_id not in self.metrics_history:
            return None
        
        metrics_list = self.metrics_history[worker_id]
        if len(metrics_list) < 10:
            return None
        
        # Calculate statistics
        memory_utilizations = [m.memory_utilization for m in metrics_list]
        avg_utilization = statistics.mean(memory_utilizations)
        max_utilization = max(memory_utilizations)
        percentile_95 = self._percentile(memory_utilizations, 95)
        percentile_99 = self._percentile(memory_utilizations, 99)
        
        current_limit = metrics_list[-1].memory_limit
        current_usage = metrics_list[-1].memory_usage
        
        # Check for memory pressure patterns
        recent_metrics = metrics_list[-20:]  # Last 20 data points
        memory_trend = self._calculate_trend([m.memory_usage for m in recent_metrics])
        
        # Determine recommendation
        if avg_utilization > 85 or percentile_95 > 95 or memory_trend > 0.1:
            # Scale up memory
            if percentile_99 > 95 or memory_trend > 0.2:
                # High pressure or growing trend
                recommended_memory = current_limit * 1.4
            else:
                # Moderate increase
                recommended_memory = current_limit * 1.2
            
            recommended_memory = min(recommended_memory, self.constraints.max_memory)
            recommended_memory = self._round_memory(recommended_memory)
            
            if recommended_memory > current_limit:
                return ResourceRecommendation(
                    worker_id=worker_id,
                    resource_type=ResourceType.MEMORY,
                    current_value=current_limit,
                    recommended_value=recommended_memory,
                    direction=ScalingDirection.UP,
                    confidence=min(0.9, avg_utilization / 100.0 + 0.1),
                    reason=f"High memory utilization: avg={avg_utilization:.1f}%, p95={percentile_95:.1f}%, trend={memory_trend:.2f}",
                    performance_impact="high"
                )
        
        elif avg_utilization < 40 and percentile_95 < 60 and memory_trend < 0.05:
            # Scale down memory
            if percentile_99 < 50:
                # Very low utilization
                recommended_memory = current_limit * 0.7
            else:
                # Moderate decrease
                recommended_memory = current_limit * 0.8
            
            recommended_memory = max(recommended_memory, self.constraints.min_memory)
            recommended_memory = self._round_memory(recommended_memory)
            
            if recommended_memory < current_limit:
                # Memory cost is higher than CPU
                potential_savings = (current_limit - recommended_memory) / (1024**3) * 0.10  # $0.10 per GB hour estimate
                
                return ResourceRecommendation(
                    worker_id=worker_id,
                    resource_type=ResourceType.MEMORY,
                    current_value=current_limit,
                    recommended_value=recommended_memory,
                    direction=ScalingDirection.DOWN,
                    confidence=max(0.7, 1.0 - (avg_utilization / 100.0)),
                    reason=f"Low memory utilization: avg={avg_utilization:.1f}%, p95={percentile_95:.1f}%",
                    potential_savings=potential_savings,
                    performance_impact="low"
                )
        
        return None
    
    def get_recommendations(self, worker_id: str) -> List[ResourceRecommendation]:
        """Get all resource recommendations for a worker."""
        recommendations = []
        
        cpu_rec = self.analyze_cpu_usage(worker_id)
        if cpu_rec:
            recommendations.append(cpu_rec)
        
        memory_rec = self.analyze_memory_usage(worker_id)
        if memory_rec:
            recommendations.append(memory_rec)
        
        return recommendations
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = int((percentile / 100.0) * (len(sorted_data) - 1))
        return sorted_data[index]
    
    def _calculate_trend(self, data: List[float]) -> float:
        """Calculate trend in data (simple linear regression slope)."""
        if len(data) < 2:
            return 0.0
        
        n = len(data)
        x_values = list(range(n))
        
        # Calculate slope using simple linear regression
        sum_x = sum(x_values)
        sum_y = sum(data)
        sum_xy = sum(x * y for x, y in zip(x_values, data))
        sum_x2 = sum(x * x for x in x_values)
        
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope / max(data) if max(data) > 0 else 0.0  # Normalize
    
    def _round_cpu(self, cpu_value: float) -> float:
        """Round CPU value to appropriate step."""
        return round(cpu_value / self.constraints.cpu_step) * self.constraints.cpu_step
    
    def _round_memory(self, memory_value: float) -> float:
        """Round memory value to appropriate step."""
        return round(memory_value / self.constraints.memory_step) * self.constraints.memory_step


class KubernetesVerticalScaler:
    """
    Kubernetes-specific vertical scaling operations.
    """
    
    def __init__(self, namespace: str = "default"):
        self.namespace = namespace
        self.k8s_apps_v1 = None
        self.k8s_core_v1 = None
        
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
            self.k8s_core_v1 = client.CoreV1Api()
    
    def _cpu_to_k8s_format(self, cpu_cores: float) -> str:
        """Convert CPU cores to Kubernetes format."""
        if cpu_cores >= 1.0:
            return f"{cpu_cores:.1f}"
        else:
            return f"{int(cpu_cores * 1000)}m"
    
    def _memory_to_k8s_format(self, memory_bytes: float) -> str:
        """Convert memory bytes to Kubernetes format."""
        if memory_bytes >= 1024**3:  # GB
            return f"{int(memory_bytes / (1024**3))}Gi"
        elif memory_bytes >= 1024**2:  # MB
            return f"{int(memory_bytes / (1024**2))}Mi"
        else:  # KB
            return f"{int(memory_bytes / 1024)}Ki"
    
    async def get_pod_resources(self, pod_name: str) -> Dict[str, Any]:
        """Get current resource configuration for a pod."""
        if not self.k8s_core_v1:
            return {}
        
        try:
            pod = self.k8s_core_v1.read_namespaced_pod(name=pod_name, namespace=self.namespace)
            
            if not pod.spec.containers:
                return {}
            
            container = pod.spec.containers[0]  # First container
            resources = container.resources
            
            result = {}
            
            if resources.limits:
                result["limits"] = dict(resources.limits)
            
            if resources.requests:
                result["requests"] = dict(resources.requests)
            
            return result
            
        except ApiException as e:
            logger.error(f"Error getting pod resources: {e}")
            return {}
    
    async def update_deployment_resources(
        self,
        deployment_name: str,
        cpu_limit: Optional[float] = None,
        memory_limit: Optional[float] = None,
        cpu_request: Optional[float] = None,
        memory_request: Optional[float] = None
    ) -> bool:
        """Update resource limits for a deployment."""
        if not self.k8s_apps_v1:
            logger.warning("Kubernetes client not available")
            return False
        
        try:
            # Get current deployment
            deployment = self.k8s_apps_v1.read_namespaced_deployment(
                name=deployment_name,
                namespace=self.namespace
            )
            
            # Update resource specifications
            container = deployment.spec.template.spec.containers[0]
            
            if not container.resources:
                container.resources = client.V1ResourceRequirements()
            
            # Update limits
            if cpu_limit is not None or memory_limit is not None:
                if not container.resources.limits:
                    container.resources.limits = {}
                
                if cpu_limit is not None:
                    container.resources.limits["cpu"] = self._cpu_to_k8s_format(cpu_limit)
                
                if memory_limit is not None:
                    container.resources.limits["memory"] = self._memory_to_k8s_format(memory_limit)
            
            # Update requests
            if cpu_request is not None or memory_request is not None:
                if not container.resources.requests:
                    container.resources.requests = {}
                
                if cpu_request is not None:
                    container.resources.requests["cpu"] = self._cpu_to_k8s_format(cpu_request)
                
                if memory_request is not None:
                    container.resources.requests["memory"] = self._memory_to_k8s_format(memory_request)
            
            # Apply the update
            self.k8s_apps_v1.patch_namespaced_deployment(
                name=deployment_name,
                namespace=self.namespace,
                body=deployment
            )
            
            logger.info(f"Updated resources for deployment {deployment_name}")
            return True
            
        except ApiException as e:
            logger.error(f"Error updating deployment resources: {e}")
            return False


class VerticalAutoScaler:
    """
    Comprehensive vertical auto-scaling system.
    
    Monitors resource usage patterns and automatically adjusts CPU and memory
    allocations to optimize performance and cost.
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
        
        # Configuration
        self.check_interval = self.config.get("check_interval", 300.0)  # 5 minutes
        self.analysis_window = self.config.get("analysis_window", 24 * 60)  # 24 hours
        self.min_confidence = self.config.get("min_confidence", 0.7)
        self.dry_run = self.config.get("dry_run", False)
        self.enable_cpu_scaling = self.config.get("enable_cpu_scaling", True)
        self.enable_memory_scaling = self.config.get("enable_memory_scaling", True)
        
        # Components
        self.analyzer = ResourceAnalyzer(self.analysis_window)
        self.k8s_scaler = KubernetesVerticalScaler(
            namespace=self.config.get("kubernetes", {}).get("namespace", "default")
        )
        
        # State management
        self.running = False
        self.recommendations_history: List[ResourceRecommendation] = []
        self.applied_recommendations: List[ResourceRecommendation] = []
        self._tasks: List[asyncio.Task] = []
        
        logger.info("VerticalAutoScaler initialized")
    
    async def start(self):
        """Start the vertical auto-scaler."""
        if self.running:
            return
        
        self.running = True
        
        # Start background tasks
        self._tasks = [
            asyncio.create_task(self._metrics_collector()),
            asyncio.create_task(self._recommendation_engine()),
            asyncio.create_task(self._scaling_executor())
        ]
        
        await self.performance_monitor.record_metric(
            "vertical_scaler_started", 1, MetricType.COUNTER
        )
        
        logger.info("VerticalAutoScaler started")
    
    async def stop(self):
        """Stop the vertical auto-scaler."""
        if not self.running:
            return
        
        self.running = False
        
        # Cancel background tasks
        for task in self._tasks:
            task.cancel()
        
        if self._tasks:
            await asyncio.wait(self._tasks, return_when=asyncio.ALL_COMPLETED)
        
        await self.performance_monitor.record_metric(
            "vertical_scaler_stopped", 1, MetricType.COUNTER
        )
        
        logger.info("VerticalAutoScaler stopped")
    
    async def collect_worker_metrics(self, worker_id: str, pod_name: str) -> Optional[ResourceMetrics]:
        """Collect resource metrics for a worker."""
        try:
            # In a real implementation, this would collect metrics from:
            # - Kubernetes metrics server
            # - Prometheus
            # - Worker health endpoints
            # - System monitoring tools
            
            # For now, simulate metrics collection
            # This would be replaced with actual metrics collection
            
            # Get resource configuration from Kubernetes
            pod_resources = await self.k8s_scaler.get_pod_resources(pod_name)
            
            # Simulate current usage (in real implementation, get from metrics)
            import random
            cpu_limit = 2.0  # Default limit
            memory_limit = 4 * 1024 * 1024 * 1024  # 4GB default
            
            if pod_resources.get("limits"):
                limits = pod_resources["limits"]
                if "cpu" in limits:
                    cpu_str = limits["cpu"]
                    if cpu_str.endswith("m"):
                        cpu_limit = float(cpu_str[:-1]) / 1000
                    else:
                        cpu_limit = float(cpu_str)
                
                if "memory" in limits:
                    memory_str = limits["memory"]
                    if memory_str.endswith("Gi"):
                        memory_limit = float(memory_str[:-2]) * 1024**3
                    elif memory_str.endswith("Mi"):
                        memory_limit = float(memory_str[:-2]) * 1024**2
                    elif memory_str.endswith("Ki"):
                        memory_limit = float(memory_str[:-2]) * 1024
            
            # Simulate current usage with some randomness
            cpu_usage = cpu_limit * (0.3 + random.random() * 0.5)  # 30-80% usage
            memory_usage = memory_limit * (0.4 + random.random() * 0.4)  # 40-80% usage
            
            return ResourceMetrics(
                worker_id=worker_id,
                cpu_usage=cpu_usage,
                cpu_limit=cpu_limit,
                memory_usage=memory_usage,
                memory_limit=memory_limit,
                cpu_utilization=(cpu_usage / cpu_limit) * 100,
                memory_utilization=(memory_usage / memory_limit) * 100
            )
            
        except Exception as e:
            logger.error(f"Error collecting metrics for worker {worker_id}: {e}")
            return None
    
    async def apply_recommendation(self, recommendation: ResourceRecommendation) -> bool:
        """Apply a resource scaling recommendation."""
        if recommendation.confidence < self.min_confidence:
            logger.info(f"Skipping recommendation for {recommendation.worker_id}: low confidence {recommendation.confidence:.2f}")
            return False
        
        # Map worker ID to deployment name (in real implementation)
        deployment_name = f"voice-agent-{recommendation.worker_id}"
        
        success = False
        
        if not self.dry_run:
            if recommendation.resource_type == ResourceType.CPU and self.enable_cpu_scaling:
                success = await self.k8s_scaler.update_deployment_resources(
                    deployment_name,
                    cpu_limit=recommendation.recommended_value,
                    cpu_request=recommendation.recommended_value * 0.8  # Request 80% of limit
                )
            elif recommendation.resource_type == ResourceType.MEMORY and self.enable_memory_scaling:
                success = await self.k8s_scaler.update_deployment_resources(
                    deployment_name,
                    memory_limit=recommendation.recommended_value,
                    memory_request=recommendation.recommended_value * 0.8
                )
        else:
            logger.info(
                f"DRY RUN: Would scale {recommendation.resource_type.value} for {recommendation.worker_id} "
                f"from {recommendation.current_value} to {recommendation.recommended_value}"
            )
            success = True
        
        if success:
            self.applied_recommendations.append(recommendation)
            
            await self.performance_monitor.record_metric(
                "vertical_scaling_applied", 1, MetricType.COUNTER,
                labels={
                    "resource_type": recommendation.resource_type.value,
                    "direction": recommendation.direction.value,
                    "worker_id": recommendation.worker_id
                }
            )
            
            if recommendation.potential_savings > 0:
                await self.performance_monitor.record_metric(
                    "vertical_scaling_savings", recommendation.potential_savings, MetricType.GAUGE,
                    labels={"worker_id": recommendation.worker_id}
                )
            
            logger.info(
                f"Applied {recommendation.resource_type.value} scaling for {recommendation.worker_id}: "
                f"{recommendation.current_value} -> {recommendation.recommended_value} "
                f"({recommendation.reason})"
            )
        
        return success
    
    async def get_scaling_status(self) -> Dict[str, Any]:
        """Get current vertical scaling status."""
        recent_recommendations = self.recommendations_history[-20:]  # Last 20
        recent_applied = self.applied_recommendations[-10:]  # Last 10 applied
        
        total_savings = sum(r.potential_savings for r in self.applied_recommendations)
        
        return {
            "enabled": self.running,
            "cpu_scaling_enabled": self.enable_cpu_scaling,
            "memory_scaling_enabled": self.enable_memory_scaling,
            "dry_run": self.dry_run,
            "min_confidence": self.min_confidence,
            "total_applied_recommendations": len(self.applied_recommendations),
            "total_potential_savings": total_savings,
            "recent_recommendations": [
                {
                    "worker_id": r.worker_id,
                    "resource_type": r.resource_type.value,
                    "direction": r.direction.value,
                    "current_value": r.current_value,
                    "recommended_value": r.recommended_value,
                    "confidence": r.confidence,
                    "reason": r.reason,
                    "potential_savings": r.potential_savings
                } for r in recent_recommendations
            ],
            "recent_applied": [
                {
                    "worker_id": r.worker_id,
                    "resource_type": r.resource_type.value,
                    "direction": r.direction.value,
                    "old_value": r.current_value,
                    "new_value": r.recommended_value,
                    "savings": r.potential_savings
                } for r in recent_applied
            ]
        }
    
    async def _metrics_collector(self):
        """Background task to collect resource metrics."""
        while self.running:
            try:
                # In a real implementation, this would:
                # 1. Discover all worker pods
                # 2. Collect metrics for each worker
                # 3. Add metrics to analyzer
                
                # Simulated worker discovery and metrics collection
                simulated_workers = ["worker-1", "worker-2", "worker-3"]
                
                for worker_id in simulated_workers:
                    pod_name = f"voice-agent-{worker_id}-pod"
                    metrics = await self.collect_worker_metrics(worker_id, pod_name)
                    
                    if metrics:
                        self.analyzer.add_metrics(metrics)
                        
                        # Record metrics
                        await self.performance_monitor.record_metric(
                            "worker_cpu_utilization", metrics.cpu_utilization, MetricType.GAUGE,
                            labels={"worker_id": worker_id}
                        )
                        
                        await self.performance_monitor.record_metric(
                            "worker_memory_utilization", metrics.memory_utilization, MetricType.GAUGE,
                            labels={"worker_id": worker_id}
                        )
                
                await asyncio.sleep(60.0)  # Collect metrics every minute
                
            except Exception as e:
                logger.error(f"Error in metrics collector: {e}")
                await asyncio.sleep(60.0)
    
    async def _recommendation_engine(self):
        """Background task to generate scaling recommendations."""
        while self.running:
            try:
                # Generate recommendations for all workers
                all_workers = list(self.analyzer.metrics_history.keys())
                
                for worker_id in all_workers:
                    recommendations = self.analyzer.get_recommendations(worker_id)
                    
                    for recommendation in recommendations:
                        self.recommendations_history.append(recommendation)
                        
                        logger.info(
                            f"Generated recommendation for {worker_id}: "
                            f"{recommendation.resource_type.value} {recommendation.direction.value} "
                            f"from {recommendation.current_value} to {recommendation.recommended_value} "
                            f"(confidence: {recommendation.confidence:.2f})"
                        )
                
                # Cleanup old recommendations
                cutoff_time = datetime.now() - timedelta(hours=24)
                self.recommendations_history = [
                    r for r in self.recommendations_history
                    if hasattr(r, 'timestamp') and r.timestamp > cutoff_time  # Add timestamp if needed
                ]
                
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in recommendation engine: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _scaling_executor(self):
        """Background task to execute scaling recommendations."""
        while self.running:
            try:
                # Process recent recommendations
                recent_recommendations = [
                    r for r in self.recommendations_history
                    if r not in self.applied_recommendations
                ]
                
                # Group by worker and resource type to avoid conflicts
                pending_by_worker = {}
                for rec in recent_recommendations:
                    key = (rec.worker_id, rec.resource_type)
                    if key not in pending_by_worker:
                        pending_by_worker[key] = []
                    pending_by_worker[key].append(rec)
                
                # Apply highest confidence recommendation for each worker/resource
                for (worker_id, resource_type), recommendations in pending_by_worker.items():
                    if recommendations:
                        # Sort by confidence and take the highest
                        best_recommendation = max(recommendations, key=lambda r: r.confidence)
                        
                        if best_recommendation.confidence >= self.min_confidence:
                            await self.apply_recommendation(best_recommendation)
                
                await asyncio.sleep(self.check_interval * 2)  # Execute less frequently than recommendations
                
            except Exception as e:
                logger.error(f"Error in scaling executor: {e}")
                await asyncio.sleep(self.check_interval * 2)


# Factory function
def create_vertical_scaler(config: Optional[Dict[str, Any]] = None) -> VerticalAutoScaler:
    """Create a new VerticalAutoScaler instance."""
    return VerticalAutoScaler(config)


# Configuration helpers
def create_resource_constraints(
    min_cpu: float = 0.1,
    max_cpu: float = 16.0,
    min_memory_gb: float = 0.125,  # 128MB
    max_memory_gb: float = 32.0
) -> ResourceConstraints:
    """Helper to create resource constraints."""
    return ResourceConstraints(
        min_cpu=min_cpu,
        max_cpu=max_cpu,
        min_memory=min_memory_gb * 1024**3,
        max_memory=max_memory_gb * 1024**3
    )