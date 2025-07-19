"""
Performance Monitoring System for Voice Pipeline Components

This module provides comprehensive performance monitoring, metrics collection,
and analysis for all voice processing pipeline components.
"""

import asyncio
import logging
import time
import psutil
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List, Callable, Union
from collections import defaultdict, deque
import statistics
import json
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics that can be collected."""
    COUNTER = "counter"           # Incrementing counter
    GAUGE = "gauge"              # Current value
    HISTOGRAM = "histogram"       # Distribution of values
    TIMER = "timer"              # Duration measurements
    RATE = "rate"                # Rate of occurrence


@dataclass
class MetricValue:
    """A single metric measurement."""
    value: Union[int, float]
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Performance metrics for a component or operation."""
    # Timing metrics
    total_duration: float = 0.0
    processing_time: float = 0.0
    queue_time: float = 0.0
    network_time: float = 0.0
    
    # Throughput metrics
    requests_per_second: float = 0.0
    bytes_per_second: float = 0.0
    items_processed: int = 0
    
    # Quality metrics
    success_rate: float = 1.0
    error_rate: float = 0.0
    retry_rate: float = 0.0
    
    # Resource metrics
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_io: float = 0.0
    network_io: float = 0.0
    
    # Component-specific metrics
    audio_quality_score: Optional[float] = None
    latency_p95: Optional[float] = None
    latency_p99: Optional[float] = None
    
    # Additional metadata
    component: Optional[str] = None
    provider: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


class PerformanceMonitor:
    """Comprehensive performance monitoring system."""
    
    def __init__(self, max_history_size: int = 10000):
        self.max_history_size = max_history_size
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history_size))
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.timers: Dict[str, List[float]] = defaultdict(list)
        
        # Component-specific monitors
        self.component_monitors: Dict[str, 'ComponentMonitor'] = {}
        
        # System resource monitoring
        self.system_monitor = SystemResourceMonitor()
        self.is_monitoring = False
        self.monitoring_interval = 1.0  # seconds
        self._monitoring_task = None
        
        # Alerting thresholds
        self.alert_thresholds = {
            "latency_p95_ms": 500,     # 500ms
            "error_rate": 0.05,        # 5%
            "cpu_usage": 0.8,          # 80%
            "memory_usage": 0.9,       # 90%
            "success_rate": 0.95       # 95%
        }
        
        # Alert callbacks
        self.alert_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
    
    async def start_monitoring(self):
        """Start continuous monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Performance monitoring started")
    
    async def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.is_monitoring = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Performance monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect system metrics
                system_metrics = self.system_monitor.get_metrics()
                self.record_metrics("system", system_metrics)
                
                # Check for alerts
                await self._check_alerts()
                
                # Cleanup old data
                self._cleanup_old_data()
                
                await asyncio.sleep(self.monitoring_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    def register_component(self, component_name: str, component_type: str = None) -> 'ComponentMonitor':
        """Register a component for monitoring."""
        if component_name not in self.component_monitors:
            self.component_monitors[component_name] = ComponentMonitor(
                component_name, component_type, self
            )
        
        return self.component_monitors[component_name]
    
    def record_metric(
        self, 
        metric_name: str, 
        value: Union[int, float], 
        metric_type: MetricType = MetricType.GAUGE,
        labels: Dict[str, str] = None,
        component: str = None
    ):
        """Record a single metric value."""
        labels = labels or {}
        if component:
            labels["component"] = component
        
        metric_value = MetricValue(value=value, labels=labels)
        
        full_metric_name = f"{component}.{metric_name}" if component else metric_name
        
        if metric_type == MetricType.COUNTER:
            self.counters[full_metric_name] += value
        elif metric_type == MetricType.GAUGE:
            self.gauges[full_metric_name] = value
        elif metric_type == MetricType.HISTOGRAM:
            self.histograms[full_metric_name].append(value)
        elif metric_type == MetricType.TIMER:
            self.timers[full_metric_name].append(value)
        
        # Add to history
        self.metrics_history[full_metric_name].append(metric_value)
    
    def record_metrics(self, component: str, metrics: Dict[str, Any]):
        """Record multiple metrics for a component."""
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.record_metric(metric_name, value, component=component)
    
    def get_metric_stats(self, metric_name: str, time_window_seconds: int = 3600) -> Dict[str, Any]:
        """Get statistical analysis of a metric over a time window."""
        current_time = time.time()
        cutoff_time = current_time - time_window_seconds
        
        if metric_name not in self.metrics_history:
            return {}
        
        # Filter values within time window
        recent_values = [
            mv.value for mv in self.metrics_history[metric_name]
            if mv.timestamp >= cutoff_time
        ]
        
        if not recent_values:
            return {}
        
        return {
            "count": len(recent_values),
            "min": min(recent_values),
            "max": max(recent_values),
            "mean": statistics.mean(recent_values),
            "median": statistics.median(recent_values),
            "p95": self._percentile(recent_values, 95),
            "p99": self._percentile(recent_values, 99),
            "stddev": statistics.stdev(recent_values) if len(recent_values) > 1 else 0,
            "time_window": time_window_seconds,
            "latest_value": recent_values[-1] if recent_values else None
        }
    
    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile value."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = (percentile / 100) * (len(sorted_values) - 1)
        
        if index.is_integer():
            return sorted_values[int(index)]
        else:
            lower_index = int(index)
            upper_index = lower_index + 1
            if upper_index < len(sorted_values):
                lower_value = sorted_values[lower_index]
                upper_value = sorted_values[upper_index]
                return lower_value + (upper_value - lower_value) * (index - lower_index)
            else:
                return sorted_values[lower_index]
    
    async def _check_alerts(self):
        """Check if any metrics exceed alert thresholds."""
        for threshold_name, threshold_value in self.alert_thresholds.items():
            # Check various metric patterns
            metric_stats = self.get_metric_stats(threshold_name, time_window_seconds=300)  # 5 minutes
            
            if not metric_stats:
                continue
            
            current_value = metric_stats.get("latest_value")
            if current_value is None:
                continue
            
            # Check threshold violation
            violated = False
            if threshold_name.endswith("_rate") or threshold_name.endswith("_usage"):
                violated = current_value > threshold_value
            elif threshold_name == "success_rate":
                violated = current_value < threshold_value
            
            if violated:
                alert_data = {
                    "metric": threshold_name,
                    "current_value": current_value,
                    "threshold": threshold_value,
                    "timestamp": time.time(),
                    "stats": metric_stats
                }
                
                await self._trigger_alert(threshold_name, alert_data)
    
    async def _trigger_alert(self, alert_type: str, alert_data: Dict[str, Any]):
        """Trigger an alert notification."""
        logger.warning(f"Performance alert: {alert_type} = {alert_data['current_value']} (threshold: {alert_data['threshold']})")
        
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert_type, alert_data)
                else:
                    callback(alert_type, alert_data)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def register_alert_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Register callback for alert notifications."""
        self.alert_callbacks.append(callback)
    
    def _cleanup_old_data(self):
        """Clean up old metric data to prevent memory leaks."""
        current_time = time.time()
        cutoff_time = current_time - 86400  # 24 hours
        
        for metric_name, history in self.metrics_history.items():
            # Remove old entries
            while history and history[0].timestamp < cutoff_time:
                history.popleft()
        
        # Clean up timers
        for timer_name, values in self.timers.items():
            if len(values) > 1000:  # Keep only recent 1000 values
                self.timers[timer_name] = values[-1000:]
    
    def get_component_summary(self, component_name: str) -> Dict[str, Any]:
        """Get performance summary for a specific component."""
        if component_name not in self.component_monitors:
            return {}
        
        return self.component_monitors[component_name].get_summary()
    
    def get_overall_summary(self) -> Dict[str, Any]:
        """Get overall system performance summary."""
        summary = {
            "timestamp": time.time(),
            "monitoring_active": self.is_monitoring,
            "components": {},
            "system_metrics": self.system_monitor.get_metrics(),
            "alert_thresholds": self.alert_thresholds,
            "total_metrics": len(self.metrics_history)
        }
        
        # Add component summaries
        for component_name, monitor in self.component_monitors.items():
            summary["components"][component_name] = monitor.get_summary()
        
        return summary
    
    def export_metrics(self, format_type: str = "json") -> str:
        """Export metrics in specified format."""
        if format_type == "json":
            return json.dumps(self.get_overall_summary(), indent=2, default=str)
        elif format_type == "prometheus":
            return self._export_prometheus_format()
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def _export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        # Export counters
        for metric_name, value in self.counters.items():
            lines.append(f"# TYPE {metric_name} counter")
            lines.append(f"{metric_name} {value}")
        
        # Export gauges
        for metric_name, value in self.gauges.items():
            lines.append(f"# TYPE {metric_name} gauge")
            lines.append(f"{metric_name} {value}")
        
        return "\n".join(lines)


class ComponentMonitor:
    """Monitor for individual pipeline components."""
    
    def __init__(self, component_name: str, component_type: str, parent_monitor: PerformanceMonitor):
        self.component_name = component_name
        self.component_type = component_type
        self.parent_monitor = parent_monitor
        
        # Component-specific metrics
        self.operation_timings: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.operation_counts: Dict[str, int] = defaultdict(int)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.success_counts: Dict[str, int] = defaultdict(int)
        
        # Real-time statistics
        self.last_operation_time = None
        self.operations_in_progress = 0
        self.total_operations = 0
        self.total_errors = 0
    
    @asynccontextmanager
    async def monitor_operation(self, operation_name: str, **metadata):
        """Context manager to monitor an operation."""
        start_time = time.time()
        self.operations_in_progress += 1
        operation_id = f"{self.component_name}.{operation_name}"
        
        try:
            yield
            
            # Operation succeeded
            end_time = time.time()
            duration = end_time - start_time
            
            self._record_operation_success(operation_name, duration, metadata)
            
        except Exception as e:
            # Operation failed
            end_time = time.time()
            duration = end_time - start_time
            
            self._record_operation_error(operation_name, duration, str(e), metadata)
            raise
        
        finally:
            self.operations_in_progress -= 1
            self.last_operation_time = end_time
    
    def _record_operation_success(self, operation_name: str, duration: float, metadata: Dict[str, Any]):
        """Record successful operation."""
        self.operation_timings[operation_name].append(duration)
        self.operation_counts[operation_name] += 1
        self.success_counts[operation_name] += 1
        self.total_operations += 1
        
        # Record in parent monitor
        self.parent_monitor.record_metric(
            f"{operation_name}.duration",
            duration * 1000,  # Convert to milliseconds
            MetricType.TIMER,
            component=self.component_name
        )
        
        self.parent_monitor.record_metric(
            f"{operation_name}.count",
            1,
            MetricType.COUNTER,
            component=self.component_name
        )
    
    def _record_operation_error(self, operation_name: str, duration: float, error: str, metadata: Dict[str, Any]):
        """Record failed operation."""
        self.operation_timings[operation_name].append(duration)
        self.operation_counts[operation_name] += 1
        self.error_counts[operation_name] += 1
        self.total_operations += 1
        self.total_errors += 1
        
        # Record in parent monitor
        self.parent_monitor.record_metric(
            f"{operation_name}.error_count",
            1,
            MetricType.COUNTER,
            component=self.component_name
        )
    
    def get_operation_stats(self, operation_name: str) -> Dict[str, Any]:
        """Get statistics for a specific operation."""
        if operation_name not in self.operation_timings:
            return {}
        
        timings = list(self.operation_timings[operation_name])
        if not timings:
            return {}
        
        total_count = self.operation_counts[operation_name]
        error_count = self.error_counts[operation_name]
        success_count = self.success_counts[operation_name]
        
        return {
            "operation": operation_name,
            "total_count": total_count,
            "success_count": success_count,
            "error_count": error_count,
            "success_rate": success_count / total_count if total_count > 0 else 0,
            "error_rate": error_count / total_count if total_count > 0 else 0,
            "avg_duration_ms": statistics.mean(timings) * 1000,
            "min_duration_ms": min(timings) * 1000,
            "max_duration_ms": max(timings) * 1000,
            "p95_duration_ms": self.parent_monitor._percentile(timings, 95) * 1000,
            "p99_duration_ms": self.parent_monitor._percentile(timings, 99) * 1000,
            "operations_per_second": self._calculate_ops_per_second(operation_name)
        }
    
    def _calculate_ops_per_second(self, operation_name: str, window_seconds: int = 60) -> float:
        """Calculate operations per second for the last window."""
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        recent_ops = 0
        for metric_value in self.parent_monitor.metrics_history.get(f"{self.component_name}.{operation_name}.count", []):
            if metric_value.timestamp >= cutoff_time:
                recent_ops += metric_value.value
        
        return recent_ops / window_seconds
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of component performance."""
        summary = {
            "component_name": self.component_name,
            "component_type": self.component_type,
            "total_operations": self.total_operations,
            "total_errors": self.total_errors,
            "overall_success_rate": (self.total_operations - self.total_errors) / self.total_operations if self.total_operations > 0 else 0,
            "operations_in_progress": self.operations_in_progress,
            "last_operation_time": self.last_operation_time,
            "operations": {}
        }
        
        # Add stats for each operation
        for operation_name in self.operation_counts.keys():
            summary["operations"][operation_name] = self.get_operation_stats(operation_name)
        
        return summary


class SystemResourceMonitor:
    """Monitor system resource usage."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.last_cpu_time = None
        self.last_measurement_time = None
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current system resource metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            process_cpu = self.process.cpu_percent()
            
            # Memory usage
            memory = psutil.virtual_memory()
            process_memory = self.process.memory_info()
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            process_io = self.process.io_counters() if hasattr(self.process, 'io_counters') else None
            
            # Network I/O
            network_io = psutil.net_io_counters()
            
            metrics = {
                "cpu_usage_percent": cpu_percent,
                "process_cpu_percent": process_cpu,
                "memory_usage_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "process_memory_mb": process_memory.rss / (1024**2),
                "disk_read_mb": disk_io.read_bytes / (1024**2) if disk_io else 0,
                "disk_write_mb": disk_io.write_bytes / (1024**2) if disk_io else 0,
                "network_sent_mb": network_io.bytes_sent / (1024**2),
                "network_recv_mb": network_io.bytes_recv / (1024**2)
            }
            
            # Process-specific I/O if available
            if process_io:
                metrics.update({
                    "process_read_mb": process_io.read_bytes / (1024**2),
                    "process_write_mb": process_io.write_bytes / (1024**2)
                })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {}


# Global performance monitor instance
global_performance_monitor = PerformanceMonitor()


def monitor_performance(component: str = None, operation: str = None):
    """Decorator for automatic performance monitoring."""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                # Get component name from instance if available
                comp_name = component
                if not comp_name and args and hasattr(args[0], 'provider_name'):
                    comp_name = args[0].provider_name
                elif not comp_name and args and hasattr(args[0], '__class__'):
                    comp_name = args[0].__class__.__name__
                
                op_name = operation or func.__name__
                
                if comp_name:
                    monitor = global_performance_monitor.register_component(comp_name)
                    async with monitor.monitor_operation(op_name):
                        return await func(*args, **kwargs)
                else:
                    return await func(*args, **kwargs)
            
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                # For sync functions, just record timing
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    comp_name = component
                    if not comp_name and args and hasattr(args[0], 'provider_name'):
                        comp_name = args[0].provider_name
                    
                    if comp_name:
                        global_performance_monitor.record_metric(
                            f"{operation or func.__name__}.duration",
                            duration * 1000,
                            MetricType.TIMER,
                            component=comp_name
                        )
                    
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    
                    comp_name = component
                    if not comp_name and args and hasattr(args[0], 'provider_name'):
                        comp_name = args[0].provider_name
                    
                    if comp_name:
                        global_performance_monitor.record_metric(
                            f"{operation or func.__name__}.error_count",
                            1,
                            MetricType.COUNTER,
                            component=comp_name
                        )
                    raise
            
            return sync_wrapper
    return decorator