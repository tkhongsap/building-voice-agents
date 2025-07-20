"""
Performance Profiler and Optimization Tools

Advanced performance monitoring, profiling, and optimization tools for voice agents.
Provides detailed insights into component performance, memory usage, and bottlenecks.

Features:
- Real-time performance monitoring
- Component-level profiling
- Memory usage tracking
- Latency analysis and optimization
- Automatic bottleneck detection
- Performance recommendations
- Resource utilization monitoring
- Benchmark comparison tools

Usage:
    from monitoring.performance_profiler import PerformanceProfiler
    
    profiler = PerformanceProfiler()
    agent.add_profiler(profiler)
    
    # Start profiling
    await profiler.start_profiling()
    
    # Get performance report
    report = await profiler.generate_report()
"""

import asyncio
import time
import sys
import gc
import threading
import tracemalloc
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from contextlib import asynccontextmanager
import statistics
import json
from pathlib import Path
import weakref


@dataclass
class PerformanceMetric:
    """Individual performance metric."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    component: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComponentProfile:
    """Performance profile for a component."""
    component_name: str
    total_calls: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    avg_time: float = 0.0
    p95_time: float = 0.0
    p99_time: float = 0.0
    error_count: int = 0
    error_rate: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    recent_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    def update(self, execution_time: float, error: bool = False) -> None:
        """Update profile with new execution data."""
        self.total_calls += 1
        self.total_time += execution_time
        self.recent_times.append(execution_time)
        
        if error:
            self.error_count += 1
        
        # Update timing statistics
        self.min_time = min(self.min_time, execution_time)
        self.max_time = max(self.max_time, execution_time)
        self.avg_time = self.total_time / self.total_calls
        self.error_rate = self.error_count / self.total_calls
        
        # Calculate percentiles from recent times
        if len(self.recent_times) > 10:
            sorted_times = sorted(self.recent_times)
            self.p95_time = sorted_times[int(len(sorted_times) * 0.95)]
            self.p99_time = sorted_times[int(len(sorted_times) * 0.99)]


@dataclass
class SystemResourceMetrics:
    """System resource usage metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_usage_mb: float
    memory_percent: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_io_sent_mb: float
    network_io_recv_mb: float
    open_files: int
    thread_count: int


@dataclass
class MemorySnapshot:
    """Memory usage snapshot."""
    timestamp: datetime
    total_memory_mb: float
    used_memory_mb: float
    free_memory_mb: float
    buffer_cache_mb: float
    swap_used_mb: float
    gc_collections: Dict[int, int]
    gc_objects: int
    tracemalloc_snapshot: Optional[Any] = None


class PerformanceTimer:
    """Context manager for timing operations."""
    
    def __init__(self, profiler: 'PerformanceProfiler', component: str, operation: str):
        self.profiler = profiler
        self.component = component
        self.operation = operation
        self.start_time = 0
        self.error_occurred = False
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.perf_counter()
        execution_time = (end_time - self.start_time) * 1000  # Convert to milliseconds
        
        self.error_occurred = exc_type is not None
        
        # Record timing
        self.profiler.record_timing(
            self.component, 
            self.operation, 
            execution_time, 
            error=self.error_occurred
        )
        
        return False  # Don't suppress exceptions


class AsyncPerformanceTimer:
    """Async context manager for timing operations."""
    
    def __init__(self, profiler: 'PerformanceProfiler', component: str, operation: str):
        self.profiler = profiler
        self.component = component
        self.operation = operation
        self.start_time = 0
        self.error_occurred = False
    
    async def __aenter__(self):
        self.start_time = time.perf_counter()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        end_time = time.perf_counter()
        execution_time = (end_time - self.start_time) * 1000
        
        self.error_occurred = exc_type is not None
        
        # Record timing
        self.profiler.record_timing(
            self.component, 
            self.operation, 
            execution_time, 
            error=self.error_occurred
        )
        
        return False


class PerformanceProfiler:
    """
    Advanced performance profiler for voice agents.
    
    Monitors component performance, system resources, and provides
    optimization recommendations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Profiling state
        self.is_profiling = False
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        
        # Component profiles
        self.component_profiles: Dict[str, ComponentProfile] = {}
        
        # System monitoring
        self.system_metrics: List[SystemResourceMetrics] = []
        self.memory_snapshots: List[MemorySnapshot] = []
        
        # Performance metrics
        self.metrics: List[PerformanceMetric] = []
        self.metric_callbacks: List[Callable] = []
        
        # Configuration
        self.sample_interval = config.get("sample_interval", 1.0)  # seconds
        self.enable_memory_profiling = config.get("enable_memory_profiling", True)
        self.enable_system_monitoring = config.get("enable_system_monitoring", True)
        self.max_metrics_history = config.get("max_metrics_history", 10000)
        
        # Monitoring task
        self.monitor_task: Optional[asyncio.Task] = None
        
        # Process info
        self.process = psutil.Process()
        
        # Weak references to avoid memory leaks
        self.agents: weakref.WeakSet = weakref.WeakSet()
    
    async def start_profiling(self) -> None:
        """Start performance profiling."""
        if self.is_profiling:
            return
        
        self.is_profiling = True
        self.start_time = datetime.now()
        
        # Start memory tracing if enabled
        if self.enable_memory_profiling:
            tracemalloc.start()
        
        # Start system monitoring
        if self.enable_system_monitoring:
            self.monitor_task = asyncio.create_task(self._monitor_system_resources())
        
        print(f"🔍 Performance profiling started")
    
    async def stop_profiling(self) -> None:
        """Stop performance profiling."""
        if not self.is_profiling:
            return
        
        self.is_profiling = False
        self.end_time = datetime.now()
        
        # Stop system monitoring
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        # Stop memory tracing
        if self.enable_memory_profiling:
            tracemalloc.stop()
        
        print(f"🔍 Performance profiling stopped")
    
    def add_agent(self, agent) -> None:
        """Add agent for monitoring."""
        self.agents.add(agent)
    
    def record_timing(
        self,
        component: str,
        operation: str,
        execution_time: float,
        error: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record timing for a component operation."""
        full_component_name = f"{component}.{operation}"
        
        # Update component profile
        if full_component_name not in self.component_profiles:
            self.component_profiles[full_component_name] = ComponentProfile(full_component_name)
        
        self.component_profiles[full_component_name].update(execution_time, error)
        
        # Record metric
        metric = PerformanceMetric(
            name=f"{component}_latency",
            value=execution_time,
            unit="ms",
            timestamp=datetime.now(),
            component=component,
            metadata=metadata or {}
        )
        
        self.metrics.append(metric)
        
        # Trim metrics if needed
        if len(self.metrics) > self.max_metrics_history:
            self.metrics = self.metrics[-self.max_metrics_history:]
        
        # Trigger callbacks
        for callback in self.metric_callbacks:
            try:
                callback(metric)
            except Exception as e:
                print(f"⚠️ Metric callback error: {e}")
    
    def record_metric(
        self,
        name: str,
        value: float,
        unit: str = "",
        component: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a custom performance metric."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=datetime.now(),
            component=component,
            metadata=metadata or {}
        )
        
        self.metrics.append(metric)
        
        # Trigger callbacks
        for callback in self.metric_callbacks:
            try:
                callback(metric)
            except Exception as e:
                print(f"⚠️ Metric callback error: {e}")
    
    def timer(self, component: str, operation: str) -> PerformanceTimer:
        """Create a timing context manager."""
        return PerformanceTimer(self, component, operation)
    
    def async_timer(self, component: str, operation: str) -> AsyncPerformanceTimer:
        """Create an async timing context manager."""
        return AsyncPerformanceTimer(self, component, operation)
    
    def add_metric_callback(self, callback: Callable) -> None:
        """Add callback for metric events."""
        self.metric_callbacks.append(callback)
    
    async def _monitor_system_resources(self) -> None:
        """Monitor system resource usage."""
        last_disk_io = self.process.io_counters()
        last_net_io = psutil.net_io_counters()
        
        while self.is_profiling:
            try:
                # Get current metrics
                cpu_percent = self.process.cpu_percent()
                memory_info = self.process.memory_info()
                
                # System memory
                system_memory = psutil.virtual_memory()
                
                # Disk I/O
                current_disk_io = self.process.io_counters()
                disk_read_mb = (current_disk_io.read_bytes - last_disk_io.read_bytes) / (1024 * 1024)
                disk_write_mb = (current_disk_io.write_bytes - last_disk_io.write_bytes) / (1024 * 1024)
                last_disk_io = current_disk_io
                
                # Network I/O
                current_net_io = psutil.net_io_counters()
                net_sent_mb = (current_net_io.bytes_sent - last_net_io.bytes_sent) / (1024 * 1024)
                net_recv_mb = (current_net_io.bytes_recv - last_net_io.bytes_recv) / (1024 * 1024)
                last_net_io = current_net_io
                
                # Create metrics object
                metrics = SystemResourceMetrics(
                    timestamp=datetime.now(),
                    cpu_percent=cpu_percent,
                    memory_usage_mb=memory_info.rss / (1024 * 1024),
                    memory_percent=system_memory.percent,
                    disk_io_read_mb=disk_read_mb,
                    disk_io_write_mb=disk_write_mb,
                    network_io_sent_mb=net_sent_mb,
                    network_io_recv_mb=net_recv_mb,
                    open_files=self.process.num_fds() if hasattr(self.process, 'num_fds') else 0,
                    thread_count=self.process.num_threads()
                )
                
                self.system_metrics.append(metrics)
                
                # Trim history
                if len(self.system_metrics) > 1000:
                    self.system_metrics = self.system_metrics[-1000:]
                
                # Take memory snapshot periodically
                if len(self.memory_snapshots) == 0 or \
                   (datetime.now() - self.memory_snapshots[-1].timestamp) > timedelta(minutes=5):
                    await self._take_memory_snapshot()
                
            except Exception as e:
                print(f"⚠️ System monitoring error: {e}")
            
            await asyncio.sleep(self.sample_interval)
    
    async def _take_memory_snapshot(self) -> None:
        """Take a memory usage snapshot."""
        if not self.enable_memory_profiling:
            return
        
        try:
            # System memory
            system_memory = psutil.virtual_memory()
            swap_memory = psutil.swap_memory()
            
            # Garbage collection stats
            gc_stats = {i: gc.get_count()[i] for i in range(len(gc.get_count()))}
            
            # Memory snapshot
            snapshot = MemorySnapshot(
                timestamp=datetime.now(),
                total_memory_mb=system_memory.total / (1024 * 1024),
                used_memory_mb=system_memory.used / (1024 * 1024),
                free_memory_mb=system_memory.free / (1024 * 1024),
                buffer_cache_mb=(system_memory.buffers + system_memory.cached) / (1024 * 1024),
                swap_used_mb=swap_memory.used / (1024 * 1024),
                gc_collections=gc_stats,
                gc_objects=len(gc.get_objects())
            )
            
            # Add tracemalloc snapshot if available
            if tracemalloc.is_tracing():
                snapshot.tracemalloc_snapshot = tracemalloc.take_snapshot()
            
            self.memory_snapshots.append(snapshot)
            
            # Trim history
            if len(self.memory_snapshots) > 100:
                self.memory_snapshots = self.memory_snapshots[-100:]
                
        except Exception as e:
            print(f"⚠️ Memory snapshot error: {e}")
    
    def get_component_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all components."""
        summary = {}
        
        for name, profile in self.component_profiles.items():
            summary[name] = {
                "total_calls": profile.total_calls,
                "avg_time_ms": profile.avg_time,
                "min_time_ms": profile.min_time,
                "max_time_ms": profile.max_time,
                "p95_time_ms": profile.p95_time,
                "p99_time_ms": profile.p99_time,
                "error_rate": profile.error_rate,
                "total_time_sec": profile.total_time / 1000
            }
        
        return summary
    
    def get_system_summary(self) -> Dict[str, float]:
        """Get system resource usage summary."""
        if not self.system_metrics:
            return {}
        
        recent_metrics = self.system_metrics[-10:]  # Last 10 samples
        
        return {
            "avg_cpu_percent": statistics.mean(m.cpu_percent for m in recent_metrics),
            "max_cpu_percent": max(m.cpu_percent for m in recent_metrics),
            "avg_memory_mb": statistics.mean(m.memory_usage_mb for m in recent_metrics),
            "max_memory_mb": max(m.memory_usage_mb for m in recent_metrics),
            "avg_memory_percent": statistics.mean(m.memory_percent for m in recent_metrics),
            "thread_count": recent_metrics[-1].thread_count,
            "open_files": recent_metrics[-1].open_files
        }
    
    def detect_bottlenecks(self) -> List[Dict[str, Any]]:
        """Detect performance bottlenecks."""
        bottlenecks = []
        
        # Check component latencies
        for name, profile in self.component_profiles.items():
            # High average latency
            if profile.avg_time > 1000:  # 1 second
                bottlenecks.append({
                    "type": "high_latency",
                    "component": name,
                    "severity": "high" if profile.avg_time > 3000 else "medium",
                    "metric": "avg_time_ms",
                    "value": profile.avg_time,
                    "threshold": 1000,
                    "description": f"Component {name} has high average latency"
                })
            
            # High error rate
            if profile.error_rate > 0.05:  # 5%
                bottlenecks.append({
                    "type": "high_error_rate",
                    "component": name,
                    "severity": "high" if profile.error_rate > 0.1 else "medium",
                    "metric": "error_rate",
                    "value": profile.error_rate,
                    "threshold": 0.05,
                    "description": f"Component {name} has high error rate"
                })
            
            # High P99 latency
            if profile.p99_time > 2000:  # 2 seconds
                bottlenecks.append({
                    "type": "high_p99_latency",
                    "component": name,
                    "severity": "medium",
                    "metric": "p99_time_ms",
                    "value": profile.p99_time,
                    "threshold": 2000,
                    "description": f"Component {name} has high P99 latency"
                })
        
        # Check system resources
        if self.system_metrics:
            recent_metrics = self.system_metrics[-10:]
            avg_cpu = statistics.mean(m.cpu_percent for m in recent_metrics)
            max_memory_percent = max(m.memory_percent for m in recent_metrics)
            
            # High CPU usage
            if avg_cpu > 80:
                bottlenecks.append({
                    "type": "high_cpu_usage",
                    "component": "system",
                    "severity": "high" if avg_cpu > 90 else "medium",
                    "metric": "cpu_percent",
                    "value": avg_cpu,
                    "threshold": 80,
                    "description": f"High CPU usage detected"
                })
            
            # High memory usage
            if max_memory_percent > 85:
                bottlenecks.append({
                    "type": "high_memory_usage",
                    "component": "system",
                    "severity": "high" if max_memory_percent > 95 else "medium",
                    "metric": "memory_percent",
                    "value": max_memory_percent,
                    "threshold": 85,
                    "description": f"High memory usage detected"
                })
        
        return bottlenecks
    
    def generate_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        bottlenecks = self.detect_bottlenecks()
        
        # Component-specific recommendations
        for bottleneck in bottlenecks:
            if bottleneck["type"] == "high_latency":
                component = bottleneck["component"]
                if "stt" in component.lower():
                    recommendations.append(f"Consider using a faster STT model or enabling streaming for {component}")
                elif "llm" in component.lower():
                    recommendations.append(f"Consider using a smaller/faster LLM model for {component}")
                elif "tts" in component.lower():
                    recommendations.append(f"Consider enabling streaming TTS for {component}")
                else:
                    recommendations.append(f"Optimize {component} - consider caching or async processing")
            
            elif bottleneck["type"] == "high_error_rate":
                component = bottleneck["component"]
                recommendations.append(f"Investigate and fix errors in {component} - add better error handling")
            
            elif bottleneck["type"] == "high_cpu_usage":
                recommendations.append("Consider scaling horizontally or upgrading CPU resources")
            
            elif bottleneck["type"] == "high_memory_usage":
                recommendations.append("Consider increasing memory or optimizing memory usage")
        
        # General optimization recommendations
        component_summary = self.get_component_summary()
        
        # Check for sequential processing opportunities
        total_latency = sum(profile["avg_time_ms"] for profile in component_summary.values())
        if total_latency > 5000:  # 5 seconds
            recommendations.append("Consider parallelizing component processing to reduce total latency")
        
        # Check for caching opportunities
        for name, profile in component_summary.items():
            if profile["total_calls"] > 100 and profile["avg_time_ms"] > 500:
                recommendations.append(f"Consider implementing caching for {name}")
        
        # Memory optimization
        if self.memory_snapshots:
            latest_snapshot = self.memory_snapshots[-1]
            if latest_snapshot.gc_objects > 100000:
                recommendations.append("High number of objects detected - consider memory optimization")
        
        return recommendations
    
    async def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.start_time:
            return {"error": "Profiling not started"}
        
        duration = (self.end_time or datetime.now()) - self.start_time
        
        report = {
            "summary": {
                "profiling_duration": str(duration),
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat() if self.end_time else None,
                "total_components": len(self.component_profiles),
                "total_metrics": len(self.metrics),
                "system_samples": len(self.system_metrics)
            },
            "component_performance": self.get_component_summary(),
            "system_resources": self.get_system_summary(),
            "bottlenecks": self.detect_bottlenecks(),
            "recommendations": self.generate_optimization_recommendations(),
            "detailed_metrics": {
                "by_component": self._get_metrics_by_component(),
                "latency_trends": self._get_latency_trends(),
                "error_analysis": self._get_error_analysis()
            }
        }
        
        # Add memory analysis if available
        if self.memory_snapshots:
            report["memory_analysis"] = self._analyze_memory_usage()
        
        return report
    
    def _get_metrics_by_component(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get metrics grouped by component."""
        by_component = defaultdict(list)
        
        for metric in self.metrics[-1000:]:  # Last 1000 metrics
            component = metric.component or "unknown"
            by_component[component].append({
                "name": metric.name,
                "value": metric.value,
                "unit": metric.unit,
                "timestamp": metric.timestamp.isoformat()
            })
        
        return dict(by_component)
    
    def _get_latency_trends(self) -> Dict[str, List[float]]:
        """Get latency trends over time."""
        trends = defaultdict(list)
        
        # Group latency metrics by component
        for metric in self.metrics[-1000:]:
            if "latency" in metric.name:
                component = metric.component or "unknown"
                trends[component].append(metric.value)
        
        return dict(trends)
    
    def _get_error_analysis(self) -> Dict[str, Any]:
        """Analyze error patterns."""
        total_errors = 0
        errors_by_component = defaultdict(int)
        
        for profile in self.component_profiles.values():
            total_errors += profile.error_count
            if profile.error_count > 0:
                errors_by_component[profile.component_name] = profile.error_count
        
        return {
            "total_errors": total_errors,
            "errors_by_component": dict(errors_by_component),
            "components_with_errors": len(errors_by_component)
        }
    
    def _analyze_memory_usage(self) -> Dict[str, Any]:
        """Analyze memory usage patterns."""
        if not self.memory_snapshots:
            return {}
        
        latest = self.memory_snapshots[-1]
        oldest = self.memory_snapshots[0]
        
        memory_growth = latest.used_memory_mb - oldest.used_memory_mb
        
        analysis = {
            "current_usage_mb": latest.used_memory_mb,
            "memory_growth_mb": memory_growth,
            "swap_usage_mb": latest.swap_used_mb,
            "gc_objects": latest.gc_objects,
            "memory_trend": "increasing" if memory_growth > 10 else "stable"
        }
        
        # Memory leak detection
        if len(self.memory_snapshots) > 3:
            recent_growth = sum(
                self.memory_snapshots[i].used_memory_mb - self.memory_snapshots[i-1].used_memory_mb
                for i in range(-3, 0)
            )
            
            if recent_growth > 50:  # 50MB growth in recent snapshots
                analysis["potential_memory_leak"] = True
                analysis["recent_growth_mb"] = recent_growth
        
        return analysis
    
    def export_report(self, file_path: str, format: str = "json") -> None:
        """Export performance report to file."""
        report = asyncio.run(self.generate_performance_report())
        
        if format == "json":
            with open(file_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        
        elif format == "csv":
            import csv
            
            # Export component metrics to CSV
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Component", "Total Calls", "Avg Time (ms)", 
                    "Min Time (ms)", "Max Time (ms)", "P95 Time (ms)", 
                    "Error Rate", "Total Time (sec)"
                ])
                
                for name, stats in report["component_performance"].items():
                    writer.writerow([
                        name,
                        stats["total_calls"],
                        stats["avg_time_ms"],
                        stats["min_time_ms"],
                        stats["max_time_ms"],
                        stats["p95_time_ms"],
                        stats["error_rate"],
                        stats["total_time_sec"]
                    ])
        
        print(f"📊 Performance report exported to: {file_path}")


# Integration helpers for voice agents
class ProfilerIntegration:
    """Helper class for integrating profiler with voice agents."""
    
    @staticmethod
    def create_agent_callbacks(profiler: PerformanceProfiler) -> Dict[str, Callable]:
        """Create callbacks for agent integration."""
        
        async def on_stt_start():
            # STT timing is handled by the timer context manager
            pass
        
        async def on_stt_complete(text: str, latency: float):
            profiler.record_metric("stt_success_rate", 1.0, component="stt")
            profiler.record_metric("stt_text_length", len(text), "chars", component="stt")
        
        async def on_llm_start():
            pass
        
        async def on_llm_complete(response: str, tokens: int, latency: float):
            profiler.record_metric("llm_tokens_generated", tokens, "tokens", component="llm")
            profiler.record_metric("llm_tokens_per_second", tokens / (latency / 1000), "tokens/sec", component="llm")
        
        async def on_tts_start():
            pass
        
        async def on_tts_complete(audio_duration: float, latency: float):
            profiler.record_metric("tts_audio_duration", audio_duration, "ms", component="tts")
            profiler.record_metric("tts_realtime_factor", audio_duration / latency, "ratio", component="tts")
        
        return {
            "on_stt_start": on_stt_start,
            "on_stt_complete": on_stt_complete,
            "on_llm_start": on_llm_start,
            "on_llm_complete": on_llm_complete,
            "on_tts_start": on_tts_start,
            "on_tts_complete": on_tts_complete
        }


# Decorator for automatic profiling
def profile_method(component: str, operation: str = None):
    """Decorator for automatic method profiling."""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(self, *args, **kwargs):
                profiler = getattr(self, '_profiler', None)
                if profiler:
                    op_name = operation or func.__name__
                    async with profiler.async_timer(component, op_name):
                        return await func(self, *args, **kwargs)
                else:
                    return await func(self, *args, **kwargs)
            return async_wrapper
        else:
            def sync_wrapper(self, *args, **kwargs):
                profiler = getattr(self, '_profiler', None)
                if profiler:
                    op_name = operation or func.__name__
                    with profiler.timer(component, op_name):
                        return func(self, *args, **kwargs)
                else:
                    return func(self, *args, **kwargs)
            return sync_wrapper
    return decorator


# Performance benchmark tool
class PerformanceBenchmark:
    """Tool for benchmarking voice agent performance."""
    
    def __init__(self, profiler: PerformanceProfiler):
        self.profiler = profiler
        self.benchmarks: Dict[str, Dict[str, float]] = {}
    
    async def run_benchmark(
        self,
        agent,
        test_phrases: List[str],
        iterations: int = 10
    ) -> Dict[str, Any]:
        """Run performance benchmark on agent."""
        print(f"🏁 Running performance benchmark ({iterations} iterations)")
        
        # Start profiling
        await self.profiler.start_profiling()
        
        results = {
            "total_iterations": iterations,
            "test_phrases": len(test_phrases),
            "results": []
        }
        
        try:
            for i in range(iterations):
                print(f"   Iteration {i+1}/{iterations}")
                
                for phrase in test_phrases:
                    # Simulate processing the phrase
                    start_time = time.perf_counter()
                    
                    # Mock STT
                    with self.profiler.timer("stt", "transcribe"):
                        await asyncio.sleep(0.1)  # Simulate STT latency
                    
                    # Mock LLM
                    with self.profiler.timer("llm", "generate"):
                        await asyncio.sleep(0.5)  # Simulate LLM latency
                    
                    # Mock TTS
                    with self.profiler.timer("tts", "synthesize"):
                        await asyncio.sleep(0.2)  # Simulate TTS latency
                    
                    total_time = (time.perf_counter() - start_time) * 1000
                    
                    results["results"].append({
                        "phrase": phrase,
                        "iteration": i + 1,
                        "total_time_ms": total_time
                    })
        
        finally:
            await self.profiler.stop_profiling()
        
        # Generate benchmark report
        report = await self.profiler.generate_performance_report()
        results["performance_report"] = report
        
        print(f"✅ Benchmark completed")
        return results
    
    def compare_benchmarks(
        self,
        baseline: Dict[str, Any],
        current: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare two benchmark results."""
        comparison = {
            "baseline_iterations": baseline["total_iterations"],
            "current_iterations": current["total_iterations"],
            "improvements": [],
            "regressions": []
        }
        
        # Compare component performance
        baseline_perf = baseline["performance_report"]["component_performance"]
        current_perf = current["performance_report"]["component_performance"]
        
        for component in baseline_perf:
            if component in current_perf:
                baseline_avg = baseline_perf[component]["avg_time_ms"]
                current_avg = current_perf[component]["avg_time_ms"]
                
                improvement = ((baseline_avg - current_avg) / baseline_avg) * 100
                
                if improvement > 5:  # 5% improvement
                    comparison["improvements"].append({
                        "component": component,
                        "improvement_percent": improvement,
                        "baseline_ms": baseline_avg,
                        "current_ms": current_avg
                    })
                elif improvement < -5:  # 5% regression
                    comparison["regressions"].append({
                        "component": component,
                        "regression_percent": abs(improvement),
                        "baseline_ms": baseline_avg,
                        "current_ms": current_avg
                    })
        
        return comparison


# Example usage
async def demo_performance_profiler():
    """Demonstrate performance profiler capabilities."""
    print("🔍 Performance Profiler Demo")
    print("="*50)
    
    # Create profiler
    profiler = PerformanceProfiler({
        "sample_interval": 0.5,
        "enable_memory_profiling": True,
        "enable_system_monitoring": True
    })
    
    # Start profiling
    await profiler.start_profiling()
    
    # Simulate some work with timing
    for i in range(5):
        # Simulate STT
        with profiler.timer("stt", "transcribe"):
            await asyncio.sleep(0.1 + (i * 0.02))  # Increasing latency
        
        # Simulate LLM with occasional errors
        try:
            with profiler.timer("llm", "generate"):
                await asyncio.sleep(0.3)
                if i == 3:  # Simulate error on 4th iteration
                    raise Exception("Mock LLM error")
        except Exception:
            pass  # Error is automatically recorded by timer
        
        # Simulate TTS
        with profiler.timer("tts", "synthesize"):
            await asyncio.sleep(0.15)
        
        print(f"   Completed iteration {i+1}")
    
    # Wait for some system monitoring samples
    await asyncio.sleep(2)
    
    # Stop profiling
    await profiler.stop_profiling()
    
    # Generate report
    report = await profiler.generate_performance_report()
    
    print("\n📊 Performance Report Summary:")
    print(f"   Profiling duration: {report['summary']['profiling_duration']}")
    print(f"   Components profiled: {report['summary']['total_components']}")
    print(f"   Total metrics: {report['summary']['total_metrics']}")
    
    print("\n⚡ Component Performance:")
    for component, stats in report["component_performance"].items():
        print(f"   {component}:")
        print(f"     Avg: {stats['avg_time_ms']:.1f}ms")
        print(f"     P95: {stats['p95_time_ms']:.1f}ms")
        print(f"     Errors: {stats['error_rate']:.1%}")
    
    print(f"\n⚠️  Detected {len(report['bottlenecks'])} bottlenecks")
    for bottleneck in report["bottlenecks"]:
        print(f"   - {bottleneck['description']}")
    
    print(f"\n💡 Recommendations:")
    for recommendation in report["recommendations"]:
        print(f"   - {recommendation}")
    
    # Export report
    profiler.export_report("performance_report.json")
    
    print("\n✅ Profiler demo completed!")


if __name__ == "__main__":
    asyncio.run(demo_performance_profiler())