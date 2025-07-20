"""
LiveKit Worker Orchestration and Job Lifecycle Management

This module provides comprehensive worker orchestration capabilities for the LiveKit
Voice Agents Platform, including job lifecycle management, health monitoring,
graceful shutdown, and worker pool management.

Features:
- Worker pool management and job distribution
- Health monitoring and automatic recovery
- Graceful shutdown and resource cleanup
- Job lifecycle tracking and metrics
- Integration with LiveKit infrastructure
- Load balancing and failover mechanisms
"""

import asyncio
import logging
import time
import threading
import signal
import os
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Set
from datetime import datetime, timedelta
import json
from contextlib import asynccontextmanager

# Optional dependencies
try:
    import livekit
    from livekit import api, rtc
    LIVEKIT_AVAILABLE = True
except ImportError:
    LIVEKIT_AVAILABLE = False
    livekit = None
    api = None
    rtc = None

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

# Local imports
from ..monitoring.performance_monitor import PerformanceMonitor, MetricType
from ..components.error_handling import ErrorHandler, ErrorSeverity
from ..sdk.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class WorkerStatus(Enum):
    """Worker status states."""
    INITIALIZING = "initializing"
    IDLE = "idle"
    BUSY = "busy"
    DRAINING = "draining"
    SHUTTING_DOWN = "shutting_down"
    FAILED = "failed"
    OFFLINE = "offline"


class JobStatus(Enum):
    """Job execution status states."""
    QUEUED = "queued"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkerInfo:
    """Information about a worker instance."""
    worker_id: str
    status: WorkerStatus = WorkerStatus.INITIALIZING
    current_job_id: Optional[str] = None
    capabilities: Set[str] = field(default_factory=set)
    resource_limits: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    last_heartbeat: datetime = field(default_factory=datetime.now)
    start_time: datetime = field(default_factory=datetime.now)
    completed_jobs: int = 0
    failed_jobs: int = 0
    host: str = ""
    port: int = 0


@dataclass
class JobInfo:
    """Information about a job."""
    job_id: str
    job_type: str
    status: JobStatus = JobStatus.QUEUED
    worker_id: Optional[str] = None
    priority: int = 0
    requirements: Dict[str, Any] = field(default_factory=dict)
    payload: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    assigned_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3


class WorkerOrchestrator:
    """
    Main orchestrator for managing LiveKit workers and job lifecycle.
    
    This class handles:
    - Worker registration and discovery
    - Job queue management and distribution
    - Health monitoring and recovery
    - Graceful shutdown procedures
    - Resource optimization
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        performance_monitor: Optional[PerformanceMonitor] = None,
        error_handler: Optional[ErrorHandler] = None,
        load_balancer: Optional[Any] = None
    ):
        self.config = config or {}
        self.performance_monitor = performance_monitor or PerformanceMonitor()
        self.error_handler = error_handler or ErrorHandler()
        self.load_balancer = load_balancer
        
        # Worker and job tracking
        self.workers: Dict[str, WorkerInfo] = {}
        self.jobs: Dict[str, JobInfo] = {}
        self.job_queue: asyncio.Queue = asyncio.Queue()
        
        # Configuration
        self.heartbeat_interval = self.config.get("heartbeat_interval", 30.0)
        self.worker_timeout = self.config.get("worker_timeout", 300.0)
        self.max_jobs_per_worker = self.config.get("max_jobs_per_worker", 5)
        self.job_retry_delay = self.config.get("job_retry_delay", 60.0)
        
        # State management
        self.running = False
        self.shutdown_event = asyncio.Event()
        self._tasks: List[asyncio.Task] = []
        self._lock = asyncio.Lock()
        
        # Callbacks
        self.job_callbacks: Dict[str, Callable] = {}
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        logger.info("WorkerOrchestrator initialized")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    async def start(self):
        """Start the worker orchestrator."""
        if self.running:
            return
        
        self.running = True
        self.shutdown_event.clear()
        
        # Start background tasks
        self._tasks = [
            asyncio.create_task(self._job_dispatcher()),
            asyncio.create_task(self._worker_monitor()),
            asyncio.create_task(self._health_checker()),
            asyncio.create_task(self._metrics_collector())
        ]
        
        await self.performance_monitor.record_metric(
            "orchestrator_started", 1, MetricType.COUNTER
        )
        
        logger.info("WorkerOrchestrator started")
    
    async def shutdown(self, timeout: float = 30.0):
        """Gracefully shutdown the orchestrator."""
        if not self.running:
            return
        
        logger.info("Starting graceful shutdown...")
        self.running = False
        self.shutdown_event.set()
        
        # Cancel all background tasks
        for task in self._tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._tasks:
            await asyncio.wait(self._tasks, timeout=timeout, return_when=asyncio.ALL_COMPLETED)
        
        # Shutdown all workers
        await self._shutdown_workers()
        
        await self.performance_monitor.record_metric(
            "orchestrator_shutdown", 1, MetricType.COUNTER
        )
        
        logger.info("WorkerOrchestrator shutdown complete")
    
    async def register_worker(
        self,
        worker_id: str,
        capabilities: Set[str],
        resource_limits: Optional[Dict[str, Any]] = None,
        host: str = "",
        port: int = 0
    ) -> bool:
        """Register a new worker with the orchestrator."""
        async with self._lock:
            if worker_id in self.workers:
                logger.warning(f"Worker {worker_id} already registered")
                return False
            
            worker_info = WorkerInfo(
                worker_id=worker_id,
                capabilities=capabilities,
                resource_limits=resource_limits or {},
                host=host,
                port=port,
                status=WorkerStatus.IDLE
            )
            
            self.workers[worker_id] = worker_info
            
            await self.performance_monitor.record_metric(
                "worker_registered", 1, MetricType.COUNTER,
                labels={"worker_id": worker_id}
            )
            
            logger.info(f"Worker {worker_id} registered with capabilities: {capabilities}")
            return True
    
    async def unregister_worker(self, worker_id: str, graceful: bool = True) -> bool:
        """Unregister a worker from the orchestrator."""
        async with self._lock:
            if worker_id not in self.workers:
                logger.warning(f"Worker {worker_id} not found")
                return False
            
            worker = self.workers[worker_id]
            
            if graceful and worker.current_job_id:
                # Mark worker as draining to finish current job
                worker.status = WorkerStatus.DRAINING
                logger.info(f"Worker {worker_id} marked as draining")
                return True
            else:
                # Force removal
                if worker.current_job_id:
                    await self._handle_job_failure(worker.current_job_id, "Worker forcefully removed")
                
                del self.workers[worker_id]
                
                await self.performance_monitor.record_metric(
                    "worker_unregistered", 1, MetricType.COUNTER,
                    labels={"worker_id": worker_id}
                )
                
                logger.info(f"Worker {worker_id} unregistered")
                return True
    
    async def submit_job(
        self,
        job_type: str,
        payload: Dict[str, Any],
        requirements: Optional[Dict[str, Any]] = None,
        priority: int = 0,
        max_retries: int = 3
    ) -> str:
        """Submit a new job to the orchestrator."""
        job_id = str(uuid.uuid4())
        
        job_info = JobInfo(
            job_id=job_id,
            job_type=job_type,
            payload=payload,
            requirements=requirements or {},
            priority=priority,
            max_retries=max_retries
        )
        
        async with self._lock:
            self.jobs[job_id] = job_info
        
        await self.job_queue.put(job_info)
        
        await self.performance_monitor.record_metric(
            "job_submitted", 1, MetricType.COUNTER,
            labels={"job_type": job_type}
        )
        
        logger.info(f"Job {job_id} submitted with type {job_type}")
        return job_id
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        async with self._lock:
            if job_id not in self.jobs:
                return False
            
            job = self.jobs[job_id]
            
            if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                return False
            
            job.status = JobStatus.CANCELLED
            
            if job.worker_id:
                worker = self.workers.get(job.worker_id)
                if worker and worker.current_job_id == job_id:
                    worker.current_job_id = None
                    worker.status = WorkerStatus.IDLE
        
        await self.performance_monitor.record_metric(
            "job_cancelled", 1, MetricType.COUNTER,
            labels={"job_type": job.job_type}
        )
        
        logger.info(f"Job {job_id} cancelled")
        return True
    
    async def get_job_status(self, job_id: str) -> Optional[JobInfo]:
        """Get the status of a job."""
        return self.jobs.get(job_id)
    
    async def get_worker_status(self, worker_id: str) -> Optional[WorkerInfo]:
        """Get the status of a worker."""
        return self.workers.get(worker_id)
    
    async def list_workers(self, status_filter: Optional[WorkerStatus] = None) -> List[WorkerInfo]:
        """List all workers, optionally filtered by status."""
        workers = list(self.workers.values())
        if status_filter:
            workers = [w for w in workers if w.status == status_filter]
        return workers
    
    async def list_jobs(self, status_filter: Optional[JobStatus] = None) -> List[JobInfo]:
        """List all jobs, optionally filtered by status."""
        jobs = list(self.jobs.values())
        if status_filter:
            jobs = [j for j in jobs if j.status == status_filter]
        return jobs
    
    async def update_worker_heartbeat(self, worker_id: str, metrics: Optional[Dict[str, Any]] = None):
        """Update worker heartbeat and metrics."""
        async with self._lock:
            if worker_id in self.workers:
                worker = self.workers[worker_id]
                worker.last_heartbeat = datetime.now()
                if metrics:
                    worker.metrics.update(metrics)
    
    async def register_job_callback(self, job_type: str, callback: Callable):
        """Register a callback for job completion."""
        self.job_callbacks[job_type] = callback
    
    async def _job_dispatcher(self):
        """Background task to dispatch jobs to workers."""
        while self.running:
            try:
                # Wait for a job with timeout to allow checking shutdown
                try:
                    job = await asyncio.wait_for(self.job_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                # Find suitable worker
                worker_id = await self._find_suitable_worker(job)
                
                if worker_id:
                    await self._assign_job_to_worker(job, worker_id)
                else:
                    # No suitable worker, put job back in queue
                    await asyncio.sleep(1.0)
                    await self.job_queue.put(job)
                
            except Exception as e:
                logger.error(f"Error in job dispatcher: {e}")
                await asyncio.sleep(1.0)
    
    async def _find_suitable_worker(self, job: JobInfo) -> Optional[str]:
        """Find a suitable worker for a job."""
        suitable_workers = []
        
        async with self._lock:
            for worker_id, worker in self.workers.items():
                if worker.status != WorkerStatus.IDLE:
                    continue
                
                # Check capabilities
                required_capabilities = job.requirements.get("capabilities", set())
                if not required_capabilities.issubset(worker.capabilities):
                    continue
                
                # Check resource requirements
                if not self._check_resource_requirements(worker, job):
                    continue
                
                suitable_workers.append(worker)
        
        if not suitable_workers:
            return None
        
        # Use load balancer if available
        if self.load_balancer:
            try:
                selected_worker_id = await self.load_balancer.select_worker(
                    suitable_workers, job.requirements
                )
                if selected_worker_id:
                    return selected_worker_id
            except Exception as e:
                logger.warning(f"Load balancer error, falling back to basic selection: {e}")
        
        # Fallback: Sort by load (prefer workers with fewer completed jobs)
        suitable_workers.sort(key=lambda x: x.completed_jobs)
        return suitable_workers[0].worker_id
    
    def _check_resource_requirements(self, worker: WorkerInfo, job: JobInfo) -> bool:
        """Check if worker meets job resource requirements."""
        resource_requirements = job.requirements.get("resources", {})
        
        for resource, required_amount in resource_requirements.items():
            worker_limit = worker.resource_limits.get(resource)
            if worker_limit is None or worker_limit < required_amount:
                return False
        
        return True
    
    async def _assign_job_to_worker(self, job: JobInfo, worker_id: str):
        """Assign a job to a worker."""
        async with self._lock:
            worker = self.workers[worker_id]
            worker.status = WorkerStatus.BUSY
            worker.current_job_id = job.job_id
            
            job.status = JobStatus.ASSIGNED
            job.worker_id = worker_id
            job.assigned_at = datetime.now()
        
        # Execute job (this would typically involve calling the worker)
        await self._execute_job(job, worker_id)
    
    async def _execute_job(self, job: JobInfo, worker_id: str):
        """Execute a job on a worker."""
        try:
            # Update job status
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now()
            
            # Record job start metric
            await self.performance_monitor.record_metric(
                "job_started", 1, MetricType.COUNTER,
                labels={"job_type": job.job_type, "worker_id": worker_id}
            )
            
            # Simulate job execution (in real implementation, this would call the worker)
            await asyncio.sleep(1.0)  # Placeholder for actual job execution
            
            # Job completed successfully
            await self._handle_job_completion(job.job_id)
            
        except Exception as e:
            logger.error(f"Job {job.job_id} failed on worker {worker_id}: {e}")
            await self._handle_job_failure(job.job_id, str(e))
    
    async def _handle_job_completion(self, job_id: str):
        """Handle successful job completion."""
        async with self._lock:
            job = self.jobs.get(job_id)
            if not job:
                return
            
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now()
            
            # Update worker status
            if job.worker_id:
                worker = self.workers.get(job.worker_id)
                if worker:
                    worker.status = WorkerStatus.IDLE
                    worker.current_job_id = None
                    worker.completed_jobs += 1
        
        # Record metrics
        if job:
            duration = (job.completed_at - job.started_at).total_seconds()
            await self.performance_monitor.record_metric(
                "job_duration", duration, MetricType.TIMER,
                labels={"job_type": job.job_type}
            )
            
            await self.performance_monitor.record_metric(
                "job_completed", 1, MetricType.COUNTER,
                labels={"job_type": job.job_type}
            )
            
            # Call job callback if registered
            callback = self.job_callbacks.get(job.job_type)
            if callback:
                try:
                    await callback(job)
                except Exception as e:
                    logger.error(f"Error in job callback for {job_id}: {e}")
        
        logger.info(f"Job {job_id} completed successfully")
    
    async def _handle_job_failure(self, job_id: str, error_message: str):
        """Handle job failure."""
        async with self._lock:
            job = self.jobs.get(job_id)
            if not job:
                return
            
            job.error_message = error_message
            job.retry_count += 1
            
            # Update worker status
            if job.worker_id:
                worker = self.workers.get(job.worker_id)
                if worker:
                    worker.status = WorkerStatus.IDLE
                    worker.current_job_id = None
                    worker.failed_jobs += 1
            
            # Determine if job should be retried
            if job.retry_count <= job.max_retries:
                job.status = JobStatus.QUEUED
                job.worker_id = None
                job.assigned_at = None
                job.started_at = None
                
                # Schedule retry with delay
                asyncio.create_task(self._schedule_retry(job))
                
                logger.warning(f"Job {job_id} failed, scheduling retry {job.retry_count}/{job.max_retries}")
            else:
                job.status = JobStatus.FAILED
                job.completed_at = datetime.now()
                
                await self.performance_monitor.record_metric(
                    "job_failed", 1, MetricType.COUNTER,
                    labels={"job_type": job.job_type}
                )
                
                logger.error(f"Job {job_id} failed permanently after {job.retry_count} retries")
    
    async def _schedule_retry(self, job: JobInfo):
        """Schedule a job retry with delay."""
        await asyncio.sleep(self.job_retry_delay)
        await self.job_queue.put(job)
    
    async def _worker_monitor(self):
        """Background task to monitor worker health."""
        while self.running:
            try:
                current_time = datetime.now()
                timeout_threshold = current_time - timedelta(seconds=self.worker_timeout)
                
                async with self._lock:
                    for worker_id, worker in list(self.workers.items()):
                        if worker.last_heartbeat < timeout_threshold:
                            logger.warning(f"Worker {worker_id} timed out")
                            
                            # Handle current job if any
                            if worker.current_job_id:
                                await self._handle_job_failure(
                                    worker.current_job_id,
                                    "Worker timed out"
                                )
                            
                            # Mark worker as failed
                            worker.status = WorkerStatus.FAILED
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Error in worker monitor: {e}")
                await asyncio.sleep(1.0)
    
    async def _health_checker(self):
        """Background task for health checking."""
        while self.running:
            try:
                # Collect health metrics
                total_workers = len(self.workers)
                active_workers = len([w for w in self.workers.values() if w.status in [WorkerStatus.IDLE, WorkerStatus.BUSY]])
                queued_jobs = self.job_queue.qsize()
                
                await self.performance_monitor.record_metric(
                    "total_workers", total_workers, MetricType.GAUGE
                )
                
                await self.performance_monitor.record_metric(
                    "active_workers", active_workers, MetricType.GAUGE
                )
                
                await self.performance_monitor.record_metric(
                    "queued_jobs", queued_jobs, MetricType.GAUGE
                )
                
                await asyncio.sleep(30.0)
                
            except Exception as e:
                logger.error(f"Error in health checker: {e}")
                await asyncio.sleep(1.0)
    
    async def _metrics_collector(self):
        """Background task for collecting and reporting metrics."""
        while self.running:
            try:
                # Collect system metrics if psutil is available
                if PSUTIL_AVAILABLE:
                    cpu_percent = psutil.cpu_percent()
                    memory_percent = psutil.virtual_memory().percent
                    
                    await self.performance_monitor.record_metric(
                        "system_cpu_percent", cpu_percent, MetricType.GAUGE
                    )
                    
                    await self.performance_monitor.record_metric(
                        "system_memory_percent", memory_percent, MetricType.GAUGE
                    )
                
                await asyncio.sleep(60.0)
                
            except Exception as e:
                logger.error(f"Error in metrics collector: {e}")
                await asyncio.sleep(1.0)
    
    async def _shutdown_workers(self):
        """Shutdown all workers gracefully."""
        async with self._lock:
            for worker_id, worker in self.workers.items():
                worker.status = WorkerStatus.SHUTTING_DOWN
        
        # Wait for workers to finish current jobs (with timeout)
        timeout = 30.0
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            active_jobs = [w for w in self.workers.values() if w.current_job_id]
            if not active_jobs:
                break
            await asyncio.sleep(1.0)
        
        logger.info("All workers shutdown")


# Factory function for creating orchestrator
def create_orchestrator(
    config: Optional[Dict[str, Any]] = None,
    load_balancer: Optional[Any] = None
) -> WorkerOrchestrator:
    """Create a new WorkerOrchestrator instance."""
    return WorkerOrchestrator(config, load_balancer=load_balancer)


# Async context manager for orchestrator lifecycle
@asynccontextmanager
async def orchestrator_context(
    config: Optional[Dict[str, Any]] = None,
    load_balancer: Optional[Any] = None
):
    """Async context manager for orchestrator lifecycle."""
    orchestrator = create_orchestrator(config, load_balancer)
    try:
        await orchestrator.start()
        yield orchestrator
    finally:
        await orchestrator.shutdown()