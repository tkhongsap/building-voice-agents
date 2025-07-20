"""
Unit tests for LiveKit Worker Orchestration and Job Lifecycle Management

This module contains comprehensive tests for the worker orchestration system,
including worker registration, job management, health monitoring, and graceful shutdown.
"""

import asyncio
import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from .worker_orchestration import (
    WorkerOrchestrator,
    WorkerInfo,
    JobInfo,
    WorkerStatus,
    JobStatus,
    create_orchestrator,
    orchestrator_context
)
from ..monitoring.performance_monitor import PerformanceMonitor
from ..components.error_handling import ErrorHandler


class TestWorkerOrchestrator:
    """Test cases for WorkerOrchestrator."""
    
    @pytest.fixture
    async def orchestrator(self):
        """Create a test orchestrator instance."""
        config = {
            "heartbeat_interval": 1.0,  # Fast for testing
            "worker_timeout": 5.0,      # Short timeout for testing
            "max_jobs_per_worker": 2,
            "job_retry_delay": 0.1      # Fast retry for testing
        }
        orchestrator = WorkerOrchestrator(config)
        await orchestrator.start()
        yield orchestrator
        await orchestrator.shutdown()
    
    @pytest.fixture
    def mock_performance_monitor(self):
        """Create a mock performance monitor."""
        monitor = Mock(spec=PerformanceMonitor)
        monitor.record_metric = AsyncMock()
        return monitor
    
    @pytest.fixture
    def mock_error_handler(self):
        """Create a mock error handler."""
        handler = Mock(spec=ErrorHandler)
        return handler
    
    async def test_orchestrator_initialization(self):
        """Test orchestrator initialization."""
        config = {"test": "value"}
        orchestrator = WorkerOrchestrator(config)
        
        assert orchestrator.config == config
        assert orchestrator.running is False
        assert len(orchestrator.workers) == 0
        assert len(orchestrator.jobs) == 0
    
    async def test_orchestrator_start_stop(self, orchestrator):
        """Test orchestrator start and stop."""
        assert orchestrator.running is True
        assert len(orchestrator._tasks) > 0
        
        await orchestrator.shutdown()
        assert orchestrator.running is False
    
    async def test_worker_registration(self, orchestrator):
        """Test worker registration."""
        worker_id = "test-worker-1"
        capabilities = {"voice_processing", "transcription"}
        
        # Register worker
        result = await orchestrator.register_worker(
            worker_id,
            capabilities,
            resource_limits={"cpu": 2, "memory": 4096},
            host="localhost",
            port=8080
        )
        
        assert result is True
        assert worker_id in orchestrator.workers
        
        worker = orchestrator.workers[worker_id]
        assert worker.worker_id == worker_id
        assert worker.capabilities == capabilities
        assert worker.status == WorkerStatus.IDLE
        assert worker.host == "localhost"
        assert worker.port == 8080
    
    async def test_duplicate_worker_registration(self, orchestrator):
        """Test duplicate worker registration."""
        worker_id = "test-worker-1"
        capabilities = {"voice_processing"}
        
        # Register worker first time
        result1 = await orchestrator.register_worker(worker_id, capabilities)
        assert result1 is True
        
        # Try to register same worker again
        result2 = await orchestrator.register_worker(worker_id, capabilities)
        assert result2 is False
    
    async def test_worker_unregistration(self, orchestrator):
        """Test worker unregistration."""
        worker_id = "test-worker-1"
        capabilities = {"voice_processing"}
        
        # Register worker
        await orchestrator.register_worker(worker_id, capabilities)
        assert worker_id in orchestrator.workers
        
        # Unregister worker
        result = await orchestrator.unregister_worker(worker_id)
        assert result is True
        assert worker_id not in orchestrator.workers
    
    async def test_worker_unregistration_nonexistent(self, orchestrator):
        """Test unregistration of non-existent worker."""
        result = await orchestrator.unregister_worker("nonexistent-worker")
        assert result is False
    
    async def test_job_submission(self, orchestrator):
        """Test job submission."""
        job_type = "voice_processing"
        payload = {"audio_url": "test.wav"}
        requirements = {"capabilities": {"voice_processing"}}
        
        job_id = await orchestrator.submit_job(
            job_type,
            payload,
            requirements,
            priority=1,
            max_retries=2
        )
        
        assert job_id is not None
        assert job_id in orchestrator.jobs
        
        job = orchestrator.jobs[job_id]
        assert job.job_type == job_type
        assert job.payload == payload
        assert job.requirements == requirements
        assert job.priority == 1
        assert job.max_retries == 2
        assert job.status == JobStatus.QUEUED
    
    async def test_job_cancellation(self, orchestrator):
        """Test job cancellation."""
        job_type = "voice_processing"
        payload = {"audio_url": "test.wav"}
        
        # Submit job
        job_id = await orchestrator.submit_job(job_type, payload)
        assert job_id in orchestrator.jobs
        
        # Cancel job
        result = await orchestrator.cancel_job(job_id)
        assert result is True
        
        job = orchestrator.jobs[job_id]
        assert job.status == JobStatus.CANCELLED
    
    async def test_job_cancellation_nonexistent(self, orchestrator):
        """Test cancellation of non-existent job."""
        result = await orchestrator.cancel_job("nonexistent-job")
        assert result is False
    
    async def test_job_assignment_to_worker(self, orchestrator):
        """Test job assignment to suitable worker."""
        # Register worker
        worker_id = "test-worker-1"
        capabilities = {"voice_processing"}
        await orchestrator.register_worker(worker_id, capabilities)
        
        # Submit job
        job_type = "voice_processing"
        payload = {"audio_url": "test.wav"}
        requirements = {"capabilities": {"voice_processing"}}
        
        job_id = await orchestrator.submit_job(job_type, payload, requirements)
        
        # Wait for job assignment
        await asyncio.sleep(0.1)
        
        # Check job was assigned
        job = orchestrator.jobs[job_id]
        assert job.worker_id == worker_id
        assert job.status in [JobStatus.ASSIGNED, JobStatus.RUNNING, JobStatus.COMPLETED]
        
        # Check worker status
        worker = orchestrator.workers[worker_id]
        if job.status in [JobStatus.ASSIGNED, JobStatus.RUNNING]:
            assert worker.status == WorkerStatus.BUSY
            assert worker.current_job_id == job_id
    
    async def test_job_assignment_no_suitable_worker(self, orchestrator):
        """Test job assignment when no suitable worker is available."""
        # Register worker with different capabilities
        worker_id = "test-worker-1"
        capabilities = {"transcription"}
        await orchestrator.register_worker(worker_id, capabilities)
        
        # Submit job requiring different capability
        job_type = "voice_processing"
        payload = {"audio_url": "test.wav"}
        requirements = {"capabilities": {"voice_processing"}}
        
        job_id = await orchestrator.submit_job(job_type, payload, requirements)
        
        # Wait a bit
        await asyncio.sleep(0.1)
        
        # Job should remain queued
        job = orchestrator.jobs[job_id]
        assert job.status == JobStatus.QUEUED
        assert job.worker_id is None
    
    async def test_worker_heartbeat_update(self, orchestrator):
        """Test worker heartbeat updates."""
        worker_id = "test-worker-1"
        capabilities = {"voice_processing"}
        await orchestrator.register_worker(worker_id, capabilities)
        
        original_heartbeat = orchestrator.workers[worker_id].last_heartbeat
        
        # Update heartbeat
        metrics = {"cpu_usage": 50.0, "memory_usage": 60.0}
        await orchestrator.update_worker_heartbeat(worker_id, metrics)
        
        worker = orchestrator.workers[worker_id]
        assert worker.last_heartbeat > original_heartbeat
        assert worker.metrics["cpu_usage"] == 50.0
        assert worker.metrics["memory_usage"] == 60.0
    
    async def test_worker_timeout_detection(self, orchestrator):
        """Test worker timeout detection."""
        worker_id = "test-worker-1"
        capabilities = {"voice_processing"}
        await orchestrator.register_worker(worker_id, capabilities)
        
        # Manually set old heartbeat
        worker = orchestrator.workers[worker_id]
        worker.last_heartbeat = datetime.now() - timedelta(seconds=10)
        
        # Wait for worker monitor to detect timeout
        await asyncio.sleep(2.0)
        
        # Worker should be marked as failed
        assert worker.status == WorkerStatus.FAILED
    
    async def test_job_status_retrieval(self, orchestrator):
        """Test job status retrieval."""
        job_type = "voice_processing"
        payload = {"audio_url": "test.wav"}
        
        job_id = await orchestrator.submit_job(job_type, payload)
        
        # Get job status
        job_info = await orchestrator.get_job_status(job_id)
        assert job_info is not None
        assert job_info.job_id == job_id
        assert job_info.job_type == job_type
        
        # Get non-existent job status
        nonexistent_status = await orchestrator.get_job_status("nonexistent")
        assert nonexistent_status is None
    
    async def test_worker_status_retrieval(self, orchestrator):
        """Test worker status retrieval."""
        worker_id = "test-worker-1"
        capabilities = {"voice_processing"}
        
        await orchestrator.register_worker(worker_id, capabilities)
        
        # Get worker status
        worker_info = await orchestrator.get_worker_status(worker_id)
        assert worker_info is not None
        assert worker_info.worker_id == worker_id
        assert worker_info.capabilities == capabilities
        
        # Get non-existent worker status
        nonexistent_status = await orchestrator.get_worker_status("nonexistent")
        assert nonexistent_status is None
    
    async def test_list_workers(self, orchestrator):
        """Test listing workers."""
        # Register multiple workers
        await orchestrator.register_worker("worker-1", {"voice"})
        await orchestrator.register_worker("worker-2", {"transcription"})
        
        # List all workers
        all_workers = await orchestrator.list_workers()
        assert len(all_workers) == 2
        
        # List workers by status
        idle_workers = await orchestrator.list_workers(WorkerStatus.IDLE)
        assert len(idle_workers) == 2
        
        busy_workers = await orchestrator.list_workers(WorkerStatus.BUSY)
        assert len(busy_workers) == 0
    
    async def test_list_jobs(self, orchestrator):
        """Test listing jobs."""
        # Submit multiple jobs
        job_id1 = await orchestrator.submit_job("type1", {"data": 1})
        job_id2 = await orchestrator.submit_job("type2", {"data": 2})
        
        # List all jobs
        all_jobs = await orchestrator.list_jobs()
        assert len(all_jobs) == 2
        
        # List jobs by status
        queued_jobs = await orchestrator.list_jobs(JobStatus.QUEUED)
        assert len(queued_jobs) == 2
        
        completed_jobs = await orchestrator.list_jobs(JobStatus.COMPLETED)
        assert len(completed_jobs) == 0
    
    async def test_job_callback_registration(self, orchestrator):
        """Test job callback registration and execution."""
        callback_called = False
        callback_job = None
        
        async def test_callback(job_info):
            nonlocal callback_called, callback_job
            callback_called = True
            callback_job = job_info
        
        # Register callback
        await orchestrator.register_job_callback("test_type", test_callback)
        
        # Submit and complete job
        job_id = await orchestrator.submit_job("test_type", {"test": "data"})
        await orchestrator._handle_job_completion(job_id)
        
        # Callback should have been called
        assert callback_called is True
        assert callback_job is not None
        assert callback_job.job_id == job_id
    
    async def test_resource_requirement_checking(self, orchestrator):
        """Test resource requirement checking for job assignment."""
        # Register worker with limited resources
        worker_id = "limited-worker"
        capabilities = {"voice_processing"}
        resource_limits = {"cpu": 2, "memory": 1024}
        
        await orchestrator.register_worker(
            worker_id, capabilities, resource_limits
        )
        
        # Submit job with high resource requirements
        job_type = "voice_processing"
        payload = {"audio_url": "test.wav"}
        requirements = {
            "capabilities": {"voice_processing"},
            "resources": {"cpu": 4, "memory": 2048}  # Exceeds worker limits
        }
        
        job_id = await orchestrator.submit_job(job_type, payload, requirements)
        
        # Wait for assignment attempt
        await asyncio.sleep(0.1)
        
        # Job should remain queued (no suitable worker)
        job = orchestrator.jobs[job_id]
        assert job.status == JobStatus.QUEUED
        assert job.worker_id is None
    
    async def test_graceful_worker_removal(self, orchestrator):
        """Test graceful worker removal when worker has active job."""
        # Register worker and assign job
        worker_id = "test-worker"
        capabilities = {"voice_processing"}
        await orchestrator.register_worker(worker_id, capabilities)
        
        job_id = await orchestrator.submit_job("voice_processing", {"test": "data"})
        await asyncio.sleep(0.1)  # Allow job assignment
        
        # Attempt graceful removal
        result = await orchestrator.unregister_worker(worker_id, graceful=True)
        assert result is True
        
        # Worker should be marked as draining
        worker = orchestrator.workers[worker_id]
        assert worker.status == WorkerStatus.DRAINING
        assert worker_id in orchestrator.workers  # Still registered
    
    async def test_forceful_worker_removal(self, orchestrator):
        """Test forceful worker removal when worker has active job."""
        # Register worker and assign job
        worker_id = "test-worker"
        capabilities = {"voice_processing"}
        await orchestrator.register_worker(worker_id, capabilities)
        
        job_id = await orchestrator.submit_job("voice_processing", {"test": "data"})
        await asyncio.sleep(0.1)  # Allow job assignment
        
        # Force removal
        result = await orchestrator.unregister_worker(worker_id, graceful=False)
        assert result is True
        
        # Worker should be completely removed
        assert worker_id not in orchestrator.workers
        
        # Job should be marked as failed
        job = orchestrator.jobs[job_id]
        assert job.status == JobStatus.FAILED


class TestFactoryFunctions:
    """Test factory functions and context managers."""
    
    async def test_create_orchestrator(self):
        """Test orchestrator factory function."""
        config = {"test": "config"}
        orchestrator = create_orchestrator(config)
        
        assert isinstance(orchestrator, WorkerOrchestrator)
        assert orchestrator.config == config
    
    async def test_orchestrator_context_manager(self):
        """Test orchestrator context manager."""
        config = {"test": "config"}
        
        async with orchestrator_context(config) as orchestrator:
            assert isinstance(orchestrator, WorkerOrchestrator)
            assert orchestrator.running is True
            assert orchestrator.config == config
        
        # Should be shutdown after context exit
        assert orchestrator.running is False


class TestWorkerInfo:
    """Test WorkerInfo dataclass."""
    
    def test_worker_info_creation(self):
        """Test WorkerInfo creation with defaults."""
        worker = WorkerInfo("test-worker")
        
        assert worker.worker_id == "test-worker"
        assert worker.status == WorkerStatus.INITIALIZING
        assert worker.current_job_id is None
        assert worker.capabilities == set()
        assert worker.resource_limits == {}
        assert worker.metrics == {}
        assert worker.completed_jobs == 0
        assert worker.failed_jobs == 0
        assert isinstance(worker.last_heartbeat, datetime)
        assert isinstance(worker.start_time, datetime)
    
    def test_worker_info_with_values(self):
        """Test WorkerInfo creation with custom values."""
        capabilities = {"voice", "transcription"}
        resource_limits = {"cpu": 4, "memory": 8192}
        
        worker = WorkerInfo(
            "test-worker",
            status=WorkerStatus.IDLE,
            capabilities=capabilities,
            resource_limits=resource_limits,
            host="localhost",
            port=8080
        )
        
        assert worker.worker_id == "test-worker"
        assert worker.status == WorkerStatus.IDLE
        assert worker.capabilities == capabilities
        assert worker.resource_limits == resource_limits
        assert worker.host == "localhost"
        assert worker.port == 8080


class TestJobInfo:
    """Test JobInfo dataclass."""
    
    def test_job_info_creation(self):
        """Test JobInfo creation with defaults."""
        job = JobInfo("test-job", "voice_processing")
        
        assert job.job_id == "test-job"
        assert job.job_type == "voice_processing"
        assert job.status == JobStatus.QUEUED
        assert job.worker_id is None
        assert job.priority == 0
        assert job.requirements == {}
        assert job.payload == {}
        assert job.retry_count == 0
        assert job.max_retries == 3
        assert isinstance(job.created_at, datetime)
    
    def test_job_info_with_values(self):
        """Test JobInfo creation with custom values."""
        requirements = {"capabilities": {"voice"}}
        payload = {"audio_url": "test.wav"}
        
        job = JobInfo(
            "test-job",
            "voice_processing",
            status=JobStatus.RUNNING,
            worker_id="worker-1",
            priority=5,
            requirements=requirements,
            payload=payload,
            max_retries=5
        )
        
        assert job.job_id == "test-job"
        assert job.job_type == "voice_processing"
        assert job.status == JobStatus.RUNNING
        assert job.worker_id == "worker-1"
        assert job.priority == 5
        assert job.requirements == requirements
        assert job.payload == payload
        assert job.max_retries == 5


if __name__ == "__main__":
    pytest.main([__file__])