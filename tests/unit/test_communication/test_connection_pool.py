"""
Unit tests for Connection Pool.

Tests connection pooling, resource optimization, lifecycle management,
and resource monitoring for efficient connection reuse.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import sys
import os
from typing import Dict, List, Any, Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from communication.connection_pool import (
    ConnectionPool,
    PoolConfig,
    PooledConnection,
    ConnectionState,
    PoolStrategy,
    ResourceMonitor,
    PoolStatistics,
    PoolEvent,
    PoolEventType,
    ConnectionFactory,
    PoolHealthChecker
)


class TestPoolConfig:
    """Test connection pool configuration."""
    
    def test_default_config(self):
        """Test default pool configuration."""
        config = PoolConfig()
        
        assert config.min_connections == 2
        assert config.max_connections == 10
        assert config.initial_connections == 3
        assert config.max_idle_time_seconds == 300
        assert config.max_lifetime_seconds == 3600
        assert config.strategy == PoolStrategy.FIFO
        assert config.enable_preemptive_creation == True
        assert config.enable_resource_monitoring == True
        assert config.health_check_interval_seconds == 60
        assert config.cleanup_interval_seconds == 30
    
    def test_custom_config(self):
        """Test custom pool configuration."""
        config = PoolConfig(
            min_connections=5,
            max_connections=20,
            initial_connections=8,
            max_idle_time_seconds=600,
            strategy=PoolStrategy.LIFO,
            enable_preemptive_creation=False
        )
        
        assert config.min_connections == 5
        assert config.max_connections == 20
        assert config.initial_connections == 8
        assert config.max_idle_time_seconds == 600
        assert config.strategy == PoolStrategy.LIFO
        assert config.enable_preemptive_creation == False
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        valid_config = PoolConfig(min_connections=2, max_connections=10, initial_connections=5)
        assert valid_config.validate() == True
        
        # Invalid config (initial > max)
        invalid_config = PoolConfig(min_connections=2, max_connections=10, initial_connections=15)
        assert invalid_config.validate() == False


class TestPoolStrategy:
    """Test pool strategy enumeration."""
    
    def test_strategy_types(self):
        """Test pool strategy types."""
        assert PoolStrategy.FIFO.value == "fifo"
        assert PoolStrategy.LIFO.value == "lifo"
        assert PoolStrategy.ROUND_ROBIN.value == "round_robin"
        assert PoolStrategy.LEAST_USED.value == "least_used"
        assert PoolStrategy.RANDOM.value == "random"
    
    def test_strategy_selection_logic(self):
        """Test strategy selection logic."""
        connections = [
            PooledConnection(connection_id="conn_1", last_used=time.time() - 100, use_count=5),
            PooledConnection(connection_id="conn_2", last_used=time.time() - 50, use_count=2),
            PooledConnection(connection_id="conn_3", last_used=time.time() - 200, use_count=8)
        ]
        
        # Test FIFO (oldest first)
        fifo_conn = PoolStrategy.FIFO.select_connection(connections)
        assert fifo_conn.connection_id == "conn_3"  # Oldest last_used
        
        # Test LEAST_USED
        least_used_conn = PoolStrategy.LEAST_USED.select_connection(connections)
        assert least_used_conn.connection_id == "conn_2"  # Lowest use_count


class TestConnectionState:
    """Test connection state enumeration."""
    
    def test_connection_states(self):
        """Test connection state values."""
        assert ConnectionState.IDLE.value == "idle"
        assert ConnectionState.ACTIVE.value == "active"
        assert ConnectionState.CONNECTING.value == "connecting"
        assert ConnectionState.DISCONNECTING.value == "disconnecting"
        assert ConnectionState.FAILED.value == "failed"
        assert ConnectionState.EXPIRED.value == "expired"
    
    def test_state_transitions(self):
        """Test valid state transitions."""
        # Valid transitions
        assert ConnectionState.IDLE.can_transition_to(ConnectionState.ACTIVE) == True
        assert ConnectionState.ACTIVE.can_transition_to(ConnectionState.IDLE) == True
        assert ConnectionState.CONNECTING.can_transition_to(ConnectionState.ACTIVE) == True
        assert ConnectionState.CONNECTING.can_transition_to(ConnectionState.FAILED) == True
        
        # Invalid transitions
        assert ConnectionState.FAILED.can_transition_to(ConnectionState.ACTIVE) == False
        assert ConnectionState.EXPIRED.can_transition_to(ConnectionState.IDLE) == False


class TestPooledConnection:
    """Test pooled connection wrapper."""
    
    def test_connection_creation(self):
        """Test pooled connection creation."""
        mock_conn = Mock()
        mock_conn.is_connected = Mock(return_value=True)
        
        pooled_conn = PooledConnection(
            connection=mock_conn,
            connection_id="test_conn_1",
            created_at=time.time(),
            state=ConnectionState.IDLE
        )
        
        assert pooled_conn.connection == mock_conn
        assert pooled_conn.connection_id == "test_conn_1"
        assert pooled_conn.state == ConnectionState.IDLE
        assert pooled_conn.use_count == 0
        assert pooled_conn.last_used is None
    
    def test_connection_usage_tracking(self):
        """Test connection usage tracking."""
        pooled_conn = PooledConnection(
            connection=Mock(),
            connection_id="test"
        )
        
        # Mark as used
        pooled_conn.mark_used()
        
        assert pooled_conn.use_count == 1
        assert pooled_conn.last_used is not None
        assert pooled_conn.state == ConnectionState.ACTIVE
        
        # Mark as returned
        pooled_conn.mark_returned()
        assert pooled_conn.state == ConnectionState.IDLE
    
    def test_connection_expiry_check(self):
        """Test connection expiry checking."""
        old_time = time.time() - 4000  # 4000 seconds ago
        
        expired_conn = PooledConnection(
            connection=Mock(),
            connection_id="expired",
            created_at=old_time
        )
        
        fresh_conn = PooledConnection(
            connection=Mock(),
            connection_id="fresh",
            created_at=time.time()
        )
        
        assert expired_conn.is_expired(max_lifetime_seconds=3600) == True
        assert fresh_conn.is_expired(max_lifetime_seconds=3600) == False
    
    def test_idle_time_calculation(self):
        """Test idle time calculation."""
        pooled_conn = PooledConnection(
            connection=Mock(),
            connection_id="test"
        )
        
        # Mark as used and returned
        pooled_conn.mark_used()
        time.sleep(0.1)
        pooled_conn.mark_returned()
        
        idle_time = pooled_conn.get_idle_time_seconds()
        assert idle_time >= 0.0
        assert idle_time < 1.0  # Should be very small


class TestConnectionFactory:
    """Test connection factory."""
    
    @pytest.fixture
    def connection_factory(self, mock_connection_factory):
        """Create connection factory for testing."""
        return mock_connection_factory
    
    @pytest.mark.asyncio
    async def test_connection_creation(self, connection_factory):
        """Test connection creation through factory."""
        connection = await connection_factory()
        
        assert connection is not None
        assert hasattr(connection, 'connect')
        assert hasattr(connection, 'close')
    
    @pytest.mark.asyncio
    async def test_connection_validation(self, connection_factory):
        """Test connection validation."""
        connection = await connection_factory()
        
        # Mock connection should be valid
        assert connection.connect is not None
        assert connection.close is not None


class TestPoolStatistics:
    """Test pool statistics tracking."""
    
    def test_statistics_creation(self):
        """Test pool statistics creation."""
        stats = PoolStatistics(
            total_connections=10,
            active_connections=3,
            idle_connections=7,
            failed_connections=0,
            total_requests=150,
            successful_requests=148,
            failed_requests=2,
            average_wait_time_ms=25.5,
            peak_connections=8
        )
        
        assert stats.total_connections == 10
        assert stats.active_connections == 3
        assert stats.idle_connections == 7
        assert stats.total_requests == 150
        assert stats.successful_requests == 148
        assert stats.average_wait_time_ms == 25.5
    
    def test_statistics_calculations(self):
        """Test statistics calculations."""
        stats = PoolStatistics(
            total_connections=10,
            active_connections=3,
            successful_requests=98,
            total_requests=100
        )
        
        assert stats.get_utilization_rate() == 0.3  # 3/10
        assert stats.get_success_rate() == 0.98  # 98/100
        assert stats.get_failure_rate() == 0.02  # 2/100
    
    def test_statistics_update(self):
        """Test statistics update operations."""
        stats = PoolStatistics()
        
        # Record request
        stats.record_request(wait_time_ms=50.0, success=True)
        
        assert stats.total_requests == 1
        assert stats.successful_requests == 1
        assert stats.average_wait_time_ms == 50.0
        
        # Record another request
        stats.record_request(wait_time_ms=30.0, success=True)
        
        assert stats.total_requests == 2
        assert stats.successful_requests == 2
        assert stats.average_wait_time_ms == 40.0  # (50+30)/2


class TestResourceMonitor:
    """Test resource monitoring."""
    
    @pytest.fixture
    def resource_monitor(self):
        """Create resource monitor for testing."""
        config = PoolConfig()
        monitor = ResourceMonitor(config)
        return monitor
    
    def test_monitor_initialization(self, resource_monitor):
        """Test resource monitor initialization."""
        assert resource_monitor.config is not None
        assert resource_monitor.is_monitoring == False
        assert len(resource_monitor.resource_history) == 0
    
    @pytest.mark.asyncio
    async def test_resource_measurement(self, resource_monitor):
        """Test resource measurement collection."""
        # Mock pool for measurement
        mock_pool = Mock()
        mock_pool.get_statistics.return_value = PoolStatistics(
            total_connections=5,
            active_connections=2,
            idle_connections=3
        )
        
        measurements = await resource_monitor.measure_resources(mock_pool)
        
        assert measurements is not None
        assert "connection_count" in measurements
        assert "memory_usage_mb" in measurements
        assert "cpu_usage_percent" in measurements
    
    @pytest.mark.asyncio
    async def test_monitoring_lifecycle(self, resource_monitor):
        """Test resource monitoring lifecycle."""
        mock_pool = Mock()
        mock_pool.get_statistics.return_value = PoolStatistics()
        
        # Start monitoring
        await resource_monitor.start_monitoring(mock_pool)
        assert resource_monitor.is_monitoring == True
        
        # Stop monitoring
        await resource_monitor.stop_monitoring()
        assert resource_monitor.is_monitoring == False
    
    def test_threshold_checking(self, resource_monitor):
        """Test resource threshold checking."""
        # High resource usage
        high_usage = {
            "memory_usage_mb": 500,
            "cpu_usage_percent": 90,
            "connection_count": 50
        }
        
        # Normal resource usage
        normal_usage = {
            "memory_usage_mb": 100,
            "cpu_usage_percent": 30,
            "connection_count": 10
        }
        
        high_violations = resource_monitor.check_thresholds(high_usage)
        normal_violations = resource_monitor.check_thresholds(normal_usage)
        
        assert len(high_violations) > 0
        assert len(normal_violations) == 0


class TestPoolHealthChecker:
    """Test pool health checking."""
    
    @pytest.fixture
    def health_checker(self):
        """Create pool health checker for testing."""
        config = PoolConfig()
        checker = PoolHealthChecker(config)
        return checker
    
    @pytest.mark.asyncio
    async def test_connection_health_check(self, health_checker):
        """Test individual connection health check."""
        # Healthy connection
        healthy_conn = Mock()
        healthy_conn.is_connected = Mock(return_value=True)
        healthy_conn.ping = AsyncMock(return_value=True)
        
        pooled_healthy = PooledConnection(
            connection=healthy_conn,
            connection_id="healthy"
        )
        
        health_result = await health_checker.check_connection_health(pooled_healthy)
        assert health_result == True
        
        # Unhealthy connection
        unhealthy_conn = Mock()
        unhealthy_conn.is_connected = Mock(return_value=False)
        
        pooled_unhealthy = PooledConnection(
            connection=unhealthy_conn,
            connection_id="unhealthy"
        )
        
        health_result = await health_checker.check_connection_health(pooled_unhealthy)
        assert health_result == False
    
    @pytest.mark.asyncio
    async def test_pool_health_assessment(self, health_checker):
        """Test overall pool health assessment."""
        connections = [
            PooledConnection(Mock(), "conn_1", state=ConnectionState.ACTIVE),
            PooledConnection(Mock(), "conn_2", state=ConnectionState.IDLE),
            PooledConnection(Mock(), "conn_3", state=ConnectionState.FAILED)
        ]
        
        health_report = await health_checker.assess_pool_health(connections)
        
        assert health_report is not None
        assert "healthy_connections" in health_report
        assert "failed_connections" in health_report
        assert "overall_health_score" in health_report


class TestConnectionPool:
    """Test connection pool main functionality."""
    
    @pytest.fixture
    def connection_pool(self, mock_connection_factory):
        """Create connection pool for testing."""
        config = PoolConfig(
            min_connections=2,
            max_connections=5,
            initial_connections=3
        )
        pool = ConnectionPool(config, mock_connection_factory)
        return pool
    
    def test_pool_initialization(self, connection_pool):
        """Test connection pool initialization."""
        assert connection_pool.config is not None
        assert connection_pool.factory is not None
        assert len(connection_pool.connections) == 0
        assert connection_pool.is_initialized == False
        assert connection_pool.statistics is not None
    
    @pytest.mark.asyncio
    async def test_pool_initialization_flow(self, connection_pool):
        """Test pool initialization flow."""
        await connection_pool.initialize()
        
        assert connection_pool.is_initialized == True
        assert len(connection_pool.connections) == connection_pool.config.initial_connections
        
        # All connections should be idle
        idle_connections = [conn for conn in connection_pool.connections if conn.state == ConnectionState.IDLE]
        assert len(idle_connections) == connection_pool.config.initial_connections
    
    @pytest.mark.asyncio
    async def test_connection_acquisition(self, connection_pool):
        """Test connection acquisition from pool."""
        await connection_pool.initialize()
        
        # Acquire connection
        connection = await connection_pool.acquire_connection()
        
        assert connection is not None
        assert connection.state == ConnectionState.ACTIVE
        assert connection.use_count == 1
        
        # Pool should have one less idle connection
        idle_count = len([conn for conn in connection_pool.connections if conn.state == ConnectionState.IDLE])
        assert idle_count == connection_pool.config.initial_connections - 1
    
    @pytest.mark.asyncio
    async def test_connection_release(self, connection_pool):
        """Test connection release back to pool."""
        await connection_pool.initialize()
        
        # Acquire and release connection
        connection = await connection_pool.acquire_connection()
        await connection_pool.release_connection(connection)
        
        assert connection.state == ConnectionState.IDLE
        
        # Pool should have all connections idle again
        idle_count = len([conn for conn in connection_pool.connections if conn.state == ConnectionState.IDLE])
        assert idle_count == connection_pool.config.initial_connections
    
    @pytest.mark.asyncio
    async def test_pool_expansion(self, connection_pool):
        """Test pool expansion when needed."""
        await connection_pool.initialize()
        
        # Acquire all initial connections
        acquired_connections = []
        for _ in range(connection_pool.config.initial_connections):
            conn = await connection_pool.acquire_connection()
            acquired_connections.append(conn)
        
        # Acquire one more (should trigger expansion)
        extra_conn = await connection_pool.acquire_connection()
        acquired_connections.append(extra_conn)
        
        assert len(connection_pool.connections) == connection_pool.config.initial_connections + 1
        assert extra_conn is not None
        
        # Clean up
        for conn in acquired_connections:
            await connection_pool.release_connection(conn)
    
    @pytest.mark.asyncio
    async def test_pool_max_limit(self, connection_pool):
        """Test pool maximum connection limit."""
        await connection_pool.initialize()
        
        # Acquire maximum connections
        acquired_connections = []
        for _ in range(connection_pool.config.max_connections):
            try:
                conn = await connection_pool.acquire_connection(timeout_seconds=0.1)
                if conn:
                    acquired_connections.append(conn)
            except asyncio.TimeoutError:
                break
        
        # Should not exceed max connections
        assert len(connection_pool.connections) <= connection_pool.config.max_connections
        
        # Clean up
        for conn in acquired_connections:
            await connection_pool.release_connection(conn)
    
    @pytest.mark.asyncio
    async def test_connection_cleanup(self, connection_pool):
        """Test connection cleanup (expired/idle connections)."""
        await connection_pool.initialize()
        
        # Create expired connection
        expired_conn = PooledConnection(
            connection=Mock(),
            connection_id="expired",
            created_at=time.time() - 4000  # 4000 seconds ago
        )
        expired_conn.state = ConnectionState.IDLE
        connection_pool.connections.append(expired_conn)
        
        initial_count = len(connection_pool.connections)
        
        # Run cleanup
        await connection_pool._cleanup_connections()
        
        # Expired connection should be removed
        assert len(connection_pool.connections) < initial_count
        assert not any(conn.connection_id == "expired" for conn in connection_pool.connections)
    
    @pytest.mark.asyncio
    async def test_pool_statistics_tracking(self, connection_pool):
        """Test pool statistics tracking."""
        await connection_pool.initialize()
        
        # Perform operations to generate statistics
        conn = await connection_pool.acquire_connection()
        await connection_pool.release_connection(conn)
        
        stats = connection_pool.get_statistics()
        
        assert stats.total_requests >= 1
        assert stats.successful_requests >= 1
        assert stats.total_connections == connection_pool.config.initial_connections
    
    def test_callback_registration(self, connection_pool):
        """Test pool event callback registration."""
        events_received = []
        
        def pool_callback(event):
            events_received.append(event)
        
        connection_pool.on_connection_created(pool_callback)
        connection_pool.on_connection_removed(pool_callback)
        connection_pool.on_pool_exhausted(pool_callback)
        
        assert len(connection_pool.connection_created_callbacks) == 1
        assert len(connection_pool.connection_removed_callbacks) == 1
        assert len(connection_pool.pool_exhausted_callbacks) == 1
    
    @pytest.mark.asyncio
    async def test_pool_shutdown(self, connection_pool):
        """Test pool shutdown and cleanup."""
        await connection_pool.initialize()
        
        initial_count = len(connection_pool.connections)
        assert initial_count > 0
        
        # Shutdown pool
        await connection_pool.shutdown()
        
        assert connection_pool.is_initialized == False
        assert len(connection_pool.connections) == 0


class TestPoolEvent:
    """Test pool event handling."""
    
    def test_pool_event_creation(self):
        """Test pool event creation."""
        event = PoolEvent(
            event_type=PoolEventType.CONNECTION_CREATED,
            connection_id="new_conn_123",
            pool_size=5,
            active_connections=2,
            message="New connection created",
            timestamp=time.time()
        )
        
        assert event.event_type == PoolEventType.CONNECTION_CREATED
        assert event.connection_id == "new_conn_123"
        assert event.pool_size == 5
        assert event.active_connections == 2
        assert event.message == "New connection created"
    
    def test_event_types(self):
        """Test pool event types."""
        assert PoolEventType.CONNECTION_CREATED.value == "connection_created"
        assert PoolEventType.CONNECTION_REMOVED.value == "connection_removed"
        assert PoolEventType.CONNECTION_ACQUIRED.value == "connection_acquired"
        assert PoolEventType.CONNECTION_RELEASED.value == "connection_released"
        assert PoolEventType.POOL_EXHAUSTED.value == "pool_exhausted"
        assert PoolEventType.POOL_EXPANDED.value == "pool_expanded"


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.fixture
    def error_pool(self):
        """Create connection pool for error testing."""
        config = PoolConfig(min_connections=1, max_connections=3)
        
        # Factory that sometimes fails
        async def failing_factory():
            if hasattr(failing_factory, 'call_count'):
                failing_factory.call_count += 1
            else:
                failing_factory.call_count = 1
            
            if failing_factory.call_count % 3 == 0:  # Fail every 3rd call
                raise Exception("Connection creation failed")
            
            conn = Mock()
            conn.connect = AsyncMock(return_value=True)
            conn.close = AsyncMock()
            return conn
        
        pool = ConnectionPool(config, failing_factory)
        return pool
    
    @pytest.mark.asyncio
    async def test_connection_creation_failure_handling(self, error_pool):
        """Test handling of connection creation failures."""
        # Should handle creation failures gracefully
        await error_pool.initialize()
        
        # May have fewer connections than requested due to failures
        assert len(error_pool.connections) <= error_pool.config.initial_connections
    
    @pytest.mark.asyncio
    async def test_connection_acquisition_timeout(self, error_pool):
        """Test connection acquisition timeout."""
        await error_pool.initialize()
        
        # Acquire all available connections
        acquired = []
        for _ in range(error_pool.config.max_connections):
            try:
                conn = await error_pool.acquire_connection(timeout_seconds=0.1)
                if conn:
                    acquired.append(conn)
            except asyncio.TimeoutError:
                break
        
        # Next acquisition should timeout
        with pytest.raises(asyncio.TimeoutError):
            await error_pool.acquire_connection(timeout_seconds=0.1)
        
        # Clean up
        for conn in acquired:
            await error_pool.release_connection(conn)
    
    @pytest.mark.asyncio
    async def test_invalid_connection_release(self, error_pool):
        """Test release of invalid/unknown connection."""
        await error_pool.initialize()
        
        # Try to release connection not from pool
        unknown_conn = PooledConnection(Mock(), "unknown")
        
        # Should handle gracefully
        result = await error_pool.release_connection(unknown_conn)
        assert result == False
    
    @pytest.mark.asyncio
    async def test_callback_error_handling(self, error_pool):
        """Test error handling in pool callbacks."""
        def failing_callback(event):
            raise Exception("Callback failed")
        
        error_pool.on_connection_created(failing_callback)
        
        # Should not crash when callback fails
        await error_pool.initialize()
        # Test continues if no exception raised


class TestPerformance:
    """Test performance characteristics."""
    
    @pytest.fixture
    def perf_pool(self):
        """Create connection pool for performance testing."""
        config = PoolConfig(
            min_connections=5,
            max_connections=20,
            initial_connections=10
        )
        
        async def fast_factory():
            conn = Mock()
            conn.connect = AsyncMock(return_value=True)
            conn.close = AsyncMock()
            return conn
        
        pool = ConnectionPool(config, fast_factory)
        return pool
    
    @pytest.mark.asyncio
    async def test_acquisition_performance(self, perf_pool):
        """Test connection acquisition performance."""
        await perf_pool.initialize()
        
        import time
        start_time = time.time()
        
        # Acquire and release connections rapidly
        for _ in range(100):
            conn = await perf_pool.acquire_connection()
            await perf_pool.release_connection(conn)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should handle rapid acquisition/release efficiently
        assert total_time < 1.0  # Less than 1 second for 100 operations
    
    @pytest.mark.asyncio
    async def test_concurrent_access_performance(self, perf_pool):
        """Test concurrent connection access performance."""
        await perf_pool.initialize()
        
        async def worker():
            for _ in range(10):
                conn = await perf_pool.acquire_connection()
                await asyncio.sleep(0.001)  # Simulate work
                await perf_pool.release_connection(conn)
        
        import time
        start_time = time.time()
        
        # Run multiple concurrent workers
        await asyncio.gather(*[worker() for _ in range(10)])
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should handle concurrent access efficiently
        assert total_time < 2.0  # Reasonable time for concurrent operations


# Integration test markers
pytestmark = pytest.mark.unit