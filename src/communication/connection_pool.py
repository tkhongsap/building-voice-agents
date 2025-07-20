"""
Connection Pooling and Resource Optimization for WebRTC

This module provides connection pooling, resource optimization, and efficient
management of WebRTC connections to minimize overhead and improve performance.
"""

import asyncio
import logging
import time
import weakref
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List, Set, Callable, Deque
from collections import deque, defaultdict
import gc
import threading

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

from monitoring.performance_monitor import monitor_performance, global_performance_monitor

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """Connection states in the pool."""
    IDLE = "idle"                    # Ready for use
    ACTIVE = "active"                # Currently in use
    CONNECTING = "connecting"        # Being established
    DISCONNECTING = "disconnecting"  # Being closed
    ERROR = "error"                  # Failed connection
    EXPIRED = "expired"              # Exceeded max lifetime


class PoolStrategy(Enum):
    """Connection pool strategies."""
    FIFO = "fifo"           # First In, First Out
    LIFO = "lifo"           # Last In, First Out
    LEAST_USED = "least_used"    # Connection with least usage
    ROUND_ROBIN = "round_robin"  # Rotate through connections


class ResourceType(Enum):
    """Types of resources to optimize."""
    MEMORY = "memory"
    CPU = "cpu"
    NETWORK = "network"
    STORAGE = "storage"


@dataclass
class ConnectionMetrics:
    """Metrics for a pooled connection."""
    connection_id: str
    created_at: float
    last_used_at: float
    total_usage_count: int = 0
    total_usage_duration: float = 0.0
    
    # Performance metrics
    avg_latency_ms: float = 0.0
    bytes_sent: int = 0
    bytes_received: int = 0
    
    # Error tracking
    error_count: int = 0
    last_error: Optional[str] = None
    last_error_at: Optional[float] = None
    
    # Resource usage
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0


@dataclass
class PoolConfiguration:
    """Configuration for connection pool."""
    # Pool sizing
    min_connections: int = 2
    max_connections: int = 20
    initial_connections: int = 5
    
    # Connection lifecycle
    max_idle_time_seconds: int = 300        # 5 minutes
    max_lifetime_seconds: int = 3600        # 1 hour
    connection_timeout_seconds: int = 30    # Connection establishment timeout
    
    # Pool strategy
    strategy: PoolStrategy = PoolStrategy.FIFO
    
    # Resource limits
    max_memory_per_connection_mb: float = 100.0
    max_cpu_per_connection_percent: float = 10.0
    
    # Optimization settings
    enable_preemptive_creation: bool = True
    enable_connection_warming: bool = True
    enable_resource_monitoring: bool = True
    
    # Cleanup intervals
    cleanup_interval_seconds: int = 60      # 1 minute
    metrics_interval_seconds: int = 30      # 30 seconds


@dataclass
class ResourceUsage:
    """Current resource usage statistics."""
    total_memory_mb: float = 0.0
    total_cpu_percent: float = 0.0
    connection_count: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    
    # System resources
    system_memory_percent: float = 0.0
    system_cpu_percent: float = 0.0
    available_memory_mb: float = 0.0


class PooledConnection:
    """Wrapper for a pooled WebRTC connection."""
    
    def __init__(self, connection_id: str, connection_factory: Callable):
        self.connection_id = connection_id
        self.connection_factory = connection_factory
        self.connection: Optional[Any] = None
        
        # State management
        self.state = ConnectionState.IDLE
        self.lock = asyncio.Lock()
        
        # Metrics
        self.metrics = ConnectionMetrics(
            connection_id=connection_id,
            created_at=time.time(),
            last_used_at=time.time()
        )
        
        # Weak reference to pool for cleanup
        self._pool_ref: Optional[weakref.ref] = None
    
    async def establish(self) -> bool:
        """Establish the actual connection."""
        async with self.lock:
            if self.state != ConnectionState.IDLE:
                return False
            
            try:
                self.state = ConnectionState.CONNECTING
                self.connection = await self.connection_factory()
                
                if self.connection:
                    self.state = ConnectionState.IDLE
                    self.metrics.created_at = time.time()
                    self.metrics.last_used_at = time.time()
                    return True
                else:
                    self.state = ConnectionState.ERROR
                    self.metrics.error_count += 1
                    self.metrics.last_error = "Connection factory returned None"
                    self.metrics.last_error_at = time.time()
                    return False
                    
            except Exception as e:
                self.state = ConnectionState.ERROR
                self.metrics.error_count += 1
                self.metrics.last_error = str(e)
                self.metrics.last_error_at = time.time()
                logger.error(f"Failed to establish connection {self.connection_id}: {e}")
                return False
    
    async def acquire(self) -> Optional[Any]:
        """Acquire the connection for use."""
        async with self.lock:
            if self.state != ConnectionState.IDLE:
                return None
            
            self.state = ConnectionState.ACTIVE
            self.metrics.total_usage_count += 1
            self.metrics.last_used_at = time.time()
            
            return self.connection
    
    async def release(self):
        """Release the connection back to idle state."""
        async with self.lock:
            if self.state == ConnectionState.ACTIVE:
                self.state = ConnectionState.IDLE
                
                # Update usage duration
                usage_duration = time.time() - self.metrics.last_used_at
                self.metrics.total_usage_duration += usage_duration
    
    async def close(self):
        """Close the connection."""
        async with self.lock:
            if self.connection and hasattr(self.connection, 'close'):
                try:
                    await self.connection.close()
                except Exception as e:
                    logger.error(f"Error closing connection {self.connection_id}: {e}")
            
            self.connection = None
            self.state = ConnectionState.DISCONNECTING
    
    def is_expired(self, max_idle_time: int, max_lifetime: int) -> bool:
        """Check if connection should be expired."""
        current_time = time.time()
        
        # Check idle time
        if self.state == ConnectionState.IDLE:
            idle_time = current_time - self.metrics.last_used_at
            if idle_time > max_idle_time:
                return True
        
        # Check lifetime
        lifetime = current_time - self.metrics.created_at
        if lifetime > max_lifetime:
            return True
        
        return False
    
    def update_resource_usage(self):
        """Update resource usage metrics."""
        if not PSUTIL_AVAILABLE:
            return
        
        try:
            # Simple resource tracking (in real implementation, would track per-connection)
            process = psutil.Process()
            self.metrics.memory_usage_mb = process.memory_info().rss / (1024 * 1024)
            self.metrics.cpu_usage_percent = process.cpu_percent()
            
        except Exception as e:
            logger.debug(f"Error updating resource usage: {e}")


class ConnectionPool:
    """Manages a pool of WebRTC connections for resource optimization."""
    
    def __init__(self, 
                 connection_factory: Callable,
                 config: PoolConfiguration = None):
        self.connection_factory = connection_factory
        self.config = config or PoolConfiguration()
        
        # Connection management
        self.connections: Dict[str, PooledConnection] = {}
        self.idle_connections: Deque[str] = deque()
        self.active_connections: Set[str] = set()
        
        # Pool state
        self.is_running = False
        self.connection_counter = 0
        self.round_robin_index = 0
        
        # Resource tracking
        self.resource_usage = ResourceUsage()
        
        # Event callbacks
        self.on_connection_created_callbacks: List[Callable] = []
        self.on_connection_destroyed_callbacks: List[Callable] = []
        self.on_resource_threshold_callbacks: List[Callable] = []
        
        # Monitoring
        self.monitor = global_performance_monitor.register_component(
            "connection_pool", "communication"
        )
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._metrics_task: Optional[asyncio.Task] = None
        
        # Thread safety
        self._lock = asyncio.Lock()
    
    async def initialize(self):
        """Initialize the connection pool."""
        logger.info("Initializing connection pool")
        
        self.is_running = True
        
        # Create initial connections
        await self._create_initial_connections()
        
        # Start background tasks
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        if self.config.enable_resource_monitoring:
            self._metrics_task = asyncio.create_task(self._metrics_loop())
        
        logger.info(f"Connection pool initialized with {len(self.connections)} connections")
    
    async def _create_initial_connections(self):
        """Create initial pool connections."""
        for i in range(self.config.initial_connections):
            await self._create_connection()
    
    async def _create_connection(self) -> Optional[str]:
        """Create a new pooled connection."""
        if len(self.connections) >= self.config.max_connections:
            logger.warning("Cannot create connection: pool at maximum capacity")
            return None
        
        connection_id = f"conn_{self.connection_counter}"
        self.connection_counter += 1
        
        try:
            pooled_conn = PooledConnection(connection_id, self.connection_factory)
            pooled_conn._pool_ref = weakref.ref(self)
            
            # Establish the connection
            if await pooled_conn.establish():
                self.connections[connection_id] = pooled_conn
                self.idle_connections.append(connection_id)
                
                # Trigger callback
                await self._trigger_connection_created(connection_id)
                
                logger.debug(f"Created connection: {connection_id}")
                return connection_id
            else:
                logger.error(f"Failed to establish connection: {connection_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating connection: {e}")
            return None
    
    @monitor_performance(component="connection_pool", operation="acquire_connection")
    async def acquire_connection(self, timeout_seconds: float = 30.0) -> Optional[Any]:
        """Acquire a connection from the pool."""
        start_time = time.time()
        
        while time.time() - start_time < timeout_seconds:
            async with self._lock:
                # Try to get an idle connection
                connection_id = await self._get_idle_connection()
                
                if connection_id:
                    pooled_conn = self.connections[connection_id]
                    connection = await pooled_conn.acquire()
                    
                    if connection:
                        # Move to active set
                        if connection_id in self.idle_connections:
                            self.idle_connections.remove(connection_id)
                        self.active_connections.add(connection_id)
                        
                        logger.debug(f"Acquired connection: {connection_id}")
                        return ConnectionWrapper(connection, pooled_conn, self)
                
                # Try to create a new connection if under limit
                if (len(self.connections) < self.config.max_connections and
                    self.config.enable_preemptive_creation):
                    
                    new_conn_id = await self._create_connection()
                    if new_conn_id:
                        # Try again with the new connection
                        continue
            
            # Wait a bit before retrying
            await asyncio.sleep(0.1)
        
        logger.warning("Failed to acquire connection: timeout exceeded")
        return None
    
    async def _get_idle_connection(self) -> Optional[str]:
        """Get an idle connection based on pool strategy."""
        if not self.idle_connections:
            return None
        
        if self.config.strategy == PoolStrategy.FIFO:
            return self.idle_connections.popleft()
        
        elif self.config.strategy == PoolStrategy.LIFO:
            return self.idle_connections.pop()
        
        elif self.config.strategy == PoolStrategy.LEAST_USED:
            # Find connection with least usage
            min_usage = float('inf')
            best_conn_id = None
            
            for conn_id in list(self.idle_connections):
                conn = self.connections[conn_id]
                if conn.metrics.total_usage_count < min_usage:
                    min_usage = conn.metrics.total_usage_count
                    best_conn_id = conn_id
            
            if best_conn_id:
                self.idle_connections.remove(best_conn_id)
                return best_conn_id
        
        elif self.config.strategy == PoolStrategy.ROUND_ROBIN:
            if self.idle_connections:
                # Round robin through idle connections
                index = self.round_robin_index % len(self.idle_connections)
                conn_id = list(self.idle_connections)[index]
                self.round_robin_index = (self.round_robin_index + 1) % len(self.idle_connections)
                self.idle_connections.remove(conn_id)
                return conn_id
        
        # Fallback to FIFO
        return self.idle_connections.popleft()
    
    async def release_connection(self, connection_id: str):
        """Release a connection back to the pool."""
        async with self._lock:
            if connection_id in self.connections:
                pooled_conn = self.connections[connection_id]
                await pooled_conn.release()
                
                # Move back to idle
                self.active_connections.discard(connection_id)
                
                # Check if connection should be kept
                if not pooled_conn.is_expired(
                    self.config.max_idle_time_seconds,
                    self.config.max_lifetime_seconds
                ):
                    self.idle_connections.append(connection_id)
                    logger.debug(f"Released connection: {connection_id}")
                else:
                    # Connection expired, remove it
                    await self._remove_connection(connection_id)
    
    async def _remove_connection(self, connection_id: str):
        """Remove a connection from the pool."""
        if connection_id in self.connections:
            pooled_conn = self.connections[connection_id]
            
            # Close the connection
            await pooled_conn.close()
            
            # Remove from all collections
            del self.connections[connection_id]
            self.active_connections.discard(connection_id)
            
            if connection_id in self.idle_connections:
                self.idle_connections.remove(connection_id)
            
            # Trigger callback
            await self._trigger_connection_destroyed(connection_id)
            
            logger.debug(f"Removed connection: {connection_id}")
    
    async def _cleanup_loop(self):
        """Background cleanup loop."""
        while self.is_running:
            try:
                await self._cleanup_expired_connections()
                await self._maintain_pool_size()
                await self._check_resource_limits()
                
                await asyncio.sleep(self.config.cleanup_interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(5.0)
    
    async def _cleanup_expired_connections(self):
        """Remove expired connections."""
        expired_connections = []
        
        for conn_id, pooled_conn in self.connections.items():
            if pooled_conn.is_expired(
                self.config.max_idle_time_seconds,
                self.config.max_lifetime_seconds
            ):
                expired_connections.append(conn_id)
        
        for conn_id in expired_connections:
            await self._remove_connection(conn_id)
    
    async def _maintain_pool_size(self):
        """Maintain minimum pool size."""
        current_idle = len(self.idle_connections)
        
        if current_idle < self.config.min_connections:
            needed = self.config.min_connections - current_idle
            
            for _ in range(min(needed, self.config.max_connections - len(self.connections))):
                await self._create_connection()
    
    async def _check_resource_limits(self):
        """Check and enforce resource limits."""
        over_limit_connections = []
        
        for conn_id, pooled_conn in self.connections.items():
            pooled_conn.update_resource_usage()
            
            # Check memory limit
            if pooled_conn.metrics.memory_usage_mb > self.config.max_memory_per_connection_mb:
                over_limit_connections.append((conn_id, "memory"))
            
            # Check CPU limit
            if pooled_conn.metrics.cpu_usage_percent > self.config.max_cpu_per_connection_percent:
                over_limit_connections.append((conn_id, "cpu"))
        
        # Remove connections that exceed limits
        for conn_id, limit_type in over_limit_connections:
            logger.warning(f"Removing connection {conn_id} due to {limit_type} limit")
            await self._remove_connection(conn_id)
    
    async def _metrics_loop(self):
        """Background metrics collection loop."""
        while self.is_running:
            try:
                await self._update_resource_usage()
                await asyncio.sleep(self.config.metrics_interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics loop: {e}")
                await asyncio.sleep(10.0)
    
    async def _update_resource_usage(self):
        """Update overall resource usage statistics."""
        self.resource_usage.connection_count = len(self.connections)
        self.resource_usage.active_connections = len(self.active_connections)
        self.resource_usage.idle_connections = len(self.idle_connections)
        
        # Calculate total resource usage
        total_memory = 0.0
        total_cpu = 0.0
        
        for pooled_conn in self.connections.values():
            total_memory += pooled_conn.metrics.memory_usage_mb
            total_cpu += pooled_conn.metrics.cpu_usage_percent
        
        self.resource_usage.total_memory_mb = total_memory
        self.resource_usage.total_cpu_percent = total_cpu
        
        # Get system resource usage
        if PSUTIL_AVAILABLE:
            try:
                memory = psutil.virtual_memory()
                self.resource_usage.system_memory_percent = memory.percent
                self.resource_usage.available_memory_mb = memory.available / (1024 * 1024)
                
                self.resource_usage.system_cpu_percent = psutil.cpu_percent(interval=1)
                
            except Exception as e:
                logger.debug(f"Error getting system resources: {e}")
    
    # Event callbacks
    async def _trigger_connection_created(self, connection_id: str):
        """Trigger connection created callbacks."""
        for callback in self.on_connection_created_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(connection_id)
                else:
                    callback(connection_id)
            except Exception as e:
                logger.error(f"Error in connection created callback: {e}")
    
    async def _trigger_connection_destroyed(self, connection_id: str):
        """Trigger connection destroyed callbacks."""
        for callback in self.on_connection_destroyed_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(connection_id)
                else:
                    callback(connection_id)
            except Exception as e:
                logger.error(f"Error in connection destroyed callback: {e}")
    
    async def _trigger_resource_threshold(self, resource_type: ResourceType, usage: float):
        """Trigger resource threshold callbacks."""
        for callback in self.on_resource_threshold_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(resource_type, usage)
                else:
                    callback(resource_type, usage)
            except Exception as e:
                logger.error(f"Error in resource threshold callback: {e}")
    
    # Callback registration
    def on_connection_created(self, callback: Callable):
        """Register callback for connection creation events."""
        self.on_connection_created_callbacks.append(callback)
    
    def on_connection_destroyed(self, callback: Callable):
        """Register callback for connection destruction events."""
        self.on_connection_destroyed_callbacks.append(callback)
    
    def on_resource_threshold(self, callback: Callable):
        """Register callback for resource threshold events."""
        self.on_resource_threshold_callbacks.append(callback)
    
    # Status and management methods
    def get_pool_status(self) -> Dict[str, Any]:
        """Get current pool status."""
        return {
            "total_connections": len(self.connections),
            "active_connections": len(self.active_connections),
            "idle_connections": len(self.idle_connections),
            "resource_usage": {
                "memory_mb": self.resource_usage.total_memory_mb,
                "cpu_percent": self.resource_usage.total_cpu_percent,
                "system_memory_percent": self.resource_usage.system_memory_percent,
                "system_cpu_percent": self.resource_usage.system_cpu_percent
            },
            "configuration": {
                "min_connections": self.config.min_connections,
                "max_connections": self.config.max_connections,
                "strategy": self.config.strategy.value
            }
        }
    
    def get_connection_metrics(self) -> List[Dict[str, Any]]:
        """Get metrics for all connections."""
        return [
            {
                "connection_id": conn.connection_id,
                "state": conn.state.value,
                "created_at": conn.metrics.created_at,
                "last_used_at": conn.metrics.last_used_at,
                "usage_count": conn.metrics.total_usage_count,
                "usage_duration": conn.metrics.total_usage_duration,
                "error_count": conn.metrics.error_count,
                "memory_mb": conn.metrics.memory_usage_mb,
                "cpu_percent": conn.metrics.cpu_usage_percent
            }
            for conn in self.connections.values()
        ]
    
    def force_cleanup(self):
        """Force garbage collection and cleanup."""
        gc.collect()
        logger.info("Forced garbage collection")
    
    async def resize_pool(self, min_connections: int, max_connections: int):
        """Resize the pool limits."""
        async with self._lock:
            self.config.min_connections = min_connections
            self.config.max_connections = max_connections
            
            # Remove excess connections if needed
            if len(self.connections) > max_connections:
                excess = len(self.connections) - max_connections
                idle_to_remove = list(self.idle_connections)[:excess]
                
                for conn_id in idle_to_remove:
                    await self._remove_connection(conn_id)
            
            logger.info(f"Pool resized: min={min_connections}, max={max_connections}")
    
    async def drain_pool(self):
        """Drain all connections from the pool."""
        async with self._lock:
            # Close all idle connections
            while self.idle_connections:
                conn_id = self.idle_connections.popleft()
                await self._remove_connection(conn_id)
            
            logger.info("Pool drained of idle connections")
    
    async def shutdown(self):
        """Shutdown the connection pool."""
        logger.info("Shutting down connection pool")
        
        self.is_running = False
        
        # Cancel background tasks
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        if self._metrics_task:
            self._metrics_task.cancel()
            try:
                await self._metrics_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        for conn_id in list(self.connections.keys()):
            await self._remove_connection(conn_id)
        
        logger.info("Connection pool shut down")


class ConnectionWrapper:
    """Wrapper that automatically releases connection when done."""
    
    def __init__(self, connection: Any, pooled_connection: PooledConnection, pool: ConnectionPool):
        self.connection = connection
        self.pooled_connection = pooled_connection
        self.pool = pool
        self._released = False
    
    async def __aenter__(self):
        return self.connection
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.release()
    
    async def release(self):
        """Release the connection back to the pool."""
        if not self._released:
            await self.pool.release_connection(self.pooled_connection.connection_id)
            self._released = True
    
    def __getattr__(self, name):
        """Delegate attribute access to the wrapped connection."""
        return getattr(self.connection, name)


# Resource optimization utilities

class ResourceOptimizer:
    """Optimizes resource usage across the system."""
    
    def __init__(self):
        self.memory_threshold_percent = 80.0
        self.cpu_threshold_percent = 75.0
        
    async def optimize_memory_usage(self):
        """Optimize memory usage."""
        if not PSUTIL_AVAILABLE:
            return
        
        memory = psutil.virtual_memory()
        
        if memory.percent > self.memory_threshold_percent:
            logger.warning(f"High memory usage detected: {memory.percent}%")
            
            # Force garbage collection
            gc.collect()
            
            # Additional cleanup could be implemented here
    
    async def optimize_cpu_usage(self):
        """Optimize CPU usage."""
        if not PSUTIL_AVAILABLE:
            return
        
        cpu_percent = psutil.cpu_percent(interval=1)
        
        if cpu_percent > self.cpu_threshold_percent:
            logger.warning(f"High CPU usage detected: {cpu_percent}%")
            
            # Could implement CPU throttling or connection limiting


# Convenience functions

async def create_connection_pool(connection_factory: Callable, **kwargs) -> ConnectionPool:
    """Create and initialize a connection pool."""
    config = PoolConfiguration(**kwargs)
    pool = ConnectionPool(connection_factory, config)
    await pool.initialize()
    return pool


# Global connection pool
_global_connection_pool: Optional[ConnectionPool] = None


def get_global_connection_pool() -> Optional[ConnectionPool]:
    """Get global connection pool."""
    return _global_connection_pool


def set_global_connection_pool(pool: ConnectionPool):
    """Set global connection pool."""
    global _global_connection_pool
    _global_connection_pool = pool