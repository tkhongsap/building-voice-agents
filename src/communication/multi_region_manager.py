"""
Multi-Region Deployment Manager for Global Latency Optimization

This module provides multi-region deployment support for optimizing global latency
by automatically selecting the best LiveKit server region for each connection.
"""

import asyncio
import logging
import time
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple, Set
import json
import random

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None

try:
    import ping3
    PING_AVAILABLE = True
except ImportError:
    PING_AVAILABLE = False
    ping3 = None

from monitoring.performance_monitor import monitor_performance, global_performance_monitor

logger = logging.getLogger(__name__)


class RegionStatus(Enum):
    """Status of a region."""
    AVAILABLE = "available"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    MAINTENANCE = "maintenance"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    LATENCY_BASED = "latency_based"        # Select lowest latency
    ROUND_ROBIN = "round_robin"            # Rotate through regions
    WEIGHTED_ROUND_ROBIN = "weighted"      # Weighted by capacity
    LEAST_CONNECTIONS = "least_connections" # Fewest active connections
    GEOGRAPHIC = "geographic"              # Closest geographic region
    CUSTOM = "custom"                      # Custom strategy


@dataclass
class RegionInfo:
    """Information about a LiveKit region."""
    region_id: str
    region_name: str
    endpoint_url: str
    geographic_location: str
    
    # Performance metrics
    average_latency_ms: float = 0.0
    current_latency_ms: Optional[float] = None
    packet_loss_percent: float = 0.0
    
    # Capacity metrics
    max_connections: int = 1000
    current_connections: int = 0
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    
    # Status
    status: RegionStatus = RegionStatus.AVAILABLE
    last_health_check: Optional[float] = None
    health_score: float = 100.0  # 0-100
    
    # Configuration
    weight: float = 1.0  # Load balancing weight
    priority: int = 1    # Higher number = higher priority
    
    # Statistics
    total_connections_served: int = 0
    error_rate: float = 0.0
    success_rate: float = 100.0


@dataclass
class ConnectionRequest:
    """Request for connection to a region."""
    client_id: str
    client_ip: Optional[str] = None
    client_location: Optional[str] = None
    required_features: List[str] = field(default_factory=list)
    performance_requirements: Dict[str, float] = field(default_factory=dict)
    fallback_regions: List[str] = field(default_factory=list)


@dataclass
class RegionSelection:
    """Result of region selection."""
    selected_region: RegionInfo
    selection_reason: str
    alternative_regions: List[RegionInfo] = field(default_factory=list)
    estimated_latency_ms: float = 0.0
    confidence_score: float = 0.0  # 0-1


class MultiRegionManager:
    """Manages multi-region deployment and optimal region selection."""
    
    def __init__(self, 
                 load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.LATENCY_BASED,
                 health_check_interval_seconds: int = 30):
        self.load_balancing_strategy = load_balancing_strategy
        self.health_check_interval_seconds = health_check_interval_seconds
        
        # Region management
        self.regions: Dict[str, RegionInfo] = {}
        self.region_groups: Dict[str, List[str]] = {}  # e.g., "us": ["us-east", "us-west"]
        
        # Load balancing state
        self.round_robin_index = 0
        self.connection_counts: Dict[str, int] = {}
        
        # Performance tracking
        self.latency_cache: Dict[str, Dict[str, float]] = {}  # client_location -> region -> latency
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Configuration
        self.failover_enabled = True
        self.max_failover_attempts = 3
        self.latency_threshold_ms = 200.0
        self.health_check_timeout_seconds = 5.0
        
        # Monitoring
        self.monitor = global_performance_monitor.register_component(
            "multi_region_manager", "communication"
        )
        
        # Background tasks
        self.is_monitoring = False
        self._health_check_task: Optional[asyncio.Task] = None
        self._metrics_collection_task: Optional[asyncio.Task] = None
    
    async def initialize(self):
        """Initialize multi-region manager."""
        logger.info("Initializing multi-region manager")
        
        # Load default regions if none configured
        if not self.regions:
            await self._load_default_regions()
        
        # Start background monitoring
        await self.start_monitoring()
        
        logger.info(f"Multi-region manager initialized with {len(self.regions)} regions")
    
    async def _load_default_regions(self):
        """Load default LiveKit Cloud regions."""
        default_regions = [
            {
                "region_id": "us-east-1",
                "region_name": "US East (Virginia)",
                "endpoint_url": "wss://us-east-1.livekit.cloud",
                "geographic_location": "North America",
                "weight": 1.0,
                "priority": 1
            },
            {
                "region_id": "us-west-2",
                "region_name": "US West (Oregon)", 
                "endpoint_url": "wss://us-west-2.livekit.cloud",
                "geographic_location": "North America",
                "weight": 1.0,
                "priority": 1
            },
            {
                "region_id": "eu-west-1",
                "region_name": "Europe (Ireland)",
                "endpoint_url": "wss://eu-west-1.livekit.cloud",
                "geographic_location": "Europe",
                "weight": 1.0,
                "priority": 1
            },
            {
                "region_id": "ap-southeast-1",
                "region_name": "Asia Pacific (Singapore)",
                "endpoint_url": "wss://ap-southeast-1.livekit.cloud",
                "geographic_location": "Asia Pacific",
                "weight": 1.0,
                "priority": 1
            }
        ]
        
        for region_data in default_regions:
            region = RegionInfo(**region_data)
            await self.add_region(region)
    
    async def add_region(self, region: RegionInfo):
        """Add a new region to the manager."""
        self.regions[region.region_id] = region
        self.connection_counts[region.region_id] = 0
        
        # Initialize performance tracking
        self.performance_history[region.region_id] = []
        
        logger.info(f"Added region: {region.region_name} ({region.region_id})")
    
    async def remove_region(self, region_id: str):
        """Remove a region from the manager."""
        if region_id in self.regions:
            region = self.regions[region_id]
            del self.regions[region_id]
            del self.connection_counts[region_id]
            
            if region_id in self.performance_history:
                del self.performance_history[region_id]
            
            logger.info(f"Removed region: {region.region_name} ({region_id})")
    
    async def start_monitoring(self):
        """Start background monitoring tasks."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        self._metrics_collection_task = asyncio.create_task(self._metrics_collection_loop())
        
        logger.info("Started multi-region monitoring")
    
    async def stop_monitoring(self):
        """Stop background monitoring."""
        self.is_monitoring = False
        
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        if self._metrics_collection_task:
            self._metrics_collection_task.cancel()
            try:
                await self._metrics_collection_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped multi-region monitoring")
    
    @monitor_performance(component="multi_region", operation="select_region")
    async def select_optimal_region(self, request: ConnectionRequest) -> RegionSelection:
        """Select the optimal region for a connection request."""
        try:
            available_regions = [
                region for region in self.regions.values()
                if region.status == RegionStatus.AVAILABLE
            ]
            
            if not available_regions:
                raise RuntimeError("No available regions")
            
            # Apply load balancing strategy
            if self.load_balancing_strategy == LoadBalancingStrategy.LATENCY_BASED:
                selected = await self._select_by_latency(request, available_regions)
            elif self.load_balancing_strategy == LoadBalancingStrategy.ROUND_ROBIN:
                selected = await self._select_round_robin(available_regions)
            elif self.load_balancing_strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
                selected = await self._select_weighted_round_robin(available_regions)
            elif self.load_balancing_strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                selected = await self._select_least_connections(available_regions)
            elif self.load_balancing_strategy == LoadBalancingStrategy.GEOGRAPHIC:
                selected = await self._select_geographic(request, available_regions)
            else:
                # Default to latency-based
                selected = await self._select_by_latency(request, available_regions)
            
            # Get alternative regions for failover
            alternatives = [r for r in available_regions if r.region_id != selected.region_id]
            alternatives.sort(key=lambda x: x.average_latency_ms)
            
            return RegionSelection(
                selected_region=selected,
                selection_reason=f"Selected by {self.load_balancing_strategy.value}",
                alternative_regions=alternatives[:3],  # Top 3 alternatives
                estimated_latency_ms=selected.average_latency_ms,
                confidence_score=self._calculate_confidence_score(selected)
            )
            
        except Exception as e:
            logger.error(f"Error selecting region: {e}")
            # Fallback to first available region
            if available_regions:
                return RegionSelection(
                    selected_region=available_regions[0],
                    selection_reason="Fallback selection",
                    confidence_score=0.5
                )
            raise
    
    async def _select_by_latency(self, request: ConnectionRequest, regions: List[RegionInfo]) -> RegionInfo:
        """Select region with lowest latency."""
        if request.client_location and request.client_location in self.latency_cache:
            # Use cached latency data
            cached_latencies = self.latency_cache[request.client_location]
            
            best_region = None
            best_latency = float('inf')
            
            for region in regions:
                if region.region_id in cached_latencies:
                    latency = cached_latencies[region.region_id]
                    if latency < best_latency:
                        best_latency = latency
                        best_region = region
            
            if best_region:
                return best_region
        
        # Fallback to average latency
        return min(regions, key=lambda r: r.average_latency_ms)
    
    async def _select_round_robin(self, regions: List[RegionInfo]) -> RegionInfo:
        """Select region using round-robin."""
        if not regions:
            raise ValueError("No regions available")
        
        selected = regions[self.round_robin_index % len(regions)]
        self.round_robin_index = (self.round_robin_index + 1) % len(regions)
        
        return selected
    
    async def _select_weighted_round_robin(self, regions: List[RegionInfo]) -> RegionInfo:
        """Select region using weighted round-robin."""
        # Calculate total weight
        total_weight = sum(r.weight for r in regions)
        
        # Generate random number
        random_value = random.uniform(0, total_weight)
        
        # Select region based on weight
        current_weight = 0
        for region in regions:
            current_weight += region.weight
            if random_value <= current_weight:
                return region
        
        # Fallback to last region
        return regions[-1]
    
    async def _select_least_connections(self, regions: List[RegionInfo]) -> RegionInfo:
        """Select region with fewest connections."""
        return min(regions, key=lambda r: r.current_connections)
    
    async def _select_geographic(self, request: ConnectionRequest, regions: List[RegionInfo]) -> RegionInfo:
        """Select region based on geographic proximity."""
        if not request.client_location:
            # Fallback to latency-based selection
            return await self._select_by_latency(request, regions)
        
        # Simple geographic matching
        location_mapping = {
            "us": "North America",
            "ca": "North America",
            "mx": "North America",
            "gb": "Europe",
            "de": "Europe",
            "fr": "Europe",
            "jp": "Asia Pacific",
            "au": "Asia Pacific",
            "sg": "Asia Pacific"
        }
        
        client_region = location_mapping.get(request.client_location.lower()[:2], "North America")
        
        # Find regions in the same geographic area
        matching_regions = [r for r in regions if r.geographic_location == client_region]
        
        if matching_regions:
            # Select best region in the same geographic area
            return min(matching_regions, key=lambda r: r.average_latency_ms)
        
        # Fallback to best overall region
        return min(regions, key=lambda r: r.average_latency_ms)
    
    def _calculate_confidence_score(self, region: RegionInfo) -> float:
        """Calculate confidence score for region selection."""
        score = 1.0
        
        # Reduce confidence for high latency
        if region.average_latency_ms > self.latency_threshold_ms:
            score *= 0.8
        
        # Reduce confidence for high load
        if region.current_connections > region.max_connections * 0.8:
            score *= 0.7
        
        # Reduce confidence for poor health
        score *= (region.health_score / 100.0)
        
        # Reduce confidence for errors
        if region.error_rate > 5.0:  # >5% error rate
            score *= 0.6
        
        return max(0.0, min(1.0, score))
    
    async def record_connection_start(self, region_id: str, client_id: str):
        """Record that a connection has started."""
        if region_id in self.regions:
            self.regions[region_id].current_connections += 1
            self.regions[region_id].total_connections_served += 1
            self.connection_counts[region_id] = self.regions[region_id].current_connections
            
            logger.debug(f"Connection started in region {region_id}: {self.regions[region_id].current_connections} active")
    
    async def record_connection_end(self, region_id: str, client_id: str, success: bool = True):
        """Record that a connection has ended."""
        if region_id in self.regions:
            self.regions[region_id].current_connections = max(0, self.regions[region_id].current_connections - 1)
            self.connection_counts[region_id] = self.regions[region_id].current_connections
            
            # Update success/error rates
            if success:
                # Implement exponential moving average for success rate
                self.regions[region_id].success_rate = (
                    self.regions[region_id].success_rate * 0.95 + 100.0 * 0.05
                )
                self.regions[region_id].error_rate = (
                    self.regions[region_id].error_rate * 0.95
                )
            else:
                self.regions[region_id].error_rate = (
                    self.regions[region_id].error_rate * 0.95 + 100.0 * 0.05
                )
                self.regions[region_id].success_rate = (
                    self.regions[region_id].success_rate * 0.95
                )
            
            logger.debug(f"Connection ended in region {region_id}: {self.regions[region_id].current_connections} active")
    
    async def update_region_latency(self, region_id: str, client_location: str, latency_ms: float):
        """Update latency measurement for a region."""
        if region_id in self.regions:
            region = self.regions[region_id]
            
            # Update current latency
            region.current_latency_ms = latency_ms
            
            # Update average latency using exponential moving average
            if region.average_latency_ms == 0:
                region.average_latency_ms = latency_ms
            else:
                region.average_latency_ms = region.average_latency_ms * 0.9 + latency_ms * 0.1
            
            # Cache latency by client location
            if client_location not in self.latency_cache:
                self.latency_cache[client_location] = {}
            
            self.latency_cache[client_location][region_id] = latency_ms
            
            logger.debug(f"Updated latency for {region_id}: {latency_ms:.1f}ms")
    
    async def _health_check_loop(self):
        """Background health check loop."""
        while self.is_monitoring:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(5.0)
    
    async def _perform_health_checks(self):
        """Perform health checks on all regions."""
        tasks = []
        
        for region_id, region in self.regions.items():
            task = asyncio.create_task(self._check_region_health(region))
            tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _check_region_health(self, region: RegionInfo):
        """Check health of a specific region."""
        try:
            start_time = time.time()
            
            # Perform health check (simplified)
            if REQUESTS_AVAILABLE:
                # Try to connect to the WebSocket endpoint
                health_url = region.endpoint_url.replace('wss://', 'https://').replace('ws://', 'http://') + '/health'
                
                try:
                    response = requests.get(health_url, timeout=self.health_check_timeout_seconds)
                    is_healthy = response.status_code == 200
                except requests.RequestException:
                    is_healthy = False
            else:
                # Mock health check
                is_healthy = True
            
            # Measure latency
            if PING_AVAILABLE:
                hostname = region.endpoint_url.split('://')[1].split('/')[0]
                latency = ping3.ping(hostname, timeout=self.health_check_timeout_seconds)
                
                if latency is not None:
                    region.current_latency_ms = latency * 1000
                    
                    # Update average latency
                    if region.average_latency_ms == 0:
                        region.average_latency_ms = region.current_latency_ms
                    else:
                        region.average_latency_ms = (
                            region.average_latency_ms * 0.9 + region.current_latency_ms * 0.1
                        )
            
            # Update region status
            if is_healthy:
                if region.status == RegionStatus.UNAVAILABLE:
                    logger.info(f"Region {region.region_id} is now available")
                region.status = RegionStatus.AVAILABLE
                region.health_score = min(100.0, region.health_score + 5.0)
            else:
                logger.warning(f"Region {region.region_id} health check failed")
                region.status = RegionStatus.UNAVAILABLE
                region.health_score = max(0.0, region.health_score - 20.0)
            
            region.last_health_check = time.time()
            
        except Exception as e:
            logger.error(f"Health check error for region {region.region_id}: {e}")
            region.status = RegionStatus.UNAVAILABLE
            region.health_score = max(0.0, region.health_score - 10.0)
    
    async def _metrics_collection_loop(self):
        """Background metrics collection loop."""
        while self.is_monitoring:
            try:
                await self._collect_metrics()
                await asyncio.sleep(60.0)  # Collect metrics every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(30.0)
    
    async def _collect_metrics(self):
        """Collect performance metrics for all regions."""
        current_time = time.time()
        
        for region_id, region in self.regions.items():
            metrics = {
                "timestamp": current_time,
                "latency_ms": region.current_latency_ms,
                "connections": region.current_connections,
                "health_score": region.health_score,
                "error_rate": region.error_rate,
                "success_rate": region.success_rate
            }
            
            # Store metrics history (keep last 24 hours)
            history = self.performance_history[region_id]
            history.append(metrics)
            
            # Keep only last 24 hours (1440 minutes)
            cutoff_time = current_time - (24 * 60 * 60)
            self.performance_history[region_id] = [
                m for m in history if m["timestamp"] > cutoff_time
            ]
    
    # Configuration and management methods
    
    def set_load_balancing_strategy(self, strategy: LoadBalancingStrategy):
        """Change load balancing strategy."""
        self.load_balancing_strategy = strategy
        logger.info(f"Changed load balancing strategy to {strategy.value}")
    
    def set_region_weight(self, region_id: str, weight: float):
        """Set weight for a region (used in weighted round-robin)."""
        if region_id in self.regions:
            self.regions[region_id].weight = weight
            logger.info(f"Set weight for region {region_id}: {weight}")
    
    def set_region_priority(self, region_id: str, priority: int):
        """Set priority for a region."""
        if region_id in self.regions:
            self.regions[region_id].priority = priority
            logger.info(f"Set priority for region {region_id}: {priority}")
    
    def enable_region(self, region_id: str):
        """Enable a region."""
        if region_id in self.regions:
            self.regions[region_id].status = RegionStatus.AVAILABLE
            logger.info(f"Enabled region {region_id}")
    
    def disable_region(self, region_id: str):
        """Disable a region."""
        if region_id in self.regions:
            self.regions[region_id].status = RegionStatus.MAINTENANCE
            logger.info(f"Disabled region {region_id}")
    
    # Status and reporting methods
    
    def get_region_status(self) -> Dict[str, Any]:
        """Get status of all regions."""
        return {
            region_id: {
                "name": region.region_name,
                "status": region.status.value,
                "latency_ms": region.average_latency_ms,
                "connections": region.current_connections,
                "health_score": region.health_score,
                "error_rate": region.error_rate,
                "last_check": region.last_health_check
            }
            for region_id, region in self.regions.items()
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across all regions."""
        if not self.regions:
            return {}
        
        total_connections = sum(r.current_connections for r in self.regions.values())
        avg_latency = statistics.mean([r.average_latency_ms for r in self.regions.values() if r.average_latency_ms > 0])
        avg_health = statistics.mean([r.health_score for r in self.regions.values()])
        
        return {
            "total_regions": len(self.regions),
            "available_regions": len([r for r in self.regions.values() if r.status == RegionStatus.AVAILABLE]),
            "total_connections": total_connections,
            "average_latency_ms": avg_latency,
            "average_health_score": avg_health,
            "load_balancing_strategy": self.load_balancing_strategy.value
        }
    
    def get_region_metrics(self, region_id: str, hours: int = 1) -> List[Dict[str, Any]]:
        """Get historical metrics for a region."""
        if region_id not in self.performance_history:
            return []
        
        cutoff_time = time.time() - (hours * 60 * 60)
        return [
            m for m in self.performance_history[region_id]
            if m["timestamp"] > cutoff_time
        ]
    
    async def cleanup(self):
        """Clean up resources."""
        await self.stop_monitoring()
        logger.info("Multi-region manager cleaned up")


# Convenience functions

async def create_multi_region_manager(**kwargs) -> MultiRegionManager:
    """Create and initialize multi-region manager."""
    manager = MultiRegionManager(**kwargs)
    await manager.initialize()
    return manager


def create_connection_request(client_id: str, **kwargs) -> ConnectionRequest:
    """Create a connection request."""
    return ConnectionRequest(client_id=client_id, **kwargs)


# Global multi-region manager
_global_region_manager: Optional[MultiRegionManager] = None


def get_global_region_manager() -> Optional[MultiRegionManager]:
    """Get global multi-region manager."""
    return _global_region_manager


def set_global_region_manager(manager: MultiRegionManager):
    """Set global multi-region manager."""
    global _global_region_manager
    _global_region_manager = manager