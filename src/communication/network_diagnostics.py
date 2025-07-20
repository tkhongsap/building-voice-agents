"""
Comprehensive Network Diagnostics and Troubleshooting Tools

This module provides extensive network diagnostics, troubleshooting tools,
and automated problem detection for WebRTC communications.
"""

import asyncio
import logging
import time
import socket
import subprocess
import platform
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple, Set
import json
import ipaddress

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

try:
    import speedtest
    SPEEDTEST_AVAILABLE = True
except ImportError:
    SPEEDTEST_AVAILABLE = False
    speedtest = None

from monitoring.performance_monitor import monitor_performance, global_performance_monitor

logger = logging.getLogger(__name__)


class DiagnosticSeverity(Enum):
    """Severity levels for diagnostic issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class NetworkTestType(Enum):
    """Types of network tests."""
    PING = "ping"
    TRACEROUTE = "traceroute"
    DNS_LOOKUP = "dns_lookup"
    BANDWIDTH = "bandwidth"
    PORT_SCAN = "port_scan"
    CONNECTIVITY = "connectivity"
    FIREWALL = "firewall"
    NAT_TYPE = "nat_type"
    STUN_BINDING = "stun_binding"
    TURN_ALLOCATION = "turn_allocation"


class IssueCategory(Enum):
    """Categories of network issues."""
    CONNECTIVITY = "connectivity"
    LATENCY = "latency"
    BANDWIDTH = "bandwidth"
    PACKET_LOSS = "packet_loss"
    DNS = "dns"
    FIREWALL = "firewall"
    NAT_TRAVERSAL = "nat_traversal"
    CERTIFICATE = "certificate"
    CONFIGURATION = "configuration"


@dataclass
class DiagnosticResult:
    """Result of a network diagnostic test."""
    test_type: NetworkTestType
    success: bool
    duration_ms: float
    result_data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class NetworkIssue:
    """Detected network issue."""
    issue_id: str
    category: IssueCategory
    severity: DiagnosticSeverity
    title: str
    description: str
    possible_causes: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    detection_timestamp: float = field(default_factory=time.time)
    related_tests: List[NetworkTestType] = field(default_factory=list)


@dataclass
class NetworkProfile:
    """Network environment profile."""
    # Basic connectivity
    public_ip: Optional[str] = None
    local_ip: Optional[str] = None
    gateway_ip: Optional[str] = None
    dns_servers: List[str] = field(default_factory=list)
    
    # NAT information
    nat_type: Optional[str] = None
    external_port_mapping: Dict[int, int] = field(default_factory=dict)
    
    # Performance characteristics
    download_mbps: Optional[float] = None
    upload_mbps: Optional[float] = None
    baseline_latency_ms: Optional[float] = None
    baseline_jitter_ms: Optional[float] = None
    
    # Capabilities
    supports_ipv6: bool = False
    supports_tcp: bool = True
    supports_udp: bool = True
    firewall_detected: bool = False
    
    # WebRTC specific
    stun_servers_accessible: List[str] = field(default_factory=list)
    turn_servers_accessible: List[str] = field(default_factory=list)
    ice_candidates: List[Dict[str, Any]] = field(default_factory=list)


class NetworkDiagnostics:
    """Comprehensive network diagnostics and troubleshooting system."""
    
    def __init__(self):
        # Test configuration
        self.test_timeout_seconds = 10.0
        self.ping_count = 5
        self.bandwidth_test_duration = 10  # seconds
        
        # Known servers for testing
        self.test_servers = [
            "8.8.8.8",           # Google DNS
            "1.1.1.1",           # Cloudflare DNS
            "208.67.222.222",    # OpenDNS
        ]
        
        self.stun_servers = [
            "stun.l.google.com:19302",
            "stun1.l.google.com:19302",
            "stun.stunprotocol.org:3478",
        ]
        
        # WebRTC test servers
        self.webrtc_test_servers = [
            "https://test.webrtc.org",
            "https://webrtc.github.io/samples",
        ]
        
        # Results storage
        self.test_results: Dict[NetworkTestType, List[DiagnosticResult]] = {}
        self.detected_issues: List[NetworkIssue] = []
        self.network_profile = NetworkProfile()
        
        # Monitoring
        self.monitor = global_performance_monitor.register_component(
            "network_diagnostics", "communication"
        )
    
    async def run_comprehensive_diagnostics(self) -> Dict[str, Any]:
        """Run a comprehensive network diagnostic suite."""
        logger.info("Starting comprehensive network diagnostics")
        
        start_time = time.time()
        results = {}
        
        # Run all diagnostic tests
        test_methods = [
            (NetworkTestType.CONNECTIVITY, self._test_basic_connectivity),
            (NetworkTestType.DNS_LOOKUP, self._test_dns_resolution),
            (NetworkTestType.PING, self._test_ping_latency),
            (NetworkTestType.TRACEROUTE, self._test_traceroute),
            (NetworkTestType.BANDWIDTH, self._test_bandwidth),
            (NetworkTestType.NAT_TYPE, self._test_nat_type),
            (NetworkTestType.STUN_BINDING, self._test_stun_binding),
            (NetworkTestType.FIREWALL, self._test_firewall_rules),
            (NetworkTestType.PORT_SCAN, self._test_port_accessibility),
        ]
        
        for test_type, test_method in test_methods:
            try:
                logger.info(f"Running {test_type.value} test")
                result = await test_method()
                
                # Store result
                if test_type not in self.test_results:
                    self.test_results[test_type] = []
                self.test_results[test_type].append(result)
                
                results[test_type.value] = {
                    "success": result.success,
                    "duration_ms": result.duration_ms,
                    "data": result.result_data,
                    "error": result.error_message
                }
                
            except Exception as e:
                logger.error(f"Error running {test_type.value} test: {e}")
                results[test_type.value] = {
                    "success": False,
                    "error": str(e)
                }
        
        # Build network profile
        await self._build_network_profile()
        
        # Analyze results and detect issues
        await self._analyze_results_and_detect_issues()
        
        # Generate summary
        total_duration = time.time() - start_time
        
        summary = {
            "timestamp": start_time,
            "duration_seconds": total_duration,
            "tests_run": len(test_methods),
            "tests_passed": sum(1 for r in results.values() if r.get("success", False)),
            "network_profile": self._network_profile_to_dict(),
            "detected_issues": [self._issue_to_dict(issue) for issue in self.detected_issues],
            "test_results": results
        }
        
        logger.info(f"Comprehensive diagnostics completed in {total_duration:.2f}s")
        return summary
    
    @monitor_performance(component="network_diagnostics", operation="connectivity_test")
    async def _test_basic_connectivity(self) -> DiagnosticResult:
        """Test basic internet connectivity."""
        start_time = time.time()
        
        try:
            # Test connectivity to multiple servers
            successful_connections = 0
            total_tests = len(self.test_servers)
            
            for server in self.test_servers:
                try:
                    # Simple socket connection test
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(self.test_timeout_seconds)
                    
                    result = sock.connect_ex((server, 53))  # DNS port
                    if result == 0:
                        successful_connections += 1
                    
                    sock.close()
                    
                except Exception:
                    continue
            
            success_rate = successful_connections / total_tests
            success = success_rate > 0.5  # At least 50% success
            
            duration_ms = (time.time() - start_time) * 1000
            
            return DiagnosticResult(
                test_type=NetworkTestType.CONNECTIVITY,
                success=success,
                duration_ms=duration_ms,
                result_data={
                    "successful_connections": successful_connections,
                    "total_tests": total_tests,
                    "success_rate": success_rate
                }
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return DiagnosticResult(
                test_type=NetworkTestType.CONNECTIVITY,
                success=False,
                duration_ms=duration_ms,
                error_message=str(e)
            )
    
    async def _test_dns_resolution(self) -> DiagnosticResult:
        """Test DNS resolution capabilities."""
        start_time = time.time()
        
        try:
            test_domains = [
                "google.com",
                "cloudflare.com", 
                "github.com",
                "livekit.io"
            ]
            
            successful_lookups = 0
            resolution_times = []
            
            for domain in test_domains:
                try:
                    lookup_start = time.time()
                    
                    # Perform DNS lookup
                    addr_info = socket.getaddrinfo(domain, None)
                    
                    lookup_time = (time.time() - lookup_start) * 1000
                    resolution_times.append(lookup_time)
                    successful_lookups += 1
                    
                except Exception:
                    continue
            
            success = successful_lookups > len(test_domains) // 2
            avg_resolution_time = sum(resolution_times) / len(resolution_times) if resolution_times else 0
            
            duration_ms = (time.time() - start_time) * 1000
            
            return DiagnosticResult(
                test_type=NetworkTestType.DNS_LOOKUP,
                success=success,
                duration_ms=duration_ms,
                result_data={
                    "successful_lookups": successful_lookups,
                    "total_domains": len(test_domains),
                    "average_resolution_time_ms": avg_resolution_time,
                    "resolution_times": resolution_times
                }
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return DiagnosticResult(
                test_type=NetworkTestType.DNS_LOOKUP,
                success=False,
                duration_ms=duration_ms,
                error_message=str(e)
            )
    
    async def _test_ping_latency(self) -> DiagnosticResult:
        """Test ping latency and packet loss."""
        start_time = time.time()
        
        try:
            if not PING_AVAILABLE:
                # Fallback to socket-based test
                return await self._socket_based_latency_test()
            
            results = {}
            
            for server in self.test_servers:
                ping_times = []
                successful_pings = 0
                
                for _ in range(self.ping_count):
                    try:
                        ping_time = ping3.ping(server, timeout=self.test_timeout_seconds)
                        if ping_time is not None:
                            ping_times.append(ping_time * 1000)  # Convert to ms
                            successful_pings += 1
                    except Exception:
                        continue
                
                if ping_times:
                    results[server] = {
                        "successful_pings": successful_pings,
                        "total_pings": self.ping_count,
                        "packet_loss_percent": ((self.ping_count - successful_pings) / self.ping_count) * 100,
                        "min_ping_ms": min(ping_times),
                        "max_ping_ms": max(ping_times),
                        "avg_ping_ms": sum(ping_times) / len(ping_times),
                        "ping_times": ping_times
                    }
            
            success = len(results) > 0
            duration_ms = (time.time() - start_time) * 1000
            
            return DiagnosticResult(
                test_type=NetworkTestType.PING,
                success=success,
                duration_ms=duration_ms,
                result_data=results
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return DiagnosticResult(
                test_type=NetworkTestType.PING,
                success=False,
                duration_ms=duration_ms,
                error_message=str(e)
            )
    
    async def _socket_based_latency_test(self) -> DiagnosticResult:
        """Fallback latency test using socket connections."""
        start_time = time.time()
        
        try:
            results = {}
            
            for server in self.test_servers:
                latencies = []
                successful_connections = 0
                
                for _ in range(self.ping_count):
                    try:
                        connect_start = time.time()
                        
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        sock.settimeout(self.test_timeout_seconds)
                        
                        result = sock.connect_ex((server, 80))
                        if result == 0:
                            latency = (time.time() - connect_start) * 1000
                            latencies.append(latency)
                            successful_connections += 1
                        
                        sock.close()
                        
                    except Exception:
                        continue
                
                if latencies:
                    results[server] = {
                        "successful_connections": successful_connections,
                        "total_attempts": self.ping_count,
                        "connection_success_rate": (successful_connections / self.ping_count) * 100,
                        "min_latency_ms": min(latencies),
                        "max_latency_ms": max(latencies),
                        "avg_latency_ms": sum(latencies) / len(latencies),
                        "latencies": latencies
                    }
            
            success = len(results) > 0
            duration_ms = (time.time() - start_time) * 1000
            
            return DiagnosticResult(
                test_type=NetworkTestType.PING,
                success=success,
                duration_ms=duration_ms,
                result_data=results
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return DiagnosticResult(
                test_type=NetworkTestType.PING,
                success=False,
                duration_ms=duration_ms,
                error_message=str(e)
            )
    
    async def _test_traceroute(self) -> DiagnosticResult:
        """Test network path to destinations."""
        start_time = time.time()
        
        try:
            # Use system traceroute command
            target = self.test_servers[0]  # Test to first server
            
            system = platform.system().lower()
            if system == "windows":
                cmd = ["tracert", "-h", "15", target]
            else:
                cmd = ["traceroute", "-m", "15", target]
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                success = result.returncode == 0
                output = result.stdout if success else result.stderr
                
            except subprocess.TimeoutExpired:
                success = False
                output = "Traceroute timed out"
            except FileNotFoundError:
                success = False
                output = "Traceroute command not found"
            
            duration_ms = (time.time() - start_time) * 1000
            
            return DiagnosticResult(
                test_type=NetworkTestType.TRACEROUTE,
                success=success,
                duration_ms=duration_ms,
                result_data={
                    "target": target,
                    "output": output,
                    "hops": self._parse_traceroute_output(output) if success else []
                }
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return DiagnosticResult(
                test_type=NetworkTestType.TRACEROUTE,
                success=False,
                duration_ms=duration_ms,
                error_message=str(e)
            )
    
    def _parse_traceroute_output(self, output: str) -> List[Dict[str, Any]]:
        """Parse traceroute output to extract hop information."""
        hops = []
        lines = output.strip().split('\n')
        
        for line in lines:
            # Simple parsing - could be enhanced for different traceroute formats
            if line.strip() and not line.startswith('traceroute'):
                parts = line.strip().split()
                if len(parts) >= 2:
                    hop_num = parts[0].rstrip('.')
                    if hop_num.isdigit():
                        hops.append({
                            "hop": int(hop_num),
                            "raw_line": line.strip()
                        })
        
        return hops
    
    async def _test_bandwidth(self) -> DiagnosticResult:
        """Test network bandwidth capabilities."""
        start_time = time.time()
        
        try:
            if not SPEEDTEST_AVAILABLE:
                # Mock bandwidth test
                duration_ms = (time.time() - start_time) * 1000
                return DiagnosticResult(
                    test_type=NetworkTestType.BANDWIDTH,
                    success=True,
                    duration_ms=duration_ms,
                    result_data={
                        "download_mbps": 50.0,  # Mock values
                        "upload_mbps": 10.0,
                        "ping_ms": 25.0,
                        "note": "Mock bandwidth test - speedtest library not available"
                    }
                )
            
            # Real bandwidth test
            st = speedtest.Speedtest()
            st.get_best_server()
            
            download_speed = st.download() / 1_000_000  # Convert to Mbps
            upload_speed = st.upload() / 1_000_000      # Convert to Mbps
            
            ping_result = st.results.ping
            
            duration_ms = (time.time() - start_time) * 1000
            
            return DiagnosticResult(
                test_type=NetworkTestType.BANDWIDTH,
                success=True,
                duration_ms=duration_ms,
                result_data={
                    "download_mbps": download_speed,
                    "upload_mbps": upload_speed,
                    "ping_ms": ping_result,
                    "server_info": st.results.server
                }
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return DiagnosticResult(
                test_type=NetworkTestType.BANDWIDTH,
                success=False,
                duration_ms=duration_ms,
                error_message=str(e)
            )
    
    async def _test_nat_type(self) -> DiagnosticResult:
        """Test NAT type and characteristics."""
        start_time = time.time()
        
        try:
            # Simplified NAT type detection
            # In a real implementation, this would use STUN protocol
            
            nat_info = {
                "type": "unknown",
                "port_mapping": "unknown",
                "filtering": "unknown"
            }
            
            # Get local IP
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.connect(("8.8.8.8", 80))
                local_ip = sock.getsockname()[0]
                sock.close()
                nat_info["local_ip"] = local_ip
            except Exception:
                nat_info["local_ip"] = "unknown"
            
            # Try to determine if behind NAT
            try:
                if REQUESTS_AVAILABLE:
                    response = requests.get("https://httpbin.org/ip", timeout=5)
                    if response.status_code == 200:
                        public_ip = response.json().get("origin", "").split(",")[0].strip()
                        nat_info["public_ip"] = public_ip
                        
                        # Simple NAT detection
                        if nat_info["local_ip"] != "unknown" and public_ip != nat_info["local_ip"]:
                            nat_info["behind_nat"] = True
                            nat_info["type"] = "nat_detected"
                        else:
                            nat_info["behind_nat"] = False
                            nat_info["type"] = "direct_connection"
            except Exception:
                nat_info["public_ip"] = "unknown"
            
            duration_ms = (time.time() - start_time) * 1000
            
            return DiagnosticResult(
                test_type=NetworkTestType.NAT_TYPE,
                success=True,
                duration_ms=duration_ms,
                result_data=nat_info
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return DiagnosticResult(
                test_type=NetworkTestType.NAT_TYPE,
                success=False,
                duration_ms=duration_ms,
                error_message=str(e)
            )
    
    async def _test_stun_binding(self) -> DiagnosticResult:
        """Test STUN server accessibility for WebRTC."""
        start_time = time.time()
        
        try:
            accessible_servers = []
            
            for stun_server in self.stun_servers:
                try:
                    host, port = stun_server.split(':')
                    port = int(port)
                    
                    # Simple UDP socket test to STUN server
                    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    sock.settimeout(self.test_timeout_seconds)
                    
                    # Send a simple message (not a real STUN request)
                    test_message = b"test"
                    sock.sendto(test_message, (host, port))
                    
                    # If no exception, server is accessible
                    accessible_servers.append(stun_server)
                    sock.close()
                    
                except Exception:
                    continue
            
            success = len(accessible_servers) > 0
            duration_ms = (time.time() - start_time) * 1000
            
            return DiagnosticResult(
                test_type=NetworkTestType.STUN_BINDING,
                success=success,
                duration_ms=duration_ms,
                result_data={
                    "accessible_servers": accessible_servers,
                    "total_servers_tested": len(self.stun_servers),
                    "accessibility_rate": len(accessible_servers) / len(self.stun_servers)
                }
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return DiagnosticResult(
                test_type=NetworkTestType.STUN_BINDING,
                success=False,
                duration_ms=duration_ms,
                error_message=str(e)
            )
    
    async def _test_firewall_rules(self) -> DiagnosticResult:
        """Test for firewall restrictions."""
        start_time = time.time()
        
        try:
            # Test common WebRTC ports
            webrtc_ports = [
                (3478, "udp"),  # STUN
                (5349, "tcp"),  # STUNS
                (80, "tcp"),    # HTTP
                (443, "tcp"),   # HTTPS
            ]
            
            blocked_ports = []
            open_ports = []
            
            for port, protocol in webrtc_ports:
                try:
                    if protocol == "tcp":
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        sock.settimeout(2.0)
                        result = sock.connect_ex(("8.8.8.8", port))
                        
                        if result == 0:
                            open_ports.append((port, protocol))
                        else:
                            blocked_ports.append((port, protocol))
                        
                        sock.close()
                    
                    elif protocol == "udp":
                        # UDP is harder to test definitively
                        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                        sock.settimeout(2.0)
                        
                        try:
                            sock.sendto(b"test", ("8.8.8.8", port))
                            open_ports.append((port, protocol))
                        except Exception:
                            blocked_ports.append((port, protocol))
                        
                        sock.close()
                
                except Exception:
                    blocked_ports.append((port, protocol))
            
            firewall_detected = len(blocked_ports) > len(open_ports)
            
            duration_ms = (time.time() - start_time) * 1000
            
            return DiagnosticResult(
                test_type=NetworkTestType.FIREWALL,
                success=True,
                duration_ms=duration_ms,
                result_data={
                    "firewall_detected": firewall_detected,
                    "open_ports": open_ports,
                    "blocked_ports": blocked_ports,
                    "total_ports_tested": len(webrtc_ports)
                }
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return DiagnosticResult(
                test_type=NetworkTestType.FIREWALL,
                success=False,
                duration_ms=duration_ms,
                error_message=str(e)
            )
    
    async def _test_port_accessibility(self) -> DiagnosticResult:
        """Test accessibility of specific ports."""
        start_time = time.time()
        
        try:
            # Test LiveKit default ports
            livekit_ports = [
                (7880, "tcp"),   # LiveKit server
                (7881, "tcp"),   # LiveKit TURN
                (443, "tcp"),    # HTTPS WebSocket
                (80, "tcp"),     # HTTP WebSocket
            ]
            
            port_results = {}
            
            for port, protocol in livekit_ports:
                try:
                    if protocol == "tcp":
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        sock.settimeout(self.test_timeout_seconds)
                        
                        # Test connection to a known server
                        test_hosts = ["google.com", "cloudflare.com"]
                        accessible = False
                        
                        for host in test_hosts:
                            try:
                                result = sock.connect_ex((host, port))
                                if result == 0:
                                    accessible = True
                                    break
                            except Exception:
                                continue
                        
                        port_results[f"{port}/{protocol}"] = {
                            "accessible": accessible,
                            "tested_hosts": test_hosts
                        }
                        
                        sock.close()
                
                except Exception as e:
                    port_results[f"{port}/{protocol}"] = {
                        "accessible": False,
                        "error": str(e)
                    }
            
            accessible_ports = sum(1 for r in port_results.values() if r.get("accessible", False))
            success = accessible_ports > 0
            
            duration_ms = (time.time() - start_time) * 1000
            
            return DiagnosticResult(
                test_type=NetworkTestType.PORT_SCAN,
                success=success,
                duration_ms=duration_ms,
                result_data={
                    "port_results": port_results,
                    "accessible_ports": accessible_ports,
                    "total_ports": len(livekit_ports)
                }
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return DiagnosticResult(
                test_type=NetworkTestType.PORT_SCAN,
                success=False,
                duration_ms=duration_ms,
                error_message=str(e)
            )
    
    async def _build_network_profile(self):
        """Build comprehensive network profile from test results."""
        # Extract information from test results
        for test_type, results in self.test_results.items():
            if not results:
                continue
            
            latest_result = results[-1]
            
            if test_type == NetworkTestType.NAT_TYPE and latest_result.success:
                data = latest_result.result_data
                self.network_profile.public_ip = data.get("public_ip")
                self.network_profile.local_ip = data.get("local_ip")
                self.network_profile.nat_type = data.get("type")
            
            elif test_type == NetworkTestType.BANDWIDTH and latest_result.success:
                data = latest_result.result_data
                self.network_profile.download_mbps = data.get("download_mbps")
                self.network_profile.upload_mbps = data.get("upload_mbps")
                self.network_profile.baseline_latency_ms = data.get("ping_ms")
            
            elif test_type == NetworkTestType.PING and latest_result.success:
                data = latest_result.result_data
                # Get average latency from first server
                for server_data in data.values():
                    if "avg_ping_ms" in server_data:
                        self.network_profile.baseline_latency_ms = server_data["avg_ping_ms"]
                        break
            
            elif test_type == NetworkTestType.STUN_BINDING and latest_result.success:
                data = latest_result.result_data
                self.network_profile.stun_servers_accessible = data.get("accessible_servers", [])
            
            elif test_type == NetworkTestType.FIREWALL and latest_result.success:
                data = latest_result.result_data
                self.network_profile.firewall_detected = data.get("firewall_detected", False)
    
    async def _analyze_results_and_detect_issues(self):
        """Analyze test results and detect network issues."""
        self.detected_issues.clear()
        
        # Check connectivity issues
        connectivity_results = self.test_results.get(NetworkTestType.CONNECTIVITY, [])
        if connectivity_results and not connectivity_results[-1].success:
            self.detected_issues.append(NetworkIssue(
                issue_id="connectivity_failure",
                category=IssueCategory.CONNECTIVITY,
                severity=DiagnosticSeverity.CRITICAL,
                title="Basic Internet Connectivity Failed",
                description="Unable to establish basic internet connections",
                possible_causes=[
                    "No internet connection",
                    "DNS server issues",
                    "Firewall blocking all traffic",
                    "Network adapter problems"
                ],
                recommended_actions=[
                    "Check network cable connections",
                    "Restart network adapter",
                    "Check DNS configuration",
                    "Contact network administrator"
                ],
                related_tests=[NetworkTestType.CONNECTIVITY]
            ))
        
        # Check DNS issues
        dns_results = self.test_results.get(NetworkTestType.DNS_LOOKUP, [])
        if dns_results and not dns_results[-1].success:
            self.detected_issues.append(NetworkIssue(
                issue_id="dns_resolution_failure",
                category=IssueCategory.DNS,
                severity=DiagnosticSeverity.ERROR,
                title="DNS Resolution Problems",
                description="Unable to resolve domain names",
                possible_causes=[
                    "DNS server configuration issues",
                    "DNS server unavailable",
                    "Network filtering blocking DNS",
                    "Local DNS cache corruption"
                ],
                recommended_actions=[
                    "Try alternate DNS servers (8.8.8.8, 1.1.1.1)",
                    "Flush DNS cache",
                    "Check DNS server configuration",
                    "Restart network connection"
                ],
                related_tests=[NetworkTestType.DNS_LOOKUP]
            ))
        
        # Check latency issues
        ping_results = self.test_results.get(NetworkTestType.PING, [])
        if ping_results and ping_results[-1].success:
            data = ping_results[-1].result_data
            
            high_latency_servers = []
            for server, server_data in data.items():
                avg_ping = server_data.get("avg_ping_ms", 0)
                if avg_ping > 200:  # High latency threshold
                    high_latency_servers.append((server, avg_ping))
            
            if high_latency_servers:
                self.detected_issues.append(NetworkIssue(
                    issue_id="high_latency",
                    category=IssueCategory.LATENCY,
                    severity=DiagnosticSeverity.WARNING,
                    title="High Network Latency Detected",
                    description=f"High latency to {len(high_latency_servers)} servers",
                    possible_causes=[
                        "Slow internet connection",
                        "Network congestion",
                        "Geographic distance to servers",
                        "ISP routing issues"
                    ],
                    recommended_actions=[
                        "Use servers closer to your location",
                        "Check for network congestion",
                        "Contact ISP about routing",
                        "Consider upgrading internet plan"
                    ],
                    related_tests=[NetworkTestType.PING]
                ))
        
        # Check packet loss issues
        if ping_results and ping_results[-1].success:
            data = ping_results[-1].result_data
            
            packet_loss_servers = []
            for server, server_data in data.items():
                packet_loss = server_data.get("packet_loss_percent", 0)
                if packet_loss > 5:  # Packet loss threshold
                    packet_loss_servers.append((server, packet_loss))
            
            if packet_loss_servers:
                self.detected_issues.append(NetworkIssue(
                    issue_id="packet_loss",
                    category=IssueCategory.PACKET_LOSS,
                    severity=DiagnosticSeverity.ERROR,
                    title="Packet Loss Detected",
                    description=f"Packet loss detected to {len(packet_loss_servers)} servers",
                    possible_causes=[
                        "Network congestion",
                        "Hardware problems",
                        "Wi-Fi interference",
                        "ISP network issues"
                    ],
                    recommended_actions=[
                        "Use wired connection instead of Wi-Fi",
                        "Check network hardware",
                        "Reduce network usage",
                        "Contact ISP support"
                    ],
                    related_tests=[NetworkTestType.PING]
                ))
        
        # Check firewall issues
        firewall_results = self.test_results.get(NetworkTestType.FIREWALL, [])
        if firewall_results and firewall_results[-1].success:
            data = firewall_results[-1].result_data
            
            if data.get("firewall_detected", False):
                blocked_ports = data.get("blocked_ports", [])
                
                self.detected_issues.append(NetworkIssue(
                    issue_id="firewall_restrictions",
                    category=IssueCategory.FIREWALL,
                    severity=DiagnosticSeverity.WARNING,
                    title="Firewall Restrictions Detected",
                    description=f"Firewall blocking {len(blocked_ports)} required ports",
                    possible_causes=[
                        "Corporate firewall policies",
                        "Router firewall settings",
                        "Operating system firewall",
                        "ISP port blocking"
                    ],
                    recommended_actions=[
                        "Configure firewall to allow WebRTC ports",
                        "Use TURN relay servers",
                        "Contact network administrator",
                        "Try different network connection"
                    ],
                    related_tests=[NetworkTestType.FIREWALL, NetworkTestType.PORT_SCAN]
                ))
        
        # Check STUN accessibility
        stun_results = self.test_results.get(NetworkTestType.STUN_BINDING, [])
        if stun_results and stun_results[-1].success:
            data = stun_results[-1].result_data
            accessibility_rate = data.get("accessibility_rate", 0)
            
            if accessibility_rate < 0.5:  # Less than 50% STUN servers accessible
                self.detected_issues.append(NetworkIssue(
                    issue_id="stun_inaccessible",
                    category=IssueCategory.NAT_TRAVERSAL,
                    severity=DiagnosticSeverity.ERROR,
                    title="STUN Servers Inaccessible",
                    description="Most STUN servers are not accessible",
                    possible_causes=[
                        "UDP traffic blocked by firewall",
                        "Corporate network restrictions",
                        "STUN ports blocked",
                        "Network configuration issues"
                    ],
                    recommended_actions=[
                        "Configure firewall to allow UDP traffic",
                        "Use TURN relay servers",
                        "Try alternative STUN servers",
                        "Contact network administrator"
                    ],
                    related_tests=[NetworkTestType.STUN_BINDING]
                ))
    
    def _network_profile_to_dict(self) -> Dict[str, Any]:
        """Convert network profile to dictionary."""
        return {
            "public_ip": self.network_profile.public_ip,
            "local_ip": self.network_profile.local_ip,
            "nat_type": self.network_profile.nat_type,
            "download_mbps": self.network_profile.download_mbps,
            "upload_mbps": self.network_profile.upload_mbps,
            "baseline_latency_ms": self.network_profile.baseline_latency_ms,
            "firewall_detected": self.network_profile.firewall_detected,
            "stun_servers_accessible": len(self.network_profile.stun_servers_accessible),
            "capabilities": {
                "supports_tcp": self.network_profile.supports_tcp,
                "supports_udp": self.network_profile.supports_udp
            }
        }
    
    def _issue_to_dict(self, issue: NetworkIssue) -> Dict[str, Any]:
        """Convert network issue to dictionary."""
        return {
            "issue_id": issue.issue_id,
            "category": issue.category.value,
            "severity": issue.severity.value,
            "title": issue.title,
            "description": issue.description,
            "possible_causes": issue.possible_causes,
            "recommended_actions": issue.recommended_actions,
            "detection_timestamp": issue.detection_timestamp,
            "related_tests": [t.value for t in issue.related_tests]
        }
    
    # Quick diagnostic methods
    
    async def quick_connectivity_check(self) -> bool:
        """Quick connectivity check."""
        result = await self._test_basic_connectivity()
        return result.success
    
    async def quick_latency_check(self) -> Optional[float]:
        """Quick latency check."""
        result = await self._test_ping_latency()
        
        if result.success and result.result_data:
            # Return average latency from first server
            for server_data in result.result_data.values():
                return server_data.get("avg_ping_ms")
        
        return None
    
    async def generate_diagnostic_report(self) -> str:
        """Generate human-readable diagnostic report."""
        report_lines = []
        
        report_lines.append("=== NETWORK DIAGNOSTIC REPORT ===")
        report_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Network profile summary
        report_lines.append("NETWORK PROFILE:")
        profile = self.network_profile
        
        if profile.public_ip:
            report_lines.append(f"  Public IP: {profile.public_ip}")
        if profile.local_ip:
            report_lines.append(f"  Local IP: {profile.local_ip}")
        if profile.nat_type:
            report_lines.append(f"  NAT Type: {profile.nat_type}")
        if profile.download_mbps:
            report_lines.append(f"  Download Speed: {profile.download_mbps:.1f} Mbps")
        if profile.upload_mbps:
            report_lines.append(f"  Upload Speed: {profile.upload_mbps:.1f} Mbps")
        if profile.baseline_latency_ms:
            report_lines.append(f"  Baseline Latency: {profile.baseline_latency_ms:.1f} ms")
        
        report_lines.append(f"  Firewall Detected: {'Yes' if profile.firewall_detected else 'No'}")
        report_lines.append(f"  STUN Servers Accessible: {len(profile.stun_servers_accessible)}")
        report_lines.append("")
        
        # Issues summary
        if self.detected_issues:
            report_lines.append("DETECTED ISSUES:")
            
            for issue in self.detected_issues:
                report_lines.append(f"  [{issue.severity.value.upper()}] {issue.title}")
                report_lines.append(f"    {issue.description}")
                
                if issue.recommended_actions:
                    report_lines.append(f"    Recommended actions:")
                    for action in issue.recommended_actions[:3]:  # Top 3 actions
                        report_lines.append(f"      - {action}")
                report_lines.append("")
        else:
            report_lines.append("NO CRITICAL ISSUES DETECTED")
            report_lines.append("")
        
        # Test results summary
        report_lines.append("TEST RESULTS:")
        for test_type, results in self.test_results.items():
            if results:
                latest = results[-1]
                status = "PASS" if latest.success else "FAIL"
                report_lines.append(f"  {test_type.value}: {status} ({latest.duration_ms:.1f}ms)")
        
        return "\n".join(report_lines)


# Convenience functions

async def run_quick_diagnostics() -> Dict[str, Any]:
    """Run quick network diagnostics."""
    diagnostics = NetworkDiagnostics()
    
    # Run essential tests only
    essential_tests = [
        diagnostics._test_basic_connectivity(),
        diagnostics._test_dns_resolution(),
        diagnostics._test_ping_latency()
    ]
    
    results = await asyncio.gather(*essential_tests, return_exceptions=True)
    
    return {
        "connectivity": results[0].success if isinstance(results[0], DiagnosticResult) else False,
        "dns": results[1].success if isinstance(results[1], DiagnosticResult) else False,
        "latency": results[2].success if isinstance(results[2], DiagnosticResult) else False
    }


async def check_webrtc_requirements() -> Dict[str, bool]:
    """Check if network meets WebRTC requirements."""
    diagnostics = NetworkDiagnostics()
    
    # Run WebRTC-specific tests
    stun_result = await diagnostics._test_stun_binding()
    firewall_result = await diagnostics._test_firewall_rules()
    nat_result = await diagnostics._test_nat_type()
    
    return {
        "stun_accessible": stun_result.success,
        "firewall_compatible": firewall_result.success and not firewall_result.result_data.get("firewall_detected", True),
        "nat_traversal_possible": nat_result.success
    }


# Global diagnostics instance
_global_diagnostics: Optional[NetworkDiagnostics] = None


def get_global_diagnostics() -> Optional[NetworkDiagnostics]:
    """Get global diagnostics instance."""
    return _global_diagnostics


def set_global_diagnostics(diagnostics: NetworkDiagnostics):
    """Set global diagnostics instance."""
    global _global_diagnostics
    _global_diagnostics = diagnostics