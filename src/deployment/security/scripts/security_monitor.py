#!/usr/bin/env python3
"""
Security Monitoring and Runtime Protection for Voice Agents Platform
Real-time security monitoring, threat detection, and alerting
"""

import os
import sys
import json
import yaml
import time
import logging
import asyncio
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import subprocess
import requests
import psutil
import docker
from collections import defaultdict, deque
import hashlib
import mmap

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ThreatLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class EventType(Enum):
    FILE_CHANGE = "file_change"
    NETWORK_ANOMALY = "network_anomaly"
    PROCESS_ANOMALY = "process_anomaly"
    CONTAINER_ANOMALY = "container_anomaly"
    AUTHENTICATION_FAILURE = "auth_failure"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    MALICIOUS_ACTIVITY = "malicious_activity"
    VULNERABILITY_EXPLOIT = "vulnerability_exploit"

@dataclass
class SecurityEvent:
    """Security event data structure"""
    event_id: str
    timestamp: str
    event_type: EventType
    threat_level: ThreatLevel
    source: str
    description: str
    details: Dict[str, Any]
    remediation_actions: List[str]
    affected_resources: List[str]

@dataclass
class ThreatIntelligence:
    """Threat intelligence data structure"""
    ioc_type: str  # IP, domain, hash, etc.
    indicator: str
    threat_level: ThreatLevel
    source: str
    description: str
    first_seen: str
    last_seen: str
    tags: List[str]

class SecurityMonitor:
    """Real-time security monitoring system"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or "/security/config.yaml"
        self.config = self._load_config()
        self.events_queue = deque(maxlen=10000)
        self.threat_intel = {}
        self.baseline_metrics = {}
        self.alert_cooldowns = defaultdict(float)
        self.running = False
        
        # Initialize monitoring components
        self.file_monitor = FileSystemMonitor(self)
        self.network_monitor = NetworkMonitor(self)
        self.process_monitor = ProcessMonitor(self)
        self.container_monitor = ContainerMonitor(self)
        
        # Load threat intelligence
        self._load_threat_intelligence()
        
    def _load_config(self) -> Dict:
        """Load security configuration"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config.get('security', {}).get('runtime_security', {})
        except Exception as e:
            logger.warning(f"Could not load config from {self.config_path}: {e}")
            return {'enabled': True}
    
    def _load_threat_intelligence(self):
        """Load threat intelligence feeds"""
        intel_file = "/security/threat_intel.json"
        try:
            if os.path.exists(intel_file):
                with open(intel_file, 'r') as f:
                    intel_data = json.load(f)
                
                for item in intel_data.get('indicators', []):
                    self.threat_intel[item['indicator']] = ThreatIntelligence(**item)
                    
                logger.info(f"Loaded {len(self.threat_intel)} threat intelligence indicators")
        except Exception as e:
            logger.warning(f"Could not load threat intelligence: {e}")
    
    def add_event(self, event: SecurityEvent):
        """Add security event to queue"""
        self.events_queue.append(event)
        
        # Send alert if threat level is high enough
        if event.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
            self._send_alert(event)
        
        # Log event
        logger.warning(f"Security Event: {event.event_type.value} - {event.description}")
    
    def _send_alert(self, event: SecurityEvent):
        """Send security alert"""
        alert_key = f"{event.event_type.value}_{event.source}"
        current_time = time.time()
        
        # Check cooldown to prevent alert spam
        if current_time - self.alert_cooldowns[alert_key] < 300:  # 5 minute cooldown
            return
        
        self.alert_cooldowns[alert_key] = current_time
        
        try:
            # Send to configured alert channels
            alerting_config = self.config.get('alerting', {})
            
            if alerting_config.get('slack', {}).get('enabled'):
                self._send_slack_alert(event)
            
            if alerting_config.get('email', {}).get('enabled'):
                self._send_email_alert(event)
            
            if alerting_config.get('pagerduty', {}).get('enabled'):
                self._send_pagerduty_alert(event)
                
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
    
    def _send_slack_alert(self, event: SecurityEvent):
        """Send Slack alert"""
        webhook_url = os.getenv('SLACK_WEBHOOK_URL')
        if not webhook_url:
            return
        
        color = "danger" if event.threat_level == ThreatLevel.CRITICAL else "warning"
        
        message = {
            "text": f"🚨 Security Alert: {event.event_type.value}",
            "attachments": [{
                "color": color,
                "title": event.description,
                "fields": [
                    {
                        "title": "Threat Level",
                        "value": event.threat_level.value.upper(),
                        "short": True
                    },
                    {
                        "title": "Source",
                        "value": event.source,
                        "short": True
                    },
                    {
                        "title": "Timestamp",
                        "value": event.timestamp,
                        "short": True
                    },
                    {
                        "title": "Affected Resources",
                        "value": ", ".join(event.affected_resources),
                        "short": False
                    }
                ]
            }]
        }
        
        response = requests.post(webhook_url, json=message, timeout=10)
        response.raise_for_status()
    
    def _send_email_alert(self, event: SecurityEvent):
        """Send email alert"""
        # Implementation would depend on email service
        logger.info(f"Email alert sent for event: {event.event_id}")
    
    def _send_pagerduty_alert(self, event: SecurityEvent):
        """Send PagerDuty alert"""
        # Implementation would depend on PagerDuty integration
        logger.info(f"PagerDuty alert sent for event: {event.event_id}")
    
    async def start_monitoring(self):
        """Start all monitoring components"""
        if not self.config.get('enabled', True):
            logger.info("Runtime security monitoring is disabled")
            return
        
        logger.info("Starting security monitoring...")
        self.running = True
        
        # Start monitoring tasks
        tasks = []
        
        if self.config.get('file_monitoring', {}).get('enabled', True):
            tasks.append(asyncio.create_task(self.file_monitor.start()))
        
        if self.config.get('network_monitoring', {}).get('enabled', True):
            tasks.append(asyncio.create_task(self.network_monitor.start()))
        
        if self.config.get('process_monitoring', {}).get('enabled', True):
            tasks.append(asyncio.create_task(self.process_monitor.start()))
        
        if self.config.get('container_monitoring', {}).get('enabled', True):
            tasks.append(asyncio.create_task(self.container_monitor.start()))
        
        # Start event processor
        tasks.append(asyncio.create_task(self._process_events()))
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            logger.info("Stopping security monitoring...")
            self.running = False
            for task in tasks:
                task.cancel()
    
    async def _process_events(self):
        """Process security events"""
        while self.running:
            try:
                # Process events in queue
                while self.events_queue:
                    event = self.events_queue.popleft()
                    await self._analyze_event(event)
                
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error processing events: {e}")
                await asyncio.sleep(5)
    
    async def _analyze_event(self, event: SecurityEvent):
        """Analyze security event for threats"""
        # Check against threat intelligence
        for resource in event.affected_resources:
            if resource in self.threat_intel:
                intel = self.threat_intel[resource]
                event.threat_level = max(event.threat_level, intel.threat_level)
                event.description += f" (Matches threat intel: {intel.description})"
        
        # Store event for analysis
        event_file = f"/security/events/{event.event_id}.json"
        os.makedirs(os.path.dirname(event_file), exist_ok=True)
        
        with open(event_file, 'w') as f:
            json.dump(asdict(event), f, indent=2, default=str)

class FileSystemMonitor:
    """File system monitoring for security events"""
    
    def __init__(self, security_monitor: SecurityMonitor):
        self.monitor = security_monitor
        self.watched_paths = [
            "/etc",
            "/usr/bin",
            "/usr/sbin",
            "/var/log",
            "/security",
            "/app"  # Application directory
        ]
        self.file_hashes = {}
        
    async def start(self):
        """Start file system monitoring"""
        logger.info("Starting file system monitoring...")
        
        # Initialize baseline
        await self._create_baseline()
        
        while self.monitor.running:
            try:
                await self._check_file_changes()
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"File monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _create_baseline(self):
        """Create baseline of critical files"""
        for path in self.watched_paths:
            if not os.path.exists(path):
                continue
                
            for root, dirs, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        file_hash = self._calculate_file_hash(file_path)
                        self.file_hashes[file_path] = file_hash
                    except Exception:
                        continue
        
        logger.info(f"Created baseline for {len(self.file_hashes)} files")
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file"""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception:
            return ""
    
    async def _check_file_changes(self):
        """Check for unauthorized file changes"""
        for file_path, baseline_hash in list(self.file_hashes.items()):
            if not os.path.exists(file_path):
                # File deleted
                event = SecurityEvent(
                    event_id=f"file_del_{int(time.time())}",
                    timestamp=datetime.utcnow().isoformat(),
                    event_type=EventType.FILE_CHANGE,
                    threat_level=ThreatLevel.HIGH,
                    source="file_monitor",
                    description=f"Critical file deleted: {file_path}",
                    details={"action": "delete", "path": file_path},
                    remediation_actions=["Investigate file deletion", "Check for malicious activity"],
                    affected_resources=[file_path]
                )
                self.monitor.add_event(event)
                del self.file_hashes[file_path]
                continue
            
            current_hash = self._calculate_file_hash(file_path)
            if current_hash and current_hash != baseline_hash:
                # File modified
                threat_level = ThreatLevel.HIGH if "/etc" in file_path or "/usr/bin" in file_path else ThreatLevel.MEDIUM
                
                event = SecurityEvent(
                    event_id=f"file_mod_{int(time.time())}",
                    timestamp=datetime.utcnow().isoformat(),
                    event_type=EventType.FILE_CHANGE,
                    threat_level=threat_level,
                    source="file_monitor",
                    description=f"Critical file modified: {file_path}",
                    details={"action": "modify", "path": file_path, "old_hash": baseline_hash, "new_hash": current_hash},
                    remediation_actions=["Review file changes", "Verify legitimacy"],
                    affected_resources=[file_path]
                )
                self.monitor.add_event(event)
                self.file_hashes[file_path] = current_hash

class NetworkMonitor:
    """Network monitoring for security events"""
    
    def __init__(self, security_monitor: SecurityMonitor):
        self.monitor = security_monitor
        self.baseline_connections = set()
        self.suspicious_ips = set()
    
    async def start(self):
        """Start network monitoring"""
        logger.info("Starting network monitoring...")
        
        await self._create_baseline()
        
        while self.monitor.running:
            try:
                await self._check_network_activity()
                await asyncio.sleep(10)  # Check every 10 seconds
            except Exception as e:
                logger.error(f"Network monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def _create_baseline(self):
        """Create baseline of normal network connections"""
        connections = psutil.net_connections(kind='inet')
        for conn in connections:
            if conn.raddr:
                self.baseline_connections.add(conn.raddr.ip)
        
        logger.info(f"Network baseline created with {len(self.baseline_connections)} known connections")
    
    async def _check_network_activity(self):
        """Check for suspicious network activity"""
        connections = psutil.net_connections(kind='inet')
        
        for conn in connections:
            if not conn.raddr:
                continue
            
            remote_ip = conn.raddr.ip
            
            # Check against threat intelligence
            if remote_ip in self.monitor.threat_intel:
                intel = self.monitor.threat_intel[remote_ip]
                event = SecurityEvent(
                    event_id=f"net_threat_{int(time.time())}",
                    timestamp=datetime.utcnow().isoformat(),
                    event_type=EventType.NETWORK_ANOMALY,
                    threat_level=intel.threat_level,
                    source="network_monitor",
                    description=f"Connection to known malicious IP: {remote_ip}",
                    details={"remote_ip": remote_ip, "local_port": conn.laddr.port, "remote_port": conn.raddr.port},
                    remediation_actions=["Block IP address", "Investigate connection"],
                    affected_resources=[remote_ip]
                )
                self.monitor.add_event(event)
            
            # Check for connections to new/unknown IPs
            elif remote_ip not in self.baseline_connections and not self._is_private_ip(remote_ip):
                event = SecurityEvent(
                    event_id=f"net_new_{int(time.time())}",
                    timestamp=datetime.utcnow().isoformat(),
                    event_type=EventType.NETWORK_ANOMALY,
                    threat_level=ThreatLevel.LOW,
                    source="network_monitor",
                    description=f"New outbound connection: {remote_ip}",
                    details={"remote_ip": remote_ip, "local_port": conn.laddr.port, "remote_port": conn.raddr.port},
                    remediation_actions=["Verify connection legitimacy"],
                    affected_resources=[remote_ip]
                )
                self.monitor.add_event(event)
                self.baseline_connections.add(remote_ip)
    
    def _is_private_ip(self, ip: str) -> bool:
        """Check if IP is in private range"""
        private_ranges = [
            "10.", "172.16.", "172.17.", "172.18.", "172.19.", "172.20.",
            "172.21.", "172.22.", "172.23.", "172.24.", "172.25.", "172.26.",
            "172.27.", "172.28.", "172.29.", "172.30.", "172.31.", "192.168."
        ]
        return any(ip.startswith(range_) for range_ in private_ranges) or ip.startswith("127.")

class ProcessMonitor:
    """Process monitoring for security events"""
    
    def __init__(self, security_monitor: SecurityMonitor):
        self.monitor = security_monitor
        self.baseline_processes = set()
        self.suspicious_commands = [
            "nc", "netcat", "ncat", "socat", "telnet", "ssh", "scp", "wget", "curl",
            "python -c", "perl -e", "ruby -e", "bash -i", "sh -i", "/dev/tcp"
        ]
    
    async def start(self):
        """Start process monitoring"""
        logger.info("Starting process monitoring...")
        
        await self._create_baseline()
        
        while self.monitor.running:
            try:
                await self._check_processes()
                await asyncio.sleep(5)  # Check every 5 seconds
            except Exception as e:
                logger.error(f"Process monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def _create_baseline(self):
        """Create baseline of normal processes"""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                proc_info = proc.info
                if proc_info['name']:
                    self.baseline_processes.add(proc_info['name'])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        logger.info(f"Process baseline created with {len(self.baseline_processes)} known processes")
    
    async def _check_processes(self):
        """Check for suspicious processes"""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'username']):
            try:
                proc_info = proc.info
                
                # Check for new processes
                if proc_info['name'] and proc_info['name'] not in self.baseline_processes:
                    self.baseline_processes.add(proc_info['name'])
                    
                    event = SecurityEvent(
                        event_id=f"proc_new_{proc_info['pid']}",
                        timestamp=datetime.utcnow().isoformat(),
                        event_type=EventType.PROCESS_ANOMALY,
                        threat_level=ThreatLevel.LOW,
                        source="process_monitor",
                        description=f"New process started: {proc_info['name']}",
                        details={"pid": proc_info['pid'], "name": proc_info['name'], "user": proc_info['username']},
                        remediation_actions=["Verify process legitimacy"],
                        affected_resources=[str(proc_info['pid'])]
                    )
                    self.monitor.add_event(event)
                
                # Check for suspicious commands
                if proc_info['cmdline']:
                    cmdline = ' '.join(proc_info['cmdline'])
                    for suspicious_cmd in self.suspicious_commands:
                        if suspicious_cmd in cmdline.lower():
                            event = SecurityEvent(
                                event_id=f"proc_susp_{proc_info['pid']}",
                                timestamp=datetime.utcnow().isoformat(),
                                event_type=EventType.MALICIOUS_ACTIVITY,
                                threat_level=ThreatLevel.HIGH,
                                source="process_monitor",
                                description=f"Suspicious command executed: {cmdline}",
                                details={"pid": proc_info['pid'], "cmdline": cmdline, "user": proc_info['username']},
                                remediation_actions=["Investigate process", "Consider terminating process"],
                                affected_resources=[str(proc_info['pid'])]
                            )
                            self.monitor.add_event(event)
                            break
                            
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

class ContainerMonitor:
    """Container monitoring for security events"""
    
    def __init__(self, security_monitor: SecurityMonitor):
        self.monitor = security_monitor
        self.docker_client = None
        self.baseline_containers = set()
        
        try:
            self.docker_client = docker.from_env()
        except Exception as e:
            logger.warning(f"Docker not available: {e}")
    
    async def start(self):
        """Start container monitoring"""
        if not self.docker_client:
            logger.info("Container monitoring disabled - Docker not available")
            return
            
        logger.info("Starting container monitoring...")
        
        await self._create_baseline()
        
        while self.monitor.running:
            try:
                await self._check_containers()
                await asyncio.sleep(15)  # Check every 15 seconds
            except Exception as e:
                logger.error(f"Container monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _create_baseline(self):
        """Create baseline of containers"""
        try:
            containers = self.docker_client.containers.list(all=True)
            for container in containers:
                self.baseline_containers.add(container.id)
            
            logger.info(f"Container baseline created with {len(self.baseline_containers)} containers")
        except Exception as e:
            logger.error(f"Failed to create container baseline: {e}")
    
    async def _check_containers(self):
        """Check for container security events"""
        try:
            containers = self.docker_client.containers.list(all=True)
            current_containers = set()
            
            for container in containers:
                current_containers.add(container.id)
                
                # Check for new containers
                if container.id not in self.baseline_containers:
                    self.baseline_containers.add(container.id)
                    
                    event = SecurityEvent(
                        event_id=f"container_new_{container.id[:12]}",
                        timestamp=datetime.utcnow().isoformat(),
                        event_type=EventType.CONTAINER_ANOMALY,
                        threat_level=ThreatLevel.MEDIUM,
                        source="container_monitor",
                        description=f"New container started: {container.name}",
                        details={"container_id": container.id, "name": container.name, "image": container.image.tags},
                        remediation_actions=["Verify container legitimacy", "Review container configuration"],
                        affected_resources=[container.id]
                    )
                    self.monitor.add_event(event)
                
                # Check for privileged containers
                if container.attrs.get('HostConfig', {}).get('Privileged', False):
                    event = SecurityEvent(
                        event_id=f"container_priv_{container.id[:12]}",
                        timestamp=datetime.utcnow().isoformat(),
                        event_type=EventType.PRIVILEGE_ESCALATION,
                        threat_level=ThreatLevel.HIGH,
                        source="container_monitor",
                        description=f"Privileged container detected: {container.name}",
                        details={"container_id": container.id, "name": container.name},
                        remediation_actions=["Review privileged access necessity", "Consider security implications"],
                        affected_resources=[container.id]
                    )
                    self.monitor.add_event(event)
                
                # Check for containers with host network
                if container.attrs.get('HostConfig', {}).get('NetworkMode') == 'host':
                    event = SecurityEvent(
                        event_id=f"container_hostnet_{container.id[:12]}",
                        timestamp=datetime.utcnow().isoformat(),
                        event_type=EventType.CONTAINER_ANOMALY,
                        threat_level=ThreatLevel.MEDIUM,
                        source="container_monitor",
                        description=f"Container using host network: {container.name}",
                        details={"container_id": container.id, "name": container.name},
                        remediation_actions=["Review host network usage"],
                        affected_resources=[container.id]
                    )
                    self.monitor.add_event(event)
            
            # Check for removed containers
            removed_containers = self.baseline_containers - current_containers
            for container_id in removed_containers:
                self.baseline_containers.remove(container_id)
                
                event = SecurityEvent(
                    event_id=f"container_removed_{container_id[:12]}",
                    timestamp=datetime.utcnow().isoformat(),
                    event_type=EventType.CONTAINER_ANOMALY,
                    threat_level=ThreatLevel.LOW,
                    source="container_monitor",
                    description=f"Container removed: {container_id}",
                    details={"container_id": container_id},
                    remediation_actions=["Verify container removal was authorized"],
                    affected_resources=[container_id]
                )
                self.monitor.add_event(event)
                
        except Exception as e:
            logger.error(f"Container check failed: {e}")

async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Security Monitoring System')
    parser.add_argument('--config', help='Path to security config file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    monitor = SecurityMonitor(args.config)
    
    try:
        await monitor.start_monitoring()
    except KeyboardInterrupt:
        logger.info("Security monitoring stopped by user")
    except Exception as e:
        logger.error(f"Security monitoring failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())