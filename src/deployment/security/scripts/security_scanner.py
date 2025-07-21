#!/usr/bin/env python3
"""
Security Scanner Orchestrator for Voice Agents Platform
Comprehensive security scanning and vulnerability management
"""

import os
import sys
import json
import subprocess
import logging
import argparse
import yaml
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import concurrent.futures
import tempfile
import requests
from dataclasses import dataclass, asdict
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ScanType(Enum):
    CONTAINER = "container"
    DEPENDENCY = "dependency"
    SAST = "sast"
    INFRASTRUCTURE = "infrastructure"
    RUNTIME = "runtime"
    COMPLIANCE = "compliance"

class SeverityLevel(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

@dataclass
class ScanResult:
    """Scan result data structure"""
    scan_type: str
    tool: str
    status: str
    vulnerabilities: List[Dict]
    errors: List[str]
    metadata: Dict
    duration: float
    timestamp: str

@dataclass
class VulnerabilityReport:
    """Comprehensive vulnerability report"""
    scan_id: str
    timestamp: str
    project_path: str
    total_scans: int
    successful_scans: int
    failed_scans: int
    total_vulnerabilities: int
    severity_counts: Dict[str, int]
    scan_results: List[ScanResult]
    recommendations: List[str]
    compliance_status: Dict[str, bool]

class SecurityScanner:
    """Main security scanner orchestrator"""
    
    def __init__(self, config_path: str = None, project_path: str = "."):
        self.config_path = config_path or "/security/config.yaml"
        self.project_path = project_path
        self.config = self._load_config()
        self.scan_id = f"scan_{int(time.time())}"
        self.report_dir = "/security/reports"
        self.scan_results = []
        
        # Ensure report directory exists
        os.makedirs(self.report_dir, exist_ok=True)
        
    def _load_config(self) -> Dict:
        """Load security configuration"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config.get('security', {})
        except Exception as e:
            logger.warning(f"Could not load config from {self.config_path}: {e}")
            return self._default_config()
    
    def _default_config(self) -> Dict:
        """Default configuration if config file not found"""
        return {
            'global': {
                'scan_frequency': 'daily',
                'severity_threshold': 'HIGH',
                'notifications_enabled': True
            },
            'container_scanning': {'enabled': True},
            'dependency_scanning': {'enabled': True},
            'sast': {'enabled': True},
            'infrastructure_scanning': {'enabled': True},
            'runtime_security': {'enabled': True},
            'compliance': {'enabled': True}
        }
    
    def _run_command(self, command: List[str], cwd: str = None, timeout: int = 1800) -> Tuple[int, str, str]:
        """Run shell command and return exit code, stdout, stderr"""
        try:
            result = subprocess.run(
                command,
                cwd=cwd or self.project_path,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out: {' '.join(command)}")
            return 1, "", "Command timed out"
        except Exception as e:
            logger.error(f"Error running command {' '.join(command)}: {e}")
            return 1, "", str(e)
    
    def scan_containers_trivy(self) -> ScanResult:
        """Scan containers with Trivy"""
        logger.info("Running Trivy container vulnerability scan")
        start_time = time.time()
        
        trivy_config = self.config.get('container_scanning', {}).get('trivy', {})
        if not trivy_config.get('enabled', True):
            return ScanResult(
                scan_type=ScanType.CONTAINER.value,
                tool="trivy",
                status="skipped",
                vulnerabilities=[],
                errors=[],
                metadata={'reason': 'Trivy scanning disabled'},
                duration=0,
                timestamp=datetime.utcnow().isoformat()
            )
        
        vulnerabilities = []
        errors = []
        
        # Scan Dockerfile and built images
        docker_files = []
        for root, dirs, files in os.walk(self.project_path):
            for file in files:
                if file.startswith('Dockerfile') or file == 'dockerfile':
                    docker_files.append(os.path.join(root, file))
        
        for dockerfile in docker_files:
            command = [
                'trivy', 'config',
                '--format', 'json',
                '--severity', 'HIGH,CRITICAL',
                dockerfile
            ]
            
            exit_code, stdout, stderr = self._run_command(command)
            
            if exit_code == 0 and stdout:
                try:
                    result = json.loads(stdout)
                    for finding in result.get('Results', []):
                        for vuln in finding.get('Vulnerabilities', []):
                            vulnerabilities.append({
                                'id': vuln.get('VulnerabilityID'),
                                'package': vuln.get('PkgName'),
                                'severity': vuln.get('Severity', '').lower(),
                                'title': vuln.get('Title'),
                                'description': vuln.get('Description'),
                                'fixed_version': vuln.get('FixedVersion'),
                                'file': dockerfile,
                                'source': 'trivy'
                            })
                except json.JSONDecodeError as e:
                    errors.append(f"Failed to parse Trivy output for {dockerfile}: {e}")
            else:
                errors.append(f"Trivy scan failed for {dockerfile}: {stderr}")
        
        # Scan running containers if Docker is available
        try:
            exit_code, stdout, stderr = self._run_command(['docker', 'ps', '--format', '{{.Names}}'])
            if exit_code == 0:
                container_names = stdout.strip().split('\n') if stdout.strip() else []
                for container in container_names:
                    if container:
                        command = ['trivy', 'image', '--format', 'json', container]
                        exit_code, stdout, stderr = self._run_command(command)
                        
                        if exit_code == 0 and stdout:
                            try:
                                result = json.loads(stdout)
                                for finding in result.get('Results', []):
                                    for vuln in finding.get('Vulnerabilities', []):
                                        vulnerabilities.append({
                                            'id': vuln.get('VulnerabilityID'),
                                            'package': vuln.get('PkgName'),
                                            'severity': vuln.get('Severity', '').lower(),
                                            'title': vuln.get('Title'),
                                            'description': vuln.get('Description'),
                                            'fixed_version': vuln.get('FixedVersion'),
                                            'container': container,
                                            'source': 'trivy'
                                        })
                            except json.JSONDecodeError as e:
                                errors.append(f"Failed to parse Trivy output for container {container}: {e}")
        except Exception as e:
            logger.warning(f"Could not scan containers: {e}")
        
        duration = time.time() - start_time
        
        return ScanResult(
            scan_type=ScanType.CONTAINER.value,
            tool="trivy",
            status="completed" if not errors else "completed_with_errors",
            vulnerabilities=vulnerabilities,
            errors=errors,
            metadata={
                'scanned_files': len(docker_files),
                'total_vulnerabilities': len(vulnerabilities)
            },
            duration=duration,
            timestamp=datetime.utcnow().isoformat()
        )
    
    def scan_dependencies(self) -> ScanResult:
        """Scan dependencies using the existing dependency scanner"""
        logger.info("Running dependency security scan")
        start_time = time.time()
        
        dep_config = self.config.get('dependency_scanning', {})
        if not dep_config.get('enabled', True):
            return ScanResult(
                scan_type=ScanType.DEPENDENCY.value,
                tool="dependency-scanner",
                status="skipped",
                vulnerabilities=[],
                errors=[],
                metadata={'reason': 'Dependency scanning disabled'},
                duration=0,
                timestamp=datetime.utcnow().isoformat()
            )
        
        # Run the existing dependency scanner
        scanner_path = os.path.join(os.path.dirname(__file__), 'dependency-scan.py')
        output_file = os.path.join(self.report_dir, f"{self.scan_id}_dependency_scan.json")
        
        command = [
            'python3', scanner_path,
            '--project-path', self.project_path,
            '--config', self.config_path,
            '--output', output_file
        ]
        
        exit_code, stdout, stderr = self._run_command(command)
        
        vulnerabilities = []
        errors = []
        
        if exit_code == 0 and os.path.exists(output_file):
            try:
                with open(output_file, 'r') as f:
                    result = json.load(f)
                
                # Extract vulnerabilities from Python scans
                for tool, tool_result in result.get('python', {}).items():
                    if isinstance(tool_result, dict) and 'vulnerabilities' in tool_result:
                        vulnerabilities.extend(tool_result['vulnerabilities'])
                
                # Extract vulnerabilities from Node.js scans
                for tool, tool_result in result.get('nodejs', {}).items():
                    if isinstance(tool_result, dict) and 'vulnerabilities' in tool_result:
                        vulnerabilities.extend(tool_result['vulnerabilities'])
                        
            except Exception as e:
                errors.append(f"Failed to parse dependency scan results: {e}")
        else:
            errors.append(f"Dependency scan failed: {stderr}")
        
        duration = time.time() - start_time
        
        return ScanResult(
            scan_type=ScanType.DEPENDENCY.value,
            tool="dependency-scanner",
            status="completed" if not errors else "failed",
            vulnerabilities=vulnerabilities,
            errors=errors,
            metadata={
                'output_file': output_file,
                'total_vulnerabilities': len(vulnerabilities)
            },
            duration=duration,
            timestamp=datetime.utcnow().isoformat()
        )
    
    def scan_sast_semgrep(self) -> ScanResult:
        """Static Application Security Testing with Semgrep"""
        logger.info("Running Semgrep SAST scan")
        start_time = time.time()
        
        sast_config = self.config.get('sast', {}).get('semgrep', {})
        if not sast_config.get('enabled', True):
            return ScanResult(
                scan_type=ScanType.SAST.value,
                tool="semgrep",
                status="skipped",
                vulnerabilities=[],
                errors=[],
                metadata={'reason': 'Semgrep scanning disabled'},
                duration=0,
                timestamp=datetime.utcnow().isoformat()
            )
        
        output_file = os.path.join(self.report_dir, f"{self.scan_id}_sast_semgrep.json")
        
        command = [
            'semgrep',
            '--config=auto',
            '--json',
            '--output', output_file,
            self.project_path
        ]
        
        # Add custom rules if configured
        rules = sast_config.get('rules', ['auto'])
        if rules != ['auto']:
            command = ['semgrep', '--config=' + ','.join(rules), '--json', '--output', output_file, self.project_path]
        
        exit_code, stdout, stderr = self._run_command(command)
        
        vulnerabilities = []
        errors = []
        
        if exit_code == 0 and os.path.exists(output_file):
            try:
                with open(output_file, 'r') as f:
                    result = json.load(f)
                
                for finding in result.get('results', []):
                    severity = finding.get('extra', {}).get('severity', 'INFO').lower()
                    vulnerabilities.append({
                        'id': finding.get('check_id'),
                        'file': finding.get('path'),
                        'line': finding.get('start', {}).get('line'),
                        'column': finding.get('start', {}).get('col'),
                        'severity': severity,
                        'message': finding.get('extra', {}).get('message'),
                        'description': finding.get('extra', {}).get('shortlink'),
                        'code': finding.get('extra', {}).get('lines'),
                        'source': 'semgrep'
                    })
                    
            except Exception as e:
                errors.append(f"Failed to parse Semgrep output: {e}")
        else:
            errors.append(f"Semgrep scan failed: {stderr}")
        
        duration = time.time() - start_time
        
        return ScanResult(
            scan_type=ScanType.SAST.value,
            tool="semgrep",
            status="completed" if not errors else "failed",
            vulnerabilities=vulnerabilities,
            errors=errors,
            metadata={
                'output_file': output_file,
                'total_vulnerabilities': len(vulnerabilities)
            },
            duration=duration,
            timestamp=datetime.utcnow().isoformat()
        )
    
    def scan_infrastructure_kube_score(self) -> ScanResult:
        """Infrastructure security scanning with kube-score"""
        logger.info("Running kube-score infrastructure scan")
        start_time = time.time()
        
        infra_config = self.config.get('infrastructure_scanning', {})
        if not infra_config.get('enabled', True):
            return ScanResult(
                scan_type=ScanType.INFRASTRUCTURE.value,
                tool="kube-score",
                status="skipped",
                vulnerabilities=[],
                errors=[],
                metadata={'reason': 'Infrastructure scanning disabled'},
                duration=0,
                timestamp=datetime.utcnow().isoformat()
            )
        
        # Find Kubernetes YAML files
        k8s_files = []
        for root, dirs, files in os.walk(self.project_path):
            for file in files:
                if file.endswith(('.yaml', '.yml')) and ('kubernetes' in root or 'k8s' in root):
                    k8s_files.append(os.path.join(root, file))
        
        vulnerabilities = []
        errors = []
        
        for k8s_file in k8s_files:
            command = ['kube-score', 'score', '--output-format', 'json', k8s_file]
            exit_code, stdout, stderr = self._run_command(command)
            
            if exit_code == 0 and stdout:
                try:
                    # kube-score output is not standard JSON, parse line by line
                    for line in stdout.strip().split('\n'):
                        if line.strip() and line.startswith('{'):
                            result = json.loads(line)
                            if result.get('score', 10) < 7:  # Score below 7 is considered a finding
                                vulnerabilities.append({
                                    'id': f"kube-score-{result.get('check', 'unknown')}",
                                    'file': k8s_file,
                                    'severity': 'medium' if result.get('score', 10) < 5 else 'low',
                                    'description': result.get('comment', 'Security configuration issue'),
                                    'check': result.get('check'),
                                    'score': result.get('score'),
                                    'source': 'kube-score'
                                })
                except Exception as e:
                    errors.append(f"Failed to parse kube-score output for {k8s_file}: {e}")
            else:
                logger.warning(f"kube-score scan failed for {k8s_file}: {stderr}")
        
        duration = time.time() - start_time
        
        return ScanResult(
            scan_type=ScanType.INFRASTRUCTURE.value,
            tool="kube-score",
            status="completed" if not errors else "completed_with_errors",
            vulnerabilities=vulnerabilities,
            errors=errors,
            metadata={
                'scanned_files': len(k8s_files),
                'total_vulnerabilities': len(vulnerabilities)
            },
            duration=duration,
            timestamp=datetime.utcnow().isoformat()
        )
    
    def scan_compliance_docker_bench(self) -> ScanResult:
        """Compliance scanning with Docker Bench Security"""
        logger.info("Running Docker Bench Security compliance scan")
        start_time = time.time()
        
        compliance_config = self.config.get('compliance', {})
        if not compliance_config.get('enabled', True):
            return ScanResult(
                scan_type=ScanType.COMPLIANCE.value,
                tool="docker-bench",
                status="skipped",
                vulnerabilities=[],
                errors=[],
                metadata={'reason': 'Compliance scanning disabled'},
                duration=0,
                timestamp=datetime.utcnow().isoformat()
            )
        
        # Run Docker Bench Security in container
        output_file = os.path.join(self.report_dir, f"{self.scan_id}_docker_bench.json")
        
        command = [
            'docker', 'run', '--rm',
            '--net', 'host',
            '--pid', 'host',
            '--userns', 'host',
            '--cap-add', 'audit_control',
            '-e', 'DOCKER_CONTENT_TRUST=$DOCKER_CONTENT_TRUST',
            '-v', '/etc:/etc:ro',
            '-v', '/usr/bin/containerd:/usr/bin/containerd:ro',
            '-v', '/usr/bin/runc:/usr/bin/runc:ro',
            '-v', '/usr/lib/systemd:/usr/lib/systemd:ro',
            '-v', '/var/lib:/var/lib:ro',
            '-v', '/var/run/docker.sock:/var/run/docker.sock:ro',
            '--label', 'docker_bench_security',
            'docker/docker-bench-security'
        ]
        
        exit_code, stdout, stderr = self._run_command(command)
        
        vulnerabilities = []
        errors = []
        
        if exit_code == 0:
            # Parse Docker Bench output (it's in text format, not JSON)
            lines = stdout.split('\n')
            for line in lines:
                if '[WARN]' in line or '[NOTE]' in line:
                    vulnerabilities.append({
                        'id': f"docker-bench-{len(vulnerabilities)}",
                        'severity': 'medium' if '[WARN]' in line else 'low',
                        'description': line.strip(),
                        'category': 'docker-security',
                        'source': 'docker-bench'
                    })
        else:
            errors.append(f"Docker Bench scan failed: {stderr}")
        
        duration = time.time() - start_time
        
        return ScanResult(
            scan_type=ScanType.COMPLIANCE.value,
            tool="docker-bench",
            status="completed" if not errors else "failed",
            vulnerabilities=vulnerabilities,
            errors=errors,
            metadata={
                'total_checks': len(vulnerabilities),
                'output_file': output_file
            },
            duration=duration,
            timestamp=datetime.utcnow().isoformat()
        )
    
    def run_all_scans(self, scan_types: List[str] = None) -> VulnerabilityReport:
        """Run all enabled security scans"""
        logger.info(f"Starting comprehensive security scan with ID: {self.scan_id}")
        
        if scan_types is None:
            scan_types = ['container', 'dependency', 'sast', 'infrastructure', 'compliance']
        
        scan_functions = {
            'container': self.scan_containers_trivy,
            'dependency': self.scan_dependencies,
            'sast': self.scan_sast_semgrep,
            'infrastructure': self.scan_infrastructure_kube_score,
            'compliance': self.scan_compliance_docker_bench
        }
        
        all_results = []
        all_vulnerabilities = []
        severity_counts = {
            'critical': 0,
            'high': 0,
            'medium': 0,
            'low': 0,
            'info': 0
        }
        
        successful_scans = 0
        failed_scans = 0
        
        # Run scans in parallel for better performance
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_scan = {}
            
            for scan_type in scan_types:
                if scan_type in scan_functions:
                    future = executor.submit(scan_functions[scan_type])
                    future_to_scan[future] = scan_type
            
            for future in concurrent.futures.as_completed(future_to_scan):
                scan_type = future_to_scan[future]
                try:
                    result = future.result()
                    all_results.append(result)
                    
                    if result.status == "completed":
                        successful_scans += 1
                    elif result.status == "failed":
                        failed_scans += 1
                    else:
                        successful_scans += 1  # completed_with_errors, skipped
                    
                    # Count vulnerabilities by severity
                    for vuln in result.vulnerabilities:
                        severity = vuln.get('severity', 'info').lower()
                        if severity in severity_counts:
                            severity_counts[severity] += 1
                        all_vulnerabilities.extend(result.vulnerabilities)
                        
                    logger.info(f"Completed {scan_type} scan: {len(result.vulnerabilities)} vulnerabilities found")
                    
                except Exception as e:
                    logger.error(f"Scan {scan_type} failed with exception: {e}")
                    failed_scans += 1
                    all_results.append(ScanResult(
                        scan_type=scan_type,
                        tool="unknown",
                        status="failed",
                        vulnerabilities=[],
                        errors=[str(e)],
                        metadata={},
                        duration=0,
                        timestamp=datetime.utcnow().isoformat()
                    ))
        
        # Generate recommendations
        recommendations = self._generate_recommendations(all_vulnerabilities, severity_counts)
        
        # Check compliance status
        compliance_status = self._check_compliance_status(severity_counts)
        
        # Create comprehensive report
        report = VulnerabilityReport(
            scan_id=self.scan_id,
            timestamp=datetime.utcnow().isoformat(),
            project_path=self.project_path,
            total_scans=len(scan_types),
            successful_scans=successful_scans,
            failed_scans=failed_scans,
            total_vulnerabilities=len(all_vulnerabilities),
            severity_counts=severity_counts,
            scan_results=all_results,
            recommendations=recommendations,
            compliance_status=compliance_status
        )
        
        # Save report
        self._save_report(report)
        
        # Send notifications if enabled
        if self.config.get('global', {}).get('notifications_enabled', True):
            self._send_notifications(report)
        
        logger.info(f"Security scan completed. Found {len(all_vulnerabilities)} total vulnerabilities.")
        return report
    
    def _generate_recommendations(self, vulnerabilities: List[Dict], severity_counts: Dict[str, int]) -> List[str]:
        """Generate security recommendations based on scan results"""
        recommendations = []
        
        if severity_counts['critical'] > 0:
            recommendations.append("URGENT: Address critical vulnerabilities immediately")
            recommendations.append("Consider temporarily isolating affected systems")
        
        if severity_counts['high'] > 5:
            recommendations.append("Prioritize fixing high-severity vulnerabilities")
            recommendations.append("Review security update processes")
        
        if severity_counts['medium'] > 20:
            recommendations.append("Schedule regular vulnerability remediation cycles")
            recommendations.append("Implement automated dependency updates where possible")
        
        # Specific recommendations based on vulnerability types
        container_vulns = [v for v in vulnerabilities if v.get('source') == 'trivy']
        if len(container_vulns) > 10:
            recommendations.append("Consider using distroless or minimal base images")
            recommendations.append("Implement container image scanning in CI/CD pipeline")
        
        dep_vulns = [v for v in vulnerabilities if v.get('source') in ['safety', 'npm-audit', 'pip-audit']]
        if len(dep_vulns) > 5:
            recommendations.append("Enable automated dependency vulnerability monitoring")
            recommendations.append("Implement dependency lock file management")
        
        sast_vulns = [v for v in vulnerabilities if v.get('source') == 'semgrep']
        if len(sast_vulns) > 0:
            recommendations.append("Provide security awareness training for developers")
            recommendations.append("Implement secure coding standards and reviews")
        
        return recommendations
    
    def _check_compliance_status(self, severity_counts: Dict[str, int]) -> Dict[str, bool]:
        """Check compliance status against security frameworks"""
        compliance_status = {}
        
        # Example compliance checks
        compliance_status['zero_critical'] = severity_counts['critical'] == 0
        compliance_status['low_high_severity'] = severity_counts['high'] < 5
        compliance_status['acceptable_medium'] = severity_counts['medium'] < 20
        compliance_status['docker_cis'] = True  # Would be based on Docker Bench results
        compliance_status['overall_passing'] = all([
            compliance_status['zero_critical'],
            compliance_status['low_high_severity']
        ])
        
        return compliance_status
    
    def _save_report(self, report: VulnerabilityReport):
        """Save vulnerability report to file"""
        report_file = os.path.join(self.report_dir, f"{self.scan_id}_security_report.json")
        
        # Convert dataclass to dict for JSON serialization
        report_dict = asdict(report)
        
        with open(report_file, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        logger.info(f"Security report saved to {report_file}")
    
    def _send_notifications(self, report: VulnerabilityReport):
        """Send notifications based on scan results"""
        alerting_config = self.config.get('alerting', {})
        if not alerting_config.get('enabled', True):
            return
        
        # Determine alert severity
        severity_mapping = alerting_config.get('severity_mapping', {})
        channels_to_notify = set()
        
        if report.severity_counts['critical'] > 0:
            channels_to_notify.update(severity_mapping.get('CRITICAL', []))
        if report.severity_counts['high'] > 0:
            channels_to_notify.update(severity_mapping.get('HIGH', []))
        if report.severity_counts['medium'] > 0:
            channels_to_notify.update(severity_mapping.get('MEDIUM', []))
        
        # Send notifications
        for channel in channels_to_notify:
            try:
                if channel == 'slack':
                    self._send_slack_notification(report)
                elif channel == 'email':
                    self._send_email_notification(report)
                elif channel == 'pagerduty':
                    self._send_pagerduty_notification(report)
            except Exception as e:
                logger.error(f"Failed to send {channel} notification: {e}")
    
    def _send_slack_notification(self, report: VulnerabilityReport):
        """Send Slack notification"""
        webhook_url = os.getenv('SLACK_WEBHOOK_URL')
        if not webhook_url:
            logger.warning("SLACK_WEBHOOK_URL not configured")
            return
        
        color = "danger" if report.severity_counts['critical'] > 0 else "warning"
        
        message = {
            "text": f"Security Scan Results - {report.scan_id}",
            "attachments": [{
                "color": color,
                "fields": [
                    {
                        "title": "Total Vulnerabilities",
                        "value": str(report.total_vulnerabilities),
                        "short": True
                    },
                    {
                        "title": "Critical",
                        "value": str(report.severity_counts['critical']),
                        "short": True
                    },
                    {
                        "title": "High",
                        "value": str(report.severity_counts['high']),
                        "short": True
                    },
                    {
                        "title": "Medium",
                        "value": str(report.severity_counts['medium']),
                        "short": True
                    }
                ]
            }]
        }
        
        response = requests.post(webhook_url, json=message)
        response.raise_for_status()
        logger.info("Slack notification sent successfully")
    
    def _send_email_notification(self, report: VulnerabilityReport):
        """Send email notification"""
        # Implementation would depend on email service configuration
        logger.info("Email notification would be sent here")
    
    def _send_pagerduty_notification(self, report: VulnerabilityReport):
        """Send PagerDuty notification"""
        # Implementation would depend on PagerDuty integration
        logger.info("PagerDuty notification would be sent here")

def main():
    parser = argparse.ArgumentParser(description='Security Scanner Orchestrator')
    parser.add_argument('--project-path', default='.', help='Path to project to scan')
    parser.add_argument('--config', help='Path to security config file')
    parser.add_argument('--scan-types', nargs='+', 
                       choices=['container', 'dependency', 'sast', 'infrastructure', 'compliance'],
                       help='Types of scans to run')
    parser.add_argument('--output-dir', default='/security/reports', help='Output directory for reports')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    scanner = SecurityScanner(args.config, args.project_path)
    scanner.report_dir = args.output_dir
    
    report = scanner.run_all_scans(args.scan_types)
    
    # Print summary
    print(f"\nSecurity Scan Complete - {report.scan_id}")
    print(f"Total vulnerabilities found: {report.total_vulnerabilities}")
    print(f"  Critical: {report.severity_counts['critical']}")
    print(f"  High: {report.severity_counts['high']}")
    print(f"  Medium: {report.severity_counts['medium']}")
    print(f"  Low: {report.severity_counts['low']}")
    print(f"  Info: {report.severity_counts['info']}")
    print(f"Compliance status: {'PASS' if report.compliance_status['overall_passing'] else 'FAIL'}")
    
    # Exit with error code if critical or high severity vulnerabilities found
    threshold = scanner.config.get('global', {}).get('severity_threshold', 'HIGH')
    if threshold == 'CRITICAL' and report.severity_counts['critical'] > 0:
        sys.exit(1)
    elif threshold == 'HIGH' and (report.severity_counts['critical'] > 0 or report.severity_counts['high'] > 0):
        sys.exit(1)
    elif threshold == 'MEDIUM' and sum([report.severity_counts['critical'], report.severity_counts['high'], report.severity_counts['medium']]) > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()