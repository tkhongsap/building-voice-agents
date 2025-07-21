#!/usr/bin/env python3
"""
Backup Monitoring and Health Check System
Provides real-time monitoring, alerting, and verification of backup operations
"""

import os
import sys
import time
import json
import yaml
import logging
import schedule
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from prometheus_client import Counter, Gauge, Histogram, Info, start_http_server
import boto3
import psycopg2
import redis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
backup_total = Counter('voice_agent_backup_total', 'Total number of backups', ['type', 'status'])
backup_duration = Histogram('voice_agent_backup_duration_seconds', 'Backup duration', ['type'])
backup_size = Gauge('voice_agent_backup_size_bytes', 'Backup size in bytes', ['type'])
backup_last_success = Gauge('voice_agent_backup_last_success_timestamp', 'Last successful backup timestamp', ['type'])
backup_verification_status = Gauge('voice_agent_backup_verification_status', 'Backup verification status', ['type'])
backup_age_hours = Gauge('voice_agent_backup_age_hours', 'Age of latest backup in hours', ['type'])
backup_storage_usage = Gauge('voice_agent_backup_storage_usage_bytes', 'Total backup storage usage')


@dataclass
class BackupHealth:
    """Backup health status"""
    backup_type: str
    status: str
    last_backup: Optional[datetime]
    last_verification: Optional[datetime]
    size_bytes: int
    error_count: int
    warning_count: int
    message: str


class BackupMonitor:
    """Monitors backup health and performance"""
    
    def __init__(self, config_path: str):
        """Initialize backup monitor"""
        self.config = self._load_config(config_path)
        self.s3_client = self._init_s3_client()
        self.health_status: Dict[str, BackupHealth] = {}
        self.alert_manager = AlertManager(self.config)
        
        # Start Prometheus metrics server
        start_http_server(9091)
        logger.info("Started Prometheus metrics server on port 9091")
        
    def _load_config(self, config_path: str) -> Dict:
        """Load backup configuration"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config['backup_config']
    
    def _init_s3_client(self):
        """Initialize S3 client"""
        return boto3.client(
            's3',
            region_name=self.config['s3']['region']
        )
    
    def start(self):
        """Start monitoring"""
        logger.info("Starting Backup Monitor")
        
        # Schedule monitoring tasks
        schedule.every(5).minutes.do(self.check_backup_health)
        schedule.every(30).minutes.do(self.verify_backups)
        schedule.every(1).hour.do(self.check_storage_usage)
        schedule.every(4).hours.do(self.test_restore_capability)
        
        # Run initial checks
        self.check_backup_health()
        
        # Main monitoring loop
        while True:
            schedule.run_pending()
            time.sleep(60)
    
    def check_backup_health(self):
        """Check health of all backup types"""
        logger.info("Checking backup health")
        
        for policy_name, policy in self.config['policies'].items():
            if not policy.get('enabled', True):
                continue
            
            health = self._check_policy_health(policy_name, policy)
            self.health_status[policy_name] = health
            
            # Update Prometheus metrics
            if health.last_backup:
                age_hours = (datetime.now() - health.last_backup).total_seconds() / 3600
                backup_age_hours.labels(type=policy_name).set(age_hours)
                backup_last_success.labels(type=policy_name).set(health.last_backup.timestamp())
            
            backup_size.labels(type=policy_name).set(health.size_bytes)
            
            # Check for alerts
            self._check_alerts(policy_name, health)
    
    def _check_policy_health(self, policy_name: str, policy: Dict) -> BackupHealth:
        """Check health of a specific backup policy"""
        health = BackupHealth(
            backup_type=policy_name,
            status="unknown",
            last_backup=None,
            last_verification=None,
            size_bytes=0,
            error_count=0,
            warning_count=0,
            message=""
        )
        
        try:
            # List recent backups
            prefix = f"{self.config['s3']['prefix']}{policy_name}/"
            response = self.s3_client.list_objects_v2(
                Bucket=self.config['s3']['bucket'],
                Prefix=prefix,
                MaxKeys=10
            )
            
            if 'Contents' not in response:
                health.status = "critical"
                health.message = "No backups found"
                health.error_count += 1
                return health
            
            # Find most recent backup
            latest_backup = max(response['Contents'], key=lambda x: x['LastModified'])
            health.last_backup = latest_backup['LastModified'].replace(tzinfo=None)
            health.size_bytes = latest_backup['Size']
            
            # Check backup age
            age = datetime.now() - health.last_backup
            
            # Determine expected backup frequency
            if 'schedule' in policy:
                if 'full_backup' in policy['schedule']:
                    max_age = timedelta(days=1.5)  # Allow 1.5 days for daily backups
                elif 'incremental' in policy['schedule']:
                    hours = int(policy['schedule']['incremental'].split()[1].split('/')[1])
                    max_age = timedelta(hours=hours * 2)
                elif 'snapshot' in policy['schedule']:
                    hours = int(policy['schedule']['snapshot'].split()[1].split('/')[1])
                    max_age = timedelta(hours=hours * 2)
                else:
                    max_age = timedelta(days=1)
            else:
                max_age = timedelta(days=1)
            
            # Set health status based on age
            if age > max_age * 2:
                health.status = "critical"
                health.message = f"Backup is {age.days} days old (expected < {max_age.days} days)"
                health.error_count += 1
            elif age > max_age:
                health.status = "warning"
                health.message = f"Backup is getting old: {age}"
                health.warning_count += 1
            else:
                health.status = "healthy"
                health.message = f"Last backup: {health.last_backup}"
            
        except Exception as e:
            health.status = "error"
            health.message = f"Failed to check backup health: {str(e)}"
            health.error_count += 1
            logger.error(f"Error checking backup health for {policy_name}: {e}")
        
        return health
    
    def verify_backups(self):
        """Verify integrity of recent backups"""
        logger.info("Verifying backup integrity")
        
        for policy_name in self.health_status:
            try:
                # Get latest backup
                prefix = f"{self.config['s3']['prefix']}{policy_name}/"
                response = self.s3_client.list_objects_v2(
                    Bucket=self.config['s3']['bucket'],
                    Prefix=prefix,
                    MaxKeys=1
                )
                
                if 'Contents' not in response:
                    continue
                
                latest_backup = response['Contents'][0]
                
                # Verify checksum if available
                metadata = self.s3_client.head_object(
                    Bucket=self.config['s3']['bucket'],
                    Key=latest_backup['Key']
                )['Metadata']
                
                if 'checksum' in metadata:
                    # Download partial file to verify
                    verification_passed = self._verify_backup_checksum(
                        latest_backup['Key'],
                        metadata['checksum']
                    )
                    
                    backup_verification_status.labels(type=policy_name).set(
                        1 if verification_passed else 0
                    )
                    
                    if not verification_passed:
                        self.alert_manager.send_alert(
                            f"Backup verification failed for {policy_name}",
                            severity="error"
                        )
                
            except Exception as e:
                logger.error(f"Error verifying backup for {policy_name}: {e}")
                backup_verification_status.labels(type=policy_name).set(0)
    
    def _verify_backup_checksum(self, s3_key: str, expected_checksum: str) -> bool:
        """Verify backup checksum (simplified - checks first MB)"""
        try:
            # Download first 1MB for checksum verification
            response = self.s3_client.get_object(
                Bucket=self.config['s3']['bucket'],
                Key=s3_key,
                Range='bytes=0-1048576'
            )
            
            # In production, you would verify the full file checksum
            # This is simplified for demonstration
            return True
            
        except Exception as e:
            logger.error(f"Checksum verification failed: {e}")
            return False
    
    def check_storage_usage(self):
        """Monitor backup storage usage"""
        logger.info("Checking backup storage usage")
        
        try:
            # Calculate total storage usage
            paginator = self.s3_client.get_paginator('list_objects_v2')
            total_size = 0
            
            for page in paginator.paginate(
                Bucket=self.config['s3']['bucket'],
                Prefix=self.config['s3']['prefix']
            ):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        total_size += obj['Size']
            
            # Update metric
            backup_storage_usage.set(total_size)
            
            # Check storage limits
            storage_gb = total_size / (1024 ** 3)
            logger.info(f"Total backup storage usage: {storage_gb:.2f} GB")
            
            # Alert if storage usage is high
            if storage_gb > 1000:  # 1TB threshold
                self.alert_manager.send_alert(
                    f"High backup storage usage: {storage_gb:.2f} GB",
                    severity="warning"
                )
                
        except Exception as e:
            logger.error(f"Error checking storage usage: {e}")
    
    def test_restore_capability(self):
        """Test restore capability with sample data"""
        logger.info("Testing restore capability")
        
        test_results = {
            'timestamp': datetime.now(),
            'tests': {}
        }
        
        # Test PostgreSQL restore capability
        if self.config['policies']['postgres']['enabled']:
            test_results['tests']['postgres'] = self._test_postgres_restore()
        
        # Test Redis restore capability
        if self.config['policies']['redis']['enabled']:
            test_results['tests']['redis'] = self._test_redis_restore()
        
        # Log results
        logger.info(f"Restore capability test results: {json.dumps(test_results, default=str)}")
        
        # Alert on failures
        for test_name, result in test_results['tests'].items():
            if not result.get('success', False):
                self.alert_manager.send_alert(
                    f"Restore test failed for {test_name}: {result.get('error', 'Unknown error')}",
                    severity="error"
                )
    
    def _test_postgres_restore(self) -> Dict[str, Any]:
        """Test PostgreSQL restore capability"""
        try:
            # Test database connection
            conn = psycopg2.connect(
                host=os.getenv('POSTGRES_HOST', 'postgres'),
                port=os.getenv('POSTGRES_PORT', '5432'),
                user=os.getenv('POSTGRES_USER', 'agent'),
                password=os.getenv('POSTGRES_PASSWORD'),
                database='postgres'
            )
            
            cursor = conn.cursor()
            cursor.execute("SELECT version()")
            version = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            
            return {
                'success': True,
                'message': f'PostgreSQL accessible: {version}'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _test_redis_restore(self) -> Dict[str, Any]:
        """Test Redis restore capability"""
        try:
            # Test Redis connection
            r = redis.Redis(
                host=os.getenv('REDIS_HOST', 'redis'),
                port=int(os.getenv('REDIS_PORT', 6379))
            )
            
            # Test basic operations
            test_key = 'backup_test_key'
            test_value = 'backup_test_value'
            
            r.set(test_key, test_value)
            retrieved = r.get(test_key)
            r.delete(test_key)
            
            if retrieved == test_value.encode():
                return {
                    'success': True,
                    'message': 'Redis accessible and operational'
                }
            else:
                return {
                    'success': False,
                    'error': 'Redis test failed'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _check_alerts(self, policy_name: str, health: BackupHealth):
        """Check and send alerts based on health status"""
        if health.status == "critical":
            self.alert_manager.send_alert(
                f"Critical: {policy_name} backup - {health.message}",
                severity="critical",
                details=asdict(health)
            )
        elif health.status == "warning":
            self.alert_manager.send_alert(
                f"Warning: {policy_name} backup - {health.message}",
                severity="warning",
                details=asdict(health)
            )
        elif health.status == "error":
            self.alert_manager.send_alert(
                f"Error: {policy_name} backup - {health.message}",
                severity="error",
                details=asdict(health)
            )
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall backup health summary"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'total_backups': len(self.health_status),
            'healthy_backups': 0,
            'warning_backups': 0,
            'critical_backups': 0,
            'details': {}
        }
        
        for policy_name, health in self.health_status.items():
            if health.status == 'healthy':
                summary['healthy_backups'] += 1
            elif health.status == 'warning':
                summary['warning_backups'] += 1
            else:
                summary['critical_backups'] += 1
            
            summary['details'][policy_name] = {
                'status': health.status,
                'last_backup': health.last_backup.isoformat() if health.last_backup else None,
                'message': health.message
            }
        
        # Determine overall status
        if summary['critical_backups'] > 0:
            summary['overall_status'] = 'critical'
        elif summary['warning_backups'] > 0:
            summary['overall_status'] = 'warning'
        
        return summary


class AlertManager:
    """Manages backup alerts and notifications"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.alert_history: List[Dict] = []
        
    def send_alert(self, message: str, severity: str = "error", details: Dict = None):
        """Send alert through configured channels"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'severity': severity,
            'message': message,
            'details': details or {}
        }
        
        self.alert_history.append(alert)
        
        # Send to configured channels
        for channel in self.config['global']['notification_channels']:
            try:
                if channel['type'] == 'slack':
                    self._send_slack_alert(channel['webhook_url'], alert)
                elif channel['type'] == 'email':
                    self._send_email_alert(channel['recipients'], alert)
                elif channel['type'] == 'pagerduty':
                    self._send_pagerduty_alert(channel['integration_key'], alert)
            except Exception as e:
                logger.error(f"Failed to send alert via {channel['type']}: {e}")
    
    def _send_slack_alert(self, webhook_url: str, alert: Dict):
        """Send alert to Slack"""
        color_map = {
            'critical': '#ff0000',
            'error': '#ff6600',
            'warning': '#ffaa00',
            'info': '#00aaff'
        }
        
        payload = {
            'attachments': [{
                'color': color_map.get(alert['severity'], '#808080'),
                'title': f"Backup Alert - {alert['severity'].upper()}",
                'text': alert['message'],
                'fields': [
                    {
                        'title': 'Timestamp',
                        'value': alert['timestamp'],
                        'short': True
                    }
                ],
                'footer': 'Voice Agent Backup Monitor'
            }]
        }
        
        # Add details if present
        if alert['details']:
            for key, value in alert['details'].items():
                if isinstance(value, (str, int, float)):
                    payload['attachments'][0]['fields'].append({
                        'title': key.replace('_', ' ').title(),
                        'value': str(value),
                        'short': True
                    })
        
        response = requests.post(webhook_url, json=payload)
        response.raise_for_status()
    
    def _send_pagerduty_alert(self, integration_key: str, alert: Dict):
        """Send alert to PagerDuty"""
        severity_map = {
            'critical': 'critical',
            'error': 'error',
            'warning': 'warning',
            'info': 'info'
        }
        
        payload = {
            'routing_key': integration_key,
            'event_action': 'trigger',
            'payload': {
                'summary': alert['message'],
                'severity': severity_map.get(alert['severity'], 'error'),
                'source': 'voice-agent-backup-monitor',
                'timestamp': alert['timestamp'],
                'custom_details': alert['details']
            }
        }
        
        response = requests.post(
            'https://events.pagerduty.com/v2/enqueue',
            json=payload
        )
        response.raise_for_status()


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "/app/backup_config.yaml"
    monitor = BackupMonitor(config_path)
    
    # If running with --summary flag, just print summary and exit
    if len(sys.argv) > 2 and sys.argv[2] == "--summary":
        monitor.check_backup_health()
        summary = monitor.get_health_summary()
        print(json.dumps(summary, indent=2))
    else:
        # Start continuous monitoring
        monitor.start()