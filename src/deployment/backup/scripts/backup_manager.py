#!/usr/bin/env python3
"""
Backup Manager for LiveKit Voice Agents Platform
Handles automated backups, monitoring, and recovery operations
"""

import os
import sys
import yaml
import json
import logging
import schedule
import time
import boto3
import subprocess
import hashlib
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import psycopg2
import redis
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BackupJob:
    """Represents a backup job"""
    id: str
    type: str
    source: str
    destination: str
    status: str = "pending"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    size_bytes: int = 0
    error: Optional[str] = None
    metadata: Dict[str, Any] = None


class BackupManager:
    """Main backup manager class"""
    
    def __init__(self, config_path: str):
        """Initialize backup manager with configuration"""
        self.config = self._load_config(config_path)
        self.s3_client = self._init_s3_client()
        self.jobs: Dict[str, BackupJob] = {}
        self.executor = ThreadPoolExecutor(
            max_workers=self.config['global']['parallel_jobs']
        )
        self.monitoring = BackupMonitoring(self.config)
        
    def _load_config(self, config_path: str) -> Dict:
        """Load backup configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config['backup_config']
    
    def _init_s3_client(self):
        """Initialize S3 client for backup storage"""
        return boto3.client(
            's3',
            region_name=self.config['s3']['region']
        )
    
    def start(self):
        """Start the backup manager"""
        logger.info("Starting Backup Manager")
        
        # Schedule backup jobs
        self._schedule_backups()
        
        # Start monitoring thread
        monitoring_thread = threading.Thread(
            target=self.monitoring.start,
            daemon=True
        )
        monitoring_thread.start()
        
        # Run scheduler
        while True:
            schedule.run_pending()
            time.sleep(60)
    
    def _schedule_backups(self):
        """Schedule all backup jobs based on configuration"""
        for policy_name, policy in self.config['policies'].items():
            if not policy.get('enabled', True):
                continue
                
            # Schedule different backup types
            if policy['type'] == 'database':
                self._schedule_database_backup(policy_name, policy)
            elif policy['type'] == 'key-value':
                self._schedule_redis_backup(policy_name, policy)
            elif policy['type'] == 'filesystem':
                self._schedule_filesystem_backup(policy_name, policy)
            elif policy['type'] == 'multimedia':
                self._schedule_multimedia_backup(policy_name, policy)
            elif policy['type'] == 'timeseries':
                self._schedule_monitoring_backup(policy_name, policy)
    
    def _schedule_database_backup(self, name: str, policy: Dict):
        """Schedule PostgreSQL database backups"""
        # Full backup
        if 'full_backup' in policy['schedule']:
            schedule_time = policy['schedule']['full_backup']
            schedule.every().day.at(schedule_time.split()[1]).do(
                self._run_postgres_full_backup, name, policy
            )
        
        # Incremental backup
        if 'incremental' in policy['schedule']:
            hours = int(policy['schedule']['incremental'].split()[1].split('/')[1])
            schedule.every(hours).hours.do(
                self._run_postgres_incremental_backup, name, policy
            )
    
    def _run_postgres_full_backup(self, name: str, policy: Dict):
        """Run full PostgreSQL backup"""
        job_id = f"postgres_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        job = BackupJob(
            id=job_id,
            type="postgres_full",
            source="postgres",
            destination=f"s3://{self.config['s3']['bucket']}/postgres/full/"
        )
        
        self.jobs[job_id] = job
        job.started_at = datetime.now()
        
        try:
            # Get database connection details from environment
            db_host = os.getenv('POSTGRES_HOST', 'postgres')
            db_port = os.getenv('POSTGRES_PORT', '5432')
            db_name = os.getenv('POSTGRES_DB', 'voice_agents')
            db_user = os.getenv('POSTGRES_USER', 'agent')
            db_password = os.getenv('POSTGRES_PASSWORD')
            
            # Create backup filename
            backup_file = f"/tmp/{job_id}.sql.gz"
            
            # Run pg_dump with compression
            dump_cmd = [
                'pg_dump',
                f'--host={db_host}',
                f'--port={db_port}',
                f'--username={db_user}',
                '--no-password',
                '--verbose',
                '--format=custom',
                '--compress=9',
                f'--file={backup_file}',
                db_name
            ]
            
            # Set PGPASSWORD environment variable
            env = os.environ.copy()
            env['PGPASSWORD'] = db_password
            
            # Execute backup
            result = subprocess.run(
                dump_cmd,
                env=env,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                raise Exception(f"pg_dump failed: {result.stderr}")
            
            # Calculate checksum
            checksum = self._calculate_checksum(backup_file)
            
            # Upload to S3
            s3_key = f"{self.config['s3']['prefix']}postgres/full/{job_id}.sql.gz"
            self._upload_to_s3(backup_file, s3_key, {
                'backup_type': 'postgres_full',
                'checksum': checksum,
                'db_name': db_name,
                'timestamp': job.started_at.isoformat()
            })
            
            # Update job status
            job.completed_at = datetime.now()
            job.status = "completed"
            job.size_bytes = os.path.getsize(backup_file)
            
            # Cleanup
            os.remove(backup_file)
            
            # Log success
            logger.info(f"PostgreSQL full backup completed: {job_id}")
            self.monitoring.record_backup_success(job)
            
        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            logger.error(f"PostgreSQL backup failed: {e}")
            self.monitoring.record_backup_failure(job)
            self._send_alert(f"PostgreSQL backup failed: {e}")
    
    def _schedule_redis_backup(self, name: str, policy: Dict):
        """Schedule Redis backups"""
        if 'snapshot' in policy['schedule']:
            hours = int(policy['schedule']['snapshot'].split()[1].split('/')[1])
            schedule.every(hours).hours.do(
                self._run_redis_backup, name, policy
            )
    
    def _run_redis_backup(self, name: str, policy: Dict):
        """Run Redis backup"""
        job_id = f"redis_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        job = BackupJob(
            id=job_id,
            type="redis_snapshot",
            source="redis",
            destination=f"s3://{self.config['s3']['bucket']}/redis/"
        )
        
        self.jobs[job_id] = job
        job.started_at = datetime.now()
        
        try:
            # Connect to Redis
            r = redis.Redis(
                host=os.getenv('REDIS_HOST', 'redis'),
                port=int(os.getenv('REDIS_PORT', 6379)),
                decode_responses=False
            )
            
            # Trigger BGSAVE
            r.bgsave()
            
            # Wait for background save to complete
            while r.lastsave() < job.started_at.timestamp():
                time.sleep(1)
            
            # Get RDB file location
            rdb_file = r.config_get('dir')['dir'] + '/' + r.config_get('dbfilename')['dbfilename']
            
            # Compress and upload
            compressed_file = f"/tmp/{job_id}.rdb.gz"
            subprocess.run([
                'gzip', '-c', rdb_file
            ], stdout=open(compressed_file, 'wb'))
            
            # Upload to S3
            s3_key = f"{self.config['s3']['prefix']}redis/{job_id}.rdb.gz"
            self._upload_to_s3(compressed_file, s3_key, {
                'backup_type': 'redis_snapshot',
                'timestamp': job.started_at.isoformat()
            })
            
            # Update job status
            job.completed_at = datetime.now()
            job.status = "completed"
            job.size_bytes = os.path.getsize(compressed_file)
            
            # Cleanup
            os.remove(compressed_file)
            
            logger.info(f"Redis backup completed: {job_id}")
            self.monitoring.record_backup_success(job)
            
        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            logger.error(f"Redis backup failed: {e}")
            self.monitoring.record_backup_failure(job)
    
    def _schedule_filesystem_backup(self, name: str, policy: Dict):
        """Schedule filesystem backups"""
        if 'full_backup' in policy['schedule']:
            schedule_time = policy['schedule']['full_backup']
            schedule.every().day.at(schedule_time.split()[1]).do(
                self._run_filesystem_backup, name, policy
            )
    
    def _run_filesystem_backup(self, name: str, policy: Dict):
        """Run filesystem backup using tar and compression"""
        job_id = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        job = BackupJob(
            id=job_id,
            type=f"filesystem_{name}",
            source=",".join(policy['paths']),
            destination=f"s3://{self.config['s3']['bucket']}/{name}/"
        )
        
        self.jobs[job_id] = job
        job.started_at = datetime.now()
        
        try:
            # Create tar archive with compression
            tar_file = f"/tmp/{job_id}.tar.zst"
            tar_cmd = [
                'tar',
                '--create',
                '--zstd',
                f'--file={tar_file}',
                '--verbose'
            ] + policy['paths']
            
            result = subprocess.run(tar_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"tar failed: {result.stderr}")
            
            # Calculate checksum
            checksum = self._calculate_checksum(tar_file)
            
            # Upload to S3 with multipart for large files
            s3_key = f"{self.config['s3']['prefix']}{name}/{job_id}.tar.zst"
            self._upload_large_file_to_s3(tar_file, s3_key, {
                'backup_type': f'filesystem_{name}',
                'checksum': checksum,
                'paths': json.dumps(policy['paths']),
                'timestamp': job.started_at.isoformat()
            })
            
            # Update job status
            job.completed_at = datetime.now()
            job.status = "completed"
            job.size_bytes = os.path.getsize(tar_file)
            
            # Cleanup
            os.remove(tar_file)
            
            logger.info(f"Filesystem backup completed: {job_id}")
            self.monitoring.record_backup_success(job)
            
        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            logger.error(f"Filesystem backup failed: {e}")
            self.monitoring.record_backup_failure(job)
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA256 checksum of a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _upload_to_s3(self, file_path: str, s3_key: str, metadata: Dict):
        """Upload file to S3 with metadata"""
        self.s3_client.upload_file(
            file_path,
            self.config['s3']['bucket'],
            s3_key,
            ExtraArgs={
                'Metadata': metadata,
                'ServerSideEncryption': self.config['s3']['server_side_encryption'],
                'StorageClass': self.config['s3']['storage_class']
            }
        )
    
    def _upload_large_file_to_s3(self, file_path: str, s3_key: str, metadata: Dict):
        """Upload large file to S3 using multipart upload"""
        # Use boto3's multipart upload for files > 100MB
        file_size = os.path.getsize(file_path)
        
        if file_size > 100 * 1024 * 1024:  # 100MB
            # Configure multipart upload
            config = boto3.s3.transfer.TransferConfig(
                multipart_threshold=1024 * 25,  # 25MB
                max_concurrency=10,
                multipart_chunksize=1024 * 25,
                use_threads=True
            )
            
            self.s3_client.upload_file(
                file_path,
                self.config['s3']['bucket'],
                s3_key,
                ExtraArgs={
                    'Metadata': metadata,
                    'ServerSideEncryption': self.config['s3']['server_side_encryption'],
                    'StorageClass': self.config['s3']['storage_class']
                },
                Config=config
            )
        else:
            self._upload_to_s3(file_path, s3_key, metadata)
    
    def _send_alert(self, message: str, severity: str = "error"):
        """Send alert through configured channels"""
        for channel in self.config['global']['notification_channels']:
            if channel['type'] == 'slack':
                self._send_slack_alert(channel['webhook_url'], message, severity)
            elif channel['type'] == 'email':
                self._send_email_alert(channel['recipients'], message, severity)
            elif channel['type'] == 'pagerduty':
                self._send_pagerduty_alert(channel['integration_key'], message, severity)
    
    def _send_slack_alert(self, webhook_url: str, message: str, severity: str):
        """Send alert to Slack"""
        color = {
            'error': '#ff0000',
            'warning': '#ffaa00',
            'info': '#00aaff'
        }.get(severity, '#808080')
        
        payload = {
            'attachments': [{
                'color': color,
                'title': f'Backup Alert - {severity.upper()}',
                'text': message,
                'timestamp': int(time.time())
            }]
        }
        
        try:
            requests.post(webhook_url, json=payload)
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
    
    def restore_backup(self, backup_id: str, target_path: str) -> bool:
        """Restore a backup from S3"""
        try:
            # Implementation for restore functionality
            logger.info(f"Restoring backup {backup_id} to {target_path}")
            # ... restore logic ...
            return True
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return False


class BackupMonitoring:
    """Monitoring and metrics for backup operations"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.metrics = {
            'backups_total': 0,
            'backups_successful': 0,
            'backups_failed': 0,
            'total_size_bytes': 0,
            'last_backup_timestamp': {}
        }
    
    def start(self):
        """Start monitoring thread"""
        while True:
            self._check_backup_health()
            self._export_metrics()
            time.sleep(300)  # Check every 5 minutes
    
    def record_backup_success(self, job: BackupJob):
        """Record successful backup"""
        self.metrics['backups_total'] += 1
        self.metrics['backups_successful'] += 1
        self.metrics['total_size_bytes'] += job.size_bytes
        self.metrics['last_backup_timestamp'][job.type] = job.completed_at
    
    def record_backup_failure(self, job: BackupJob):
        """Record failed backup"""
        self.metrics['backups_total'] += 1
        self.metrics['backups_failed'] += 1
    
    def _check_backup_health(self):
        """Check if backups are running as expected"""
        now = datetime.now()
        
        # Check each backup type
        for backup_type, last_timestamp in self.metrics['last_backup_timestamp'].items():
            time_since_backup = now - last_timestamp
            
            # Alert if backup is overdue (customize thresholds per type)
            if time_since_backup > timedelta(hours=24):
                logger.warning(f"Backup overdue: {backup_type} last run {time_since_backup} ago")
    
    def _export_metrics(self):
        """Export metrics to Prometheus"""
        # This would integrate with your Prometheus setup
        metrics_data = {
            'backup_manager_total': self.metrics['backups_total'],
            'backup_manager_successful': self.metrics['backups_successful'],
            'backup_manager_failed': self.metrics['backups_failed'],
            'backup_manager_size_bytes': self.metrics['total_size_bytes']
        }
        
        # Export to Prometheus pushgateway or expose via HTTP endpoint
        logger.debug(f"Metrics: {metrics_data}")


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "/app/backup_config.yaml"
    manager = BackupManager(config_path)
    manager.start()