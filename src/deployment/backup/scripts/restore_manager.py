#!/usr/bin/env python3
"""
Restore Manager for LiveKit Voice Agents Platform
Handles disaster recovery and restore operations
"""

import os
import sys
import yaml
import json
import logging
import boto3
import subprocess
import tempfile
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import psycopg2
import redis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RestoreJob:
    """Represents a restore job"""
    id: str
    backup_id: str
    type: str
    source: str
    target: str
    status: str = "pending"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


class RestoreManager:
    """Handles backup restoration and disaster recovery"""
    
    def __init__(self, config_path: str):
        """Initialize restore manager"""
        self.config = self._load_config(config_path)
        self.s3_client = self._init_s3_client()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config['backup_config']
    
    def _init_s3_client(self):
        """Initialize S3 client"""
        return boto3.client(
            's3',
            region_name=self.config['s3']['region']
        )
    
    def list_available_backups(self, backup_type: str = None) -> List[Dict]:
        """List available backups from S3"""
        backups = []
        prefix = self.config['s3']['prefix']
        
        if backup_type:
            prefix += f"{backup_type}/"
        
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(
                Bucket=self.config['s3']['bucket'],
                Prefix=prefix
            )
            
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        # Get object metadata
                        metadata = self.s3_client.head_object(
                            Bucket=self.config['s3']['bucket'],
                            Key=obj['Key']
                        )['Metadata']
                        
                        backups.append({
                            'key': obj['Key'],
                            'size': obj['Size'],
                            'last_modified': obj['LastModified'],
                            'type': metadata.get('backup_type', 'unknown'),
                            'checksum': metadata.get('checksum', ''),
                            'timestamp': metadata.get('timestamp', '')
                        })
            
            # Sort by timestamp descending
            backups.sort(key=lambda x: x['last_modified'], reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to list backups: {e}")
            
        return backups
    
    def restore_postgres(self, backup_key: str, target_db: str = None) -> bool:
        """Restore PostgreSQL database from backup"""
        job_id = f"restore_postgres_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        job = RestoreJob(
            id=job_id,
            backup_id=backup_key,
            type="postgres",
            source=f"s3://{self.config['s3']['bucket']}/{backup_key}",
            target=target_db or "postgres"
        )
        
        job.started_at = datetime.now()
        
        try:
            # Download backup from S3
            with tempfile.NamedTemporaryFile(suffix='.sql.gz', delete=False) as tmp_file:
                logger.info(f"Downloading backup from S3: {backup_key}")
                self.s3_client.download_file(
                    self.config['s3']['bucket'],
                    backup_key,
                    tmp_file.name
                )
                
                # Get database connection details
                db_host = os.getenv('POSTGRES_HOST', 'postgres')
                db_port = os.getenv('POSTGRES_PORT', '5432')
                db_name = target_db or os.getenv('POSTGRES_DB', 'voice_agents')
                db_user = os.getenv('POSTGRES_USER', 'agent')
                db_password = os.getenv('POSTGRES_PASSWORD')
                
                # Drop existing database and recreate
                logger.info(f"Preparing database: {db_name}")
                conn = psycopg2.connect(
                    host=db_host,
                    port=db_port,
                    user=db_user,
                    password=db_password,
                    database='postgres'
                )
                conn.autocommit = True
                cursor = conn.cursor()
                
                # Terminate existing connections
                cursor.execute(f"""
                    SELECT pg_terminate_backend(pg_stat_activity.pid)
                    FROM pg_stat_activity
                    WHERE pg_stat_activity.datname = '{db_name}'
                    AND pid <> pg_backend_pid()
                """)
                
                # Drop and recreate database
                cursor.execute(f"DROP DATABASE IF EXISTS {db_name}")
                cursor.execute(f"CREATE DATABASE {db_name}")
                cursor.close()
                conn.close()
                
                # Restore backup
                logger.info(f"Restoring database from backup")
                restore_cmd = [
                    'pg_restore',
                    f'--host={db_host}',
                    f'--port={db_port}',
                    f'--username={db_user}',
                    f'--dbname={db_name}',
                    '--no-password',
                    '--verbose',
                    '--no-owner',
                    '--no-privileges',
                    tmp_file.name
                ]
                
                env = os.environ.copy()
                env['PGPASSWORD'] = db_password
                
                result = subprocess.run(
                    restore_cmd,
                    env=env,
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    # pg_restore returns warnings as errors, check if critical
                    if "ERROR" in result.stderr:
                        raise Exception(f"pg_restore failed: {result.stderr}")
                    else:
                        logger.warning(f"pg_restore warnings: {result.stderr}")
                
                # Verify restore
                conn = psycopg2.connect(
                    host=db_host,
                    port=db_port,
                    user=db_user,
                    password=db_password,
                    database=db_name
                )
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'")
                table_count = cursor.fetchone()[0]
                cursor.close()
                conn.close()
                
                logger.info(f"Database restored successfully with {table_count} tables")
                
                # Cleanup
                os.unlink(tmp_file.name)
                
                job.completed_at = datetime.now()
                job.status = "completed"
                return True
                
        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            logger.error(f"PostgreSQL restore failed: {e}")
            return False
    
    def restore_redis(self, backup_key: str) -> bool:
        """Restore Redis from backup"""
        job_id = f"restore_redis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        job = RestoreJob(
            id=job_id,
            backup_id=backup_key,
            type="redis",
            source=f"s3://{self.config['s3']['bucket']}/{backup_key}",
            target="redis"
        )
        
        job.started_at = datetime.now()
        
        try:
            # Download backup from S3
            with tempfile.NamedTemporaryFile(suffix='.rdb.gz', delete=False) as tmp_file:
                logger.info(f"Downloading Redis backup from S3: {backup_key}")
                self.s3_client.download_file(
                    self.config['s3']['bucket'],
                    backup_key,
                    tmp_file.name
                )
                
                # Decompress
                rdb_file = tmp_file.name.replace('.gz', '')
                subprocess.run(['gunzip', '-c', tmp_file.name], stdout=open(rdb_file, 'wb'))
                
                # Connect to Redis
                r = redis.Redis(
                    host=os.getenv('REDIS_HOST', 'redis'),
                    port=int(os.getenv('REDIS_PORT', 6379))
                )
                
                # Clear existing data
                logger.info("Flushing Redis database")
                r.flushall()
                
                # Get Redis data directory
                redis_dir = r.config_get('dir')['dir']
                redis_dbfile = r.config_get('dbfilename')['dbfilename']
                target_path = os.path.join(redis_dir, redis_dbfile)
                
                # Copy RDB file to Redis directory
                logger.info(f"Copying RDB file to {target_path}")
                shutil.copy2(rdb_file, target_path)
                
                # Restart Redis to load the RDB file
                logger.info("Restarting Redis to load backup")
                r.shutdown(nosave=True)
                
                # Wait for Redis to come back up
                import time
                time.sleep(5)
                
                # Verify restore
                r = redis.Redis(
                    host=os.getenv('REDIS_HOST', 'redis'),
                    port=int(os.getenv('REDIS_PORT', 6379))
                )
                db_size = r.dbsize()
                logger.info(f"Redis restored successfully with {db_size} keys")
                
                # Cleanup
                os.unlink(tmp_file.name)
                os.unlink(rdb_file)
                
                job.completed_at = datetime.now()
                job.status = "completed"
                return True
                
        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            logger.error(f"Redis restore failed: {e}")
            return False
    
    def restore_filesystem(self, backup_key: str, target_path: str) -> bool:
        """Restore filesystem backup"""
        job_id = f"restore_filesystem_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        job = RestoreJob(
            id=job_id,
            backup_id=backup_key,
            type="filesystem",
            source=f"s3://{self.config['s3']['bucket']}/{backup_key}",
            target=target_path
        )
        
        job.started_at = datetime.now()
        
        try:
            # Download backup from S3
            with tempfile.NamedTemporaryFile(suffix='.tar.zst', delete=False) as tmp_file:
                logger.info(f"Downloading filesystem backup from S3: {backup_key}")
                self.s3_client.download_file(
                    self.config['s3']['bucket'],
                    backup_key,
                    tmp_file.name
                )
                
                # Create target directory if not exists
                os.makedirs(target_path, exist_ok=True)
                
                # Extract archive
                logger.info(f"Extracting backup to {target_path}")
                extract_cmd = [
                    'tar',
                    '--extract',
                    '--zstd',
                    f'--file={tmp_file.name}',
                    '--directory', target_path,
                    '--verbose'
                ]
                
                result = subprocess.run(extract_cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    raise Exception(f"tar extraction failed: {result.stderr}")
                
                logger.info(f"Filesystem restored successfully to {target_path}")
                
                # Cleanup
                os.unlink(tmp_file.name)
                
                job.completed_at = datetime.now()
                job.status = "completed"
                return True
                
        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            logger.error(f"Filesystem restore failed: {e}")
            return False
    
    def verify_restore(self, restore_type: str, target: str) -> Dict[str, Any]:
        """Verify restore operation completed successfully"""
        verification_results = {
            'status': 'unknown',
            'checks': {},
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            if restore_type == 'postgres':
                # Verify PostgreSQL restore
                conn = psycopg2.connect(
                    host=os.getenv('POSTGRES_HOST', 'postgres'),
                    port=os.getenv('POSTGRES_PORT', '5432'),
                    user=os.getenv('POSTGRES_USER', 'agent'),
                    password=os.getenv('POSTGRES_PASSWORD'),
                    database=target
                )
                cursor = conn.cursor()
                
                # Check tables
                cursor.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'")
                table_count = cursor.fetchone()[0]
                verification_results['checks']['table_count'] = table_count
                
                # Check critical tables
                critical_tables = ['voice_sessions', 'transcripts', 'conversation_history']
                for table in critical_tables:
                    cursor.execute(f"SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = '{table}')")
                    exists = cursor.fetchone()[0]
                    verification_results['checks'][f'table_{table}_exists'] = exists
                
                cursor.close()
                conn.close()
                
                verification_results['status'] = 'success' if table_count > 0 else 'failed'
                
            elif restore_type == 'redis':
                # Verify Redis restore
                r = redis.Redis(
                    host=os.getenv('REDIS_HOST', 'redis'),
                    port=int(os.getenv('REDIS_PORT', 6379))
                )
                
                db_size = r.dbsize()
                verification_results['checks']['key_count'] = db_size
                verification_results['status'] = 'success' if db_size > 0 else 'failed'
                
            elif restore_type == 'filesystem':
                # Verify filesystem restore
                if os.path.exists(target):
                    file_count = sum([len(files) for _, _, files in os.walk(target)])
                    dir_count = sum([len(dirs) for _, dirs, _ in os.walk(target)])
                    
                    verification_results['checks']['file_count'] = file_count
                    verification_results['checks']['directory_count'] = dir_count
                    verification_results['status'] = 'success' if file_count > 0 else 'failed'
                else:
                    verification_results['status'] = 'failed'
                    verification_results['error'] = 'Target path does not exist'
                    
        except Exception as e:
            verification_results['status'] = 'error'
            verification_results['error'] = str(e)
            
        return verification_results
    
    def perform_disaster_recovery(self, recovery_point: datetime = None) -> Dict[str, Any]:
        """Perform full disaster recovery to specified point in time"""
        logger.info("Starting disaster recovery procedure")
        
        recovery_results = {
            'started_at': datetime.now(),
            'recovery_point': recovery_point or 'latest',
            'components': {},
            'overall_status': 'in_progress'
        }
        
        try:
            # Step 1: Find appropriate backups for recovery point
            backups_to_restore = self._find_recovery_backups(recovery_point)
            
            # Step 2: Restore PostgreSQL
            if 'postgres' in backups_to_restore:
                logger.info("Restoring PostgreSQL database")
                postgres_success = self.restore_postgres(
                    backups_to_restore['postgres']['key'],
                    'voice_agents_recovery'
                )
                recovery_results['components']['postgres'] = {
                    'status': 'success' if postgres_success else 'failed',
                    'backup_used': backups_to_restore['postgres']['key']
                }
            
            # Step 3: Restore Redis
            if 'redis' in backups_to_restore:
                logger.info("Restoring Redis data")
                redis_success = self.restore_redis(backups_to_restore['redis']['key'])
                recovery_results['components']['redis'] = {
                    'status': 'success' if redis_success else 'failed',
                    'backup_used': backups_to_restore['redis']['key']
                }
            
            # Step 4: Restore voice data
            if 'voice_data' in backups_to_restore:
                logger.info("Restoring voice data")
                voice_success = self.restore_filesystem(
                    backups_to_restore['voice_data']['key'],
                    '/data/recovery/voice_data'
                )
                recovery_results['components']['voice_data'] = {
                    'status': 'success' if voice_success else 'failed',
                    'backup_used': backups_to_restore['voice_data']['key']
                }
            
            # Step 5: Restore configurations
            if 'configs' in backups_to_restore:
                logger.info("Restoring configuration files")
                config_success = self.restore_filesystem(
                    backups_to_restore['configs']['key'],
                    '/data/recovery/configs'
                )
                recovery_results['components']['configs'] = {
                    'status': 'success' if config_success else 'failed',
                    'backup_used': backups_to_restore['configs']['key']
                }
            
            # Step 6: Verify all restores
            all_successful = all(
                comp['status'] == 'success' 
                for comp in recovery_results['components'].values()
            )
            
            recovery_results['completed_at'] = datetime.now()
            recovery_results['overall_status'] = 'success' if all_successful else 'partial_failure'
            recovery_results['duration'] = (
                recovery_results['completed_at'] - recovery_results['started_at']
            ).total_seconds()
            
            logger.info(f"Disaster recovery completed with status: {recovery_results['overall_status']}")
            
        except Exception as e:
            recovery_results['overall_status'] = 'failed'
            recovery_results['error'] = str(e)
            logger.error(f"Disaster recovery failed: {e}")
            
        return recovery_results
    
    def _find_recovery_backups(self, recovery_point: datetime = None) -> Dict[str, Dict]:
        """Find appropriate backups for recovery point"""
        recovery_backups = {}
        
        # List all backup types
        backup_types = ['postgres', 'redis', 'voice_data', 'configs', 'logs']
        
        for backup_type in backup_types:
            backups = self.list_available_backups(backup_type)
            
            if recovery_point:
                # Find backup closest to but before recovery point
                suitable_backup = None
                for backup in backups:
                    backup_time = datetime.fromisoformat(backup['timestamp'])
                    if backup_time <= recovery_point:
                        suitable_backup = backup
                        break
                
                if suitable_backup:
                    recovery_backups[backup_type] = suitable_backup
            else:
                # Use latest backup
                if backups:
                    recovery_backups[backup_type] = backups[0]
        
        return recovery_backups


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: restore_manager.py <command> [options]")
        print("Commands:")
        print("  list [type] - List available backups")
        print("  restore-postgres <backup_key> [target_db] - Restore PostgreSQL")
        print("  restore-redis <backup_key> - Restore Redis")
        print("  restore-filesystem <backup_key> <target_path> - Restore filesystem")
        print("  disaster-recovery [recovery_point] - Full disaster recovery")
        sys.exit(1)
    
    config_path = os.getenv('BACKUP_CONFIG_PATH', '/app/backup_config.yaml')
    manager = RestoreManager(config_path)
    
    command = sys.argv[1]
    
    if command == "list":
        backup_type = sys.argv[2] if len(sys.argv) > 2 else None
        backups = manager.list_available_backups(backup_type)
        
        print(f"Available backups ({backup_type or 'all'}):")
        for backup in backups[:10]:  # Show latest 10
            print(f"  - {backup['key']}")
            print(f"    Size: {backup['size'] / 1024 / 1024:.2f} MB")
            print(f"    Type: {backup['type']}")
            print(f"    Timestamp: {backup['timestamp']}")
            print()
    
    elif command == "restore-postgres":
        if len(sys.argv) < 3:
            print("Error: backup_key required")
            sys.exit(1)
        
        backup_key = sys.argv[2]
        target_db = sys.argv[3] if len(sys.argv) > 3 else None
        
        success = manager.restore_postgres(backup_key, target_db)
        sys.exit(0 if success else 1)
    
    elif command == "restore-redis":
        if len(sys.argv) < 3:
            print("Error: backup_key required")
            sys.exit(1)
        
        backup_key = sys.argv[2]
        success = manager.restore_redis(backup_key)
        sys.exit(0 if success else 1)
    
    elif command == "restore-filesystem":
        if len(sys.argv) < 4:
            print("Error: backup_key and target_path required")
            sys.exit(1)
        
        backup_key = sys.argv[2]
        target_path = sys.argv[3]
        
        success = manager.restore_filesystem(backup_key, target_path)
        sys.exit(0 if success else 1)
    
    elif command == "disaster-recovery":
        recovery_point = None
        if len(sys.argv) > 2:
            recovery_point = datetime.fromisoformat(sys.argv[2])
        
        results = manager.perform_disaster_recovery(recovery_point)
        print(json.dumps(results, indent=2, default=str))
        sys.exit(0 if results['overall_status'] == 'success' else 1)
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)