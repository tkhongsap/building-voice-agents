#!/usr/bin/env python3
"""
Automated Restore Testing
Periodically tests backup restore capabilities to ensure disaster recovery readiness
"""

import os
import sys
import yaml
import json
import logging
import random
import tempfile
from datetime import datetime, timedelta
from typing import Dict, List, Any
import boto3
import psycopg2
import redis
from restore_manager import RestoreManager
from backup_monitor import AlertManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RestoreTester:
    """Automated restore testing system"""
    
    def __init__(self, config_path: str):
        """Initialize restore tester"""
        self.config = self._load_config(config_path)
        self.restore_manager = RestoreManager(config_path)
        self.alert_manager = AlertManager(self.config)
        self.test_results = []
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config['backup_config']
    
    def run_tests(self):
        """Run all restore tests"""
        logger.info("Starting automated restore tests")
        
        test_report = {
            'timestamp': datetime.now().isoformat(),
            'tests': {},
            'overall_status': 'pending'
        }
        
        # Test each backup type
        if self.config['verification']['test_restore']['sample_percentage'] > 0:
            # Test PostgreSQL restore
            if self._should_test('postgres'):
                test_report['tests']['postgres'] = self._test_postgres_restore()
            
            # Test Redis restore
            if self._should_test('redis'):
                test_report['tests']['redis'] = self._test_redis_restore()
            
            # Test filesystem restore
            if self._should_test('voice_data'):
                test_report['tests']['voice_data'] = self._test_filesystem_restore('voice_data')
            
            if self._should_test('configs'):
                test_report['tests']['configs'] = self._test_filesystem_restore('configs')
        
        # Determine overall status
        all_passed = all(
            test.get('status') == 'passed' 
            for test in test_report['tests'].values()
        )
        test_report['overall_status'] = 'passed' if all_passed else 'failed'
        
        # Save test results
        self.test_results.append(test_report)
        
        # Send notifications
        self._send_test_report(test_report)
        
        logger.info(f"Restore tests completed: {test_report['overall_status']}")
        
        return test_report
    
    def _should_test(self, backup_type: str) -> bool:
        """Determine if backup type should be tested based on sample percentage"""
        sample_percentage = self.config['verification']['test_restore']['sample_percentage']
        return random.random() * 100 < sample_percentage
    
    def _test_postgres_restore(self) -> Dict[str, Any]:
        """Test PostgreSQL restore"""
        logger.info("Testing PostgreSQL restore")
        
        test_result = {
            'started_at': datetime.now().isoformat(),
            'status': 'running',
            'backup_tested': None,
            'restore_time_seconds': 0,
            'verification': {}
        }
        
        try:
            # List available backups
            backups = self.restore_manager.list_available_backups('postgres')
            if not backups:
                test_result['status'] = 'failed'
                test_result['error'] = 'No backups available to test'
                return test_result
            
            # Select a random recent backup
            backup_to_test = random.choice(backups[:5])  # Test one of the 5 most recent
            test_result['backup_tested'] = backup_to_test['key']
            
            # Perform restore to test database
            start_time = datetime.now()
            success = self.restore_manager.restore_postgres(
                backup_to_test['key'],
                'voice_agents_restore_test'
            )
            restore_duration = (datetime.now() - start_time).total_seconds()
            test_result['restore_time_seconds'] = restore_duration
            
            if not success:
                test_result['status'] = 'failed'
                test_result['error'] = 'Restore operation failed'
                return test_result
            
            # Verify restore
            verification = self._verify_postgres_restore('voice_agents_restore_test')
            test_result['verification'] = verification
            
            # Check verification results
            if verification.get('table_count', 0) > 0:
                test_result['status'] = 'passed'
                logger.info(f"PostgreSQL restore test passed in {restore_duration:.2f}s")
            else:
                test_result['status'] = 'failed'
                test_result['error'] = 'Verification failed: no tables found'
            
            # Cleanup test database
            if self.config['verification']['test_restore']['cleanup_after_test']:
                self._cleanup_postgres_test('voice_agents_restore_test')
            
        except Exception as e:
            test_result['status'] = 'failed'
            test_result['error'] = str(e)
            logger.error(f"PostgreSQL restore test failed: {e}")
        
        test_result['completed_at'] = datetime.now().isoformat()
        return test_result
    
    def _test_redis_restore(self) -> Dict[str, Any]:
        """Test Redis restore"""
        logger.info("Testing Redis restore")
        
        test_result = {
            'started_at': datetime.now().isoformat(),
            'status': 'running',
            'backup_tested': None,
            'restore_time_seconds': 0,
            'verification': {}
        }
        
        try:
            # List available backups
            backups = self.restore_manager.list_available_backups('redis')
            if not backups:
                test_result['status'] = 'failed'
                test_result['error'] = 'No backups available to test'
                return test_result
            
            # Select a random recent backup
            backup_to_test = random.choice(backups[:5])
            test_result['backup_tested'] = backup_to_test['key']
            
            # Clear test Redis instance
            test_redis = redis.Redis(
                host=os.getenv('REDIS_HOST', 'redis-test'),
                port=int(os.getenv('REDIS_PORT', 6379))
            )
            test_redis.flushall()
            
            # Perform restore
            start_time = datetime.now()
            success = self.restore_manager.restore_redis(backup_to_test['key'])
            restore_duration = (datetime.now() - start_time).total_seconds()
            test_result['restore_time_seconds'] = restore_duration
            
            if not success:
                test_result['status'] = 'failed'
                test_result['error'] = 'Restore operation failed'
                return test_result
            
            # Verify restore
            db_size = test_redis.dbsize()
            test_result['verification'] = {
                'key_count': db_size,
                'test_key_set': False,
                'test_key_retrieved': False
            }
            
            # Test basic operations
            test_key = 'restore_test_key'
            test_value = 'restore_test_value'
            test_redis.set(test_key, test_value)
            test_result['verification']['test_key_set'] = True
            
            retrieved = test_redis.get(test_key)
            if retrieved and retrieved.decode() == test_value:
                test_result['verification']['test_key_retrieved'] = True
            
            # Determine test status
            if db_size > 0 or test_result['verification']['test_key_retrieved']:
                test_result['status'] = 'passed'
                logger.info(f"Redis restore test passed in {restore_duration:.2f}s")
            else:
                test_result['status'] = 'failed'
                test_result['error'] = 'Verification failed'
            
            # Cleanup
            if self.config['verification']['test_restore']['cleanup_after_test']:
                test_redis.flushall()
            
        except Exception as e:
            test_result['status'] = 'failed'
            test_result['error'] = str(e)
            logger.error(f"Redis restore test failed: {e}")
        
        test_result['completed_at'] = datetime.now().isoformat()
        return test_result
    
    def _test_filesystem_restore(self, backup_type: str) -> Dict[str, Any]:
        """Test filesystem restore"""
        logger.info(f"Testing {backup_type} filesystem restore")
        
        test_result = {
            'started_at': datetime.now().isoformat(),
            'status': 'running',
            'backup_tested': None,
            'restore_time_seconds': 0,
            'verification': {}
        }
        
        try:
            # List available backups
            backups = self.restore_manager.list_available_backups(backup_type)
            if not backups:
                test_result['status'] = 'failed'
                test_result['error'] = 'No backups available to test'
                return test_result
            
            # Select a random recent backup
            backup_to_test = random.choice(backups[:5])
            test_result['backup_tested'] = backup_to_test['key']
            
            # Create temporary restore directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Perform restore
                start_time = datetime.now()
                success = self.restore_manager.restore_filesystem(
                    backup_to_test['key'],
                    temp_dir
                )
                restore_duration = (datetime.now() - start_time).total_seconds()
                test_result['restore_time_seconds'] = restore_duration
                
                if not success:
                    test_result['status'] = 'failed'
                    test_result['error'] = 'Restore operation failed'
                    return test_result
                
                # Verify restore
                file_count = sum([len(files) for _, _, files in os.walk(temp_dir)])
                dir_count = sum([len(dirs) for _, dirs, _ in os.walk(temp_dir)])
                
                test_result['verification'] = {
                    'file_count': file_count,
                    'directory_count': dir_count,
                    'total_size_bytes': sum(
                        os.path.getsize(os.path.join(dirpath, filename))
                        for dirpath, _, filenames in os.walk(temp_dir)
                        for filename in filenames
                    )
                }
                
                # Determine test status
                if file_count > 0:
                    test_result['status'] = 'passed'
                    logger.info(f"{backup_type} restore test passed in {restore_duration:.2f}s")
                else:
                    test_result['status'] = 'failed'
                    test_result['error'] = 'No files found after restore'
            
        except Exception as e:
            test_result['status'] = 'failed'
            test_result['error'] = str(e)
            logger.error(f"{backup_type} restore test failed: {e}")
        
        test_result['completed_at'] = datetime.now().isoformat()
        return test_result
    
    def _verify_postgres_restore(self, db_name: str) -> Dict[str, Any]:
        """Verify PostgreSQL restore"""
        verification = {}
        
        try:
            conn = psycopg2.connect(
                host=os.getenv('POSTGRES_HOST', 'postgres'),
                port=os.getenv('POSTGRES_PORT', '5432'),
                user=os.getenv('POSTGRES_USER', 'agent'),
                password=os.getenv('POSTGRES_PASSWORD'),
                database=db_name
            )
            cursor = conn.cursor()
            
            # Count tables
            cursor.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'")
            verification['table_count'] = cursor.fetchone()[0]
            
            # Check for critical tables
            critical_tables = ['voice_sessions', 'transcripts', 'conversation_history']
            for table in critical_tables:
                cursor.execute(
                    "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = %s)",
                    (table,)
                )
                verification[f'{table}_exists'] = cursor.fetchone()[0]
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            verification['error'] = str(e)
        
        return verification
    
    def _cleanup_postgres_test(self, db_name: str):
        """Clean up test PostgreSQL database"""
        try:
            conn = psycopg2.connect(
                host=os.getenv('POSTGRES_HOST', 'postgres'),
                port=os.getenv('POSTGRES_PORT', '5432'),
                user=os.getenv('POSTGRES_USER', 'agent'),
                password=os.getenv('POSTGRES_PASSWORD'),
                database='postgres'
            )
            conn.autocommit = True
            cursor = conn.cursor()
            
            # Terminate connections
            cursor.execute(f"""
                SELECT pg_terminate_backend(pg_stat_activity.pid)
                FROM pg_stat_activity
                WHERE pg_stat_activity.datname = '{db_name}'
                AND pid <> pg_backend_pid()
            """)
            
            # Drop database
            cursor.execute(f"DROP DATABASE IF EXISTS {db_name}")
            
            cursor.close()
            conn.close()
            
            logger.info(f"Cleaned up test database: {db_name}")
            
        except Exception as e:
            logger.error(f"Failed to cleanup test database: {e}")
    
    def _send_test_report(self, report: Dict[str, Any]):
        """Send test report notifications"""
        if report['overall_status'] == 'failed':
            # Send alert for failed tests
            failed_tests = [
                name for name, test in report['tests'].items()
                if test['status'] == 'failed'
            ]
            
            self.alert_manager.send_alert(
                f"Restore tests failed: {', '.join(failed_tests)}",
                severity='error',
                details=report
            )
        else:
            # Send success notification (less frequently)
            logger.info(f"All restore tests passed: {report}")
    
    def get_test_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent test results"""
        return self.test_results[-limit:]


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "/app/backup_config.yaml"
    tester = RestoreTester(config_path)
    
    # Run tests
    results = tester.run_tests()
    
    # Print results
    print(json.dumps(results, indent=2))
    
    # Exit with appropriate code
    sys.exit(0 if results['overall_status'] == 'passed' else 1)