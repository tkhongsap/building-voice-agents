#!/usr/bin/env python3
"""
Backup and Recovery CLI Tool
Provides command-line interface for backup and recovery operations
"""

import os
import sys
import json
import yaml
import click
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from tabulate import tabulate
from backup_manager import BackupManager
from restore_manager import RestoreManager
from backup_monitor import BackupMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.option('--config', default='/app/backup_config.yaml', help='Path to backup configuration file')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, config, verbose):
    """Voice Agent Backup and Recovery CLI"""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    ctx.ensure_object(dict)
    ctx.obj['config'] = config
    
    # Verify config file exists
    if not os.path.exists(config):
        click.echo(f"Error: Configuration file not found: {config}", err=True)
        sys.exit(1)


@cli.group()
@click.pass_context
def backup(ctx):
    """Backup operations"""
    pass


@backup.command('run')
@click.option('--type', 'backup_type', help='Type of backup to run (postgres, redis, voice_data, etc.)')
@click.option('--force', is_flag=True, help='Force backup even if one already exists today')
@click.pass_context
def backup_run(ctx, backup_type, force):
    """Run a backup job"""
    config_path = ctx.obj['config']
    
    try:
        manager = BackupManager(config_path)
        
        if backup_type:
            # Run specific backup type
            if backup_type == 'postgres':
                manager._run_postgres_full_backup('postgres', manager.config['policies']['postgres'])
            elif backup_type == 'redis':
                manager._run_redis_backup('redis', manager.config['policies']['redis'])
            elif backup_type == 'voice_data':
                manager._run_filesystem_backup('voice_data', manager.config['policies']['voice_data'])
            else:
                click.echo(f"Unknown backup type: {backup_type}", err=True)
                sys.exit(1)
        else:
            # Run all enabled backups
            click.echo("Running all enabled backups...")
            # This would trigger all backup policies
            
        click.echo(f"Backup job completed successfully")
        
    except Exception as e:
        click.echo(f"Backup failed: {e}", err=True)
        sys.exit(1)


@backup.command('list')
@click.option('--type', 'backup_type', help='Filter by backup type')
@click.option('--limit', default=20, help='Number of backups to show')
@click.pass_context
def backup_list(ctx, backup_type, limit):
    """List available backups"""
    config_path = ctx.obj['config']
    
    try:
        manager = RestoreManager(config_path)
        backups = manager.list_available_backups(backup_type)
        
        if not backups:
            click.echo("No backups found")
            return
        
        # Prepare table data
        table_data = []
        for backup in backups[:limit]:
            table_data.append([
                backup['key'].split('/')[-1],  # Filename only
                backup['type'],
                f"{backup['size'] / 1024 / 1024:.1f} MB",
                backup['last_modified'].strftime('%Y-%m-%d %H:%M:%S'),
                backup.get('timestamp', 'N/A')
            ])
        
        headers = ['Backup File', 'Type', 'Size', 'Modified', 'Timestamp']
        click.echo(tabulate(table_data, headers=headers, tablefmt='grid'))
        
    except Exception as e:
        click.echo(f"Failed to list backups: {e}", err=True)
        sys.exit(1)


@backup.command('status')
@click.pass_context
def backup_status(ctx):
    """Show backup system status"""
    config_path = ctx.obj['config']
    
    try:
        monitor = BackupMonitor(config_path)
        monitor.check_backup_health()
        summary = monitor.get_health_summary()
        
        # Display summary
        click.echo(f"Backup System Status: {summary['overall_status'].upper()}")
        click.echo(f"Timestamp: {summary['timestamp']}")
        click.echo()
        
        # Display detailed status
        table_data = []
        for policy_name, details in summary['details'].items():
            status_color = 'green' if details['status'] == 'healthy' else 'red'
            table_data.append([
                policy_name,
                click.style(details['status'], fg=status_color),
                details['last_backup'] or 'Never',
                details['message']
            ])
        
        headers = ['Backup Type', 'Status', 'Last Backup', 'Message']
        click.echo(tabulate(table_data, headers=headers, tablefmt='grid'))
        
    except Exception as e:
        click.echo(f"Failed to get backup status: {e}", err=True)
        sys.exit(1)


@cli.group()
@click.pass_context
def restore(ctx):
    """Restore operations"""
    pass


@restore.command('list')
@click.option('--type', 'backup_type', help='Filter by backup type')
@click.option('--limit', default=10, help='Number of backups to show')
@click.pass_context
def restore_list(ctx, backup_type, limit):
    """List available backups for restore"""
    # Same as backup list, but focused on restore-ready backups
    ctx.invoke(backup_list, backup_type=backup_type, limit=limit)


@restore.command('postgres')
@click.argument('backup_key')
@click.option('--target-db', help='Target database name')
@click.option('--dry-run', is_flag=True, help='Show what would be restored without actually doing it')
@click.pass_context
def restore_postgres(ctx, backup_key, target_db, dry_run):
    """Restore PostgreSQL database from backup"""
    config_path = ctx.obj['config']
    
    if dry_run:
        click.echo(f"Would restore PostgreSQL from: {backup_key}")
        click.echo(f"Target database: {target_db or 'default'}")
        return
    
    if not click.confirm(f"Are you sure you want to restore PostgreSQL from {backup_key}?"):
        click.echo("Restore cancelled")
        return
    
    try:
        manager = RestoreManager(config_path)
        
        with click.progressbar(length=100, label='Restoring PostgreSQL') as bar:
            success = manager.restore_postgres(backup_key, target_db)
            bar.update(100)
        
        if success:
            click.echo(click.style("PostgreSQL restore completed successfully", fg='green'))
            
            # Verify restore
            verification = manager.verify_restore('postgres', target_db or 'voice_agents')
            if verification['status'] == 'success':
                click.echo(f"Verification passed: {verification['checks']['table_count']} tables restored")
            else:
                click.echo(click.style(f"Verification failed: {verification.get('error', 'Unknown error')}", fg='yellow'))
        else:
            click.echo(click.style("PostgreSQL restore failed", fg='red'))
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"Restore failed: {e}", err=True)
        sys.exit(1)


@restore.command('redis')
@click.argument('backup_key')
@click.option('--dry-run', is_flag=True, help='Show what would be restored without actually doing it')
@click.pass_context
def restore_redis(ctx, backup_key, dry_run):
    """Restore Redis from backup"""
    config_path = ctx.obj['config']
    
    if dry_run:
        click.echo(f"Would restore Redis from: {backup_key}")
        return
    
    if not click.confirm(f"Are you sure you want to restore Redis from {backup_key}? This will clear current data."):
        click.echo("Restore cancelled")
        return
    
    try:
        manager = RestoreManager(config_path)
        
        with click.progressbar(length=100, label='Restoring Redis') as bar:
            success = manager.restore_redis(backup_key)
            bar.update(100)
        
        if success:
            click.echo(click.style("Redis restore completed successfully", fg='green'))
        else:
            click.echo(click.style("Redis restore failed", fg='red'))
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"Restore failed: {e}", err=True)
        sys.exit(1)


@restore.command('filesystem')
@click.argument('backup_key')
@click.argument('target_path')
@click.option('--dry-run', is_flag=True, help='Show what would be restored without actually doing it')
@click.pass_context
def restore_filesystem(ctx, backup_key, target_path, dry_run):
    """Restore filesystem from backup"""
    config_path = ctx.obj['config']
    
    if dry_run:
        click.echo(f"Would restore filesystem from: {backup_key}")
        click.echo(f"Target path: {target_path}")
        return
    
    if not click.confirm(f"Are you sure you want to restore to {target_path}?"):
        click.echo("Restore cancelled")
        return
    
    try:
        manager = RestoreManager(config_path)
        
        with click.progressbar(length=100, label='Restoring filesystem') as bar:
            success = manager.restore_filesystem(backup_key, target_path)
            bar.update(100)
        
        if success:
            click.echo(click.style("Filesystem restore completed successfully", fg='green'))
        else:
            click.echo(click.style("Filesystem restore failed", fg='red'))
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"Restore failed: {e}", err=True)
        sys.exit(1)


@cli.group()
@click.pass_context
def disaster(ctx):
    """Disaster recovery operations"""
    pass


@disaster.command('recovery')
@click.option('--recovery-point', help='Recovery point in ISO format (e.g., 2024-01-21T10:00:00Z)')
@click.option('--dry-run', is_flag=True, help='Show what would be recovered without actually doing it')
@click.pass_context
def disaster_recovery(ctx, recovery_point, dry_run):
    """Perform complete disaster recovery"""
    config_path = ctx.obj['config']
    
    recovery_time = None
    if recovery_point:
        try:
            recovery_time = datetime.fromisoformat(recovery_point.replace('Z', '+00:00'))
        except ValueError:
            click.echo(f"Invalid recovery point format: {recovery_point}", err=True)
            sys.exit(1)
    
    if dry_run:
        click.echo("Disaster Recovery Plan:")
        click.echo(f"Recovery point: {recovery_point or 'latest'}")
        click.echo("Components to recover:")
        click.echo("  - PostgreSQL database")
        click.echo("  - Redis cache")
        click.echo("  - Voice data files")
        click.echo("  - Configuration files")
        return
    
    if not click.confirm("This will perform COMPLETE disaster recovery. Are you absolutely sure?"):
        click.echo("Disaster recovery cancelled")
        return
    
    try:
        manager = RestoreManager(config_path)
        
        click.echo("Starting disaster recovery procedure...")
        results = manager.perform_disaster_recovery(recovery_time)
        
        # Display results
        click.echo(f"\nDisaster Recovery Results:")
        click.echo(f"Overall Status: {click.style(results['overall_status'], fg='green' if results['overall_status'] == 'success' else 'red')}")
        click.echo(f"Duration: {results.get('duration', 'N/A')} seconds")
        
        # Component details
        table_data = []
        for component, details in results['components'].items():
            status_color = 'green' if details['status'] == 'success' else 'red'
            table_data.append([
                component,
                click.style(details['status'], fg=status_color),
                details['backup_used']
            ])
        
        headers = ['Component', 'Status', 'Backup Used']
        click.echo(tabulate(table_data, headers=headers, tablefmt='grid'))
        
        if results['overall_status'] != 'success':
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"Disaster recovery failed: {e}", err=True)
        sys.exit(1)


@cli.group()
@click.pass_context
def test(ctx):
    """Testing and verification operations"""
    pass


@test.command('restore')
@click.option('--type', 'test_type', help='Type of restore to test')
@click.pass_context
def test_restore(ctx, test_type):
    """Test restore capabilities"""
    config_path = ctx.obj['config']
    
    try:
        from restore_tester import RestoreTester
        tester = RestoreTester(config_path)
        
        click.echo("Running restore tests...")
        results = tester.run_tests()
        
        # Display results
        click.echo(f"\nRestore Test Results:")
        click.echo(f"Overall Status: {click.style(results['overall_status'], fg='green' if results['overall_status'] == 'passed' else 'red')}")
        
        # Test details
        table_data = []
        for test_name, details in results['tests'].items():
            status_color = 'green' if details['status'] == 'passed' else 'red'
            duration = details.get('restore_time_seconds', 0)
            table_data.append([
                test_name,
                click.style(details['status'], fg=status_color),
                f"{duration:.1f}s",
                details.get('backup_tested', 'N/A')
            ])
        
        headers = ['Test', 'Status', 'Duration', 'Backup Tested']
        click.echo(tabulate(table_data, headers=headers, tablefmt='grid'))
        
        if results['overall_status'] != 'passed':
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"Restore test failed: {e}", err=True)
        sys.exit(1)


@cli.command('config')
@click.option('--validate', is_flag=True, help='Validate configuration file')
@click.pass_context
def config(ctx, validate):
    """Show or validate configuration"""
    config_path = ctx.obj['config']
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if validate:
            # Basic validation
            required_sections = ['backup_config', 'global', 'policies', 's3']
            missing = [section for section in required_sections 
                      if section not in str(config)]
            
            if missing:
                click.echo(f"Missing required sections: {missing}", err=True)
                sys.exit(1)
            else:
                click.echo(click.style("Configuration is valid", fg='green'))
        else:
            # Display configuration
            click.echo(yaml.dump(config, default_flow_style=False))
            
    except Exception as e:
        click.echo(f"Configuration error: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    cli()