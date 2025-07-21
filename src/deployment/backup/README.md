# Backup and Disaster Recovery System

## Overview

This directory contains a comprehensive backup and disaster recovery (B&DR) system for the LiveKit Voice Agents Platform. The system provides automated backups, monitoring, alerting, and recovery capabilities to ensure business continuity and data protection.

## Features

### 🔄 Automated Backups
- **PostgreSQL Database**: Full and incremental backups with WAL archiving
- **Redis Cache**: RDB snapshots and AOF backups
- **Voice Data**: Audio recordings, transcripts, and conversation history
- **Application Logs**: Structured logging data for debugging and auditing
- **Configuration Files**: System and application configurations
- **LiveKit Sessions**: WebRTC session data and recordings
- **Monitoring Data**: Prometheus metrics and Grafana dashboards

### 🚨 Monitoring & Alerting
- Real-time backup health monitoring
- Backup verification and integrity checks
- Storage usage monitoring
- Cross-region replication monitoring
- Prometheus metrics integration
- Multi-channel alerting (Slack, Email, PagerDuty)

### 🔧 Disaster Recovery
- Complete disaster recovery procedures
- Point-in-time recovery capabilities
- Automated restore testing
- Cross-region failover support
- Recovery time/point objective tracking

### 🛠️ Management Tools
- Command-line interface for all operations
- Web dashboard for monitoring
- Automated restore testing
- Recovery runbooks and procedures

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │  Backup System  │    │    Storage      │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ PostgreSQL      │───▶│ Backup Manager  │───▶│ S3 Primary      │
│ Redis           │    │ Monitor         │    │ S3 Cross-Region │
│ Voice Data      │    │ Scheduler       │    │ Local Cache     │
│ Logs            │    │ Verifier        │    └─────────────────┘
│ Configs         │    └─────────────────┘
└─────────────────┘             │
                                ▼
                    ┌─────────────────┐
                    │   Alerting      │
                    ├─────────────────┤
                    │ Slack           │
                    │ Email           │
                    │ PagerDuty       │
                    │ Prometheus      │
                    └─────────────────┘
```

## Quick Start

### Prerequisites

1. **AWS Account** with S3 access
2. **Docker** and Docker Compose
3. **Kubernetes** cluster (for production)
4. **Python 3.11+** with required packages

### Installation

1. **Clone and setup**:
   ```bash
   cd src/deployment/backup
   pip install -r requirements.backup.txt
   ```

2. **Configure environment**:
   ```bash
   # Copy and edit configuration
   cp backup_config.yaml.example backup_config.yaml
   
   # Set environment variables
   export AWS_ACCESS_KEY_ID=your_access_key
   export AWS_SECRET_ACCESS_KEY=your_secret_key
   export POSTGRES_PASSWORD=your_db_password
   ```

3. **Start backup services**:
   ```bash
   # For development/testing
   docker-compose -f docker-compose.backup.yml up -d
   
   # For production (Kubernetes)
   kubectl apply -f kubernetes/backup-cronjob.yaml
   ```

### Basic Usage

**CLI Commands**:
```bash
# Check backup status
python scripts/backup_cli.py backup status

# List available backups
python scripts/backup_cli.py backup list --type postgres

# Run manual backup
python scripts/backup_cli.py backup run --type postgres

# Restore from backup
python scripts/backup_cli.py restore postgres backup_key

# Test restore capability
python scripts/backup_cli.py test restore

# Full disaster recovery
python scripts/backup_cli.py disaster recovery
```

## Configuration

### Backup Policies

The system supports multiple backup policies defined in `backup_config.yaml`:

```yaml
policies:
  postgres:
    schedule:
      full_backup: "0 2 * * *"     # Daily at 2 AM
      incremental: "0 */4 * * *"   # Every 4 hours
    retention:
      full_backups: 30             # Keep 30 days
      incremental: 7               # Keep 7 days
  
  redis:
    schedule:
      snapshot: "0 */2 * * *"      # Every 2 hours
    retention:
      snapshots: 7                 # Keep 7 days
```

### Storage Configuration

```yaml
s3:
  bucket: voice-agents-backups
  region: us-east-1
  storage_class: STANDARD_IA
  encryption: AES256
  lifecycle_rules:
    - id: transition-to-glacier
      days: 30
      storage_class: GLACIER
```

### Alerting Configuration

```yaml
notification_channels:
  - type: slack
    webhook_url: ${SLACK_WEBHOOK_URL}
  - type: email
    recipients:
      - ops-team@example.com
  - type: pagerduty
    integration_key: ${PAGERDUTY_KEY}
```

## Monitoring

### Prometheus Metrics

The system exports various metrics:

- `voice_agent_backup_total`: Total number of backups
- `voice_agent_backup_duration_seconds`: Backup duration
- `voice_agent_backup_size_bytes`: Backup size
- `voice_agent_backup_last_success_timestamp`: Last successful backup
- `voice_agent_backup_verification_status`: Verification status

### Grafana Dashboards

Pre-built dashboards for:
- Backup performance and success rates
- Storage usage trends
- Recovery time metrics
- Alert status overview

### Health Checks

Regular health checks monitor:
- Backup job success/failure rates
- Backup age and freshness
- Storage capacity and usage
- Cross-region replication lag
- Restore test results

## Disaster Recovery

### Recovery Objectives

| Component | RTO | RPO |
|-----------|-----|-----|
| Database  | 60 min | 15 min |
| Cache     | 30 min | 2 hours |
| Voice Data| 90 min | 6 hours |
| Full System| 3 hours | 30 min |

### Recovery Procedures

1. **Assessment Phase** (0-15 min)
   - Determine scope of failure
   - Activate incident response
   - Choose recovery strategy

2. **Recovery Phase** (15-120 min)
   - Deploy infrastructure
   - Restore data components
   - Verify system integrity

3. **Validation Phase** (90-180 min)
   - Test all services
   - Verify data consistency
   - Switch traffic to recovered system

### Runbooks

Detailed runbooks are provided for:
- [Database Recovery](disaster-recovery/runbooks/database-recovery.md)
- [Complete Disaster Recovery](disaster-recovery/runbooks/complete-disaster-recovery.md)
- [Network Failure Recovery](disaster-recovery/runbooks/network-recovery.md)
- [Security Incident Response](disaster-recovery/runbooks/security-incident.md)

## Testing

### Automated Testing

The system includes automated restore testing:

```bash
# Run restore tests
python scripts/restore_tester.py

# View test history
python scripts/backup_cli.py test history
```

### Manual Testing

Regular manual tests should include:
- Full disaster recovery drills
- Cross-region failover tests
- Performance benchmarks
- Security assessments

## Security

### Data Protection

- **Encryption at rest**: All backups encrypted with AES-256
- **Encryption in transit**: TLS 1.3 for all data transfers
- **Access control**: IAM roles with least privilege
- **Audit logging**: All operations logged and monitored

### Compliance

The system supports compliance with:
- SOC 2 Type II
- GDPR data protection
- HIPAA requirements
- PCI DSS standards

## Troubleshooting

### Common Issues

**Backup Failures**:
```bash
# Check backup logs
kubectl logs -l component=backup -n voice-agents-prod

# Verify credentials
aws s3 ls s3://voice-agents-backups/

# Test database connectivity
kubectl exec -it postgres-0 -- psql -U agent -c "SELECT 1"
```

**Storage Issues**:
```bash
# Check storage usage
python scripts/backup_cli.py backup status

# Clean old backups
aws s3 rm s3://voice-agents-backups/old/ --recursive

# Check retention policies
aws s3api get-bucket-lifecycle-configuration --bucket voice-agents-backups
```

**Recovery Problems**:
```bash
# Verify backup integrity
python scripts/backup_cli.py backup verify --backup-key key

# Test restore process
python scripts/backup_cli.py restore postgres key --dry-run

# Check restore logs
tail -f /var/log/restore.log
```

### Performance Optimization

- Use parallel backup jobs for large datasets
- Implement incremental backups for frequently changing data
- Optimize compression settings based on data type
- Use cross-region replication for faster recovery

## API Reference

### REST API Endpoints

```bash
# Backup status
GET /api/v1/backup/status

# Trigger backup
POST /api/v1/backup/run
{
  "type": "postgres",
  "force": false
}

# List backups
GET /api/v1/backup/list?type=postgres&limit=10

# Restore from backup
POST /api/v1/restore/postgres
{
  "backup_key": "backup_file.sql.gz",
  "target_db": "voice_agents"
}
```

### CLI Reference

```bash
# Backup operations
backup run [--type TYPE] [--force]
backup list [--type TYPE] [--limit N]
backup status
backup verify [--backup-key KEY]

# Restore operations
restore postgres BACKUP_KEY [--target-db DB]
restore redis BACKUP_KEY
restore filesystem BACKUP_KEY TARGET_PATH

# Disaster recovery
disaster recovery [--recovery-point TIME] [--dry-run]

# Testing
test restore [--type TYPE]
test dr-drill
```

## Support

For issues and questions:

- **Documentation**: Check runbooks and troubleshooting guides
- **Monitoring**: Review Grafana dashboards and Prometheus alerts
- **Logs**: Check application and system logs
- **Support**: Contact the platform team or file an issue

## Contributing

1. Review the system architecture and configuration
2. Test changes in development environment
3. Update documentation and runbooks
4. Add appropriate monitoring and alerts
5. Submit pull request with detailed description

## License

This backup and disaster recovery system is part of the LiveKit Voice Agents Platform and follows the same licensing terms.