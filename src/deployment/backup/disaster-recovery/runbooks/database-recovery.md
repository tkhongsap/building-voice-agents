# Database Recovery Runbook

## Overview
This runbook provides step-by-step instructions for recovering the PostgreSQL database in case of failure or data corruption.

## Prerequisites
- Access to backup storage (S3)
- PostgreSQL client tools
- Appropriate database credentials
- Access to Kubernetes cluster (for production)

## Recovery Scenarios

### Scenario 1: Complete Database Failure

**Symptoms:**
- Database is not responding
- Connection timeouts
- Critical alerts firing

**Steps:**

1. **Assess the situation**
   ```bash
   # Check database pod status
   kubectl get pods -n voice-agents-prod -l component=postgres
   
   # Check pod logs
   kubectl logs -n voice-agents-prod postgres-primary-0 --tail=100
   
   # Check persistent volume status
   kubectl get pv,pvc -n voice-agents-prod
   ```

2. **List available backups**
   ```bash
   # Using restore manager
   python /app/scripts/restore_manager.py list postgres
   
   # Or check S3 directly
   aws s3 ls s3://voice-agents-backups/backups/postgres/full/ --recursive | sort -r | head -20
   ```

3. **Choose recovery point**
   - Select the most recent successful backup
   - Note the backup key for restoration

4. **Perform database restore**
   ```bash
   # Scale down applications using the database
   kubectl scale deployment voice-agent --replicas=0 -n voice-agents-prod
   
   # Restore database
   python /app/scripts/restore_manager.py restore-postgres \
     "backups/postgres/full/postgres_full_20240121_020000.sql.gz" \
     voice_agents
   ```

5. **Verify restoration**
   ```bash
   # Connect to database
   psql -h postgres-primary -U agent -d voice_agents
   
   # Check critical tables
   SELECT COUNT(*) FROM voice_sessions;
   SELECT COUNT(*) FROM transcripts;
   SELECT COUNT(*) FROM conversation_history;
   
   # Check latest data
   SELECT MAX(created_at) FROM voice_sessions;
   ```

6. **Resume services**
   ```bash
   # Scale applications back up
   kubectl scale deployment voice-agent --replicas=3 -n voice-agents-prod
   
   # Monitor application logs
   kubectl logs -f deployment/voice-agent -n voice-agents-prod
   ```

### Scenario 2: Data Corruption

**Symptoms:**
- Inconsistent query results
- Application errors related to data integrity
- Checksum failures

**Steps:**

1. **Identify corruption extent**
   ```sql
   -- Check for corruption
   SELECT schemaname, tablename, 
          pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
   FROM pg_tables
   WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
   ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
   
   -- Run integrity checks
   VACUUM ANALYZE;
   ```

2. **Determine recovery approach**
   - If limited to specific tables: Restore only affected tables
   - If widespread: Full database restore

3. **Partial table restore**
   ```bash
   # Extract specific table from backup
   pg_restore -t voice_sessions -d voice_agents_temp backup_file.sql.gz
   
   # Copy data to production
   pg_dump -t voice_sessions voice_agents_temp | psql -d voice_agents
   ```

### Scenario 3: Point-in-Time Recovery

**Use Case:** Need to recover to a specific moment before an erroneous operation

**Steps:**

1. **Identify target recovery time**
   ```bash
   # Find the exact timestamp of the incident
   kubectl logs deployment/voice-agent -n voice-agents-prod --since=2h | grep -i error
   ```

2. **Find appropriate backup and WAL files**
   ```bash
   # List WAL archives around the target time
   aws s3 ls s3://voice-agents-backups/backups/postgres/wal/ \
     --recursive | grep "20240121_14"
   ```

3. **Perform PITR**
   ```bash
   # Restore base backup
   pg_restore -d postgres://agent:password@postgres-recovery:5432/voice_agents \
     base_backup.sql.gz
   
   # Apply WAL files up to target time
   recovery_target_time = '2024-01-21 14:30:00'
   ```

## Validation Checklist

After any recovery operation:

- [ ] Database is accessible
- [ ] All critical tables exist
- [ ] Row counts match expectations
- [ ] Application can connect successfully
- [ ] No error logs in database
- [ ] Monitoring shows healthy metrics
- [ ] Test transactions work correctly
- [ ] Backup jobs are running again

## Rollback Procedure

If recovery fails:

1. **Document current state**
   ```bash
   kubectl describe pods -n voice-agents-prod > recovery_state.log
   kubectl logs postgres-primary-0 -n voice-agents-prod > postgres_recovery.log
   ```

2. **Attempt alternative backup**
   - Use previous backup if latest is corrupted
   - Try cross-region backup if available

3. **Escalation**
   - Contact database team
   - Engage vendor support if needed
   - Consider disaster recovery failover

## Post-Recovery Actions

1. **Root cause analysis**
   - Investigate failure cause
   - Document timeline
   - Identify preventive measures

2. **Update monitoring**
   - Add alerts for detected issues
   - Adjust thresholds if needed

3. **Test backups**
   ```bash
   python /app/scripts/restore_tester.py
   ```

4. **Communication**
   - Update status page
   - Notify stakeholders
   - Document lessons learned

## Contact Information

- On-call DBA: +1-xxx-xxx-xxxx
- Platform Team: platform-oncall@example.com
- Escalation: engineering-leadership@example.com