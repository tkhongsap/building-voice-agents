# Complete Disaster Recovery Runbook

## Overview
This runbook provides comprehensive procedures for complete disaster recovery of the LiveKit Voice Agents Platform in case of catastrophic failure affecting multiple systems.

## Emergency Contact Information
- Incident Commander: +1-xxx-xxx-xxxx
- Platform Team Lead: +1-xxx-xxx-xxxx
- Infrastructure Team: +1-xxx-xxx-xxxx
- Business Continuity: +1-xxx-xxx-xxxx

## Disaster Scenarios

### Scenario 1: Complete Data Center Failure

**Triggers:**
- Primary AWS region unavailable
- All services down
- Network connectivity lost

**Recovery Steps:**

#### Phase 1: Assessment (0-15 minutes)

1. **Incident Declaration**
   ```bash
   # Activate incident response
   # Set up war room
   # Notify stakeholders
   ```

2. **Damage Assessment**
   ```bash
   # Check AWS Service Health Dashboard
   curl -s https://status.aws.amazon.com/
   
   # Test connectivity to secondary region
   aws ec2 describe-instances --region us-west-2
   
   # Verify backup storage accessibility
   aws s3 ls s3://voice-agents-backups-west/ --region us-west-2
   ```

3. **Decision Point**
   - If partial failure: Follow component-specific runbooks
   - If complete failure: Proceed with full DR

#### Phase 2: Infrastructure Recovery (15-60 minutes)

1. **Activate Secondary Region**
   ```bash
   # Switch to disaster recovery region
   export AWS_DEFAULT_REGION=us-west-2
   
   # Deploy infrastructure
   cd /path/to/terraform-dr
   terraform init
   terraform plan -var="dr_mode=true"
   terraform apply -auto-approve
   ```

2. **Deploy Kubernetes Cluster**
   ```bash
   # Create EKS cluster in DR region
   eksctl create cluster --config-file=eks-dr-config.yaml
   
   # Apply base manifests
   kubectl apply -k kubernetes/base/
   ```

3. **Verify Network Connectivity**
   ```bash
   # Test internal networking
   kubectl run test-pod --image=busybox --rm -it -- nslookup kubernetes.default
   
   # Test external connectivity
   kubectl run test-pod --image=busybox --rm -it -- wget -qO- http://httpbin.org/ip
   ```

#### Phase 3: Data Recovery (30-90 minutes)

1. **Restore PostgreSQL**
   ```bash
   # Find latest backup before disaster
   python /app/scripts/restore_manager.py list postgres | head -5
   
   # Restore database
   python /app/scripts/restore_manager.py disaster-recovery \
     --recovery-point "2024-01-21T10:00:00Z"
   ```

2. **Restore Redis**
   ```bash
   # Restore Redis from backup
   python /app/scripts/restore_manager.py restore-redis \
     "backups/redis/redis_snapshot_20240121_100000.rdb.gz"
   ```

3. **Restore Voice Data**
   ```bash
   # Restore voice recordings and transcripts
   python /app/scripts/restore_manager.py restore-filesystem \
     "backups/voice_data/voice_data_20240121_030000.tar.zst" \
     "/data/voice"
   ```

4. **Restore Configuration**
   ```bash
   # Restore application configurations
   python /app/scripts/restore_manager.py restore-filesystem \
     "backups/configs/configs_20240121_120000.tar.zst" \
     "/etc/voice-agents"
   ```

#### Phase 4: Service Recovery (60-120 minutes)

1. **Deploy Core Services**
   ```bash
   # Deploy database services
   kubectl apply -f kubernetes/postgres/
   kubectl apply -f kubernetes/redis/
   
   # Wait for databases to be ready
   kubectl wait --for=condition=ready pod -l app=postgres --timeout=300s
   kubectl wait --for=condition=ready pod -l app=redis --timeout=300s
   ```

2. **Deploy Voice Agent Services**
   ```bash
   # Deploy main application
   kubectl apply -f kubernetes/voice-agent/
   
   # Scale to minimum required replicas
   kubectl scale deployment voice-agent --replicas=2
   ```

3. **Deploy Support Services**
   ```bash
   # Deploy monitoring
   kubectl apply -f kubernetes/monitoring/
   
   # Deploy backup services
   kubectl apply -f kubernetes/backup/
   ```

#### Phase 5: Verification (90-150 minutes)

1. **Health Checks**
   ```bash
   # Check all pods are running
   kubectl get pods --all-namespaces | grep -v Running
   
   # Test database connectivity
   kubectl exec -it postgres-0 -- psql -U agent -d voice_agents -c "SELECT version();"
   
   # Test Redis connectivity
   kubectl exec -it redis-0 -- redis-cli ping
   ```

2. **Functional Testing**
   ```bash
   # Test voice agent API
   curl -X POST https://voice-api-dr.example.com/health
   
   # Test session creation
   curl -X POST https://voice-api-dr.example.com/sessions \
     -H "Content-Type: application/json" \
     -d '{"user_id": "test", "language": "en"}'
   
   # Test WebRTC connectivity
   # Use LiveKit test client
   ```

3. **Data Integrity Verification**
   ```bash
   # Verify restored data
   python /app/scripts/verify_restore.py
   
   # Check critical tables row counts
   kubectl exec -it postgres-0 -- psql -U agent -d voice_agents -c "
     SELECT 'voice_sessions' as table_name, COUNT(*) as rows FROM voice_sessions
     UNION ALL
     SELECT 'transcripts', COUNT(*) FROM transcripts
     UNION ALL
     SELECT 'conversation_history', COUNT(*) FROM conversation_history;
   "
   ```

#### Phase 6: Traffic Cutover (120-180 minutes)

1. **DNS Cutover**
   ```bash
   # Update DNS to point to DR region
   aws route53 change-resource-record-sets \
     --hosted-zone-id Z123456789 \
     --change-batch file://dns-cutover.json
   ```

2. **SSL/TLS Verification**
   ```bash
   # Verify SSL certificates
   curl -vI https://voice-api.example.com 2>&1 | grep -i certificate
   
   # Test WebSocket connections
   wscat -c wss://voice-api.example.com/ws
   ```

3. **Load Balancer Configuration**
   ```bash
   # Update load balancer health checks
   kubectl patch service voice-agent-service -p '{"spec":{"externalTrafficPolicy":"Local"}}'
   ```

### Scenario 2: Ransomware Attack

**Immediate Actions:**

1. **Isolation**
   ```bash
   # Immediately isolate affected systems
   kubectl delete networkpolicy --all -n voice-agents-prod
   kubectl apply -f security/emergency-isolation.yaml
   
   # Block external traffic
   aws ec2 revoke-security-group-ingress --group-id sg-12345678 --protocol all
   ```

2. **Assessment**
   ```bash
   # Check file system integrity
   find /data -name "*.encrypted" -o -name "*ransom*" | head -20
   
   # Check backup integrity
   python /app/scripts/backup_monitor.py --verify-all
   ```

3. **Clean Recovery**
   ```bash
   # Wipe and rebuild from clean backups
   python /app/scripts/restore_manager.py disaster-recovery \
     --recovery-point "2024-01-20T23:59:59Z" \
     --clean-slate
   ```

## Recovery Time Objectives (RTO)

| Component | RTO Target | Critical Path |
|-----------|------------|---------------|
| Infrastructure | 60 minutes | AWS resources, networking |
| Database | 90 minutes | PostgreSQL restore |
| Voice Services | 120 minutes | Application deployment |
| Full Service | 180 minutes | End-to-end testing |

## Recovery Point Objectives (RPO)

| Data Type | RPO Target | Backup Frequency |
|-----------|------------|------------------|
| Database | 15 minutes | Continuous WAL |
| Voice Data | 2 hours | Every 2 hours |
| Configuration | 12 hours | Twice daily |
| Logs | 4 hours | Every 4 hours |

## Communication Plan

### Status Updates

1. **Every 30 minutes during active recovery**
   ```bash
   # Post to status page
   curl -X POST https://api.statuspage.io/v1/pages/YOUR_PAGE_ID/incidents \
     -H "Authorization: OAuth YOUR_API_KEY" \
     -d "name=Data Center Failure&status=investigating"
   ```

2. **Key stakeholders notification**
   - CEO/CTO: Immediate
   - Engineering: Within 15 minutes
   - Customer Success: Within 30 minutes
   - All hands: Within 1 hour

### Communication Templates

#### Initial Notification
```
INCIDENT: Complete service outage due to data center failure
STATUS: Recovery in progress
ETA: 3 hours for full restoration
IMPACT: All voice agent services unavailable
NEXT UPDATE: In 30 minutes
```

#### Progress Update
```
INCIDENT UPDATE: Infrastructure restored, data recovery 60% complete
STATUS: On track
ETA: 90 minutes remaining
PROGRESS: Database restored, voice services deploying
NEXT UPDATE: In 30 minutes
```

#### Resolution
```
INCIDENT RESOLVED: All services restored and operational
DURATION: 2 hours 45 minutes
ROOT CAUSE: AWS region-wide outage
FOLLOW-UP: Post-incident review scheduled for tomorrow 2 PM
```

## Post-Recovery Checklist

### Immediate (0-24 hours)
- [ ] Verify all services operational
- [ ] Resume backup jobs
- [ ] Update monitoring dashboards
- [ ] Document recovery timeline
- [ ] Customer communication
- [ ] Stakeholder debrief

### Short-term (1-7 days)
- [ ] Post-incident review
- [ ] Update runbooks based on lessons learned
- [ ] Test backup integrity
- [ ] Performance optimization
- [ ] Security review
- [ ] Process improvements

### Long-term (1-4 weeks)
- [ ] Implement identified improvements
- [ ] Update disaster recovery plan
- [ ] Conduct tabletop exercises
- [ ] Review insurance claims
- [ ] Capacity planning updates
- [ ] Staff training updates

## Validation Scripts

```bash
#!/bin/bash
# complete-dr-validation.sh

echo "Starting complete disaster recovery validation..."

# Test database
kubectl exec -it postgres-0 -- psql -U agent -d voice_agents -c "SELECT COUNT(*) FROM voice_sessions;" || exit 1

# Test Redis
kubectl exec -it redis-0 -- redis-cli ping || exit 1

# Test API
curl -f https://voice-api.example.com/health || exit 1

# Test WebRTC
python /app/scripts/test_webrtc.py || exit 1

# Test monitoring
curl -f http://prometheus:9090/-/healthy || exit 1

echo "All validation tests passed!"
```

## Emergency Procedures Override

In extreme situations where standard procedures fail:

1. **Manual Infrastructure Deployment**
   - Use pre-configured AMIs
   - Deploy minimal viable service
   - Focus on data recovery first

2. **Alternative Backup Sources**
   - Check cross-region replicas
   - Use offline backup tapes if available
   - Consider third-party backup services

3. **Emergency Contacts**
   - AWS Enterprise Support: +1-xxx-xxx-xxxx
   - LiveKit Support: support@livekit.io
   - Security Team: security@example.com