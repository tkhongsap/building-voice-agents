# Security Infrastructure for Voice Agents Platform

This directory contains comprehensive security scanning and vulnerability management infrastructure for the LiveKit Voice Agents Platform.

## 🔒 Overview

The security infrastructure provides:
- **Container Vulnerability Scanning** with Trivy and Grype
- **Dependency Security Scanning** for Python and Node.js
- **Static Application Security Testing (SAST)** with Semgrep
- **Infrastructure Security Scanning** for Kubernetes and Docker
- **Runtime Security Monitoring** with real-time threat detection
- **Compliance Checking** against security frameworks
- **CI/CD Integration** for automated security gates
- **Comprehensive Alerting** via Slack, email, and PagerDuty

## 🚀 Quick Start

### 1. Deploy Security Infrastructure

```bash
# Full deployment (recommended)
./scripts/deploy_security_infrastructure.sh

# Preview deployment without making changes
./scripts/deploy_security_infrastructure.sh --dry-run

# Partial deployment (skip certain components)
./scripts/deploy_security_infrastructure.sh --skip-docker --skip-monitoring
```

### 2. Run Security Scan

```bash
# Comprehensive security scan
python3 scripts/security_scanner.py --project-path /path/to/project

# Specific scan types
python3 scripts/security_scanner.py --scan-types container dependency sast

# View results
ls -la /security/reports/
```

### 3. Start Real-time Monitoring

```bash
# Start security monitoring daemon
python3 scripts/security_monitor.py --config config.yaml

# Or using systemd (if installed)
systemctl start security-monitor
```

## 📁 Directory Structure

```
src/deployment/security/
├── README.md                           # This file
├── config.yaml                         # Main security configuration
├── docker-compose.security.yml         # Docker services for scanning
├── bandit.yaml                        # Bandit SAST configuration
├── semgrep.yaml                       # Semgrep SAST rules
├── trivy.yaml                         # Trivy scanner configuration
├── scripts/                           # Security tools and scripts
│   ├── deploy_security_infrastructure.sh  # Main deployment script
│   ├── security_scanner.py            # Comprehensive security scanner
│   ├── dependency-scan.py             # Dependency vulnerability scanner
│   ├── security_monitor.py            # Real-time security monitoring
│   └── ci_security_integration.py     # CI/CD integration generator
└── monitoring/                        # Monitoring configurations
    └── backup_alerts.yml              # Alert configurations
```

## 🛠️ Components

### 1. Container Security Scanning

**Tools:** Trivy, Grype  
**Purpose:** Scan Docker images and containers for vulnerabilities

```bash
# Manual container scan
trivy image --format json --output report.json your-image:tag

# Comprehensive scan via orchestrator
python3 scripts/security_scanner.py --scan-types container
```

**Features:**
- Multi-layer vulnerability detection
- Secret scanning in container images
- Configuration misconfigurations detection
- SBOM (Software Bill of Materials) generation
- Integration with container registries

### 2. Dependency Security Scanning

**Tools:** Safety, Bandit, pip-audit, npm audit, yarn audit  
**Purpose:** Identify vulnerable dependencies in Python and Node.js projects

```bash
# Manual dependency scan
python3 scripts/dependency-scan.py --project-path .

# Scan specific language
python3 scripts/dependency-scan.py --project-path . --language python
```

**Features:**
- Python package vulnerability detection
- Node.js package vulnerability detection
- License compliance checking
- Automated fix suggestions
- Integration with package managers

### 3. Static Application Security Testing (SAST)

**Tools:** Semgrep, Bandit, ESLint Security  
**Purpose:** Find security vulnerabilities in source code

```bash
# Manual SAST scan
semgrep --config=auto --json --output=report.json .

# Comprehensive SAST via orchestrator
python3 scripts/security_scanner.py --scan-types sast
```

**Features:**
- Multi-language support (Python, JavaScript, TypeScript)
- Custom security rules
- OWASP Top 10 coverage
- False positive reduction
- IDE integration support

### 4. Infrastructure Security Scanning

**Tools:** kube-score, Docker Bench Security  
**Purpose:** Scan infrastructure configurations for security issues

```bash
# Kubernetes security scan
kube-score score --output-format json your-k8s-config.yaml

# Docker security scan
docker run --rm --net host --pid host --cap-add audit_control \
  docker/docker-bench-security
```

**Features:**
- Kubernetes CIS benchmark compliance
- Docker CIS benchmark compliance
- Network policy validation
- RBAC configuration analysis
- Resource limit validation

### 5. Runtime Security Monitoring

**Tools:** Custom monitoring system with Falco integration  
**Purpose:** Real-time threat detection and incident response

```bash
# Start monitoring
python3 scripts/security_monitor.py

# View security events
tail -f /security/logs/security-events.log
```

**Features:**
- File system integrity monitoring
- Network anomaly detection
- Process behavior analysis
- Container runtime monitoring
- Automated threat response

### 6. Compliance and Reporting

**Frameworks:** SOC 2, GDPR, HIPAA, CIS Benchmarks  
**Purpose:** Ensure compliance with security standards

```bash
# Generate compliance report
python3 scripts/security_scanner.py --compliance-report

# View compliance status
cat /security/reports/compliance-status.json
```

**Features:**
- Multi-framework compliance checking
- Automated compliance reporting
- Gap analysis and recommendations
- Audit trail maintenance
- Policy enforcement

## ⚙️ Configuration

### Main Configuration (`config.yaml`)

The main configuration file controls all security scanning and monitoring behavior:

```yaml
security:
  global:
    scan_frequency: "daily"
    severity_threshold: "HIGH"
    notifications_enabled: true
    
  container_scanning:
    enabled: true
    scanners:
      trivy:
        enabled: true
        severity_levels: ["HIGH", "CRITICAL"]
        
  dependency_scanning:
    enabled: true
    python:
      safety:
        enabled: true
      bandit:
        enabled: true
        
  sast:
    enabled: true
    semgrep:
      enabled: true
      rules: ["security", "owasp-top-ten"]
      
  runtime_security:
    enabled: true
    file_monitoring:
      enabled: true
    network_monitoring:
      enabled: true
      
  alerting:
    enabled: true
    channels:
      slack:
        enabled: true
        webhook_url: "${SLACK_WEBHOOK_URL}"
      email:
        enabled: true
        recipients: ["security@company.com"]
```

### Environment Variables

```bash
# Required for notifications
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
export SMTP_SERVER="smtp.company.com"
export SMTP_USERNAME="alerts@company.com"
export SMTP_PASSWORD="password"

# Optional for enhanced features
export PAGERDUTY_KEY="your-pagerduty-integration-key"
export GITHUB_TOKEN="ghp_your-github-token"
```

## 🔧 CI/CD Integration

### GitHub Actions

Generate GitHub Actions workflow:

```bash
python3 scripts/ci_security_integration.py --platform github_actions --output-dir .
```

This creates `.github/workflows/security.yml` with:
- Automated security scanning on push/PR
- Security gate checks
- SARIF upload to GitHub Security tab
- Artifact preservation
- Notification integration

### GitLab CI

Generate GitLab CI configuration:

```bash
python3 scripts/ci_security_integration.py --platform gitlab_ci --output-dir .
```

This creates `.gitlab-ci-security.yml` with:
- Security scanning stages
- Security report artifacts
- Security gate policies
- Integration with GitLab Security Dashboard

### Jenkins

Generate Jenkins pipeline:

```bash
python3 scripts/ci_security_integration.py --platform jenkins --output-dir .
```

This creates `Jenkinsfile.security` with:
- Parallel security scanning
- Security gate enforcement
- Report publishing
- Notification integration

### Pre-commit Hooks

Generate pre-commit configuration:

```bash
python3 scripts/ci_security_integration.py
# Creates .pre-commit-config.yaml
```

Install pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

## 📊 Monitoring and Alerting

### Dashboard Access

- **Grafana Security Dashboard:** http://localhost:3001
- **Prometheus Metrics:** http://localhost:9091
- **Trivy Server:** http://localhost:4954
- **Security Reports:** `/security/reports/`

### Alert Channels

Configure multiple alert channels based on severity:

1. **Critical/High:** Slack + Email + PagerDuty
2. **Medium:** Slack + Email
3. **Low:** Email only

### Metrics Collected

- Vulnerability counts by severity
- Scan duration and success rates
- Container security metrics
- Dependency freshness metrics
- Compliance status metrics
- Threat detection metrics

## 🚨 Incident Response

### Automated Response Actions

1. **Critical Vulnerabilities:** Automatic Slack alert + PagerDuty incident
2. **Malicious Activity:** Process termination + Network isolation
3. **Container Threats:** Container quarantine + Security team notification
4. **File Integrity:** Backup restoration + Forensic analysis

### Manual Response Procedures

1. **Investigate Alert:** Review security event details
2. **Assess Impact:** Determine scope and severity
3. **Contain Threat:** Isolate affected systems
4. **Remediate:** Apply fixes and patches
5. **Document:** Record incident details and lessons learned

## 🔍 Troubleshooting

### Common Issues

#### 1. Scan Failures

```bash
# Check tool availability
which trivy semgrep bandit safety

# Verify configuration
python3 scripts/ci_security_integration.py --validate

# Check logs
tail -f /security/logs/scanner.log
```

#### 2. Missing Dependencies

```bash
# Install required tools
./scripts/deploy_security_infrastructure.sh --skip-docker

# Update Python packages
pip3 install --upgrade safety bandit pip-audit semgrep
```

#### 3. Alert Delivery Issues

```bash
# Test Slack webhook
curl -X POST -H 'Content-type: application/json' \
  --data '{"text":"Test alert"}' \
  $SLACK_WEBHOOK_URL

# Check SMTP configuration
python3 -c "
import smtplib
server = smtplib.SMTP('$SMTP_SERVER', 587)
server.starttls()
server.login('$SMTP_USERNAME', '$SMTP_PASSWORD')
print('SMTP connection successful')
"
```

#### 4. Performance Issues

```bash
# Check resource usage
docker stats

# Monitor scan duration
grep "duration" /security/reports/*.json

# Tune scan frequency
# Edit config.yaml to reduce scan frequency
```

### Log Locations

- **Scanner Logs:** `/security/logs/scanner.log`
- **Monitor Logs:** `/security/logs/monitor.log`
- **Security Events:** `/security/events/`
- **Scan Reports:** `/security/reports/`
- **Docker Logs:** `docker-compose -f docker-compose.security.yml logs`

## 🎯 Best Practices

### Security Scanning

1. **Regular Scans:** Enable daily automated scans
2. **CI/CD Integration:** Fail builds on critical vulnerabilities
3. **Baseline Management:** Establish security baselines
4. **False Positive Management:** Maintain ignore lists for confirmed false positives
5. **Tool Diversity:** Use multiple tools for comprehensive coverage

### Monitoring

1. **Baseline Establishment:** Create behavioral baselines
2. **Alert Tuning:** Reduce false positives through tuning
3. **Correlation:** Correlate events across multiple sources
4. **Response Time:** Maintain low response times for critical alerts
5. **Documentation:** Document all security events and responses

### Compliance

1. **Regular Audits:** Conduct periodic compliance audits
2. **Policy Updates:** Keep security policies current
3. **Training:** Provide security training for development teams
4. **Documentation:** Maintain comprehensive security documentation
5. **Incident Response:** Test incident response procedures regularly

## 📚 Additional Resources

### Documentation

- [Security Architecture Guide](../../docs/security-architecture.md)
- [Incident Response Playbook](./incident-response-playbook.md)
- [Compliance Framework Guide](./compliance-guide.md)

### External Resources

- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [CIS Benchmarks](https://www.cisecurity.org/cis-benchmarks/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cybersecurity/framework)
- [Docker Security Best Practices](https://docs.docker.com/engine/security/)
- [Kubernetes Security Best Practices](https://kubernetes.io/docs/concepts/security/)

### Tools Documentation

- [Trivy Documentation](https://aquasecurity.github.io/trivy/)
- [Semgrep Documentation](https://semgrep.dev/docs/)
- [Bandit Documentation](https://bandit.readthedocs.io/)
- [Safety Documentation](https://pyup.io/safety/)

## 🤝 Contributing

To contribute to the security infrastructure:

1. Review existing security policies
2. Test changes in isolated environment
3. Update documentation
4. Submit security-focused pull requests
5. Participate in security reviews

## 📞 Support

For security-related issues:

- **Security Team:** security@company.com
- **Incident Response:** incident-response@company.com
- **Emergency Hotline:** +1-XXX-XXX-XXXX
- **Documentation:** This README and linked resources

---

**⚠️ Security Notice:** This infrastructure contains sensitive security tools and configurations. Ensure appropriate access controls and review security implications before deployment in production environments.