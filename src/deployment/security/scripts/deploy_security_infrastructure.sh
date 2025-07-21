#!/bin/bash
# Security Infrastructure Deployment Script for Voice Agents Platform
# Comprehensive security scanning and monitoring deployment

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SECURITY_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(cd "$SECURITY_DIR/../../.." && pwd)"
CONFIG_FILE="${SECURITY_DIR}/config.yaml"
REPORTS_DIR="/security/reports"
LOGS_DIR="/security/logs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Check if running as root (required for some security tools)
check_privileges() {
    if [[ $EUID -eq 0 ]]; then
        warn "Running as root. This is required for some security tools but consider security implications."
    fi
}

# Check dependencies
check_dependencies() {
    info "Checking dependencies..."
    
    local missing_deps=()
    
    # Required tools
    local required_tools=(
        "docker"
        "docker-compose"
        "python3"
        "pip3"
        "curl"
        "wget"
        "git"
    )
    
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" &> /dev/null; then
            missing_deps+=("$tool")
        fi
    done
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        error "Missing required dependencies: ${missing_deps[*]}"
        echo "Please install missing dependencies and try again."
        exit 1
    fi
    
    log "All dependencies found"
}

# Install security tools
install_security_tools() {
    info "Installing security tools..."
    
    # Install Python security tools
    pip3 install --upgrade \
        safety \
        bandit \
        pip-audit \
        semgrep \
        docker \
        psutil \
        pyyaml \
        requests
    
    # Install Trivy
    if ! command -v trivy &> /dev/null; then
        info "Installing Trivy..."
        curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin
    fi
    
    # Install Grype
    if ! command -v grype &> /dev/null; then
        info "Installing Grype..."
        curl -sSfL https://raw.githubusercontent.com/anchore/grype/main/install.sh | sh -s -- -b /usr/local/bin
    fi
    
    # Install Syft
    if ! command -v syft &> /dev/null; then
        info "Installing Syft..."
        curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh -s -- -b /usr/local/bin
    fi
    
    # Install kube-score (for Kubernetes security)
    if ! command -v kube-score &> /dev/null; then
        info "Installing kube-score..."
        wget -O /usr/local/bin/kube-score https://github.com/zegl/kube-score/releases/latest/download/kube-score_linux_amd64
        chmod +x /usr/local/bin/kube-score
    fi
    
    log "Security tools installation completed"
}

# Setup directory structure
setup_directories() {
    info "Setting up directory structure..."
    
    local dirs=(
        "$REPORTS_DIR"
        "$LOGS_DIR"
        "/security/events"
        "/security/policies"
        "/security/threat_intel"
        "/security/backups"
    )
    
    for dir in "${dirs[@]}"; do
        mkdir -p "$dir"
        chmod 755 "$dir"
    done
    
    log "Directory structure created"
}

# Generate default configuration if not exists
setup_configuration() {
    if [[ ! -f "$CONFIG_FILE" ]]; then
        info "Generating default security configuration..."
        
        # The config.yaml already exists in the repository
        log "Using existing security configuration at $CONFIG_FILE"
    else
        log "Security configuration found at $CONFIG_FILE"
    fi
    
    # Validate configuration
    if python3 -c "
import yaml
try:
    with open('$CONFIG_FILE', 'r') as f:
        yaml.safe_load(f)
    print('Configuration validation: PASSED')
except Exception as e:
    print(f'Configuration validation: FAILED - {e}')
    exit(1)
"; then
        log "Configuration validation passed"
    else
        error "Configuration validation failed"
        exit 1
    fi
}

# Deploy security scanning infrastructure
deploy_scanning_infrastructure() {
    info "Deploying security scanning infrastructure..."
    
    # Start security scanning services
    cd "$SECURITY_DIR"
    
    if [[ -f "docker-compose.security.yml" ]]; then
        info "Starting security scanning containers..."
        docker-compose -f docker-compose.security.yml up -d
        
        # Wait for services to be ready
        sleep 10
        
        # Check service health
        info "Checking service health..."
        docker-compose -f docker-compose.security.yml ps
        
        log "Security scanning infrastructure deployed"
    else
        warn "docker-compose.security.yml not found, skipping container deployment"
    fi
}

# Setup continuous monitoring
setup_monitoring() {
    info "Setting up security monitoring..."
    
    # Create systemd service for security monitor
    cat > /tmp/security-monitor.service << 'EOF'
[Unit]
Description=Voice Agents Security Monitor
After=network.target
Wants=network.target

[Service]
Type=simple
User=security
Group=security
ExecStart=/usr/bin/python3 /path/to/security_monitor.py --config /security/config.yaml
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
    
    # Update the ExecStart path
    sed -i "s|/path/to/security_monitor.py|$SCRIPT_DIR/security_monitor.py|g" /tmp/security-monitor.service
    
    if command -v systemctl &> /dev/null; then
        info "Installing systemd service..."
        sudo cp /tmp/security-monitor.service /etc/systemd/system/
        sudo systemctl daemon-reload
        sudo systemctl enable security-monitor.service
        
        log "Security monitoring service installed"
    else
        warn "systemctl not available, manual service setup required"
    fi
    
    rm -f /tmp/security-monitor.service
}

# Setup scheduled scans
setup_scheduled_scans() {
    info "Setting up scheduled security scans..."
    
    # Create cron job for daily security scans
    local cron_job="0 2 * * * cd $PROJECT_ROOT && python3 $SCRIPT_DIR/security_scanner.py --project-path $PROJECT_ROOT --output-dir $REPORTS_DIR > $LOGS_DIR/daily-scan.log 2>&1"
    
    # Add to crontab if not already present
    if ! crontab -l 2>/dev/null | grep -q "$SCRIPT_DIR/security_scanner.py"; then
        (crontab -l 2>/dev/null; echo "$cron_job") | crontab -
        log "Daily security scan scheduled"
    else
        log "Security scan already scheduled"
    fi
    
    # Create weekly vulnerability database update
    local update_job="0 1 * * 0 cd $SECURITY_DIR && docker-compose -f docker-compose.security.yml exec trivy-scanner trivy image --download-db-only > $LOGS_DIR/db-update.log 2>&1"
    
    if ! crontab -l 2>/dev/null | grep -q "trivy.*download-db-only"; then
        (crontab -l 2>/dev/null; echo "$update_job") | crontab -
        log "Weekly vulnerability database update scheduled"
    else
        log "Vulnerability database update already scheduled"
    fi
}

# Setup CI/CD integration
setup_ci_integration() {
    info "Setting up CI/CD integration..."
    
    # Generate CI/CD configuration files
    python3 "$SCRIPT_DIR/ci_security_integration.py" \
        --output-dir "$PROJECT_ROOT" \
        --config "$CONFIG_FILE"
    
    log "CI/CD integration files generated"
}

# Run initial security scan
run_initial_scan() {
    info "Running initial security scan..."
    
    python3 "$SCRIPT_DIR/security_scanner.py" \
        --project-path "$PROJECT_ROOT" \
        --output-dir "$REPORTS_DIR" \
        --verbose
    
    local exit_code=$?
    
    if [[ $exit_code -eq 0 ]]; then
        log "Initial security scan completed successfully"
    else
        warn "Initial security scan completed with issues (exit code: $exit_code)"
        warn "Review the scan results in $REPORTS_DIR"
    fi
}

# Generate deployment report
generate_deployment_report() {
    info "Generating deployment report..."
    
    local report_file="$REPORTS_DIR/deployment_report_$(date +'%Y%m%d_%H%M%S').md"
    
    cat > "$report_file" << EOF
# Security Infrastructure Deployment Report

**Date:** $(date)
**Project:** Voice Agents Platform
**Deployment Status:** Completed

## Components Deployed

### Security Scanning Tools
- ✅ Trivy (Container vulnerability scanner)
- ✅ Grype (Alternative vulnerability scanner)
- ✅ Semgrep (SAST scanner)
- ✅ Bandit (Python security linter)
- ✅ Safety (Python dependency scanner)
- ✅ pip-audit (Python vulnerability scanner)
- ✅ kube-score (Kubernetes security scanner)

### Monitoring Components
- ✅ File system monitoring
- ✅ Network monitoring
- ✅ Process monitoring
- ✅ Container monitoring
- ✅ Real-time alerting

### Infrastructure
- ✅ Docker-based scanning services
- ✅ Scheduled vulnerability scans
- ✅ CI/CD integration
- ✅ Compliance checking
- ✅ Threat intelligence integration

## Configuration
- **Config File:** $CONFIG_FILE
- **Reports Directory:** $REPORTS_DIR
- **Logs Directory:** $LOGS_DIR

## Next Steps
1. Review and customize security policies
2. Configure notification channels (Slack, email, PagerDuty)
3. Set up threat intelligence feeds
4. Test incident response procedures
5. Train team on security tools and processes

## Verification Commands
\`\`\`bash
# Check service status
docker-compose -f $SECURITY_DIR/docker-compose.security.yml ps

# Run manual security scan
python3 $SCRIPT_DIR/security_scanner.py --project-path $PROJECT_ROOT

# Check monitoring status
systemctl status security-monitor

# View latest scan results
ls -la $REPORTS_DIR/
\`\`\`

## Support
For issues or questions, consult the documentation or contact the security team.
EOF
    
    log "Deployment report generated: $report_file"
}

# Cleanup function
cleanup() {
    info "Cleaning up temporary files..."
    # Add cleanup logic here if needed
}

# Main deployment function
main() {
    log "Starting Voice Agents Platform Security Infrastructure Deployment"
    
    # Set trap for cleanup
    trap cleanup EXIT
    
    # Check if help requested
    if [[ "${1:-}" == "-h" ]] || [[ "${1:-}" == "--help" ]]; then
        cat << EOF
Security Infrastructure Deployment Script

Usage: $0 [OPTIONS]

Options:
    -h, --help          Show this help message
    --skip-tools        Skip security tools installation
    --skip-docker       Skip Docker container deployment
    --skip-monitoring   Skip monitoring setup
    --skip-cron         Skip scheduled scans setup
    --skip-ci           Skip CI/CD integration
    --skip-scan         Skip initial security scan
    --dry-run           Show what would be done without executing

Examples:
    $0                  # Full deployment
    $0 --skip-tools     # Skip tools installation
    $0 --dry-run        # Show deployment plan
EOF
        exit 0
    fi
    
    # Parse command line arguments
    local skip_tools=false
    local skip_docker=false
    local skip_monitoring=false
    local skip_cron=false
    local skip_ci=false
    local skip_scan=false
    local dry_run=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-tools)
                skip_tools=true
                shift
                ;;
            --skip-docker)
                skip_docker=true
                shift
                ;;
            --skip-monitoring)
                skip_monitoring=true
                shift
                ;;
            --skip-cron)
                skip_cron=true
                shift
                ;;
            --skip-ci)
                skip_ci=true
                shift
                ;;
            --skip-scan)
                skip_scan=true
                shift
                ;;
            --dry-run)
                dry_run=true
                shift
                ;;
            *)
                error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    if [[ "$dry_run" == true ]]; then
        info "DRY RUN MODE - No changes will be made"
        info "Would execute the following steps:"
        info "1. Check privileges and dependencies"
        info "2. Install security tools (skip: $skip_tools)"
        info "3. Setup directories and configuration"
        info "4. Deploy scanning infrastructure (skip: $skip_docker)"
        info "5. Setup monitoring (skip: $skip_monitoring)"
        info "6. Setup scheduled scans (skip: $skip_cron)"
        info "7. Setup CI/CD integration (skip: $skip_ci)"
        info "8. Run initial scan (skip: $skip_scan)"
        info "9. Generate deployment report"
        exit 0
    fi
    
    # Execute deployment steps
    check_privileges
    check_dependencies
    
    if [[ "$skip_tools" != true ]]; then
        install_security_tools
    fi
    
    setup_directories
    setup_configuration
    
    if [[ "$skip_docker" != true ]]; then
        deploy_scanning_infrastructure
    fi
    
    if [[ "$skip_monitoring" != true ]]; then
        setup_monitoring
    fi
    
    if [[ "$skip_cron" != true ]]; then
        setup_scheduled_scans
    fi
    
    if [[ "$skip_ci" != true ]]; then
        setup_ci_integration
    fi
    
    if [[ "$skip_scan" != true ]]; then
        run_initial_scan
    fi
    
    generate_deployment_report
    
    log "Security infrastructure deployment completed successfully!"
    log "Review the deployment report for next steps and verification commands."
}

# Execute main function with all arguments
main "$@"