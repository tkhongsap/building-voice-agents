#!/bin/bash
# Script to deploy and manage backup infrastructure for Voice Agents Platform

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKUP_DIR="$(dirname "$SCRIPT_DIR")"
DEPLOYMENT_DIR="$(dirname "$BACKUP_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check AWS CLI (optional but recommended)
    if ! command -v aws &> /dev/null; then
        warning "AWS CLI not found. Consider installing for S3 operations"
    fi
    
    # Check required environment variables
    required_vars=("AWS_ACCESS_KEY_ID" "AWS_SECRET_ACCESS_KEY" "POSTGRES_PASSWORD")
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            error "Required environment variable $var is not set"
            exit 1
        fi
    done
    
    success "Prerequisites check passed"
}

# Setup backup configuration
setup_config() {
    log "Setting up backup configuration..."
    
    cd "$BACKUP_DIR"
    
    # Check if config exists
    if [[ ! -f "backup_config.yaml" ]]; then
        if [[ -f "backup_config.yaml.example" ]]; then
            cp backup_config.yaml.example backup_config.yaml
            warning "Created backup_config.yaml from example. Please review and customize."
        else
            error "backup_config.yaml not found and no example available"
            exit 1
        fi
    fi
    
    # Validate configuration
    if command -v python3 &> /dev/null; then
        python3 -c "
import yaml
try:
    with open('backup_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print('Configuration is valid YAML')
except Exception as e:
    print(f'Configuration error: {e}')
    exit(1)
"
    fi
    
    success "Configuration setup complete"
}

# Build backup services
build_services() {
    log "Building backup services..."
    
    cd "$BACKUP_DIR"
    
    # Build backup manager image
    docker build -f Dockerfile.backup -t voice-agents/backup-manager:latest .
    
    success "Backup services built successfully"
}

# Deploy backup infrastructure
deploy_backup() {
    log "Deploying backup infrastructure..."
    
    cd "$BACKUP_DIR"
    
    # Create backup network if it doesn't exist
    if ! docker network ls | grep -q voice-agents-network; then
        docker network create voice-agents-network
        log "Created voice-agents-network"
    fi
    
    # Deploy backup services
    docker-compose -f docker-compose.backup.yml up -d
    
    # Wait for services to start
    log "Waiting for services to start..."
    sleep 30
    
    # Check service health
    check_service_health
    
    success "Backup infrastructure deployed successfully"
}

# Check service health
check_service_health() {
    log "Checking service health..."
    
    services=("backup-manager" "backup-monitor")
    
    for service in "${services[@]}"; do
        if docker ps | grep -q "$service"; then
            if docker exec "$service" python -c "import requests; requests.get('http://localhost:9091/metrics')" &> /dev/null; then
                success "$service is healthy"
            else
                warning "$service is running but health check failed"
            fi
        else
            error "$service is not running"
        fi
    done
}

# Setup monitoring integration
setup_monitoring() {
    log "Setting up monitoring integration..."
    
    # Copy backup alerts to Prometheus
    if [[ -d "$DEPLOYMENT_DIR/docker_config/prometheus" ]]; then
        cp "$BACKUP_DIR/monitoring/backup_alerts.yml" "$DEPLOYMENT_DIR/docker_config/prometheus/"
        success "Backup alerts copied to Prometheus"
        
        # Reload Prometheus configuration
        if docker ps | grep -q prometheus; then
            docker exec prometheus-container kill -HUP 1 || warning "Failed to reload Prometheus config"
        fi
    else
        warning "Prometheus directory not found. Manual integration required."
    fi
    
    # Setup Grafana dashboards
    if [[ -d "$DEPLOYMENT_DIR/docker_config/grafana/dashboards" ]]; then
        if [[ -f "$BACKUP_DIR/monitoring/backup-dashboard.json" ]]; then
            cp "$BACKUP_DIR/monitoring/backup-dashboard.json" "$DEPLOYMENT_DIR/docker_config/grafana/dashboards/"
            success "Backup dashboard copied to Grafana"
        fi
    fi
}

# Test backup functionality
test_backup() {
    log "Testing backup functionality..."
    
    cd "$BACKUP_DIR"
    
    # Test backup manager CLI
    if docker exec backup-manager python /app/scripts/backup_cli.py backup status; then
        success "Backup CLI is working"
    else
        error "Backup CLI test failed"
        return 1
    fi
    
    # Test backup monitor
    if docker exec backup-monitor wget -q -O- http://localhost:9091/metrics | grep -q voice_agent_backup; then
        success "Backup monitor is exposing metrics"
    else
        error "Backup monitor test failed"
        return 1
    fi
    
    # Test S3 connectivity
    if docker exec backup-manager aws s3 ls s3://${S3_BUCKET:-voice-agents-backups}/ &> /dev/null; then
        success "S3 connectivity test passed"
    else
        warning "S3 connectivity test failed - check credentials and bucket"
    fi
}

# Setup Kubernetes CronJobs (if in Kubernetes environment)
setup_kubernetes() {
    log "Setting up Kubernetes backup jobs..."
    
    if command -v kubectl &> /dev/null; then
        cd "$BACKUP_DIR"
        
        # Apply Kubernetes manifests
        if kubectl apply -f kubernetes/backup-cronjob.yaml; then
            success "Kubernetes backup jobs deployed"
        else
            error "Failed to deploy Kubernetes backup jobs"
        fi
    else
        warning "kubectl not found. Skipping Kubernetes setup."
    fi
}

# Show status
show_status() {
    log "Backup Infrastructure Status:"
    echo
    
    # Service status
    echo "=== Service Status ==="
    docker-compose -f "$BACKUP_DIR/docker-compose.backup.yml" ps
    echo
    
    # Backup health
    echo "=== Backup Health ==="
    docker exec backup-manager python /app/scripts/backup_cli.py backup status 2>/dev/null || echo "Unable to get backup status"
    echo
    
    # Storage usage
    echo "=== Storage Usage ==="
    if docker exec backup-manager aws s3 ls s3://${S3_BUCKET:-voice-agents-backups}/ --recursive --summarize 2>/dev/null; then
        echo "S3 storage check complete"
    else
        echo "Unable to check S3 storage"
    fi
    echo
    
    # Recent logs
    echo "=== Recent Logs ==="
    docker logs backup-manager --tail 10 2>/dev/null || echo "Unable to get backup-manager logs"
}

# Cleanup function
cleanup() {
    log "Cleaning up backup infrastructure..."
    
    cd "$BACKUP_DIR"
    
    # Stop services
    docker-compose -f docker-compose.backup.yml down
    
    # Remove images (optional)
    if [[ "${1:-}" == "--remove-images" ]]; then
        docker rmi voice-agents/backup-manager:latest || true
    fi
    
    success "Cleanup complete"
}

# Main function
main() {
    case "${1:-}" in
        "deploy"|"start")
            check_prerequisites
            setup_config
            build_services
            deploy_backup
            setup_monitoring
            test_backup
            show_status
            ;;
        "stop")
            cd "$BACKUP_DIR"
            docker-compose -f docker-compose.backup.yml stop
            success "Backup services stopped"
            ;;
        "restart")
            cd "$BACKUP_DIR"
            docker-compose -f docker-compose.backup.yml restart
            success "Backup services restarted"
            ;;
        "status")
            show_status
            ;;
        "test")
            test_backup
            ;;
        "cleanup")
            cleanup "${2:-}"
            ;;
        "k8s"|"kubernetes")
            setup_kubernetes
            ;;
        "logs")
            service="${2:-backup-manager}"
            docker logs "$service" -f
            ;;
        "shell")
            service="${2:-backup-manager}"
            docker exec -it "$service" /bin/bash
            ;;
        *)
            echo "Usage: $0 {deploy|start|stop|restart|status|test|cleanup|k8s|logs|shell}"
            echo ""
            echo "Commands:"
            echo "  deploy/start  - Deploy backup infrastructure"
            echo "  stop          - Stop backup services"
            echo "  restart       - Restart backup services"
            echo "  status        - Show backup system status"
            echo "  test          - Test backup functionality"
            echo "  cleanup       - Clean up backup infrastructure"
            echo "  k8s           - Setup Kubernetes CronJobs"
            echo "  logs [service]- Show logs for service"
            echo "  shell [service]- Open shell in service container"
            echo ""
            echo "Environment variables required:"
            echo "  AWS_ACCESS_KEY_ID"
            echo "  AWS_SECRET_ACCESS_KEY"
            echo "  POSTGRES_PASSWORD"
            echo ""
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"