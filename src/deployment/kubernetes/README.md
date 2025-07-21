# Kubernetes Deployment Guide for LiveKit Voice Agents Platform

This directory contains comprehensive Kubernetes manifests for deploying the LiveKit Voice Agents Platform in production-ready environments.

## 📁 File Structure

```
kubernetes/
├── 00-namespaces.yaml           # Environment namespaces (dev, staging, prod, monitoring)
├── 01-configmaps.yaml           # Application configuration and NGINX config
├── 02-secrets.yaml              # Sensitive data (API keys, passwords, certificates)
├── 03-storage.yaml              # PersistentVolumes and StorageClasses
├── 04-voice-agent-deployment.yaml  # Main application deployments
├── 05-supporting-services.yaml  # PostgreSQL, Redis, LiveKit, NGINX deployments
├── 06-services.yaml             # Service discovery and load balancing
├── 07-ingress.yaml              # External access with SSL/TLS termination
├── 08-autoscaling.yaml          # HPA, VPA, and PodDisruptionBudgets
├── 09-rbac.yaml                 # ServiceAccounts, Roles, and RoleBindings
├── 10-network-policies.yaml     # Network security micro-segmentation
├── 11-monitoring.yaml           # Prometheus, Grafana, Jaeger observability
└── README.md                    # This deployment guide
```

## 🚀 Quick Start

### Prerequisites

1. **Kubernetes Cluster** (v1.24+)
2. **kubectl** configured with cluster access
3. **NGINX Ingress Controller** installed
4. **cert-manager** for SSL certificates (optional but recommended)
5. **Metrics Server** for HPA functionality
6. **Storage Classes** configured for your cloud provider

### Basic Deployment (Development)

```bash
# 1. Apply namespaces
kubectl apply -f 00-namespaces.yaml

# 2. Apply storage configurations
kubectl apply -f 03-storage.yaml

# 3. Apply ConfigMaps and Secrets
kubectl apply -f 01-configmaps.yaml
kubectl apply -f 02-secrets.yaml

# 4. Apply RBAC
kubectl apply -f 09-rbac.yaml

# 5. Deploy supporting services
kubectl apply -f 05-supporting-services.yaml

# 6. Deploy main application
kubectl apply -f 04-voice-agent-deployment.yaml

# 7. Apply services
kubectl apply -f 06-services.yaml

# 8. Apply ingress (configure domains first)
kubectl apply -f 07-ingress.yaml

# 9. Apply monitoring
kubectl apply -f 11-monitoring.yaml

# 10. Apply autoscaling and network policies
kubectl apply -f 08-autoscaling.yaml
kubectl apply -f 10-network-policies.yaml
```

### Production Deployment

```bash
# Deploy to production namespace
kubectl apply -f 00-namespaces.yaml

# Update secrets with production values
kubectl create secret generic voice-agents-secrets \
  --namespace=voice-agents-prod \
  --from-literal=POSTGRES_PASSWORD=<secure-password> \
  --from-literal=LIVEKIT_API_KEY=<livekit-key> \
  --from-literal=LIVEKIT_API_SECRET=<livekit-secret> \
  --from-literal=OPENAI_API_KEY=<openai-key> \
  --from-literal=ANTHROPIC_API_KEY=<anthropic-key> \
  --from-literal=JWT_SECRET=<jwt-secret>

# Deploy infrastructure
kubectl apply -f 03-storage.yaml
kubectl apply -f 01-configmaps.yaml
kubectl apply -f 09-rbac.yaml

# Deploy services
kubectl apply -f 05-supporting-services.yaml
kubectl apply -f 04-voice-agent-deployment.yaml
kubectl apply -f 06-services.yaml

# Configure and deploy ingress
# Update domain names in 07-ingress.yaml first
kubectl apply -f 07-ingress.yaml

# Deploy monitoring and policies
kubectl apply -f 11-monitoring.yaml
kubectl apply -f 08-autoscaling.yaml
kubectl apply -f 10-network-policies.yaml
```

## 🏗️ Architecture Overview

### Components

- **Voice Agent**: Main application pods with auto-scaling
- **PostgreSQL**: Primary database for persistent data
- **Redis**: Caching and session management
- **LiveKit**: WebRTC server for real-time communication
- **NGINX**: Load balancer and reverse proxy
- **Prometheus**: Metrics collection and alerting
- **Grafana**: Monitoring dashboards
- **Jaeger**: Distributed tracing

### Environments

- **Development**: Single replica, relaxed security, local storage
- **Staging**: Multi-replica, production-like configuration
- **Production**: High availability, strict security, performance optimized

## 🔧 Configuration

### Environment Variables

Key configuration is managed through ConfigMaps and Secrets:

#### ConfigMap Variables
- `ENVIRONMENT`: deployment environment (development/staging/production)
- `LOG_LEVEL`: logging verbosity
- `MAX_WORKERS`: worker process count
- Database and service connection details

#### Secret Variables
- `POSTGRES_PASSWORD`: database password
- `LIVEKIT_API_KEY/SECRET`: LiveKit authentication
- `OPENAI_API_KEY`: OpenAI API access
- `ANTHROPIC_API_KEY`: Anthropic API access
- `JWT_SECRET`: JWT token signing key

### Customization

1. **Resource Limits**: Adjust CPU/memory in deployment files
2. **Replica Counts**: Modify based on expected load
3. **Storage Sizes**: Update PVC sizes in storage.yaml
4. **Domain Names**: Configure in ingress.yaml
5. **Auto-scaling**: Tune HPA thresholds in autoscaling.yaml

## 📊 Monitoring & Observability

### Access Monitoring Tools

```bash
# Port forward to access locally
kubectl port-forward -n voice-agents-monitoring svc/grafana-service 3000:3000
kubectl port-forward -n voice-agents-monitoring svc/prometheus-service 9090:9090
kubectl port-forward -n voice-agents-monitoring svc/jaeger-service 16686:16686
```

### Key Metrics

- **Application Metrics**: Available at `/metrics` endpoint
- **HTTP Request Rate**: Tracked per endpoint
- **Voice Session Count**: Active concurrent sessions
- **WebSocket Connections**: Real-time connection count
- **Database Performance**: Query times and connection pools
- **Cache Hit Rates**: Redis performance metrics

### Alerts

Configured alerts for:
- High CPU/memory usage (>80%)
- Service downtime
- High error rates (>10%)
- Pod crash loops
- Database connection issues

## 🔒 Security Features

### Network Security
- **Network Policies**: Micro-segmentation between components
- **Pod Security**: Non-root containers, read-only filesystems
- **RBAC**: Least-privilege access controls
- **TLS Encryption**: End-to-end encryption for all traffic

### Secret Management
- **Kubernetes Secrets**: For sensitive configuration
- **Service Account Tokens**: For inter-service authentication
- **External Secret Integration**: Ready for HashiCorp Vault, AWS Secrets Manager

### Ingress Security
- **SSL/TLS Termination**: Automatic certificate management
- **Rate Limiting**: Protection against abuse
- **CORS Configuration**: Cross-origin request controls
- **Security Headers**: XSS, CSRF, and clickjacking protection

## 🔄 Auto-scaling

### Horizontal Pod Autoscaler (HPA)
- **CPU-based**: Scale based on CPU utilization
- **Memory-based**: Scale based on memory usage
- **Custom Metrics**: Scale based on application-specific metrics

### Vertical Pod Autoscaler (VPA)
- **Resource Optimization**: Automatic resource request/limit tuning
- **Conservative Mode**: For stateful services like databases

### Cluster Autoscaling
- **Node Scaling**: Automatic node provisioning based on resource demands

## 🚨 Troubleshooting

### Common Issues

1. **Pod Not Starting**
   ```bash
   kubectl describe pod <pod-name> -n <namespace>
   kubectl logs <pod-name> -n <namespace>
   ```

2. **Service Unreachable**
   ```bash
   kubectl get svc -n <namespace>
   kubectl describe svc <service-name> -n <namespace>
   ```

3. **Ingress Issues**
   ```bash
   kubectl get ingress -n <namespace>
   kubectl describe ingress <ingress-name> -n <namespace>
   ```

4. **Storage Issues**
   ```bash
   kubectl get pvc -n <namespace>
   kubectl describe pvc <pvc-name> -n <namespace>
   ```

### Health Checks

All services include:
- **Liveness Probes**: Restart unhealthy containers
- **Readiness Probes**: Remove from load balancer when not ready
- **Startup Probes**: Allow slow-starting containers

## 📝 Maintenance

### Updates and Rollbacks

```bash
# Rolling update
kubectl set image deployment/voice-agent voice-agent=new-image:tag -n voice-agents-prod

# Check rollout status
kubectl rollout status deployment/voice-agent -n voice-agents-prod

# Rollback if needed
kubectl rollout undo deployment/voice-agent -n voice-agents-prod
```

### Backup Procedures

1. **Database Backups**: PostgreSQL scheduled backups
2. **Configuration Backups**: Export ConfigMaps and Secrets
3. **Persistent Volume Backups**: Snapshot storage volumes

### Scaling Operations

```bash
# Manual scaling
kubectl scale deployment voice-agent --replicas=10 -n voice-agents-prod

# Check HPA status
kubectl get hpa -n voice-agents-prod
```

## 🔧 Advanced Configuration

### Storage Classes

Update storage provisioner in `03-storage.yaml` based on your cloud provider:

- **AWS**: `kubernetes.io/aws-ebs`
- **GCP**: `kubernetes.io/gce-pd`
- **Azure**: `kubernetes.io/azure-disk`

### Cloud Provider Specifics

#### AWS
- Update LoadBalancer annotations for ALB/NLB
- Configure IAM roles for service accounts
- Use AWS Secrets Manager integration

#### GCP
- Configure Google Cloud Load Balancer
- Use Google Secret Manager
- Enable GKE Autopilot features

#### Azure
- Configure Azure Load Balancer
- Use Azure Key Vault integration
- Enable Azure Monitor integration

## 📞 Support

For deployment issues or questions:

1. Check the troubleshooting section above
2. Review Kubernetes and application logs
3. Consult the monitoring dashboards
4. Contact the platform team with specific error messages and steps to reproduce

## 🔄 CI/CD Integration

This configuration is designed to work with:
- **GitOps workflows** (ArgoCD, Flux)
- **Helm charts** (can be converted)
- **Kustomize overlays** for environment-specific configurations
- **CI/CD pipelines** with proper secret management