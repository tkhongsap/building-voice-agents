# Makefile for LiveKit Voice Agents Platform Kubernetes Deployment
# Provides convenient commands for deployment, management, and troubleshooting

# Variables
NAMESPACE_DEV=voice-agents-dev
NAMESPACE_STAGING=voice-agents-staging
NAMESPACE_PROD=voice-agents-prod
NAMESPACE_MONITORING=voice-agents-monitoring

# Default environment
ENV?=dev
IMAGE_TAG?=latest
REGISTRY?=your-registry.com/voice-agents

# Color codes for output
RED=\033[0;31m
GREEN=\033[0;32m
YELLOW=\033[1;33m
BLUE=\033[0;34m
NC=\033[0m # No Color

.PHONY: help
help: ## Display this help message
	@echo "$(BLUE)LiveKit Voice Agents Platform - Kubernetes Deployment$(NC)"
	@echo ""
	@echo "$(YELLOW)Usage:$(NC)"
	@echo "  make [target] [ENV=dev|staging|prod]"
	@echo ""
	@echo "$(YELLOW)Targets:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: check-env
check-env: ## Check if environment is valid
	@if [ "$(ENV)" != "dev" ] && [ "$(ENV)" != "staging" ] && [ "$(ENV)" != "prod" ]; then \
		echo "$(RED)Error: ENV must be dev, staging, or prod$(NC)"; \
		exit 1; \
	fi

.PHONY: check-kubectl
check-kubectl: ## Check if kubectl is available and configured
	@kubectl version --client > /dev/null 2>&1 || (echo "$(RED)Error: kubectl not found or not configured$(NC)" && exit 1)
	@echo "$(GREEN)kubectl is available and configured$(NC)"

.PHONY: check-cluster
check-cluster: check-kubectl ## Check cluster connectivity
	@kubectl cluster-info > /dev/null 2>&1 || (echo "$(RED)Error: Cannot connect to Kubernetes cluster$(NC)" && exit 1)
	@echo "$(GREEN)Connected to Kubernetes cluster$(NC)"

# Namespace management
.PHONY: create-namespaces
create-namespaces: check-cluster ## Create all namespaces
	@echo "$(BLUE)Creating namespaces...$(NC)"
	kubectl apply -f 00-namespaces.yaml
	@echo "$(GREEN)Namespaces created successfully$(NC)"

.PHONY: delete-namespaces
delete-namespaces: check-cluster ## Delete all namespaces (WARNING: This will delete all data!)
	@echo "$(RED)WARNING: This will delete all namespaces and data!$(NC)"
	@read -p "Are you sure? (y/N) " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		kubectl delete -f 00-namespaces.yaml; \
		echo "$(GREEN)Namespaces deleted$(NC)"; \
	else \
		echo "$(YELLOW)Operation cancelled$(NC)"; \
	fi

# Storage management
.PHONY: deploy-storage
deploy-storage: check-cluster ## Deploy storage configurations
	@echo "$(BLUE)Deploying storage configurations...$(NC)"
	kubectl apply -f 03-storage.yaml
	@echo "$(GREEN)Storage configurations deployed$(NC)"

# Configuration management
.PHONY: deploy-config
deploy-config: check-cluster ## Deploy ConfigMaps and basic secrets
	@echo "$(BLUE)Deploying configurations...$(NC)"
	kubectl apply -f 01-configmaps.yaml
	kubectl apply -f 02-secrets.yaml
	@echo "$(YELLOW)Note: Update secrets with production values before deploying to production$(NC)"
	@echo "$(GREEN)Configurations deployed$(NC)"

# RBAC
.PHONY: deploy-rbac
deploy-rbac: check-cluster ## Deploy RBAC configurations
	@echo "$(BLUE)Deploying RBAC configurations...$(NC)"
	kubectl apply -f 09-rbac.yaml
	@echo "$(GREEN)RBAC configurations deployed$(NC)"

# Application deployment
.PHONY: deploy-supporting-services
deploy-supporting-services: check-cluster ## Deploy supporting services (PostgreSQL, Redis, LiveKit, NGINX)
	@echo "$(BLUE)Deploying supporting services...$(NC)"
	kubectl apply -f 05-supporting-services.yaml
	@echo "$(GREEN)Supporting services deployed$(NC)"

.PHONY: deploy-voice-agent
deploy-voice-agent: check-cluster check-env ## Deploy voice agent application
	@echo "$(BLUE)Deploying voice agent to $(ENV) environment...$(NC)"
	kubectl apply -f 04-voice-agent-deployment.yaml
	@echo "$(GREEN)Voice agent deployed$(NC)"

.PHONY: deploy-services
deploy-services: check-cluster ## Deploy Kubernetes services
	@echo "$(BLUE)Deploying services...$(NC)"
	kubectl apply -f 06-services.yaml
	@echo "$(GREEN)Services deployed$(NC)"

.PHONY: deploy-ingress
deploy-ingress: check-cluster ## Deploy ingress configurations
	@echo "$(BLUE)Deploying ingress configurations...$(NC)"
	@echo "$(YELLOW)Make sure to update domain names in 07-ingress.yaml before deployment$(NC)"
	kubectl apply -f 07-ingress.yaml
	@echo "$(GREEN)Ingress configurations deployed$(NC)"

.PHONY: deploy-monitoring
deploy-monitoring: check-cluster ## Deploy monitoring stack (Prometheus, Grafana, Jaeger)
	@echo "$(BLUE)Deploying monitoring stack...$(NC)"
	kubectl apply -f 11-monitoring.yaml
	@echo "$(GREEN)Monitoring stack deployed$(NC)"

.PHONY: deploy-autoscaling
deploy-autoscaling: check-cluster ## Deploy autoscaling configurations
	@echo "$(BLUE)Deploying autoscaling configurations...$(NC)"
	kubectl apply -f 08-autoscaling.yaml
	@echo "$(GREEN)Autoscaling configurations deployed$(NC)"

.PHONY: deploy-network-policies
deploy-network-policies: check-cluster ## Deploy network policies
	@echo "$(BLUE)Deploying network policies...$(NC)"
	kubectl apply -f 10-network-policies.yaml
	@echo "$(GREEN)Network policies deployed$(NC)"

# Full deployment
.PHONY: deploy-infrastructure
deploy-infrastructure: create-namespaces deploy-storage deploy-config deploy-rbac ## Deploy basic infrastructure
	@echo "$(GREEN)Infrastructure deployment completed$(NC)"

.PHONY: deploy-applications
deploy-applications: deploy-supporting-services deploy-voice-agent deploy-services ## Deploy applications
	@echo "$(GREEN)Application deployment completed$(NC)"

.PHONY: deploy-all
deploy-all: deploy-infrastructure deploy-applications deploy-ingress deploy-monitoring deploy-autoscaling deploy-network-policies ## Deploy everything
	@echo "$(GREEN)Full deployment completed$(NC)"

# Environment-specific deployments
.PHONY: deploy-dev
deploy-dev: ## Deploy to development environment
	@$(MAKE) deploy-all ENV=dev

.PHONY: deploy-staging
deploy-staging: ## Deploy to staging environment
	@$(MAKE) deploy-all ENV=staging

.PHONY: deploy-prod
deploy-prod: ## Deploy to production environment
	@echo "$(RED)Deploying to PRODUCTION environment$(NC)"
	@echo "$(YELLOW)Make sure all secrets are properly configured!$(NC)"
	@read -p "Continue? (y/N) " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		$(MAKE) deploy-all ENV=prod; \
	else \
		echo "$(YELLOW)Production deployment cancelled$(NC)"; \
	fi

# Status and monitoring
.PHONY: status
status: check-cluster check-env ## Show deployment status for environment
	@echo "$(BLUE)Status for $(ENV) environment:$(NC)"
	@kubectl get pods -n voice-agents-$(ENV) -o wide
	@echo ""
	@kubectl get svc -n voice-agents-$(ENV)
	@echo ""
	@kubectl get ingress -n voice-agents-$(ENV)

.PHONY: status-all
status-all: check-cluster ## Show status for all environments
	@echo "$(BLUE)Development Environment:$(NC)"
	@kubectl get pods -n $(NAMESPACE_DEV) 2>/dev/null || echo "No resources found"
	@echo ""
	@echo "$(BLUE)Staging Environment:$(NC)"
	@kubectl get pods -n $(NAMESPACE_STAGING) 2>/dev/null || echo "No resources found"
	@echo ""
	@echo "$(BLUE)Production Environment:$(NC)"
	@kubectl get pods -n $(NAMESPACE_PROD) 2>/dev/null || echo "No resources found"
	@echo ""
	@echo "$(BLUE)Monitoring:$(NC)"
	@kubectl get pods -n $(NAMESPACE_MONITORING) 2>/dev/null || echo "No resources found"

.PHONY: logs
logs: check-cluster check-env ## Show logs for voice agent pods
	@echo "$(BLUE)Logs for voice agent in $(ENV) environment:$(NC)"
	kubectl logs -n voice-agents-$(ENV) -l app=voice-agents-platform,component=voice-agent --tail=100 -f

.PHONY: describe-pods
describe-pods: check-cluster check-env ## Describe voice agent pods for troubleshooting
	@echo "$(BLUE)Describing voice agent pods in $(ENV) environment:$(NC)"
	kubectl describe pods -n voice-agents-$(ENV) -l app=voice-agents-platform,component=voice-agent

# Port forwarding for local access
.PHONY: port-forward-grafana
port-forward-grafana: check-cluster ## Port forward Grafana (localhost:3000)
	@echo "$(BLUE)Port forwarding Grafana to localhost:3000$(NC)"
	@echo "$(YELLOW)Access Grafana at http://localhost:3000$(NC)"
	kubectl port-forward -n $(NAMESPACE_MONITORING) svc/grafana-service 3000:3000

.PHONY: port-forward-prometheus
port-forward-prometheus: check-cluster ## Port forward Prometheus (localhost:9090)
	@echo "$(BLUE)Port forwarding Prometheus to localhost:9090$(NC)"
	@echo "$(YELLOW)Access Prometheus at http://localhost:9090$(NC)"
	kubectl port-forward -n $(NAMESPACE_MONITORING) svc/prometheus-service 9090:9090

.PHONY: port-forward-jaeger
port-forward-jaeger: check-cluster ## Port forward Jaeger (localhost:16686)
	@echo "$(BLUE)Port forwarding Jaeger to localhost:16686$(NC)"
	@echo "$(YELLOW)Access Jaeger at http://localhost:16686$(NC)"
	kubectl port-forward -n $(NAMESPACE_MONITORING) svc/jaeger-service 16686:16686

.PHONY: port-forward-app
port-forward-app: check-cluster check-env ## Port forward voice agent application (localhost:8080)
	@echo "$(BLUE)Port forwarding voice agent to localhost:8080$(NC)"
	@echo "$(YELLOW)Access application at http://localhost:8080$(NC)"
	kubectl port-forward -n voice-agents-$(ENV) svc/voice-agent-service 8080:8080

# Scaling operations
.PHONY: scale
scale: check-cluster check-env ## Scale voice agent replicas (usage: make scale ENV=prod REPLICAS=5)
	@if [ -z "$(REPLICAS)" ]; then \
		echo "$(RED)Error: REPLICAS not specified$(NC)"; \
		echo "$(YELLOW)Usage: make scale ENV=prod REPLICAS=5$(NC)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Scaling voice agent to $(REPLICAS) replicas in $(ENV) environment$(NC)"
	kubectl scale deployment voice-agent --replicas=$(REPLICAS) -n voice-agents-$(ENV)
	@echo "$(GREEN)Scaling completed$(NC)"

.PHONY: hpa-status
hpa-status: check-cluster check-env ## Show HPA status
	@echo "$(BLUE)HPA status for $(ENV) environment:$(NC)"
	kubectl get hpa -n voice-agents-$(ENV)

# Cleanup operations
.PHONY: clean-env
clean-env: check-cluster check-env ## Clean specific environment
	@echo "$(RED)WARNING: This will delete all resources in $(ENV) environment!$(NC)"
	@read -p "Are you sure? (y/N) " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		kubectl delete all --all -n voice-agents-$(ENV); \
		echo "$(GREEN)Environment $(ENV) cleaned$(NC)"; \
	else \
		echo "$(YELLOW)Operation cancelled$(NC)"; \
	fi

.PHONY: clean-monitoring
clean-monitoring: check-cluster ## Clean monitoring namespace
	@echo "$(RED)WARNING: This will delete all monitoring resources!$(NC)"
	@read -p "Are you sure? (y/N) " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		kubectl delete all --all -n $(NAMESPACE_MONITORING); \
		echo "$(GREEN)Monitoring resources cleaned$(NC)"; \
	else \
		echo "$(YELLOW)Operation cancelled$(NC)"; \
	fi

# Secret management
.PHONY: create-prod-secrets
create-prod-secrets: check-cluster ## Create production secrets (interactive)
	@echo "$(BLUE)Creating production secrets...$(NC)"
	@echo "$(YELLOW)You will be prompted for each secret value$(NC)"
	@read -p "PostgreSQL password: " -s POSTGRES_PASSWORD && echo "" && \
	read -p "LiveKit API key: " LIVEKIT_API_KEY && \
	read -p "LiveKit API secret: " -s LIVEKIT_API_SECRET && echo "" && \
	read -p "OpenAI API key: " -s OPENAI_API_KEY && echo "" && \
	read -p "Anthropic API key: " -s ANTHROPIC_API_KEY && echo "" && \
	read -p "JWT secret: " -s JWT_SECRET && echo "" && \
	kubectl create secret generic voice-agents-secrets \
		--namespace=voice-agents-prod \
		--from-literal=POSTGRES_PASSWORD=$$POSTGRES_PASSWORD \
		--from-literal=LIVEKIT_API_KEY=$$LIVEKIT_API_KEY \
		--from-literal=LIVEKIT_API_SECRET=$$LIVEKIT_API_SECRET \
		--from-literal=OPENAI_API_KEY=$$OPENAI_API_KEY \
		--from-literal=ANTHROPIC_API_KEY=$$ANTHROPIC_API_KEY \
		--from-literal=JWT_SECRET=$$JWT_SECRET \
		--dry-run=client -o yaml | kubectl apply -f -
	@echo "$(GREEN)Production secrets created$(NC)"

# Update operations
.PHONY: update-image
update-image: check-cluster check-env ## Update voice agent image (usage: make update-image ENV=prod IMAGE_TAG=v1.2.3)
	@echo "$(BLUE)Updating voice agent image to $(REGISTRY):$(IMAGE_TAG) in $(ENV) environment$(NC)"
	kubectl set image deployment/voice-agent voice-agent=$(REGISTRY):$(IMAGE_TAG) -n voice-agents-$(ENV)
	kubectl rollout status deployment/voice-agent -n voice-agents-$(ENV)
	@echo "$(GREEN)Image update completed$(NC)"

.PHONY: rollback
rollback: check-cluster check-env ## Rollback voice agent deployment
	@echo "$(BLUE)Rolling back voice agent deployment in $(ENV) environment$(NC)"
	kubectl rollout undo deployment/voice-agent -n voice-agents-$(ENV)
	kubectl rollout status deployment/voice-agent -n voice-agents-$(ENV)
	@echo "$(GREEN)Rollback completed$(NC)"

# Health checks
.PHONY: health-check
health-check: check-cluster check-env ## Run health checks for environment
	@echo "$(BLUE)Running health checks for $(ENV) environment$(NC)"
	@echo "Checking pods..."
	@kubectl get pods -n voice-agents-$(ENV) | grep -E "(Running|Ready)" || echo "$(RED)Some pods are not healthy$(NC)"
	@echo "Checking services..."
	@kubectl get svc -n voice-agents-$(ENV) | grep -v "No resources found" || echo "$(RED)No services found$(NC)"
	@echo "Checking ingress..."
	@kubectl get ingress -n voice-agents-$(ENV) | grep -v "No resources found" || echo "$(RED)No ingress found$(NC)"
	@echo "$(GREEN)Health check completed$(NC)"

# Backup operations
.PHONY: backup-config
backup-config: check-cluster ## Backup ConfigMaps and Secrets to local files
	@echo "$(BLUE)Backing up configurations...$(NC)"
	@mkdir -p backups/$(shell date +%Y%m%d_%H%M%S)
	@kubectl get configmaps -n voice-agents-dev -o yaml > backups/$(shell date +%Y%m%d_%H%M%S)/configmaps-dev.yaml
	@kubectl get secrets -n voice-agents-dev -o yaml > backups/$(shell date +%Y%m%d_%H%M%S)/secrets-dev.yaml
	@kubectl get configmaps -n voice-agents-staging -o yaml > backups/$(shell date +%Y%m%d_%H%M%S)/configmaps-staging.yaml 2>/dev/null || true
	@kubectl get secrets -n voice-agents-staging -o yaml > backups/$(shell date +%Y%m%d_%H%M%S)/secrets-staging.yaml 2>/dev/null || true
	@kubectl get configmaps -n voice-agents-prod -o yaml > backups/$(shell date +%Y%m%d_%H%M%S)/configmaps-prod.yaml 2>/dev/null || true
	@kubectl get secrets -n voice-agents-prod -o yaml > backups/$(shell date +%Y%m%d_%H%M%S)/secrets-prod.yaml 2>/dev/null || true
	@echo "$(GREEN)Configuration backup completed$(NC)"

# Development helpers
.PHONY: dev-setup
dev-setup: ## Setup development environment quickly
	@echo "$(BLUE)Setting up development environment...$(NC)"
	@$(MAKE) create-namespaces
	@$(MAKE) deploy-storage
	@$(MAKE) deploy-config
	@$(MAKE) deploy-rbac
	@$(MAKE) deploy-supporting-services
	@$(MAKE) deploy-voice-agent ENV=dev
	@$(MAKE) deploy-services
	@$(MAKE) deploy-monitoring
	@echo "$(GREEN)Development environment setup completed$(NC)"
	@echo "$(YELLOW)You can now access your application at the configured ingress URL$(NC)"

.PHONY: monitoring-urls
monitoring-urls: ## Display monitoring access URLs
	@echo "$(BLUE)Monitoring Access URLs:$(NC)"
	@echo "$(YELLOW)Port forward commands (run in separate terminals):$(NC)"
	@echo "  Grafana:    make port-forward-grafana    # http://localhost:3000"
	@echo "  Prometheus: make port-forward-prometheus # http://localhost:9090"
	@echo "  Jaeger:     make port-forward-jaeger     # http://localhost:16686"
	@echo ""
	@echo "$(YELLOW)Or access via ingress (if configured):$(NC)"
	@kubectl get ingress -n $(NAMESPACE_MONITORING) 2>/dev/null || echo "  No ingress configured for monitoring"