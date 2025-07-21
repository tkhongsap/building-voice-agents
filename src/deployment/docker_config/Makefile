# Makefile for LiveKit Voice Agents Platform Docker Operations
# Provides convenient commands for building, testing, and deploying

# Variables
PROJECT_NAME = voice-agents
VERSION ?= latest
REGISTRY ?= your-registry.com
IMAGE_NAME = $(REGISTRY)/$(PROJECT_NAME)
COMPOSE_FILE = docker-compose.yml
COMPOSE_PROD_FILE = docker-compose.prod.yml

# Docker build targets
.PHONY: build build-dev build-prod build-all
.PHONY: push pull clean test
.PHONY: up down logs shell
.PHONY: up-prod down-prod logs-prod
.PHONY: security-scan optimize

# Default target
.DEFAULT_GOAL := help

# Help target
help: ## Show this help message
	@echo "LiveKit Voice Agents Platform Docker Commands"
	@echo "============================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Build targets
build: ## Build development image
	@echo "Building development image..."
	docker build \
		--target development \
		--tag $(PROJECT_NAME):dev-$(VERSION) \
		--tag $(PROJECT_NAME):dev-latest \
		--file Dockerfile \
		../../..

build-prod: ## Build production image
	@echo "Building production image..."
	docker build \
		--target production \
		--tag $(PROJECT_NAME):$(VERSION) \
		--tag $(PROJECT_NAME):latest \
		--tag $(IMAGE_NAME):$(VERSION) \
		--tag $(IMAGE_NAME):latest \
		--file Dockerfile \
		../../..

build-all: build build-prod ## Build both development and production images

# Multi-platform build for production
build-multi: ## Build multi-platform production image
	@echo "Building multi-platform production image..."
	docker buildx build \
		--platform linux/amd64,linux/arm64 \
		--target production \
		--tag $(IMAGE_NAME):$(VERSION) \
		--tag $(IMAGE_NAME):latest \
		--file Dockerfile \
		--push \
		../../..

# Push and pull
push: ## Push production image to registry
	@echo "Pushing production image to registry..."
	docker push $(IMAGE_NAME):$(VERSION)
	docker push $(IMAGE_NAME):latest

pull: ## Pull production image from registry
	@echo "Pulling production image from registry..."
	docker pull $(IMAGE_NAME):$(VERSION)

# Development environment
up: ## Start development environment
	@echo "Starting development environment..."
	docker-compose -f $(COMPOSE_FILE) up -d
	@echo "Development environment started. Services available at:"
	@echo "  Voice Agent: http://localhost:8080"
	@echo "  Grafana: http://localhost:3000 (admin/admin)"
	@echo "  Prometheus: http://localhost:9090"
	@echo "  Jaeger: http://localhost:16686"

down: ## Stop development environment
	@echo "Stopping development environment..."
	docker-compose -f $(COMPOSE_FILE) down

restart: down up ## Restart development environment

logs: ## Show logs from development environment
	docker-compose -f $(COMPOSE_FILE) logs -f

logs-app: ## Show logs from voice agent only
	docker-compose -f $(COMPOSE_FILE) logs -f voice-agent voice-agent-worker-1 voice-agent-worker-2

shell: ## Open shell in voice agent container
	docker-compose -f $(COMPOSE_FILE) exec voice-agent /bin/bash

# Production environment
up-prod: ## Start production environment
	@echo "Starting production environment..."
	@if [ ! -f .env.prod ]; then echo "Error: .env.prod file not found"; exit 1; fi
	docker-compose -f $(COMPOSE_PROD_FILE) --env-file .env.prod up -d
	@echo "Production environment started."

down-prod: ## Stop production environment
	@echo "Stopping production environment..."
	docker-compose -f $(COMPOSE_PROD_FILE) down

restart-prod: down-prod up-prod ## Restart production environment

logs-prod: ## Show logs from production environment
	docker-compose -f $(COMPOSE_PROD_FILE) logs -f

# Testing
test: ## Run tests in container
	@echo "Running tests in container..."
	docker run --rm \
		-v $(PWD)/../../..:/app \
		$(PROJECT_NAME):dev-latest \
		python -m pytest tests/ -v

test-integration: ## Run integration tests with full environment
	@echo "Running integration tests..."
	docker-compose -f $(COMPOSE_FILE) run --rm \
		-e ENVIRONMENT=test \
		voice-agent \
		python -m pytest tests/integration/ -v

# Performance and load testing
load-test: ## Run load tests against development environment
	@echo "Running load tests..."
	docker-compose -f $(COMPOSE_FILE) run --rm \
		voice-agent \
		python -m pytest tests/load/ -v

benchmark: ## Run performance benchmarks
	@echo "Running performance benchmarks..."
	docker run --rm \
		--network voice_agents_voice-agents-network \
		$(PROJECT_NAME):dev-latest \
		python scripts/benchmark.py --target http://voice-agent:8080

# Security and optimization
security-scan: ## Run security scan on production image
	@echo "Running security scan..."
	@command -v trivy >/dev/null 2>&1 || { echo "trivy not installed. Install with: curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin"; exit 1; }
	trivy image $(PROJECT_NAME):latest

optimize: ## Optimize images and clean up
	@echo "Optimizing Docker images..."
	docker system prune -f
	docker image prune -f
	@echo "Removing unused volumes..."
	docker volume prune -f

# Database operations
db-migrate: ## Run database migrations
	docker-compose -f $(COMPOSE_FILE) exec voice-agent \
		python scripts/migrate.py

db-seed: ## Seed database with test data
	docker-compose -f $(COMPOSE_FILE) exec voice-agent \
		python scripts/seed.py

db-backup: ## Backup database
	@echo "Creating database backup..."
	docker-compose -f $(COMPOSE_FILE) exec postgres \
		pg_dump -U agent voice_agents > backups/db_backup_$(shell date +%Y%m%d_%H%M%S).sql

db-restore: ## Restore database from backup (set BACKUP_FILE variable)
	@if [ -z "$(BACKUP_FILE)" ]; then echo "Error: BACKUP_FILE variable not set"; exit 1; fi
	@echo "Restoring database from $(BACKUP_FILE)..."
	docker-compose -f $(COMPOSE_FILE) exec -T postgres \
		psql -U agent -d voice_agents < $(BACKUP_FILE)

# Monitoring and debugging
monitor: ## Open monitoring dashboard
	@echo "Opening monitoring dashboard..."
	@command -v open >/dev/null 2>&1 && open http://localhost:3000 || echo "Open http://localhost:3000 in your browser"

health-check: ## Check health of all services
	@echo "Checking service health..."
	@docker-compose -f $(COMPOSE_FILE) ps
	@echo ""
	@echo "Health check endpoints:"
	@curl -s http://localhost:8080/health | jq '.' || echo "Voice Agent: UNHEALTHY"
	@curl -s http://localhost:9090/-/healthy | grep -q "Prometheus is Healthy" && echo "Prometheus: HEALTHY" || echo "Prometheus: UNHEALTHY"
	@curl -s http://localhost:3000/api/health | jq '.' || echo "Grafana: UNHEALTHY"

# Cleanup
clean: ## Clean up all containers, images, and volumes
	@echo "Cleaning up Docker resources..."
	docker-compose -f $(COMPOSE_FILE) down -v --remove-orphans
	docker-compose -f $(COMPOSE_PROD_FILE) down -v --remove-orphans
	docker system prune -a -f
	docker volume prune -f

clean-images: ## Remove all project images
	@echo "Removing project images..."
	docker rmi $(PROJECT_NAME):dev-latest $(PROJECT_NAME):latest $(IMAGE_NAME):latest 2>/dev/null || true

# Configuration
setup-dev: ## Set up development environment
	@echo "Setting up development environment..."
	@cp .env.example .env
	@echo "Created .env file. Please edit it with your configuration."
	@mkdir -p ssl logs cache backups
	@echo "Created necessary directories."

setup-prod: ## Set up production environment
	@echo "Setting up production environment..."
	@cp .env.prod.example .env.prod
	@echo "Created .env.prod file. Please edit it with your production configuration."
	@mkdir -p ssl logs cache backups
	@echo "Created necessary directories."
	@echo "Don't forget to:"
	@echo "  1. Configure SSL certificates in ssl/ directory"
	@echo "  2. Set up proper secrets management"
	@echo "  3. Configure backup storage"

# Development utilities
format: ## Format code in container
	docker run --rm -v $(PWD)/../../..:/app $(PROJECT_NAME):dev-latest \
		black src/ tests/

lint: ## Lint code in container
	docker run --rm -v $(PWD)/../../..:/app $(PROJECT_NAME):dev-latest \
		flake8 src/ tests/

type-check: ## Type check code in container
	docker run --rm -v $(PWD)/../../..:/app $(PROJECT_NAME):dev-latest \
		mypy src/

# Release
release: build-prod test security-scan ## Build, test, scan and prepare for release
	@echo "Release preparation completed successfully!"
	@echo "Next steps:"
	@echo "  1. Tag the release: git tag v$(VERSION)"
	@echo "  2. Push to registry: make push"
	@echo "  3. Deploy to production: make up-prod"