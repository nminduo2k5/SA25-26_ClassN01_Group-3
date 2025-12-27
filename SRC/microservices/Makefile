# Multi-Agent Stock Analysis - Docker Management
.PHONY: help build up down logs clean dev prod test backup

# Default environment
ENV ?= dev

help: ## Show this help message
	@echo "Multi-Agent Stock Analysis - Docker Commands"
	@echo "============================================="
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Development Commands
dev: ## Start development environment with hot reload
	@echo "🚀 Starting development environment..."
	docker-compose -f docker-compose.dev.yml up --build -d
	@echo "✅ Development environment ready!"
	@echo "   Frontend: http://localhost:8502"
	@echo "   API Docs: http://localhost:8001/docs"
	@echo "   Database: http://localhost:8081 (adminer)"
	@echo "   Redis: http://localhost:8082 (redis-commander)"

dev-logs: ## Show development logs
	docker-compose -f docker-compose.dev.yml logs -f

dev-down: ## Stop development environment
	docker-compose -f docker-compose.dev.yml down

# Production Commands
prod: ## Start production environment
	@echo "🚀 Starting production environment..."
	docker-compose -f docker-compose.production.yml up --build -d
	@echo "✅ Production environment ready!"
	@echo "   Frontend: http://localhost:8502"
	@echo "   Monitoring: http://localhost:3000 (grafana)"
	@echo "   Metrics: http://localhost:9090 (prometheus)"
	@echo "   Logs: http://localhost:5601 (kibana)"

prod-logs: ## Show production logs
	docker-compose -f docker-compose.production.yml logs -f

prod-down: ## Stop production environment
	docker-compose -f docker-compose.production.yml down

# Basic Commands
up: ## Start default environment
	docker-compose up --build -d

down: ## Stop all services
	docker-compose down

build: ## Build all services
	docker-compose build --no-cache

logs: ## Show logs for all services
	docker-compose logs -f

# Service-specific commands
logs-frontend: ## Show frontend logs
	docker-compose logs -f frontend

logs-price: ## Show price predictor logs
	docker-compose logs -f price-predictor

logs-investment: ## Show investment expert logs
	docker-compose logs -f investment-expert

logs-llm: ## Show LLM hub logs
	docker-compose logs -f llm-hub

# Health checks
health: ## Check health of all services
	@echo "🔍 Checking service health..."
	@curl -s http://localhost:8010/health | jq . || echo "❌ LLM Hub not responding"
	@curl -s http://localhost:8001/health | jq . || echo "❌ Price Predictor not responding"
	@curl -s http://localhost:8002/health | jq . || echo "❌ Investment Expert not responding"
	@curl -s http://localhost:8080/health || echo "❌ Gateway not responding"

# Database operations
db-backup: ## Backup database
	@echo "💾 Creating database backup..."
	docker-compose exec postgres pg_dump -U vnstock vnstock_db > backups/backup_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "✅ Backup created in backups/ directory"

db-restore: ## Restore database from latest backup
	@echo "🔄 Restoring database from latest backup..."
	docker-compose exec -T postgres psql -U vnstock vnstock_db < $(shell ls -t backups/*.sql | head -1)
	@echo "✅ Database restored"

db-shell: ## Connect to database shell
	docker-compose exec postgres psql -U vnstock vnstock_db

# Cache operations
cache-clear: ## Clear Redis cache
	docker-compose exec redis redis-cli FLUSHALL
	@echo "✅ Cache cleared"

cache-info: ## Show Redis cache info
	docker-compose exec redis redis-cli INFO

# Monitoring
monitor: ## Open monitoring dashboard
	@echo "📊 Opening monitoring dashboards..."
	@open http://localhost:3000 || start http://localhost:3000 || xdg-open http://localhost:3000

metrics: ## Show system metrics
	@echo "📈 System Metrics:"
	@docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"

# Testing
test: ## Run tests for all services
	@echo "🧪 Running tests..."
	docker-compose -f docker-compose.test.yml up --build --abort-on-container-exit
	docker-compose -f docker-compose.test.yml down

test-unit: ## Run unit tests only
	docker-compose exec price-predictor python -m pytest tests/unit/
	docker-compose exec investment-expert python -m pytest tests/unit/

test-integration: ## Run integration tests
	docker-compose exec price-predictor python -m pytest tests/integration/

# Cleanup
clean: ## Clean up Docker resources
	@echo "🧹 Cleaning up Docker resources..."
	docker-compose down -v --remove-orphans
	docker system prune -f
	docker volume prune -f
	@echo "✅ Cleanup completed"

clean-all: ## Clean everything including images
	@echo "🧹 Cleaning all Docker resources..."
	docker-compose down -v --remove-orphans
	docker system prune -af
	docker volume prune -f
	@echo "✅ Full cleanup completed"

# Security
security-scan: ## Run security scan on images
	@echo "🔒 Running security scan..."
	docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
		aquasec/trivy image vnstock-price-predictor:latest
	docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
		aquasec/trivy image vnstock-investment-expert:latest

# Performance
benchmark: ## Run performance benchmarks
	@echo "⚡ Running performance benchmarks..."
	docker run --rm --network host \
		williamyeh/wrk -t4 -c100 -d30s http://localhost:8001/health

# Quick commands
restart: down up ## Restart all services

rebuild: clean build up ## Rebuild and restart everything

status: ## Show status of all services
	docker-compose ps

# Environment setup
setup: ## Initial setup for development
	@echo "🔧 Setting up development environment..."
	cp .env.template .env
	@echo "✅ Environment file created. Please edit .env with your API keys."
	@echo "📝 Next steps:"
	@echo "   1. Edit .env file with your API keys"
	@echo "   2. Run 'make dev' to start development environment"

# Documentation
docs: ## Generate API documentation
	@echo "📚 Generating API documentation..."
	docker-compose exec price-predictor python -c "import main; print('Price Predictor API docs at http://localhost:8001/docs')"
	docker-compose exec investment-expert python -c "import main; print('Investment Expert API docs at http://localhost:8002/docs')"