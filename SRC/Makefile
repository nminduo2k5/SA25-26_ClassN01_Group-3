# Multi-Agent Vietnam Stock System - Docker Management
.PHONY: help build up down logs clean dev prod restart status update

# Default target
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "Multi-Agent Vietnam Stock System - Docker Commands"
	@echo "================================================="
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Production Commands
up: ## Start production environment
	@echo "🚀 Starting production environment..."
	docker-compose down 2>/dev/null || true
	docker-compose up --build -d
	@echo "✅ Production ready at http://localhost:8501"

prod: up ## Alias for production start

build: ## Build Docker images
	@echo "🏗️ Building Docker images..."
	docker-compose build --no-cache

# Development Commands  
dev: ## Start development environment with hot reload
	@echo "🔧 Starting development environment..."
	docker-compose -f docker-compose.dev.yml down 2>/dev/null || true
	docker-compose -f docker-compose.dev.yml up --build -d
	@echo "✅ Development ready at http://localhost:8501"
	@echo "🔧 Redis UI at http://localhost:8081"

# Control Commands
down: ## Stop all services
	@echo "🛑 Stopping all services..."
	docker-compose down 2>/dev/null || true
	docker-compose -f docker-compose.dev.yml down 2>/dev/null || true
	@echo "✅ All services stopped"

stop: down ## Alias for stop

restart: ## Restart services
	@echo "🔄 Restarting services..."
	$(MAKE) down
	sleep 2
	$(MAKE) up

# Logging Commands
logs: ## Show logs for all services
	docker-compose logs -f

logs-app: ## Show application logs only
	docker-compose logs -f app

logs-redis: ## Show Redis logs only
	docker-compose logs -f redis

# Maintenance Commands
update: ## Update with latest code changes
	@echo "🔄 Updating with latest code..."
	docker-compose down
	docker-compose build --no-cache
	docker-compose up -d
	@echo "✅ Update completed!"

clean: ## Clean up Docker resources
	@echo "🧹 Cleaning up Docker resources..."
	docker-compose down -v 2>/dev/null || true
	docker-compose -f docker-compose.dev.yml down -v 2>/dev/null || true
	docker system prune -f
	docker volume prune -f
	@echo "✅ Cleanup completed"

clean-all: ## Deep clean all Docker resources
	@echo "🧹 Deep cleaning all Docker resources..."
	$(MAKE) down
	docker system prune -af --volumes
	@echo "✅ Deep cleanup completed"

# Status Commands
status: ## Show system status
	@echo "📊 System Status:"
	@echo "=================="
	@echo "🐳 Containers:"
	@docker-compose ps 2>/dev/null || echo "No containers running"
	@echo ""
	@echo "📦 Images:"
	@docker images | grep vnstock || echo "No vnstock images found"
	@echo ""
	@echo "💾 Volumes:"
	@docker volume ls | grep vnstock || echo "No vnstock volumes found"

health: ## Check service health
	@echo "🔍 Checking service health..."
	@curl -s http://localhost:8501/_stcore/health > /dev/null && echo "✅ App is healthy" || echo "❌ App is not responding"
	@docker exec vnstock-redis redis-cli ping > /dev/null 2>&1 && echo "✅ Redis is healthy" || echo "❌ Redis is not responding"

ps: ## Show running containers
	docker-compose ps

# Testing Commands
test: ## Run tests in container
	docker-compose exec app python -m pytest tests/ -v

shell: ## Open shell in app container
	docker-compose exec app /bin/bash

redis-cli: ## Open Redis CLI
	docker-compose exec redis redis-cli

# Quick Commands
quick-start: ## Quick start for development
	@echo "⚡ Quick start for development..."
	docker-compose -f docker-compose.dev.yml up -d
	@echo "✅ Ready at http://localhost:8501"

# Backup Commands
backup: ## Backup application data
	@echo "💾 Creating backup..."
	@mkdir -p backups
	@docker cp vnstock-app:/app/duong_trading.db backups/duong_trading_$(shell date +%Y%m%d_%H%M%S).db
	@echo "✅ Backup created in backups/ directory"

# Monitoring Commands
monitor: ## Show real-time resource usage
	docker stats

top: ## Show container processes
	docker-compose top

# Network Commands
network: ## Show network information
	docker network ls | grep vnstock

# Environment Commands
env-check: ## Check environment setup
	@echo "🔍 Environment Check:"
	@echo "===================="
	@docker --version
	@docker-compose --version
	@echo "✅ Docker environment ready"

# Development Helpers
format: ## Format code in container
	docker-compose exec app black /app --exclude __pycache__

lint: ## Lint code in container
	docker-compose exec app flake8 /app --exclude __pycache__

# Documentation
docs: ## Generate documentation
	@echo "📚 Available endpoints:"
	@echo "  http://localhost:8501 - Main Application"
	@echo "  http://localhost:8081 - Redis UI (dev mode)"
	@echo "  http://localhost:6379 - Redis Server"

# Installation
install: ## Install and setup everything
	@echo "🔧 Installing Multi-Agent Vietnam Stock System..."
	$(MAKE) build
	$(MAKE) up
	@echo "🎉 Installation completed!"
	@echo "🌐 Access your application at http://localhost:8501"