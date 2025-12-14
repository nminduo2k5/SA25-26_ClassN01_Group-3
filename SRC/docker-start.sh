#!/bin/bash

# Multi-Agent Vietnam Stock System Docker Startup Script

set -e

echo "🚀 Starting Multi-Agent Vietnam Stock System..."

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        echo "❌ Docker is not running. Please start Docker first."
        exit 1
    fi
    echo "✅ Docker is running"
}

# Function to build and start services
start_services() {
    local env=${1:-dev}
    
    echo "📦 Building Docker images..."
    
    case $env in
        "dev"|"development")
            echo "🔧 Starting Development Environment..."
            docker-compose -f docker-compose.dev.yml down
            docker-compose -f docker-compose.dev.yml build
            docker-compose -f docker-compose.dev.yml up -d
            echo "✅ Development services started!"
            echo "🌐 Streamlit: http://localhost:8501"
            echo "🔗 API: http://localhost:8000"
            echo "📚 API Docs: http://localhost:8000/api/docs"
            ;;
        "prod"|"production")
            echo "🏭 Starting Production Environment..."
            docker-compose -f docker-compose.prod.yml down
            docker-compose -f docker-compose.prod.yml build
            docker-compose -f docker-compose.prod.yml up -d
            echo "✅ Production services started!"
            echo "🌐 Application: http://localhost"
            echo "🔗 API: http://localhost:8000"
            ;;
        "simple")
            echo "🎯 Starting Simple Environment..."
            docker-compose down
            docker-compose build
            docker-compose up -d
            echo "✅ Simple services started!"
            echo "🌐 Streamlit: http://localhost:8501"
            echo "🔗 API: http://localhost:8000"
            ;;
        "fast")
            echo "⚡ Starting Fast Environment..."
            docker-compose -f docker-compose.fast.yml down
            docker-compose -f docker-compose.fast.yml build
            docker-compose -f docker-compose.fast.yml up -d
            echo "✅ Fast services started!"
            echo "🌐 Streamlit: http://localhost:8501"
            echo "🔗 API: http://localhost:8000"
            ;;
        *)
            echo "❌ Invalid environment. Use: dev, prod, or simple"
            exit 1
            ;;
    esac
}

# Function to stop services
stop_services() {
    echo "🛑 Stopping all services..."
    docker-compose -f docker-compose.dev.yml down 2>/dev/null || true
    docker-compose -f docker-compose.prod.yml down 2>/dev/null || true
    docker-compose down 2>/dev/null || true
    echo "✅ All services stopped"
}

# Function to show logs
show_logs() {
    local service=${1:-}
    if [ -z "$service" ]; then
        echo "📋 Showing all logs..."
        docker-compose logs -f
    else
        echo "📋 Showing logs for $service..."
        docker-compose logs -f "$service"
    fi
}

# Function to show status
show_status() {
    echo "📊 Service Status:"
    docker-compose ps
    echo ""
    echo "🐳 Docker Images:"
    docker images | grep vnstock || echo "No vnstock images found"
    echo ""
    echo "📦 Docker Volumes:"
    docker volume ls | grep vnstock || echo "No vnstock volumes found"
}

# Main script logic
case ${1:-help} in
    "start")
        check_docker
        start_services ${2:-dev}
        ;;
    "stop")
        stop_services
        ;;
    "restart")
        check_docker
        stop_services
        start_services ${2:-dev}
        ;;
    "logs")
        show_logs $2
        ;;
    "status")
        show_status
        ;;
    "clean")
        echo "🧹 Cleaning up Docker resources..."
        stop_services
        docker system prune -f
        docker volume prune -f
        echo "✅ Cleanup completed"
        ;;
    "help"|*)
        echo "🔧 Multi-Agent Vietnam Stock System Docker Manager"
        echo ""
        echo "Usage: $0 <command> [options]"
        echo ""
        echo "Commands:"
        echo "  start [env]    Start services (env: dev, prod, simple)"
        echo "  stop           Stop all services"
        echo "  restart [env]  Restart services"
        echo "  logs [service] Show logs"
        echo "  status         Show service status"
        echo "  clean          Clean up Docker resources"
        echo "  help           Show this help"
        echo ""
        echo "Examples:"
        echo "  $0 start dev       # Start development environment"
        echo "  $0 start prod      # Start production environment"
        echo "  $0 logs api        # Show API logs"
        echo "  $0 status          # Show all service status"
        ;;
esac