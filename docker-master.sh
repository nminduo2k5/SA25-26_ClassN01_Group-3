#!/bin/bash

# Multi-Agent Stock Analysis - Master Docker Manager
# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}🚀 Multi-Agent Stock Analysis - Master Docker Manager${NC}"
    echo -e "${BLUE}=====================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Check if Docker is running
check_docker() {
    if ! docker info &> /dev/null; then
        print_error "Docker is not running. Please start Docker first."
        exit 1
    fi
}

# Original System Management
manage_original() {
    local command=$1
    print_info "Managing Original System (SRC/)"
    echo "================================="
    
    cd SRC || { print_error "SRC directory not found"; exit 1; }
    
    case $command in
        "start"|"")
            print_info "Starting Original System..."
            make up
            ;;
        "dev")
            print_info "Starting Original System in Development mode..."
            make dev
            ;;
        "prod")
            print_info "Starting Original System in Production mode..."
            make prod
            ;;
        "stop")
            print_info "Stopping Original System..."
            make down
            ;;
        "logs")
            print_info "Showing Original System logs..."
            make logs
            ;;
        "status")
            print_info "Original System status..."
            make status
            ;;
        "clean")
            print_info "Cleaning Original System..."
            make clean
            ;;
        *)
            print_error "Unknown command for original system: $command"
            show_help_original
            ;;
    esac
}

# Microservices System Management
manage_microservices() {
    local command=$1
    print_info "Managing Microservices System (SRC/microservices/)"
    echo "=================================================="
    
    cd SRC/microservices || { print_error "SRC/microservices directory not found"; exit 1; }
    
    case $command in
        "start"|"")
            print_info "Starting Microservices System..."
            make up
            ;;
        "dev")
            print_info "Starting Microservices in Development mode..."
            make dev
            print_success "Microservices Development ready!"
            echo "🌐 Frontend: http://localhost:8502"
            echo "🔧 Redis UI: http://localhost:8081"
            ;;
        "prod")
            print_info "Starting Microservices in Production mode..."
            make prod
            print_success "Microservices Production ready!"
            echo "🌐 Frontend: http://localhost:8502"
            echo "📊 Monitoring: http://localhost:3000"
            ;;
        "basic")
            print_info "Starting Microservices Basic mode..."
            make up
            print_success "Microservices Basic ready!"
            echo "🌐 Frontend: http://localhost:8502"
            ;;
        "stop")
            print_info "Stopping Microservices System..."
            make down
            ;;
        "logs")
            print_info "Showing Microservices logs..."
            make logs
            ;;
        "status")
            print_info "Microservices status..."
            make status
            ;;
        "health")
            print_info "Checking Microservices health..."
            make health
            ;;
        "clean")
            print_info "Cleaning Microservices System..."
            make clean
            ;;
        *)
            print_error "Unknown command for microservices: $command"
            show_help_microservices
            ;;
    esac
}

# Show status of all systems
show_status_all() {
    print_info "System Status Overview"
    echo "======================"
    echo
    
    echo "🏠 Original System (Port 8501):"
    cd SRC && docker-compose ps 2>/dev/null || echo "   Not running"
    echo
    
    echo "🏗️ Microservices System (Port 8502):"
    cd ../SRC/microservices && docker-compose ps 2>/dev/null || echo "   Not running"
    echo
    
    echo "🐳 All Docker Containers:"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    echo
    
    echo "💾 Docker Resources:"
    docker system df
}

# Stop all systems
stop_all() {
    print_info "Stopping All Systems"
    echo "===================="
    echo
    
    echo "1. Stopping Original System..."
    cd SRC && docker-compose down 2>/dev/null
    echo
    
    echo "2. Stopping Microservices System..."
    cd ../SRC/microservices
    docker-compose down 2>/dev/null
    docker-compose -f docker-compose.dev.yml down 2>/dev/null
    docker-compose -f docker-compose.production.yml down 2>/dev/null
    echo
    
    print_success "All systems stopped"
}

# Clean all systems
clean_all() {
    print_info "Cleaning All Systems"
    echo "==================="
    echo
    
    echo "1. Stopping all containers..."
    stop_all
    echo
    
    echo "2. Cleaning Docker resources..."
    docker system prune -f
    docker volume prune -f
    echo
    
    print_success "Complete cleanup finished"
}

# Help functions
show_help() {
    print_header
    echo
    echo "Usage: ./docker-master.sh <system> [command]"
    echo
    echo "Systems:"
    echo "  original       Original monolithic system (SRC/)"
    echo "  micro          Microservices system (SRC/microservices/)"
    echo "  microservices  Alias for micro"
    echo
    echo "Commands:"
    echo "  start          Start system (default)"
    echo "  dev            Start in development mode"
    echo "  prod           Start in production mode"
    echo "  basic          Start basic mode (microservices only)"
    echo "  stop           Stop system"
    echo "  logs           Show logs"
    echo "  status         Show system status"
    echo "  health         Check service health (microservices only)"
    echo "  clean          Clean system resources"
    echo
    echo "Global Commands:"
    echo "  status         Show status of all systems"
    echo "  stop-all       Stop all systems"
    echo "  clean-all      Clean all Docker resources"
    echo "  help           Show this help"
    echo
    echo "Examples:"
    echo "  $0 original start     # Start original system"
    echo "  $0 micro dev          # Start microservices in dev mode"
    echo "  $0 micro prod         # Start microservices in production"
    echo "  $0 original stop      # Stop original system"
    echo "  $0 status             # Show all systems status"
    echo "  $0 stop-all           # Stop everything"
    echo
    echo "🌐 URLs:"
    echo "  Original System:      http://localhost:8501"
    echo "  Microservices:        http://localhost:8502"
    echo "  Microservices API:    http://localhost:8080"
    echo
    show_help_details
}

show_help_original() {
    echo
    echo "🏠 Original System Commands:"
    echo "  $0 original start     # Quick start (production-ready)"
    echo "  $0 original dev       # Development with hot reload"
    echo "  $0 original prod      # Production optimized"
    echo "  $0 original stop      # Stop services"
    echo "  $0 original logs      # View logs"
    echo "  $0 original status    # System status"
    echo "  $0 original clean     # Cleanup resources"
    echo
}

show_help_microservices() {
    echo
    echo "🏗️ Microservices System Commands:"
    echo "  $0 micro start        # Start with monitoring"
    echo "  $0 micro dev          # Development with hot reload"
    echo "  $0 micro prod         # Production with full monitoring"
    echo "  $0 micro basic        # Basic setup"
    echo "  $0 micro stop         # Stop all services"
    echo "  $0 micro logs         # View logs"
    echo "  $0 micro status       # Service status"
    echo "  $0 micro health       # Health checks"
    echo "  $0 micro clean        # Cleanup resources"
    echo
}

show_help_details() {
    echo "📋 Detailed Information:"
    echo
    echo "🏠 Original System (Monolithic):"
    echo "  - Single container with all agents"
    echo "  - Port: 8501"
    echo "  - Best for: Quick start, testing, demo"
    echo "  - Resource usage: ~4GB RAM"
    echo
    echo "🏗️ Microservices System (Distributed):"
    echo "  - 6 separate services + monitoring"
    echo "  - Ports: 8502, 8080, 8001, 8002, 8010"
    echo "  - Best for: Production, scaling, enterprise"
    echo "  - Resource usage: ~6GB RAM"
    echo
    echo "💡 Recommendations:"
    echo "  - New users: Start with 'original'"
    echo "  - Developers: Use 'micro dev'"
    echo "  - Production: Use 'micro prod'"
    echo
}

# Main execution
main() {
    local system=$1
    local command=$2
    
    # Check Docker first
    check_docker
    
    case $system in
        "original")
            manage_original "$command"
            ;;
        "micro"|"microservices")
            manage_microservices "$command"
            ;;
        "status")
            show_status_all
            ;;
        "stop-all")
            stop_all
            ;;
        "clean-all")
            clean_all
            ;;
        "help"|""|*)
            show_help
            ;;
    esac
}

# Run main function with all arguments
main "$@"