#!/bin/bash

# Multi-Agent Stock Analysis - Automated Setup Script
# This script sets up the complete Docker environment

set -e

echo "🚀 Multi-Agent Stock Analysis - Automated Setup"
echo "================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker Desktop first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if Docker is running
    if ! docker info &> /dev/null; then
        print_error "Docker is not running. Please start Docker Desktop first."
        exit 1
    fi
    
    print_success "All prerequisites met!"
}

# Setup environment
setup_environment() {
    print_status "Setting up environment..."
    
    # Create .env file if it doesn't exist
    if [ ! -f .env ]; then
        cp .env.template .env
        print_success "Environment file created from template"
        print_warning "Please edit .env file with your actual API keys"
    else
        print_warning ".env file already exists"
    fi
    
    # Create necessary directories
    mkdir -p logs backups monitoring/grafana monitoring/prometheus database gateway/ssl
    print_success "Directory structure created"
}

# Setup monitoring configuration
setup_monitoring() {
    print_status "Setting up monitoring configuration..."
    
    # Prometheus configuration
    cat > monitoring/prometheus.yml << EOF
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'vnstock-services'
    static_configs:
      - targets: ['price-predictor:8001', 'investment-expert:8002', 'llm-hub:8010']
    metrics_path: /metrics
    scrape_interval: 30s

  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx:80']
    metrics_path: /metrics

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']
EOF

    # Grafana dashboard configuration
    mkdir -p monitoring/grafana/dashboards monitoring/grafana/datasources
    
    cat > monitoring/grafana/datasources/prometheus.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF

    print_success "Monitoring configuration created"
}

# Setup database initialization
setup_database() {
    print_status "Setting up database initialization..."
    
    cat > database/init/01-init.sql << EOF
-- Create database schema for VNStock analysis
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(10) NOT NULL,
    predicted_price DECIMAL(10,2),
    current_price DECIMAL(10,2),
    change_percent DECIMAL(5,2),
    confidence DECIMAL(5,2),
    method_used VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Investment recommendations table
CREATE TABLE IF NOT EXISTS recommendations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(10) NOT NULL,
    recommendation VARCHAR(20),
    confidence DECIMAL(5,2),
    target_price DECIMAL(10,2),
    stop_loss DECIMAL(10,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User sessions table
CREATE TABLE IF NOT EXISTS user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(100) UNIQUE,
    user_data JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_predictions_symbol ON predictions(symbol);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_recommendations_symbol ON recommendations(symbol);
CREATE INDEX IF NOT EXISTS idx_recommendations_created_at ON recommendations(created_at);

-- Insert sample data
INSERT INTO predictions (symbol, predicted_price, current_price, change_percent, confidence, method_used)
VALUES 
    ('VCB', 95000, 92000, 3.26, 75.5, 'Technical Analysis'),
    ('BID', 48500, 47200, 2.75, 68.2, 'LSTM Neural Network'),
    ('CTG', 32100, 31800, 0.94, 82.1, 'Ensemble Method')
ON CONFLICT DO NOTHING;
EOF

    print_success "Database initialization script created"
}

# Setup SSL certificates (self-signed for development)
setup_ssl() {
    print_status "Setting up SSL certificates..."
    
    if [ ! -f gateway/ssl/cert.pem ]; then
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout gateway/ssl/key.pem \
            -out gateway/ssl/cert.pem \
            -subj "/C=VN/ST=HCM/L=HoChiMinh/O=VNStock/CN=localhost" 2>/dev/null || {
            print_warning "OpenSSL not available, skipping SSL setup"
            return
        }
        print_success "Self-signed SSL certificates created"
    else
        print_warning "SSL certificates already exist"
    fi
}

# Build and start services
start_services() {
    print_status "Building and starting services..."
    
    # Choose environment
    echo "Select environment:"
    echo "1) Development (with hot reload)"
    echo "2) Production (with monitoring)"
    echo "3) Basic (simple setup)"
    read -p "Enter choice [1-3]: " choice
    
    case $choice in
        1)
            print_status "Starting development environment..."
            docker-compose -f docker-compose.dev.yml up --build -d
            print_success "Development environment started!"
            echo ""
            echo "🌐 Development URLs:"
            echo "   Frontend: http://localhost:8502"
            echo "   API Gateway: http://localhost:8080"
            echo "   Database Admin: http://localhost:8081"
            echo "   Redis Admin: http://localhost:8082"
            ;;
        2)
            print_status "Starting production environment..."
            docker-compose -f docker-compose.production.yml up --build -d
            print_success "Production environment started!"
            echo ""
            echo "🌐 Production URLs:"
            echo "   Frontend: http://localhost:8502"
            echo "   API Gateway: http://localhost:8080"
            echo "   Monitoring: http://localhost:3000 (admin/admin123)"
            echo "   Metrics: http://localhost:9090"
            echo "   Logs: http://localhost:5601"
            ;;
        3)
            print_status "Starting basic environment..."
            docker-compose up --build -d
            print_success "Basic environment started!"
            echo ""
            echo "🌐 Basic URLs:"
            echo "   Frontend: http://localhost:8502"
            echo "   API Gateway: http://localhost:8080"
            ;;
        *)
            print_error "Invalid choice"
            exit 1
            ;;
    esac
}

# Wait for services to be ready
wait_for_services() {
    print_status "Waiting for services to be ready..."
    
    # Wait for key services
    for i in {1..30}; do
        if curl -s http://localhost:8080/health > /dev/null 2>&1; then
            print_success "Services are ready!"
            return 0
        fi
        echo -n "."
        sleep 2
    done
    
    print_warning "Services may still be starting up. Check logs if needed."
}

# Run health checks
run_health_checks() {
    print_status "Running health checks..."
    
    services=("llm-hub:8010" "price-predictor:8001" "investment-expert:8002")
    
    for service in "${services[@]}"; do
        name=$(echo $service | cut -d: -f1)
        port=$(echo $service | cut -d: -f2)
        
        if curl -s http://localhost:$port/health > /dev/null 2>&1; then
            print_success "$name service is healthy"
        else
            print_warning "$name service may not be ready yet"
        fi
    done
}

# Main execution
main() {
    echo ""
    check_prerequisites
    echo ""
    setup_environment
    echo ""
    setup_monitoring
    echo ""
    setup_database
    echo ""
    setup_ssl
    echo ""
    start_services
    echo ""
    wait_for_services
    echo ""
    run_health_checks
    echo ""
    
    print_success "🎉 Setup completed successfully!"
    echo ""
    echo "📚 Next steps:"
    echo "   1. Edit .env file with your API keys"
    echo "   2. Visit http://localhost:8502 to use the application"
    echo "   3. Check service logs: docker-compose logs -f"
    echo "   4. Stop services: docker-compose down"
    echo ""
    echo "📖 For more commands, see the Makefile or run 'make help'"
}

# Run main function
main "$@"