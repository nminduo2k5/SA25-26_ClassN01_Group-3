#!/bin/bash

echo "🚀 Starting Multi-Agent Stock Analysis Microservices System"
echo "============================================================"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose is not installed. Please install docker-compose first."
    exit 1
fi

# Set environment variables
export GEMINI_API_KEY=${GEMINI_API_KEY:-"your_gemini_api_key_here"}
export OPENAI_API_KEY=${OPENAI_API_KEY:-"your_openai_api_key_here"}

echo "🔧 Environment variables set"
echo "   GEMINI_API_KEY: ${GEMINI_API_KEY:0:10}..."
echo "   OPENAI_API_KEY: ${OPENAI_API_KEY:0:10}..."

# Build and start services
echo ""
echo "🏗️ Building and starting services..."
docker-compose up --build -d

# Wait for services to be ready
echo ""
echo "⏳ Waiting for services to be ready..."
sleep 30

# Check service health
echo ""
echo "🔍 Checking service health..."

services=("nginx:80" "redis:6379" "postgres:5432" "llm-hub:8010" "price-predictor:8001" "investment-expert:8002" "frontend:8501")

for service in "${services[@]}"; do
    IFS=':' read -r name port <<< "$service"
    if docker-compose ps | grep -q "$name.*Up"; then
        echo "✅ $name service is running"
    else
        echo "❌ $name service is not running"
    fi
done

echo ""
echo "🌐 Service URLs:"
echo "   Frontend (Streamlit): http://localhost:8501"
echo "   API Gateway (Nginx): http://localhost:80"
echo "   Price Predictor API: http://localhost:8001"
echo "   Investment Expert API: http://localhost:8002"
echo "   LLM Hub API: http://localhost:8010"
echo "   Redis: localhost:6379"
echo "   PostgreSQL: localhost:5432"

echo ""
echo "📊 To view logs:"
echo "   docker-compose logs -f [service_name]"
echo ""
echo "🛑 To stop all services:"
echo "   docker-compose down"
echo ""
echo "🎉 Multi-Agent Stock Analysis System is ready!"
echo "   Open http://localhost:8501 in your browser to start using the system."