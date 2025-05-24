#!/bin/bash

# ==============================================
# Memory-Enhanced AI Platform Setup Script
# ==============================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'  
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="memory-enhanced-ai-platform"
REQUIRED_MEMORY_GB=16
REQUIRED_DISK_GB=50

# Print functions
print_header() {
    echo -e "${BLUE}=========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}=========================================${NC}"
}

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check system requirements
check_requirements() {
    print_header "Checking System Requirements"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    print_info "Docker: $(docker --version)"
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    print_info "Docker Compose: $(docker-compose --version)"
    
    # Check available memory
    if command -v free &> /dev/null; then
        AVAILABLE_MEMORY_KB=$(free | grep '^Mem:' | awk '{print $2}')
        AVAILABLE_MEMORY_GB=$((AVAILABLE_MEMORY_KB / 1024 / 1024))
        if [ $AVAILABLE_MEMORY_GB -lt $REQUIRED_MEMORY_GB ]; then
            print_warning "Available memory: ${AVAILABLE_MEMORY_GB}GB (recommended: ${REQUIRED_MEMORY_GB}GB)"
        else
            print_info "Available memory: ${AVAILABLE_MEMORY_GB}GB ✓"
        fi
    fi
    
    # Check available disk space
    AVAILABLE_DISK_GB=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
    if [ $AVAILABLE_DISK_GB -lt $REQUIRED_DISK_GB ]; then
        print_warning "Available disk space: ${AVAILABLE_DISK_GB}GB (recommended: ${REQUIRED_DISK_GB}GB)"
    else
        print_info "Available disk space: ${AVAILABLE_DISK_GB}GB ✓"
    fi
}

# Create project structure
create_project_structure() {
    print_header "Creating Project Structure"
    
    # Create main project directory
    if [ ! -d "$PROJECT_NAME" ]; then
        mkdir "$PROJECT_NAME"
        print_info "Created project directory: $PROJECT_NAME"
    fi
    
    cd "$PROJECT_NAME"
    
    # Create directory structure
    directories=(
        "services/mcp-memory-server"
        "services/rag-memory-service"
        "services/agent-memory-orchestrator"
        "services/memory-analytics"
        "services/memory-health-monitor"
        "config"
        "database"
        "nginx/ssl"
        "nginx/logs"
        "monitoring/prometheus"
        "monitoring/grafana/memory-dashboards"
        "monitoring/grafana/datasources"
        "logs"
        "models"
        "data/documents"
        "agent-config"
        "workflows"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        print_info "Created directory: $dir"
    done
}

# Generate environment file
generate_env_file() {
    print_header "Generating Environment Configuration"
    
    if [ ! -f ".env" ]; then
        cat > .env << 'EOF'
# ==============================================
# Memory-Enhanced AI Platform Configuration
# ==============================================

# API Keys (REQUIRED - Replace with your actual keys)
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here
MCP_ACCESS_TOKEN=secure-memory-token-123

# Database Passwords (Change these for production!)
REDIS_PASSWORD=secure-redis-password-123
POSTGRES_PASSWORD=secure-postgres-password-123
NEO4J_PASSWORD=secure-neo4j-password-123
TIMESCALE_PASSWORD=secure-timescale-password-123

# Admin Passwords (Change these for production!)
GRAFANA_PASSWORD=secure-grafana-password-123
PGADMIN_PASSWORD=secure-pgadmin-password-123

# Memory System Configuration
WORKING_MEMORY_CAPACITY=7
WORKING_MEMORY_TTL=1800
MEMORY_CONSOLIDATION_INTERVAL=21600
MEMORY_CLEANUP_INTERVAL=3600
EPISODIC_MEMORY_DECAY_RATE=0.99
SEMANTIC_MEMORY_SIMILARITY_THRESHOLD=0.85

# Monitoring and Alerts
ALERT_WEBHOOK_URL=https://hooks.slack.com/your-webhook-url
ENABLE_MEMORY_ANALYTICS=true
ENABLE_AUTOMATIC_CONSOLIDATION=true

# Performance Settings
MAX_CONCURRENT_AGENTS=10
AGENT_MEMORY_SYNC_INTERVAL=300
ENABLE_MULTI_AGENT_COORDINATION=true
ENABLE_AGENT_LEARNING=true
WORKING_MEMORY_SHARING=true
EOF
        print_info "Created .env configuration file"
        print_warning "Please edit .env file and add your API keys!"
    else
        print_info ".env file already exists"
    fi
}

# Create basic configuration files
create_config_files() {
    print_header "Creating Configuration Files"
    
    # Redis configuration (basic version)
    if [ ! -f "config/redis-memory.conf" ]; then
        cat > config/redis-memory.conf << 'EOF'
bind 0.0.0.0
port 6379
protected-mode yes
requirepass memorypass123
maxmemory 1.5gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
loglevel notice
EOF
        print_info "Created Redis configuration"
    fi
    
    # ChromaDB configuration (basic version)
    if [ ! -f "config/chroma-config.yaml" ]; then
        cat > config/chroma-config.yaml << 'EOF'
server:
  host: "0.0.0.0"
  port: 8000

storage:
  persist_directory: "/chroma/chroma"
  is_persistent: true

auth:
  enabled: false

logging:
  level: "INFO"
EOF
        print_info "Created ChromaDB configuration"
    fi
}

# Download and setup docker-compose file
setup_docker_compose() {
    print_header "Setting up Docker Compose"
    
    if [ ! -f "docker-compose.yml" ]; then
        print_info "Please paste your docker-compose.yml content in the current directory"
        print_info "The docker-compose.yml file should be provided separately"
        
        # Create a basic docker-compose.yml template
        cat > docker-compose.yml << 'EOF'
# Please replace this with the complete docker-compose.yml content
# provided in the Memory-Enhanced AI Platform package

version: '3.8'

services:
  # Add your services here
  placeholder:
    image: alpine:latest
    command: echo "Please replace with complete docker-compose.yml"
EOF
        print_warning "Created placeholder docker-compose.yml - please replace with actual content"
    else
        print_info "docker-compose.yml already exists"
    fi
}

# Setup application code
setup_application_code() {
    print_header "Setting up Application Code"
    
    # Create placeholder files to indicate where code should go
    services=("mcp-memory-server" "rag-memory-service" "agent-memory-orchestrator" "memory-analytics" "memory-health-monitor")
    
    for service in "${services[@]}"; do
        service_dir="services/$service"
        
        # Create requirements.txt
        if [ ! -f "$service_dir/requirements.txt" ]; then
            echo "# Add service requirements here" > "$service_dir/requirements.txt"
        fi
        
        # Create Dockerfile
        if [ ! -f "$service_dir/Dockerfile" ]; then
            cat > "$service_dir/Dockerfile" << 'EOF'
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "main.py"]
EOF
        fi
        
        # Create placeholder main.py
        if [ ! -f "$service_dir/main.py" ]; then
            echo "# $service implementation goes here" > "$service_dir/main.py"
        fi
    done
    
    print_info "Created placeholder application structure"
    print_warning "Please add the actual service implementations to the services/ directories"
}

# Interactive API key setup
setup_api_keys() {
    print_header "API Key Configuration"
    
    echo -e "${YELLOW}The system can work without API keys, but AI features will be limited.${NC}"
    echo ""
    
    read -p "Do you have an OpenAI API key? (y/n): " has_openai
    if [[ $has_openai =~ ^[Yy]$ ]]; then
        read -p "Enter your OpenAI API key: " openai_key
        if [ ! -z "$openai_key" ]; then
            sed -i.bak "s/OPENAI_API_KEY=your-openai-api-key-here/OPENAI_API_KEY=$openai_key/" .env
            print_info "OpenAI API key configured"
        fi
    fi
    
    read -p "Do you have an Anthropic API key? (y/n): " has_anthropic
    if [[ $has_anthropic =~ ^[Yy]$ ]]; then
        read -p "Enter your Anthropic API key: " anthropic_key
        if [ ! -z "$anthropic_key" ]; then
            sed -i.bak "s/ANTHROPIC_API_KEY=your-anthropic-api-key-here/ANTHROPIC_API_KEY=$anthropic_key/" .env
            print_info "Anthropic API key configured"
        fi
    fi
    
    # Generate secure access token
    access_token=$(openssl rand -hex 32 2>/dev/null || date +%s | sha256sum | base64 | head -c 32)
    sed -i.bak "s/MCP_ACCESS_TOKEN=secure-memory-token-123/MCP_ACCESS_TOKEN=$access_token/" .env
    print_info "Generated secure MCP access token"
}

# Test Docker setup
test_docker_setup() {
    print_header "Testing Docker Setup"
    
    # Test basic Docker functionality
    if docker run --rm hello-world > /dev/null 2>&1; then
        print_info "Docker is working correctly ✓"
    else
        print_error "Docker test failed"
        return 1
    fi
    
    # Check if docker-compose file is valid
    if docker-compose config > /dev/null 2>&1; then
        print_info "Docker Compose configuration is valid ✓"
    else
        print_warning "Docker Compose configuration may have issues"
        print_info "This is expected if you haven't replaced the placeholder docker-compose.yml"
    fi
}

# Start services
start_services() {
    print_header "Starting Services"
    
    echo -e "${YELLOW}Choose deployment option:${NC}"
    echo "1) Full production deployment"
    echo "2) Development deployment (includes admin tools)"
    echo "3) Skip deployment (setup only)"
    read -p "Enter choice (1-3): " deploy_choice
    
    case $deploy_choice in
        1)
            print_info "Starting full production deployment..."
            docker-compose up -d
            ;;
        2)
            print_info "Starting development deployment..."
            docker-compose --profile development up -d
            ;;
        3)
            print_info "Skipping deployment - you can start services later with:"
            print_info "  docker-compose up -d"
            return
            ;;
        *)
            print_warning "Invalid choice, skipping deployment"
            return
            ;;
    esac
    
    # Wait a moment for services to start
    print_info "Waiting for services to start..."
    sleep 10
    
    # Check service status
    print_info "Service status:"
    docker-compose ps
}

# Display final information
show_completion_info() {
    print_header "Setup Complete!"
    
    echo -e "${GREEN}Your Memory-Enhanced AI Platform is ready!${NC}"
    echo ""
    echo -e "${BLUE}Next Steps:${NC}"
    echo "1. Review and update the .env file with your API keys"
    echo "2. Replace placeholder service code with actual implementations"
    echo "3. Update docker-compose.yml with the complete configuration"
    echo ""
    echo -e "${BLUE}Service URLs (when running):${NC}"
    echo "• MCP Memory Server:     http://localhost:8080"
    echo "• RAG Service:           http://localhost:8001"
    echo "• Agent Orchestrator:    http://localhost:8003"
    echo "• Memory Analytics:      http://localhost:8005"
    echo "• Grafana Dashboard:     http://localhost:3000"
    echo "• Prometheus Metrics:    http://localhost:9090"
    echo ""
    echo -e "${BLUE}Useful Commands:${NC}"
    echo "• Start services:        docker-compose up -d"
    echo "• Stop services:         docker-compose down"
    echo "• View logs:             docker-compose logs -f [service-name]"
    echo "• Check status:          docker-compose ps"
    echo "• Development mode:      docker-compose --profile development up -d"
    echo ""
    echo -e "${YELLOW}Important:${NC}"
    echo "• Change default passwords in .env before production use"
    echo "• Add your actual service implementations"
    echo "• Configure SSL certificates for production"
    echo "• Set up monitoring alerts (webhook URL in .env)"
    echo ""
    echo -e "${GREEN}For support and documentation, refer to the deployment guide.${NC}"
}

# Main execution
main() {
    print_header "Memory-Enhanced AI Platform Setup"
    echo -e "${BLUE}This script will set up your Memory-Enhanced AI Platform${NC}"
    echo ""
    
    # Confirmation
    read -p "Continue with setup? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Setup cancelled"
        exit 0
    fi
    
    # Run setup steps
    check_requirements
    create_project_structure
    generate_env_file
    create_config_files
    setup_docker_compose
    setup_application_code
    setup_api_keys
    test_docker_setup
    
    # Ask about starting services
    echo ""
    read -p "Start the services now? (y/n): " start_now
    if [[ $start_now =~ ^[Yy]$ ]]; then
        start_services
    fi
    
    show_completion_info
}

# Handle script interruption
trap 'print_error "Setup interrupted by user"; exit 1' INT

# Run main function
main "$@"

# ==============================================
# Additional utility functions
# ==============================================

# Quick start function (separate script: quick-start.sh)
cat > quick-start.sh << 'EOF'
#!/bin/bash

# Quick start script for Memory-Enhanced AI Platform

echo "🚀 Quick Starting Memory-Enhanced AI Platform..."

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "❌ .env file not found. Please run setup.sh first."
    exit 1
fi

# Start services
echo "📦 Starting services..."
docker-compose up -d

# Wait for services
echo "⏳ Waiting for services to start..."
sleep 30

# Health check
echo "🔍 Checking service health..."
services=("http://localhost:8080/health" "http://localhost:8001/health" "http://localhost:8003/health")

for service in "${services[@]}"; do
    if curl -s "$service" > /dev/null; then
        echo "✅ ${service} is healthy"
    else
        echo "⚠️  ${service} is not responding"
    fi
done

echo ""
echo "🎉 Platform is starting up!"
echo "📊 Access Grafana: http://localhost:3000"
echo "🔍 View logs: docker-compose logs -f"
echo "🛑 Stop services: docker-compose down"
EOF

chmod +x quick-start.sh

echo ""