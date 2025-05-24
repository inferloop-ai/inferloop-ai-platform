#!/bin/bash
# scripts/backup-memories.sh
# Comprehensive backup script for the memory system

set -e

# Configuration
BACKUP_DIR="/app/backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_NAME="memory_backup_${TIMESTAMP}"
POSTGRES_CONTAINER="postgres-memory"
REDIS_CONTAINER="redis-memory"
NEO4J_CONTAINER="neo4j-memory"
CHROMA_CONTAINER="chroma-memory"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Create backup directory
create_backup_dir() {
    log_step "Creating backup directory..."
    mkdir -p "${BACKUP_DIR}/${BACKUP_NAME}"
    
    # Create subdirectories
    mkdir -p "${BACKUP_DIR}/${BACKUP_NAME}/postgres"
    mkdir -p "${BACKUP_DIR}/${BACKUP_NAME}/redis"
    mkdir -p "${BACKUP_DIR}/${BACKUP_NAME}/neo4j"
    mkdir -p "${BACKUP_DIR}/${BACKUP_NAME}/chroma"
    mkdir -p "${BACKUP_DIR}/${BACKUP_NAME}/config"
    mkdir -p "${BACKUP_DIR}/${BACKUP_NAME}/logs"
    
    log_info "Backup directory created: ${BACKUP_DIR}/${BACKUP_NAME}"
}

# Backup PostgreSQL
backup_postgres() {
    log_step "Backing up PostgreSQL database..."
    
    # Full database dump
    docker exec $POSTGRES_CONTAINER pg_dump -U memory_user -d memories > \
        "${BACKUP_DIR}/${BACKUP_NAME}/postgres/memories_full.sql"
    
    # Schema only dump
    docker exec $POSTGRES_CONTAINER pg_dump -U memory_user -d memories --schema-only > \
        "${BACKUP_DIR}/${BACKUP_NAME}/postgres/memories_schema.sql"
    
    # Data only dump
    docker exec $POSTGRES_CONTAINER pg_dump -U memory_user -d memories --data-only > \
        "${BACKUP_DIR}/${BACKUP_NAME}/postgres/memories_data.sql"
    
    # Individual table dumps
    TABLES=("episodic_memories" "semantic_concepts" "semantic_relationships" "agent_memory_profiles" "memory_consolidation_log")
    
    for table in "${TABLES[@]}"; do
        docker exec $POSTGRES_CONTAINER pg_dump -U memory_user -d memories -t $table > \
            "${BACKUP_DIR}/${BACKUP_NAME}/postgres/${table}.sql"
    done
    
    log_info "PostgreSQL backup completed"
}

# Backup Redis
backup_redis() {
    log_step "Backing up Redis data..."
    
    # Trigger Redis save
    docker exec $REDIS_CONTAINER redis-cli -a "${REDIS_PASSWORD:-memorypass123}" BGSAVE
    
    # Wait for save to complete
    while [ "$(docker exec $REDIS_CONTAINER redis-cli -a "${REDIS_PASSWORD:-memorypass123}" LASTSAVE)" = "$(docker exec $REDIS_CONTAINER redis-cli -a "${REDIS_PASSWORD:-memorypass123}" LASTSAVE)" ]; do
        sleep 1
    done
    
    # Copy RDB file
    docker cp $REDIS_CONTAINER:/data/dump.rdb "${BACKUP_DIR}/${BACKUP_NAME}/redis/"
    
    # Export all keys to JSON
    docker exec $REDIS_CONTAINER redis-cli -a "${REDIS_PASSWORD:-memorypass123}" --json KEYS "*" > \
        "${BACKUP_DIR}/${BACKUP_NAME}/redis/keys.json"
    
    log_info "Redis backup completed"
}

# Backup Neo4j
backup_neo4j() {
    log_step "Backing up Neo4j database..."
    
    # Create Neo4j dump
    docker exec $NEO4J_CONTAINER neo4j-admin dump --database=neo4j --to=/tmp/neo4j-backup.dump
    
    # Copy dump file
    docker cp $NEO4J_CONTAINER:/tmp/neo4j-backup.dump "${BACKUP_DIR}/${BACKUP_NAME}/neo4j/"
    
    # Export Cypher script
    docker exec $NEO4J_CONTAINER cypher-shell -u neo4j -p "${NEO4J_PASSWORD:-memorypass123}" \
        "CALL apoc.export.cypher.all('/tmp/neo4j-export.cypher', {})" || log_warning "Cypher export failed (APOC might not be available)"
    
    # Copy export if it exists
    docker cp $NEO4J_CONTAINER:/tmp/neo4j-export.cypher "${BACKUP_DIR}/${BACKUP_NAME}/neo4j/" 2>/dev/null || true
    
    log_info "Neo4j backup completed"
}

# Backup ChromaDB
backup_chroma() {
    log_step "Backing up ChromaDB data..."
    
    # Copy ChromaDB persistent data
    docker cp $CHROMA_CONTAINER:/chroma/chroma/. "${BACKUP_DIR}/${BACKUP_NAME}/chroma/"
    
    # Export collections metadata via API
    curl -s "http://localhost:8004/api/v1/collections" > "${BACKUP_DIR}/${BACKUP_NAME}/chroma/collections.json" || \
        log_warning "Could not export ChromaDB collections metadata"
    
    log_info "ChromaDB backup completed"
}

# Backup configurations
backup_config() {
    log_step "Backing up configuration files..."
    
    # Copy configuration files
    cp -r config/* "${BACKUP_DIR}/${BACKUP_NAME}/config/" 2>/dev/null || log_warning "Config directory not found"
    
    # Copy environment file (without sensitive data)
    if [ -f ".env" ]; then
        # Remove sensitive lines and copy
        grep -v -E "(PASSWORD|SECRET|KEY)" .env > "${BACKUP_DIR}/${BACKUP_NAME}/config/env_template" || true
    fi
    
    # Copy docker-compose file
    cp docker-compose.yml "${BACKUP_DIR}/${BACKUP_NAME}/config/" 2>/dev/null || log_warning "docker-compose.yml not found"
    
    log_info "Configuration backup completed"
}

# Backup logs
backup_logs() {
    log_step "Backing up recent logs..."
    
    # Copy logs from last 7 days
    find logs/ -name "*.log" -mtime -7 -exec cp {} "${BACKUP_DIR}/${BACKUP_NAME}/logs/" \; 2>/dev/null || \
        log_warning "Logs directory not found or empty"
    
    # Export Docker container logs
    CONTAINERS=("mcp-memory-server" "rag-memory-service" "agent-memory-orchestrator" "memory-analytics" "memory-health-monitor")
    
    for container in "${CONTAINERS[@]}"; do
        docker logs --since="7d" $container > "${BACKUP_DIR}/${BACKUP_NAME}/logs/${container}.log" 2>&1 || \
            log_warning "Could not export logs for container: $container"
    done
    
    log_info "Logs backup completed"
}

# Create backup metadata
create_metadata() {
    log_step "Creating backup metadata..."
    
    cat > "${BACKUP_DIR}/${BACKUP_NAME}/backup_info.json" << EOF
{
    "backup_name": "${BACKUP_NAME}",
    "timestamp": "$(date -Iseconds)",
    "version": "1.0.0",
    "components": [
        "postgresql",
        "redis", 
        "neo4j",
        "chromadb",
        "configuration",
        "logs"
    ],
    "system_info": {
        "hostname": "$(hostname)",
        "docker_version": "$(docker --version)",
        "backup_size_mb": "$(du -sm "${BACKUP_DIR}/${BACKUP_NAME}" | cut -f1)"
    }
}
EOF
    
    log_info "Backup metadata created"
}

# Compress backup
compress_backup() {
    log_step "Compressing backup..."
    
    cd "$BACKUP_DIR"
    tar -czf "${BACKUP_NAME}.tar.gz" "${BACKUP_NAME}/"
    
    if [ $? -eq 0 ]; then
        rm -rf "${BACKUP_NAME}/"
        BACKUP_SIZE=$(ls -lh "${BACKUP_NAME}.tar.gz" | awk '{print $5}')
        log_info "Backup compressed successfully: ${BACKUP_NAME}.tar.gz (${BACKUP_SIZE})"
    else
        log_error "Backup compression failed"
        exit 1
    fi
}

# Cleanup old backups
cleanup_old_backups() {
    log_step "Cleaning up old backups..."
    
    # Keep only last 7 backups
    cd "$BACKUP_DIR"
    ls -t memory_backup_*.tar.gz | tail -n +8 | xargs -r rm -f
    
    REMAINING=$(ls -1 memory_backup_*.tar.gz 2>/dev/null | wc -l)
    log_info "Cleanup completed. ${REMAINING} backups retained."
}

# Main backup process
main() {
    log_info "Starting memory system backup..."
    
    # Check if containers are running
    REQUIRED_CONTAINERS=($POSTGRES_CONTAINER $REDIS_CONTAINER $NEO4J_CONTAINER $CHROMA_CONTAINER)
    for container in "${REQUIRED_CONTAINERS[@]}"; do
        if ! docker ps | grep -q "$container"; then
            log_error "Container $container is not running. Please start the services first."
            exit 1
        fi
    done
    
    create_backup_dir
    backup_postgres
    backup_redis
    backup_neo4j
    backup_chroma
    backup_config
    backup_logs
    create_metadata
    compress_backup
    cleanup_old_backups
    
    log_info "Memory system backup completed successfully!"
    log_info "Backup location: ${BACKUP_DIR}/${BACKUP_NAME}.tar.gz"
}

# Run backup
main "$@"

#!/bin/bash
# scripts/restore-memories.sh
# Comprehensive restore script for the memory system

set -e

# Configuration
BACKUP_DIR="/app/backups"
POSTGRES_CONTAINER="postgres-memory"
REDIS_CONTAINER="redis-memory"
NEO4J_CONTAINER="neo4j-memory"
CHROMA_CONTAINER="chroma-memory"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Show usage
usage() {
    echo "Usage: $0 <backup_file> [options]"
    echo ""
    echo "Options:"
    echo "  --postgres-only    Restore only PostgreSQL data"
    echo "  --redis-only       Restore only Redis data"
    echo "  --neo4j-only       Restore only Neo4j data"
    echo "  --chroma-only      Restore only ChromaDB data"
    echo "  --config-only      Restore only configuration files"
    echo "  --force            Skip confirmation prompts"
    echo ""
    echo "Examples:"
    echo "  $0 memory_backup_20241223_120000.tar.gz"
    echo "  $0 memory_backup_20241223_120000.tar.gz --postgres-only"
    echo "  $0 memory_backup_20241223_120000.tar.gz --force"
}

# Parse command line arguments
BACKUP_FILE=""
POSTGRES_ONLY=false
REDIS_ONLY=false
NEO4J_ONLY=false
CHROMA_ONLY=false
CONFIG_ONLY=false
FORCE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --postgres-only)
            POSTGRES_ONLY=true
            shift
            ;;
        --redis-only)
            REDIS_ONLY=true
            shift
            ;;
        --neo4j-only)
            NEO4J_ONLY=true
            shift
            ;;
        --chroma-only)
            CHROMA_ONLY=true
            shift
            ;;
        --config-only)
            CONFIG_ONLY=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            if [ -z "$BACKUP_FILE" ]; then
                BACKUP_FILE="$1"
            else
                log_error "Unknown option: $1"
                usage
                exit 1
            fi
            shift
            ;;
    esac
done

# Validate backup file
if [ -z "$BACKUP_FILE" ]; then
    log_error "Backup file not specified"
    usage
    exit 1
fi

if [ ! -f "$BACKUP_FILE" ]; then
    # Try to find in backup directory
    if [ -f "${BACKUP_DIR}/${BACKUP_FILE}" ]; then
        BACKUP_FILE="${BACKUP_DIR}/${BACKUP_FILE}"
    else
        log_error "Backup file not found: $BACKUP_FILE"
        exit 1
    fi
fi

# Extract backup
extract_backup() {
    log_step "Extracting backup file..."
    
    BACKUP_NAME=$(basename "$BACKUP_FILE" .tar.gz)
    RESTORE_DIR="/tmp/memory_restore_${BACKUP_NAME}"
    
    rm -rf "$RESTORE_DIR"
    mkdir -p "$RESTORE_DIR"
    
    tar -xzf "$BACKUP_FILE" -C "$RESTORE_DIR" --strip-components=1
    
    if [ ! -f "$RESTORE_DIR/backup_info.json" ]; then
        log_error "Invalid backup file: missing backup metadata"
        exit 1
    fi
    
    log_info "Backup extracted to: $RESTORE_DIR"
    echo "$RESTORE_DIR"
}

# Restore PostgreSQL
restore_postgres() {
    log_step "Restoring PostgreSQL database..."
    
    # Drop and recreate database
    docker exec $POSTGRES_CONTAINER psql -U memory_user -d postgres -c "DROP DATABASE IF EXISTS memories;"
    docker exec $POSTGRES_CONTAINER psql -U memory_user -d postgres -c "CREATE DATABASE memories;"
    
    # Restore full database
    if [ -f "$RESTORE_DIR/postgres/memories_full.sql" ]; then
        docker exec -i $POSTGRES_CONTAINER psql -U memory_user -d memories < "$RESTORE_DIR/postgres/memories_full.sql"
        log_info "PostgreSQL full restore completed"
    else
        log_error "PostgreSQL backup file not found"
        return 1
    fi
}

# Restore Redis
restore_redis() {
    log_step "Restoring Redis data..."
    
    # Flush existing data
    docker exec $REDIS_CONTAINER redis-cli -a "${REDIS_PASSWORD:-memorypass123}" FLUSHDB
    
    # Stop Redis temporarily
    docker exec $REDIS_CONTAINER redis-cli -a "${REDIS_PASSWORD:-memorypass123}" SHUTDOWN NOSAVE || true
    
    # Copy RDB file
    if [ -f "$RESTORE_DIR/redis/dump.rdb" ]; then
        docker cp "$RESTORE_DIR/redis/dump.rdb" $REDIS_CONTAINER:/data/dump.rdb
        
        # Restart Redis
        docker restart $REDIS_CONTAINER
        
        # Wait for Redis to start
        sleep 5
        
        log_info "Redis restore completed"
    else
        log_error "Redis backup file not found"
        return 1
    fi
}

# Restore Neo4j
restore_neo4j() {
    log_step "Restoring Neo4j database..."
    
    # Stop Neo4j
    docker stop $NEO4J_CONTAINER
    
    if [ -f "$RESTORE_DIR/neo4j/neo4j-backup.dump" ]; then
        # Copy dump file
        docker cp "$RESTORE_DIR/neo4j/neo4j-backup.dump" $NEO4J_CONTAINER:/tmp/neo4j-backup.dump
        
        # Start Neo4j
        docker start $NEO4J_CONTAINER
        
        # Wait for Neo4j to start
        sleep 10
        
        # Load dump
        docker exec $NEO4J_CONTAINER neo4j-admin load --database=neo4j --from=/tmp/neo4j-backup.dump --force
        
        # Restart to apply changes
        docker restart $NEO4J_CONTAINER
        
        log_info "Neo4j restore completed"
    else
        log_error "Neo4j backup file not found"
        docker start $NEO4J_CONTAINER  # Restart anyway
        return 1
    fi
}

# Restore ChromaDB
restore_chroma() {
    log_step "Restoring ChromaDB data..."
    
    # Stop ChromaDB
    docker stop $CHROMA_CONTAINER
    
    if [ -d "$RESTORE_DIR/chroma" ]; then
        # Remove existing data
        docker exec $CHROMA_CONTAINER rm -rf /chroma/chroma/* 2>/dev/null || true
        
        # Copy backup data
        docker cp "$RESTORE_DIR/chroma/." $CHROMA_CONTAINER:/chroma/chroma/
        
        # Start ChromaDB
        docker start $CHROMA_CONTAINER
        
        log_info "ChromaDB restore completed"
    else
        log_error "ChromaDB backup directory not found"
        docker start $CHROMA_CONTAINER  # Restart anyway
        return 1
    fi
}

# Restore configuration
restore_config() {
    log_step "Restoring configuration files..."
    
    if [ -d "$RESTORE_DIR/config" ]; then
        # Backup current config
        if [ -d "config" ]; then
            mv config config.backup.$(date +%s)
            log_info "Current config backed up"
        fi
        
        # Restore config
        cp -r "$RESTORE_DIR/config" .
        
        log_info "Configuration restore completed"
        log_warning "Please review and update sensitive values in .env file"
    else
        log_error "Configuration backup directory not found"
        return 1
    fi
}

# Verification
verify_restore() {
    log_step "Verifying restore..."
    
    # Check container health
    CONTAINERS=($POSTGRES_CONTAINER $REDIS_CONTAINER $NEO4J_CONTAINER $CHROMA_CONTAINER)
    
    for container in "${CONTAINERS[@]}"; do
        if docker ps --filter "name=$container" --filter "status=running" | grep -q "$container"; then
            log_info "✓ $container is running"
        else
            log_warning "✗ $container is not running"
        fi
    done
    
    # Test database connections
    if docker exec $POSTGRES_CONTAINER pg_isready -U memory_user >/dev/null 2>&1; then
        log_info "✓ PostgreSQL connection successful"
    else
        log_warning "✗ PostgreSQL connection failed"
    fi
    
    if docker exec $REDIS_CONTAINER redis-cli -a "${REDIS_PASSWORD:-memorypass123}" ping >/dev/null 2>&1; then
        log_info "✓ Redis connection successful"
    else
        log_warning "✗ Redis connection failed"
    fi
}

# Main restore process
main() {
    log_info "Starting memory system restore..."
    log_info "Backup file: $BACKUP_FILE"
    
    # Confirmation
    if [ "$FORCE" != true ]; then
        echo ""
        log_warning "This will REPLACE all existing data in the memory system!"
        read -p "Are you sure you want to continue? (yes/no): " confirm
        if [ "$confirm" != "yes" ]; then
            log_info "Restore cancelled"
            exit 0
        fi
    fi
    
    # Extract backup
    RESTORE_DIR=$(extract_backup)
    
    # Determine what to restore
    if [ "$POSTGRES_ONLY" = true ]; then
        restore_postgres
    elif [ "$REDIS_ONLY" = true ]; then
        restore_redis
    elif [ "$NEO4J_ONLY" = true ]; then
        restore_neo4j
    elif [ "$CHROMA_ONLY" = true ]; then
        restore_chroma
    elif [ "$CONFIG_ONLY" = true ]; then
        restore_config
    else
        # Restore everything
        restore_postgres
        restore_redis
        restore_neo4j
        restore_chroma
        restore_config
    fi
    
    # Verify restore
    verify_restore
    
    # Cleanup
    rm -rf "$RESTORE_DIR"
    
    log_info "Memory system restore completed!"
    log_warning "Please restart all services to ensure proper operation:"
    log_warning "  docker-compose restart"
}

# Show available backups if no file specified
if [ -z "$BACKUP_FILE" ]; then
    echo "Available backups:"
    ls -lht "${BACKUP_DIR}"/memory_backup_*.tar.gz 2>/dev/null || echo "No backups found"
    echo ""
    usage
    exit 1
fi

# Run restore
main "$@"

#!/bin/bash
# scripts/health-check.sh
# Comprehensive health check script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Health check results
HEALTH_SCORE=0
MAX_SCORE=0
FAILED_CHECKS=()

# Track score
add_score() {
    local points=$1
    local max_points=$2
    HEALTH_SCORE=$((HEALTH_SCORE + points))
    MAX_SCORE=$((MAX_SCORE + max_points))
}

# Check Docker services
check_docker_services() {
    log_step "Checking Docker services..."
    
    SERVICES=("postgres-memory" "redis-memory" "neo4j-memory" "chroma-memory" "mcp-memory-server" "rag-memory-service" "agent-memory-orchestrator" "memory-analytics" "memory-health-monitor")
    
    for service in "${SERVICES[@]}"; do
        if docker ps --filter "name=$service" --filter "status=running" | grep -q "$service"; then
            log_info "✓ $service is running"
            add_score 1 1
        else
            log_error "✗ $service is not running"
            FAILED_CHECKS+=("$service not running")
            add_score 0 1
        fi
    done
}

# Check service health endpoints
check_health_endpoints() {
    log_step "Checking service health endpoints..."
    
    ENDPOINTS=(
        "http://localhost:8080/health:MCP Memory Server"
        "http://localhost:8001/health:RAG Service"
        "http://localhost:8003/health:Agent Orchestrator"
        "http://localhost:8005/health:Memory Analytics"
    )
    
    for endpoint_info in "${ENDPOINTS[@]}"; do
        IFS=':' read -r endpoint name <<< "$endpoint_info"
        
        if curl -sf "$endpoint" >/dev/null 2>&1; then
            log_info "✓ $name health check passed"
            add_score 2 2
        else
            log_error "✗ $name health check failed"
            FAILED_CHECKS+=("$name health endpoint failed")
            add_score 0 2
        fi
    done
}

# Check database connections
check_database_connections() {
    log_step "Checking database connections..."
    
    # PostgreSQL
    if docker exec postgres-memory pg_isready -U memory_user >/dev/null 2>&1; then
        log_info "✓ PostgreSQL connection successful"
        add_score 2 2
    else
        log_error "✗ PostgreSQL connection failed"
        FAILED_CHECKS+=("PostgreSQL connection failed")
        add_score 0 2
    fi
    
    # Redis
    if docker exec redis-memory redis-cli -a "${REDIS_PASSWORD:-memorypass123}" ping >/dev/null 2>&1; then
        log_info "✓ Redis connection successful"
        add_score 2 2
    else
        log_error "✗ Redis connection failed"
        FAILED_CHECKS+=("Redis connection failed")
        add_score 0 2
    fi
    
    # Neo4j
    if docker exec neo4j-memory cypher-shell -u neo4j -p "${NEO4J_PASSWORD:-memorypass123}" "RETURN 1" >/dev/null 2>&1; then
        log_info "✓ Neo4j connection successful"
        add_score 2 2
    else
        log_error "✗ Neo4j connection failed"
        FAILED_CHECKS+=("Neo4j connection failed")
        add_score 0 2
    fi
    
    # ChromaDB
    if curl -sf "http://localhost:8004/api/v1/heartbeat" >/dev/null 2>&1; then
        log_info "✓ ChromaDB connection successful"
        add_score 2 2
    else
        log_error "✗ ChromaDB connection failed"
        FAILED_CHECKS+=("ChromaDB connection failed")
        add_score 0 2
    fi
}

# Check memory operations
check_memory_operations() {
    log_step "Checking basic memory operations..."
    
    # Test memory storage
    TEST_MEMORY='{
        "content": "Health check test memory",
        "memory_type": "working",
        "agent_id": "health-check-agent",
        "session_id": "health-check-session",
        "importance": 0.5,
        "tags": ["health_check", "test"]
    }'
    
    RESPONSE=$(curl -sf -X POST "http://localhost:8080/memory/store" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer ${MCP_ACCESS_TOKEN:-secure-memory-token}" \
        -d "$TEST_MEMORY" 2>/dev/null)
    
    if [ $? -eq 0 ] && echo "$RESPONSE" | grep -q "memory_id"; then
        log_info "✓ Memory storage operation successful"
        add_score 3 3
        
        # Test memory query
        TEST_QUERY='{
            "query": "health check test",
            "agent_id": "health-check-agent",
            "session_id": "health-check-session",
            "limit": 1
        }'
        
        QUERY_RESPONSE=$(curl -sf -X POST "http://localhost:8080/memory/query" \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer ${MCP_ACCESS_TOKEN:-secure-memory-token}" \
            -d "$TEST_QUERY" 2>/dev/null)
        
        if [ $? -eq 0 ] && echo "$QUERY_RESPONSE" | grep -q "working"; then
            log_info "✓ Memory query operation successful"
            add_score 3 3
        else
            log_error "✗ Memory query operation failed"
            FAILED_CHECKS+=("Memory query failed")
            add_score 0 3
        fi
    else
        log_error "✗ Memory storage operation failed"
        FAILED_CHECKS+=("Memory storage failed")
        add_score 0 3
        add_score 0 3  # Skip query test
    fi
}

# Check system resources
check_system_resources() {
    log_step "Checking system resources..."
    
    # Memory usage
    MEMORY_USAGE=$(free | grep '^Mem:' | awk '{printf "%.1f", $3/$2 * 100.0}')
    if (( $(echo "$MEMORY_USAGE < 85.0" | bc -l) )); then
        log_info "✓ Memory usage: ${MEMORY_USAGE}%"
        add_score 2 2
    else
        log_warning "⚠ High memory usage: ${MEMORY_USAGE}%"
        add_score 1 2
    fi
    
    # Disk usage
    DISK_USAGE=$(df . | tail -1 | awk '{print $5}' | sed 's/%//')
    if [ "$DISK_USAGE" -lt 85 ]; then
        log_info "✓ Disk usage: ${DISK_USAGE}%"
        add_score 2 2
    else
        log_warning "⚠ High disk usage: ${DISK_USAGE}%"
        add_score 1 2
    fi
    
    # Docker system
    DOCKER_SYSTEM=$(docker system df --format "table {{.Type}}\t{{.TotalCount}}\t{{.Size}}" | grep -v TYPE)
    log_info "Docker system usage:"
    echo "$DOCKER_SYSTEM"
    add_score 1 1  # Basic check that docker system command works
}

# Check log errors
check_recent_errors() {
    log_step "Checking for recent errors..."
    
    ERROR_COUNT=0
    
    # Check Docker logs for errors in last hour
    CONTAINERS=("mcp-memory-server" "rag-memory-service" "agent-memory-orchestrator" "memory-analytics")
    
    for container in "${CONTAINERS[@]}"; do
        ERRORS=$(docker logs --since="1h" "$container" 2>&1 | grep -i error | wc -l)
        if [ "$ERRORS" -gt 0 ]; then
            log_warning "⚠ $container has $ERRORS error(s) in the last hour"
            ERROR_COUNT=$((ERROR_COUNT + ERRORS))
        fi
    done
    
    if [ "$ERROR_COUNT" -eq 0 ]; then
        log_info "✓ No recent errors found in container logs"
        add_score 3 3
    elif [ "$ERROR_COUNT" -lt 5 ]; then
        log_warning "⚠ Found $ERROR_COUNT recent error(s)"
        add_score 2 3
    else
        log_error "✗ Found $ERROR_COUNT recent error(s)"
        FAILED_CHECKS+=("High error count: $ERROR_COUNT")
        add_score 0 3
    fi
}

# Generate health report
generate_report() {
    log_step "Generating health report..."
    
    HEALTH_PERCENTAGE=$((HEALTH_SCORE * 100 / MAX_SCORE))
    
    echo ""
    echo "=============================================="
    echo "           MEMORY SYSTEM HEALTH REPORT"
    echo "=============================================="
    echo "Timestamp: $(date)"
    echo "Health Score: ${HEALTH_SCORE}/${MAX_SCORE} (${HEALTH_PERCENTAGE}%)"
    echo ""
    
    if [ "$HEALTH_PERCENTAGE" -ge 90 ]; then
        echo -e "${GREEN}Overall Status: EXCELLENT${NC}"
    elif [ "$HEALTH_PERCENTAGE" -ge 75 ]; then
        echo -e "${YELLOW}Overall Status: GOOD${NC}"
    elif [ "$HEALTH_PERCENTAGE" -ge 50 ]; then
        echo -e "${YELLOW}Overall Status: DEGRADED${NC}"
    else
        echo -e "${RED}Overall Status: CRITICAL${NC}"
    fi
    
    if [ ${#FAILED_CHECKS[@]} -gt 0 ]; then
        echo ""
        echo "Failed Checks:"
        for check in "${FAILED_CHECKS[@]}"; do
            echo "  - $check"
        done
    fi
    
    echo ""
    echo "=============================================="
    
    # Return appropriate exit code
    if [ "$HEALTH_PERCENTAGE" -ge 75 ]; then
        exit 0
    elif [ "$HEALTH_PERCENTAGE" -ge 50 ]; then
        exit 1
    else
        exit 2
    fi
}

# Main health check
main() {
    log_info "Starting comprehensive health check..."
    
    check_docker_services
    check_health_endpoints
    check_database_connections
    check_memory_operations
    check_system_resources
    check_recent_errors
    
    generate_report
}

# Run health check
main "$@"