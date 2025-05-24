# Memory-Enhanced AI Platform Deployment Guide

## 🚀 Quick Start

This guide will help you deploy the complete Memory-Enhanced AI Platform with MCP Server.

## 📋 Prerequisites

- Docker and Docker Compose installed
- At least 16GB RAM recommended
- 50GB+ free disk space
- OpenAI API Key (optional but recommended)
- Anthropic API Key (optional but recommended)

## 🛠️ Installation Steps

### 1. Clone and Setup Directory Structure

```bash
# Create project directory
mkdir memory-enhanced-ai-platform
cd memory-enhanced-ai-platform

# Create required directory structure
mkdir -p {services/{mcp-memory-server,rag-memory-service,agent-memory-orchestrator,memory-analytics,memory-health-monitor},config,database,nginx/{ssl,logs},monitoring/{prometheus,grafana/{memory-dashboards,datasources}},logs,models,data/documents,agent-config,workflows}
```

### 2. Environment Configuration

Create `.env` file in the root directory:

```bash
# .env file
# API Keys
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here
MCP_ACCESS_TOKEN=secure-memory-token-123

# Database Passwords
REDIS_PASSWORD=secure-redis-password-123
POSTGRES_PASSWORD=secure-postgres-password-123
NEO4J_PASSWORD=secure-neo4j-password-123
TIMESCALE_PASSWORD=secure-timescale-password-123

# Admin Passwords  
GRAFANA_PASSWORD=secure-grafana-password-123
PGADMIN_PASSWORD=secure-pgadmin-password-123

# Monitoring
ALERT_WEBHOOK_URL=https://hooks.slack.com/your-webhook-url
```

### 3. Deploy Core Services

```bash
# Start the complete platform
docker-compose up -d

# Or start with development tools
docker-compose --profile development up -d

# Or start with TimescaleDB alternative
docker-compose --profile timescale up -d
```

### 4. Verify Deployment

```bash
# Check all services are running
docker-compose ps

# Check logs
docker-compose logs -f mcp-memory-server

# Test health endpoints
curl http://localhost:8080/health  # MCP Server
curl http://localhost:8001/health  # RAG Service  
curl http://localhost:8003/health  # Agent Orchestrator
```

## 🔧 Configuration Details

### Memory System Settings

The memory system can be configured via environment variables:

```bash
# Memory Configuration
WORKING_MEMORY_CAPACITY=7              # Number of items in working memory
WORKING_MEMORY_TTL=1800               # Working memory TTL in seconds
MEMORY_CONSOLIDATION_INTERVAL=21600   # Consolidation interval (6 hours)
MEMORY_CLEANUP_INTERVAL=3600          # Cleanup interval (1 hour)
EPISODIC_MEMORY_DECAY_RATE=0.99      # Memory decay rate
SEMANTIC_MEMORY_SIMILARITY_THRESHOLD=0.85  # Similarity threshold
```

### Database Initialization

The PostgreSQL database will be automatically initialized with the required schema. The initialization includes:

- Vector extension for embeddings
- Memory tables (episodic, semantic, procedural)
- Indexes for performance
- Triggers for automatic updates
- Sample data for testing

### Memory Types

The system supports four types of memory:

1. **Working Memory** (Redis) - Temporary, high-speed access
2. **Episodic Memory** (PostgreSQL + pgvector) - Event-based memories with temporal context
3. **Semantic Memory** (ChromaDB) - Concept-based knowledge storage
4. **Procedural Memory** (Neo4j) - Process and relationship knowledge

## 📊 Monitoring and Analytics

### Access Monitoring Dashboards

- **Grafana Dashboard**: http://localhost:3000 (admin/password from .env)
- **Memory Analytics**: http://localhost:8005
- **Prometheus Metrics**: http://localhost:9090
- **Redis Commander**: http://localhost:8081 (development profile)
- **pgAdmin**: http://localhost:5050 (development profile)

### Key Metrics Monitored

- Memory utilization across all types
- Query response times
- Consolidation performance
- Error rates and health status
- Agent activity and learning patterns

## 🧠 Using the Memory System

### Basic API Usage

```python
import requests
import json

# Store a memory
memory_data = {
    "content": "User prefers morning meetings and dislikes interruptions",
    "memory_type": "working",
    "agent_id": "assistant-001", 
    "session_id": "session-123",
    "importance": 0.8,
    "tags": ["preference", "scheduling"]
}

response = requests.post(
    "http://localhost:8080/memory/store",
    json=memory_data,
    headers={"Authorization": "Bearer secure-memory-token-123"}
)

# Query memories
query_data = {
    "query": "What are the user's meeting preferences?",
    "agent_id": "assistant-001",
    "session_id": "session-123",
    "memory_types": ["working", "episodic", "semantic"],
    "limit": 5
}

response = requests.post(
    "http://localhost:8080/memory/query",
    json=query_data,
    headers={"Authorization": "Bearer secure-memory-token-123"}
)

memories = response.json()
```

### Memory Consolidation

```python
# Trigger memory consolidation
consolidation_data = {
    "agent_id": "assistant-001",
    "session_id": "session-123",
    "force": True  # Force immediate consolidation
}

response = requests.post(
    "http://localhost:8080/memory/consolidate",
    json=consolidation_data,
    headers={"Authorization": "Bearer secure-memory-token-123"}
)
```

## 🔄 Memory Lifecycle

### 1. Working Memory
- New memories start in working memory (Redis)
- Limited capacity (default: 7 items)
- Fast access for immediate use
- TTL-based expiration (default: 30 minutes)

### 2. Consolidation Process
- Automatic consolidation every 6 hours
- Importance-based routing to long-term storage
- High importance → All memory types
- Medium importance → Episodic + Semantic  
- Low importance → Semantic only

### 3. Long-term Storage
- **Episodic**: Time-ordered events with decay
- **Semantic**: Conceptual knowledge with similarity search
- **Procedural**: Structured knowledge graphs

### 4. Retrieval and Decay
- Memories decay over time based on access patterns
- Frequently accessed memories maintain higher importance
- Automatic cleanup of very old, low-importance memories

## 🛡️ Security Considerations

### Production Deployment

1. **Enable Authentication**:
   ```yaml
   # In config/chroma-config.yaml
   auth:
     enabled: true
     provider: "basic"
     credentials:
       - username: "memory_user"
         password: "secure_password"
   ```

2. **Use HTTPS**:
   - Uncomment HTTPS server block in nginx configuration
   - Add SSL certificates to `nginx/ssl/`

3. **Secure Database Connections**:
   - Use strong passwords
   - Enable SSL/TLS for database connections
   - Restrict network access

4. **Rate Limiting**:
   - Already configured in Nginx
   - Adjust limits based on your requirements

## 🐛 Troubleshooting

### Common Issues

1. **Services not starting**:
   ```bash
   # Check logs
   docker-compose logs [service-name]
   
   # Restart specific service
   docker-compose restart [service-name]
   ```

2. **Memory errors**:
   ```bash
   # Check system resources
   docker stats
   
   # Increase memory limits in docker-compose.yml
   ```

3. **Database connection issues**:
   ```bash
   # Check database health
   docker-compose exec postgres-memory pg_isready -U memory_user
   
   # Reset database
   docker-compose down -v
   docker-compose up -d
   ```

4. **Vector similarity not working**:
   ```bash
   # Ensure pgvector extension is installed
   docker-compose exec postgres-memory psql -U memory_user -d memories -c "CREATE EXTENSION IF NOT EXISTS vector;"
   ```

### Performance Tuning

1. **Database Optimization**:
   - Adjust PostgreSQL memory settings in docker-compose.yml
   - Monitor index usage and add custom indexes as needed

2. **Redis Optimization**:
   - Tune memory policy in redis-memory.conf
   - Monitor memory usage and eviction patterns

3. **Vector Search Optimization**:
   - Adjust HNSW parameters in ChromaDB config
   - Consider using GPU acceleration for embeddings

## 📈 Scaling

### Horizontal Scaling

```bash
# Scale MCP servers
docker-compose up -d --scale mcp-memory-server=3

# Add load balancer endpoints in nginx configuration
```

### Vertical Scaling

- Increase memory limits in docker-compose.yml
- Adjust CPU allocations based on workload
- Monitor resource usage with Grafana dashboards

## 🔧 Development Setup

### Running with Development Tools

```bash
# Start with development profile
docker-compose --profile development up -d

# This includes:
# - Redis Commander (http://localhost:8081)
# - pgAdmin (http://localhost:5050)
# - Additional debugging tools
```

### Custom Service Development

1. Create your service in `services/your-service/`
2. Add Dockerfile and requirements
3. Update docker-compose.yml with your service
4. Connect to memory system via HTTP API

## 📚 API Documentation

Once deployed, access the interactive API documentation:
- **MCP Server API**: http://localhost:8080/docs
- **RAG Service API**: http://localhost:8001/docs
- **Agent Orchestrator API**: http://localhost:8003/docs

## 🧪 Testing

### Basic Functionality Test

```bash
# Create test script
cat << 'EOF' > test_memory_system.py
#!/usr/bin/env python3
import requests
import json
import time

BASE_URL = "http://localhost:8080"
TOKEN = "secure-memory-token-123"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

def test_store_memory():
    memory = {
        "content": "Test memory content",
        "memory_type": "working",
        "agent_id": "test-agent",
        "session_id": "test-session",
        "importance": 0.7
    }
    
    response = requests.post(f"{BASE_URL}/memory/store", json=memory, headers=HEADERS)
    print(f"Store Memory: {response.status_code} - {response.json()}")
    return response.json().get("memory_id")

def test_query_memory():
    query = {
        "query": "test content",
        "agent_id": "test-agent", 
        "session_id": "test-session"
    }
    
    response = requests.post(f"{BASE_URL}/memory/query", json=query, headers=HEADERS)
    print(f"Query Memory: {response.status_code} - {response.json()}")

def test_consolidation():
    consolidation = {
        "agent_id": "test-agent",
        "session_id": "test-session",
        "force": True
    }
    
    response = requests.post(f"{BASE_URL}/memory/consolidate", json=consolidation, headers=HEADERS)
    print(f"Consolidation: {response.status_code} - {response.json()}")

if __name__ == "__main__":
    print("Testing Memory System...")
    test_store_memory()
    time.sleep(1)
    test_query_memory()
    time.sleep(1)
    test_consolidation()
    print("Testing complete!")
EOF

# Run test
python3 test_memory_system.py
```

## 📞 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review logs: `docker-compose logs -f [service-name]`
3. Monitor system health via Grafana dashboards
4. Check service health endpoints

---

## 🎉 Congratulations!

Your Memory-Enhanced AI Platform is now deployed and ready for use. The system provides:

✅ **Multi-modal memory storage** (Working, Episodic, Semantic, Procedural)  
✅ **Automatic memory consolidation** with intelligent routing  
✅ **Vector-based similarity search** for semantic retrieval  
✅ **Knowledge graph storage** for procedural memory  
✅ **Comprehensive monitoring** and analytics  
✅ **Scalable architecture** with load balancing  
✅ **RESTful API** for easy integration  

Start building memory-enhanced AI agents that can learn, remember, and improve over time!