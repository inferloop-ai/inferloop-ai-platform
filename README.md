# Memory-Enhanced AI Platform - File Purpose Documentation

## 🎯 **Root Directory Files**

| File | Purpose |
|------|---------|
| `README.md` | Project overview, installation instructions, and quick start guide |
| `.env` | **Environment variables** - API keys, passwords, configuration settings |
| `.env.example` | Template for environment variables (safe to commit to git) |
| `docker-compose.yml` | **Main deployment file** - orchestrates all services and dependencies |
| `setup.sh` | **Automated setup script** - creates directories, configures environment |
| `quick-start.sh` | **Fast deployment script** - starts services with health checks |
| `.gitignore` | Git ignore rules (excludes logs, secrets, temp files) |

---

## 🔧 **Services Directory (`/services/`)**

### **MCP Memory Server (`/services/mcp-memory-server/`)**
| File | Purpose |
|------|---------|
| `main.py` | **Core memory server** - FastAPI app with working/episodic/semantic/procedural memory |
| `requirements.txt` | Python dependencies (FastAPI, asyncpg, aioredis, chromadb, neo4j, etc.) |
| `Dockerfile` | Container configuration for the memory server |
| `config/memory.yaml` | Service-specific memory system configuration |
| `tests/test_memory_operations.py` | Unit tests for memory storage, retrieval, consolidation |
| `tests/test_api_endpoints.py` | API endpoint testing (store, query, consolidate) |

### **RAG Memory Service (`/services/rag-memory-service/`)**
| File | Purpose |
|------|---------|
| `main.py` | **RAG with memory integration** - retrieval-augmented generation using memory context |
| `requirements.txt` | Dependencies for document processing, embeddings, retrieval |
| `Dockerfile` | Container for RAG service |
| `models/embeddings/` | Stored embedding models for document similarity |

### **Agent Memory Orchestrator (`/services/agent-memory-orchestrator/`)**
| File | Purpose |
|------|---------|
| `main.py` | **Multi-agent coordination** - manages agent interactions and shared memory |
| `requirements.txt` | Dependencies for agent management and coordination |
| `Dockerfile` | Container for agent orchestrator |
| `agents/base_agent.py` | Base agent class with memory integration |
| `agents/memory_agent.py` | Specialized agent that focuses on memory operations |

### **Memory Analytics (`/services/memory-analytics/`)**
| File | Purpose |
|------|---------|
| `main.py` | **Analytics dashboard** - memory usage statistics, performance metrics |
| `requirements.txt` | Dependencies for data visualization and analytics |
| `Dockerfile` | Container for analytics service |
| `static/` | CSS, JavaScript, images for dashboard UI |
| `templates/` | HTML templates for analytics dashboard |

### **Memory Health Monitor (`/services/memory-health-monitor/`)**
| File | Purpose |
|------|---------|
| `main.py` | **System health monitoring** - checks all services, sends alerts |
| `requirements.txt` | Dependencies for health checks and notifications |
| `Dockerfile` | Container for health monitoring service |

---

## ⚙️ **Configuration Directory (`/config/`)**

| File | Purpose |
|------|---------|
| `redis-memory.conf` | **Redis configuration** - memory limits, persistence, security |
| `chroma-config.yaml` | **ChromaDB configuration** - vector database settings for semantic memory |
| `memory-config.yaml` | **Main memory system config** - consolidation rules, thresholds, decay rates |
| `neo4j-memory.conf` | **Neo4j configuration** - graph database settings for procedural memory |
| `logging.yaml` | **Logging configuration** - log levels, formats, output destinations |

---

## 🗄️ **Database Directory (`/database/`)**

| File | Purpose |
|------|---------|
| `init-memory.sql` | **PostgreSQL initialization** - creates extensions, users, permissions |
| `memory-schema.sql` | **Complete database schema** - tables, indexes, functions, triggers |
| `timescale-init.sql` | **TimescaleDB setup** - time-series database for temporal memory analysis |
| `seed-data.sql` | **Sample data** - test agents, memories, relationships for development |
| `migrations/001_initial_schema.sql` | Database version 1 - initial schema creation |
| `migrations/002_add_indexes.sql` | Database version 2 - performance optimization indexes |
| `migrations/003_memory_analytics.sql` | Database version 3 - analytics tables and views |

---

## 🌐 **Nginx Directory (`/nginx/`)**

| File | Purpose |
|------|---------|
| `memory-nginx.conf` | **Main Nginx config** - API gateway, load balancing, SSL, rate limiting |
| `ssl/cert.pem` | **SSL certificate** for HTTPS (production) |
| `ssl/key.pem` | **SSL private key** for HTTPS (production) |
| `logs/` | **Nginx access/error logs** directory |
| `conf.d/upstream.conf` | **Upstream server definitions** - backend service pools |
| `conf.d/security.conf` | **Security headers and policies** - CORS, CSP, etc. |

---

## 📊 **Monitoring Directory (`/monitoring/`)**

### **Prometheus (`/monitoring/prometheus/`)**
| File | Purpose |
|------|---------|
| `prometheus-memory.yml` | **Metrics collection config** - scrape targets, intervals |
| `rules/memory-alerts.yml` | **Alert rules** - memory usage, performance thresholds |
| `rules/performance-rules.yml` | **Performance alerts** - response times, error rates |

### **Grafana (`/monitoring/grafana/`)**
| File | Purpose |
|------|---------|
| `memory-dashboards/memory-overview.json` | **Main dashboard** - overall system health and memory usage |
| `memory-dashboards/agent-performance.json` | **Agent metrics** - individual agent performance and activity |
| `memory-dashboards/system-health.json` | **Infrastructure dashboard** - database, Redis, service health |
| `memory-dashboards/consolidation-metrics.json` | **Memory consolidation** - consolidation rates, effectiveness |
| `datasources/prometheus.yaml` | **Prometheus data source** configuration |
| `datasources/postgres.yaml` | **PostgreSQL data source** for memory analytics |

---

## 📁 **Data Directory (`/data/`)**

### **Documents (`/data/documents/`)**
| Directory | Purpose |
|-----------|---------|
| `knowledge-base/` | **Curated documents** for RAG - policies, procedures, knowledge articles |
| `uploads/` | **User-uploaded documents** - dynamic content for processing |

### **Models (`/data/models/`)**
| Directory | Purpose |
|-----------|---------|
| `sentence-transformers/` | **Pre-trained embedding models** - cached transformers |
| `custom-models/` | **Fine-tuned models** - domain-specific embeddings |

### **Exports (`/data/exports/`)**
| Directory | Purpose |
|-----------|---------|
| `memory-backups/` | **Memory system backups** - periodic exports of all memories |
| `analytics-reports/` | **Generated reports** - memory usage, agent performance |

---

## 🤖 **Agent Config Directory (`/agent-config/`)**

| File | Purpose |
|------|---------|
| `default-agent.yaml` | **Default agent settings** - standard memory behavior, thresholds |
| `specialized-agents/research-agent.yaml` | **Research agent config** - high memory retention, fact-focused |
| `specialized-agents/support-agent.yaml` | **Support agent config** - conversation memory, empathy settings |
| `specialized-agents/creative-agent.yaml` | **Creative agent config** - associative memory, inspiration focus |
| `memory-profiles/conservative-memory.yaml` | **Conservative profile** - slow forgetting, high importance threshold |
| `memory-profiles/aggressive-learning.yaml` | **Aggressive profile** - fast learning, low consolidation threshold |
| `memory-profiles/balanced-retention.yaml` | **Balanced profile** - standard memory decay and consolidation |

---

## 🔄 **Workflows Directory (`/workflows/`)**

| File | Purpose |
|------|---------|
| `memory-consolidation.yaml` | **Consolidation workflow** - when/how to move memories to long-term storage |
| `agent-coordination.yaml` | **Multi-agent workflow** - how agents share and coordinate memories |
| `data-processing.yaml` | **Data pipeline** - document ingestion, embedding, storage workflow |

---

## 📋 **Logs Directory (`/logs/`)**

| Directory | Purpose |
|-----------|---------|
| `mcp-server/` | **Memory server logs** - API requests, memory operations, errors |
| `memory-analytics/` | **Analytics service logs** - dashboard access, report generation |
| `health-monitor/` | **Health monitoring logs** - service checks, alerts sent |
| `consolidated/` | **Aggregated logs** - combined logs from all services |

---

## 🛠️ **Scripts Directory (`/scripts/`)**

| File | Purpose |
|------|---------|
| `backup-memories.sh` | **Backup script** - exports all memories to files |
| `restore-memories.sh` | **Restore script** - imports memories from backup files |
| `health-check.sh` | **Manual health check** - tests all services without docker |
| `performance-test.py` | **Performance testing** - load testing for memory operations |
| `data-migration.py` | **Data migration** - moves data between database versions |

---

## 🧪 **Tests Directory (`/tests/`)**

### **Unit Tests (`/tests/unit/`)**
| File | Purpose |
|------|---------|
| `test_memory_operations.py` | **Memory function tests** - store, retrieve, decay, consolidation |
| `test_consolidation.py` | **Consolidation logic tests** - importance-based routing |
| `test_vector_search.py` | **Vector similarity tests** - embedding and search accuracy |

### **Integration Tests (`/tests/integration/`)**
| File | Purpose |
|------|---------|
| `test_api_endpoints.py` | **API integration tests** - full request/response cycles |
| `test_memory_flow.py` | **Memory lifecycle tests** - working → long-term storage flow |
| `test_multi_agent.py` | **Multi-agent tests** - agent coordination and memory sharing |

### **Performance Tests (`/tests/performance/`)**
| File | Purpose |
|------|---------|
| `load_test.py` | **Load testing** - concurrent requests, high memory volume |
| `memory_benchmark.py` | **Memory performance** - consolidation speed, query response times |

### **Test Fixtures (`/tests/fixtures/`)**
| File | Purpose |
|------|---------|
| `sample_memories.json` | **Test memory data** - realistic memory examples for testing |
| `test_agents.yaml` | **Test agent configs** - agents configured for testing scenarios |

---

## 📚 **Documentation Directory (`/docs/`)**

### **API Documentation (`/docs/api/`)**
| File | Purpose |
|------|---------|
| `memory-server-api.md` | **Memory server API docs** - endpoints, parameters, examples |
| `rag-service-api.md` | **RAG service API docs** - document processing, retrieval |
| `orchestrator-api.md` | **Orchestrator API docs** - agent coordination, memory sharing |

### **Deployment (`/docs/deployment/`)**
| File | Purpose |
|------|---------|
| `production-setup.md` | **Production deployment** - SSL, scaling, security hardening |
| `kubernetes-deployment.md` | **Kubernetes guide** - K8s manifests, scaling, monitoring |
| `scaling-guide.md` | **Scaling strategies** - horizontal/vertical scaling approaches |

### **Architecture (`/docs/architecture/`)**
| File | Purpose |
|------|---------|
| `memory-system-design.md` | **System architecture** - memory types, data flow, consolidation |
| `data-flow-diagrams.md` | **Visual diagrams** - how data moves through the system |
| `security-model.md` | **Security design** - authentication, authorization, data protection |

### **Examples (`/docs/examples/`)**
| File | Purpose |
|------|---------|
| `basic-usage.py` | **Getting started code** - simple memory operations |
| `agent-integration.py` | **Agent integration** - building memory-aware agents |
| `custom-memory-types.py` | **Extensibility** - adding new memory types |

---

## 🔒 **Security Directory (`/security/`)**

### **Certificates (`/security/certificates/`)**
| Directory | Purpose |
|-----------|---------|
| SSL certificate storage | **Production certificates** for HTTPS |

### **Secrets (`/security/secrets/`)**
| File | Purpose |
|------|---------|
| `api-keys.env.template` | **API key template** - secure format for API keys |
| `database-credentials.env.template` | **Database credentials template** - secure database access |

### **Policies (`/security/policies/`)**
| File | Purpose |
|------|---------|
| `access-control.yaml` | **Access control rules** - who can access what memories |
| `data-retention.yaml` | **Data retention policy** - how long to keep different memory types |

---

## 🔧 **Tools Directory (`/tools/`)**

| File | Purpose |
|------|---------|
| `memory-inspector.py` | **Memory inspection tool** - visualize memory contents, relationships |
| `performance-profiler.py` | **Performance analysis** - identify bottlenecks, optimize queries |
| `data-visualizer.py` | **Memory visualization** - graphs of memory networks, decay patterns |
| `migration-assistant.py` | **Data migration helper** - assists with schema changes, data moves |

---

## 🎯 **Key File Categories Summary**

### **🚀 Deployment & Setup**
- `docker-compose.yml`, `setup.sh`, `.env` → **Get system running**

### **🧠 Core Memory Logic** 
- `services/mcp-memory-server/main.py` → **Heart of the memory system**

### **🗄️ Data Storage**
- `database/memory-schema.sql` → **Complete database structure**

### **⚙️ Configuration**
- `config/memory-config.yaml` → **Memory behavior settings**

### **📊 Monitoring**
- `monitoring/grafana/memory-dashboards/` → **Visual system monitoring**

### **🤖 Agent Behavior**
- `agent-config/specialized-agents/` → **Different agent personalities**

This comprehensive file structure provides everything needed for a **production-ready, memory-enhanced AI platform** with clear organization and purpose! 🎯