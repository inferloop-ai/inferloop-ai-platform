# services/memory-health-monitor/main.py
"""
Memory System Health Monitor
Monitors all memory system components and provides alerts
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import aioredis
import asyncpg
import httpx
import neo4j
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthStatus(BaseModel):
    service: str
    status: str  # 'healthy', 'degraded', 'unhealthy'
    response_time_ms: Optional[float] = None
    error_message: Optional[str] = None
    timestamp: datetime
    metadata: Dict = {}

class SystemHealth(BaseModel):
    overall_status: str
    services: List[HealthStatus]
    timestamp: datetime
    summary: Dict

class MemoryHealthMonitor:
    def __init__(self):
        self.redis_url = os.getenv("REDIS_URL", "redis://:memorypass123@redis-memory:6379/0")
        self.postgres_url = os.getenv("POSTGRES_URL", "postgresql://memory_user:memorypass123@postgres-memory:5432/memories")
        self.chroma_url = os.getenv("CHROMA_URL", "http://chroma-memory:8000")
        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://neo4j-memory:7687")
        self.neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", "memorypass123")
        self.check_interval = int(os.getenv("CHECK_INTERVAL", "60"))
        self.alert_webhook_url = os.getenv("ALERT_WEBHOOK_URL")
        
        self.redis_client = None
        self.postgres_pool = None
        self.neo4j_driver = None
        self.http_client = httpx.AsyncClient(timeout=10.0)
        
        self.last_health_status = {}

    async def initialize(self):
        """Initialize connections to all services"""
        try:
            # Redis connection
            self.redis_client = aioredis.from_url(self.redis_url, decode_responses=True)
            await self.redis_client.ping()
            
            # PostgreSQL connection pool
            self.postgres_pool = await asyncpg.create_pool(self.postgres_url, min_size=1, max_size=3)
            
            # Neo4j driver
            self.neo4j_driver = neo4j.AsyncGraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_username, self.neo4j_password)
            )
            
            logger.info("Health monitor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize health monitor: {e}")
            raise

    async def cleanup(self):
        """Cleanup all connections"""
        if self.redis_client:
            await self.redis_client.close()
        if self.postgres_pool:
            await self.postgres_pool.close()
        if self.neo4j_driver:
            await self.neo4j_driver.close()
        await self.http_client.aclose()

    async def check_redis_health(self) -> HealthStatus:
        """Check Redis health"""
        start_time = time.time()
        try:
            # Test basic operations
            await self.redis_client.ping()
            await self.redis_client.set("health_check", "test", ex=60)
            result = await self.redis_client.get("health_check")
            
            # Get Redis info
            info = await self.redis_client.info()
            memory_usage = info.get('used_memory_human', 'unknown')
            connected_clients = info.get('connected_clients', 0)
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthStatus(
                service="redis",
                status="healthy",
                response_time_ms=response_time,
                timestamp=datetime.utcnow(),
                metadata={
                    "memory_usage": memory_usage,
                    "connected_clients": connected_clients,
                    "uptime_seconds": info.get('uptime_in_seconds', 0)
                }
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthStatus(
                service="redis",
                status="unhealthy",
                response_time_ms=response_time,
                error_message=str(e),
                timestamp=datetime.utcnow()
            )

    async def check_postgres_health(self) -> HealthStatus:
        """Check PostgreSQL health"""
        start_time = time.time()
        try:
            async with self.postgres_pool.acquire() as conn:
                # Test basic query
                result = await conn.fetchval("SELECT 1")
                
                # Check memory table
                memory_count = await conn.fetchval("SELECT COUNT(*) FROM episodic_memories")
                
                # Check database size
                db_size = await conn.fetchval("""
                    SELECT pg_size_pretty(pg_database_size(current_database()))
                """)
                
                response_time = (time.time() - start_time) * 1000
                
                return HealthStatus(
                    service="postgresql",
                    status="healthy",
                    response_time_ms=response_time,
                    timestamp=datetime.utcnow(),
                    metadata={
                        "episodic_memory_count": memory_count,
                        "database_size": db_size,
                        "connection_pool_size": self.postgres_pool.get_size()
                    }
                )
                
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthStatus(
                service="postgresql",
                status="unhealthy",
                response_time_ms=response_time,
                error_message=str(e),
                timestamp=datetime.utcnow()
            )

    async def check_chroma_health(self) -> HealthStatus:
        """Check ChromaDB health"""
        start_time = time.time()
        try:
            # Test ChromaDB heartbeat
            response = await self.http_client.get(f"{self.chroma_url}/api/v1/heartbeat")
            response.raise_for_status()
            
            # Test collection operations
            collections_response = await self.http_client.get(f"{self.chroma_url}/api/v1/collections")
            collections_data = collections_response.json()
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthStatus(
                service="chromadb",
                status="healthy",
                response_time_ms=response_time,
                timestamp=datetime.utcnow(),
                metadata={
                    "collections_count": len(collections_data),
                    "status": "online"
                }
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthStatus(
                service="chromadb",
                status="unhealthy",
                response_time_ms=response_time,
                error_message=str(e),
                timestamp=datetime.utcnow()
            )

    async def check_neo4j_health(self) -> HealthStatus:
        """Check Neo4j health"""
        start_time = time.time()
        try:
            async with self.neo4j_driver.session() as session:
                # Test basic query
                result = await session.run("RETURN 1 as test")
                await result.single()
                
                # Get node count
                node_result = await session.run("MATCH (n) RETURN count(n) as node_count")
                node_count = (await node_result.single())["node_count"]
                
                # Get relationship count
                rel_result = await session.run("MATCH ()-[r]->() RETURN count(r) as rel_count")
                rel_count = (await rel_result.single())["rel_count"]
                
                response_time = (time.time() - start_time) * 1000
                
                return HealthStatus(
                    service="neo4j",
                    status="healthy",
                    response_time_ms=response_time,
                    timestamp=datetime.utcnow(),
                    metadata={
                        "node_count": node_count,
                        "relationship_count": rel_count
                    }
                )
                
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthStatus(
                service="neo4j",
                status="unhealthy",
                response_time_ms=response_time,
                error_message=str(e),
                timestamp=datetime.utcnow()
            )

    async def check_mcp_server_health(self) -> HealthStatus:
        """Check MCP Memory Server health"""
        start_time = time.time()
        try:
            response = await self.http_client.get("http://mcp-memory-server:8080/health")
            response.raise_for_status()
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthStatus(
                service="mcp-memory-server",
                status="healthy",
                response_time_ms=response_time,
                timestamp=datetime.utcnow(),
                metadata=response.json()
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthStatus(
                service="mcp-memory-server",
                status="unhealthy",
                response_time_ms=response_time,
                error_message=str(e),
                timestamp=datetime.utcnow()
            )

    async def perform_health_check(self) -> SystemHealth:
        """Perform comprehensive health check"""
        logger.info("Starting health check...")
        
        # Run all health checks concurrently
        health_checks = await asyncio.gather(
            self.check_redis_health(),
            self.check_postgres_health(),
            self.check_chroma_health(),
            self.check_neo4j_health(),
            self.check_mcp_server_health(),
            return_exceptions=True
        )
        
        # Filter out exceptions and create health statuses
        services = []
        for check in health_checks:
            if isinstance(check, HealthStatus):
                services.append(check)
            elif isinstance(check, Exception):
                logger.error(f"Health check failed: {check}")
                services.append(HealthStatus(
                    service="unknown",
                    status="unhealthy",
                    error_message=str(check),
                    timestamp=datetime.utcnow()
                ))
        
        # Determine overall status
        healthy_count = sum(1 for s in services if s.status == "healthy")
        unhealthy_count = sum(1 for s in services if s.status == "unhealthy")
        degraded_count = sum(1 for s in services if s.status == "degraded")
        
        if unhealthy_count == 0 and degraded_count == 0:
            overall_status = "healthy"
        elif unhealthy_count > 0:
            overall_status = "unhealthy"
        else:
            overall_status = "degraded"
        
        # Calculate summary statistics
        avg_response_time = sum(s.response_time_ms for s in services if s.response_time_ms) / len([s for s in services if s.response_time_ms])
        
        summary = {
            "total_services": len(services),
            "healthy_services": healthy_count,
            "degraded_services": degraded_count,
            "unhealthy_services": unhealthy_count,
            "average_response_time_ms": round(avg_response_time, 2)
        }
        
        system_health = SystemHealth(
            overall_status=overall_status,
            services=services,
            timestamp=datetime.utcnow(),
            summary=summary
        )
        
        logger.info(f"Health check completed: {overall_status} ({healthy_count}/{len(services)} services healthy)")
        return system_health

    async def send_alert(self, message: str, severity: str = "warning"):
        """Send alert notification"""
        if not self.alert_webhook_url:
            logger.warning(f"Alert (no webhook configured): {message}")
            return
        
        try:
            alert_payload = {
                "text": f"🚨 Memory System Alert - {severity.upper()}",
                "attachments": [{
                    "color": "danger" if severity == "critical" else "warning",
                    "fields": [{
                        "title": "Message",
                        "value": message,
                        "short": False
                    }, {
                        "title": "Timestamp",
                        "value": datetime.utcnow().isoformat(),
                        "short": True
                    }]
                }]
            }
            
            response = await self.http_client.post(
                self.alert_webhook_url,
                json=alert_payload
            )
            response.raise_for_status()
            logger.info(f"Alert sent successfully: {message}")
            
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")

    async def check_for_alerts(self, current_health: SystemHealth):
        """Check if alerts need to be sent"""
        # Check for status changes
        for service in current_health.services:
            service_name = service.service
            current_status = service.status
            previous_status = self.last_health_status.get(service_name)
            
            if previous_status and previous_status != current_status:
                if current_status == "unhealthy":
                    await self.send_alert(
                        f"Service {service_name} is now UNHEALTHY: {service.error_message}",
                        "critical"
                    )
                elif current_status == "healthy" and previous_status == "unhealthy":
                    await self.send_alert(
                        f"Service {service_name} has RECOVERED and is now healthy",
                        "info"
                    )
            
            self.last_health_status[service_name] = current_status
        
        # Check for overall system health
        if current_health.overall_status == "unhealthy":
            unhealthy_services = [s.service for s in current_health.services if s.status == "unhealthy"]
            await self.send_alert(
                f"Memory system is UNHEALTHY. Failed services: {', '.join(unhealthy_services)}",
                "critical"
            )

    async def log_health_metrics(self, health: SystemHealth):
        """Log health metrics to Redis for monitoring"""
        try:
            metrics_key = f"health_metrics:{int(time.time())}"
            await self.redis_client.setex(
                metrics_key,
                3600,  # Keep for 1 hour
                health.model_dump_json()
            )
            
            # Keep only last 100 metrics
            pattern = "health_metrics:*"
            keys = await self.redis_client.keys(pattern)
            if len(keys) > 100:
                oldest_keys = sorted(keys)[:len(keys)-100]
                await self.redis_client.delete(*oldest_keys)
                
        except Exception as e:
            logger.error(f"Failed to log health metrics: {e}")

    async def run_monitoring_loop(self):
        """Main monitoring loop"""
        logger.info(f"Starting health monitoring loop (interval: {self.check_interval}s)")
        
        while True:
            try:
                # Perform health check
                health = await self.perform_health_check()
                
                # Check for alerts
                await self.check_for_alerts(health)
                
                # Log metrics
                await self.log_health_metrics(health)
                
                # Print summary
                logger.info(f"System Status: {health.overall_status} | "
                          f"Services: {health.summary['healthy_services']}/{health.summary['total_services']} healthy | "
                          f"Avg Response: {health.summary['average_response_time_ms']}ms")
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await self.send_alert(f"Health monitoring error: {str(e)}", "critical")
            
            # Wait for next check
            await asyncio.sleep(self.check_interval)

async def main():
    """Main function"""
    monitor = MemoryHealthMonitor()
    
    try:
        await monitor.initialize()
        await monitor.run_monitoring_loop()
    except KeyboardInterrupt:
        logger.info("Shutting down health monitor...")
    except Exception as e:
        logger.error(f"Fatal error in health monitor: {e}")
    finally:
        await monitor.cleanup()

if __name__ == "__main__":
    asyncio.run(main())

# =====================================================
# services/memory-health-monitor/requirements.txt
# =====================================================

aioredis==2.0.1
asyncpg==0.29.0
httpx==0.25.2
neo4j==5.14.1
pydantic==2.5.0
asyncio-mqtt==0.13.0

# =====================================================
# services/memory-health-monitor/Dockerfile
# =====================================================

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY main.py .

# Health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import asyncio; print('Health monitor running')" || exit 1

# Run the health monitor
CMD ["python", "main.py"]