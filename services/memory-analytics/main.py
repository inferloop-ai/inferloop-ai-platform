# services/memory-analytics/main.py
"""
Memory Analytics Service
Real-time Analytics and Dashboards for Memory System
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import aioredis
import asyncpg
import httpx
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================================
# Configuration & Models
# ==============================================

class AnalyticsConfig:
    def __init__(self):
        self.memory_system_url = os.getenv("MEMORY_SYSTEM_URL", "http://mcp-memory-server:8080")
        self.postgres_url = os.getenv("POSTGRES_URL", "postgresql://memory_user:memorypass123@postgres-memory:5432/memories")
        self.redis_url = os.getenv("REDIS_URL", "redis://:memorypass123@redis-memory:6379/0")
        self.refresh_interval = int(os.getenv("REFRESH_INTERVAL", "300"))  # 5 minutes
        self.data_retention_days = int(os.getenv("DATA_RETENTION_DAYS", "30"))

config = AnalyticsConfig()

class AnalyticsQuery(BaseModel):
    metric_type: str  # "memory_usage", "agent_performance", "consolidation_stats", "system_health"
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    time_range: str = "24h"  # "1h", "24h", "7d", "30d"
    granularity: str = "hour"  # "minute", "hour", "day"

class MemoryMetrics(BaseModel):
    total_memories: int
    working_memory_count: int
    episodic_memory_count: int
    semantic_memory_count: int
    procedural_memory_count: int
    avg_importance: float
    memory_growth_rate: float
    consolidation_rate: float

class AgentMetrics(BaseModel):
    agent_id: str
    total_memories: int
    avg_memory_importance: float
    consolidation_frequency: float
    last_activity: datetime
    memory_types_distribution: Dict[str, int]
    performance_score: float

# ==============================================
# Analytics Service
# ==============================================

class MemoryAnalyticsService:
    def __init__(self):
        self.postgres_pool = None
        self.redis_client = None
        self.http_client = httpx.AsyncClient(timeout=30.0)
        self.templates = Jinja2Templates(directory="templates")
        
    async def initialize(self):
        """Initialize database connections"""
        # PostgreSQL connection pool
        self.postgres_pool = await asyncpg.create_pool(config.postgres_url)
        
        # Redis connection
        self.redis_client = aioredis.from_url(config.redis_url, decode_responses=True)
        
        logger.info("Memory Analytics Service initialized")
    
    async def cleanup(self):
        """Cleanup connections"""
        if self.postgres_pool:
            await self.postgres_pool.close()
        if self.redis_client:
            await self.redis_client.close()
        await self.http_client.aclose()

    # ==============================================
    # Data Collection
    # ==============================================
    
    async def get_memory_statistics(self, time_range: str = "24h") -> MemoryMetrics:
        """Get overall memory system statistics"""
        time_filter = self._get_time_filter(time_range)
        
        async with self.postgres_pool.acquire() as conn:
            # Total memory counts
            total_memories = await conn.fetchval(
                "SELECT COUNT(*) FROM episodic_memories WHERE created_at >= $1",
                time_filter
            )
            
            # Memory type distribution
            memory_type_counts = await conn.fetch("""
                SELECT 
                    'episodic' as type, COUNT(*) as count 
                FROM episodic_memories WHERE created_at >= $1
                UNION ALL
                SELECT 
                    'semantic' as type, COUNT(*) as count 
                FROM semantic_concepts WHERE created_at >= $1
            """, time_filter)
            
            # Average importance
            avg_importance = await conn.fetchval(
                "SELECT AVG(importance) FROM episodic_memories WHERE created_at >= $1",
                time_filter
            ) or 0.0
            
            # Growth rate (memories created in last period vs previous period)
            previous_period = time_filter - (datetime.utcnow() - time_filter)
            current_count = await conn.fetchval(
                "SELECT COUNT(*) FROM episodic_memories WHERE created_at >= $1",
                time_filter
            )
            previous_count = await conn.fetchval(
                "SELECT COUNT(*) FROM episodic_memories WHERE created_at >= $1 AND created_at < $2",
                previous_period, time_filter
            )
            
            growth_rate = ((current_count - previous_count) / max(previous_count, 1)) * 100 if previous_count else 0
            
            # Consolidation stats
            consolidation_stats = await conn.fetchrow("""
                SELECT 
                    COUNT(*) as total_consolidations,
                    AVG(memories_consolidated::float / NULLIF(memories_processed, 0)) as avg_consolidation_rate
                FROM memory_consolidation_log 
                WHERE started_at >= $1 AND status = 'completed'
            """, time_filter)
            
        # Working memory count from Redis
        working_memory_keys = await self.redis_client.keys("working:*")
        working_memory_count = len(working_memory_keys)
        
        return MemoryMetrics(
            total_memories=total_memories,
            working_memory_count=working_memory_count,
            episodic_memory_count=next((row['count'] for row in memory_type_counts if row['type'] == 'episodic'), 0),
            semantic_memory_count=next((row['count'] for row in memory_type_counts if row['type'] == 'semantic'), 0),
            procedural_memory_count=0,  # Would come from Neo4j
            avg_importance=round(avg_importance, 3),
            memory_growth_rate=round(growth_rate, 2),
            consolidation_rate=round(consolidation_stats['avg_consolidation_rate'] or 0, 3)
        )
    
    async def get_agent_metrics(self, agent_id: str = None, time_range: str = "24h") -> List[AgentMetrics]:
        """Get agent-specific metrics"""
        time_filter = self._get_time_filter(time_range)
        
        async with self.postgres_pool.acquire() as conn:
            if agent_id:
                # Single agent metrics
                agent_stats = await conn.fetchrow("""
                    SELECT 
                        agent_id,
                        COUNT(*) as total_memories,
                        AVG(importance) as avg_importance,
                        MAX(created_at) as last_activity
                    FROM episodic_memories 
                    WHERE agent_id = $1 AND created_at >= $2
                    GROUP BY agent_id
                """, agent_id, time_filter)
                
                if not agent_stats:
                    return []
                
                # Memory type distribution
                type_distribution = await conn.fetch("""
                    SELECT 
                        'episodic' as memory_type, COUNT(*) as count
                    FROM episodic_memories 
                    WHERE agent_id = $1 AND created_at >= $2
                    UNION ALL
                    SELECT 
                        'semantic' as memory_type, COUNT(*) as count
                    FROM semantic_concepts 
                    WHERE agent_id = $1 AND created_at >= $2
                """, agent_id, time_filter)
                
                # Consolidation frequency
                consolidation_freq = await conn.fetchval("""
                    SELECT COUNT(*) 
                    FROM memory_consolidation_log 
                    WHERE agent_id = $1 AND started_at >= $2
                """, agent_id, time_filter) or 0
                
                distribution = {row['memory_type']: row['count'] for row in type_distribution}
                
                return [AgentMetrics(
                    agent_id=agent_stats['agent_id'],
                    total_memories=agent_stats['total_memories'],
                    avg_memory_importance=round(agent_stats['avg_importance'], 3),
                    consolidation_frequency=consolidation_freq,
                    last_activity=agent_stats['last_activity'],
                    memory_types_distribution=distribution,
                    performance_score=self._calculate_performance_score(agent_stats, consolidation_freq)
                )]
            
            else:
                # All agents metrics
                all_agents = await conn.fetch("""
                    SELECT 
                        agent_id,
                        COUNT(*) as total_memories,
                        AVG(importance) as avg_importance,
                        MAX(created_at) as last_activity
                    FROM episodic_memories 
                    WHERE created_at >= $1
                    GROUP BY agent_id
                    ORDER BY total_memories DESC
                """, time_filter)
                
                metrics = []
                for agent_stat in all_agents:
                    # Get consolidation frequency for each agent
                    consolidation_freq = await conn.fetchval("""
                        SELECT COUNT(*) 
                        FROM memory_consolidation_log 
                        WHERE agent_id = $1 AND started_at >= $2
                    """, agent_stat['agent_id'], time_filter) or 0
                    
                    metrics.append(AgentMetrics(
                        agent_id=agent_stat['agent_id'],
                        total_memories=agent_stat['total_memories'],
                        avg_memory_importance=round(agent_stat['avg_importance'], 3),
                        consolidation_frequency=consolidation_freq,
                        last_activity=agent_stat['last_activity'],
                        memory_types_distribution={},  # Simplified for performance
                        performance_score=self._calculate_performance_score(agent_stat, consolidation_freq)
                    ))
                
                return metrics
    
    async def get_system_health_metrics(self) -> Dict[str, Any]:
        """Get system health and performance metrics"""
        async with self.postgres_pool.acquire() as conn:
            # Database performance
            db_stats = await conn.fetchrow("""
                SELECT 
                    pg_database_size(current_database()) as db_size,
                    (SELECT COUNT(*) FROM episodic_memories) as total_memories,
                    (SELECT COUNT(*) FROM memory_consolidation_log WHERE status = 'failed') as failed_consolidations
            """)
            
            # Recent errors
            recent_errors = await conn.fetch("""
                SELECT error_message, COUNT(*) as error_count
                FROM memory_consolidation_log 
                WHERE status = 'failed' AND started_at >= $1
                GROUP BY error_message
                ORDER BY error_count DESC
                LIMIT 10
            """, datetime.utcnow() - timedelta(hours=24))
        
        # Redis stats
        redis_info = await self.redis_client.info()
        
        # Memory system health check
        try:
            health_response = await self.http_client.get(
                f"{config.memory_system_url}/health",
                timeout=5.0
            )
            memory_system_healthy = health_response.status_code == 200
        except:
            memory_system_healthy = False
        
        return {
            "database": {
                "size_bytes": db_stats['db_size'],
                "total_memories": db_stats['total_memories'],
                "failed_consolidations": db_stats['failed_consolidations']
            },
            "redis": {
                "used_memory": redis_info.get('used_memory_human', 'Unknown'),
                "connected_clients": redis_info.get('connected_clients', 0),
                "uptime_seconds": redis_info.get('uptime_in_seconds', 0)
            },
            "memory_system": {
                "healthy": memory_system_healthy
            },
            "recent_errors": [dict(row) for row in recent_errors]
        }
    
    # ==============================================
    # Visualization Generation
    # ==============================================
    
    async def create_memory_usage_chart(self, time_range: str = "24h") -> Dict[str, Any]:
        """Create memory usage over time chart"""
        time_filter = self._get_time_filter(time_range)
        granularity = self._get_granularity(time_range)
        
        async with self.postgres_pool.acquire() as conn:
            data = await conn.fetch(f"""
                SELECT 
                    DATE_TRUNC('{granularity}', created_at) as time_bucket,
                    COUNT(*) as memory_count,
                    AVG(importance) as avg_importance
                FROM episodic_memories 
                WHERE created_at >= $1
                GROUP BY time_bucket
                ORDER BY time_bucket
            """, time_filter)
        
        if not data:
            return {}
        
        df = pd.DataFrame([dict(row) for row in data])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['time_bucket'],
            y=df['memory_count'],
            mode='lines+markers',
            name='Memory Count',
            yaxis='y'
        ))
        
        fig.add_trace(go.Scatter(
            x=df['time_bucket'],
            y=df['avg_importance'],
            mode='lines+markers',
            name='Avg Importance',
            yaxis='y2'
        ))
        
        fig.update_layout(
            title=f'Memory Usage Over Time ({time_range})',
            xaxis_title='Time',
            yaxis=dict(title='Memory Count', side='left'),
            yaxis2=dict(title='Average Importance', side='right', overlaying='y'),
            hovermode='x unified'
        )
        
        return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
    
    async def create_agent_performance_chart(self, time_range: str = "24h") -> Dict[str, Any]:
        """Create agent performance comparison chart"""
        agent_metrics = await self.get_agent_metrics(time_range=time_range)
        
        if not agent_metrics:
            return {}
        
        agents = [m.agent_id for m in agent_metrics]
        memory_counts = [m.total_memories for m in agent_metrics]
        importance_scores = [m.avg_memory_importance for m in agent_metrics]
        performance_scores = [m.performance_score for m in agent_metrics]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Memory Count',
            x=agents,
            y=memory_counts,
            yaxis='y'
        ))
        
        fig.add_trace(go.Scatter(
            name='Performance Score',
            x=agents,
            y=performance_scores,
            mode='lines+markers',
            yaxis='y2'
        ))
        
        fig.update_layout(
            title=f'Agent Performance Comparison ({time_range})',
            xaxis_title='Agents',
            yaxis=dict(title='Memory Count', side='left'),
            yaxis2=dict(title='Performance Score', side='right', overlaying='y', range=[0, 100]),
            barmode='group'
        )
        
        return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
    
    async def create_memory_types_pie_chart(self) -> Dict[str, Any]:
        """Create memory types distribution pie chart"""
        metrics = await self.get_memory_statistics()
        
        labels = ['Working Memory', 'Episodic Memory', 'Semantic Memory', 'Procedural Memory']
        values = [
            metrics.working_memory_count,
            metrics.episodic_memory_count,
            metrics.semantic_memory_count,
            metrics.procedural_memory_count
        ]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.3
        )])
        
        fig.update_layout(
            title='Memory Types Distribution',
            annotations=[dict(text='Memory<br>Types', x=0.5, y=0.5, font_size=20, showarrow=False)]
        )
        
        return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))
    
    # ==============================================
    # Helper Methods
    # ==============================================
    
    def _get_time_filter(self, time_range: str) -> datetime:
        """Convert time range string to datetime filter"""
        now = datetime.utcnow()
        if time_range == "1h":
            return now - timedelta(hours=1)
        elif time_range == "24h":
            return now - timedelta(hours=24)
        elif time_range == "7d":
            return now - timedelta(days=7)
        elif time_range == "30d":
            return now - timedelta(days=30)
        else:
            return now - timedelta(hours=24)
    
    def _get_granularity(self, time_range: str) -> str:
        """Get appropriate time granularity for range"""
        if time_range == "1h":
            return "minute"
        elif time_range == "24h":
            return "hour"
        elif time_range in ["7d", "30d"]:
            return "day"
        else:
            return "hour"
    
    def _calculate_performance_score(self, agent_stats: Dict, consolidation_freq: int) -> float:
        """Calculate agent performance score"""
        # Simple scoring algorithm
        memory_score = min(agent_stats['total_memories'] / 100 * 30, 30)  # Max 30 points
        importance_score = agent_stats['avg_importance'] * 40  # Max 40 points
        consolidation_score = min(consolidation_freq / 10 * 30, 30)  # Max 30 points
        
        return round(memory_score + importance_score + consolidation_score, 1)

# ==============================================
# FastAPI Application
# ==============================================

app = FastAPI(
    title="Memory Analytics Service",
    description="Real-time Analytics and Dashboards for Memory System",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")

# Security
security = HTTPBearer()
access_token = os.getenv("ACCESS_TOKEN", "secure-memory-token")

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != access_token:
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials

# Global analytics service
analytics_service = MemoryAnalyticsService()

@app.on_event("startup")
async def startup_event():
    await analytics_service.initialize()

@app.on_event("shutdown")
async def shutdown_event():
    await analytics_service.cleanup()

# ==============================================
# API Endpoints
# ==============================================

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "service": "memory-analytics"
    }

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main analytics dashboard"""
    return analytics_service.templates.TemplateResponse(
        "dashboard.html",
        {"request": request, "title": "Memory System Analytics"}
    )

@app.get("/api/metrics/memory")
async def get_memory_metrics(
    time_range: str = "24h",
    credentials: HTTPAuthorizationCredentials = Depends(verify_token)
):
    """Get memory system metrics"""
    metrics = await analytics_service.get_memory_statistics(time_range)
    return metrics

@app.get("/api/metrics/agents")
async def get_agent_metrics(
    agent_id: Optional[str] = None,
    time_range: str = "24h",
    credentials: HTTPAuthorizationCredentials = Depends(verify_token)
):
    """Get agent performance metrics"""
    metrics = await analytics_service.get_agent_metrics(agent_id, time_range)
    return {"agents": metrics}

@app.get("/api/metrics/system-health")
async def get_system_health(
    credentials: HTTPAuthorizationCredentials = Depends(verify_token)
):
    """Get system health metrics"""
    health = await analytics_service.get_system_health_metrics()
    return health

@app.get("/api/charts/memory-usage")
async def get_memory_usage_chart(
    time_range: str = "24h",
    credentials: HTTPAuthorizationCredentials = Depends(verify_token)
):
    """Get memory usage chart data"""
    chart = await analytics_service.create_memory_usage_chart(time_range)
    return chart

@app.get("/api/charts/agent-performance")
async def get_agent_performance_chart(
    time_range: str = "24h",
    credentials: HTTPAuthorizationCredentials = Depends(verify_token)
):
    """Get agent performance chart data"""
    chart = await analytics_service.create_agent_performance_chart(time_range)
    return chart

@app.get("/api/charts/memory-types")
async def get_memory_types_chart(
    credentials: HTTPAuthorizationCredentials = Depends(verify_token)
):
    """Get memory types distribution chart"""
    chart = await analytics_service.create_memory_types_pie_chart()
    return chart

@app.get("/api/export/csv")
async def export_data_csv(
    table: str = "episodic_memories",
    time_range: str = "24h",
    credentials: HTTPAuthorizationCredentials = Depends(verify_token)
):
    """Export data as CSV"""
    if table not in ["episodic_memories", "memory_consolidation_log", "agent_memory_profiles"]:
        raise HTTPException(status_code=400, detail="Invalid table name")
    
    time_filter = analytics_service._get_time_filter(time_range)
    
    async with analytics_service.postgres_pool.acquire() as conn:
        if table == "episodic_memories":
            data = await conn.fetch("""
                SELECT agent_id, content, importance, emotional_valence, 
                       tags, created_at, last_accessed, access_count
                FROM episodic_memories 
                WHERE created_at >= $1
                ORDER BY created_at DESC
            """, time_filter)
        elif table == "memory_consolidation_log":
            data = await conn.fetch("""
                SELECT agent_id, session_id, consolidation_type, 
                       memories_processed, memories_consolidated, 
                       started_at, completed_at, status
                FROM memory_consolidation_log 
                WHERE started_at >= $1
                ORDER BY started_at DESC
            """, time_filter)
        else:
            data = await conn.fetch("SELECT * FROM agent_memory_profiles")
    
    # Convert to CSV format
    df = pd.DataFrame([dict(row) for row in data])
    csv_data = df.to_csv(index=False)
    
    return JSONResponse(
        content={"csv_data": csv_data, "filename": f"{table}_{time_range}.csv"},
        headers={"Content-Type": "application/json"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


