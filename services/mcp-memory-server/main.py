# services/mcp-memory-server/main.py
"""
Memory-Enhanced MCP Server
Advanced Multi-Modal Memory System with Agentic Capabilities
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import aioredis
import asyncpg
import chromadb
import neo4j
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import openai
from anthropic import Anthropic

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================================
# Configuration & Models
# ==============================================

class MemoryConfig:
    def __init__(self):
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.postgres_url = os.getenv("POSTGRES_URL", "postgresql://memory_user:memorypass123@localhost:5432/memories")
        self.chroma_host = os.getenv("CHROMA_HOST", "localhost")
        self.chroma_port = int(os.getenv("CHROMA_PORT", "8000"))
        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", "memorypass123")
        
        # Memory system settings
        self.working_memory_capacity = int(os.getenv("WORKING_MEMORY_CAPACITY", "7"))
        self.working_memory_ttl = int(os.getenv("WORKING_MEMORY_TTL", "1800"))
        self.consolidation_interval = int(os.getenv("MEMORY_CONSOLIDATION_INTERVAL", "21600"))
        self.cleanup_interval = int(os.getenv("MEMORY_CLEANUP_INTERVAL", "3600"))
        self.similarity_threshold = float(os.getenv("SEMANTIC_MEMORY_SIMILARITY_THRESHOLD", "0.85"))
        self.decay_rate = float(os.getenv("EPISODIC_MEMORY_DECAY_RATE", "0.99"))

config = MemoryConfig()

# Pydantic Models
class MemoryItem(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    memory_type: str  # working, episodic, semantic, procedural
    agent_id: str
    session_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    importance: float = Field(default=0.5, ge=0, le=1)
    emotional_valence: float = Field(default=0.0, ge=-1, le=1)
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None

class MemoryQuery(BaseModel):
    query: str
    agent_id: str
    session_id: str
    memory_types: List[str] = Field(default=["working", "episodic", "semantic"])
    limit: int = Field(default=10, le=100)
    similarity_threshold: float = Field(default=0.7, ge=0, le=1)
    time_range: Optional[Dict[str, datetime]] = None
    include_context: bool = True

class ConsolidationRequest(BaseModel):
    agent_id: str
    session_id: str
    force: bool = False

# ==============================================
# Memory System Components
# ==============================================

class MemorySystem:
    def __init__(self):
        self.redis_client = None
        self.postgres_pool = None
        self.chroma_client = None
        self.neo4j_driver = None
        self.embedding_model = None
        self.openai_client = None
        self.anthropic_client = None
        
    async def initialize(self):
        """Initialize all memory system connections"""
        # Redis connection
        self.redis_client = aioredis.from_url(config.redis_url, decode_responses=True)
        
        # PostgreSQL connection pool
        self.postgres_pool = await asyncpg.create_pool(config.postgres_url)
        
        # ChromaDB client
        self.chroma_client = chromadb.HttpClient(
            host=config.chroma_host,
            port=config.chroma_port
        )
        
        # Neo4j driver
        self.neo4j_driver = neo4j.AsyncGraphDatabase.driver(
            config.neo4j_uri,
            auth=(config.neo4j_username, config.neo4j_password)
        )
        
        # Embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # AI clients
        self.openai_client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
        logger.info("Memory system initialized successfully")

    async def cleanup(self):
        """Cleanup all connections"""
        if self.redis_client:
            await self.redis_client.close()
        if self.postgres_pool:
            await self.postgres_pool.close()
        if self.neo4j_driver:
            await self.neo4j_driver.close()

    # ==============================================
    # Working Memory (Redis)
    # ==============================================
    
    async def store_working_memory(self, memory: MemoryItem) -> str:
        """Store memory item in working memory (Redis)"""
        key = f"working:{memory.agent_id}:{memory.session_id}:{memory.id}"
        
        # Generate embedding
        if not memory.embedding:
            memory.embedding = self.embedding_model.encode(memory.content).tolist()
        
        # Store in Redis with TTL
        await self.redis_client.setex(
            key,
            config.working_memory_ttl,
            memory.model_dump_json()
        )
        
        # Maintain working memory capacity
        await self._maintain_working_memory_capacity(memory.agent_id, memory.session_id)
        
        logger.info(f"Stored working memory: {memory.id}")
        return memory.id

    async def _maintain_working_memory_capacity(self, agent_id: str, session_id: str):
        """Maintain working memory capacity by removing oldest items"""
        pattern = f"working:{agent_id}:{session_id}:*"
        keys = await self.redis_client.keys(pattern)
        
        if len(keys) > config.working_memory_capacity:
            # Get timestamps and sort
            memory_items = []
            for key in keys:
                data = await self.redis_client.get(key)
                if data:
                    memory = MemoryItem.model_validate_json(data)
                    memory_items.append((key, memory.timestamp))
            
            # Sort by timestamp and remove oldest
            memory_items.sort(key=lambda x: x[1])
            to_remove = memory_items[:len(memory_items) - config.working_memory_capacity]
            
            for key, _ in to_remove:
                await self.redis_client.delete(key)

    async def get_working_memory(self, agent_id: str, session_id: str) -> List[MemoryItem]:
        """Retrieve all working memory items for an agent/session"""
        pattern = f"working:{agent_id}:{session_id}:*"
        keys = await self.redis_client.keys(pattern)
        
        memories = []
        for key in keys:
            data = await self.redis_client.get(key)
            if data:
                memories.append(MemoryItem.model_validate_json(data))
        
        return sorted(memories, key=lambda x: x.timestamp, reverse=True)

    # ==============================================
    # Episodic Memory (PostgreSQL)
    # ==============================================
    
    async def store_episodic_memory(self, memory: MemoryItem) -> str:
        """Store memory item in episodic memory (PostgreSQL)"""
        if not memory.embedding:
            memory.embedding = self.embedding_model.encode(memory.content).tolist()
        
        async with self.postgres_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO episodic_memories 
                (id, content, agent_id, session_id, timestamp, importance, emotional_valence, tags, metadata, embedding)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """, 
                memory.id, memory.content, memory.agent_id, memory.session_id,
                memory.timestamp, memory.importance, memory.emotional_valence,
                json.dumps(memory.tags), json.dumps(memory.metadata), memory.embedding
            )
        
        logger.info(f"Stored episodic memory: {memory.id}")
        return memory.id

    async def query_episodic_memory(self, query: MemoryQuery) -> List[MemoryItem]:
        """Query episodic memory using vector similarity"""
        query_embedding = self.embedding_model.encode(query.query).tolist()
        
        async with self.postgres_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id, content, agent_id, session_id, timestamp, importance, 
                       emotional_valence, tags, metadata, embedding,
                       (embedding <=> $1::vector) as distance
                FROM episodic_memories 
                WHERE agent_id = $2 
                  AND (embedding <=> $1::vector) < $3
                ORDER BY distance
                LIMIT $4
            """, query_embedding, query.agent_id, 1 - query.similarity_threshold, query.limit)
        
        memories = []
        for row in rows:
            memory_data = dict(row)
            memory_data['tags'] = json.loads(memory_data['tags'])
            memory_data['metadata'] = json.loads(memory_data['metadata'])
            memories.append(MemoryItem(**memory_data))
        
        return memories

    # ==============================================
    # Semantic Memory (ChromaDB)
    # ==============================================
    
    async def store_semantic_memory(self, memory: MemoryItem) -> str:
        """Store memory item in semantic memory (ChromaDB)"""
        collection_name = f"semantic_{memory.agent_id.replace('-', '_')}"
        
        try:
            collection = self.chroma_client.get_or_create_collection(collection_name)
            
            collection.add(
                documents=[memory.content],
                metadatas=[{
                    "agent_id": memory.agent_id,
                    "session_id": memory.session_id,
                    "timestamp": memory.timestamp.isoformat(),
                    "importance": memory.importance,
                    "tags": json.dumps(memory.tags)
                }],
                ids=[memory.id]
            )
            
            logger.info(f"Stored semantic memory: {memory.id}")
            return memory.id
            
        except Exception as e:
            logger.error(f"Error storing semantic memory: {e}")
            raise

    async def query_semantic_memory(self, query: MemoryQuery) -> List[MemoryItem]:
        """Query semantic memory using ChromaDB"""
        collection_name = f"semantic_{query.agent_id.replace('-', '_')}"
        
        try:
            collection = self.chroma_client.get_collection(collection_name)
            
            results = collection.query(
                query_texts=[query.query],
                n_results=query.limit,
                where={"agent_id": query.agent_id}
            )
            
            memories = []
            if results['documents']:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i]
                    memory = MemoryItem(
                        id=results['ids'][0][i],
                        content=doc,
                        memory_type="semantic",
                        agent_id=metadata['agent_id'],
                        session_id=metadata['session_id'],
                        timestamp=datetime.fromisoformat(metadata['timestamp']),
                        importance=metadata['importance'],
                        tags=json.loads(metadata.get('tags', '[]'))
                    )
                    memories.append(memory)
            
            return memories
            
        except Exception as e:
            logger.error(f"Error querying semantic memory: {e}")
            return []

    # ==============================================
    # Procedural Memory (Neo4j)
    # ==============================================
    
    async def store_procedural_memory(self, memory: MemoryItem) -> str:
        """Store procedural memory as knowledge graph in Neo4j"""
        async with self.neo4j_driver.session() as session:
            # Extract entities and relationships using AI
            entities_and_relations = await self._extract_knowledge_graph(memory.content)
            
            # Store in Neo4j
            query = """
            MERGE (m:Memory {id: $memory_id})
            SET m.content = $content,
                m.agent_id = $agent_id,
                m.session_id = $session_id,
                m.timestamp = $timestamp,
                m.importance = $importance
            """
            
            await session.run(query, {
                'memory_id': memory.id,
                'content': memory.content,
                'agent_id': memory.agent_id,
                'session_id': memory.session_id,
                'timestamp': memory.timestamp.isoformat(),
                'importance': memory.importance
            })
            
            # Add extracted entities and relationships
            for entity in entities_and_relations.get('entities', []):
                entity_query = """
                MERGE (e:Entity {name: $name, type: $type})
                MERGE (m:Memory {id: $memory_id})
                MERGE (m)-[:CONTAINS]->(e)
                """
                await session.run(entity_query, {
                    'name': entity['name'],
                    'type': entity['type'],
                    'memory_id': memory.id
                })
        
        logger.info(f"Stored procedural memory: {memory.id}")
        return memory.id

    async def _extract_knowledge_graph(self, content: str) -> Dict[str, Any]:
        """Extract entities and relationships using AI"""
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "Extract entities and relationships from the text. Return JSON with 'entities' and 'relationships' arrays."
                    },
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                temperature=0.1
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            logger.error(f"Error extracting knowledge graph: {e}")
            return {"entities": [], "relationships": []}

    # ==============================================
    # Memory Consolidation
    # ==============================================
    
    async def consolidate_memories(self, agent_id: str, session_id: str) -> Dict[str, int]:
        """Consolidate working memory into long-term storage"""
        working_memories = await self.get_working_memory(agent_id, session_id)
        
        consolidated = {"episodic": 0, "semantic": 0, "procedural": 0}
        
        for memory in working_memories:
            # Determine consolidation strategy based on importance and content
            if memory.importance > 0.7:
                # High importance -> store in all memory types
                await self.store_episodic_memory(memory)
                await self.store_semantic_memory(memory)
                await self.store_procedural_memory(memory)
                consolidated["episodic"] += 1
                consolidated["semantic"] += 1
                consolidated["procedural"] += 1
                
            elif memory.importance > 0.4:
                # Medium importance -> episodic and semantic
                await self.store_episodic_memory(memory)
                await self.store_semantic_memory(memory)
                consolidated["episodic"] += 1
                consolidated["semantic"] += 1
                
            else:
                # Low importance -> semantic only
                await self.store_semantic_memory(memory)
                consolidated["semantic"] += 1
        
        logger.info(f"Consolidated {len(working_memories)} memories: {consolidated}")
        return consolidated

    async def query_all_memories(self, query: MemoryQuery) -> Dict[str, List[MemoryItem]]:
        """Query all memory types and return combined results"""
        results = {"working": [], "episodic": [], "semantic": [], "procedural": []}
        
        # Query working memory
        if "working" in query.memory_types:
            results["working"] = await self.get_working_memory(query.agent_id, query.session_id)
        
        # Query episodic memory
        if "episodic" in query.memory_types:
            results["episodic"] = await self.query_episodic_memory(query)
        
        # Query semantic memory
        if "semantic" in query.memory_types:
            results["semantic"] = await self.query_semantic_memory(query)
        
        return results

# ==============================================
# FastAPI Application
# ==============================================

app = FastAPI(
    title="Memory-Enhanced MCP Server",
    description="Advanced Multi-Modal Memory System with Agentic Capabilities",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()
access_token = os.getenv("ACCESS_TOKEN", "secure-memory-token")

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != access_token:
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials

# Global memory system instance
memory_system = MemorySystem()

@app.on_event("startup")
async def startup_event():
    await memory_system.initialize()
    
    # Start background tasks
    asyncio.create_task(periodic_consolidation())
    asyncio.create_task(periodic_cleanup())

@app.on_event("shutdown")
async def shutdown_event():
    await memory_system.cleanup()

# ==============================================
# API Endpoints
# ==============================================

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}

@app.post("/memory/store")
async def store_memory(
    memory: MemoryItem,
    credentials: HTTPAuthorizationCredentials = Depends(verify_token)
):
    """Store a memory item"""
    if memory.memory_type == "working":
        memory_id = await memory_system.store_working_memory(memory)
    elif memory.memory_type == "episodic":
        memory_id = await memory_system.store_episodic_memory(memory)
    elif memory.memory_type == "semantic":
        memory_id = await memory_system.store_semantic_memory(memory)
    elif memory.memory_type == "procedural":
        memory_id = await memory_system.store_procedural_memory(memory)
    else:
        raise HTTPException(status_code=400, detail="Invalid memory type")
    
    return {"memory_id": memory_id, "status": "stored"}

@app.post("/memory/query")
async def query_memory(
    query: MemoryQuery,
    credentials: HTTPAuthorizationCredentials = Depends(verify_token)
):
    """Query memories across all types"""
    results = await memory_system.query_all_memories(query)
    return results

@app.post("/memory/consolidate")
async def consolidate_memory(
    request: ConsolidationRequest,
    background_tasks: BackgroundTasks,
    credentials: HTTPAuthorizationCredentials = Depends(verify_token)
):
    """Trigger memory consolidation"""
    if request.force:
        result = await memory_system.consolidate_memories(request.agent_id, request.session_id)
        return {"status": "completed", "consolidated": result}
    else:
        background_tasks.add_task(
            memory_system.consolidate_memories,
            request.agent_id,
            request.session_id
        )
        return {"status": "scheduled"}

@app.get("/memory/working/{agent_id}/{session_id}")
async def get_working_memory(
    agent_id: str,
    session_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(verify_token)
):
    """Get all working memory for an agent/session"""
    memories = await memory_system.get_working_memory(agent_id, session_id)
    return {"memories": memories, "count": len(memories)}

@app.get("/memory/stats/{agent_id}")
async def get_memory_stats(
    agent_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(verify_token)
):
    """Get memory statistics for an agent"""
    # Implementation for memory analytics
    stats = {
        "agent_id": agent_id,
        "working_memory_count": 0,
        "episodic_memory_count": 0,
        "semantic_memory_count": 0,
        "procedural_memory_count": 0,
        "last_consolidation": None
    }
    
    # Get working memory count
    pattern = f"working:{agent_id}:*"
    working_keys = await memory_system.redis_client.keys(pattern)
    stats["working_memory_count"] = len(working_keys)
    
    # Get episodic memory count
    async with memory_system.postgres_pool.acquire() as conn:
        result = await conn.fetchval(
            "SELECT COUNT(*) FROM episodic_memories WHERE agent_id = $1",
            agent_id
        )
        stats["episodic_memory_count"] = result
    
    return stats

# ==============================================
# Background Tasks
# ==============================================

async def periodic_consolidation():
    """Periodic memory consolidation task"""
    while True:
        try:
            # Get all active agent/session pairs
            pattern = "working:*"
            keys = await memory_system.redis_client.keys(pattern)
            
            agent_sessions = set()
            for key in keys:
                parts = key.split(":")
                if len(parts) >= 3:
                    agent_id, session_id = parts[1], parts[2]
                    agent_sessions.add((agent_id, session_id))
            
            # Consolidate memories for each agent/session
            for agent_id, session_id in agent_sessions:
                await memory_system.consolidate_memories(agent_id, session_id)
                await asyncio.sleep(1)  # Prevent overwhelming the system
            
            logger.info(f"Completed consolidation for {len(agent_sessions)} agent/session pairs")
            
        except Exception as e:
            logger.error(f"Error in periodic consolidation: {e}")
        
        await asyncio.sleep(config.consolidation_interval)

async def periodic_cleanup():
    """Periodic memory cleanup task"""
    while True:
        try:
            # Clean up expired working memory
            pattern = "working:*"
            keys = await memory_system.redis_client.keys(pattern)
            
            for key in keys:
                ttl = await memory_system.redis_client.ttl(key)
                if ttl == -1:  # No expiration set
                    await memory_system.redis_client.expire(key, config.working_memory_ttl)
            
            # Clean up old episodic memories with low importance
            async with memory_system.postgres_pool.acquire() as conn:
                cutoff_date = datetime.utcnow() - timedelta(days=30)
                await conn.execute("""
                    DELETE FROM episodic_memories 
                    WHERE timestamp < $1 AND importance < 0.3
                """, cutoff_date)
            
            logger.info("Completed memory cleanup")
            
        except Exception as e:
            logger.error(f"Error in periodic cleanup: {e}")
        
        await asyncio.sleep(config.cleanup_interval)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)