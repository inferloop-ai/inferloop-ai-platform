# tests/integration/test_api_endpoints.py
"""
Integration tests for API endpoints
"""

import pytest
import httpx
import json
import asyncio
from datetime import datetime

class TestMemoryAPI:
    
    @pytest.fixture
    def api_client(self):
        """Create HTTP client for API testing"""
        return httpx.AsyncClient(
            base_url="http://localhost:8080",
            headers={"Authorization": "Bearer secure-memory-token"}
        )
    
    @pytest.mark.asyncio
    async def test_health_endpoint(self, api_client):
        """Test health check endpoint"""
        response = await api_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
    
    @pytest.mark.asyncio
    async def test_store_memory_endpoint(self, api_client):
        """Test memory storage endpoint"""
        memory_data = {
            "content": "Integration test memory",
            "memory_type": "working",
            "agent_id": "integration-test-agent",
            "session_id": "integration-test-session",
            "importance": 0.6,
            "tags": ["integration", "test"]
        }
        
        response = await api_client.post("/memory/store", json=memory_data)
        assert response.status_code == 200
        data = response.json()
        assert "memory_id" in data
        assert data["status"] == "stored"
    
    @pytest.mark.asyncio
    async def test_query_memory_endpoint(self, api_client):
        """Test memory query endpoint"""
        # First store a memory
        memory_data = {
            "content": "Queryable test memory",
            "memory_type": "working",
            "agent_id": "integration-test-agent",
            "session_id": "integration-test-session",
            "importance": 0.7
        }
        
        store_response = await api_client.post("/memory/store", json=memory_data)
        assert store_response.status_code == 200
        
        # Then query for it
        query_data = {
            "query": "queryable test",
            "agent_id": "integration-test-agent",
            "session_id": "integration-test-session",
            "memory_types": ["working"],
            "limit": 5
        }
        
        query_response = await api_client.post("/memory/query", json=query_data)
        assert query_response.status_code == 200
        results = query_response.json()
        assert "working" in results
    
    @pytest.mark.asyncio
    async def test_consolidation_endpoint(self, api_client):
        """Test memory consolidation endpoint"""
        consolidation_data = {
            "agent_id": "integration-test-agent",
            "session_id": "integration-test-session",
            "force": True
        }
        
        response = await api_client.post("/memory/consolidate", json=consolidation_data)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["completed", "scheduled"]
    
    @pytest.mark.asyncio
    async def test_memory_stats_endpoint(self, api_client):
        """Test memory statistics endpoint"""
        response = await api_client.get("/memory/stats/integration-test-agent")
        assert response.status_code == 200
        data = response.json()
        assert "agent_id" in data
        assert "working_memory_count" in data

class TestRAGAPI:
    
    @pytest.fixture
    def rag_client(self):
        """Create HTTP client for RAG service testing"""
        return httpx.AsyncClient(
            base_url="http://localhost:8001",
            headers={"Authorization": "Bearer secure-memory-token"}
        )
    
    @pytest.mark.asyncio
    async def test_rag_query_endpoint(self, rag_client):
        """Test RAG query endpoint"""
        query_data = {
            "query": "What is machine learning?",
            "agent_id": "rag-test-agent",
            "session_id": "rag-test-session",
            "max_results": 3,
            "include_memory_context": True
        }
        
        response = await rag_client.post("/query", json=query_data)
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "confidence_score" in data

class TestAgentOrchestratorAPI:
    
    @pytest.fixture
    def orchestrator_client(self):
        """Create HTTP client for agent orchestrator testing"""
        return httpx.AsyncClient(
            base_url="http://localhost:8003",
            headers={"Authorization": "Bearer secure-memory-token"}
        )
    
    @pytest.mark.asyncio
    async def test_agent_registration(self, orchestrator_client):
        """Test agent registration endpoint"""
        agent_data = {
            "agent_id": "test-integration-agent",
            "name": "Integration Test Agent",
            "type": "general",
            "capabilities": ["testing", "integration"],
            "max_concurrent_tasks": 2
        }
        
        response = await orchestrator_client.post("/agents/register", json=agent_data)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "agent_id" in data
