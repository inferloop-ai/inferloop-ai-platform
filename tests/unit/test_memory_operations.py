# tests/unit/test_memory_operations.py
"""
Unit tests for memory operations
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import sys
import os

# Add the services directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../services/mcp-memory-server'))

from main import MemorySystem, MemoryItem

class TestMemoryOperations:
    
    @pytest.fixture
    def memory_system(self):
        """Create a memory system instance for testing"""
        system = MemorySystem()
        system.redis_client = AsyncMock()
        system.postgres_pool = AsyncMock()
        system.chroma_client = Mock()
        system.neo4j_driver = AsyncMock()
        system.embedding_model = Mock()
        system.openai_client = AsyncMock()
        return system
    
    @pytest.fixture
    def sample_memory(self):
        """Create sample memory item for testing"""
        return MemoryItem(
            content="Test memory content for unit testing",
            memory_type="working",
            agent_id="test-agent-001",
            session_id="test-session-001",
            importance=0.7,
            emotional_valence=0.2,
            tags=["test", "unit_test", "memory"],
            metadata={"test_context": "unit_testing"}
        )
    
    @pytest.mark.asyncio
    async def test_store_working_memory(self, memory_system, sample_memory):
        """Test storing memory in working memory (Redis)"""
        # Mock embedding generation
        memory_system.embedding_model.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3, 0.4]
        
        # Mock Redis operations
        memory_system.redis_client.setex = AsyncMock()
        memory_system.redis_client.keys = AsyncMock(return_value=[])
        
        # Store memory
        memory_id = await memory_system.store_working_memory(sample_memory)
        
        # Assertions
        assert memory_id == sample_memory.id
        memory_system.redis_client.setex.assert_called_once()
        memory_system.embedding_model.encode.assert_called_once_with(sample_memory.content)
    
    @pytest.mark.asyncio
    async def test_store_episodic_memory(self, memory_system, sample_memory):
        """Test storing memory in episodic memory (PostgreSQL)"""
        # Mock embedding generation
        memory_system.embedding_model.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3, 0.4]
        
        # Mock database connection
        mock_conn = AsyncMock()
        memory_system.postgres_pool.acquire.return_value.__aenter__.return_value = mock_conn
        
        # Store memory
        memory_id = await memory_system.store_episodic_memory(sample_memory)
        
        # Assertions
        assert memory_id == sample_memory.id
        mock_conn.execute.assert_called_once()
        memory_system.embedding_model.encode.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_query_episodic_memory(self, memory_system):
        """Test querying episodic memory"""
        from main import MemoryQuery
        
        # Create test query
        query = MemoryQuery(
            query="test query",
            agent_id="test-agent-001",
            session_id="test-session-001"
        )
        
        # Mock embedding generation
        memory_system.embedding_model.encode.return_value.tolist.return_value = [0.1, 0.2, 0.3, 0.4]
        
        # Mock database results
        mock_conn = AsyncMock()
        mock_conn.fetch.return_value = [
            {
                'id': 'memory-001',
                'content': 'Test memory content',
                'agent_id': 'test-agent-001',
                'session_id': 'test-session-001',
                'timestamp': datetime.utcnow(),
                'importance': 0.8,
                'emotional_valence': 0.3,
                'tags': '["test", "memory"]',
                'metadata': '{"context": "test"}',
                'embedding': [0.1, 0.2, 0.3, 0.4],
                'distance': 0.2
            }
        ]
        memory_system.postgres_pool.acquire.return_value.__aenter__.return_value = mock_conn
        
        # Query memories
        results = await memory_system.query_episodic_memory(query)
        
        # Assertions
        assert len(results) == 1
        assert results[0].content == 'Test memory content'
        assert results[0].agent_id == 'test-agent-001'
        mock_conn.fetch.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_consolidate_memories(self, memory_system):
        """Test memory consolidation process"""
        # Mock working memory retrieval
        working_memories = [
            MemoryItem(
                content="High importance memory",
                memory_type="working",
                agent_id="test-agent-001",
                session_id="test-session-001",
                importance=0.8
            ),
            MemoryItem(
                content="Medium importance memory",
                memory_type="working",
                agent_id="test-agent-001",
                session_id="test-session-001",
                importance=0.5
            ),
            MemoryItem(
                content="Low importance memory",
                memory_type="working",
                agent_id="test-agent-001",
                session_id="test-session-001",
                importance=0.2
            )
        ]
        
        memory_system.get_working_memory = AsyncMock(return_value=working_memories)
        memory_system.store_episodic_memory = AsyncMock()
        memory_system.store_semantic_memory = AsyncMock()
        memory_system.store_procedural_memory = AsyncMock()
        
        # Run consolidation
        result = await memory_system.consolidate_memories("test-agent-001", "test-session-001")
        
        # Assertions
        assert result["episodic"] >= 1  # At least high importance memory
        assert result["semantic"] >= 1
        memory_system.get_working_memory.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_memory_capacity_maintenance(self, memory_system):
        """Test working memory capacity maintenance"""
        # Mock Redis operations for capacity management
        memory_system.redis_client.keys = AsyncMock(return_value=[
            f"working:test-agent:test-session:{i}" for i in range(10)  # More than capacity
        ])
        
        # Mock memory retrieval
        old_memories = []
        for i in range(10):
            memory = MemoryItem(
                content=f"Memory {i}",
                memory_type="working",
                agent_id="test-agent",
                session_id="test-session",
                timestamp=datetime.utcnow() - timedelta(minutes=i)
            )
            old_memories.append(memory.model_dump_json())
        
        memory_system.redis_client.get = AsyncMock(side_effect=old_memories)
        memory_system.redis_client.delete = AsyncMock()
        
        # Test capacity maintenance
        await memory_system._maintain_working_memory_capacity("test-agent", "test-session")
        
        # Should delete oldest memories
        memory_system.redis_client.delete.assert_called()

class TestMemoryImportanceCalculation:
    
    def test_importance_calculation_factors(self):
        """Test importance calculation with different factors"""
        # This would test the importance calculation algorithm
        # Including content length, emotional valence, recency, access frequency
        pass
    
    def test_memory_decay_over_time(self):
        """Test memory importance decay over time"""
        # Test that memories lose importance over time if not accessed
        pass

class TestMemoryConsolidation:
    
    @pytest.mark.asyncio
    async def test_consolidation_strategy_selection(self):
        """Test different consolidation strategies based on importance"""
        pass
    
    @pytest.mark.asyncio
    async def test_consolidation_batch_processing(self):
        """Test batch processing during consolidation"""
        pass


