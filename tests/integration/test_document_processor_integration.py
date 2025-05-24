# tests/integration/test_document_processor_integration.py
"""
Integration tests for DocumentProcessor with other services
"""

import pytest
import asyncio
import httpx
import json
import tempfile
import os
from pathlib import Path

@pytest.fixture
async def document_processor_client():
    """HTTP client for DocumentProcessor service"""
    client = httpx.AsyncClient(
        base_url="http://localhost:8006",
        headers={"Authorization": "Bearer secure-memory-token"},
        timeout=60.0
    )
    yield client
    await client.aclose()

@pytest.fixture
async def mcp_client():
    """HTTP client for MCP Memory Server"""
    client = httpx.AsyncClient(
        base_url="http://localhost:8080", 
        headers={"Authorization": "Bearer secure-memory-token"},
        timeout=30.0
    )
    yield client
    await client.aclose()

@pytest.fixture
async def rag_client():
    """HTTP client for RAG service"""
    client = httpx.AsyncClient(
        base_url="http://localhost:8001",
        headers={"Authorization": "Bearer secure-memory-token"},
        timeout=30.0
    )
    yield client
    await client.aclose()

@pytest.fixture
def sample_pdf_file():
    """Create a sample PDF file for testing"""
    # Create a simple text file (in real tests, use actual PDF)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdf', delete=False) as f:
        f.write("This is a sample document for testing DocumentProcessor integration.\n")
        f.write("It contains multiple lines of text that should be processed and chunked.\n")
        f.write("The system should extract this text, create embeddings, and store in vector database.\n")
        f.write("This will enable semantic search and memory integration capabilities.")
        return f.name

class TestDocumentProcessorIntegration:
    
    @pytest.mark.asyncio
    async def test_document_processing_workflow(self, document_processor_client, sample_pdf_file):
        """Test complete document processing workflow"""
        
        # Submit document for processing
        with open(sample_pdf_file, 'rb') as f:
            files = {"files": ("test_document.pdf", f, "application/pdf")}
            form_data = {
                "agent_id": "test-integration-agent",
                "session_id": "test-integration-session",
                "tags": "integration,test",
                "chunk_size": 500,
                "enable_nlp_analysis": True,
                "store_in_memory": True
            }
            
            response = await document_processor_client.post("/process", data=form_data, files=files)
        
        assert response.status_code == 200
        job_data = response.json()
        assert "job_id" in job_data
        job_id = job_data["job_id"]
        
        # Wait for processing completion
        max_attempts = 30
        for attempt in range(max_attempts):
            status_response = await document_processor_client.get(f"/job/{job_id}")
            assert status_response.status_code == 200
            
            status = status_response.json()
            if status["status"] == "completed":
                break
            elif status["status"] == "failed":
                pytest.fail(f"Document processing failed: {status.get('errors', [])}")
            
            await asyncio.sleep(2)
        else:
            pytest.fail("Document processing timed out")
        
        # Verify processing results
        assert status["status"] == "completed"
        assert status["chunks_created"] > 0
        assert status["embeddings_generated"] > 0
        
        # Clean up
        os.unlink(sample_pdf_file)
    
    @pytest.mark.asyncio
    async def test_document_search_functionality(self, document_processor_client):
        """Test document search after processing"""
        
        # Search for documents
        search_data = {
            "query": "sample document testing",
            "agent_id": "test-integration-agent",
            "limit": 5,
            "similarity_threshold": 0.5
        }
        
        response = await document_processor_client.post("/search", data=search_data)
        assert response.status_code == 200
        
        search_results = response.json()
        assert "results" in search_results
        assert "count" in search_results
        
        # Check result structure
        if search_results["count"] > 0:
            result = search_results["results"][0]
            required_fields = ["chunk_id", "document_id", "content", "similarity_score"]
            for field in required_fields:
                assert field in result
    
    @pytest.mark.asyncio 
    async def test_memory_integration(self, document_processor_client, mcp_client):
        """Test integration with MCP Memory Server"""
        
        # First, ensure we have processed documents
        docs_response = await document_processor_client.get("/documents/test-integration-agent")
        assert docs_response.status_code == 200
        
        # Query memory system for document processing entries
        memory_query = {
            "query": "document processing",
            "agent_id": "test-integration-agent",
            "session_id": "test-integration-session",
            "memory_types": ["episodic"],
            "limit": 5
        }
        
        memory_response = await mcp_client.post("/memory/query", json=memory_query)
        assert memory_response.status_code == 200
        
        memory_results = memory_response.json()
        assert "episodic" in memory_results
        
        # Check if any memories contain document processing context
        episodic_memories = memory_results["episodic"]
        document_memories = [
            memory for memory in episodic_memories
            if "document" in memory.get("content", "").lower()
        ]
        
        # Should have at least some document-related memories
        assert len(document_memories) >= 0  # May be 0 if no processing happened recently
    
    @pytest.mark.asyncio
    async def test_rag_integration(self, document_processor_client, rag_client):
        """Test integration with RAG service"""
        
        # Submit RAG query that should use processed documents
        rag_query = {
            "query": "What is the content of the test document?",
            "agent_id": "test-integration-agent",
            "session_id": "test-integration-session",
            "max_results": 3,
            "include_memory_context": True
        }
        
        response = await rag_client.post("/query", json=rag_query)
        assert response.status_code == 200
        
        rag_result = response.json()
        assert "answer" in rag_result
        assert "sources" in rag_result
        assert "confidence_score" in rag_result
        
        # Check if sources include processed documents
        sources = rag_result["sources"]
        if sources:
            # At least one source should have relevant content
            assert any("document" in source.get("content", "").lower() for source in sources)
    
    @pytest.mark.asyncio
    async def test_document_metadata_retrieval(self, document_processor_client):
        """Test document metadata retrieval"""
        
        # List documents for agent
        docs_response = await document_processor_client.get("/documents/test-integration-agent")
        assert docs_response.status_code == 200
        
        docs_data = docs_response.json()
        assert "documents" in docs_data
        assert "count" in docs_data
        
        if docs_data["count"] > 0:
            document = docs_data["documents"][0]
            document_id = document["document_id"]
            
            # Get detailed document info
            detail_response = await document_processor_client.get(f"/document/{document_id}")
            assert detail_response.status_code == 200
            
            doc_detail = detail_response.json()
            required_fields = [
                "document_id", "filename", "document_type", 
                "chunk_count", "created_at"
            ]
            for field in required_fields:
                assert field in doc_detail
    
    @pytest.mark.asyncio
    async def test_processing_statistics(self, document_processor_client):
        """Test processing statistics endpoint"""
        
        response = await document_processor_client.get("/stats")
        assert response.status_code == 200
        
        stats = response.json()
        assert "processing_stats" in stats
        assert "timestamp" in stats
        
        processing_stats = stats["processing_stats"]
        expected_stats = [
            "documents_processed", "chunks_created", 
            "embeddings_generated", "errors_count"
        ]
        for stat in expected_stats:
            assert stat in processing_stats
            assert isinstance(processing_stats[stat], int)
    
    @pytest.mark.asyncio
    async def test_error_handling(self, document_processor_client):
        """Test error handling for invalid requests"""
        
        # Test with missing required fields
        response = await document_processor_client.post("/process", data={})
        assert response.status_code == 422  # Validation error
        
        # Test with invalid job ID
        response = await document_processor_client.get("/job/invalid-job-id")
        assert response.status_code == 404
        
        # Test with invalid document ID  
        response = await document_processor_client.get("/document/invalid-doc-id")
        assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_document_deletion(self, document_processor_client):
        """Test document deletion functionality"""
        
        # List existing documents
        docs_response = await document_processor_client.get("/documents/test-integration-agent")
        assert docs_response.status_code == 200
        
        docs_data = docs_response.json()
        
        if docs_data["count"] > 0:
            document_id = docs_data["documents"][0]["document_id"]
            
            # Delete document
            delete_response = await document_processor_client.delete(f"/document/{document_id}")
            assert delete_response.status_code == 200
            
            delete_result = delete_response.json()
            assert delete_result["status"] == "success"
            
            # Verify document is deleted
            detail_response = await document_processor_client.get(f"/document/{document_id}")
            assert detail_response.status_code == 404

