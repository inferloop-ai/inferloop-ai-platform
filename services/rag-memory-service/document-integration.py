# services/rag-memory-service/document_integration.py
"""
Enhanced RAG service integration with DocumentProcessor
"""

import httpx
import json
from typing import List, Dict, Any, Optional

class DocumentProcessorIntegration:
    """Integration class for RAG service to work with DocumentProcessor"""
    
    def __init__(self, document_processor_url: str, access_token: str):
        self.document_processor_url = document_processor_url
        self.access_token = access_token
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def search_processed_documents(self, query: str, agent_id: str, 
                                       limit: int = 10) -> List[Dict[str, Any]]:
        """Search processed documents for RAG context"""
        try:
            response = await self.client.post(
                f"{self.document_processor_url}/search",
                headers={"Authorization": f"Bearer {self.access_token}"},
                data={
                    "query": query,
                    "agent_id": agent_id,
                    "limit": limit,
                    "similarity_threshold": 0.7
                }
            )
            
            if response.status_code == 200:
                return response.json()["results"]
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error searching processed documents: {e}")
            return []
    
    async def get_document_context(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get full document context for RAG"""
        try:
            response = await self.client.get(
                f"{self.document_processor_url}/document/{document_id}",
                headers={"Authorization": f"Bearer {self.access_token}"}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error getting document context: {e}")
            return None
    
    async def get_related_chunks(self, document_id: str, chunk_id: str, 
                               context_window: int = 2) -> List[Dict[str, Any]]:
        """Get related chunks around a specific chunk for better context"""
        # This would require additional API endpoint in DocumentProcessor
        # For now, return empty list
        return []

# Enhanced RAG query with document integration
async def enhanced_rag_query_with_documents(self, query: RAGQuery) -> RAGResponse:
    """Enhanced RAG query that includes processed documents"""
    import time
    start_time = time.time()
    
    try:
        # Step 1: Get memory context (existing)
        memory_context = []
        if query.include_memory_context:
            memory_context = await self.get_memory_context(
                query.query, query.agent_id, query.session_id
            )
        
        # Step 2: Search processed documents
        document_processor_integration = DocumentProcessorIntegration(
            document_processor_url="http://document-processor:8000",
            access_token=os.getenv("ACCESS_TOKEN", "secure-memory-token")
        )
        
        processed_doc_results = await document_processor_integration.search_processed_documents(
            query.query, query.agent_id, query.max_results
        )
        
        # Step 3: Expand query with memory context (existing)
        expanded_query = query.query
        if query.expand_query:
            expanded_query = await self.expand_query_with_memory(query.query, memory_context)
        
        # Step 4: Retrieve from ChromaDB (existing) + processed documents
        chroma_sources = await self.retrieve_documents(
            expanded_query, query.agent_id, query.max_results
        )
        
        # Combine and rank sources
        all_sources = []
        
        # Add ChromaDB results
        for source in chroma_sources:
            source["source_type"] = "chroma_collection"
            all_sources.append(source)
        
        # Add processed document results
        for doc_result in processed_doc_results:
            source = {
                "content": doc_result["content"],
                "document_name": doc_result["document_filename"],
                "chunk_index": doc_result["chunk_index"],
                "similarity_score": doc_result["similarity_score"],
                "source_type": "processed_document",
                "document_id": doc_result["document_id"],
                "keywords": doc_result["keywords"],
                "entities": doc_result["entities"],
                "importance_score": doc_result["importance_score"],
                "metadata": {
                    "document_type": doc_result["document_type"],
                    "chunk_id": doc_result["chunk_id"]
                }
            }
            all_sources.append(source)
        
        # Sort by similarity score and importance
        all_sources.sort(key=lambda x: (x["similarity_score"], x.get("importance_score", 0.5)), reverse=True)
        
        # Take top results
        top_sources = all_sources[:query.max_results]
        
        # Step 5: Generate response with enhanced context
        answer = await self.generate_response_with_documents(
            query.query, top_sources, memory_context, query.model, query.temperature
        )
        
        # Step 6: Calculate confidence score
        confidence_score = self.calculate_enhanced_confidence_score(top_sources, memory_context)
        
        # Step 7: Store interaction in memory with document context
        await self.store_enhanced_interaction_memory(
            query.query, answer, top_sources, query.agent_id, query.session_id
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return RAGResponse(
            answer=answer,
            sources=top_sources,
            memory_context=memory_context,
            query_expansion=expanded_query if expanded_query != query.query else None,
            confidence_score=confidence_score,
            processing_time_ms=round(processing_time, 2)
        )
        
    except Exception as e:
        logger.error(f"Error processing enhanced RAG query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

