# services/rag-memory-service/main.py
"""
RAG Memory Service
Retrieval-Augmented Generation with Memory Integration
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import chromadb
import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import openai
from anthropic import Anthropic
import PyPDF2
import docx
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================================
# Configuration & Models
# ==============================================

class RAGConfig:
    def __init__(self):
        self.memory_system_url = os.getenv("MEMORY_SYSTEM_URL", "http://mcp-memory-server:8080")
        self.chroma_url = os.getenv("CHROMA_URL", "http://chroma-memory:8000")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
        self.enable_memory_guided_retrieval = os.getenv("ENABLE_MEMORY_GUIDED_RETRIEVAL", "true").lower() == "true"
        self.enable_query_expansion = os.getenv("ENABLE_QUERY_EXPANSION", "true").lower() == "true"
        self.enable_personalization = os.getenv("ENABLE_PERSONALIZATION", "true").lower() == "true"
        self.cache_ttl = int(os.getenv("CACHE_TTL", "3600"))
        self.max_context_length = int(os.getenv("MAX_CONTEXT_LENGTH", "4000"))
        self.similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))

config = RAGConfig()

# Pydantic Models
class DocumentChunk(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    document_id: str
    document_name: str
    chunk_index: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None

class RAGQuery(BaseModel):
    query: str
    agent_id: str
    session_id: str
    max_results: int = Field(default=5, le=20)
    include_memory_context: bool = True
    expand_query: bool = True
    personalize: bool = True
    temperature: float = Field(default=0.7, ge=0, le=2)
    model: str = "gpt-3.5-turbo"

class RAGResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    memory_context: List[Dict[str, Any]] = []
    query_expansion: Optional[str] = None
    confidence_score: float
    processing_time_ms: float

class DocumentUpload(BaseModel):
    filename: str
    content_type: str
    agent_id: str
    session_id: str
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

# ==============================================
# RAG Memory Service
# ==============================================

class RAGMemoryService:
    def __init__(self):
        self.chroma_client = None
        self.embedding_model = None
        self.openai_client = None
        self.anthropic_client = None
        self.memory_client = None
        self.document_collection = None
        
    async def initialize(self):
        """Initialize all connections and models"""
        # ChromaDB client
        self.chroma_client = chromadb.HttpClient(
            host=config.chroma_url.split("://")[1].split(":")[0],
            port=int(config.chroma_url.split(":")[-1])
        )
        
        # Get or create document collection
        self.document_collection = self.chroma_client.get_or_create_collection(
            name="rag_documents",
            metadata={"description": "RAG document chunks with embeddings"}
        )
        
        # Embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # AI clients
        if config.openai_api_key:
            self.openai_client = openai.AsyncOpenAI(api_key=config.openai_api_key)
        
        if config.anthropic_api_key:
            self.anthropic_client = Anthropic(api_key=config.anthropic_api_key)
        
        # HTTP client for memory system
        self.memory_client = httpx.AsyncClient(timeout=30.0)
        
        logger.info("RAG Memory Service initialized successfully")

    async def cleanup(self):
        """Cleanup connections"""
        if self.memory_client:
            await self.memory_client.aclose()

    # ==============================================
    # Document Processing
    # ==============================================
    
    def extract_text_from_file(self, file_content: bytes, content_type: str) -> str:
        """Extract text from uploaded file"""
        if content_type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        
        elif content_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
            doc = docx.Document(BytesIO(file_content))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        
        elif content_type.startswith("text/"):
            return file_content.decode('utf-8')
        
        else:
            raise ValueError(f"Unsupported content type: {content_type}")

    def chunk_document(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split document into overlapping chunks"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                for i in range(end, max(start + chunk_size - 200, start), -1):
                    if text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
        
        return chunks

    async def process_document(self, document: DocumentUpload, file_content: bytes) -> List[DocumentChunk]:
        """Process document and create chunks with embeddings"""
        # Extract text
        text = self.extract_text_from_file(file_content, document.content_type)
        
        # Create chunks
        chunks = self.chunk_document(text)
        
        # Create document chunks
        document_chunks = []
        for i, chunk_text in enumerate(chunks):
            # Generate embedding
            embedding = self.embedding_model.encode(chunk_text).tolist()
            
            chunk = DocumentChunk(
                content=chunk_text,
                document_id=document.filename,
                document_name=document.filename,
                chunk_index=i,
                embedding=embedding,
                metadata={
                    **document.metadata,
                    "agent_id": document.agent_id,
                    "session_id": document.session_id,
                    "tags": document.tags,
                    "chunk_count": len(chunks),
                    "processed_at": datetime.utcnow().isoformat()
                }
            )
            document_chunks.append(chunk)
        
        return document_chunks

    async def store_document_chunks(self, chunks: List[DocumentChunk]) -> str:
        """Store document chunks in ChromaDB"""
        try:
            # Prepare data for ChromaDB
            documents = [chunk.content for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            ids = [chunk.id for chunk in chunks]
            embeddings = [chunk.embedding for chunk in chunks]
            
            # Add to collection
            self.document_collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )
            
            logger.info(f"Stored {len(chunks)} document chunks")
            return f"Successfully stored {len(chunks)} chunks"
            
        except Exception as e:
            logger.error(f"Error storing document chunks: {e}")
            raise

    # ==============================================
    # Memory Integration
    # ==============================================
    
    async def get_memory_context(self, query: str, agent_id: str, session_id: str) -> List[Dict[str, Any]]:
        """Get relevant memory context from MCP server"""
        if not config.enable_memory_guided_retrieval:
            return []
        
        try:
            memory_query = {
                "query": query,
                "agent_id": agent_id,
                "session_id": session_id,
                "memory_types": ["working", "episodic", "semantic"],
                "limit": 5,
                "similarity_threshold": 0.7
            }
            
            response = await self.memory_client.post(
                f"{config.memory_system_url}/memory/query",
                json=memory_query,
                headers={"Authorization": f"Bearer {os.getenv('MCP_ACCESS_TOKEN', 'secure-memory-token')}"}
            )
            
            if response.status_code == 200:
                memory_results = response.json()
                context = []
                
                for memory_type, memories in memory_results.items():
                    for memory in memories:
                        context.append({
                            "type": memory_type,
                            "content": memory.get("content", ""),
                            "importance": memory.get("importance", 0.5),
                            "timestamp": memory.get("timestamp")
                        })
                
                return context
            
        except Exception as e:
            logger.error(f"Error getting memory context: {e}")
        
        return []

    async def expand_query_with_memory(self, original_query: str, memory_context: List[Dict[str, Any]]) -> str:
        """Expand query using memory context"""
        if not config.enable_query_expansion or not memory_context:
            return original_query
        
        try:
            # Create context from memories
            context_text = "\n".join([
                f"- {mem['content'][:100]}..." if len(mem['content']) > 100 else f"- {mem['content']}"
                for mem in memory_context[:3]
            ])
            
            expansion_prompt = f"""
            Based on the user's previous interactions and context, expand this query to be more specific and relevant:
            
            Original Query: {original_query}
            
            Context from previous interactions:
            {context_text}
            
            Provide an expanded, more specific query that incorporates relevant context:
            """
            
            if self.openai_client:
                response = await self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": expansion_prompt}],
                    temperature=0.3,
                    max_tokens=150
                )
                
                expanded_query = response.choices[0].message.content.strip()
                return expanded_query
            
        except Exception as e:
            logger.error(f"Error expanding query: {e}")
        
        return original_query

    # ==============================================
    # Retrieval and Generation
    # ==============================================
    
    async def retrieve_documents(self, query: str, agent_id: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant documents from ChromaDB"""
        try:
            # Query the collection
            results = self.document_collection.query(
                query_texts=[query],
                n_results=max_results,
                where={"agent_id": agent_id} if config.enable_personalization else None
            )
            
            sources = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    
                    sources.append({
                        "content": doc,
                        "document_name": metadata.get("document_name", "Unknown"),
                        "chunk_index": metadata.get("chunk_index", 0),
                        "similarity_score": 1 - results['distances'][0][i] if results['distances'] else 0.0,
                        "metadata": metadata
                    })
            
            return sources
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []

    async def generate_response(self, query: str, sources: List[Dict[str, Any]], 
                             memory_context: List[Dict[str, Any]], model: str, temperature: float) -> str:
        """Generate response using retrieved sources and memory context"""
        # Prepare context
        document_context = "\n\n".join([
            f"Source: {source['document_name']}\nContent: {source['content']}"
            for source in sources[:3]
        ])
        
        memory_context_text = "\n".join([
            f"Memory: {mem['content']}"
            for mem in memory_context[:2]
        ]) if memory_context else "No relevant memory context."
        
        # Create prompt
        prompt = f"""
        You are an AI assistant with access to documents and memory context. Answer the user's question based on the provided information.
        
        User Question: {query}
        
        Document Context:
        {document_context}
        
        Memory Context (previous interactions):
        {memory_context_text}
        
        Please provide a comprehensive answer that:
        1. Directly addresses the user's question
        2. Uses information from the provided documents
        3. Considers the memory context when relevant
        4. Cites specific sources when possible
        5. Is clear and well-structured
        
        Answer:
        """
        
        try:
            if model.startswith("gpt") and self.openai_client:
                response = await self.openai_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=config.max_context_length
                )
                return response.choices[0].message.content
            
            elif model.startswith("claude") and self.anthropic_client:
                message = self.anthropic_client.messages.create(
                    model=model,
                    max_tokens=config.max_context_length,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                return message.content[0].text
                
            else:
                # Fallback: simple concatenation
                return f"Based on the available documents: {document_context[:500]}..."
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I encountered an error generating a response. Please try again."

    async def store_interaction_memory(self, query: str, response: str, sources: List[Dict[str, Any]], 
                                     agent_id: str, session_id: str):
        """Store the RAG interaction in memory system"""
        try:
            # Create memory entry for the interaction
            memory_content = f"User asked: '{query}'. I provided information from {len(sources)} sources including {', '.join([s['document_name'] for s in sources[:2]])}."
            
            memory_data = {
                "content": memory_content,
                "memory_type": "episodic",
                "agent_id": agent_id,
                "session_id": session_id,
                "importance": 0.6,  # Medium importance for RAG interactions
                "tags": ["rag_interaction", "knowledge_retrieval"],
                "metadata": {
                    "query": query,
                    "response_length": len(response),
                    "sources_used": len(sources),
                    "interaction_type": "rag_query"
                }
            }
            
            await self.memory_client.post(
                f"{config.memory_system_url}/memory/store",
                json=memory_data,
                headers={"Authorization": f"Bearer {os.getenv('MCP_ACCESS_TOKEN', 'secure-memory-token')}"}
            )
            
        except Exception as e:
            logger.error(f"Error storing interaction memory: {e}")

    # ==============================================
    # Main RAG Pipeline
    # ==============================================
    
    async def process_rag_query(self, query: RAGQuery) -> RAGResponse:
        """Main RAG processing pipeline"""
        import time
        start_time = time.time()
        
        try:
            # Step 1: Get memory context
            memory_context = []
            if query.include_memory_context:
                memory_context = await self.get_memory_context(
                    query.query, query.agent_id, query.session_id
                )
            
            # Step 2: Expand query with memory context
            expanded_query = query.query
            if query.expand_query:
                expanded_query = await self.expand_query_with_memory(query.query, memory_context)
            
            # Step 3: Retrieve relevant documents
            sources = await self.retrieve_documents(
                expanded_query, query.agent_id, query.max_results
            )
            
            # Step 4: Generate response
            answer = await self.generate_response(
                query.query, sources, memory_context, query.model, query.temperature
            )
            
            # Step 5: Calculate confidence score
            confidence_score = self.calculate_confidence_score(sources, memory_context)
            
            # Step 6: Store interaction in memory
            await self.store_interaction_memory(
                query.query, answer, sources, query.agent_id, query.session_id
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            return RAGResponse(
                answer=answer,
                sources=sources,
                memory_context=memory_context,
                query_expansion=expanded_query if expanded_query != query.query else None,
                confidence_score=confidence_score,
                processing_time_ms=round(processing_time, 2)
            )
            
        except Exception as e:
            logger.error(f"Error processing RAG query: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    def calculate_confidence_score(self, sources: List[Dict[str, Any]], 
                                 memory_context: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on source quality and relevance"""
        if not sources:
            return 0.0
        
        # Average similarity score from sources
        avg_similarity = sum([s.get('similarity_score', 0) for s in sources]) / len(sources)
        
        # Boost confidence if we have memory context
        memory_boost = 0.1 if memory_context else 0.0
        
        # Source count factor (more sources = higher confidence, up to a point)
        source_factor = min(len(sources) / 5.0, 1.0)
        
        confidence = (avg_similarity * 0.7) + (source_factor * 0.2) + memory_boost
        return min(confidence, 1.0)

# ==============================================
# FastAPI Application
# ==============================================

app = FastAPI(
    title="RAG Memory Service",
    description="Retrieval-Augmented Generation with Memory Integration",
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

# Global service instance
rag_service = RAGMemoryService()

@app.on_event("startup")
async def startup_event():
    await rag_service.initialize()

@app.on_event("shutdown")
async def shutdown_event():
    await rag_service.cleanup()

# ==============================================
# API Endpoints
# ==============================================

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow(), "service": "rag-memory"}

@app.post("/query", response_model=RAGResponse)
async def process_query(
    query: RAGQuery,
    credentials: HTTPAuthorizationCredentials = Depends(verify_token)
):
    """Process RAG query with memory integration"""
    return await rag_service.process_rag_query(query)

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    agent_id: str = "",
    session_id: str = "",
    tags: str = "",
    credentials: HTTPAuthorizationCredentials = Depends(verify_token)
):
    """Upload and process document for RAG"""
    try:
        # Read file content
        file_content = await file.read()
        
        # Create document upload object
        document = DocumentUpload(
            filename=file.filename,
            content_type=file.content_type,
            agent_id=agent_id,
            session_id=session_id,
            tags=tags.split(",") if tags else [],
            metadata={"upload_timestamp": datetime.utcnow().isoformat()}
        )
        
        # Process document
        chunks = await rag_service.process_document(document, file_content)
        
        # Store chunks
        result = await rag_service.store_document_chunks(chunks)
        
        return {
            "status": "success",
            "message": result,
            "document_name": file.filename,
            "chunks_created": len(chunks)
        }
        
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/{agent_id}")
async def list_documents(
    agent_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(verify_token)
):
    """List documents for an agent"""
    try:
        # Query ChromaDB for agent's documents
        results = rag_service.document_collection.get(
            where={"agent_id": agent_id},
            include=['metadatas']
        )
        
        # Group by document
        documents = {}
        for metadata in results['metadatas']:
            doc_name = metadata.get('document_name', 'Unknown')
            if doc_name not in documents:
                documents[doc_name] = {
                    "name": doc_name,
                    "chunks": 0,
                    "upload_date": metadata.get('processed_at'),
                    "tags": metadata.get('tags', [])
                }
            documents[doc_name]["chunks"] += 1
        
        return {"documents": list(documents.values())}
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{agent_id}/{document_name}")
async def delete_document(
    agent_id: str,
    document_name: str,
    credentials: HTTPAuthorizationCredentials = Depends(verify_token)
):
    """Delete a document and its chunks"""
    try:
        # Get document chunks to delete
        results = rag_service.document_collection.get(
            where={
                "agent_id": agent_id,
                "document_name": document_name
            },
            include=['ids']
        )
        
        if results['ids']:
            # Delete chunks
            rag_service.document_collection.delete(ids=results['ids'])
            
            return {
                "status": "success", 
                "message": f"Deleted {len(results['ids'])} chunks for document {document_name}"
            }
        else:
            raise HTTPException(status_code=404, detail="Document not found")
            
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# ==============================================
# services/rag-memory-service/requirements.txt
# ==============================================

fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
chromadb==0.4.18
sentence-transformers==2.2.2
openai==1.3.6
anthropic==0.7.8
httpx==0.25.2
numpy==1.24.3
PyPDF2==3.0.1
python-docx==1.1.0
python-multipart==0.0.6
aiofiles==23.2.0

# ==============================================
# services/rag-memory-service/Dockerfile
# ==============================================

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create directories
RUN mkdir -p /app/documents /app/models

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run application