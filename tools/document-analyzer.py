# tools/document-analyzer.py
#!/usr/bin/env python3
"""
Advanced Document Analysis Tool
Provides detailed analysis of processed documents and their memory integration
"""

import asyncio
import argparse
import json
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import httpx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict

class DocumentAnalyzer:
    """Advanced analysis tool for processed documents"""
    
    def __init__(self, base_url: str = "http://localhost:8006", 
                 mcp_url: str = "http://localhost:8080",
                 access_token: str = "secure-memory-token"):
        self.base_url = base_url
        self.mcp_url = mcp_url
        self.headers = {"Authorization": f"Bearer {access_token}"}
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def analyze_agent_documents(self, agent_id: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Comprehensive analysis of agent's documents"""
        
        print(f"🔍 Analyzing documents for agent: {agent_id}")
        if session_id:
            print(f"📋 Session: {session_id}")
        
        analysis = {
            "agent_id": agent_id,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "document_stats": {},
            "content_analysis": {},
            "memory_integration": {},
            "performance_metrics": {},
            "recommendations": []
        }
        
        try:
            # Get document list
            params = {"limit": 100}
            if session_id:
                params["session_id"] = session_id
            
            response = await self.client.get(
                f"{self.base_url}/documents/{agent_id}",
                headers=self.headers,
                params=params
            )
            
            if response.status_code != 200:
                raise Exception(f"Failed to get documents: {response.status_code}")
            
            documents = response.json()["documents"]
            print(f"📚 Found {len(documents)} documents")
            
            # Basic document statistics
            analysis["document_stats"] = await self._analyze_document_stats(documents)
            
            # Content analysis
            analysis["content_analysis"] = await self._analyze_content(agent_id, documents)
            
            # Memory integration analysis
            analysis["memory_integration"] = await self._analyze_memory_integration(agent_id, session_id)
            
            # Performance metrics
            analysis["performance_metrics"] = await self._analyze_performance(agent_id, documents)
            
            # Generate recommendations
            analysis["recommendations"] = self._generate_recommendations(analysis)
            
            return analysis
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            return {"error": str(e)}
    
    async def _analyze_document_stats(self, documents: List[Dict]) -> Dict[str, Any]:
        """Analyze basic document statistics"""
        
        if not documents:
            return {"total_documents": 0}
        
        stats = {
            "total_documents": len(documents),
            "document_types": Counter(doc["document_type"] for doc in documents),
            "total_chunks": sum(doc.get("chunk_count", 0) for doc in documents),
            "avg_chunks_per_doc": sum(doc.get("chunk_count", 0) for doc in documents) / len(documents),
            "total_file_size": sum(doc.get("file_size", 0) for doc in documents),
            "processing_timeline": []
        }
        
        # Processing timeline
        processed_docs = [doc for doc in documents if doc.get("processed_at")]
        if processed_docs:
            timeline = defaultdict(int)
            for doc in processed_docs:
                date = doc["processed_at"][:10]  # YYYY-MM-DD
                timeline[date] += 1
            
            stats["processing_timeline"] = dict(timeline)
            stats["most_active_day"] = max(timeline.items(), key=lambda x: x[1]) if timeline else None
        
        return stats
    
    async def _analyze_content(self, agent_id: str, documents: List[Dict]) -> Dict[str, Any]:
        """Analyze content patterns and themes"""
        
        content_analysis = {
            "keyword_analysis": {"top_keywords": [], "keyword_frequency": {}},
            "entity_analysis": {"top_entities": [], "entity_types": {}},
            "topic_distribution": {},
            "content_quality": {"avg_importance": 0.0, "quality_distribution": {}}
        }
        
        try:
            # Search for all content to get keywords and entities
            search_response = await self.client.post(
                f"{self.base_url}/search",
                headers=self.headers,
                data={
                    "query": "*",  # Broad search
                    "agent_id": agent_id,
                    "limit": 100,
                    "similarity_threshold": 0.3
                }
            )
            
            if search_response.status_code == 200:
                search_results = search_response.json()["results"]
                
                # Analyze keywords
                all_keywords = []
                all_entities = []
                importance_scores = []
                
                for result in search_results:
                    all_keywords.extend(result.get("keywords", []))
                    all_entities.extend(result.get("entities", []))
                    importance_scores.append(result.get("importance_score", 0.5))
                
                # Keyword analysis
                keyword_counter = Counter(all_keywords)
                content_analysis["keyword_analysis"] = {
                    "top_keywords": keyword_counter.most_common(20),
                    "total_unique_keywords": len(keyword_counter),
                    "keyword_frequency": dict(keyword_counter.most_common(10))
                }
                
                # Entity analysis
                entity_texts = [ent.get("text", "") for ent in all_entities if isinstance(ent, dict)]
                entity_types = [ent.get("label", "") for ent in all_entities if isinstance(ent, dict)]
                
                content_analysis["entity_analysis"] = {
                    "top_entities": Counter(entity_texts).most_common(15),
                    "entity_types": dict(Counter(entity_types)),
                    "total_entities": len(entity_texts)
                }
                
                # Content quality
                if importance_scores:
                    content_analysis["content_quality"] = {
                        "avg_importance": sum(importance_scores) / len(importance_scores),
                        "max_importance": max(importance_scores),
                        "min_importance": min(importance_scores),
                        "high_quality_chunks": len([s for s in importance_scores if s > 0.7]),
                        "quality_distribution": self._categorize_quality(importance_scores)
                    }
        
        except Exception as e:
            print(f"⚠️ Content analysis error: {e}")
        
        return content_analysis
    
    async def _analyze_memory_integration(self, agent_id: str, session_id: Optional[str]) -> Dict[str, Any]:
        """Analyze integration with memory system"""
        
        memory_analysis = {
            "document_memories": 0,
            "processing_memories": 0,
            "memory_types_used": {},
            "memory_importance_avg": 0.0,
            "recent_memory_activity": []
        }
        
        try:
            # Query memory system for document-related memories
            memory_query = {
                "query": "document processing",
                "agent_id": agent_id,
                "session_id": session_id,
                "memory_types": ["working", "episodic", "semantic"],
                "limit": 50
            }
            
            response = await self.client.post(
                f"{self.mcp_url}/memory/query",
                headers=self.headers,
                json=memory_query
            )
            
            if response.status_code == 200:
                memory_results = response.json()
                
                total_memories = 0
                all_importance_scores = []
                memory_types_count = {}
                
                for memory_type, memories in memory_results.items():
                    count = len(memories)
                    total_memories += count
                    memory_types_count[memory_type] = count
                    
                    for memory in memories:
                        importance = memory.get("importance", 0.5)
                        all_importance_scores.append(importance)
                
                memory_analysis.update({
                    "document_memories": total_memories,
                    "memory_types_used": memory_types_count,
                    "memory_importance_avg": sum(all_importance_scores) / len(all_importance_scores) if all_importance_scores else 0.0,
                    "total_document_related_memories": total_memories
                })
        
        except Exception as e:
            print(f"⚠️ Memory integration analysis error: {e}")
        
        return memory_analysis
    
    async def _analyze_performance(self, agent_id: str, documents: List[Dict]) -> Dict[str, Any]:
        """Analyze processing performance metrics"""
        
        performance = {
            "processing_efficiency": {},
            "throughput_metrics": {},
            "resource_utilization": {},
            "trends": {}
        }
        
        try:
            # Get processing statistics
            stats_response = await self.client.get(
                f"{self.base_url}/stats",
                headers=self.headers
            )
            
            if stats_response.status_code == 200:
                stats = stats_response.json()["processing_stats"]
                
                performance["processing_efficiency"] = {
                    "total_documents_processed": stats.get("documents_processed", 0),
                    "total_chunks_created": stats.get("chunks_created", 0),
                    "total_embeddings_generated": stats.get("embeddings_generated", 0),
                    "error_rate": stats.get("errors_count", 0) / max(stats.get("documents_processed", 1), 1),
                    "avg_chunks_per_document": stats.get("chunks_created", 0) / max(stats.get("documents_processed", 1), 1)
                }
            
            # Analyze document sizes and processing patterns
            if documents:
                file_sizes = [doc.get("file_size", 0) for doc in documents if doc.get("file_size")]
                chunk_counts = [doc.get("chunk_count", 0) for doc in documents if doc.get("chunk_count")]
                
                if file_sizes:
                    performance["throughput_metrics"] = {
                        "avg_file_size_mb": sum(file_sizes) / len(file_sizes) / (1024*1024),
                        "max_file_size_mb": max(file_sizes) / (1024*1024),
                        "avg_chunks_per_doc": sum(chunk_counts) / len(chunk_counts) if chunk_counts else 0
                    }
        
        except Exception as e:
            print(f"⚠️ Performance analysis error: {e}")
        
        return performance
    
    def _categorize_quality(self, importance_scores: List[float]) -> Dict[str, int]:
        """Categorize content quality based on importance scores"""
        categories = {"high": 0, "medium": 0, "low": 0}
        
        for score in importance_scores:
            if score > 0.7:
                categories["high"] += 1
            elif score > 0.4:
                categories["medium"] += 1
            else:
                categories["low"] += 1
        
        return categories
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        doc_stats = analysis.get("document_stats", {})
        content_analysis = analysis.get("content_analysis", {})
        memory_integration = analysis.get("memory_integration", {})
        performance = analysis.get("performance_metrics", {})
        
        # Document volume recommendations
        total_docs = doc_stats.get("total_documents", 0)
        if total_docs < 5:
            recommendations.append("📈 Consider processing more documents to improve analysis quality and training data")
        elif total_docs > 100:
            recommendations.append("🗂️ Large document collection detected - consider implementing automated categorization")
        
        # Content quality recommendations
        quality = content_analysis.get("content_quality", {})
        avg_importance = quality.get("avg_importance", 0)
        if avg_importance < 0.5:
            recommendations.append("⚡ Consider adjusting importance scoring criteria or improving document selection quality")
        elif avg_importance > 0.8:
            recommendations.append("✨ High-quality content detected - consider increasing memory retention for these documents")
        
        # Memory integration recommendations
        doc_memories = memory_integration.get("document_memories", 0)
        if doc_memories < total_docs * 0.5:
            recommendations.append("🧠 Low memory integration rate - ensure 'store_in_memory' is enabled for important documents")
        
        # Performance recommendations
        efficiency = performance.get("processing_efficiency", {})
        error_rate = efficiency.get("error_rate", 0)
        if error_rate > 0.1:
            recommendations.append("🔧 High error rate detected - review document formats and processing configuration")
        
        # Keyword diversity recommendations
        keyword_analysis = content_analysis.get("keyword_analysis", {})
        unique_keywords = keyword_analysis.get("total_unique_keywords", 0)
        if unique_keywords > 0 and total_docs > 0:
            keyword_diversity = unique_keywords / total_docs
            if keyword_diversity < 10:
                recommendations.append("📊 Low keyword diversity - consider processing documents from different domains")
        
        if not recommendations:
            recommendations.append("✅ System performance looks good - no immediate optimizations needed")
        
        return recommendations
    
    def print_analysis_report(self, analysis: Dict[str, Any]):
        """Print a formatted analysis report"""
        
        print("\n" + "="*80)
        print("📊 DOCUMENT ANALYSIS REPORT")
        print("="*80)
        print(f"Agent: {analysis['agent_id']}")
        if analysis.get("session_id"):
            print(f"Session: {analysis['session_id']}")
        print(f"Generated: {analysis['timestamp']}")
        
        # Document Statistics
        doc_stats = analysis.get("document_stats", {})
        print(f"\n📚 DOCUMENT STATISTICS")
        print(f"  Total Documents: {doc_stats.get('total_documents', 0)}")
        print(f"  Total Chunks: {doc_stats.get('total_chunks', 0)}")
        print(f"  Avg Chunks/Doc: {doc_stats.get('avg_chunks_per_doc', 0):.1f}")
        print(f"  Total Size: {doc_stats.get('total_file_size', 0) / (1024*1024):.1f} MB")
        
        # Document types
        doc_types = doc_stats.get("document_types", {})
        if doc_types:
            print(f"  Document Types:")
            for doc_type, count in doc_types.items():
                print(f"    - {doc_type}: {count}")
        
        # Content Analysis
        content = analysis.get("content_analysis", {})
        print(f"\n🔍 CONTENT ANALYSIS")
        
        # Keywords
        keywords = content.get("keyword_analysis", {})
        top_keywords = keywords.get("top_keywords", [])
        if top_keywords:
            print(f"  Top Keywords:")
            for keyword, count in top_keywords[:10]:
                print(f"    - {keyword}: {count}")
        
        # Entities
        entities = content.get("entity_analysis", {})
        top_entities = entities.get("top_entities", [])
        if top_entities:
            print(f"  Top Entities:")
            for entity, count in top_entities[:5]:
                print(f"    - {entity}: {count}")
        
        # Quality metrics
        quality = content.get("content_quality", {})
        if quality:
            print(f"  Content Quality:")
            print(f"    - Avg Importance: {quality.get('avg_importance', 0):.3f}")
            print(f"    - High Quality Chunks: {quality.get('high_quality_chunks', 0)}")
        
        # Memory Integration
        memory = analysis.get("memory_integration", {})
        print(f"\n🧠 MEMORY INTEGRATION")
        print(f"  Document-related Memories: {memory.get('document_memories', 0)}")
        print(f"  Avg Memory Importance: {memory.get('memory_importance_avg', 0):.3f}")
        
        memory_types = memory.get("memory_types_used", {})
        if memory_types:
            print(f"  Memory Types:")
            for mem_type, count in memory_types.items():
                print(f"    - {mem_type}: {count}")
        
        # Performance Metrics
        performance = analysis.get("performance_metrics", {})
        efficiency = performance.get("processing_efficiency", {})
        if efficiency:
            print(f"\n⚡ PERFORMANCE METRICS")
            print(f"  Error Rate: {efficiency.get('error_rate', 0):.1%}")
            print(f"  Avg Chunks/Document: {efficiency.get('avg_chunks_per_document', 0):.1f}")
        
        # Recommendations
        recommendations = analysis.get("recommendations", [])
        if recommendations:
            print(f"\n💡 RECOMMENDATIONS")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        
        print("\n" + "="*80)
    
    async def export_analysis(self, analysis: Dict[str, Any], format: str = "json", 
                            filename: Optional[str] = None):
        """Export analysis results to file"""
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            agent_id = analysis.get("agent_id", "unknown").replace("-", "_")
            filename = f"document_analysis_{agent_id}_{timestamp}.{format}"
        
        try:
            if format == "json":
                with open(filename, 'w') as f:
                    json.dump(analysis, f, indent=2, default=str)
            
            elif format == "csv":
                # Export key metrics to CSV
                data = []
                
                # Document stats
                doc_stats = analysis.get("document_stats", {})
                data.append({
                    "metric": "total_documents",
                    "value": doc_stats.get("total_documents", 0),
                    "category": "documents"
                })
                
                # Add more metrics as needed
                df = pd.DataFrame(data)
                df.to_csv(filename, index=False)
            
            print(f"✅ Analysis exported to: {filename}")
            
        except Exception as e:
            print(f"❌ Export failed: {e}")
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()

async def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Advanced Document Analysis Tool")
    parser.add_argument("agent_id", help="Agent ID to analyze")
    parser.add_argument("--session-id", help="Optional session ID filter")
    parser.add_argument("--export", choices=["json", "csv"], help="Export format")
    parser.add_argument("--output", help="Output filename")
    parser.add_argument("--url", default="http://localhost:8006", help="DocumentProcessor URL")
    parser.add_argument("--mcp-url", default="http://localhost:8080", help="MCP Server URL")
    parser.add_argument("--token", default="secure-memory-token", help="Access token")
    
    args = parser.parse_args()
    
    analyzer = DocumentAnalyzer(
        base_url=args.url,
        mcp_url=args.mcp_url,
        access_token=args.token
    )
    
    try:
        # Run analysis
        analysis = await analyzer.analyze_agent_documents(args.agent_id, args.session_id)
        
        if "error" in analysis:
            print(f"❌ Analysis failed: {analysis['error']}")
            sys.exit(1)
        
        # Print report
        analyzer.print_analysis_report(analysis)
        
        # Export if requested
        if args.export:
            await analyzer.export_analysis(analysis, args.export, args.output)
    
    except KeyboardInterrupt:
        print("\n⚠️ Analysis interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)
    finally:
        await analyzer.close()

if __name__ == "__main__":
    asyncio.run(main())

---

# Enhanced MCP Memory Server integration for DocumentProcessor
# Add to services/mcp-memory-server/main.py

@app.post("/memory/query-documents")
async def query_documents_with_memory(
    query: str = Form(...),
    agent_id: str = Form(...),
    session_id: str = Form(...),
    include_document_chunks: bool = Form(True),
    memory_weight: float = Form(0.3),
    document_weight: float = Form(0.7),
    limit: int = Form(10),
    credentials: HTTPAuthorizationCredentials = Depends(verify_token)
):
    """Enhanced query that combines memory and document search"""
    
    try:
        # Query memory system
        memory_query = MemoryQuery(
            query=query,
            agent_id=agent_id,
            session_id=session_id,
            memory_types=["working", "episodic", "semantic"],
            limit=max(int(limit * memory_weight * 2), 3)
        )
        
        memory_results = await memory_system.query_all_memories(memory_query)
        
        # Query document processor if enabled
        document_results = []
        if include_document_chunks:
            try:
                async with httpx.AsyncClient() as client:
                    doc_response = await client.post(
                        "http://document-processor:8000/search",
                        headers={"Authorization": f"Bearer {access_token}"},
                        data={
                            "query": query,
                            "agent_id": agent_id,
                            "limit": max(int(limit * document_weight * 2), 3),
                            "similarity_threshold": 0.6
                        }
                    )
                    
                    if doc_response.status_code == 200:
                        document_results = doc_response.json().get("results", [])
            except Exception as e:
                logger.warning(f"Document search failed: {e}")
        
        # Combine and rank results
        combined_results = {
            "query": query,
            "memory_results": memory_results,
            "document_results": document_results,
            "combined_ranking": [],
            "total_results": 0
        }
        
        # Create combined ranking
        all_items = []
        
        # Add memory results
        for memory_type, memories in memory_results.items():
            for memory in memories:
                all_items.append({
                    "type": "memory",
                    "subtype": memory_type,
                    "content": memory.get("content", ""),
                    "importance": memory.get("importance", 0.5),
                    "timestamp": memory.get("timestamp"),
                    "source_score": memory.get("similarity_score", 0.8),
                    "weighted_score": memory.get("similarity_score", 0.8) * memory_weight,
                    "metadata": memory.get("metadata", {})
                })
        
        # Add document results
        for doc_result in document_results:
            all_items.append({
                "type": "document",
                "subtype": "chunk",
                "content": doc_result.get("content", ""),
                "importance": doc_result.get("importance_score", 0.5),
                "document_filename": doc_result.get("document_filename", ""),
                "source_score": doc_result.get("similarity_score", 0.8),
                "weighted_score": doc_result.get("similarity_score", 0.8) * document_weight,
                "keywords": doc_result.get("keywords", []),
                "entities": doc_result.get("entities", [])
            })
        
        # Sort by weighted score and take top results
        all_items.sort(key=lambda x: x["weighted_score"], reverse=True)
        combined_results["combined_ranking"] = all_items[:limit]
        combined_results["total_results"] = len(combined_results["combined_ranking"])
        
        return combined_results
        
    except Exception as e:
        logger.error(f"Enhanced query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/store-document-insight")
async def store_document_insight(
    document_id: str = Form(...),
    insight: str = Form(...),
    agent_id: str = Form(...),
    session_id: str = Form(...),
    importance: float = Form(0.7),
    insight_type: str = Form("analysis"),
    credentials: HTTPAuthorizationCredentials = Depends(verify_token)
):
    """Store insights derived from document analysis"""
    
    try:
        # Get document information
        async with httpx.AsyncClient() as client:
            doc_response = await client.get(
                f"http://document-processor:8000/document/{document_id}",
                headers={"Authorization": f"Bearer {access_token}"}
            )
            
            if doc_response.status_code == 200:
                doc_info = doc_response.json()
                
                # Create insight memory
                memory = MemoryItem(
                    content=f"Document insight from '{doc_info['filename']}': {insight}",
                    memory_type="episodic",
                    agent_id=agent_id,
                    session_id=session_id,
                    importance=importance,
                    tags=["document_insight", insight_type, "analysis"],
                    metadata={
                        "document_id": document_id,
                        "document_filename": doc_info["filename"],
                        "document_type": doc_info["document_type"],
                        "insight_type": insight_type,
                        "derived_from": "document_analysis",
                        "original_insight": insight
                    }
                )
                
                # Store insight memory
                memory_id = await memory_system.store_episodic_memory(memory)
                
                return {
                    "status": "success",
                    "memory_id": memory_id,
                    "insight_stored": True,
                    "document_linked": True
                }
            else:
                raise HTTPException(status_code=404, detail="Document not found")
                
    except Exception as e:
        logger.error(f"Error storing document insight: {e}")
        raise HTTPException(status_code=500, detail=str(e))

