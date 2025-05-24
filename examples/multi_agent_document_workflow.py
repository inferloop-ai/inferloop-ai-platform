# examples/multi_agent_document_workflow.py
"""
Multi-Agent Document Processing Workflow Example
Shows how different agents collaborate on document processing
"""

import asyncio
import httpx
import json
from datetime import datetime
from typing import List, Dict, Any

class MultiAgentDocumentWorkflow:
    """Orchestrates multiple agents for comprehensive document processing"""
    
    def __init__(self):
        self.access_token = "secure-memory-token"
        self.base_headers = {"Authorization": f"Bearer {self.access_token}"}
        
        # Service URLs
        self.document_processor_url = "http://localhost:8006"
        self.mcp_server_url = "http://localhost:8080"
        self.rag_service_url = "http://localhost:8001"
        self.orchestrator_url = "http://localhost:8003"
        
        self.client = httpx.AsyncClient(timeout=300.0)
    
    async def register_specialized_agents(self):
        """Register specialized agents for document processing workflow"""
        
        agents = [
            {
                "agent_id": "document-processor-001",
                "name": "Document Processing Specialist",
                "type": "specialist",
                "capabilities": ["document_parsing", "text_extraction", "nlp_analysis"],
                "specialization": "document_processing",
                "max_concurrent_tasks": 3
            },
            {
                "agent_id": "research-analyst-001", 
                "name": "Research Analyst",
                "type": "specialist",
                "capabilities": ["research", "analysis", "summarization"],
                "specialization": "research",
                "max_concurrent_tasks": 5
            },
            {
                "agent_id": "knowledge-curator-001",
                "name": "Knowledge Curator",
                "type": "specialist", 
                "capabilities": ["knowledge_organization", "relationship_mapping", "insight_extraction"],
                "specialization": "knowledge_management",
                "max_concurrent_tasks": 4
            }
        ]
        
        for agent in agents:
            try:
                response = await self.client.post(
                    f"{self.orchestrator_url}/agents/register",
                    headers=self.base_headers,
                    json=agent
                )
                if response.status_code == 200:
                    print(f"✅ Registered {agent['name']}")
                else:
                    print(f"❌ Failed to register {agent['name']}: {response.status_code}")
            except Exception as e:
                print(f"❌ Error registering {agent['name']}: {e}")
    
    async def process_document_collection(self, document_paths: List[str], 
                                        project_name: str) -> Dict[str, Any]:
        """Process a collection of documents with multiple agents"""
        
        session_id = f"document-workflow-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        workflow_results = {
            "session_id": session_id,
            "project_name": project_name,
            "processed_documents": [],
            "analysis_results": {},
            "knowledge_insights": {},
            "errors": []
        }
        
        print(f"🚀 Starting document workflow for project: {project_name}")
        print(f"📋 Session ID: {session_id}")
        
        # Step 1: Process documents with DocumentProcessor agent
        print("\n📄 Step 1: Processing documents...")
        processing_jobs = []
        
        for doc_path in document_paths:
            try:
                # Upload and process document
                with open(doc_path, 'rb') as f:
                    files = {"files": (doc_path.split('/')[-1], f, "application/octet-stream")}
                    form_data = {
                        "agent_id": "document-processor-001",
                        "session_id": session_id,
                        "tags": f"project_{project_name},workflow",
                        "custom_metadata": json.dumps({
                            "project": project_name,
                            "workflow_stage": "initial_processing"
                        }),
                        "chunk_size": 1000,
                        "enable_nlp_analysis": True,
                        "store_in_memory": True,
                        "memory_importance": 0.7
                    }
                    
                    response = await self.client.post(
                        f"{self.document_processor_url}/process",
                        headers=self.base_headers,
                        data=form_data,
                        files=files
                    )
                    
                    if response.status_code == 200:
                        job_id = response.json()["job_id"]
                        processing_jobs.append({
                            "job_id": job_id,
                            "document_path": doc_path,
                            "filename": doc_path.split('/')[-1]
                        })
                        print(f"  📤 Submitted {doc_path.split('/')[-1]} (Job: {job_id})")
                    else:
                        error_msg = f"Failed to process {doc_path}: {response.status_code}"
                        workflow_results["errors"].append(error_msg)
                        print(f"  ❌ {error_msg}")
                        
            except Exception as e:
                error_msg = f"Error processing {doc_path}: {str(e)}"
                workflow_results["errors"].append(error_msg)
                print(f"  ❌ {error_msg}")
        
        # Wait for all processing jobs to complete
        print("\n⏳ Waiting for document processing to complete...")
        for job in processing_jobs:
            try:
                final_status = await self.wait_for_job_completion(job["job_id"])
                if final_status["status"] == "completed":
                    workflow_results["processed_documents"].append({
                        "filename": job["filename"],
                        "job_id": job["job_id"],
                        "result_summary": final_status["result_summary"]
                    })
                    print(f"  ✅ {job['filename']} processed successfully")
                else:
                    error_msg = f"Processing failed for {job['filename']}: {final_status.get('errors', [])}"
                    workflow_results["errors"].append(error_msg)
                    print(f"  ❌ {error_msg}")
            except Exception as e:
                error_msg = f"Error waiting for {job['filename']}: {str(e)}"
                workflow_results["errors"].append(error_msg)
                print(f"  ❌ {error_msg}")
        
        # Step 2: Research analysis using processed documents
        print("\n🔍 Step 2: Conducting research analysis...")
        try:
            analysis_task = {
                "title": f"Research Analysis for {project_name}",
                "description": f"Analyze processed documents and extract key insights for {project_name}",
                "requester_id": "workflow-orchestrator",
                "session_id": session_id,
                "assigned_agent_id": "research-analyst-001",
                "priority": 8,
                "required_capabilities": ["research", "analysis"],
                "context": {
                    "project_name": project_name,
                    "processed_document_count": len(workflow_results["processed_documents"]),
                    "analysis_type": "comprehensive_research"
                }
            }
            
            response = await self.client.post(
                f"{self.orchestrator_url}/tasks/submit",
                headers=self.base_headers,
                json=analysis_task
            )
            
            if response.status_code == 200:
                task_id = response.json()["task_id"]
                print(f"  📊 Research analysis task submitted (Task: {task_id})")
                
                # Simulate research analysis using RAG
                research_results = await self.conduct_research_analysis(session_id, project_name)
                workflow_results["analysis_results"] = research_results
                
                # Complete the task
                await self.client.put(
                    f"{self.orchestrator_url}/tasks/{task_id}/complete",
                    headers=self.base_headers,
                    json={
                        "result": research_results,
                        "agent_id": "research-analyst-001"
                    }
                )
                print(f"  ✅ Research analysis completed")
            else:
                error_msg = f"Failed to submit research analysis task: {response.status_code}"
                workflow_results["errors"].append(error_msg)
                print(f"  ❌ {error_msg}")
                
        except Exception as e:
            error_msg = f"Error in research analysis: {str(e)}"
            workflow_results["errors"].append(error_msg)
            print(f"  ❌ {error_msg}")
        
        # Step 3: Knowledge curation and insight extraction
        print("\n🧠 Step 3: Curating knowledge and extracting insights...")
        try:
            curation_task = {
                "title": f"Knowledge Curation for {project_name}",
                "description": f"Organize knowledge and extract actionable insights from {project_name} documents",
                "requester_id": "workflow-orchestrator",
                "session_id": session_id,
                "assigned_agent_id": "knowledge-curator-001",
                "priority": 7,
                "required_capabilities": ["knowledge_organization", "insight_extraction"],
                "context": {
                    "project_name": project_name,
                    "research_results": workflow_results["analysis_results"],
                    "curation_focus": "actionable_insights"
                }
            }
            
            response = await self.client.post(
                f"{self.orchestrator_url}/tasks/submit",
                headers=self.base_headers,
                json=curation_task
            )
            
            if response.status_code == 200:
                task_id = response.json()["task_id"]
                print(f"  🎯 Knowledge curation task submitted (Task: {task_id})")
                
                # Conduct knowledge curation
                knowledge_insights = await self.conduct_knowledge_curation(session_id, project_name)
                workflow_results["knowledge_insights"] = knowledge_insights
                
                # Complete the task
                await self.client.put(
                    f"{self.orchestrator_url}/tasks/{task_id}/complete",
                    headers=self.base_headers,
                    json={
                        "result": knowledge_insights,
                        "agent_id": "knowledge-curator-001"
                    }
                )
                print(f"  ✅ Knowledge curation completed")
            else:
                error_msg = f"Failed to submit knowledge curation task: {response.status_code}"
                workflow_results["errors"].append(error_msg)
                print(f"  ❌ {error_msg}")
                
        except Exception as e:
            error_msg = f"Error in knowledge curation: {str(e)}"
            workflow_results["errors"].append(error_msg)
            print(f"  ❌ {error_msg}")
        
        # Step 4: Store workflow results in memory
        print("\n💾 Step 4: Storing workflow results in memory...")
        await self.store_workflow_memory(session_id, project_name, workflow_results)
        
        return workflow_results
    
    async def wait_for_job_completion(self, job_id: str, poll_interval: int = 10) -> Dict[str, Any]:
        """Wait for document processing job to complete"""
        while True:
            try:
                response = await self.client.get(
                    f"{self.document_processor_url}/job/{job_id}",
                    headers=self.base_headers
                )
                
                if response.status_code == 200:
                    status = response.json()
                    if status["status"] in ["completed", "failed"]:
                        return status
                    
                    print(f"    📊 Job {job_id}: {status['progress_percentage']:.1f}% - {status.get('current_step', 'Processing...')}")
                    await asyncio.sleep(poll_interval)
                else:
                    raise Exception(f"Failed to get job status: {response.status_code}")
                    
            except Exception as e:
                raise Exception(f"Error waiting for job completion: {str(e)}")
    
    async def conduct_research_analysis(self, session_id: str, project_name: str) -> Dict[str, Any]:
        """Conduct research analysis using RAG service"""
        try:
            # Use RAG to analyze the processed documents
            research_queries = [
                f"What are the main themes in {project_name}?",
                f"What are the key findings and conclusions in {project_name}?", 
                f"What recommendations or next steps are suggested in {project_name}?",
                f"What are the potential risks or challenges mentioned in {project_name}?"
            ]
            
            analysis_results = {
                "main_themes": [],
                "key_findings": [],
                "recommendations": [],
                "risks_challenges": [],
                "summary": ""
            }
            
            for i, query in enumerate(research_queries):
                try:
                    rag_query = {
                        "query": query,
                        "agent_id": "research-analyst-001",
                        "session_id": session_id,
                        "max_results": 5,
                        "include_memory_context": True,
                        "expand_query": True
                    }
                    
                    response = await self.client.post(
                        f"{self.rag_service_url}/query",
                        headers=self.base_headers,
                        json=rag_query
                    )
                    
                    if response.status_code == 200:
                        rag_result = response.json()
                        
                        if i == 0:  # Main themes
                            analysis_results["main_themes"] = self.extract_themes_from_answer(rag_result["answer"])
                        elif i == 1:  # Key findings
                            analysis_results["key_findings"] = self.extract_findings_from_answer(rag_result["answer"])
                        elif i == 2:  # Recommendations
                            analysis_results["recommendations"] = self.extract_recommendations_from_answer(rag_result["answer"])
                        elif i == 3:  # Risks/challenges
                            analysis_results["risks_challenges"] = self.extract_risks_from_answer(rag_result["answer"])
                        
                        print(f"    📊 Completed analysis query {i+1}/{len(research_queries)}")
                    else:
                        print(f"    ⚠️ RAG query {i+1} failed: {response.status_code}")
                        
                except Exception as e:
                    print(f"    ❌ Error in research query {i+1}: {str(e)}")
            
            # Generate summary
            analysis_results["summary"] = self.generate_analysis_summary(analysis_results)
            
            return analysis_results
            
        except Exception as e:
            print(f"❌ Error in research analysis: {str(e)}")
            return {"error": str(e)}
    
    async def conduct_knowledge_curation(self, session_id: str, project_name: str) -> Dict[str, Any]:
        """Conduct knowledge curation and insight extraction"""
        try:
            # Search for all processed documents in this session
            search_results = await self.client.post(
                f"{self.document_processor_url}/search",
                headers=self.base_headers,
                data={
                    "query": project_name,
                    "agent_id": "knowledge-curator-001", 
                    "limit": 20,
                    "similarity_threshold": 0.5
                }
            )
            
            knowledge_insights = {
                "key_concepts": [],
                "relationships": [],
                "actionable_insights": [],
                "knowledge_gaps": [],
                "confidence_score": 0.0
            }
            
            if search_results.status_code == 200:
                results = search_results.json()["results"]
                
                # Extract key concepts from keywords and entities
                all_keywords = []
                all_entities = []
                
                for result in results:
                    all_keywords.extend(result.get("keywords", []))
                    all_entities.extend(result.get("entities", []))
                
                # Get most frequent concepts
                from collections import Counter
                keyword_counts = Counter(all_keywords)
                entity_counts = Counter([entity.get("text", "") for entity in all_entities if isinstance(entity, dict)])
                
                knowledge_insights["key_concepts"] = [
                    {"concept": concept, "frequency": count, "type": "keyword"}
                    for concept, count in keyword_counts.most_common(10)
                ]
                
                knowledge_insights["key_concepts"].extend([
                    {"concept": concept, "frequency": count, "type": "entity"}
                    for concept, count in entity_counts.most_common(5)
                ])
                
                # Generate actionable insights (simplified)
                knowledge_insights["actionable_insights"] = [
                    f"Focus on {concept['concept']} which appears {concept['frequency']} times across documents"
                    for concept in knowledge_insights["key_concepts"][:3]
                ]
                
                # Calculate confidence score
                knowledge_insights["confidence_score"] = min(len(results) / 10.0, 1.0)
                
                print(f"    🎯 Extracted {len(knowledge_insights['key_concepts'])} key concepts")
                print(f"    💡 Generated {len(knowledge_insights['actionable_insights'])} actionable insights")
            
            return knowledge_insights
            
        except Exception as e:
            print(f"❌ Error in knowledge curation: {str(e)}")
            return {"error": str(e)}
    
    async def store_workflow_memory(self, session_id: str, project_name: str, 
                                  workflow_results: Dict[str, Any]):
        """Store workflow results in memory system"""
        try:
            memory_content = f"Completed multi-agent document workflow for {project_name}. Processed {len(workflow_results['processed_documents'])} documents with comprehensive analysis and knowledge curation."
            
            memory_data = {
                "content": memory_content,
                "memory_type": "episodic",
                "agent_id": "workflow-orchestrator", 
                "session_id": session_id,
                "importance": 0.8,
                "tags": ["workflow", "document_processing", "multi_agent", project_name.lower().replace(" ", "_")],
                "metadata": {
                    "workflow_type": "multi_agent_document_processing",
                    "project_name": project_name,
                    "documents_processed": len(workflow_results["processed_documents"]),
                    "analysis_completed": bool(workflow_results["analysis_results"]),
                    "insights_generated": bool(workflow_results["knowledge_insights"]),
                    "error_count": len(workflow_results["errors"]),
                    "workflow_results": workflow_results
                }
            }
            
            response = await self.client.post(
                f"{self.mcp_server_url}/memory/store",
                headers=self.base_headers,
                json=memory_data
            )
            
            if response.status_code == 200:
                print(f"  ✅ Workflow results stored in memory")
            else:
                print(f"  ⚠️ Failed to store workflow memory: {response.status_code}")
                
        except Exception as e:
            print(f"  ❌ Error storing workflow memory: {str(e)}")
    
    # Helper methods for extracting information from RAG responses
    def extract_themes_from_answer(self, answer: str) -> List[str]:
        """Extract themes from RAG answer (simplified)"""
        # This would be more sophisticated in practice
        themes = []
        if "theme" in answer.lower():
            # Simple extraction - in practice, use NLP
            sentences = answer.split('.')
            for sentence in sentences:
                if "theme" in sentence.lower():
                    themes.append(sentence.strip()[:100])  # Limit length
        return themes[:5]  # Return top 5
    
    def extract_findings_from_answer(self, answer: str) -> List[str]:
        """Extract key findings from RAG answer"""
        findings = []
        keywords = ["finding", "result", "conclusion", "shows", "indicates"]
        sentences = answer.split('.')
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in keywords):
                findings.append(sentence.strip()[:150])
        return findings[:5]
    
    def extract_recommendations_from_answer(self, answer: str) -> List[str]:
        """Extract recommendations from RAG answer"""
        recommendations = []
        keywords = ["recommend", "suggest", "should", "must", "need to"]
        sentences = answer.split('.')
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in keywords):
                recommendations.append(sentence.strip()[:150])
        return recommendations[:5]
    
    def extract_risks_from_answer(self, answer: str) -> List[str]:
        """Extract risks and challenges from RAG answer"""
        risks = []
        keywords = ["risk", "challenge", "problem", "issue", "concern", "threat"]
        sentences = answer.split('.')
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in keywords):
                risks.append(sentence.strip()[:150])
        return risks[:5]
    
    def generate_analysis_summary(self, analysis_results: Dict[str, Any]) -> str:
        """Generate a summary of the analysis results"""
        summary_parts = []
        
        if analysis_results["main_themes"]:
            summary_parts.append(f"Identified {len(analysis_results['main_themes'])} main themes")
        
        if analysis_results["key_findings"]:
            summary_parts.append(f"extracted {len(analysis_results['key_findings'])} key findings")
        
        if analysis_results["recommendations"]:
            summary_parts.append(f"generated {len(analysis_results['recommendations'])} recommendations")
        
        if analysis_results["risks_challenges"]:
            summary_parts.append(f"identified {len(analysis_results['risks_challenges'])} risks/challenges")
        
        if summary_parts:
            return f"Analysis completed: {', '.join(summary_parts)}."
        else:
            return "Analysis completed with limited results."
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()

# Example usage
async def main():
    """Example multi-agent document workflow"""
    
    workflow = MultiAgentDocumentWorkflow()
    
    try:
        # Register agents
        print("🤖 Registering specialized agents...")
        await workflow.register_specialized_agents()
        
        # Process document collection
        document_paths = [
            "./documents/research_paper_1.pdf",
            "./documents/market_analysis.docx", 
            "./documents/technical_specifications.txt"
        ]
        
        results = await workflow.process_document_collection(
            document_paths=document_paths,
            project_name="AI Research Initiative 2024"
        )
        
        # Display results
        print("\n" + "="*60)
        print("📊 WORKFLOW RESULTS SUMMARY")
        print("="*60)
        print(f"Project: {results['project_name']}")
        print(f"Session: {results['session_id']}")
        print(f"Documents Processed: {len(results['processed_documents'])}")
        print(f"Errors: {len(results['errors'])}")
        
        if results["analysis_results"]:
            analysis = results["analysis_results"] 
            print(f"\n🔍 Research Analysis:")
            print(f"  - Themes: {len(analysis.get('main_themes', []))}")
            print(f"  - Findings: {len(analysis.get('key_findings', []))}")
            print(f"  - Recommendations: {len(analysis.get('recommendations', []))}")
            print(f"  - Summary: {analysis.get('summary', 'N/A')}")
        
        if results["knowledge_insights"]:
            insights = results["knowledge_insights"]
            print(f"\n🧠 Knowledge Insights:")
            print(f"  - Key Concepts: {len(insights.get('key_concepts', []))}")
            print(f"  - Actionable Insights: {len(insights.get('actionable_insights', []))}")
            print(f"  - Confidence Score: {insights.get('confidence_score', 0):.2f}")
        
        if results["errors"]:
            print(f"\n❌ Errors encountered:")
            for error in results["errors"]:
                print(f"  - {error}")
        
        print("\n✅ Multi-agent document workflow completed!")
        
    except Exception as e:
        print(f"❌ Workflow failed: {e}")
    
    finally:
        await workflow.close()

if __name__ == "__main__":
    asyncio.run(main())

