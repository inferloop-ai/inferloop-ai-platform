# services/agent-memory-orchestrator/main.py
"""
Agent Memory Orchestrator
Multi-Agent Coordination with Shared Memory Management
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4
from enum import Enum

import httpx
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import openai
from anthropic import Anthropic

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==============================================
# Configuration & Models
# ==============================================

class AgentStatus(str, Enum):
    IDLE = "idle"
    ACTIVE = "active"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"

class TaskStatus(str, Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class OrchestratorConfig:
    def __init__(self):
        self.memory_system_url = os.getenv("MEMORY_SYSTEM_URL", "http://mcp-memory-server:8080")
        self.rag_service_url = os.getenv("RAG_SERVICE_URL", "http://rag-memory-service:8000")
        self.mcp_service_url = os.getenv("MCP_SERVICE_URL", "http://mcp-memory-server:8080")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        
        # Agent configuration
        self.max_concurrent_agents = int(os.getenv("MAX_CONCURRENT_AGENTS", "10"))
        self.agent_memory_sync_interval = int(os.getenv("AGENT_MEMORY_SYNC_INTERVAL", "300"))  # 5 minutes
        self.enable_multi_agent_coordination = os.getenv("ENABLE_MULTI_AGENT_COORDINATION", "true").lower() == "true"
        self.enable_agent_learning = os.getenv("ENABLE_AGENT_LEARNING", "true").lower() == "true"
        self.working_memory_sharing = os.getenv("WORKING_MEMORY_SHARING", "true").lower() == "true"

config = OrchestratorConfig()

# Pydantic Models
class AgentProfile(BaseModel):
    agent_id: str
    name: str
    type: str  # "general", "specialist", "coordinator", "learner"
    capabilities: List[str]
    specialization: Optional[str] = None
    max_concurrent_tasks: int = 3
    memory_profile: str = "balanced"  # "conservative", "aggressive", "balanced"
    collaboration_style: str = "cooperative"  # "cooperative", "competitive", "independent"
    learning_rate: float = 0.1
    status: AgentStatus = AgentStatus.IDLE
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_active: datetime = Field(default_factory=datetime.utcnow)

class Task(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    description: str
    requester_id: str
    session_id: str
    assigned_agent_id: Optional[str] = None
    collaborating_agents: List[str] = Field(default_factory=list)
    priority: int = Field(default=5, ge=1, le=10)  # 1 = low, 10 = high
    estimated_duration_minutes: Optional[int] = None
    required_capabilities: List[str] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    assigned_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

class AgentCommunication(BaseModel):
    sender_id: str
    receiver_id: str
    session_id: str
    message_type: str  # "request", "response", "broadcast", "memory_share"
    content: str
    memory_context: Optional[Dict[str, Any]] = None
    requires_response: bool = False
    priority: int = 5
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class CollaborationRequest(BaseModel):
    initiator_agent_id: str
    target_agent_ids: List[str]
    session_id: str
    collaboration_type: str  # "parallel", "sequential", "hierarchical"
    shared_context: Dict[str, Any]
    expected_outcome: str
    max_duration_minutes: Optional[int] = 30

# ==============================================
# Agent Management
# ==============================================

class AgentManager:
    def __init__(self):
        self.agents: Dict[str, AgentProfile] = {}
        self.agent_connections: Dict[str, WebSocket] = {}
        self.active_tasks: Dict[str, Task] = {}
        self.task_queue: List[Task] = []
        self.collaboration_sessions: Dict[str, Dict] = {}
        
    def register_agent(self, agent: AgentProfile) -> str:
        """Register a new agent"""
        self.agents[agent.agent_id] = agent
        logger.info(f"Registered agent: {agent.agent_id} ({agent.type})")
        return agent.agent_id
    
    def get_agent(self, agent_id: str) -> Optional[AgentProfile]:
        """Get agent profile"""
        return self.agents.get(agent_id)
    
    def update_agent_status(self, agent_id: str, status: AgentStatus):
        """Update agent status"""
        if agent_id in self.agents:
            self.agents[agent_id].status = status
            self.agents[agent_id].last_active = datetime.utcnow()
    
    def get_available_agents(self, required_capabilities: List[str] = None) -> List[AgentProfile]:
        """Get agents that are available and have required capabilities"""
        available = []
        for agent in self.agents.values():
            if agent.status in [AgentStatus.IDLE, AgentStatus.ACTIVE]:
                # Check if agent has required capabilities
                if required_capabilities:
                    if not all(cap in agent.capabilities for cap in required_capabilities):
                        continue
                
                # Check if agent has capacity for more tasks
                current_tasks = sum(1 for task in self.active_tasks.values() 
                                  if task.assigned_agent_id == agent.agent_id)
                if current_tasks >= agent.max_concurrent_tasks:
                    continue
                
                available.append(agent)
        
        return available
    
    def find_best_agent_for_task(self, task: Task) -> Optional[AgentProfile]:
        """Find the best agent for a specific task"""
        available_agents = self.get_available_agents(task.required_capabilities)
        
        if not available_agents:
            return None
        
        # Score agents based on various factors
        scored_agents = []
        for agent in available_agents:
            score = 0
            
            # Base score from idle status
            if agent.status == AgentStatus.IDLE:
                score += 10
            elif agent.status == AgentStatus.ACTIVE:
                score += 5
            
            # Specialization match
            if task.required_capabilities:
                matching_caps = sum(1 for cap in task.required_capabilities if cap in agent.capabilities)
                score += matching_caps * 3
            
            # Type-specific bonuses
            if agent.specialization and task.context.get("domain") == agent.specialization:
                score += 15
            
            # Priority handling
            if task.priority >= 8 and "high_priority" in agent.capabilities:
                score += 10
            
            # Current workload (prefer less busy agents)
            current_tasks = sum(1 for t in self.active_tasks.values() 
                              if t.assigned_agent_id == agent.agent_id)
            score -= current_tasks * 2
            
            scored_agents.append((agent, score))
        
        # Return the highest scoring agent
        scored_agents.sort(key=lambda x: x[1], reverse=True)
        return scored_agents[0][0] if scored_agents else None

# ==============================================
# Memory Coordination
# ==============================================

class MemoryCoordinator:
    def __init__(self, memory_system_url: str, access_token: str):
        self.memory_system_url = memory_system_url
        self.access_token = access_token
        self.http_client = httpx.AsyncClient(timeout=30.0)
        self.shared_memory_sessions: Dict[str, Set[str]] = {}  # session_id -> set of agent_ids
        
    async def share_memory_between_agents(self, source_agent_id: str, target_agent_ids: List[str], 
                                        session_id: str, memory_types: List[str] = None) -> Dict[str, Any]:
        """Share memories between agents"""
        if not config.working_memory_sharing:
            return {"status": "disabled", "message": "Memory sharing is disabled"}
        
        try:
            # Get memories from source agent
            memory_query = {
                "query": "recent important memories",
                "agent_id": source_agent_id,
                "session_id": session_id,
                "memory_types": memory_types or ["working", "episodic"],
                "limit": 10,
                "similarity_threshold": 0.6
            }
            
            response = await self.http_client.post(
                f"{self.memory_system_url}/memory/query",
                json=memory_query,
                headers={"Authorization": f"Bearer {self.access_token}"}
            )
            
            if response.status_code != 200:
                return {"status": "error", "message": "Failed to retrieve source memories"}
            
            source_memories = response.json()
            
            # Share relevant memories with target agents
            shared_count = 0
            for target_agent_id in target_agent_ids:
                for memory_type, memories in source_memories.items():
                    for memory in memories:
                        # Create shared memory entry
                        shared_memory = {
                            "content": f"[SHARED FROM {source_agent_id}] {memory['content']}",
                            "memory_type": "working",  # Start in working memory
                            "agent_id": target_agent_id,
                            "session_id": session_id,
                            "importance": memory.get("importance", 0.5) * 0.8,  # Slightly reduce importance
                            "tags": memory.get("tags", []) + ["shared_memory", f"from_{source_agent_id}"],
                            "metadata": {
                                **memory.get("metadata", {}),
                                "shared_from": source_agent_id,
                                "shared_at": datetime.utcnow().isoformat(),
                                "original_memory_type": memory_type
                            }
                        }
                        
                        store_response = await self.http_client.post(
                            f"{self.memory_system_url}/memory/store",
                            json=shared_memory,
                            headers={"Authorization": f"Bearer {self.access_token}"}
                        )
                        
                        if store_response.status_code == 200:
                            shared_count += 1
            
            # Track shared memory session
            if session_id not in self.shared_memory_sessions:
                self.shared_memory_sessions[session_id] = set()
            self.shared_memory_sessions[session_id].add(source_agent_id)
            self.shared_memory_sessions[session_id].update(target_agent_ids)
            
            return {
                "status": "success",
                "memories_shared": shared_count,
                "source_agent": source_agent_id,
                "target_agents": target_agent_ids
            }
            
        except Exception as e:
            logger.error(f"Error sharing memory: {e}")
            return {"status": "error", "message": str(e)}
    
    async def synchronize_session_memories(self, session_id: str) -> Dict[str, Any]:
        """Synchronize memories across all agents in a session"""
        if session_id not in self.shared_memory_sessions:
            return {"status": "no_session", "message": "No active memory sharing session"}
        
        agent_ids = list(self.shared_memory_sessions[session_id])
        sync_results = []
        
        # Get important recent memories from each agent
        for agent_id in agent_ids:
            try:
                memory_query = {
                    "query": "important recent interactions",
                    "agent_id": agent_id,
                    "session_id": session_id,
                    "memory_types": ["working", "episodic"],
                    "limit": 5,
                    "similarity_threshold": 0.7
                }
                
                response = await self.http_client.post(
                    f"{self.memory_system_url}/memory/query",
                    json=memory_query,
                    headers={"Authorization": f"Bearer {self.access_token}"}
                )
                
                if response.status_code == 200:
                    memories = response.json()
                    high_importance_memories = []
                    
                    for memory_type, memory_list in memories.items():
                        for memory in memory_list:
                            if memory.get("importance", 0) > 0.7:  # Only share high-importance memories
                                high_importance_memories.append(memory)
                    
                    if high_importance_memories:
                        # Share with other agents in the session
                        other_agents = [aid for aid in agent_ids if aid != agent_id]
                        if other_agents:
                            share_result = await self.share_memory_between_agents(
                                agent_id, other_agents, session_id, ["working", "episodic"]
                            )
                            sync_results.append({
                                "agent_id": agent_id,
                                "result": share_result
                            })
                            
            except Exception as e:
                logger.error(f"Error synchronizing memories for agent {agent_id}: {e}")
                sync_results.append({
                    "agent_id": agent_id,
                    "result": {"status": "error", "message": str(e)}
                })
        
        return {
            "status": "completed",
            "session_id": session_id,
            "synchronization_results": sync_results
        }

# ==============================================
# Main Orchestrator Service
# ==============================================

class AgentOrchestrator:
    def __init__(self):
        self.agent_manager = AgentManager()
        self.memory_coordinator = MemoryCoordinator(
            config.memory_system_url,
            os.getenv("MCP_ACCESS_TOKEN", "secure-memory-token")
        )
        self.http_client = httpx.AsyncClient(timeout=30.0)
        self.openai_client = None
        self.anthropic_client = None
        
        # Initialize AI clients
        if config.openai_api_key:
            self.openai_client = openai.AsyncOpenAI(api_key=config.openai_api_key)
        if config.anthropic_api_key:
            self.anthropic_client = Anthropic(api_key=config.anthropic_api_key)
    
    async def initialize(self):
        """Initialize the orchestrator"""
        # Start background tasks
        asyncio.create_task(self.memory_sync_task())
        asyncio.create_task(self.agent_health_monitor())
        asyncio.create_task(self.task_dispatcher())
        
        logger.info("Agent Orchestrator initialized")
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.http_client.aclose()
        await self.memory_coordinator.http_client.aclose()
    
    # ==============================================
    # Task Management
    # ==============================================
    
    async def submit_task(self, task: Task) -> str:
        """Submit a new task for processing"""
        # Add to active tasks
        self.agent_manager.active_tasks[task.task_id] = task
        
        # Try to assign immediately
        best_agent = self.agent_manager.find_best_agent_for_task(task)
        if best_agent:
            await self.assign_task_to_agent(task.task_id, best_agent.agent_id)
        else:
            # Add to queue
            self.agent_manager.task_queue.append(task)
            logger.info(f"Task {task.task_id} queued - no available agents")
        
        return task.task_id
    
    async def assign_task_to_agent(self, task_id: str, agent_id: str) -> bool:
        """Assign a task to a specific agent"""
        if task_id not in self.agent_manager.active_tasks:
            return False
        
        task = self.agent_manager.active_tasks[task_id]
        agent = self.agent_manager.get_agent(agent_id)
        
        if not agent or agent.status not in [AgentStatus.IDLE, AgentStatus.ACTIVE]:
            return False
        
        # Update task
        task.assigned_agent_id = agent_id
        task.assigned_at = datetime.utcnow()
        task.status = TaskStatus.ASSIGNED
        
        # Update agent status
        self.agent_manager.update_agent_status(agent_id, AgentStatus.BUSY)
        
        # Notify agent (if connected via WebSocket)
        if agent_id in self.agent_manager.agent_connections:
            await self.notify_agent_of_task(agent_id, task)
        
        logger.info(f"Assigned task {task_id} to agent {agent_id}")
        return True
    
    async def notify_agent_of_task(self, agent_id: str, task: Task):
        """Notify agent of new task via WebSocket"""
        websocket = self.agent_manager.agent_connections.get(agent_id)
        if websocket:
            try:
                notification = {
                    "type": "task_assignment",
                    "task": task.dict(),
                    "timestamp": datetime.utcnow().isoformat()
                }
                await websocket.send_text(json.dumps(notification))
            except Exception as e:
                logger.error(f"Error notifying agent {agent_id}: {e}")
    
    async def complete_task(self, task_id: str, result: Dict[str, Any], agent_id: str) -> bool:
        """Mark task as completed"""
        if task_id not in self.agent_manager.active_tasks:
            return False
        
        task = self.agent_manager.active_tasks[task_id]
        if task.assigned_agent_id != agent_id:
            return False
        
        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.utcnow()
        task.result = result
        
        # Free up the agent
        self.agent_manager.update_agent_status(agent_id, AgentStatus.ACTIVE)
        
        # Store task completion in memory
        await self.store_task_completion_memory(task)
        
        logger.info(f"Task {task_id} completed by agent {agent_id}")
        return True
    
    async def store_task_completion_memory(self, task: Task):
        """Store task completion in memory system"""
        try:
            memory_content = f"Completed task '{task.title}': {task.description}. Result: {str(task.result)[:200]}"
            
            memory_data = {
                "content": memory_content,
                "memory_type": "episodic",
                "agent_id": task.assigned_agent_id,
                "session_id": task.session_id,
                "importance": 0.7,  # Task completions are fairly important
                "tags": ["task_completion", task.title.lower().replace(" ", "_")],
                "metadata": {
                    "task_id": task.task_id,
                    "task_type": task.title,
                    "completion_time": task.completed_at.isoformat() if task.completed_at else None,
                    "duration_minutes": ((task.completed_at - task.assigned_at).total_seconds() / 60) if task.completed_at and task.assigned_at else None
                }
            }
            
            await self.http_client.post(
                f"{config.memory_system_url}/memory/store",
                json=memory_data,
                headers={"Authorization": f"Bearer {os.getenv('MCP_ACCESS_TOKEN', 'secure-memory-token')}"}
            )
            
        except Exception as e:
            logger.error(f"Error storing task completion memory: {e}")
    
    # ==============================================
    # Background Tasks
    # ==============================================
    
    async def memory_sync_task(self):
        """Background task for memory synchronization"""
        while True:
            try:
                if config.working_memory_sharing:
                    # Synchronize memories for active sessions
                    for session_id in list(self.memory_coordinator.shared_memory_sessions.keys()):
                        await self.memory_coordinator.synchronize_session_memories(session_id)
                
                await asyncio.sleep(config.agent_memory_sync_interval)
                
            except Exception as e:
                logger.error(f"Error in memory sync task: {e}")
                await asyncio.sleep(60)  # Wait a minute before retrying
    
    async def agent_health_monitor(self):
        """Monitor agent health and status"""
        while True:
            try:
                current_time = datetime.utcnow()
                
                for agent_id, agent in self.agent_manager.agents.items():
                    # Check for inactive agents
                    time_since_active = current_time - agent.last_active
                    if time_since_active > timedelta(minutes=10):
                        if agent.status != AgentStatus.OFFLINE:
                            self.agent_manager.update_agent_status(agent_id, AgentStatus.OFFLINE)
                            logger.warning(f"Agent {agent_id} marked as offline due to inactivity")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in agent health monitor: {e}")
                await asyncio.sleep(60)
    
    async def task_dispatcher(self):
        """Background task dispatcher"""
        while True:
            try:
                if self.agent_manager.task_queue:
                    task = self.agent_manager.task_queue[0]
                    best_agent = self.agent_manager.find_best_agent_for_task(task)
                    
                    if best_agent:
                        # Remove from queue and assign
                        self.agent_manager.task_queue.pop(0)
                        await self.assign_task_to_agent(task.task_id, best_agent.agent_id)
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in task dispatcher: {e}")
                await asyncio.sleep(30)

# ==============================================
# FastAPI Application
# ==============================================

app = FastAPI(
    title="Agent Memory Orchestrator",
    description="Multi-Agent Coordination with Shared Memory Management",
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

# Global orchestrator instance
orchestrator = AgentOrchestrator()

@app.on_event("startup")
async def startup_event():
    await orchestrator.initialize()

@app.on_event("shutdown")
async def shutdown_event():
    await orchestrator.cleanup()

# ==============================================
# API Endpoints
# ==============================================

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "service": "agent-orchestrator",
        "active_agents": len(orchestrator.agent_manager.agents),
        "active_tasks": len(orchestrator.agent_manager.active_tasks),
        "queued_tasks": len(orchestrator.agent_manager.task_queue)
    }

@app.post("/agents/register")
async def register_agent(
    agent: AgentProfile,
    credentials: HTTPAuthorizationCredentials = Depends(verify_token)
):
    """Register a new agent"""
    agent_id = orchestrator.agent_manager.register_agent(agent)
    return {"status": "success", "agent_id": agent_id}

@app.get("/agents")
async def list_agents(
    credentials: HTTPAuthorizationCredentials = Depends(verify_token)
):
    """List all registered agents"""
    return {"agents": list(orchestrator.agent_manager.agents.values())}

@app.get("/agents/{agent_id}")
async def get_agent(
    agent_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(verify_token)
):
    """Get specific agent details"""
    agent = orchestrator.agent_manager.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent

@app.post("/tasks/submit")
async def submit_task(
    task: Task,
    credentials: HTTPAuthorizationCredentials = Depends(verify_token)
):
    """Submit a new task"""
    task_id = await orchestrator.submit_task(task)
    return {"status": "success", "task_id": task_id}

@app.get("/tasks")
async def list_tasks(
    status: Optional[TaskStatus] = None,
    agent_id: Optional[str] = None,
    credentials: HTTPAuthorizationCredentials = Depends(verify_token)
):
    """List tasks with optional filtering"""
    tasks = list(orchestrator.agent_manager.active_tasks.values())
    
    if status:
        tasks = [t for t in tasks if t.status == status]
    
    if agent_id:
        tasks = [t for t in tasks if t.assigned_agent_id == agent_id]
    
    return {"tasks": tasks}

@app.put("/tasks/{task_id}/complete")
async def complete_task(
    task_id: str,
    result: Dict[str, Any],
    agent_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(verify_token)
):
    """Mark task as completed"""
    success = await orchestrator.complete_task(task_id, result, agent_id)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to complete task")
    return {"status": "success"}

@app.post("/memory/share")
async def share_memory(
    source_agent_id: str,
    target_agent_ids: List[str],
    session_id: str,
    memory_types: List[str] = None,
    credentials: HTTPAuthorizationCredentials = Depends(verify_token)
):
    """Share memories between agents"""
    result = await orchestrator.memory_coordinator.share_memory_between_agents(
        source_agent_id, target_agent_ids, session_id, memory_types
    )
    return result

@app.post("/collaboration/request")
async def request_collaboration(
    request: CollaborationRequest,
    credentials: HTTPAuthorizationCredentials = Depends(verify_token)
):
    """Request collaboration between agents"""
    # This is a simplified implementation
    # In practice, you'd implement more sophisticated collaboration logic
    return {
        "status": "collaboration_initiated",
        "collaboration_id": str(uuid4()),
        "participating_agents": [request.initiator_agent_id] + request.target_agent_ids
    }

# WebSocket endpoint for real-time agent communication
@app.websocket("/agents/{agent_id}/connect")
async def agent_websocket(websocket: WebSocket, agent_id: str):
    """WebSocket connection for real-time agent communication"""
    await websocket.accept()
    
    # Register WebSocket connection
    orchestrator.agent_manager.agent_connections[agent_id] = websocket
    orchestrator.agent_manager.update_agent_status(agent_id, AgentStatus.ACTIVE)
    
    try:
        while True:
            # Receive messages from agent
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            if message.get("type") == "status_update":
                status = AgentStatus(message.get("status", "active"))
                orchestrator.agent_manager.update_agent_status(agent_id, status)
            
            elif message.get("type") == "task_update":
                task_id = message.get("task_id")
                if task_id in orchestrator.agent_manager.active_tasks:
                    task = orchestrator.agent_manager.active_tasks[task_id]
                    if message.get("status") == "completed":
                        await orchestrator.complete_task(
                            task_id, 
                            message.get("result", {}), 
                            agent_id
                        )
            
    except WebSocketDisconnect:
        # Clean up on disconnect
        orchestrator.agent_manager.agent_connections.pop(agent_id, None)
        orchestrator.agent_manager.update_agent_status(agent_id, AgentStatus.OFFLINE)
        logger.info(f"Agent {agent_id} disconnected")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

