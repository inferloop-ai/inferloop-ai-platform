# tests/performance/load_test.py
"""
Load testing for memory system
"""

import asyncio
import httpx
import time
import random
import string
from concurrent.futures import ThreadPoolExecutor
import statistics

class MemoryLoadTester:
    
    def __init__(self, base_url="http://localhost:8080", num_agents=10, num_sessions=5):
        self.base_url = base_url
        self.num_agents = num_agents
        self.num_sessions = num_sessions
        self.headers = {"Authorization": "Bearer secure-memory-token"}
        self.results = {
            "store_times": [],
            "query_times": [],
            "errors": 0,
            "total_operations": 0
        }
    
    def generate_random_content(self, length=100):
        """Generate random content for testing"""
        return ''.join(random.choices(string.ascii_letters + string.digits + ' ', k=length))
    
    async def store_memory_operation(self, client, agent_id, session_id):
        """Perform a single memory store operation"""
        memory_data = {
            "content": self.generate_random_content(),
            "memory_type": random.choice(["working", "episodic"]),
            "agent_id": agent_id,
            "session_id": session_id,
            "importance": random.uniform(0.1, 1.0),
            "tags": [f"load_test_{random.randint(1, 100)}"]
        }
        
        start_time = time.time()
        try:
            response = await client.post("/memory/store", json=memory_data, headers=self.headers)
            if response.status_code == 200:
                elapsed = (time.time() - start_time) * 1000  # Convert to ms
                self.results["store_times"].append(elapsed)
            else:
                self.results["errors"] += 1
        except Exception:
            self.results["errors"] += 1
        
        self.results["total_operations"] += 1
    
    async def query_memory_operation(self, client, agent_id, session_id):
        """Perform a single memory query operation"""
        query_data = {
            "query": f"load test query {random.randint(1, 1000)}",
            "agent_id": agent_id,
            "session_id": session_id,
            "memory_types": ["working", "episodic"],
            "limit": random.randint(1, 10)
        }
        
        start_time = time.time()
        try:
            response = await client.post("/memory/query", json=query_data, headers=self.headers)
            if response.status_code == 200:
                elapsed = (time.time() - start_time) * 1000  # Convert to ms
                self.results["query_times"].append(elapsed)
            else:
                self.results["errors"] += 1
        except Exception:
            self.results["errors"] += 1
        
        self.results["total_operations"] += 1
    
    async def run_load_test(self, duration_seconds=60, operations_per_second=10):
        """Run load test for specified duration"""
        print(f"Starting load test for {duration_seconds} seconds at {operations_per_second} ops/sec")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            start_time = time.time()
            
            while (time.time() - start_time) < duration_seconds:
                # Create batch of operations
                tasks = []
                
                for _ in range(operations_per_second):
                    agent_id = f"load-test-agent-{random.randint(1, self.num_agents)}"
                    session_id = f"load-test-session-{random.randint(1, self.num_sessions)}"
                    
                    # Randomly choose operation type
                    if random.random() < 0.7:  # 70% store operations
                        task = self.store_memory_operation(client, agent_id, session_id)
                    else:  # 30% query operations
                        task = self.query_memory_operation(client, agent_id, session_id)
                    
                    tasks.append(task)
                
                # Execute batch
                await asyncio.gather(*tasks, return_exceptions=True)
                
                # Wait for next second
                await asyncio.sleep(1.0)
        
        self.print_results()
    
    def print_results(self):
        """Print load test results"""
        print("\n" + "="*50)
        print("LOAD TEST RESULTS")
        print("="*50)
        
        print(f"Total Operations: {self.results['total_operations']}")
        print(f"Errors: {self.results['errors']}")
        print(f"Error Rate: {(self.results['errors'] / self.results['total_operations']) * 100:.2f}%")
        
        if self.results["store_times"]:
            print(f"\nStore Operations:")
            print(f"  Count: {len(self.results['store_times'])}")
            print(f"  Avg Response Time: {statistics.mean(self.results['store_times']):.2f}ms")
            print(f"  95th Percentile: {statistics.quantiles(self.results['store_times'], n=20)[18]:.2f}ms")
            print(f"  Max Response Time: {max(self.results['store_times']):.2f}ms")
        
        if self.results["query_times"]:
            print(f"\nQuery Operations:")
            print(f"  Count: {len(self.results['query_times'])}")
            print(f"  Avg Response Time: {statistics.mean(self.results['query_times']):.2f}ms")
            print(f"  95th Percentile: {statistics.quantiles(self.results['query_times'], n=20)[18]:.2f}ms")
            print(f"  Max Response Time: {max(self.results['query_times']):.2f}ms")

# Run load test
if __name__ == "__main__":
    tester = MemoryLoadTester()
    asyncio.run(tester.run_load_test(duration_seconds=120, operations_per_second=20))
