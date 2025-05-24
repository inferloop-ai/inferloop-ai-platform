# tests/performance/memory_benchmark.py
"""
Memory system performance benchmarking
"""

import asyncio
import time
import statistics
import httpx
import json
from datetime import datetime

class MemoryBenchmark:
    
    def __init__(self, base_url="http://localhost:8080"):
        self.base_url = base_url
        self.headers = {"Authorization": "Bearer secure-memory-token"}
        self.benchmarks = {}
    
    async def benchmark_memory_storage(self, num_memories=1000):
        """Benchmark memory storage performance"""
        print(f"Benchmarking memory storage with {num_memories} memories...")
        
        times = []
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            for i in range(num_memories):
                memory_data = {
                    "content": f"Benchmark memory {i} with detailed content for performance testing",
                    "memory_type": "working",
                    "agent_id": "benchmark-agent",
                    "session_id": "benchmark-session",
                    "importance": 0.5,
                    "tags": [f"benchmark_{i}", "performance_test"]
                }
                
                start_time = time.time()
                response = await client.post("/memory/store", json=memory_data, headers=self.headers)
                elapsed = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    times.append(elapsed)
                
                if i % 100 == 0:
                    print(f"  Processed: {i}/{num_memories}")
        
        self.benchmarks["storage"] = {
            "count": len(times),
            "avg_time_ms": statistics.mean(times),
            "median_time_ms": statistics.median(times),
            "p95_time_ms": statistics.quantiles(times, n=20)[18] if len(times) >= 20 else max(times),
            "max_time_ms": max(times),
            "min_time_ms": min(times),
            "throughput_ops_per_sec": len(times) / (sum(times) / 1000)
        }
        
        print(f"Storage benchmark completed: {self.benchmarks['storage']['avg_time_ms']:.2f}ms avg")
    
    async def benchmark_memory_query(self, num_queries=500):
        """Benchmark memory query performance"""
        print(f"Benchmarking memory queries with {num_queries} queries...")
        
        times = []
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            for i in range(num_queries):
                query_data = {
                    "query": f"benchmark memory {i % 100}",
                    "agent_id": "benchmark-agent",
                    "session_id": "benchmark-session",
                    "memory_types": ["working", "episodic"],
                    "limit": 10
                }
                
                start_time = time.time()
                response = await client.post("/memory/query", json=query_data, headers=self.headers)
                elapsed = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    times.append(elapsed)
                
                if i % 50 == 0:
                    print(f"  Processed: {i}/{num_queries}")
        
        self.benchmarks["query"] = {
            "count": len(times),
            "avg_time_ms": statistics.mean(times),
            "median_time_ms": statistics.median(times),
            "p95_time_ms": statistics.quantiles(times, n=20)[18] if len(times) >= 20 else max(times),
            "max_time_ms": max(times),
            "min_time_ms": min(times),
            "throughput_ops_per_sec": len(times) / (sum(times) / 1000)
        }
        
        print(f"Query benchmark completed: {self.benchmarks['query']['avg_time_ms']:.2f}ms avg")
    
    async def benchmark_consolidation(self):
        """Benchmark memory consolidation performance"""
        print("Benchmarking memory consolidation...")
        
        async with httpx.AsyncClient(timeout=300.0) as client:  # 5 minute timeout
            consolidation_data = {
                "agent_id": "benchmark-agent",
                "session_id": "benchmark-session",
                "force": True
            }
            
            start_time = time.time()
            response = await client.post("/memory/consolidate", json=consolidation_data, headers=self.headers)
            elapsed = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                result = response.json()
                self.benchmarks["consolidation"] = {
                    "time_ms": elapsed,
                    "status": result.get("status"),
                    "consolidated": result.get("consolidated", {})
                }
                
                print(f"Consolidation benchmark completed: {elapsed:.2f}ms")
            else:
                print(f"Consolidation benchmark failed: {response.status_code}")
    
    async def benchmark_concurrent_operations(self, concurrency=10, operations_per_client=100):
        """Benchmark concurrent operations"""
        print(f"Benchmarking concurrent operations: {concurrency} clients, {operations_per_client} ops each")
        
        async def client_operations(client_id):
            times = []
            errors = 0
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                for i in range(operations_per_client):
                    memory_data = {
                        "content": f"Concurrent test memory {client_id}-{i}",
                        "memory_type": "working",
                        "agent_id": f"concurrent-agent-{client_id}",
                        "session_id": f"concurrent-session-{client_id}",
                        "importance": 0.5
                    }
                    
                    start_time = time.time()
                    try:
                        response = await client.post("/memory/store", json=memory_data, headers=self.headers)
                        elapsed = (time.time() - start_time) * 1000
                        
                        if response.status_code == 200:
                            times.append(elapsed)
                        else:
                            errors += 1
                    except Exception:
                        errors += 1
            
            return times, errors
        
        # Run concurrent clients
        start_time = time.time()
        tasks = [client_operations(i) for i in range(concurrency)]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Aggregate results
        all_times = []
        total_errors = 0
        
        for times, errors in results:
            all_times.extend(times)
            total_errors += errors
        
        self.benchmarks["concurrent"] = {
            "concurrency": concurrency,
            "operations_per_client": operations_per_client,
            "total_operations": len(all_times),
            "total_errors": total_errors,
            "total_time_seconds": total_time,
            "avg_time_ms": statistics.mean(all_times) if all_times else 0,
            "p95_time_ms": statistics.quantiles(all_times, n=20)[18] if len(all_times) >= 20 else (max(all_times) if all_times else 0),
            "throughput_ops_per_sec": len(all_times) / total_time if total_time > 0 else 0,
            "error_rate": total_errors / (len(all_times) + total_errors) if (len(all_times) + total_errors) > 0 else 0
        }
        
        print(f"Concurrent benchmark completed: {self.benchmarks['concurrent']['throughput_ops_per_sec']:.2f} ops/sec")
    
    def print_benchmark_report(self):
        """Print comprehensive benchmark report"""
        print("\n" + "="*70)
        print("MEMORY SYSTEM PERFORMANCE BENCHMARK REPORT")
        print("="*70)
        print(f"Timestamp: {datetime.now().isoformat()}")
        print()
        
        for benchmark_name, results in self.benchmarks.items():
            print(f"{benchmark_name.upper()} BENCHMARK:")
            for key, value in results.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")
            print()
        
        # Performance summary
        if "storage" in self.benchmarks and "query" in self.benchmarks:
            print("PERFORMANCE SUMMARY:")
            print(f"  Storage Throughput: {self.benchmarks['storage']['throughput_ops_per_sec']:.2f} ops/sec")
            print(f"  Query Throughput: {self.benchmarks['query']['throughput_ops_per_sec']:.2f} ops/sec")
            
            if "concurrent" in self.benchmarks:
                print(f"  Concurrent Throughput: {self.benchmarks['concurrent']['throughput_ops_per_sec']:.2f} ops/sec")
                print(f"  Concurrent Error Rate: {self.benchmarks['concurrent']['error_rate']*100:.2f}%")
    
    async def run_full_benchmark(self):
        """Run complete benchmark suite"""
        print("Starting comprehensive memory system benchmark...")
        
        await self.benchmark_memory_storage(1000)
        await self.benchmark_memory_query(500)
        await self.benchmark_consolidation()
        await self.benchmark_concurrent_operations(10, 50)
        
        self.print_benchmark_report()

# Run benchmark
if __name__ == "__main__":
    benchmark = MemoryBenchmark()
    asyncio.run(benchmark.run_full_benchmark())
