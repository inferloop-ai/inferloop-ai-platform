-- database/migrations/002_add_indexes.sql  
-- Performance optimization indexes migration

-- Migration metadata
INSERT INTO schema_migrations (version, description, applied_at) VALUES
('002', 'Performance optimization indexes', NOW())
ON CONFLICT (version) DO NOTHING;

-- Additional performance indexes
CREATE INDEX IF NOT EXISTS idx_episodic_session_time ON episodic_memories(session_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_episodic_importance_time ON episodic_memories(importance DESC, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_episodic_agent_importance ON episodic_memories(agent_id, importance DESC);

-- Partial indexes for common queries
CREATE INDEX IF NOT EXISTS idx_episodic_high_importance 
ON episodic_memories(agent_id, created_at DESC) 
WHERE importance > 0.7;

CREATE INDEX IF NOT EXISTS idx_episodic_recent_active
ON episodic_memories(agent_id, importance DESC)
WHERE created_at > (NOW() - INTERVAL '7 days');

-- Compound indexes for complex queries
CREATE INDEX IF NOT EXISTS idx_memory_access_agent_type_time
ON memory_access_patterns(agent_id, memory_type, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_consolidation_agent_status_time
ON memory_consolidation_log(agent_id, status, started_at DESC);

-- Update migration version
UPDATE episodic_memories SET migration_version = 2 WHERE migration_version < 2;
UPDATE semantic_concepts SET migration_version = 2 WHERE migration_version < 2;
