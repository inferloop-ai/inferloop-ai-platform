-- database/init-memory.sql
-- PostgreSQL initialization script for memory system

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS btree_gin;

-- Create database user if not exists
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'memory_user') THEN
        CREATE ROLE memory_user WITH LOGIN PASSWORD 'memorypass123';
    END IF;
END
$$;

-- Grant necessary permissions
GRANT ALL PRIVILEGES ON DATABASE memories TO memory_user;
GRANT ALL ON SCHEMA public TO memory_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO memory_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO memory_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO memory_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO memory_user;

-- =====================================================
-- database/memory-schema.sql
-- Memory system database schema
-- =====================================================

-- Episodic Memory Table
CREATE TABLE IF NOT EXISTS episodic_memories (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid()::text,
    content TEXT NOT NULL,
    agent_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    importance FLOAT NOT NULL DEFAULT 0.5 CHECK (importance >= 0 AND importance <= 1),
    emotional_valence FLOAT NOT NULL DEFAULT 0.0 CHECK (emotional_valence >= -1 AND emotional_valence <= 1),
    tags TEXT[] DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    embedding vector(384), -- Using sentence-transformers dimension
    decay_factor FLOAT DEFAULT 1.0,
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    consolidated BOOLEAN DEFAULT FALSE,
    source_memory_id VARCHAR(36),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Semantic Memory Concepts Table
CREATE TABLE IF NOT EXISTS semantic_concepts (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid()::text,
    concept_name VARCHAR(500) NOT NULL,
    concept_type VARCHAR(100) NOT NULL,
    definition TEXT,
    agent_id VARCHAR(255) NOT NULL,
    embedding vector(384),
    strength FLOAT DEFAULT 1.0 CHECK (strength >= 0 AND strength <= 1),
    connections_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(concept_name, agent_id)
);

-- Semantic Memory Relationships Table
CREATE TABLE IF NOT EXISTS semantic_relationships (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid()::text,
    source_concept_id VARCHAR(36) REFERENCES semantic_concepts(id) ON DELETE CASCADE,
    target_concept_id VARCHAR(36) REFERENCES semantic_concepts(id) ON DELETE CASCADE,
    relationship_type VARCHAR(100) NOT NULL,
    strength FLOAT DEFAULT 1.0 CHECK (strength >= 0 AND strength <= 1),
    agent_id VARCHAR(255) NOT NULL,
    evidence TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source_concept_id, target_concept_id, relationship_type)
);

-- Memory Consolidation Log
CREATE TABLE IF NOT EXISTS memory_consolidation_log (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid()::text,
    agent_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255) NOT NULL,
    consolidation_type VARCHAR(50) NOT NULL, -- 'automatic', 'manual', 'scheduled'
    memories_processed INTEGER NOT NULL DEFAULT 0,
    memories_consolidated INTEGER NOT NULL DEFAULT 0,
    consolidation_strategy JSONB DEFAULT '{}',
    performance_metrics JSONB DEFAULT '{}',
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'running', 'completed', 'failed'
    error_message TEXT
);

-- Agent Memory Profiles
CREATE TABLE IF NOT EXISTS agent_memory_profiles (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid()::text,
    agent_id VARCHAR(255) UNIQUE NOT NULL,
    memory_preferences JSONB DEFAULT '{}',
    consolidation_frequency INTEGER DEFAULT 21600, -- 6 hours in seconds
    working_memory_capacity INTEGER DEFAULT 7,
    importance_threshold FLOAT DEFAULT 0.3,
    emotional_sensitivity FLOAT DEFAULT 0.5,
    learning_rate FLOAT DEFAULT 0.01,
    forgetting_curve_params JSONB DEFAULT '{"decay_rate": 0.99, "retention_strength": 0.5}',
    personality_traits JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Memory Access Patterns (for analytics)
CREATE TABLE IF NOT EXISTS memory_access_patterns (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid()::text,
    agent_id VARCHAR(255) NOT NULL,
    memory_id VARCHAR(36) NOT NULL,
    memory_type VARCHAR(50) NOT NULL, -- 'working', 'episodic', 'semantic', 'procedural'
    access_type VARCHAR(50) NOT NULL, -- 'read', 'write', 'update', 'delete'
    query_context TEXT,
    similarity_score FLOAT,
    retrieval_time_ms INTEGER,
    session_id VARCHAR(255),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

-- Memory Quality Metrics
CREATE TABLE IF NOT EXISTS memory_quality_metrics (
    id VARCHAR(36) PRIMARY KEY DEFAULT gen_random_uuid()::text,
    agent_id VARCHAR(255) NOT NULL,
    metric_type VARCHAR(100) NOT NULL, -- 'coherence', 'completeness', 'accuracy', 'relevance'
    metric_value FLOAT NOT NULL,
    memory_type VARCHAR(50) NOT NULL,
    calculation_method VARCHAR(100),
    sample_size INTEGER,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

-- =====================================================
-- Indexes for Performance
-- =====================================================

-- Episodic Memory Indexes
CREATE INDEX IF NOT EXISTS idx_episodic_agent_session ON episodic_memories(agent_id, session_id);
CREATE INDEX IF NOT EXISTS idx_episodic_timestamp ON episodic_memories(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_episodic_importance ON episodic_memories(importance DESC);
CREATE INDEX IF NOT EXISTS idx_episodic_embedding ON episodic_memories USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_episodic_tags ON episodic_memories USING GIN(tags);
CREATE INDEX IF NOT EXISTS idx_episodic_metadata ON episodic_memories USING GIN(metadata);
CREATE INDEX IF NOT EXISTS idx_episodic_consolidated ON episodic_memories(consolidated);

-- Semantic Concepts Indexes
CREATE INDEX IF NOT EXISTS idx_semantic_concepts_agent ON semantic_concepts(agent_id);
CREATE INDEX IF NOT EXISTS idx_semantic_concepts_type ON semantic_concepts(concept_type);
CREATE INDEX IF NOT EXISTS idx_semantic_concepts_embedding ON semantic_concepts USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50);
CREATE INDEX IF NOT EXISTS idx_semantic_concepts_name ON semantic_concepts USING gin(concept_name gin_trgm_ops);

-- Semantic Relationships Indexes
CREATE INDEX IF NOT EXISTS idx_semantic_rel_source ON semantic_relationships(source_concept_id);
CREATE INDEX IF NOT EXISTS idx_semantic_rel_target ON semantic_relationships(target_concept_id);
CREATE INDEX IF NOT EXISTS idx_semantic_rel_agent ON semantic_relationships(agent_id);
CREATE INDEX IF NOT EXISTS idx_semantic_rel_type ON semantic_relationships(relationship_type);

-- Consolidation Log Indexes
CREATE INDEX IF NOT EXISTS idx_consolidation_agent_session ON memory_consolidation_log(agent_id, session_id);
CREATE INDEX IF NOT EXISTS idx_consolidation_status ON memory_consolidation_log(status);
CREATE INDEX IF NOT EXISTS idx_consolidation_started ON memory_consolidation_log(started_at DESC);

-- Agent Profiles Indexes
CREATE INDEX IF NOT EXISTS idx_agent_profiles_agent_id ON agent_memory_profiles(agent_id);

-- Access Patterns Indexes
CREATE INDEX IF NOT EXISTS idx_access_patterns_agent ON memory_access_patterns(agent_id);
CREATE INDEX IF NOT EXISTS idx_access_patterns_memory ON memory_access_patterns(memory_id);
CREATE INDEX IF NOT EXISTS idx_access_patterns_timestamp ON memory_access_patterns(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_access_patterns_type ON memory_access_patterns(memory_type, access_type);

-- Quality Metrics Indexes
CREATE INDEX IF NOT EXISTS idx_quality_metrics_agent ON memory_quality_metrics(agent_id);
CREATE INDEX IF NOT EXISTS idx_quality_metrics_type ON memory_quality_metrics(metric_type);
CREATE INDEX IF NOT EXISTS idx_quality_metrics_timestamp ON memory_quality_metrics(timestamp DESC);

-- =====================================================
-- Triggers for Automatic Updates
-- =====================================================

-- Update timestamp trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply trigger to tables with updated_at column
CREATE TRIGGER update_episodic_memories_updated_at 
    BEFORE UPDATE ON episodic_memories 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_semantic_concepts_updated_at 
    BEFORE UPDATE ON semantic_concepts 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_semantic_relationships_updated_at 
    BEFORE UPDATE ON semantic_relationships 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_agent_memory_profiles_updated_at 
    BEFORE UPDATE ON agent_memory_profiles 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =====================================================
-- Functions for Memory Operations
-- =====================================================

-- Function to calculate memory importance based on various factors
CREATE OR REPLACE FUNCTION calculate_memory_importance(
    p_content TEXT,
    p_emotional_valence FLOAT,
    p_access_count INTEGER,
    p_age_hours FLOAT
) RETURNS FLOAT AS $$
DECLARE
    base_importance FLOAT := 0.5;
    content_factor FLOAT;
    emotional_factor FLOAT;
    recency_factor FLOAT;
    access_factor FLOAT;
    final_importance FLOAT;
BEGIN
    -- Content length factor (longer content might be more important)
    content_factor := LEAST(LENGTH(p_content) / 1000.0, 1.0);
    
    -- Emotional valence factor (more emotional content is more memorable)
    emotional_factor := ABS(p_emotional_valence);
    
    -- Recency factor (newer memories start with higher importance)
    recency_factor := EXP(-p_age_hours / 168.0); -- Decay over a week
    
    -- Access frequency factor
    access_factor := LEAST(p_access_count / 10.0, 1.0);
    
    -- Combine factors
    final_importance := (
        base_importance * 0.3 +
        content_factor * 0.2 +
        emotional_factor * 0.2 +
        recency_factor * 0.2 +
        access_factor * 0.1
    );
    
    RETURN GREATEST(LEAST(final_importance, 1.0), 0.0);
END;
$$ LANGUAGE plpgsql;

-- Function to find similar memories using vector similarity
CREATE OR REPLACE FUNCTION find_similar_memories(
    p_embedding vector(384),
    p_agent_id VARCHAR(255),
    p_threshold FLOAT DEFAULT 0.8,
    p_limit INTEGER DEFAULT 10
) RETURNS TABLE(
    memory_id VARCHAR(36),
    similarity_score FLOAT,
    content TEXT,
    importance FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        em.id,
        1 - (em.embedding <=> p_embedding) as similarity_score,
        em.content,
        em.importance
    FROM episodic_memories em
    WHERE em.agent_id = p_agent_id
      AND em.embedding IS NOT NULL
      AND (1 - (em.embedding <=> p_embedding)) >= p_threshold
    ORDER BY em.embedding <=> p_embedding
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Function to update memory decay over time
CREATE OR REPLACE FUNCTION update_memory_decay() RETURNS VOID AS $$
BEGIN
    UPDATE episodic_memories 
    SET 
        decay_factor = decay_factor * 0.99,
        importance = importance * 0.995
    WHERE 
        last_accessed < CURRENT_TIMESTAMP - INTERVAL '24 hours'
        AND decay_factor > 0.1;
        
    -- Remove very old, low-importance memories
    DELETE FROM episodic_memories 
    WHERE 
        importance < 0.05 
        AND created_at < CURRENT_TIMESTAMP - INTERVAL '90 days';
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- Views for Common Queries
-- =====================================================

-- Active memories view (non-decayed, recent, or frequently accessed)
CREATE OR REPLACE VIEW active_memories AS
SELECT 
    em.*,
    EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - em.created_at)) / 3600 as age_hours,
    CASE 
        WHEN em.access_count > 5 THEN 'frequently_accessed'
        WHEN em.created_at > CURRENT_TIMESTAMP - INTERVAL '24 hours' THEN 'recent'
        WHEN em.importance > 0.7 THEN 'high_importance'
        ELSE 'standard'
    END as memory_category
FROM episodic_memories em
WHERE 
    em.decay_factor > 0.3
    AND (
        em.access_count > 2 
        OR em.created_at > CURRENT_TIMESTAMP - INTERVAL '7 days'
        OR em.importance > 0.5
    );

-- Memory statistics view per agent
CREATE OR REPLACE VIEW agent_memory_stats AS
SELECT 
    agent_id,
    COUNT(*) as total_memories,
    AVG(importance) as avg_importance,
    AVG(emotional_valence) as avg_emotional_valence,
    COUNT(*) FILTER (WHERE consolidated = true) as consolidated_memories,
    COUNT(*) FILTER (WHERE created_at > CURRENT_TIMESTAMP - INTERVAL '24 hours') as recent_memories,
    COUNT(*) FILTER (WHERE access_count > 5) as frequently_accessed,
    MAX(created_at) as last_memory_created,
    MAX(last_accessed) as last_memory_accessed
FROM episodic_memories
GROUP BY agent_id;

-- =====================================================
-- Sample Data for Testing
-- =====================================================

-- Insert sample agent profile
INSERT INTO agent_memory_profiles (agent_id, memory_preferences) 
VALUES ('test-agent-001', '{"consolidation_strategy": "importance_based", "emotional_weighting": 0.7}')
ON CONFLICT (agent_id) DO NOTHING;

-- Insert sample semantic concepts
INSERT INTO semantic_concepts (concept_name, concept_type, definition, agent_id)
VALUES 
    ('artificial_intelligence', 'technology', 'Computer systems able to perform tasks that typically require human intelligence', 'test-agent-001'),
    ('machine_learning', 'technology', 'Method of data analysis that automates analytical model building', 'test-agent-001'),
    ('neural_networks', 'technology', 'Computing systems inspired by biological neural networks', 'test-agent-001')
ON CONFLICT (concept_name, agent_id) DO NOTHING;

-- Insert sample semantic relationships
INSERT INTO semantic_relationships (source_concept_id, target_concept_id, relationship_type, agent_id)
SELECT 
    sc1.id, sc2.id, 'is_type_of', 'test-agent-001'
FROM semantic_concepts sc1, semantic_concepts sc2
WHERE sc1.concept_name = 'machine_learning' 
  AND sc2.concept_name = 'artificial_intelligence'
  AND sc1.agent_id = 'test-agent-001' 
  AND sc2.agent_id = 'test-agent-001'
ON CONFLICT (source_concept_id, target_concept_id, relationship_type) DO NOTHING;

-- Grant final permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO memory_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO memory_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO memory_user;