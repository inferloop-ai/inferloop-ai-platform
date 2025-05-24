-- database/migrations/003_memory_analytics.sql
-- Memory analytics enhancements migration

-- Migration metadata  
INSERT INTO schema_migrations (version, description, applied_at) VALUES
('003', 'Memory analytics enhancements', NOW')
ON CONFLICT (version) DO NOTHING;

-- Create analytics views
CREATE OR REPLACE VIEW memory_analytics_summary AS
SELECT 
    em.agent_id,
    COUNT(*) as total_memories,
    AVG(em.importance) as avg_importance,
    AVG(em.emotional_valence) as avg_emotional_valence,
    COUNT(DISTINCT em.session_id) as unique_sessions,
    MAX(em.created_at) as last_memory_created,
    SUM(em.access_count) as total_accesses,
    COUNT(*) FILTER (WHERE em.consolidated = true) as consolidated_memories
FROM episodic_memories em
GROUP BY em.agent_id;

CREATE OR REPLACE VIEW agent_performance_metrics AS
SELECT 
    amp.agent_id,
    ams.total_memories,
    ams.avg_importance,
    amp.learning_rate,
    COUNT(mcl.id) as consolidation_count,
    AVG(mcl.memories_consolidated::float / NULLIF(mcl.memories_processed, 0)) as consolidation_efficiency
FROM agent_memory_profiles amp
LEFT JOIN memory_analytics_summary ams ON amp.agent_id = ams.agent_id
LEFT JOIN memory_consolidation_log mcl ON amp.agent_id = mcl.agent_id 
    AND mcl.status = 'completed'
    AND mcl.started_at > (NOW() - INTERVAL '30 days')
GROUP BY amp.agent_id, ams.total_memories, ams.avg_importance, amp.learning_rate;

-- Create schema migrations tracking table if not exists
CREATE TABLE IF NOT EXISTS schema_migrations (
    version VARCHAR(10) PRIMARY KEY,
    description TEXT,
    applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Functions for analytics
CREATE OR REPLACE FUNCTION get_memory_growth_rate(p_agent_id VARCHAR(255), p_days INTEGER DEFAULT 7)
RETURNS FLOAT AS $$
DECLARE
    current_count INTEGER;
    previous_count INTEGER;
    growth_rate FLOAT;
BEGIN
    -- Get current memory count
    SELECT COUNT(*) INTO current_count
    FROM episodic_memories 
    WHERE agent_id = p_agent_id 
    AND created_at >= (NOW() - INTERVAL '1 day' * p_days);
    
    -- Get previous period count
    SELECT COUNT(*) INTO previous_count
    FROM episodic_memories 
    WHERE agent_id = p_agent_id 
    AND created_at >= (NOW() - INTERVAL '1 day' * (p_days * 2))
    AND created_at < (NOW() - INTERVAL '1 day' * p_days);
    
    -- Calculate growth rate
    IF previous_count > 0 THEN
        growth_rate := ((current_count - previous_count)::float / previous_count) * 100;
    ELSE
        growth_rate := 0;
    END IF;
    
    RETURN growth_rate;
END;
$$ LANGUAGE plpgsql;

-- Update migration version
UPDATE episodic_memories SET migration_version = 3 WHERE migration_version < 3;
UPDATE semantic_concepts SET migration_version = 3 WHERE migration_version < 3;