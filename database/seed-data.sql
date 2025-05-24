-- database/seed-data.sql
-- Sample data for testing and development

-- Insert sample agent memory profiles
INSERT INTO agent_memory_profiles (agent_id, memory_preferences, personality_traits) VALUES
('research-agent-001', 
 '{"consolidation_frequency": 10800, "importance_threshold": 0.4, "learning_rate": 0.15}',
 '{"curiosity": 0.9, "analytical": 0.85, "methodical": 0.8}'),
('support-agent-001',
 '{"consolidation_frequency": 14400, "importance_threshold": 0.5, "emotional_sensitivity": 0.9}',
 '{"empathy": 0.95, "patience": 0.9, "helpfulness": 0.9}'),
('creative-agent-001',
 '{"consolidation_frequency": 28800, "importance_threshold": 0.2, "novelty_preference": 0.8}',
 '{"creativity": 0.95, "openness": 0.9, "experimentation": 0.85}')
ON CONFLICT (agent_id) DO NOTHING;

-- Insert sample episodic memories
INSERT INTO episodic_memories (
    content, agent_id, session_id, importance, emotional_valence, tags, metadata, embedding
) VALUES
('User asked about machine learning fundamentals and showed particular interest in neural networks',
 'research-agent-001', 'session-001', 0.8, 0.3, 
 ARRAY['education', 'machine_learning', 'neural_networks'],
 '{"topic": "education", "user_interest_level": "high"}',
 ARRAY[0.1, 0.2, 0.3, 0.4]::float[] -- Simplified embedding
),
('Successfully helped user resolve login issue by guiding through password reset process',
 'support-agent-001', 'session-002', 0.7, 0.6,
 ARRAY['support', 'login_issue', 'resolution'],
 '{"issue_type": "authentication", "resolution_time_minutes": 5, "user_satisfaction": "high"}',
 ARRAY[0.2, 0.3, 0.4, 0.5]::float[]
),
('Created an engaging story about space exploration that incorporated user\'s love for adventure',
 'creative-agent-001', 'session-003', 0.6, 0.8,
 ARRAY['creative_writing', 'space', 'adventure'],
 '{"genre": "science_fiction", "word_count": 850, "user_feedback": "loved_it"}',
 ARRAY[0.3, 0.4, 0.5, 0.6]::float[]
);

-- Insert sample semantic concepts
INSERT INTO semantic_concepts (concept_name, concept_type, definition, agent_id, embedding, strength) VALUES
('neural_network', 'technology', 'A computing system inspired by biological neural networks', 'research-agent-001', ARRAY[0.5, 0.6, 0.7, 0.8]::float[], 0.9),
('customer_satisfaction', 'metric', 'Measure of how products and services meet customer expectations', 'support-agent-001', ARRAY[0.4, 0.5, 0.6, 0.7]::float[], 0.8),
('narrative_structure', 'literary', 'The framework of a story including plot, character development, and theme', 'creative-agent-001', ARRAY[0.6, 0.7, 0.8, 0.9]::float[], 0.85)
ON CONFLICT (concept_name, agent_id) DO NOTHING;

-- Insert sample semantic relationships
INSERT INTO semantic_relationships (source_concept_id, target_concept_id, relationship_type, strength, agent_id, evidence)
SELECT 
    sc1.id, sc2.id, 'enables', 0.8, 'research-agent-001', 'Machine learning techniques enable neural network training'
FROM semantic_concepts sc1, semantic_concepts sc2
WHERE sc1.concept_name = 'neural_network' AND sc2.concept_name = 'machine_learning' 
  AND sc1.agent_id = 'research-agent-001' AND sc2.agent_id = 'research-agent-001'
ON CONFLICT (source_concept_id, target_concept_id, relationship_type) DO NOTHING;

-- Insert sample consolidation log entries
INSERT INTO memory_consolidation_log (
    agent_id, session_id, consolidation_type, memories_processed, 
    memories_consolidated, performance_metrics, started_at, completed_at, status
) VALUES
('research-agent-001', 'session-001', 'automatic', 25, 18,
 '{"processing_time_seconds": 45.2, "success_rate": 0.92, "memory_types": {"episodic": 12, "semantic": 6}}',
 NOW() - INTERVAL '2 hours', NOW() - INTERVAL '2 hours' + INTERVAL '45 seconds', 'completed'),
('support-agent-001', 'session-002', 'scheduled', 15, 12,
 '{"processing_time_seconds": 32.1, "success_rate": 0.95, "memory_types": {"episodic": 8, "semantic": 4}}',
 NOW() - INTERVAL '4 hours', NOW() - INTERVAL '4 hours' + INTERVAL '32 seconds', 'completed');

-- Insert sample memory access patterns
INSERT INTO memory_access_patterns (
    agent_id, memory_id, memory_type, access_type, similarity_score, retrieval_time_ms, session_id
) VALUES
('research-agent-001', (SELECT id FROM episodic_memories WHERE agent_id = 'research-agent-001' LIMIT 1), 
 'episodic', 'read', 0.85, 120, 'session-001'),
('support-agent-001', (SELECT id FROM episodic_memories WHERE agent_id = 'support-agent-001' LIMIT 1),
 'episodic', 'read', 0.92, 95, 'session-002');

-- Insert sample quality metrics
INSERT INTO memory_quality_metrics (
    agent_id, metric_type, metric_value, memory_type, calculation_method, sample_size
) VALUES
('research-agent-001', 'coherence', 0.87, 'episodic', 'semantic_similarity_clustering', 50),
('research-agent-001', 'completeness', 0.92, 'semantic', 'concept_coverage_analysis', 30),
('support-agent-001', 'relevance', 0.89, 'episodic', 'user_satisfaction_correlation', 40),
('creative-agent-001', 'creativity_score', 0.78, 'episodic', 'novelty_measurement', 25);
