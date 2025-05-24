-- database/migrations/001_initial_schema.sql
-- Initial schema migration

-- Migration metadata
INSERT INTO schema_migrations (version, description, applied_at) VALUES
('001', 'Initial memory system schema', NOW())
ON CONFLICT (version) DO NOTHING;

-- Core tables (already defined in memory-schema.sql)
-- This file serves as a placeholder for migration tracking

-- Add any schema changes specific to version 001
ALTER TABLE episodic_memories ADD COLUMN IF NOT EXISTS migration_version INTEGER DEFAULT 1;
ALTER TABLE semantic_concepts ADD COLUMN IF NOT EXISTS migration_version INTEGER DEFAULT 1;
