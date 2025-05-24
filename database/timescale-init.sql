-- database/timescale-init.sql
-- TimescaleDB initialization for time-series memory analytics

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create time-series tables for memory analytics
CREATE TABLE IF NOT EXISTS memory_time_series (
    time TIMESTAMPTZ NOT NULL,
    agent_id VARCHAR(255) NOT NULL,
    session_id VARCHAR(255),
    memory_type VARCHAR(50) NOT NULL,
    operation_type VARCHAR(50) NOT NULL, -- 'store', 'retrieve', 'consolidate', 'decay'
    importance_score FLOAT,
    response_time_ms INTEGER,
    success BOOLEAN DEFAULT true,
    metadata JSONB DEFAULT '{}'
);

-- Convert to hypertable (TimescaleDB specific)
SELECT create_hypertable('memory_time_series', 'time', if_not_exists => TRUE);

-- Create continuous aggregates for common queries
CREATE MATERIALIZED VIEW IF NOT EXISTS memory_hourly_stats
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 hour', time) AS hour,
    agent_id,
    memory_type,
    COUNT(*) as operation_count,
    AVG(importance_score) as avg_importance,
    AVG(response_time_ms) as avg_response_time,
    SUM(CASE WHEN success THEN 1 ELSE 0 END)::float / COUNT(*) as success_rate
FROM memory_time_series
GROUP BY hour, agent_id, memory_type
WITH NO DATA;

CREATE MATERIALIZED VIEW IF NOT EXISTS memory_daily_stats  
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 day', time) AS day,
    agent_id,
    COUNT(*) as total_operations,
    COUNT(DISTINCT session_id) as unique_sessions,
    AVG(importance_score) as avg_importance,
    AVG(response_time_ms) as avg_response_time
FROM memory_time_series
GROUP BY day, agent_id
WITH NO DATA;

-- Add data retention policy (keep detailed data for 30 days, aggregated for 1 year)
SELECT add_retention_policy('memory_time_series', INTERVAL '30 days');

-- Create indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_memory_ts_agent_time ON memory_time_series (agent_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_memory_ts_type_time ON memory_time_series (memory_type, time DESC);
CREATE INDEX IF NOT EXISTS idx_memory_ts_operation ON memory_time_series (operation_type, time DESC);

-- Performance monitoring table
CREATE TABLE IF NOT EXISTS system_performance_metrics (
    time TIMESTAMPTZ NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    component VARCHAR(50), -- 'mcp_server', 'postgres', 'redis', 'chroma', 'neo4j'
    instance_id VARCHAR(100),
    metadata JSONB DEFAULT '{}'
);

SELECT create_hypertable('system_performance_metrics', 'time', if_not_exists => TRUE);

-- Add retention policy for performance metrics
SELECT add_retention_policy('system_performance_metrics', INTERVAL '90 days');

-- Create sample data insertion function
CREATE OR REPLACE FUNCTION insert_memory_operation_metric(
    p_agent_id VARCHAR(255),
    p_session_id VARCHAR(255),
    p_memory_type VARCHAR(50),
    p_operation_type VARCHAR(50),
    p_importance_score FLOAT DEFAULT NULL,
    p_response_time_ms INTEGER DEFAULT NULL,
    p_success BOOLEAN DEFAULT true,
    p_metadata JSONB DEFAULT '{}'
) RETURNS VOID AS $$
BEGIN
    INSERT INTO memory_time_series (
        time, agent_id, session_id, memory_type, operation_type,
        importance_score, response_time_ms, success, metadata
    ) VALUES (
        NOW(), p_agent_id, p_session_id, p_memory_type, p_operation_type,
        p_importance_score, p_response_time_ms, p_success, p_metadata
    );
END;
$$ LANGUAGE plpgsql;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO memory_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO memory_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO memory_user;

