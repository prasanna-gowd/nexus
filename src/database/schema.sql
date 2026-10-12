-- PostgreSQL 16 + pgvector Schema for NEXUS Market Intelligence

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

-- Relational Tables
CREATE TABLE IF NOT EXISTS jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_job_id VARCHAR(255) UNIQUE,
    source VARCHAR(50) NOT NULL DEFAULT 'unknown',
    title VARCHAR(255) NOT NULL,
    company VARCHAR(255) NOT NULL DEFAULT 'Unknown',
    clean_text TEXT,
    date_posted TIMESTAMP,
    remote_status VARCHAR(50) DEFAULT 'unknown',
    location VARCHAR(255) DEFAULT '',
    url TEXT DEFAULT '',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS skills (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    standard_name VARCHAR(255) NOT NULL UNIQUE,
    category VARCHAR(100) NOT NULL DEFAULT 'tool',
    esco_uri TEXT DEFAULT '',
    is_emerging BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS job_skills (
    job_id UUID REFERENCES jobs(id) ON DELETE CASCADE,
    skill_id UUID REFERENCES skills(id) ON DELETE CASCADE,
    confidence_score FLOAT DEFAULT 1.0,
    is_required BOOLEAN DEFAULT TRUE,
    PRIMARY KEY (job_id, skill_id)
);

-- Vector Table for Multi-Agent Semantic Retrieval (RAG)
CREATE TABLE IF NOT EXISTS job_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID REFERENCES jobs(id) ON DELETE CASCADE,
    chunk_text TEXT NOT NULL,
    embedding vector(384), -- BAAI/bge-small-en-v1.5 produces 384-dimensional dense vectors
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_jobs_title ON jobs(title);
CREATE INDEX IF NOT EXISTS idx_jobs_company ON jobs(company);
CREATE INDEX IF NOT EXISTS idx_skills_standard_name ON skills(standard_name);
CREATE INDEX IF NOT EXISTS idx_job_skills_skill_id ON job_skills(skill_id);
CREATE INDEX IF NOT EXISTS idx_job_skills_job_id ON job_skills(job_id);

-- HNSW Vector Index for approximate nearest neighbor (ANN) search via cosine distance
CREATE INDEX IF NOT EXISTS idx_job_embeddings_hnsw 
ON job_embeddings USING hnsw (embedding vector_cosine_ops);
