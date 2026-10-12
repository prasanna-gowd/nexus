# NEXUS Phase 3: Hybrid Storage & Trend Analytics Engine Implementation Plan

## 1. Objective
Migrate the normalized, NLP-extracted job data from the local DuckDB database into a production-grade PostgreSQL system. This phase calculates critical market trends and prepares the semantic vectors required for the Multi-Agent ecosystem.

## 2. Infrastructure Setup (PostgreSQL + pgvector)
Using PostgreSQL with the `pgvector` extension provides a unified platform for both transactional SQL queries and high-performance similarity search, without the overhead of managing a separate vector database.

*   **Database Deployment:** Use the official Docker image to spin up a local instance.
    ```bash
    docker run --name pgvector-container -e POSTGRES_USER=nexus -e POSTGRES_PASSWORD=nexus -e POSTGRES_DB=nexus_db -p 5432:5432 -d pgvector/pgvector:pg16
    ```
*   **Python Dependencies:** 
    ```bash
    pip install psycopg2-binary langchain-postgres sentence-transformers pandas sqlalchemy
    ```

## 3. Database Schema

Execute this SQL to structure your hybrid storage:

```sql
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

-- Core Relational Tables
CREATE TABLE jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title VARCHAR NOT NULL,
    company VARCHAR NOT NULL,
    clean_text TEXT,
    date_posted TIMESTAMP NOT NULL,
    remote_status VARCHAR
);

CREATE TABLE skills (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    standard_name VARCHAR NOT NULL,
    category VARCHAR NOT NULL
);

CREATE TABLE job_skills (
    job_id UUID REFERENCES jobs(id) ON DELETE CASCADE,
    skill_id UUID REFERENCES skills(id) ON DELETE CASCADE,
    is_required BOOLEAN DEFAULT TRUE,
    PRIMARY KEY (job_id, skill_id)
);

-- Vector Table for Multi-Agent RAG
CREATE TABLE job_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID REFERENCES jobs(id) ON DELETE CASCADE,
    chunk_text TEXT NOT NULL,
    embedding vector(384) -- Dimension size for BAAI/bge-small-en-v1.5
);

-- HNSW Index for rapid approximate nearest neighbor (ANN) search
CREATE INDEX idx_job_embeddings ON job_embeddings USING hnsw (embedding vector_cosine_ops);