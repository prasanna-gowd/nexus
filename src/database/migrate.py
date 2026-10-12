"""
Data migration script: DuckDB to PostgreSQL + pgvector.

Transfers jobs, normalized skills, and relationships from DuckDB into PostgreSQL.
Generates 384-dimensional dense embeddings for job description chunks
using BAAI/bge-small-en-v1.5 (or deterministic normalized vector fallback).
"""
import os
import sys
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import duckdb
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

DUCKDB_PATH = os.getenv("DUCKDB_PATH", "data/nexus_jobs.duckdb")
PG_HOST = os.getenv("POSTGRES_HOST", "localhost")
PG_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
PG_USER = os.getenv("POSTGRES_USER", "nexus")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD", "nexus")
PG_DB = os.getenv("POSTGRES_DB", "nexus_db")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")


def get_pg_connection():
    """Establish connection to PostgreSQL."""
    return psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        user=PG_USER,
        password=PG_PASSWORD,
        dbname=PG_DB
    )


def initialize_pg_schema(pg_conn):
    """Execute schema.sql to ensure tables, extensions, and HNSW indexes exist."""
    schema_path = Path(__file__).parent / "schema.sql"
    if schema_path.exists():
        with open(schema_path, "r", encoding="utf-8") as f:
            sql = f.read()
        with pg_conn.cursor() as cur:
            cur.execute(sql)
        pg_conn.commit()
        logger.info("PostgreSQL schema validated/initialized successfully.")


class VectorEmbeddingGenerator:
    """Handles text embeddings generation with BAAI/bge-small-en-v1.5 or deterministic fallback."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME, dim: int = 384):
        self.dim = dim
        self.model = None
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model '{model_name}'...")
            self.model = SentenceTransformer(model_name)
            logger.info(f"Loaded '{model_name}' successfully.")
        except Exception as e:
            logger.warning(f"Could not load SentenceTransformer ({e}). Using deterministic 384-d vector generator.")

    def encode(self, texts: List[str]) -> List[List[float]]:
        if self.model is not None:
            vectors = self.model.encode(texts, normalize_embeddings=True)
            return vectors.tolist()

        # Deterministic 384-dimensional pseudo-embedding fallback
        import hashlib
        import math
        fallback_vectors = []
        for text in texts:
            vec = [0.0] * self.dim
            words = text.lower().split()
            if not words:
                words = ["unknown"]
            for i, word in enumerate(words):
                h = int(hashlib.md5(f"{word}_{i % 384}".encode()).hexdigest(), 16)
                idx = h % self.dim
                val = ((h % 1000) / 500.0) - 1.0
                vec[idx] += val
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            vec = [round(x / norm, 6) for x in vec]
            fallback_vectors.append(vec)
        return fallback_vectors


def parse_date_safely(date_str: Optional[str]) -> Optional[datetime]:
    if not date_str:
        return datetime.utcnow()
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(date_str[:19], fmt)
        except (ValueError, TypeError):
            continue
    return datetime.utcnow()


def run_migration(duckdb_path: str = DUCKDB_PATH) -> dict:
    """
    Executes full migration from DuckDB to PostgreSQL.
    """
    stats = {
        "jobs_migrated": 0,
        "skills_migrated": 0,
        "job_skills_migrated": 0,
        "embeddings_created": 0,
        "errors": []
    }

    if not Path(duckdb_path).exists():
        msg = f"DuckDB database not found at {duckdb_path}. Run Phase 1 & 2 first."
        logger.error(msg)
        stats["errors"].append(msg)
        return stats

    logger.info("Connecting to DuckDB and PostgreSQL...")
    duck_conn = duckdb.connect(duckdb_path)
    try:
        pg_conn = get_pg_connection()
    except Exception as e:
        msg = f"Failed to connect to PostgreSQL: {e}"
        logger.error(msg)
        stats["errors"].append(msg)
        duck_conn.close()
        return stats

    try:
        initialize_pg_schema(pg_conn)

        # 1. Migrate Skills
        logger.info("Migrating Skills from DuckDB to PostgreSQL...")
        duck_skills = duck_conn.execute("""
            SELECT skill_id, canonical_name, category, esco_uri, is_emerging
            FROM skills_taxonomy
        """).fetchall()

        skill_id_map = {} # canonical_name.lower() -> postgres UUID
        with pg_conn.cursor() as cur:
            for s_id, name, cat, esco_uri, is_em in duck_skills:
                cur.execute("""
                    INSERT INTO skills (standard_name, category, esco_uri, is_emerging)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (standard_name) DO UPDATE 
                    SET category = EXCLUDED.category,
                        esco_uri = COALESCE(NULLIF(EXCLUDED.esco_uri, ''), skills.esco_uri)
                    RETURNING id;
                """, (name, cat or "tool", esco_uri or "", bool(is_em)))
                pg_skill_id = cur.fetchone()[0]
                skill_id_map[name.lower()] = pg_skill_id
                stats["skills_migrated"] += 1
        pg_conn.commit()
        logger.info(f"Skills migrated: {stats['skills_migrated']}")

        # 2. Migrate Jobs
        logger.info("Migrating Jobs from DuckDB to PostgreSQL...")
        duck_jobs = duck_conn.execute("""
            SELECT job_id, source, title, company, location, remote_status, date_posted, description, url
            FROM raw_jobs
        """).fetchall()

        embed_gen = VectorEmbeddingGenerator()
        job_id_map = {} # duckdb job_id -> postgres UUID
        chunks_to_embed = [] # list of (postgres_job_id, chunk_text)

        with pg_conn.cursor() as cur:
            for j_id, src, title, comp, loc, rem, dt_str, desc, url in duck_jobs:
                dt_val = parse_date_safely(dt_str)
                cur.execute("""
                    INSERT INTO jobs (source_job_id, source, title, company, location, remote_status, date_posted, clean_text, url)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (source_job_id) DO UPDATE
                    SET title = EXCLUDED.title,
                        company = EXCLUDED.company,
                        clean_text = EXCLUDED.clean_text
                    RETURNING id;
                """, (j_id, src, title, comp or "Unknown", loc or "", rem or "unknown", dt_val, desc, url or ""))
                pg_job_id = cur.fetchone()[0]
                job_id_map[j_id] = pg_job_id
                stats["jobs_migrated"] += 1

                # Prepare summary text chunk for vector embedding
                chunk_text = f"Role: {title} at {comp}. Description: {desc[:1000]}"
                chunks_to_embed.append((pg_job_id, chunk_text))
        pg_conn.commit()
        logger.info(f"Jobs migrated: {stats['jobs_migrated']}")

        # 3. Migrate Job_Skills relationships
        logger.info("Migrating Job-Skill relationships...")
        duck_links = duck_conn.execute("""
            SELECT js.job_id, st.canonical_name, js.confidence_score
            FROM job_skills js
            JOIN skills_taxonomy st ON js.skill_id = st.skill_id
        """).fetchall()

        with pg_conn.cursor() as cur:
            for duck_jid, canonical_name, conf in duck_links:
                pg_jid = job_id_map.get(duck_jid)
                pg_sid = skill_id_map.get(canonical_name.lower())
                if pg_jid and pg_sid:
                    cur.execute("""
                        INSERT INTO job_skills (job_id, skill_id, confidence_score, is_required)
                        VALUES (%s, %s, %s, TRUE)
                        ON CONFLICT (job_id, skill_id) DO NOTHING;
                    """, (pg_jid, pg_sid, float(conf or 1.0)))
                    stats["job_skills_migrated"] += 1
        pg_conn.commit()
        logger.info(f"Job-Skill links migrated: {stats['job_skills_migrated']}")

        # 4. Generate & Insert Embeddings
        logger.info("Generating and storing vector embeddings...")
        if chunks_to_embed:
            texts = [c[1] for c in chunks_to_embed]
            vectors = embed_gen.encode(texts)
            with pg_conn.cursor() as cur:
                cur.execute("DELETE FROM job_embeddings;") # Refresh embeddings
                insert_data = [
                    (chunks_to_embed[i][0], chunks_to_embed[i][1], vectors[i])
                    for i in range(len(chunks_to_embed))
                ]
                execute_values(
                    cur,
                    "INSERT INTO job_embeddings (job_id, chunk_text, embedding) VALUES %s",
                    insert_data,
                    template="(%s, %s, %s::vector)"
                )
                stats["embeddings_created"] = len(insert_data)
            pg_conn.commit()
            logger.info(f"Embeddings stored: {stats['embeddings_created']}")

    except Exception as e:
        logger.error(f"Migration error: {e}", exc_info=True)
        stats["errors"].append(str(e))
        pg_conn.rollback()
    finally:
        duck_conn.close()
        pg_conn.close()

    logger.info("Migration finished. Summary: %s", stats)
    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_migration()
