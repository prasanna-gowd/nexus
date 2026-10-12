"""
DuckDB database connection and schema initialization for NEXUS.

Handles creating the raw_jobs table with deduplication support
and the NLP-related tables (skills_taxonomy, job_skills).
"""
import os
import hashlib
import logging
import duckdb
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = os.getenv("DUCKDB_PATH", "data/nexus_jobs.duckdb")


def get_connection(db_path: str = None) -> duckdb.DuckDBPyConnection:
    """Get a DuckDB connection, creating the data directory if needed."""
    db_path = db_path or DEFAULT_DB_PATH
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(db_path)
    logger.info(f"Connected to DuckDB at {db_path}")
    return conn


def init_schema(conn: duckdb.DuckDBPyConnection) -> None:
    """Initialize all DuckDB tables for Phases 1 and 2."""
    # Phase 1: Raw jobs table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw_jobs (
            job_id VARCHAR PRIMARY KEY,
            source VARCHAR NOT NULL,
            title VARCHAR NOT NULL,
            company VARCHAR DEFAULT 'Unknown',
            location VARCHAR DEFAULT '',
            remote_status VARCHAR DEFAULT '',
            date_posted VARCHAR DEFAULT '',
            description TEXT NOT NULL,
            url VARCHAR DEFAULT '',
            dedup_hash VARCHAR NOT NULL,
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Phase 2: Skills taxonomy
    conn.execute("""
        CREATE TABLE IF NOT EXISTS skills_taxonomy (
            skill_id VARCHAR PRIMARY KEY,
            canonical_name VARCHAR NOT NULL UNIQUE,
            category VARCHAR NOT NULL,
            esco_uri VARCHAR DEFAULT '',
            confidence FLOAT DEFAULT 0.0,
            is_emerging BOOLEAN DEFAULT FALSE
        )
    """)

    # Phase 2: Job-to-skill relationships
    conn.execute("""
        CREATE TABLE IF NOT EXISTS job_skills (
            job_id VARCHAR NOT NULL,
            skill_id VARCHAR NOT NULL,
            confidence_score FLOAT DEFAULT 0.0,
            PRIMARY KEY (job_id, skill_id)
        )
    """)

    # Dedup index
    conn.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_dedup_hash 
        ON raw_jobs(dedup_hash)
    """)

    logger.info("DuckDB schema initialized successfully")


def compute_dedup_hash(title: str, company: str, description: str) -> str:
    """Generate a deterministic deduplication hash from job content."""
    content = f"{title.strip().lower()}|{company.strip().lower()}|{description.strip().lower()[:500]}"
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def insert_jobs(conn: duckdb.DuckDBPyConnection, jobs: list[dict]) -> dict:
    """
    Insert jobs into raw_jobs with deduplication.
    
    Returns dict with counts: inserted, skipped, failed.
    """
    stats = {"inserted": 0, "skipped": 0, "failed": 0}

    for job in jobs:
        try:
            dedup_hash = compute_dedup_hash(
                job.get("title", ""),
                job.get("company", ""),
                job.get("description", "")
            )

            # Check if already exists
            existing = conn.execute(
                "SELECT 1 FROM raw_jobs WHERE dedup_hash = ?", [dedup_hash]
            ).fetchone()

            if existing:
                stats["skipped"] += 1
                continue

            conn.execute("""
                INSERT INTO raw_jobs (job_id, source, title, company, location,
                    remote_status, date_posted, description, url, dedup_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                job.get("job_id", dedup_hash),
                job.get("source", "unknown"),
                job.get("title", "Untitled"),
                job.get("company", "Unknown"),
                job.get("location", ""),
                job.get("remote_status", ""),
                job.get("date_posted", ""),
                job.get("description", ""),
                job.get("url", ""),
                dedup_hash
            ])
            stats["inserted"] += 1

        except Exception as e:
            logger.warning(f"Failed to insert job '{job.get('title', '?')}': {e}")
            stats["failed"] += 1

    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    conn = get_connection()
    init_schema(conn)
    print("DuckDB schema initialized successfully.")
    tables = conn.execute("SHOW TABLES").fetchall()
    print(f"Tables: {[t[0] for t in tables]}")
    conn.close()
