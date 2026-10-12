"""
Database and Semantic Vector Retrieval manager for NEXUS API.
"""
import os
import logging
from typing import List, Dict, Any, Optional
import duckdb
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

DUCKDB_PATH = os.getenv("DUCKDB_PATH", "data/nexus_jobs.duckdb")
PG_HOST = os.getenv("POSTGRES_HOST", "localhost")
PG_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
PG_USER = os.getenv("POSTGRES_USER", "nexus")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD", "nexus")
PG_DB = os.getenv("POSTGRES_DB", "nexus_db")


def check_duckdb_health() -> bool:
    try:
        conn = duckdb.connect(DUCKDB_PATH)
        conn.execute("SELECT 1")
        conn.close()
        return True
    except Exception:
        return False


def check_postgres_health() -> bool:
    try:
        import psycopg2
        conn = psycopg2.connect(
            host=PG_HOST, port=PG_PORT, user=PG_USER, password=PG_PASSWORD, dbname=PG_DB,
            connect_timeout=2
        )
        conn.close()
        return True
    except Exception:
        return False


def retrieve_semantic_job_context(skill_name: str, limit: int = 3) -> Dict[str, Any]:
    """
    Retrieves real job description context snippets for a skill.
    Uses pgvector ANN search if PostgreSQL is available.
    Falls back to DuckDB keyword search or curated industry templates.
    """
    skill_clean = skill_name.strip()

    # 1. Try PostgreSQL + pgvector
    if check_postgres_health():
        try:
            import psycopg2
            from src.database.migrate import VectorEmbeddingGenerator

            embed_gen = VectorEmbeddingGenerator()
            query_vec = embed_gen.encode([f"Technical requirements and production usage of {skill_clean}"])[0]

            conn = psycopg2.connect(
                host=PG_HOST, port=PG_PORT, user=PG_USER, password=PG_PASSWORD, dbname=PG_DB
            )
            with conn.cursor() as cur:
                # Cosine distance operator (<=>)
                cur.execute("""
                    SELECT j.title, j.company, je.chunk_text, (1 - (je.embedding <=> %s::vector)) as similarity
                    FROM job_embeddings je
                    JOIN jobs j ON je.job_id = j.id
                    ORDER BY je.embedding <=> %s::vector ASC
                    LIMIT %s;
                """, (query_vec, query_vec, limit))
                rows = cur.fetchall()
            conn.close()

            if rows:
                contexts = []
                for title, company, chunk, score in rows:
                    contexts.append({
                        "job_title": title,
                        "company": company,
                        "snippet": chunk[:400] + "..." if len(chunk) > 400 else chunk,
                        "score": round(float(score), 3) if score is not None else None
                    })
                return {
                    "skill_name": skill_clean,
                    "contexts": contexts,
                    "is_fallback": False
                }
        except Exception as e:
            logger.warning(f"Vector search failed, falling back to DuckDB: {e}")

    # 2. Try DuckDB keyword search
    if check_duckdb_health():
        try:
            conn = duckdb.connect(DUCKDB_PATH)
            pattern = f"%{skill_clean.lower()}%"
            rows = conn.execute("""
                SELECT title, company, description 
                FROM raw_jobs 
                WHERE LOWER(description) LIKE ?
                LIMIT ?
            """, [pattern, limit]).fetchall()
            conn.close()

            if rows:
                contexts = []
                for title, company, desc in rows:
                    # Find snippet around the skill
                    lower_desc = desc.lower()
                    idx = lower_desc.find(skill_clean.lower())
                    start = max(0, idx - 100)
                    end = min(len(desc), idx + 300)
                    snippet = desc[start:end].strip()
                    contexts.append({
                        "job_title": title,
                        "company": company,
                        "snippet": f"...{snippet}..." if start > 0 else f"{snippet}...",
                        "score": 0.85
                    })
                return {
                    "skill_name": skill_clean,
                    "contexts": contexts,
                    "is_fallback": False
                }
        except Exception as e:
            logger.warning(f"DuckDB search failed: {e}")

    # 3. Honest fallback templates clearly flagged as fallback
    curated_fallbacks = {
        "Docker": "Engineers utilize Docker to package microservices with container runtimes, ensuring consistent deployment across staging and production environments.",
        "Python": "Production backend systems rely on Python for high-performance data processing pipelines, asynchronous microservices, and machine learning model serving.",
        "SQL": "Data infrastructure teams write optimized SQL queries, partition large analytics tables, and tune execution plans for relational OLTP and OLAP workloads.",
        "React": "Frontend architectures leverage React with modern state management, component decomposition, and responsive UI interfaces integrated with RESTful APIs.",
        "Kubernetes": "DevOps teams configure Kubernetes Helm charts and manifest files for multi-cluster container orchestration, rolling deployments, and automated horizontal pod scaling.",
        "Git": "Distributed engineering teams manage branch strategies, pull request code reviews, and GitOps CI/CD deployment automation.",
    }
    fallback_text = curated_fallbacks.get(
        skill_clean,
        f"Modern production software environments demand hands-on proficiency in {skill_clean} to maintain reliability, scalability, and technical excellence."
    )

    return {
        "skill_name": skill_clean,
        "contexts": [{
            "job_title": "Software Engineer",
            "company": "Industry Benchmark",
            "snippet": fallback_text,
            "score": 0.70
        }],
        "is_fallback": True
    }
