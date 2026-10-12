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


def get_duckdb_connection() -> duckdb.DuckDBPyConnection:
    """Get a DuckDB connection, creating parent directory if necessary."""
    from pathlib import Path
    Path(DUCKDB_PATH).parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(DUCKDB_PATH)


def init_session_tables():
    """Create session and learning event tables in DuckDB if they don't exist."""
    try:
        con = get_duckdb_connection()
        # Create sequence if not exists
        try:
            con.execute("CREATE SEQUENCE IF NOT EXISTS learning_event_seq START 1")
        except:
            pass
        con.execute("""
            CREATE TABLE IF NOT EXISTS learner_sessions (
                student_id TEXT PRIMARY KEY,
                state_json TEXT NOT NULL,
                target_role TEXT DEFAULT '',
                skills_count INTEGER DEFAULT 0,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        con.execute("""
            CREATE TABLE IF NOT EXISTS learning_events (
                id INTEGER DEFAULT (nextval('learning_event_seq')),
                student_id TEXT NOT NULL,
                skill_name TEXT NOT NULL,
                mastery_score REAL NOT NULL,
                event_type TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        con.close()
    except Exception as e:
        logger.warning(f"Failed to init session tables: {e}")


def save_session(student_id: str, state: dict):
    """Persist learner session state to DuckDB."""
    import json
    try:
        con = get_duckdb_connection()
        state_json = json.dumps(state, default=str)
        target_role = state.get('target_role', '')
        skills_count = len(state.get('verified_skills', []))
        con.execute("""
            INSERT OR REPLACE INTO learner_sessions (student_id, state_json, target_role, skills_count, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, [student_id, state_json, target_role, skills_count])
        con.close()
    except Exception as e:
        logger.warning(f"Failed to save session: {e}")


def load_session(student_id: str) -> dict:
    """Load learner session state from DuckDB. Returns empty dict if not found."""
    import json
    try:
        con = get_duckdb_connection()
        result = con.execute(
            "SELECT state_json FROM learner_sessions WHERE student_id = ?",
            [student_id]
        ).fetchone()
        con.close()
        if result:
            return json.loads(result[0])
    except Exception as e:
        logger.warning(f"Failed to load session: {e}")
    return {}


def record_learning_event(student_id: str, skill_name: str, mastery_score: float, event_type: str):
    """Record a learning event for progress analytics."""
    try:
        con = get_duckdb_connection()
        con.execute("""
            INSERT INTO learning_events (student_id, skill_name, mastery_score, event_type, created_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, [student_id, skill_name, mastery_score, event_type])
        con.close()
    except Exception as e:
        logger.warning(f"Failed to record learning event: {e}")


def get_learning_history(student_id: str) -> list:
    """Retrieve all learning events for a student, ordered by time."""
    try:
        con = get_duckdb_connection()
        results = con.execute("""
            SELECT skill_name, mastery_score, event_type, created_at
            FROM learning_events
            WHERE student_id = ?
            ORDER BY created_at ASC
        """, [student_id]).fetchall()
        con.close()
        return [
            {
                "skill_name": r[0],
                "mastery_score": r[1],
                "event_type": r[2],
                "timestamp": str(r[3])
            }
            for r in results
        ]
    except Exception as e:
        logger.warning(f"Failed to get learning history: {e}")
        return []

