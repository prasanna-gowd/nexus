"""
Market Analytics Engine for NEXUS.

Calculates:
1. Overall skill demand & frequency rankings.
2. Role-specific skill demand.
3. Skill velocity (week-over-week changes).
4. Technology co-occurrence matrix (frequently paired technologies).

Supports both PostgreSQL (primary production analytical store)
and DuckDB (seamless fallback/local dev store).
"""
import os
import logging
from collections import defaultdict
from typing import Dict, List, Any, Optional
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

DUCKDB_PATH = os.getenv("DUCKDB_PATH", "data/nexus_jobs.duckdb")
PG_HOST = os.getenv("POSTGRES_HOST", "localhost")
PG_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
PG_USER = os.getenv("POSTGRES_USER", "nexus")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD", "nexus")
PG_DB = os.getenv("POSTGRES_DB", "nexus_db")


class MarketAnalytics:
    """Computes real job market intelligence and skill demand metrics."""

    def __init__(self, use_postgres: bool = True):
        self.use_postgres = use_postgres
        self.pg_available = False
        if use_postgres:
            self.pg_available = self._test_pg_connection()

    def _test_pg_connection(self) -> bool:
        try:
            import psycopg2
            conn = psycopg2.connect(
                host=PG_HOST, port=PG_PORT, user=PG_USER, password=PG_PASSWORD, dbname=PG_DB,
                connect_timeout=3
            )
            conn.close()
            return True
        except Exception:
            return False

    def _get_pg_conn(self):
        import psycopg2
        return psycopg2.connect(
            host=PG_HOST, port=PG_PORT, user=PG_USER, password=PG_PASSWORD, dbname=PG_DB
        )

    def _get_duck_conn(self):
        import duckdb
        return duckdb.connect(DUCKDB_PATH)

    def get_top_skills(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Returns the most frequently requested technical skills across all jobs."""
        if self.pg_available:
            try:
                conn = self._get_pg_conn()
                query = """
                    SELECT s.standard_name as skill_name, s.category, COUNT(js.job_id) as demand_count
                    FROM skills s
                    JOIN job_skills js ON s.id = js.skill_id
                    GROUP BY s.standard_name, s.category
                    ORDER BY demand_count DESC
                    LIMIT %s;
                """
                df = pd.read_sql(query, conn, params=(limit,))
                conn.close()
                return df.to_dict(orient="records")
            except Exception as e:
                logger.warning(f"PostgreSQL query failed, falling back to DuckDB: {e}")

        # DuckDB fallback
        try:
            conn = self._get_duck_conn()
            query = """
                SELECT st.canonical_name as skill_name, st.category, COUNT(js.job_id) as demand_count
                FROM skills_taxonomy st
                JOIN job_skills js ON st.skill_id = js.skill_id
                GROUP BY st.canonical_name, st.category
                ORDER BY demand_count DESC
                LIMIT ?;
            """
            df = conn.execute(query, [limit]).fetchdf()
            conn.close()
            return df.to_dict(orient="records")
        except Exception as e:
            logger.error(f"Failed to query top skills from DuckDB: {e}")
            return []

    def get_skills_by_role(self, role_keyword: str, limit: int = 15) -> List[Dict[str, Any]]:
        """Returns required skills for jobs matching a specific role keyword."""
        pattern = f"%{role_keyword.lower()}%"
        if self.pg_available:
            try:
                conn = self._get_pg_conn()
                query = """
                    SELECT s.standard_name as skill_name, s.category, COUNT(DISTINCT j.id) as demand_count
                    FROM jobs j
                    JOIN job_skills js ON j.id = js.job_id
                    JOIN skills s ON js.skill_id = s.id
                    WHERE LOWER(j.title) LIKE %s
                    GROUP BY s.standard_name, s.category
                    ORDER BY demand_count DESC
                    LIMIT %s;
                """
                df = pd.read_sql(query, conn, params=(pattern, limit))
                conn.close()
                return df.to_dict(orient="records")
            except Exception as e:
                logger.warning(f"PG role query failed, falling back to DuckDB: {e}")

        # DuckDB fallback
        try:
            conn = self._get_duck_conn()
            query = """
                SELECT st.canonical_name as skill_name, st.category, COUNT(DISTINCT rj.job_id) as demand_count
                FROM raw_jobs rj
                JOIN job_skills js ON rj.job_id = js.job_id
                JOIN skills_taxonomy st ON js.skill_id = st.skill_id
                WHERE LOWER(rj.title) LIKE ?
                GROUP BY st.canonical_name, st.category
                ORDER BY demand_count DESC
                LIMIT ?;
            """
            df = conn.execute(query, [pattern, limit]).fetchdf()
            conn.close()
            return df.to_dict(orient="records")
        except Exception as e:
            logger.error(f"Failed to query skills by role: {e}")
            return []

    def get_skill_co_occurrences(self, target_skill: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Discovers complementary technologies frequently appearing alongside target_skill.
        """
        target_skill_clean = target_skill.strip().lower()
        if self.pg_available:
            try:
                conn = self._get_pg_conn()
                query = """
                    WITH target_jobs AS (
                        SELECT js.job_id
                        FROM job_skills js
                        JOIN skills s ON js.skill_id = s.id
                        WHERE LOWER(s.standard_name) = %s
                    )
                    SELECT s.standard_name as related_skill, s.category, COUNT(js.job_id) as co_occurrence_count
                    FROM job_skills js
                    JOIN skills s ON js.skill_id = s.id
                    WHERE js.job_id IN (SELECT job_id FROM target_jobs)
                      AND LOWER(s.standard_name) != %s
                    GROUP BY s.standard_name, s.category
                    ORDER BY co_occurrence_count DESC
                    LIMIT %s;
                """
                df = pd.read_sql(query, conn, params=(target_skill_clean, target_skill_clean, limit))
                conn.close()
                return df.to_dict(orient="records")
            except Exception as e:
                logger.warning(f"PG co-occurrence query failed: {e}")

        # DuckDB fallback
        try:
            conn = self._get_duck_conn()
            query = """
                WITH target_jobs AS (
                    SELECT js.job_id
                    FROM job_skills js
                    JOIN skills_taxonomy st ON js.skill_id = st.skill_id
                    WHERE LOWER(st.canonical_name) = ?
                )
                SELECT st.canonical_name as related_skill, st.category, COUNT(js.job_id) as co_occurrence_count
                FROM job_skills js
                JOIN skills_taxonomy st ON js.skill_id = st.skill_id
                WHERE js.job_id IN (SELECT job_id FROM target_jobs)
                  AND LOWER(st.canonical_name) != ?
                GROUP BY st.canonical_name, st.category
                ORDER BY co_occurrence_count DESC
                LIMIT ?;
            """
            df = conn.execute(query, [target_skill_clean, target_skill_clean, limit]).fetchdf()
            conn.close()
            return df.to_dict(orient="records")
        except Exception as e:
            logger.error(f"Failed to query co-occurrences: {e}")
            return []

    def get_skill_velocity(self) -> Dict[str, Any]:
        """
        Calculates week-over-week frequency shifts when sufficient temporal data exists.
        Handles sparse dates with transparent indicators.
        """
        # Read jobs with dates from available database
        df_jobs = None
        if self.pg_available:
            try:
                conn = self._get_pg_conn()
                df_jobs = pd.read_sql("""
                    SELECT j.id as job_id, j.date_posted, s.standard_name as skill_name
                    FROM jobs j
                    JOIN job_skills js ON j.id = js.job_id
                    JOIN skills s ON js.skill_id = s.id
                    WHERE j.date_posted IS NOT NULL;
                """, conn)
                conn.close()
            except Exception:
                pass

        if df_jobs is None or df_jobs.empty:
            try:
                conn = self._get_duck_conn()
                df_jobs = conn.execute("""
                    SELECT rj.job_id, rj.date_posted, st.canonical_name as skill_name
                    FROM raw_jobs rj
                    JOIN job_skills js ON rj.job_id = js.job_id
                    JOIN skills_taxonomy st ON js.skill_id = st.skill_id
                    WHERE rj.date_posted != '';
                """).fetchdf()
                conn.close()
            except Exception:
                pass

        if df_jobs is None or df_jobs.empty or len(df_jobs) < 10:
            return {
                "status": "insufficient_temporal_data",
                "message": "Insufficient dated records to compute honest WoW growth rates. Showing total counts instead.",
                "trends": self.get_top_skills(limit=10)
            }

        try:
            df_jobs["date"] = pd.to_datetime(df_jobs["date_posted"], errors="coerce")
            df_jobs = df_jobs.dropna(subset=["date"])
            if df_jobs.empty:
                return {
                    "status": "insufficient_temporal_data",
                    "message": "Job dates could not be parsed into continuous date intervals.",
                    "trends": self.get_top_skills(limit=10)
                }

            # Split into recent period (last 14 days) vs previous period (14-28 days ago)
            max_date = df_jobs["date"].max()
            split_date = max_date - pd.Timedelta(days=14)
            prev_date = max_date - pd.Timedelta(days=28)

            recent_counts = df_jobs[df_jobs["date"] > split_date]["skill_name"].value_counts()
            prev_counts = df_jobs[(df_jobs["date"] <= split_date) & (df_jobs["date"] >= prev_date)]["skill_name"].value_counts()

            trends = []
            for skill, r_cnt in recent_counts.head(15).items():
                p_cnt = prev_counts.get(skill, 0)
                velocity = ((r_cnt - p_cnt) / max(p_cnt, 1)) * 100 if p_cnt > 0 else 100.0
                trends.append({
                    "skill_name": skill,
                    "recent_count": int(r_cnt),
                    "previous_count": int(p_cnt),
                    "velocity_pct": round(velocity, 1)
                })

            return {
                "status": "calculated",
                "window": "14-day comparison",
                "trends": trends
            }
        except Exception as e:
            logger.error(f"Error computing velocity: {e}")
            return {
                "status": "error",
                "message": str(e),
                "trends": self.get_top_skills(limit=10)
            }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    analytics = MarketAnalytics()
    print("Top skills:")
    for s in analytics.get_top_skills(5):
        print(" ", s)
