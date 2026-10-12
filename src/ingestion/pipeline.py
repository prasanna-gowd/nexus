"""
Data ingestion pipeline orchestrator for NEXUS.

Coordinates the Remotive and Hacker News clients, handles deduplication,
and stores results in DuckDB. Safe to re-run.
"""
import sys
import os
import logging
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.ingestion.db import get_connection, init_schema, insert_jobs
from src.ingestion.remotive_client import fetch_jobs as fetch_remotive
from src.ingestion.hackernews_client import fetch_jobs as fetch_hn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def run_pipeline(db_path: str = None, skip_remotive: bool = False, skip_hn: bool = False) -> dict:
    """
    Execute the complete data ingestion pipeline.
    
    Args:
        db_path: Optional custom DuckDB path.
        skip_remotive: Skip Remotive API.
        skip_hn: Skip Hacker News API.
    
    Returns:
        Summary dict with counts per source and totals.
    """
    summary = {
        "start_time": datetime.now(timezone.utc).isoformat(),
        "sources": {},
        "total_fetched": 0,
        "total_inserted": 0,
        "total_skipped": 0,
        "total_failed": 0,
    }
    
    # Initialize database
    logger.info("=" * 60)
    logger.info("NEXUS Data Ingestion Pipeline")
    logger.info("=" * 60)
    
    conn = get_connection(db_path)
    init_schema(conn)
    
    # Count existing records
    existing_count = conn.execute("SELECT COUNT(*) FROM raw_jobs").fetchone()[0]
    logger.info(f"Existing records in database: {existing_count}")
    
    # Fetch from Remotive
    if not skip_remotive:
        logger.info("-" * 40)
        logger.info("Fetching from Remotive API...")
        try:
            remotive_jobs = fetch_remotive()
            stats = insert_jobs(conn, remotive_jobs)
            summary["sources"]["remotive"] = {
                "fetched": len(remotive_jobs),
                **stats
            }
            summary["total_fetched"] += len(remotive_jobs)
            summary["total_inserted"] += stats["inserted"]
            summary["total_skipped"] += stats["skipped"]
            summary["total_failed"] += stats["failed"]
            logger.info(f"Remotive: fetched={len(remotive_jobs)}, "
                       f"inserted={stats['inserted']}, "
                       f"skipped={stats['skipped']}, "
                       f"failed={stats['failed']}")
        except Exception as e:
            logger.error(f"Remotive ingestion failed: {e}")
            summary["sources"]["remotive"] = {"error": str(e)}
    
    # Fetch from Hacker News
    if not skip_hn:
        logger.info("-" * 40)
        logger.info("Fetching from Hacker News API...")
        try:
            hn_jobs = fetch_hn(max_comments=200)
            stats = insert_jobs(conn, hn_jobs)
            summary["sources"]["hackernews"] = {
                "fetched": len(hn_jobs),
                **stats
            }
            summary["total_fetched"] += len(hn_jobs)
            summary["total_inserted"] += stats["inserted"]
            summary["total_skipped"] += stats["skipped"]
            summary["total_failed"] += stats["failed"]
            logger.info(f"HackerNews: fetched={len(hn_jobs)}, "
                       f"inserted={stats['inserted']}, "
                       f"skipped={stats['skipped']}, "
                       f"failed={stats['failed']}")
        except Exception as e:
            logger.error(f"HackerNews ingestion failed: {e}")
            summary["sources"]["hackernews"] = {"error": str(e)}
    
    # Final summary
    final_count = conn.execute("SELECT COUNT(*) FROM raw_jobs").fetchone()[0]
    summary["end_time"] = datetime.now(timezone.utc).isoformat()
    summary["final_record_count"] = final_count
    
    logger.info("=" * 60)
    logger.info("INGESTION SUMMARY")
    logger.info(f"  Total fetched:  {summary['total_fetched']}")
    logger.info(f"  Total inserted: {summary['total_inserted']}")
    logger.info(f"  Total skipped:  {summary['total_skipped']}")
    logger.info(f"  Total failed:   {summary['total_failed']}")
    logger.info(f"  Records in DB:  {final_count}")
    logger.info("=" * 60)
    
    # Show sample records
    samples = conn.execute("""
        SELECT job_id, source, title, company 
        FROM raw_jobs 
        ORDER BY ingested_at DESC 
        LIMIT 5
    """).fetchall()
    
    if samples:
        logger.info("Sample records:")
        for s in samples:
            logger.info(f"  [{s[1]}] {s[2]} @ {s[3]}")
    
    conn.close()
    return summary


if __name__ == "__main__":
    summary = run_pipeline()
