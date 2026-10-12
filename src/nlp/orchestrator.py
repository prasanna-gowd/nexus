"""
NLP orchestrator for batch processing job descriptions.

Coordinates the cleaner, extractor, and taxonomy mapper
to process unprocessed jobs in DuckDB.
"""
import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.ingestion.db import get_connection, init_schema
from src.nlp.cleaner import clean_description
from src.nlp.extractor import SkillExtractor
from src.nlp.taxonomy import TaxonomyMapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def run_nlp_pipeline(db_path: str = None, use_gliner: bool = False) -> dict:
    """
    Process unprocessed jobs through the NLP pipeline.
    
    Args:
        db_path: Optional DuckDB path.
        use_gliner: Whether to use GLiNER (requires model download).
    
    Returns:
        Summary with processing statistics.
    """
    summary = {
        "jobs_processed": 0,
        "skills_extracted": 0,
        "skills_mapped_esco": 0,
        "skills_emerging": 0,
        "job_skill_links": 0,
        "errors": 0,
    }
    
    conn = get_connection(db_path)
    init_schema(conn)
    
    # Find unprocessed jobs (not yet in job_skills)
    unprocessed = conn.execute("""
        SELECT r.job_id, r.description 
        FROM raw_jobs r
        LEFT JOIN job_skills js ON r.job_id = js.job_id
        WHERE js.job_id IS NULL
    """).fetchall()
    
    total = len(unprocessed)
    logger.info(f"Found {total} unprocessed jobs")
    
    if total == 0:
        logger.info("No unprocessed jobs found. Pipeline is up to date.")
        conn.close()
        return summary
    
    # Initialize NLP components
    extractor = SkillExtractor(use_gliner=use_gliner)
    mapper = TaxonomyMapper()
    
    # Track unique skills to avoid duplicates
    seen_skills = set()
    
    # Load existing skills
    existing = conn.execute("SELECT canonical_name FROM skills_taxonomy").fetchall()
    for row in existing:
        seen_skills.add(row[0].lower())
    
    for i, (job_id, description) in enumerate(unprocessed):
        try:
            # Step 1: Clean the description
            cleaned = clean_description(description)
            
            if not cleaned or len(cleaned) < 20:
                logger.debug(f"Skipping job {job_id}: description too short after cleaning")
                continue
            
            # Step 2: Extract skills
            raw_skills = extractor.extract_skills(cleaned)
            
            if not raw_skills:
                logger.debug(f"No skills extracted from job {job_id}")
                continue
            
            # Step 3: Map to taxonomy and store
            for skill_data in raw_skills:
                try:
                    mapped = mapper.map_skill(
                        skill_data["skill"],
                        skill_data.get("category", "tool")
                    )
                    
                    canonical = mapped["canonical_name"]
                    skill_id = mapped["skill_id"]
                    
                    # Insert into skills_taxonomy if new
                    if canonical.lower() not in seen_skills:
                        try:
                            conn.execute("""
                                INSERT INTO skills_taxonomy 
                                (skill_id, canonical_name, category, esco_uri, confidence, is_emerging)
                                VALUES (?, ?, ?, ?, ?, ?)
                            """, [
                                skill_id,
                                canonical,
                                mapped["category"],
                                mapped.get("esco_uri", ""),
                                mapped.get("confidence", 0.0),
                                mapped.get("is_emerging", False)
                            ])
                            seen_skills.add(canonical.lower())
                            summary["skills_extracted"] += 1
                            
                            if mapped.get("is_emerging"):
                                summary["skills_emerging"] += 1
                            else:
                                summary["skills_mapped_esco"] += 1
                                
                        except Exception as e:
                            if "Duplicate" not in str(e) and "UNIQUE" not in str(e):
                                logger.debug(f"Skill insert error for '{canonical}': {e}")
                    
                    # Insert job-skill relationship
                    try:
                        conn.execute("""
                            INSERT INTO job_skills (job_id, skill_id, confidence_score)
                            VALUES (?, ?, ?)
                        """, [
                            job_id,
                            skill_id,
                            skill_data.get("confidence", 0.5)
                        ])
                        summary["job_skill_links"] += 1
                    except Exception:
                        pass  # Duplicate link, skip
                    
                except Exception as e:
                    logger.debug(f"Error mapping skill '{skill_data.get('skill', '?')}': {e}")
            
            summary["jobs_processed"] += 1
            
            if (i + 1) % 50 == 0:
                logger.info(f"Processed {i + 1}/{total} jobs...")
                
        except Exception as e:
            logger.warning(f"Error processing job {job_id}: {e}")
            summary["errors"] += 1
    
    # Final summary
    total_skills = conn.execute("SELECT COUNT(*) FROM skills_taxonomy").fetchone()[0]
    total_links = conn.execute("SELECT COUNT(*) FROM job_skills").fetchone()[0]
    
    logger.info("=" * 60)
    logger.info("NLP PIPELINE SUMMARY")
    logger.info(f"  Jobs processed:      {summary['jobs_processed']}")
    logger.info(f"  New skills found:    {summary['skills_extracted']}")
    logger.info(f"  ESCO matched:        {summary['skills_mapped_esco']}")
    logger.info(f"  Emerging skills:     {summary['skills_emerging']}")
    logger.info(f"  Job-skill links:     {summary['job_skill_links']}")
    logger.info(f"  Errors:              {summary['errors']}")
    logger.info(f"  Total skills in DB:  {total_skills}")
    logger.info(f"  Total links in DB:   {total_links}")
    logger.info("=" * 60)
    
    # Show top skills
    top_skills = conn.execute("""
        SELECT st.canonical_name, st.category, COUNT(js.job_id) as job_count
        FROM skills_taxonomy st
        JOIN job_skills js ON st.skill_id = js.skill_id
        GROUP BY st.canonical_name, st.category
        ORDER BY job_count DESC
        LIMIT 15
    """).fetchall()
    
    if top_skills:
        logger.info("Top extracted skills:")
        for name, category, count in top_skills:
            logger.info(f"  {name} ({category}): {count} jobs")
    
    conn.close()
    return summary


if __name__ == "__main__":
    summary = run_nlp_pipeline(use_gliner=False)
