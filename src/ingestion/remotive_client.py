"""
Remotive API client for fetching remote tech job postings.

Uses the free public Remotive API (https://remotive.com/api/remote-jobs).
No API key required.
"""
import logging
import requests
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

REMOTIVE_API_URL = "https://remotive.com/api/remote-jobs"
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3


def fetch_jobs(category: str = "software-dev", limit: Optional[int] = None) -> list[dict]:
    """
    Fetch job listings from the Remotive API.
    
    Args:
        category: Job category filter (default: software-dev)
        limit: Max number of jobs to return (None = all available)
    
    Returns:
        List of normalized job dictionaries.
    """
    jobs = []
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(f"Fetching Remotive jobs (attempt {attempt}/{MAX_RETRIES})...")
            params = {"category": category}
            if limit:
                params["limit"] = limit
            
            response = requests.get(
                REMOTIVE_API_URL,
                params=params,
                timeout=REQUEST_TIMEOUT,
                headers={"User-Agent": "NEXUS-Capstone/1.0"}
            )
            response.raise_for_status()
            data = response.json()
            
            raw_jobs = data.get("jobs", [])
            logger.info(f"Received {len(raw_jobs)} jobs from Remotive API")
            
            for rj in raw_jobs:
                try:
                    job = _normalize_job(rj)
                    if job and job.get("description"):
                        jobs.append(job)
                except Exception as e:
                    logger.warning(f"Failed to normalize Remotive job {rj.get('id', '?')}: {e}")
            
            break  # Success - exit retry loop
            
        except requests.exceptions.Timeout:
            logger.warning(f"Remotive API timeout (attempt {attempt})")
        except requests.exceptions.HTTPError as e:
            logger.error(f"Remotive API HTTP error: {e}")
            break  # Don't retry on HTTP errors (4xx, 5xx)
        except requests.exceptions.RequestException as e:
            logger.error(f"Remotive API request error: {e}")
        except ValueError as e:
            logger.error(f"Remotive API JSON parse error: {e}")
            break
    
    logger.info(f"Normalized {len(jobs)} valid jobs from Remotive")
    return jobs


def _normalize_job(raw: dict) -> Optional[dict]:
    """Convert a Remotive API job object to our normalized schema."""
    title = (raw.get("title") or "").strip()
    company = (raw.get("company_name") or "Unknown").strip()
    description = (raw.get("description") or "").strip()
    
    if not title or not description:
        return None
    
    # Parse date
    date_posted = raw.get("publication_date", "")
    if date_posted:
        try:
            dt = datetime.fromisoformat(date_posted.replace("Z", "+00:00"))
            date_posted = dt.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            date_posted = ""
    
    # Determine remote status
    job_type = raw.get("job_type", "")
    candidate_required_location = raw.get("candidate_required_location", "")
    
    location = candidate_required_location or "Remote"
    remote_status = "remote" if not candidate_required_location or "remote" in candidate_required_location.lower() else "hybrid"
    
    return {
        "job_id": f"remotive_{raw.get('id', '')}",
        "source": "remotive",
        "title": title,
        "company": company,
        "location": location,
        "remote_status": remote_status,
        "date_posted": date_posted,
        "description": description,
        "url": raw.get("url", ""),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    jobs = fetch_jobs(limit=5)
    for j in jobs[:3]:
        print(f"  [{j['source']}] {j['title']} @ {j['company']}")
        print(f"    Location: {j['location']} | Remote: {j['remote_status']}")
        print(f"    Description length: {len(j['description'])} chars")
        print()
