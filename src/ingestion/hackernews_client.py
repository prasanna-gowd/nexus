"""
Hacker News API client for extracting job postings from 'Who is hiring?' threads.

Uses the HN Algolia API to find hiring threads and the Firebase API
to fetch individual comments/stories.
"""
import re
import logging
import requests
from datetime import datetime, timezone
from typing import Optional
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

ALGOLIA_API = "https://hn.algolia.com/api/v1/search"
HN_ITEM_API = "https://hacker-news.firebaseio.com/v0/item"
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3


def fetch_jobs(max_comments: int = 200) -> list[dict]:
    """
    Fetch job postings from the latest 'Ask HN: Who is hiring?' thread.
    
    Args:
        max_comments: Maximum number of top-level comments to process.
    
    Returns:
        List of normalized job dictionaries extracted from HN comments.
    """
    jobs = []
    
    # Step 1: Find the latest hiring thread
    thread_id = _find_latest_hiring_thread()
    if not thread_id:
        logger.warning("Could not find a 'Who is hiring?' thread on HN")
        return jobs
    
    logger.info(f"Found hiring thread: {thread_id}")
    
    # Step 2: Get the thread and its comment IDs
    thread_data = _fetch_hn_item(thread_id)
    if not thread_data:
        logger.error(f"Could not fetch thread {thread_id}")
        return jobs
    
    thread_title = thread_data.get("title", "")
    thread_date = ""
    thread_time = thread_data.get("time")
    if thread_time:
        try:
            thread_date = datetime.fromtimestamp(thread_time, tz=timezone.utc).strftime("%Y-%m-%d")
        except (ValueError, OSError):
            pass
    
    comment_ids = thread_data.get("kids", [])[:max_comments]
    logger.info(f"Processing {len(comment_ids)} comments from '{thread_title}'")
    
    # Step 3: Fetch and parse each top-level comment as a job posting
    for i, comment_id in enumerate(comment_ids):
        try:
            comment = _fetch_hn_item(comment_id)
            if not comment or comment.get("deleted") or comment.get("dead"):
                continue
            
            text = comment.get("text", "")
            if not text or len(text) < 50:
                continue
            
            job = _parse_comment_as_job(comment, thread_title, thread_date)
            if job:
                jobs.append(job)
                
        except Exception as e:
            logger.warning(f"Error processing comment {comment_id}: {e}")
        
        if (i + 1) % 50 == 0:
            logger.info(f"Processed {i + 1}/{len(comment_ids)} comments...")
    
    logger.info(f"Extracted {len(jobs)} job postings from HN")
    return jobs


def _find_latest_hiring_thread() -> Optional[int]:
    """Find the latest 'Ask HN: Who is hiring?' thread via Algolia."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                ALGOLIA_API,
                params={
                    "query": "Ask HN: Who is hiring?",
                    "tags": "story,ask_hn",
                    "hitsPerPage": 5,
                },
                timeout=REQUEST_TIMEOUT,
                headers={"User-Agent": "NEXUS-Capstone/1.0"}
            )
            response.raise_for_status()
            hits = response.json().get("hits", [])
            
            # Filter for actual hiring threads (not "Who wants to be hired")
            for hit in hits:
                title = hit.get("title", "").lower()
                if "who is hiring" in title and "wants to be" not in title:
                    return int(hit["objectID"])
            
            logger.warning("No matching hiring thread found in Algolia results")
            return None
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Algolia search attempt {attempt} failed: {e}")
        except (ValueError, KeyError) as e:
            logger.error(f"Error parsing Algolia response: {e}")
            return None
    
    return None


def _fetch_hn_item(item_id: int) -> Optional[dict]:
    """Fetch a single HN item (story or comment) from the Firebase API."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            url = f"{HN_ITEM_API}/{item_id}.json"
            response = requests.get(
                url,
                timeout=REQUEST_TIMEOUT,
                headers={"User-Agent": "NEXUS-Capstone/1.0"}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.debug(f"HN item {item_id} fetch attempt {attempt} failed: {e}")
    return None


def _parse_comment_as_job(comment: dict, thread_title: str, thread_date: str) -> Optional[dict]:
    """
    Parse a top-level HN comment as a job posting.
    
    HN hiring comments typically start with: Company | Location | Role | ...
    """
    raw_text = comment.get("text", "")
    
    # Clean HTML from the comment
    soup = BeautifulSoup(raw_text, "html.parser")
    clean_text = soup.get_text(separator="\n").strip()
    
    if not clean_text or len(clean_text) < 30:
        return None
    
    # Extract company name from the first line
    lines = clean_text.split("\n")
    first_line = lines[0].strip()
    
    # Parse the pipe-separated header: Company | Location | Role | ...
    parts = [p.strip() for p in first_line.split("|")]
    
    company = parts[0] if parts else "Unknown"
    # Clean company name - remove URLs and common prefixes
    company = re.sub(r'https?://\S+', '', company).strip()
    company = company[:100] if company else "Unknown"
    
    location = ""
    title = ""
    remote_status = ""
    
    for part in parts[1:]:
        part_lower = part.lower()
        if any(word in part_lower for word in ["remote", "onsite", "hybrid", "on-site"]):
            if "remote" in part_lower:
                remote_status = "remote"
            elif "hybrid" in part_lower:
                remote_status = "hybrid"
            else:
                remote_status = "onsite"
            if not location:
                location = part
        elif any(word in part_lower for word in ["engineer", "developer", "manager", "designer",
                                                   "scientist", "analyst", "architect", "lead",
                                                   "senior", "junior", "intern", "full stack",
                                                   "backend", "frontend", "devops", "sre", "ml"]):
            title = part
        elif not location:
            location = part
    
    if not title:
        title = f"Engineering Role at {company}"
    
    # Extract URLs from the original text
    urls = re.findall(r'https?://[^\s<>"\']+', raw_text)
    url = urls[0] if urls else f"https://news.ycombinator.com/item?id={comment.get('id', '')}"
    
    comment_id = comment.get("id", "")
    
    return {
        "job_id": f"hn_{comment_id}",
        "source": "hackernews",
        "title": title.strip(),
        "company": company.strip() or "Unknown",
        "location": location.strip() or "Not specified",
        "remote_status": remote_status or "not_specified",
        "date_posted": thread_date,
        "description": clean_text,
        "url": url,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    jobs = fetch_jobs(max_comments=10)
    for j in jobs[:5]:
        print(f"  [{j['source']}] {j['title']} @ {j['company']}")
        print(f"    Location: {j['location']} | Remote: {j['remote_status']}")
        print(f"    Description length: {len(j['description'])} chars")
        print()
