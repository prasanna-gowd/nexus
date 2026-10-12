"""
Unit & Integration Tests for Phase 1: Data Ingestion & DuckDB Raw Storage.
"""
import os
import tempfile
import duckdb
import pytest

from src.ingestion.db import (
    get_connection,
    init_schema,
    compute_dedup_hash,
    insert_jobs
)
from src.ingestion.remotive_client import _normalize_job
from src.ingestion.hackernews_client import _parse_comment_as_job


@pytest.fixture
def temp_duckdb():
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_nexus.duckdb")
    conn = get_connection(db_path)
    init_schema(conn)
    yield conn, db_path
    conn.close()


def test_dedup_hash_deterministic():
    h1 = compute_dedup_hash("Software Engineer", "Google", "Python and distributed systems")
    h2 = compute_dedup_hash("software engineer", "google", "Python and distributed systems")
    h3 = compute_dedup_hash("Software Engineer", "Meta", "Python and distributed systems")
    assert h1 == h2, "Hash must be case-insensitive for title and company"
    assert h1 != h3, "Different companies must generate distinct hashes"


def test_insert_jobs_and_deduplication(temp_duckdb):
    conn, _ = temp_duckdb
    sample_jobs = [
        {
            "job_id": "test_1",
            "source": "test_source",
            "title": "Backend Developer",
            "company": "Acme Corp",
            "description": "Building microservices with Python and FastAPI",
            "location": "Remote",
            "remote_status": "remote",
        },
        {
            "job_id": "test_2",
            "source": "test_source",
            "title": "ML Engineer",
            "company": "Neural Corp",
            "description": "Training PyTorch models on GPUs",
            "location": "San Francisco",
            "remote_status": "onsite",
        }
    ]

    # First insertion
    stats1 = insert_jobs(conn, sample_jobs)
    assert stats1["inserted"] == 2
    assert stats1["skipped"] == 0
    assert stats1["failed"] == 0

    count1 = conn.execute("SELECT COUNT(*) FROM raw_jobs").fetchone()[0]
    assert count1 == 2

    # Second insertion with same data (deduplication check)
    stats2 = insert_jobs(conn, sample_jobs)
    assert stats2["inserted"] == 0
    assert stats2["skipped"] == 2
    assert stats2["failed"] == 0

    count2 = conn.execute("SELECT COUNT(*) FROM raw_jobs").fetchone()[0]
    assert count2 == 2, "Duplicate rows should not be inserted"


def test_remotive_job_normalization():
    raw_remotive = {
        "id": 12345,
        "title": "Senior Python Developer ",
        "company_name": "TechStart Inc",
        "description": "<p>We are seeking a Python engineer.</p>",
        "publication_date": "2026-03-15T12:00:00Z",
        "candidate_required_location": "USA Remote",
        "url": "https://remotive.com/jobs/12345"
    }

    normalized = _normalize_job(raw_remotive)
    assert normalized is not None
    assert normalized["job_id"] == "remotive_12345"
    assert normalized["title"] == "Senior Python Developer"
    assert normalized["company"] == "TechStart Inc"
    assert normalized["remote_status"] == "remote"
    assert normalized["date_posted"] == "2026-03-15"


def test_hn_comment_parser():
    comment = {
        "id": 987654,
        "text": "<p>Stripe | San Francisco, CA | Full Stack Engineer | REMOTE | https://stripe.com<br>We build global economic infrastructure.</p>"
    }
    parsed = _parse_comment_as_job(comment, "Ask HN: Who is hiring? (March 2026)", "2026-03-01")
    assert parsed is not None
    assert "stripe" in parsed["company"].lower()
    assert parsed["remote_status"] == "remote"
    assert "economic infrastructure" in parsed["description"].lower()


def test_malformed_record_handling(temp_duckdb):
    conn, _ = temp_duckdb
    malformed = [
        {"title": "", "company": ""}, # missing description
        {"description": "Just some text without title"}
    ]
    # Ingestion should not crash
    stats = insert_jobs(conn, malformed)
    assert stats["failed"] == 0
