# NEXUS Phase 1: Ingestion & Extraction Layer Implementation Plan

This plan covers the implementation of **Phase 1: Ingestion & Extraction Layer** as outlined in your `plan.md`. The goal is to reliably gather structured and unstructured job data and store it in a local database for downstream NLP extraction.

## User Review Required
> [!IMPORTANT]
> Please review this plan. Once approved, I will begin executing the steps to create the Python scripts and set up the local DuckDB database.

## Open Questions
> [!NOTE]
> 1. Are you okay with using **DuckDB** as the initial raw storage solution before moving to PostgreSQL? It is a highly performant, serverless SQL database that is perfect for local data engineering and development.
> 2. I plan to use the **Remotive Public API** and the **Hacker News Firebase API**, as they are free and require no API keys to get started. Is this acceptable?

## Proposed Changes

We will set up a new Python project structure for the data pipeline.

### Project Setup and Dependencies
- Create a `requirements.txt` with necessary libraries: `requests`, `duckdb`, `pandas`, `beautifulsoup4`.

### Ingestion Clients

#### [NEW] `src/ingestion/db.py`
Sets up the DuckDB connection and initializes the raw storage table schema with deduplication logic (using MD5 hashes of company + title + text).

#### [NEW] `src/ingestion/remotive_client.py`
A client to fetch job listings from the public Remotive API (`https://remotive.com/api/remote-jobs`), parse the required fields (job_id, title, company, location, date, description), and return normalized dictionaries.

#### [NEW] `src/ingestion/hackernews_client.py`
A client to query the Hacker News API to find the latest "Ask HN: Who is hiring?" thread and parse the top-level comments as job postings.

#### [NEW] `src/ingestion/pipeline.py`
The main orchestrator script that:
1. Initializes the DuckDB database.
2. Calls the APIs to fetch data.
3. Hashes incoming jobs for deduplication.
4. Inserts new records into the DuckDB `raw_jobs` table.

## Verification Plan

### Automated/Manual Verification
- Run the ingestion pipeline (`python src/ingestion/pipeline.py`).
- Query the DuckDB database to verify that jobs were successfully inserted, and that no duplicates exist.
- Output a sample row to the console to ensure all metadata (title, company, description) is correctly captured.
