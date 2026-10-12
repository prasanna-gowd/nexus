# NEXUS Phase 2: NLP, Skill Extraction & Taxonomy Mapping Implementation Plan

## 1. Objective
Transform unstructured, noisy job postings stored in the local DuckDB `raw_jobs` table[cite: 3] into a clean, normalized relational graph of verified technical skills and emerging market trends[cite: 2].

## 2. Prerequisites & Environment
*   **Local Data:** The official ESCO dataset (`skills_en.csv`) downloaded and placed in the `data/` directory[cite: 2].
*   **Dependencies:** `pip install gliner sentence-transformers beautifulsoup4 duckdb pandas scikit-learn`
*   **Upstream Data:** A populated DuckDB database containing raw job postings from Phase 1[cite: 3].

## 3. Pipeline Architecture

The Phase 2 pipeline consists of four sequential Python modules:

### Component 1: `src/nlp/cleaner.py` (Noise Reduction)
Raw job descriptions contain HTML tags, equal opportunity disclaimers, and irrelevant company perks[cite: 2].
*   **Action:** Use `BeautifulSoup` to strip HTML.
*   **Action:** Use regex heuristics to isolate the `Requirements` or `Qualifications` sections, discarding the rest of the noise[cite: 2].

### Component 2: `src/nlp/extractor.py` (Zero-Shot NER)
Extract raw skill strings from the cleaned requirements text without paying for LLM API calls[cite: 2].
*   **Action:** Initialize `GLiNER` (e.g., `urchade/gliner_base`)[cite: 2].
*   **Action:** Extract entities matching the labels: `TOOL`, `PROGRAMMING_LANGUAGE`, `FRAMEWORK`, `CONCEPT`, `MINIMUM_EXPERIENCE`[cite: 2].
*   **Action:** Apply a contextual filter to remove generic non-technical adjectives (e.g., "team player")[cite: 2].

### Component 3: `src/nlp/taxonomy.py` (Normalization & Zero-Day Detection)
Map extracted raw strings (e.g., "ReactJS") to the canonical ESCO standard (e.g., "React")[cite: 2].
*   **Action:** Encode the entire ESCO dataset using `SentenceTransformer('BAAI/bge-small-en-v1.5')`[cite: 2].
*   **Action:** Encode the newly extracted skill from GLiNER.
*   **Action:** Calculate cosine similarity against the ESCO baseline[cite: 2].
    *   **Match:** If similarity $> 0.82$, map the extracted string to the official ESCO canonical ID[cite: 2].
    *   **Discover:** If similarity $< 0.82$, flag the extracted string as a "Zero-Day Emerging Skill" and assign it a new ID[cite: 2].

### Component 4: `src/nlp/orchestrator.py` (Batch Processing)
Coordinates the workflow between DuckDB and the NLP models.
*   **Action:** Fetch unprocessed rows from the `raw_jobs` table[cite: 3].
*   **Action:** Run the Cleaner $\to$ Extractor $\to$ Taxonomy sequence.
*   **Action:** Insert new unique skills into the `skills_taxonomy` table and create relational edges in the `job_skills` table[cite: 2].

## 4. DuckDB Schema Updates

Execute these SQL commands in DuckDB to prepare for the extracted data:

```sql
-- Store unique, normalized skills and zero-day discoveries
CREATE TABLE IF NOT EXISTS skills_taxonomy (
    skill_id UUID DEFAULT uuid(),
    canonical_name VARCHAR NOT NULL UNIQUE,
    category VARCHAR NOT NULL,
    esco_uri VARCHAR,
    is_emerging BOOLEAN DEFAULT FALSE,
    PRIMARY KEY (skill_id)
);

-- Store the mapping between a specific job and its required skills
CREATE TABLE IF NOT EXISTS job_skills (
    job_id VARCHAR,
    skill_id UUID,
    confidence_score FLOAT,
    FOREIGN KEY (job_id) REFERENCES raw_jobs(job_id),
    FOREIGN KEY (skill_id) REFERENCES skills_taxonomy(skill_id),
    PRIMARY KEY (job_id, skill_id)
);