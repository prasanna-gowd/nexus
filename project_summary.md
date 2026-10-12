# NEXUS Project Implementation Summary

This document summarizes all the work, pipelines, and architecture implemented in the NEXUS project across all five phases.

---

## Phase 1: Data Ingestion & Warehouse Setup
**Objective:** Gather unstructured job postings from tech-focused job boards and establish a local, lightweight data warehouse.
*   **API Clients (`src/ingestion/`)**: 
    *   `hackernews_client.py`: Connected to Algolia and Firebase APIs to scrape monthly "Ask HN: Who is hiring?" threads.
    *   `remotive_client.py`: Connected to the Remotive API to fetch structured remote tech jobs.
*   **Storage Setup**: Initialized DuckDB (`data/nexus_jobs.duckdb`) and created the `raw_jobs` table with deduplication hashes.

---

## Phase 2: NLP, Skill Extraction & Taxonomy Mapping
**Objective:** Transform raw job postings into a normalized relational graph of verified technical skills without expensive paid LLM APIs.
*   **Noise Reduction (`src/nlp/cleaner.py`)**: Stripped raw HTML with `BeautifulSoup` and isolated requirements sections using regex heuristics.
*   **Zero-Shot NER (`src/nlp/extractor.py`)**: Extracted entities (`TOOL`, `PROGRAMMING_LANGUAGE`, `FRAMEWORK`) using local **GLiNER** (`urchade/gliner_base`).
*   **Taxonomy Normalization (`src/nlp/taxonomy.py`)**: 
    *   Mapped skills to the **ESCO** taxonomy standard (`skills_en.csv`).
    *   Embedded skills using **SentenceTransformers** (`BAAI/bge-small-en-v1.5`) and applied cosine similarity thresholds (> 0.82) to resolve canonical standards vs. emerging skills.
*   **Orchestration (`src/nlp/orchestrator.py`)**: DuckDB pipeline populating `skills_taxonomy` and `job_skills`.

---

## Phase 3: Hybrid Storage & Trend Analytics Engine
**Objective:** Migrate into a production-grade PostgreSQL + `pgvector` database supporting both relational queries and semantic vector retrieval.
*   **Infrastructure (`src/database/schema.sql`)**: 
    *   Deployed PostgreSQL 16 with `pgvector` in Docker container `pgvector-container` on port 5432.
    *   Created relational schema (`jobs`, `skills`, `job_skills`) and vector table (`job_embeddings`) indexed with `HNSW` (`vector_cosine_ops`).
*   **Migration (`src/database/migrate.py`)**: Automated porting of DuckDB data to PostgreSQL, generating 384-dimensional dense vectors for all job descriptions.
*   **Trend Analytics (`src/analytics/trends.py`)**: 
    *   **Skill Velocity Metric**: Calculates week-over-week frequency shifts to identify trending tools.
    *   **Co-occurrence Matrix**: Discovers complementary technologies based on industry hiring demand.

---

## Phase 4: Retrieval API & Agent Tool Integration
**Objective:** Expose live market intelligence as low-latency REST endpoints and standard agent tool hooks.
*   **DAG Scheduler (`src/engine/dag_scheduler.py`)**: Employs `NetworkX` to construct topological prerequisite DAGs and detect cyclic skill dependencies.
*   **FastAPI Gateway (`src/api/`)**:
    *   `schemas.py`: Pydantic request/response models for gap analysis, market trends, and semantic context retrieval.
    *   `main.py`: Endpoints (`/api/v1/market/trends`, `/api/v1/roadmaps/gap-analysis`, `/api/v1/context/retrieve-jobs`).
*   **Agent Tool Hooks (`src/agents/nexus_tools.py`)**: Exposes dual-callable tool wrappers (`analyze_student_skill_gap`, `fetch_real_world_context`, `query_market_trends`) supporting both plain Python calls and LangChain `.invoke()` with offline failover.

---

## Phase 5: Multi-Agent Orchestration & Socratic Pedagogy
**Objective:** Construct an autonomous closed-loop pedagogical state machine using **LangGraph** to dynamically tutor students.
*   **Learner State (`src/agents/state.py`)**: `LearnerState` TypedDict maintaining `student_id`, `target_role`, `verified_skills`, `active_trajectory`, `current_skill`, `chat_history`, and `mastery_score`.
*   **Prompts (`src/agents/prompts.py`)**: Formulates strict Socratic discovery rules (`SOCRATES_PROMPT`) and structured scoring instructions (`PROFILER_PROMPT`).
*   **Agent Nodes (`src/agents/nodes.py`)**:
    *   `Roadmap_Agent`: Re-aligns active learning trajectory against market DAGs.
    *   `Socrates_Agent`: Grounds inquiry in real-world company contexts and asks Socratic questions.
    *   `Cognitive_Profiler`: Assesses conceptual depth, calculates mastery score (0.0 to 1.0), and promotes mastered skills ($\ge 0.8$).
    *   **Dual LLM Engine**: Supports `ChatOpenAI(model="gpt-4o")` when `OPENAI_API_KEY` is present, with an intelligent local fallback simulator for zero-cost testing.
*   **StateGraph Orchestration (`src/agents/orchestrator.py`)**:
    *   Cyclic state machine compiled with `nexus_app`.
    *   `session_entry_router`: Inspects chat state to enter via `Cognitive_Profiler` on user reply or `Roadmap_Agent` on new session.
    *   `mastery_router`: Intelligently branches to `Roadmap_Agent` upon mastery or returns to `Socrates_Agent` for scaffolding hints.
    *   Halts cleanly after question generation to wait for human student input.
*   **Validation (`test_flow.py`)**: End-to-end test script demonstrating the complete loop across both remediation and mastery scenarios.

---

## How to Verify System Status

Run these commands from the repository root:

### 1. Verify DuckDB (Phases 1 & 2)
```bash
venv\Scripts\python.exe -c "import duckdb; con = duckdb.connect('data/nexus_jobs.duckdb'); print(con.execute('SHOW TABLES').fetchall())"
```
*Expected: `job_skills`, `raw_jobs`, `skills_taxonomy`*

### 2. Verify Docker & PostgreSQL (Phase 3)
```bash
docker ps
docker exec -it pgvector-container psql -U nexus -d nexus_db -c "SELECT title, company FROM jobs LIMIT 3;"
```

### 3. Verify Trend Analytics (Phase 3)
```bash
venv\Scripts\python.exe src\analytics\trends.py
```

### 4. Verify Phase 4 DAG Scheduler & Tools
```bash
venv\Scripts\python.exe test_agent.py
```

### 5. Verify Phase 5 Multi-Agent Orchestration
```bash
venv\Scripts\python.exe test_flow.py
```
*Expected: Displays Step 1 initial session, Scenario B remediation routing (score < 0.8), and Scenario C mastery progression (score $\ge 0.8$) advancing from `SQL` to `Docker`.*

### 6. Launch & Use the Interactive Web Application Tool
Run the dedicated web dashboard to visually experience the full multi-agent pedagogical loop:
```bash
venv\Scripts\python.exe run_app.py
```
* Open your browser at: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**
* **Features:**
  * **Interactive Socratic Chat:** Converse directly with Socrates Agent.
  * **Live Prerequisite DAG Visualizer:** Dynamic topological roadmap of skills updating in real time.
  * **Cognitive Profiler Dial:** Animated mastery confidence gauge with 80% advancement threshold.
  * **Market Intelligence Signals:** Live demand ratios and trending framework metrics.
  * **FastAPI Swagger API Documentation:** Available at **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**.

