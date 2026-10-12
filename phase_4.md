# NEXUS Phase 4: Retrieval API & Multi-Agent Integration Implementation Plan

## 1. Overview & Scope
This plan guides the implementation of **Phase 4** of the NEXUS pipeline. Phases 1–3 established raw ingestion, NLP skill extraction/taxonomy alignment, and PostgreSQL/pgvector storage with velocity analytics. Phase 4 develops the intelligence integration layer:
1. Converting relational skill co-occurrences and dependencies into an actionable **Topological Prerequisite DAG (Directed Acyclic Graph)** using `NetworkX`.
2. Constructing a low-latency **FastAPI Gateway Service** exposing endpoints tailored for agent tool use.
3. Defining **Agent Tool Hooks** compatible with agent frameworks (e.g., LangGraph, CrewAI, AutoGen) for direct consumption by the **Roadmap Agent** and **Socrates Agent**.

---

## 2. Directory Structure & New Files
The following files must be created in the repository:

```text
nexus-market-pipeline/
├── src/
│   ├── engine/
│   │   ├── __init__.py
│   │   └── dag_scheduler.py       # Prerequisite DAG generation & topological ordering
│   ├── api/
│   │   ├── __init__.py
│   │   ├── schemas.py             # Pydantic request/response models
│   │   ├── db.py                  # PostgreSQL & pgvector async session management
│   │   └── main.py                # FastAPI REST endpoints for agent access
│   └── agents/
│       ├── __init__.py
│       └── nexus_tools.py         # Standardized agent tool wrappers for LangGraph/CrewAI
├── tests/
│   ├── test_dag_scheduler.py      # Unit tests for cycle detection and path generation
│   └── test_api.py                # Integration tests for FastAPI endpoints
└── requirements_phase4.txt

requirements_phase4- download these mentioned below
fastapi>=0.110.0
uvicorn[standard]>=0.28.0
pydantic>=2.6.0
networkx>=3.2.1
asyncpg>=0.29.0
sqlalchemy>=2.0.28
pgvector>=0.2.5
sentence-transformers>=2.5.1
httpx>=0.27.0
pytest>=8.1.0
pytest-asyncio>=0.23.0
```