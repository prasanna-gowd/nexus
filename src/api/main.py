"""
FastAPI Gateway Service for NEXUS Multi-Agent Learning Ecosystem.
"""
import os
import logging
from typing import Dict, Any
from pathlib import Path
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.api.schemas import (
    HealthResponse,
    GapAnalysisRequest,
    GapAnalysisResponse,
    MarketTrendsResponse,
    ContextRetrieveRequest,
    ContextRetrieveResponse,
    AgentChatRequest,
    AgentChatResponse,
)
from src.api.db import check_duckdb_health, check_postgres_health, retrieve_semantic_job_context
from src.engine.dag_scheduler import perform_skill_gap_analysis
from src.analytics.trends import MarketAnalytics

logger = logging.getLogger(__name__)

app = FastAPI(
    title="NEXUS Autonomous Multi-Agent Learning Platform",
    description="REST API Gateway connecting job market demand, prerequisite DAG scheduling, and adaptive multi-agent tutoring.",
    version="1.0.0"
)

# Enable CORS for local dev / frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
@app.get("/api/v1/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """System health and database readiness check."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        duckdb_connected=check_duckdb_health(),
        postgres_connected=check_postgres_health(),
    )


@app.get("/api/v1/market/trends", response_model=MarketTrendsResponse, tags=["Market Analytics"])
async def get_market_trends(limit: int = 15):
    """Returns top in-demand technical skills and week-over-week velocity metrics."""
    try:
        analytics = MarketAnalytics()
        top_skills = analytics.get_top_skills(limit=limit)
        velocity = analytics.get_skill_velocity()
        return MarketTrendsResponse(
            top_skills=top_skills,
            velocity_analysis=velocity,
            status="success"
        )
    except Exception as e:
        logger.error(f"Error querying market trends: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compute market trends: {str(e)}"
        )


@app.post("/api/v1/roadmaps/gap-analysis", response_model=GapAnalysisResponse, tags=["Skill-Gap Engine"])
async def analyze_gap(request: GapAnalysisRequest):
    """
    Performs skill gap analysis against job market demand and generates
    a topological prerequisite DAG learning trajectory.
    """
    try:
        result = perform_skill_gap_analysis(
            target_role=request.target_role,
            student_skills=request.student_skills,
            role_skills_override=request.role_skills_override
        )
        return GapAnalysisResponse(**result)
    except Exception as e:
        logger.error(f"Error computing gap analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gap analysis calculation failed: {str(e)}"
        )


@app.post("/api/v1/context/retrieve-jobs", response_model=ContextRetrieveResponse, tags=["Context Retrieval"])
async def retrieve_job_context(request: ContextRetrieveRequest):
    """
    Performs semantic vector retrieval via pgvector (or fallback)
    to find real-world company usages of a technical skill.
    """
    try:
        result = retrieve_semantic_job_context(
            skill_name=request.skill_name,
            limit=request.limit
        )
        return ContextRetrieveResponse(**result)
    except Exception as e:
        logger.error(f"Error retrieving job context: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve context: {str(e)}"
        )


@app.post("/api/v1/tutor/chat", response_model=AgentChatResponse, tags=["Multi-Agent Tutoring"])
async def tutor_chat(request: AgentChatRequest):
    """
    Interactive endpoint for the LangGraph multi-agent tutoring loop.
    Maintains persistent learner state across conversation turns.
    """
    try:
        from src.agents.orchestrator import run_agent_turn

        result = run_agent_turn(
            student_id=request.student_id,
            target_role=request.target_role,
            verified_skills=request.verified_skills,
            user_message=request.user_message,
            reset_session=request.reset_session
        )
        return AgentChatResponse(**result)
    except Exception as e:
        logger.error(f"Multi-agent execution error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Multi-agent tutoring turn failed: {str(e)}"
        )


# Mount static directory for frontend web UI
static_dir = Path(__file__).parent.parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_index():
        index_file = static_dir / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return {"message": "NEXUS API is running. Visit /docs for Swagger documentation."}
