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
    JobPostAnalyzeRequest,
    JobPostAnalyzeResponse,
    SurveyQuestion,
    SurveyGenerateRequest,
    SurveyGenerateResponse,
    SurveyEvaluateRequest,
    SurveyEvaluateResponse,
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


@app.post("/api/v1/roadmaps/analyze-job-post", response_model=JobPostAnalyzeResponse, tags=["Skill-Gap Engine"])
async def analyze_job_post(request: JobPostAnalyzeRequest):
    """
    Parses raw job posting text in real time, extracts technical skills,
    compares against verified student skills, and generates the prerequisite DAG.
    """
    try:
        from src.nlp.cleaner import clean_description
        from src.nlp.extractor import extract_skills

        cleaned = clean_description(request.job_description)
        extracted_entities = extract_skills(cleaned, use_gliner=False)
        extracted_names = []
        for e in extracted_entities:
            s_name = e.get("skill", "").strip()
            if s_name and s_name not in extracted_names:
                extracted_names.append(s_name)

        if not extracted_names:
            extracted_names = ["Python", "SQL", "Git", "Docker"]

        result = perform_skill_gap_analysis(
            target_role=request.job_title or "Target Job Posting",
            student_skills=request.student_skills,
            role_skills_override=extracted_names
        )

        return JobPostAnalyzeResponse(
            job_title=request.job_title or "Target Job Posting",
            extracted_skills=extracted_names,
            match_percentage=result["match_percentage"],
            matched_skills=result["matched_skills"],
            missing_skills=result["missing_skills"],
            total_required=result["total_required"],
            ordered_trajectory=result["ordered_trajectory"]
        )
    except Exception as e:
        logger.error(f"Error analyzing job posting: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze job posting: {str(e)}"
        )


@app.post("/api/v1/survey/generate", response_model=SurveyGenerateResponse, tags=["Diagnostic Survey"])
async def generate_survey(request: SurveyGenerateRequest):
    """
    Generates a diagnostic skill survey assessing baseline competencies
    for the target role's key prerequisite technologies.
    """
    try:
        from src.analytics.trends import MarketAnalytics
        analytics = MarketAnalytics()
        
        target_skills = request.skills or []
        if not target_skills:
            role_demand = analytics.get_skills_by_role(request.target_role, limit=8)
            if role_demand:
                target_skills = [item["skill_name"] for item in role_demand]
            else:
                role_defaults = {
                    "machine learning": ["Python", "SQL", "Pandas", "NumPy", "scikit-learn", "PyTorch", "Docker"],
                    "data engineer": ["Python", "SQL", "PostgreSQL", "Docker", "Apache Kafka", "Linux"],
                    "frontend": ["HTML/CSS", "JavaScript", "TypeScript", "React", "Next.js", "Git"],
                    "backend": ["Python", "PostgreSQL", "Docker", "FastAPI", "Redis", "Linux"],
                    "full stack": ["JavaScript", "Python", "React", "PostgreSQL", "Docker", "Git"],
                    "devops": ["Linux", "Git", "Docker", "Kubernetes", "AWS", "CI/CD"],
                }
                matched_key = next((k for k in role_defaults if k in request.target_role.lower()), None)
                target_skills = role_defaults.get(matched_key, ["Python", "SQL", "Docker", "Git", "React"])

        survey_questions = []
        options = [
            "Level 1: No exposure / Just beginning",
            "Level 2: Basic conceptual familiarity (syntax/tutorials)",
            "Level 3: Intermediate hands-on project experience",
            "Level 4: Production mastery (architectural trade-offs & debugging)"
        ]

        for skill in target_skills[:8]:
            survey_questions.append(SurveyQuestion(
                skill_name=skill,
                question=f"How would you rate your hands-on production engineering competence in **{skill}**?",
                options=options
            ))

        return SurveyGenerateResponse(
            target_role=request.target_role,
            questions=survey_questions
        )
    except Exception as e:
        logger.error(f"Error generating survey: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Survey generation failed: {str(e)}"
        )


@app.post("/api/v1/survey/evaluate", response_model=SurveyEvaluateResponse, tags=["Diagnostic Survey"])
async def evaluate_survey(request: SurveyEvaluateRequest):
    """
    Evaluates the student's diagnostic survey answers, establishes baseline
    verified competencies, and constructs the personalized learning map.
    """
    try:
        from src.agents.orchestrator import SESSION_STORE, LearnerState

        verified_skills = []
        for skill, answer in request.answers.items():
            ans_clean = answer.lower()
            if "intermediate" in ans_clean or "production" in ans_clean or "level 3" in ans_clean or "level 4" in ans_clean:
                if skill not in verified_skills:
                    verified_skills.append(skill)

        # Re-compute skill gap and DAG
        gap_res = perform_skill_gap_analysis(
            target_role=request.target_role,
            student_skills=verified_skills
        )

        # Update session store
        state: LearnerState = {
            "student_id": request.student_id,
            "target_role": request.target_role,
            "verified_skills": verified_skills,
            "active_trajectory": [step["skill_name"] for step in gap_res["ordered_trajectory"]],
            "current_skill": gap_res["ordered_trajectory"][0]["skill_name"] if gap_res["ordered_trajectory"] else "",
            "chat_history": [{
                "role": "agent",
                "content": f"📋 **Diagnostic Survey Completed!**\n\nBased on your assessment, you have verified proficiency in: {', '.join(verified_skills) if verified_skills else 'None yet'}.\n\nYour personalized prerequisite DAG has been mapped out. Your next learning milestone is **{gap_res['ordered_trajectory'][0]['skill_name'] if gap_res['ordered_trajectory'] else 'None'}**."
            }],
            "mastery_score": 0.0,
            "attempt_count": 0,
            "question_index": 0,
            "pending_response": False,
            "is_completed": len(gap_res["ordered_trajectory"]) == 0
        }
        SESSION_STORE[request.student_id] = state

        return SurveyEvaluateResponse(
            student_id=request.student_id,
            target_role=request.target_role,
            verified_skills=verified_skills,
            match_percentage=gap_res["match_percentage"],
            missing_skills=gap_res["missing_skills"],
            ordered_trajectory=gap_res["ordered_trajectory"],
            initial_feedback=f"Identified {len(verified_skills)} verified skills and {len(gap_res['missing_skills'])} prerequisite learning objectives."
        )
    except Exception as e:
        logger.error(f"Error evaluating survey: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Survey evaluation failed: {str(e)}"
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
