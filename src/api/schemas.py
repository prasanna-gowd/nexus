"""
Pydantic Request and Response Schemas for NEXUS API.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


# --- Health & Readiness ---
class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    duckdb_connected: bool
    postgres_connected: bool


# --- Skill Gap Analysis ---
class GapAnalysisRequest(BaseModel):
    target_role: str = Field(..., description="Target job title")
    student_skills: List[str] = Field(default_factory=list, description="Verified student skills")
    role_skills_override: Optional[List[str]] = None


class TrajectoryStep(BaseModel):
    step: int
    skill_name: str
    prerequisites: List[str] = Field(default_factory=list)
    is_target_skill: bool = True
    status: str = "pending"


class GapAnalysisResponse(BaseModel):
    target_role: str
    match_percentage: float
    verified_skills: List[str]
    matched_skills: List[str]
    missing_skills: List[str]
    total_required: int
    ordered_trajectory: List[TrajectoryStep]


# --- Market Trends ---
class SkillDemandItem(BaseModel):
    skill_name: str
    category: str
    demand_count: int


class MarketTrendsResponse(BaseModel):
    top_skills: List[SkillDemandItem]
    velocity_analysis: Dict[str, Any]
    status: str = "success"


# --- Semantic Context Retrieval ---
class ContextRetrieveRequest(BaseModel):
    skill_name: str = Field(..., description="Technical skill name")
    limit: int = Field(default=3, ge=1, le=10)


class ContextSnippet(BaseModel):
    job_title: str
    company: str
    snippet: str
    score: Optional[float] = None


class ContextRetrieveResponse(BaseModel):
    skill_name: str
    contexts: List[ContextSnippet]
    is_fallback: bool = False


# --- Multi-Agent Tutoring Dialogue ---
class AgentChatRequest(BaseModel):
    student_id: str = Field(default="STU-001")
    target_role: str = Field(default="Machine Learning Engineer")
    verified_skills: List[str] = Field(default_factory=list)
    user_message: Optional[str] = None
    reset_session: bool = False


class AgentChatResponse(BaseModel):
    student_id: str
    current_skill: str
    agent_message: str
    mastery_score: float
    verified_skills: List[str]
    active_trajectory: List[str]
    is_completed: bool = False
    chat_history: List[Dict[str, str]]
