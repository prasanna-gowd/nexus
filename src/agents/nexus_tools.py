"""
Standardized Agent Tool Wrappers for NEXUS Multi-Agent System.

Provides dual-callable interfaces:
1. Direct Python function invocation.
2. LangChain StructuredTool / .invoke() compatibility.
Includes reliable offline failover.
"""
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def analyze_student_skill_gap(target_role: str, student_skills: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Computes missing skills and topological prerequisite trajectory for a student.

    Args:
        target_role: Desired job title (e.g. 'Machine Learning Engineer').
        student_skills: List of verified skills the student already possesses.
    """
    if student_skills is None:
        student_skills = []
    from src.engine.dag_scheduler import perform_skill_gap_analysis
    return perform_skill_gap_analysis(target_role=target_role, student_skills=student_skills)


def fetch_real_world_context(skill_name: str, limit: int = 3) -> Dict[str, Any]:
    """
    Retrieves real-world job posting excerpts demonstrating how a skill is used in industry.

    Args:
        skill_name: Technical skill to look up (e.g. 'Docker', 'SQL').
        limit: Max number of context snippets.
    """
    from src.api.db import retrieve_semantic_job_context
    return retrieve_semantic_job_context(skill_name=skill_name, limit=limit)


def query_market_trends(limit: int = 15) -> Dict[str, Any]:
    """
    Queries current market demand rankings and trending technologies.
    """
    from src.analytics.trends import MarketAnalytics
    analytics = MarketAnalytics()
    top_skills = analytics.get_top_skills(limit=limit)
    velocity = analytics.get_skill_velocity()
    return {
        "top_skills": top_skills,
        "velocity": velocity
    }


# LangChain Tool wrappers if langchain is available
try:
    from langchain_core.tools import tool

    @tool
    def tool_analyze_student_skill_gap(target_role: str, student_skills: List[str]) -> str:
        """Analyze missing skills and compute prerequisite learning sequence for a target role."""
        import json
        res = analyze_student_skill_gap(target_role, student_skills)
        return json.dumps(res)

    @tool
    def tool_fetch_real_world_context(skill_name: str) -> str:
        """Fetch production job context and industry usage examples for a skill."""
        import json
        res = fetch_real_world_context(skill_name)
        return json.dumps(res)

    @tool
    def tool_query_market_trends() -> str:
        """Fetch current tech market skill demand rankings."""
        import json
        res = query_market_trends()
        return json.dumps(res)

except ImportError:
    tool_analyze_student_skill_gap = analyze_student_skill_gap
    tool_fetch_real_world_context = fetch_real_world_context
    tool_query_market_trends = query_market_trends


if __name__ == "__main__":
    print("Testing nexus_tools:")
    gap = analyze_student_skill_gap("Machine Learning Engineer", ["Python"])
    print("Gap analysis:", gap["match_percentage"], "% match, missing:", gap["missing_skills"])
    ctx = fetch_real_world_context("Docker")
    print("Context:", ctx["contexts"][0]["snippet"][:100])
