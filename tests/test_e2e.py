"""
End-to-End Application Workflow Tests.
"""
import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_complete_student_workflow_e2e():
    """
    Simulates a full student user session:
    1. Checks system health.
    2. Explores market trends.
    3. Performs skill gap analysis for 'DevOps Engineer'.
    4. Retrieves real-world context for missing skill.
    5. Starts tutoring session.
    6. Submits struggling answer (receives scaffolding hint).
    7. Submits proficient answer (demonstrates mastery and advances).
    """
    # 1. Health
    res_health = client.get("/health")
    assert res_health.status_code == 200

    # 2. Market Trends
    res_trends = client.get("/api/v1/market/trends?limit=5")
    assert res_trends.status_code == 200

    # 3. Gap Analysis
    student_id = "E2E-STUDENT-WORKFLOW"
    res_gap = client.post("/api/v1/roadmaps/gap-analysis", json={
        "target_role": "DevOps Engineer",
        "student_skills": ["Linux", "Git"],
        "role_skills_override": ["Linux", "Git", "Docker", "Kubernetes", "CI/CD"]
    })
    assert res_gap.status_code == 200
    gap_data = res_gap.json()
    assert "Linux" in gap_data["matched_skills"]
    assert len(gap_data["ordered_trajectory"]) > 0

    first_needed_skill = gap_data["ordered_trajectory"][0]["skill_name"]

    # 4. Context Retrieval
    res_ctx = client.post("/api/v1/context/retrieve-jobs", json={
        "skill_name": first_needed_skill,
        "limit": 1
    })
    assert res_ctx.status_code == 200

    # 5. Start Tutoring Session
    res_tutor_start = client.post("/api/v1/tutor/chat", json={
        "student_id": student_id,
        "target_role": "DevOps Engineer",
        "verified_skills": ["Linux", "Git"],
        "reset_session": True
    })
    assert res_tutor_start.status_code == 200
    tutor_data = res_tutor_start.json()
    assert tutor_data["current_skill"] != ""
    assert len(tutor_data["agent_message"]) > 0

    # 6. Struggle / Partial Answer
    res_tutor_weak = client.post("/api/v1/tutor/chat", json={
        "student_id": student_id,
        "user_message": "not sure"
    })
    assert res_tutor_weak.status_code == 200
    assert res_tutor_weak.json()["mastery_score"] < 0.8

    # 7. Mastery Answer
    res_tutor_strong = client.post("/api/v1/tutor/chat", json={
        "student_id": student_id,
        "user_message": "We containerize using multi-stage builds to optimize image layer caching and minimize the attack surface by deploying non-root distroless runtime containers."
    })
    assert res_tutor_strong.status_code == 200
    assert res_tutor_strong.json()["mastery_score"] >= 0.8
