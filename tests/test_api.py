"""
Integration Tests for FastAPI Gateway Service (Phase 4).
"""
import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "duckdb_connected" in data
    assert "postgres_connected" in data


def test_market_trends_endpoint():
    response = client.get("/api/v1/market/trends?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert "top_skills" in data
    assert "velocity_analysis" in data
    assert data["status"] == "success"


def test_gap_analysis_endpoint_valid():
    payload = {
        "target_role": "Machine Learning Engineer",
        "student_skills": ["Python", "SQL"],
        "role_skills_override": ["Python", "SQL", "PyTorch", "Docker"]
    }
    response = client.post("/api/v1/roadmaps/gap-analysis", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["target_role"] == "Machine Learning Engineer"
    assert "Python" in data["matched_skills"]
    assert "SQL" in data["matched_skills"]
    assert len(data["ordered_trajectory"]) > 0
    assert "step" in data["ordered_trajectory"][0]


def test_gap_analysis_endpoint_invalid():
    # Missing required target_role
    response = client.post("/api/v1/roadmaps/gap-analysis", json={})
    assert response.status_code == 422


def test_context_retrieval_endpoint():
    payload = {
        "skill_name": "Docker",
        "limit": 2
    }
    response = client.post("/api/v1/context/retrieve-jobs", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["skill_name"] == "Docker"
    assert len(data["contexts"]) > 0
    assert "snippet" in data["contexts"][0]


def test_tutor_chat_endpoint():
    payload = {
        "student_id": "TEST-API-STU",
        "target_role": "Frontend Developer",
        "verified_skills": ["HTML/CSS"],
        "reset_session": True
    }
    response = client.post("/api/v1/tutor/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["student_id"] == "TEST-API-STU"
    assert len(data["agent_message"]) > 0
    assert data["current_skill"] != ""
