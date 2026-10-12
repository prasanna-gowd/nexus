"""
Unit & Integration Tests for Phase 5: Multi-Agent LangGraph Pedagogical Loop.
"""
import pytest

from src.agents.state import LearnerState
from src.agents.nodes import (
    roadmap_agent_node,
    socrates_agent_node,
    cognitive_profiler_node,
    MASTERY_THRESHOLD
)
from src.agents.orchestrator import run_agent_turn, mastery_router, build_nexus_graph


def test_roadmap_node_initial_trajectory():
    state: LearnerState = {
        "student_id": "TEST-1",
        "target_role": "Machine Learning Engineer",
        "verified_skills": ["Python"],
        "active_trajectory": [],
        "current_skill": "",
        "chat_history": [],
        "mastery_score": 0.0,
        "attempt_count": 0,
        "pending_response": False,
        "is_completed": False
    }

    result = roadmap_agent_node(state)
    assert len(result["active_trajectory"]) > 0
    assert result["current_skill"] != ""
    assert result["current_skill"] not in ["Python"]


def test_socrates_node_asks_question():
    state: LearnerState = {
        "student_id": "TEST-1",
        "target_role": "Machine Learning Engineer",
        "verified_skills": ["Python"],
        "active_trajectory": ["Docker"],
        "current_skill": "Docker",
        "chat_history": [],
        "mastery_score": 0.0,
        "attempt_count": 0,
        "pending_response": False,
        "is_completed": False
    }

    result = socrates_agent_node(state)
    assert len(result["chat_history"]) == 1
    assert result["chat_history"][0]["role"] == "agent"
    assert len(result["chat_history"][0]["content"]) > 20
    assert result["pending_response"] is True


def test_cognitive_profiler_low_score_remediation():
    state: LearnerState = {
        "student_id": "TEST-1",
        "target_role": "Machine Learning Engineer",
        "verified_skills": ["Python"],
        "active_trajectory": ["Docker"],
        "current_skill": "Docker",
        "chat_history": [
            {"role": "agent", "content": "Why is multi-stage Dockerfile important?"},
            {"role": "user", "content": "I don't know, maybe to build it faster."}
        ],
        "mastery_score": 0.0,
        "attempt_count": 0,
        "pending_response": True,
        "is_completed": False
    }

    result = cognitive_profiler_node(state)
    assert result["mastery_score"] < MASTERY_THRESHOLD
    assert "Docker" not in result.get("verified_skills", [])
    assert result["attempt_count"] == 1


def test_cognitive_profiler_high_score_mastery():
    state: LearnerState = {
        "student_id": "TEST-1",
        "target_role": "Machine Learning Engineer",
        "verified_skills": ["Python"],
        "active_trajectory": ["Docker", "Kubernetes"],
        "current_skill": "Docker",
        "chat_history": [
            {"role": "agent", "content": "Why is multi-stage Dockerfile important?"},
            {"role": "user", "content": "Multi-stage builds allow compiling binaries in a build container and copying only the artifacts into a minimal distroless image to minimize layer size and reduce attack surface."}
        ],
        "mastery_score": 0.0,
        "attempt_count": 0,
        "pending_response": True,
        "is_completed": False
    }

    result = cognitive_profiler_node(state)
    assert result["mastery_score"] >= MASTERY_THRESHOLD
    assert "Docker" in result["verified_skills"]
    assert "Docker" not in result["active_trajectory"]


def test_mastery_router():
    state_remediation: LearnerState = {
        "student_id": "T", "target_role": "R", "verified_skills": [],
        "active_trajectory": ["Docker"], "current_skill": "Docker",
        "chat_history": [], "mastery_score": 0.4, "attempt_count": 1,
        "pending_response": False, "is_completed": False
    }
    assert mastery_router(state_remediation) == "Socrates_Agent"

    state_mastery: LearnerState = {
        "student_id": "T", "target_role": "R", "verified_skills": ["Docker"],
        "active_trajectory": ["Kubernetes"], "current_skill": "Docker",
        "chat_history": [], "mastery_score": 0.85, "attempt_count": 0,
        "pending_response": False, "is_completed": False
    }
    assert mastery_router(state_mastery) == "Roadmap_Agent"


def test_multi_turn_session_persistence():
    student_id = "SESSION-TEST-001"
    # Turn 1: Init
    turn1 = run_agent_turn(student_id, "Backend Developer", ["Python"], reset_session=True)
    assert turn1["current_skill"] != ""

    # Turn 2: Remediation answer
    turn2 = run_agent_turn(student_id, user_message="idk how to do that")
    assert turn2["mastery_score"] < MASTERY_THRESHOLD
    assert turn2["current_skill"] == turn1["current_skill"]

    # Turn 3: Competent answer
    turn3 = run_agent_turn(
        student_id,
        user_message="We optimize queries by analyzing the execution plan with explain analyze and adding b-tree indexes on foreign keys."
    )
    assert turn3["mastery_score"] >= 0.8
