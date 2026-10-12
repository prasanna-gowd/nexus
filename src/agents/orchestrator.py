"""
LangGraph Multi-Agent Orchestrator for NEXUS.

Assembles Roadmap Agent, Socrates Agent, and Cognitive Profiler
into a closed-loop pedagogical state machine with session persistence.
"""
import logging
from typing import Dict, Any, Optional

from langgraph.graph import StateGraph, END
from src.agents.state import LearnerState
from src.agents.nodes import (
    roadmap_agent_node,
    socrates_agent_node,
    cognitive_profiler_node,
    MASTERY_THRESHOLD
)
from src.api.db import save_session, load_session, record_learning_event

logger = logging.getLogger(__name__)

# In-memory session store mapping student_id -> LearnerState
SESSION_STORE: Dict[str, LearnerState] = {}


def mastery_router(state: LearnerState) -> str:
    """
    Conditional routing edge after Cognitive Profiler evaluates student response:
    - If student has completed all skills -> END
    - If mastery score >= threshold -> Roadmap_Agent (advance to next skill)
    - If mastery score < threshold -> Socrates_Agent (provide remediation hint)
    """
    if state.get("is_completed") or not state.get("current_skill"):
        return END

    score = state.get("mastery_score", 0.0)
    if score >= MASTERY_THRESHOLD:
        logger.info(f"Routing to Roadmap_Agent: mastery score {score} >= {MASTERY_THRESHOLD}")
        return "Roadmap_Agent"
    else:
        logger.info(f"Routing to Socrates_Agent: mastery score {score} < {MASTERY_THRESHOLD}")
        return "Socrates_Agent"


# --- 1. Assemble Full StateGraph ---
def build_nexus_graph() -> StateGraph:
    workflow = StateGraph(LearnerState)

    workflow.add_node("Roadmap_Agent", roadmap_agent_node)
    workflow.add_node("Socrates_Agent", socrates_agent_node)
    workflow.add_node("Cognitive_Profiler", cognitive_profiler_node)

    # Entry point
    workflow.set_entry_point("Roadmap_Agent")
    workflow.add_edge("Roadmap_Agent", "Socrates_Agent")
    workflow.add_edge("Socrates_Agent", END) # Halts cleanly after asking question to wait for student!

    # Profiler evaluates student input, then branches conditionally
    workflow.add_conditional_edges(
        "Cognitive_Profiler",
        mastery_router,
        {
            "Roadmap_Agent": "Roadmap_Agent",
            "Socrates_Agent": "Socrates_Agent",
            END: END
        }
    )

    return workflow


# Compile the graph
nexus_app = build_nexus_graph().compile()


# --- 2. Multi-Turn Session Orchestration ---
def run_agent_turn(
    student_id: str,
    target_role: str = "Machine Learning Engineer",
    verified_skills: Optional[list] = None,
    user_message: Optional[str] = None,
    reset_session: bool = False
) -> Dict[str, Any]:
    """
    Orchestrates one interactive turn in the multi-agent tutoring loop.
    Persists learner state across API requests.

    Args:
        student_id: Unique learner identifier.
        target_role: Target career goal (e.g. 'Machine Learning Engineer').
        verified_skills: Initial skills known by the student.
        user_message: Student's answer to the previous Socratic question.
        reset_session: Force a fresh learning session.
    """
    if verified_skills is None:
        verified_skills = []

    # 1. Retrieve or initialize session state
    if reset_session:
        logger.info(f"Initializing new learning session for {student_id} (Role: {target_role})")
        state: LearnerState = {
            "student_id": student_id,
            "target_role": target_role,
            "verified_skills": list(verified_skills),
            "active_trajectory": [],
            "current_skill": "",
            "chat_history": [],
            "mastery_score": 0.0,
            "attempt_count": 0,
            "question_index": 0,
            "pending_response": False,
            "is_completed": False
        }
    elif student_id in SESSION_STORE:
        state = SESSION_STORE[student_id]
        # Allow updating verified skills or role if supplied
        if target_role and target_role != state.get("target_role"):
            state["target_role"] = target_role
        for s in verified_skills:
            if s not in state.get("verified_skills", []):
                state["verified_skills"].append(s)
    elif (db_session := load_session(student_id)):
        logger.info(f"Restored session for {student_id} from persistent DuckDB storage")
        state = db_session
        SESSION_STORE[student_id] = state
        if target_role and target_role != state.get("target_role"):
            state["target_role"] = target_role
        for s in verified_skills:
            if s not in state.get("verified_skills", []):
                state["verified_skills"].append(s)
    else:
        logger.info(f"Initializing new learning session for {student_id} (Role: {target_role})")
        state: LearnerState = {
            "student_id": student_id,
            "target_role": target_role,
            "verified_skills": list(verified_skills),
            "active_trajectory": [],
            "current_skill": "",
            "chat_history": [],
            "mastery_score": 0.0,
            "attempt_count": 0,
            "question_index": 0,
            "pending_response": False,
            "is_completed": False
        }

    # 2. Process Turn
    if user_message and user_message.strip():
        # Ensure active trajectory and current_skill are initialized if empty
        if not state.get("current_skill"):
            roadmap_updates = roadmap_agent_node(state)
            state.update(roadmap_updates)

        # Human provided a response -> Enter via Cognitive Profiler
        logger.info(f"Processing student message: '{user_message[:60]}...'")
        state["chat_history"].append({"role": "user", "content": user_message.strip()})
        
        # Step A: Evaluate response
        profiler_updates = cognitive_profiler_node(state)
        eval_score = profiler_updates.get("mastery_score", 0.0)
        state.update(profiler_updates)

        # Step A.1: Record learning event for progress analytics
        current_skill = state.get("current_skill", "")
        if current_skill:
            event_type = "mastery_verified" if eval_score >= MASTERY_THRESHOLD else "evaluation_attempt"
            record_learning_event(student_id, current_skill, eval_score, event_type)

        # Step B: Route conditionally based on evaluation
        next_node = mastery_router(state)
        if next_node == "Roadmap_Agent":
            roadmap_updates = roadmap_agent_node(state)
            state.update(roadmap_updates)
            socrates_updates = socrates_agent_node(state)
            state.update(socrates_updates)
        elif next_node == "Socrates_Agent":
            socrates_updates = socrates_agent_node(state)
            state.update(socrates_updates)

    else:
        eval_score = state.get("mastery_score", 0.0)
        # Initial invocation or resumption without message -> Roadmap -> Socrates
        if not state.get("current_skill") or not state.get("chat_history"):
            roadmap_updates = roadmap_agent_node(state)
            state.update(roadmap_updates)
            socrates_updates = socrates_agent_node(state)
            state.update(socrates_updates)

    # 3. Persist updated session
    SESSION_STORE[student_id] = state
    save_session(student_id, state)

    # Extract latest agent message
    latest_agent_msg = ""
    for msg in reversed(state.get("chat_history", [])):
        if msg["role"] == "agent":
            latest_agent_msg = msg["content"]
            break

    return {
        "student_id": student_id,
        "current_skill": state.get("current_skill", ""),
        "agent_message": latest_agent_msg,
        "mastery_score": eval_score,
        "verified_skills": state.get("verified_skills", []),
        "active_trajectory": state.get("active_trajectory", []),
        "is_completed": state.get("is_completed", False),
        "chat_history": state.get("chat_history", []),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing NEXUS Multi-Agent Orchestrator...")
    res1 = run_agent_turn("TEST-STU", "Machine Learning Engineer", ["Python"], reset_session=True)
    print(f"Agent Asked: {res1['agent_message'][:120]}...")
    print(f"Current Skill: {res1['current_skill']}")
