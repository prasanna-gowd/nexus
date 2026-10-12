"""
Learner State schema for NEXUS multi-agent system.
"""
from typing import TypedDict, List, Dict, Any, Optional


class LearnerState(TypedDict):
    """
    Shared state maintaining the student's dynamic learning trajectory,
    active dialogue, cognitive evaluations, and knowledge tracing.
    """
    student_id: str
    target_role: str
    verified_skills: List[str]
    active_trajectory: List[str]      # Ordered DAG sequence of remaining skills
    current_skill: str                # Skill currently being taught (s_k)
    chat_history: List[Dict[str, str]] # [{'role': 'user'|'agent', 'content': str}]
    mastery_score: float              # 0.0 to 1.0 confidence from profiler
    attempt_count: int                # Number of attempts for current skill
    pending_response: bool            # True if waiting for student response
    is_completed: bool                # True if all skills in trajectory are mastered
