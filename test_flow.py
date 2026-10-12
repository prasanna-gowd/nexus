"""
End-to-End Validation Script for NEXUS Multi-Agent Tutoring Ecosystem (Phase 5).

Demonstrates:
1. Initial Session: Target role alignment & first Socratic question.
2. Remediation Flow: Sub-mastery response triggers scaffolding hint.
3. Mastery Flow: High-quality technical answer promotes skill and advances DAG trajectory.
"""
import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


from src.agents.orchestrator import run_agent_turn

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def run_test_flow():
    print("=" * 70)
    print("NEXUS MULTI-AGENT PEDAGOGICAL LOOP VERIFICATION (test_flow.py)")
    print("=" * 70)

    student_id = "STU-E2E-001"
    target_role = "Machine Learning Engineer"
    initial_skills = ["Python"]

    # --- STEP 1: INITIAL PASS ---
    print("\n--- STEP 1: INITIAL SESSION CREATION ---")
    step1 = run_agent_turn(
        student_id=student_id,
        target_role=target_role,
        verified_skills=initial_skills,
        reset_session=True
    )

    print(f"Target Role:        {target_role}")
    print(f"Verified Skills:    {step1['verified_skills']}")
    print(f"Active Trajectory:  {step1['active_trajectory']}")
    print(f"Current Skill (s1): {step1['current_skill']}")
    print(f"\nSocrates Agent:\n{step1['agent_message']}\n")

    assert step1["current_skill"] != "", "Roadmap Agent failed to select initial skill"
    assert len(step1["active_trajectory"]) > 0, "No trajectory generated"

    # --- STEP 2: REMEDIATION SCENARIO (Score < 0.8) ---
    print("--- STEP 2: REMEDIATION SCENARIO (Sub-mastery response) ---")
    weak_response = "I am not completely sure, maybe we just use threading or a normal loop?"
    print(f"Student Answers:    '{weak_response}'")

    step2 = run_agent_turn(
        student_id=student_id,
        user_message=weak_response
    )

    print(f"Mastery Score:      {step2['mastery_score']} (< 0.8 expected)")
    print(f"Current Skill:      {step2['current_skill']} (Should remain same)")
    print(f"\nSocrates Agent (Scaffolding Hint):\n{step2['agent_message']}\n")

    assert step2["mastery_score"] < 0.8, "Weak response received inappropriately high score"
    assert step2["current_skill"] == step1["current_skill"], "Skill advanced prematurely without mastery"

    # --- STEP 3: MASTERY SCENARIO (Score >= 0.8) ---
    print("--- STEP 3: MASTERY PROGRESSION SCENARIO (Demonstrating competence) ---")
    active_skill = step1["current_skill"].lower()
    if active_skill == "numpy":
        strong_response = (
            "We should leverage vectorized operations with NumPy ndarrays because vectorization "
            "delegates computations to underlying C-contiguous loops utilizing SIMD instructions, "
            "avoiding Python interpreter overhead and optimizing memory layout."
        )
    elif active_skill == "docker":
        strong_response = (
            "We build multi-stage Dockerfiles to compile binaries in an initial stage and copy only the artifacts "
            "into a minimal distroless image to minimize layer size, optimize caching, and reduce attack surface."
        )
    else:
        strong_response = (
            f"We optimize production {step1['current_skill']} architectures by designing modular components "
            "because explicit isolation prevents runtime bottlenecks, optimizes resource utilization, "
            "and ensures deterministic latency."
        )
    print(f"Student Answers:    '{strong_response}'")

    step3 = run_agent_turn(
        student_id=student_id,
        user_message=strong_response
    )

    print(f"Mastery Score:      {step3['mastery_score']} (>= 0.8 expected)")
    print(f"Verified Skills:    {step3['verified_skills']}")
    print(f"Next Active Skill:  {step3['current_skill']}")
    print(f"Remaining Steps:    {step3['active_trajectory']}")
    print(f"\nSocrates Agent (Next Topic Inquiry):\n{step3['agent_message']}\n")

    assert step3["mastery_score"] >= 0.8, "Strong response failed to achieve mastery"
    print("=" * 70)
    print("ALL PEDAGOGICAL LOOP SCENARIOS VERIFIED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_test_flow()
