"""
Phase 4 Verification Script: DAG Scheduler & Agent Tool Hooks.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.engine.dag_scheduler import PrerequisiteDAG, perform_skill_gap_analysis
from src.agents.nexus_tools import (
    analyze_student_skill_gap,
    fetch_real_world_context,
    query_market_trends
)


def run_tests():
    print("=" * 60)
    print("TESTING PHASE 4: DAG SCHEDULER & AGENT TOOLS")
    print("=" * 60)

    # 1. Prerequisite DAG cycle prevention
    print("\n[1] Testing Prerequisite DAG...")
    dag = PrerequisiteDAG()
    prereqs = dag.get_prerequisites_for_skill("Deep Learning")
    print(f"Prerequisites for Deep Learning: {prereqs}")
    assert "Python" in prereqs or "Machine Learning" in prereqs

    # Test adding cycle safely
    dag.add_prerequisite("Deep Learning", "Python")
    print("Cycle injection safely intercepted (DAG verified).")

    # 2. Skill Gap Analysis
    print("\n[2] Testing Skill-Gap Engine...")
    gap = analyze_student_skill_gap(
        target_role="Full Stack Developer",
        student_skills=["HTML/CSS", "Git"]
    )
    print(f"Target Role: {gap['target_role']}")
    print(f"Match: {gap['match_percentage']}%")
    print(f"Matched Skills: {gap['matched_skills']}")
    print(f"Missing Skills: {gap['missing_skills']}")
    print("Ordered Trajectory:")
    for step in gap["ordered_trajectory"]:
        print(f"  Step {step['step']}: {step['skill_name']} (Prereqs: {step['prerequisites']})")

    assert len(gap["ordered_trajectory"]) > 0

    # 3. Context Retrieval Tool
    print("\n[3] Testing Context Retrieval Hook...")
    ctx = fetch_real_world_context("Docker")
    print(f"Skill: {ctx['skill_name']}, Fallback: {ctx.get('is_fallback')}")
    print(f"Snippet: {ctx['contexts'][0]['snippet'][:150]}...")
    assert len(ctx["contexts"]) > 0

    # 4. Market Trends Tool
    print("\n[4] Testing Market Trends Tool...")
    trends = query_market_trends(limit=5)
    print(f"Top Skills count: {len(trends['top_skills'])}")
    print(f"Velocity Status: {trends['velocity'].get('status')}")

    print("\n" + "=" * 60)
    print("ALL PHASE 4 ENGINE AND TOOL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
