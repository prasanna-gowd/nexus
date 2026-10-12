"""
Unit Tests for Phase 4: Topological Prerequisite DAG Scheduler & Skill Gap Engine.
"""
import pytest
import networkx as nx

from src.engine.dag_scheduler import PrerequisiteDAG, perform_skill_gap_analysis


def test_dag_initialization_acyclic():
    dag = PrerequisiteDAG()
    assert nx.is_directed_acyclic_graph(dag.graph), "Curated base graph must be acyclic"


def test_cycle_detection_and_rejection():
    dag = PrerequisiteDAG()
    # Add valid sequence A -> B -> C
    dag.add_prerequisite("Concept_A", "Concept_B")
    dag.add_prerequisite("Concept_B", "Concept_C")
    assert nx.is_directed_acyclic_graph(dag.graph)

    # Attempt to inject cycle C -> A
    dag.add_prerequisite("Concept_C", "Concept_A")
    assert nx.is_directed_acyclic_graph(dag.graph), "Injected cycle must be detected and prevented"


def test_topological_ancestor_order():
    dag = PrerequisiteDAG()
    prereqs = dag.get_prerequisites_for_skill("Deep Learning")
    assert "Python" in prereqs or "Machine Learning" in prereqs


def test_learning_trajectory_generation():
    dag = PrerequisiteDAG()
    required = ["PyTorch", "Docker"]
    verified = ["Python", "Linux"]

    trajectory = dag.compute_learning_trajectory(required, verified)
    assert len(trajectory) > 0

    skills_in_order = [t["skill_name"] for t in trajectory]
    assert "PyTorch" in skills_in_order
    assert "Docker" in skills_in_order

    # If Deep Learning is a prerequisite for PyTorch, it must appear before PyTorch
    if "Deep Learning" in skills_in_order and "PyTorch" in skills_in_order:
        dl_idx = skills_in_order.index("Deep Learning")
        pt_idx = skills_in_order.index("PyTorch")
        assert dl_idx < pt_idx, "Prerequisite 'Deep Learning' must precede 'PyTorch'"


def test_skill_gap_analysis_calculation():
    result = perform_skill_gap_analysis(
        target_role="Machine Learning Engineer",
        student_skills=["Python"]
    )
    assert result["target_role"] == "Machine Learning Engineer"
    assert "Python" in result["matched_skills"]
    assert len(result["missing_skills"]) > 0
    assert len(result["ordered_trajectory"]) > 0
    assert 0 <= result["match_percentage"] <= 100


def test_empty_student_skills():
    result = perform_skill_gap_analysis(
        target_role="Full Stack Developer",
        student_skills=[]
    )
    assert result["match_percentage"] == 0.0
    assert len(result["matched_skills"]) == 0
    assert len(result["missing_skills"]) == result["total_required"]
    assert len(result["ordered_trajectory"]) > 0


def test_completed_roadmap_100_percent():
    full_stack_skills = [
        "HTML/CSS", "JavaScript", "TypeScript", "React", "Node.js",
        "REST API", "SQL", "PostgreSQL", "Git", "Docker"
    ]
    result = perform_skill_gap_analysis(
        target_role="Full Stack Developer",
        student_skills=full_stack_skills
    )
    assert result["match_percentage"] == 100.0
    assert len(result["missing_skills"]) == 0
    assert len(result["matched_skills"]) == result["total_required"]
    assert len(result["ordered_trajectory"]) == 0


def test_duplicate_aliases_reconciliation():
    # Pass various casing and alias duplicates: reactjs, React, postgres, PostgreSQL
    result = perform_skill_gap_analysis(
        target_role="Full Stack Developer",
        student_skills=["reactjs", "React", "REACT", "postgres", "PostgreSQL", "psql", "js", "JavaScript"]
    )
    # Verified skills must be deduplicated canonical names
    verified = result["verified_skills"]
    assert "React" in verified
    assert "PostgreSQL" in verified
    assert "JavaScript" in verified
    # No duplicate entries
    assert len(verified) == len(set(verified))
    # Matched skills must not have duplicates
    assert len(result["matched_skills"]) == len(set(result["matched_skills"]))
    assert "React" in result["matched_skills"]
    assert "PostgreSQL" in result["matched_skills"]
    assert "JavaScript" in result["matched_skills"]


def test_full_stack_developer_branching_curriculum():
    result = perform_skill_gap_analysis(
        target_role="Full Stack Developer",
        student_skills=[]
    )
    traj_skills = [step["skill_name"] for step in result["ordered_trajectory"]]
    # HTML/CSS must precede JavaScript
    assert traj_skills.index("HTML/CSS") < traj_skills.index("JavaScript")
    # JavaScript must precede React and Node.js
    assert traj_skills.index("JavaScript") < traj_skills.index("React")
    assert traj_skills.index("JavaScript") < traj_skills.index("Node.js")
    # SQL must precede PostgreSQL
    assert traj_skills.index("SQL") < traj_skills.index("PostgreSQL")
    # Git must precede Docker
    assert traj_skills.index("Git") < traj_skills.index("Docker")


def test_self_loop_and_invalid_prerequisites():
    dag = PrerequisiteDAG()
    initial_edge_count = dag.graph.number_of_edges()
    # Self loops should be ignored
    dag.add_prerequisite("Python", "Python")
    assert dag.graph.number_of_edges() == initial_edge_count
    assert nx.is_directed_acyclic_graph(dag.graph)


def test_role_switching_regenerates_requirements():
    ml_result = perform_skill_gap_analysis(
        target_role="Machine Learning Engineer",
        student_skills=["Python"]
    )
    devops_result = perform_skill_gap_analysis(
        target_role="DevOps Engineer",
        student_skills=["Python"]
    )
    assert ml_result["target_role"] != devops_result["target_role"]
    assert set(ml_result["missing_skills"]) != set(devops_result["missing_skills"])

