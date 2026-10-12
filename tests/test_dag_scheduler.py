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
