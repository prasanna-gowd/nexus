"""
Topological Prerequisite DAG Scheduler for NEXUS.

Builds and validates prerequisite dependency graphs using NetworkX.
Distinguishes pedagogical prerequisites from mere co-occurrence.
Detects and resolves cycles to generate valid topological learning trajectories.
"""
import logging
from typing import List, Dict, Set, Tuple, Optional, Any
import networkx as nx

logger = logging.getLogger(__name__)

# Standard pedagogical prerequisite knowledge base.
# Key: prerequisite skill -> Value: dependent skill (Key MUST be learned BEFORE Value)
CURATED_PREREQUISITES: List[Tuple[str, str]] = [
    # Programming fundamentals to advanced concepts / frameworks
    ("Python", "Django"),
    ("Python", "FastAPI"),
    ("Python", "Flask"),
    ("Python", "Pandas"),
    ("Python", "NumPy"),
    ("Python", "PyTorch"),
    ("Python", "TensorFlow"),
    ("Python", "scikit-learn"),
    ("JavaScript", "TypeScript"),
    ("JavaScript", "React"),
    ("JavaScript", "Vue.js"),
    ("JavaScript", "Angular"),
    ("JavaScript", "Node.js"),
    ("TypeScript", "Angular"),
    ("TypeScript", "Next.js"),
    ("React", "Next.js"),
    ("Node.js", "Express.js"),
    ("HTML/CSS", "JavaScript"),
    ("HTML/CSS", "React"),

    # Data & Database hierarchies
    ("SQL", "PostgreSQL"),
    ("SQL", "MySQL"),
    ("SQL", "Database Design"),
    ("SQL", "Data Warehouse"),
    ("PostgreSQL", "Database Optimization"),

    # Data Science & ML hierarchies
    ("Python", "Machine Learning"),
    ("Pandas", "Machine Learning"),
    ("NumPy", "Machine Learning"),
    ("Machine Learning", "Deep Learning"),
    ("Machine Learning", "scikit-learn"),
    ("Deep Learning", "PyTorch"),
    ("Deep Learning", "TensorFlow"),
    ("Deep Learning", "Natural Language Processing"),
    ("Deep Learning", "Computer Vision"),
    ("Natural Language Processing", "Large Language Models"),

    # DevOps & Infrastructure
    ("Linux", "Docker"),
    ("Linux", "Bash"),
    ("Git", "CI/CD"),
    ("Docker", "Kubernetes"),
    ("Docker", "Microservices"),
    ("CI/CD", "DevOps"),
    ("AWS", "Kubernetes"),
    ("AWS", "Terraform"),
    ("Cloud Platform", "Terraform"),

    # General Systems
    ("Data Structures", "Algorithms"),
    ("Algorithms", "System Design"),
    ("REST API", "Microservices"),
]


class PrerequisiteDAG:
    """Manages the directed acyclic graph of skill prerequisites."""

    def __init__(self):
        self.graph = nx.DiGraph()
        self._build_base_graph()

    def _build_base_graph(self):
        """Initializes the graph with curated technical prerequisites."""
        for prereq, dependent in CURATED_PREREQUISITES:
            self.add_prerequisite(prereq, dependent, weight=1.0)

    def add_prerequisite(self, prereq: str, dependent: str, weight: float = 1.0):
        """
        Adds a directed edge from prereq to dependent.
        Ensures no cycles are introduced.
        """
        prereq_clean = prereq.strip()
        dependent_clean = dependent.strip()

        if prereq_clean.lower() == dependent_clean.lower():
            return

        self.graph.add_edge(prereq_clean, dependent_clean, weight=weight)

        # Cycle check
        if not nx.is_directed_acyclic_graph(self.graph):
            logger.warning(f"Cycle detected when adding edge ({prereq_clean} -> {dependent_clean}). Removing edge.")
            self.graph.remove_edge(prereq_clean, dependent_clean)

    def get_prerequisites_for_skill(self, skill: str) -> List[str]:
        """Returns all direct and indirect ancestors required before learning `skill`."""
        if skill not in self.graph:
            return []
        try:
            ancestors = nx.ancestors(self.graph, skill)
            # Order ancestors topologically
            subgraph = self.graph.subgraph(ancestors | {skill})
            ordered = list(nx.topological_sort(subgraph))
            return [s for s in ordered if s != skill]
        except Exception as e:
            logger.warning(f"Error computing ancestors for {skill}: {e}")
            return []

    def compute_learning_trajectory(
        self,
        required_skills: List[str],
        verified_skills: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Generates an ordered topological trajectory for missing skills.

        Args:
            required_skills: Skills needed for the role.
            verified_skills: Skills the student already has mastered.

        Returns:
            List of dicts containing ordered skills with their prerequisite status.
        """
        verified_set = {s.strip().lower() for s in verified_skills}
        
        # 1. Identify missing target skills
        missing_skills = [s for s in required_skills if s.strip().lower() not in verified_set]
        if not missing_skills:
            return []

        # 2. Gather all missing skills and their necessary prerequisites
        all_needed_skills: Set[str] = set()
        for skill in missing_skills:
            all_needed_skills.add(skill)
            # Also include ancestors if not verified
            ancestors = self.get_prerequisites_for_skill(skill)
            for anc in ancestors:
                if anc.lower() not in verified_set:
                    all_needed_skills.add(anc)

        # 3. Create subgraph of all needed skills
        subgraph = nx.DiGraph()
        for skill in all_needed_skills:
            subgraph.add_node(skill)

        for u, v in self.graph.edges():
            if u in all_needed_skills and v in all_needed_skills:
                subgraph.add_edge(u, v)

        # 4. Ensure subgraph is acyclic
        if not nx.is_directed_acyclic_graph(subgraph):
            cycles = list(nx.simple_cycles(subgraph))
            for cycle in cycles:
                # Remove feedback edge
                subgraph.remove_edge(cycle[-1], cycle[0])

        # 5. Topological Sort
        ordered_skills = list(nx.topological_sort(subgraph))

        # 6. Format trajectory output
        trajectory = []
        for idx, skill in enumerate(ordered_skills):
            prereqs = [p for p in self.graph.predecessors(skill)] if skill in self.graph else []
            trajectory.append({
                "step": idx + 1,
                "skill_name": skill,
                "prerequisites": prereqs,
                "is_target_skill": skill in missing_skills,
                "status": "pending"
            })

        return trajectory


def perform_skill_gap_analysis(
    target_role: str,
    student_skills: List[str],
    role_skills_override: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Core Skill-Gap Engine:
    Compares student's verified skills against target role requirements.
    """
    from src.analytics.trends import MarketAnalytics

    analytics = MarketAnalytics()
    verified_lower = {s.strip().lower() for s in student_skills}

    # Retrieve required skills for role
    if role_skills_override:
        required_skill_names = role_skills_override
    else:
        role_demand = analytics.get_skills_by_role(target_role, limit=12)
        if role_demand:
            required_skill_names = [item["skill_name"] for item in role_demand]
        else:
            # Fallback default technical skills for the role
            role_defaults = {
                "machine learning engineer": ["Python", "SQL", "Pandas", "NumPy", "scikit-learn", "PyTorch", "Docker"],
                "data engineer": ["Python", "SQL", "PostgreSQL", "Docker", "Apache Kafka", "Apache Spark", "Linux"],
                "frontend developer": ["HTML/CSS", "JavaScript", "TypeScript", "React", "Next.js", "Git"],
                "backend developer": ["Python", "PostgreSQL", "Docker", "REST API", "Redis", "Linux", "FastAPI"],
                "full stack developer": ["JavaScript", "Python", "React", "PostgreSQL", "Docker", "Git", "REST API"],
                "devops engineer": ["Linux", "Git", "Docker", "Kubernetes", "CI/CD", "AWS", "Terraform"],
            }
            matched_key = next((k for k in role_defaults if k in target_role.lower()), None)
            if matched_key:
                required_skill_names = role_defaults[matched_key]
            else:
                top_general = analytics.get_top_skills(limit=8)
                required_skill_names = [s["skill_name"] for s in top_general] or ["Python", "SQL", "Git", "Docker"]

    matched_skills = [s for s in required_skill_names if s.strip().lower() in verified_lower]
    missing_skills = [s for s in required_skill_names if s.strip().lower() not in verified_lower]

    # Compute topological DAG trajectory
    dag = PrerequisiteDAG()
    trajectory = dag.compute_learning_trajectory(missing_skills, student_skills)

    match_pct = round((len(matched_skills) / max(len(required_skill_names), 1)) * 100, 1)

    return {
        "target_role": target_role,
        "match_percentage": match_pct,
        "verified_skills": student_skills,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "total_required": len(required_skill_names),
        "ordered_trajectory": trajectory,
    }


if __name__ == "__main__":
    result = perform_skill_gap_analysis(
        target_role="Machine Learning Engineer",
        student_skills=["Python"]
    )
    print(f"Match: {result['match_percentage']}%")
    print(f"Matched: {result['matched_skills']}")
    print(f"Missing: {result['missing_skills']}")
    print("Trajectory:")
    for step in result["ordered_trajectory"]:
        print(f"  Step {step['step']}: {step['skill_name']} (Prereqs: {step['prerequisites']})")
