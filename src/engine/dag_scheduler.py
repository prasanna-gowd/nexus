"""
Topological Prerequisite DAG Scheduler for NEXUS.

Builds and validates prerequisite dependency graphs using NetworkX.
Distinguishes pedagogical prerequisites from mere co-occurrence.
Detects and resolves cycles to generate valid topological learning trajectories.
"""
import logging
from typing import List, Dict, Set, Tuple, Optional, Any
import networkx as nx

from src.nlp.extractor import canonicalize_skill_name

logger = logging.getLogger(__name__)

# Standard pedagogical prerequisite knowledge base.
# Key: prerequisite skill -> Value: dependent skill (Key MUST be learned BEFORE Value)
CURATED_PREREQUISITES: List[Tuple[str, str]] = [
    # Web Foundations & Frontend
    ("HTML/CSS", "JavaScript"),
    ("HTML/CSS", "React"),
    ("JavaScript", "TypeScript"),
    ("JavaScript", "React"),
    ("JavaScript", "Vue.js"),
    ("JavaScript", "Angular"),
    ("JavaScript", "Node.js"),
    ("TypeScript", "Angular"),
    ("TypeScript", "Next.js"),
    ("React", "Next.js"),

    # Backend & Server-Side APIs
    ("Node.js", "Express.js"),
    ("Python", "FastAPI"),
    ("Python", "Django"),
    ("Python", "Flask"),
    ("FastAPI", "REST API"),
    ("Node.js", "REST API"),

    # Data & Database hierarchies
    ("SQL", "PostgreSQL"),
    ("SQL", "MySQL"),
    ("SQL", "Database Design"),
    ("SQL", "Data Warehouse"),
    ("PostgreSQL", "Database Optimization"),
    ("PostgreSQL", "Redis"),

    # Data Science & Machine Learning
    ("Python", "Pandas"),
    ("Python", "NumPy"),
    ("Python", "Machine Learning"),
    ("Pandas", "Machine Learning"),
    ("NumPy", "Machine Learning"),
    ("Machine Learning", "scikit-learn"),
    ("Machine Learning", "Deep Learning"),
    ("Deep Learning", "PyTorch"),
    ("Deep Learning", "TensorFlow"),
    ("Deep Learning", "Natural Language Processing"),
    ("Deep Learning", "Computer Vision"),
    ("Natural Language Processing", "Large Language Models"),

    # Version Control, DevOps, Cloud & Containerization
    ("Git", "CI/CD"),
    ("Git", "Docker"),
    ("Linux", "Docker"),
    ("Linux", "Bash"),
    ("Docker", "Kubernetes"),
    ("Docker", "Microservices"),
    ("Docker", "CI/CD"),
    ("REST API", "Microservices"),
    ("CI/CD", "DevOps"),
    ("AWS", "Kubernetes"),
    ("AWS", "Terraform"),
    ("Cloud Platform", "Terraform"),

    # Computer Science & Architecture Fundamentals
    ("Data Structures", "Algorithms"),
    ("Algorithms", "System Design"),
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
        verified_skills: List[str],
        return_edges: bool = False
    ) -> Any:
        """
        Generates an ordered topological trajectory for missing skills with generation levels
        and directed prerequisite edges.

        Args:
            required_skills: Skills needed for the role.
            verified_skills: Skills the student already has mastered.
            return_edges: If True, returns a tuple (trajectory, dag_edges).

        Returns:
            List of trajectory steps (or tuple of trajectory and dag_edges).
        """
        clean_verified = [canonicalize_skill_name(s) for s in (verified_skills or []) if canonicalize_skill_name(s)]
        verified_set = {s.lower() for s in clean_verified}

        clean_required = []
        seen_req = set()
        for s in (required_skills or []):
            canon = canonicalize_skill_name(s)
            if canon and canon.lower() not in seen_req:
                seen_req.add(canon.lower())
                clean_required.append(canon)

        # 1. Identify missing target skills
        missing_skills = [s for s in clean_required if s.lower() not in verified_set]
        if not missing_skills:
            return ([], []) if return_edges else []

        # 2. Gather all missing skills and their necessary prerequisites
        # If required_skills defines an explicit curriculum/role scope, respect that scope
        # while including foundational prerequisites (like Git, Linux, SQL)
        all_needed_skills: Set[str] = set()
        required_lower = {s.lower() for s in clean_required}

        for skill in missing_skills:
            all_needed_skills.add(skill)
            # Pull direct predecessors that belong to required_skills or are direct prerequisites
            ancestors = self.get_prerequisites_for_skill(skill)
            for anc in ancestors:
                anc_canon = canonicalize_skill_name(anc)
                if anc_canon.lower() in verified_set:
                    continue
                # If required_skills already explicitly defines the scope (len >= 5), only include ancestors that are in required_skills
                # or have direct relevance, avoiding unrelated language/framework stacks
                if len(clean_required) >= 5 and anc_canon.lower() not in required_lower:
                    # Only include essential cross-stack foundations (Git, Linux, SQL, HTML/CSS if web)
                    if anc_canon in ["Git", "Linux", "SQL"]:
                        all_needed_skills.add(anc_canon)
                else:
                    all_needed_skills.add(anc_canon)

        # 3. Create subgraph of all needed skills
        subgraph = nx.DiGraph()
        for skill in all_needed_skills:
            subgraph.add_node(skill)

        for u, v in self.graph.edges():
            u_c = canonicalize_skill_name(u)
            v_c = canonicalize_skill_name(v)
            if u_c in all_needed_skills and v_c in all_needed_skills:
                subgraph.add_edge(u_c, v_c)

        # 4. Ensure subgraph is acyclic
        if not nx.is_directed_acyclic_graph(subgraph):
            cycles = list(nx.simple_cycles(subgraph))
            for cycle in cycles:
                subgraph.remove_edge(cycle[-1], cycle[0])

        # 5. Topological Generations (Level Indexing)
        levels: Dict[str, int] = {}
        for lvl_idx, gen in enumerate(nx.topological_generations(subgraph)):
            for s in gen:
                levels[s] = lvl_idx

        # Pedagogical tie-breaker priority for skills on the same level:
        # Web & Foundations -> Databases -> Languages -> Frameworks -> APIs -> DevOps
        domain_priority = {
            "SQL": 1, "HTML/CSS": 2, "Git": 3, "Linux": 4, "Data Structures": 5,
            "JavaScript": 10, "Python": 11, "TypeScript": 12, "Bash": 13,
            "React": 20, "Vue.js": 21, "Angular": 22, "Node.js": 23, "FastAPI": 24, "Django": 25, "Flask": 26,
            "PostgreSQL": 30, "MySQL": 31, "Redis": 32, "Database Design": 33, "Pandas": 34, "NumPy": 35,
            "REST API": 40, "Express.js": 41, "Next.js": 42, "Machine Learning": 43, "Docker": 44,
            "scikit-learn": 50, "PyTorch": 51, "TensorFlow": 52, "Deep Learning": 53,
            "CI/CD": 60, "Kubernetes": 61, "AWS": 62, "Microservices": 63, "Terraform": 64
        }

        def sort_key(skill_name: str):
            return (levels.get(skill_name, 0), domain_priority.get(skill_name, 90), skill_name)

        ordered_skills = sorted(list(subgraph.nodes()), key=sort_key)

        # 6. Format trajectory output
        trajectory = []
        for idx, skill in enumerate(ordered_skills):
            prereqs = [p for p in subgraph.predecessors(skill)]
            trajectory.append({
                "step": idx + 1,
                "skill_name": skill,
                "prerequisites": prereqs,
                "is_target_skill": skill in missing_skills,
                "level": levels.get(skill, 0),
                "status": "pending"
            })

        dag_edges = [{"source": u, "target": v} for u, v in subgraph.edges()]

        if return_edges:
            return trajectory, dag_edges
        return trajectory


# Sensible, comprehensive curriculum definitions for core industry career roles
ROLE_CURRICULUM = {
    "full stack developer": [
        "HTML/CSS", "JavaScript", "TypeScript", "React", "Node.js",
        "REST API", "SQL", "PostgreSQL", "Git", "Docker"
    ],
    "full stack engineer": [
        "HTML/CSS", "JavaScript", "TypeScript", "React", "Node.js",
        "REST API", "SQL", "PostgreSQL", "Git", "Docker"
    ],
    "full stack": [
        "HTML/CSS", "JavaScript", "TypeScript", "React", "Node.js",
        "REST API", "SQL", "PostgreSQL", "Git", "Docker"
    ],
    "machine learning engineer": [
        "Python", "SQL", "Pandas", "NumPy", "scikit-learn",
        "PyTorch", "Docker", "Machine Learning"
    ],
    "data engineer": [
        "Python", "SQL", "PostgreSQL", "Apache Spark", "Apache Kafka",
        "Docker", "Linux", "Data Warehouse"
    ],
    "frontend developer": [
        "HTML/CSS", "JavaScript", "TypeScript", "React", "Next.js",
        "Git", "REST API"
    ],
    "backend developer": [
        "Python", "SQL", "PostgreSQL", "FastAPI", "REST API",
        "Redis", "Docker", "Linux", "Git"
    ],
    "devops engineer": [
        "Linux", "Git", "Docker", "Kubernetes", "CI/CD",
        "AWS", "Terraform"
    ],
}


def perform_skill_gap_analysis(
    target_role: str,
    student_skills: List[str],
    role_skills_override: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Core Skill-Gap Engine:
    Compares student's verified skills against target role requirements.
    Uses canonical skill IDs to reconcile verified skills, avoiding loose matching,
    duplicate aliases, and false readiness percentages.
    Ensures sensible, industry-standard prerequisite sequences.
    """
    from src.analytics.trends import MarketAnalytics

    analytics = MarketAnalytics()

    # 1. Canonicalize and deduplicate student verified skills
    clean_student_skills = []
    seen_student = set()
    for s in (student_skills or []):
        canon = canonicalize_skill_name(s)
        if canon and canon.lower() not in seen_student:
            seen_student.add(canon.lower())
            clean_student_skills.append(canon)

    # 2. Retrieve required skills for role
    raw_required = []
    if role_skills_override:
        raw_required = list(role_skills_override)
    else:
        # Check standard curriculum first to guarantee sensible pedagogical sequencing
        matched_curriculum_key = next((k for k in ROLE_CURRICULUM if k in target_role.lower()), None)
        if matched_curriculum_key:
            raw_required = list(ROLE_CURRICULUM[matched_curriculum_key])
            # Enrich with real market demand signals if high-frequency skills exist in DB
            try:
                role_demand = analytics.get_skills_by_role(target_role, limit=10)
                existing_lower = {s.lower() for s in raw_required}
                for item in role_demand:
                    s_name = item.get("skill_name", "").strip()
                    if s_name and s_name.lower() not in existing_lower and item.get("demand_count", 0) >= 2:
                        raw_required.append(s_name)
                        existing_lower.add(s_name.lower())
            except Exception:
                pass
        else:
            role_demand = analytics.get_skills_by_role(target_role, limit=12)
            if role_demand:
                raw_required = [item["skill_name"] for item in role_demand]
            else:
                top_general = analytics.get_top_skills(limit=8)
                raw_required = [s["skill_name"] for s in top_general] or ["Python", "SQL", "Git", "Docker"]

    # 3. Canonicalize and deduplicate required skills
    canonical_required = []
    seen_req = set()
    for s in raw_required:
        canon = canonicalize_skill_name(s)
        if canon and canon.lower() not in seen_req:
            seen_req.add(canon.lower())
            canonical_required.append(canon)

    # 4. Reconcile matched and missing skills
    matched_skills = [s for s in canonical_required if s.lower() in seen_student]
    missing_skills = [s for s in canonical_required if s.lower() not in seen_student]

    total_required = len(canonical_required)
    if total_required == 0:
        match_pct = 0.0
    elif len(missing_skills) == 0:
        match_pct = 100.0
    else:
        match_pct = round((len(matched_skills) / total_required) * 100, 1)

    # 5. Compute topological DAG trajectory and edges
    dag = PrerequisiteDAG()
    trajectory, dag_edges = dag.compute_learning_trajectory(missing_skills, clean_student_skills, return_edges=True)

    return {
        "target_role": target_role,
        "match_percentage": match_pct,
        "verified_skills": clean_student_skills,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "total_required": total_required,
        "ordered_trajectory": trajectory,
        "dag_edges": dag_edges,
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
