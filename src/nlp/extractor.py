"""
Technical skill extraction using GLiNER zero-shot NER.

Extracts programming languages, frameworks, tools, databases,
cloud technologies, and concepts from cleaned job descriptions.
"""
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Labels for GLiNER entity extraction
ENTITY_LABELS = [
    "PROGRAMMING_LANGUAGE",
    "FRAMEWORK",
    "TOOL",
    "DATABASE",
    "CLOUD_PLATFORM",
    "CONCEPT",
    "MINIMUM_EXPERIENCE",
]

# Generic phrases to filter out (false positives)
GENERIC_FILTERS = {
    "team player", "self-starter", "detail-oriented", "problem solver",
    "team", "experience", "years", "strong", "excellent", "good",
    "communication", "written", "verbal", "ability", "skills",
    "understanding", "knowledge", "familiarity", "proficiency",
    "work", "working", "environment", "company", "role", "position",
    "responsibilities", "requirements", "qualifications", "preferred",
    "required", "must", "should", "will", "can", "able", "plus",
    "bonus", "nice", "ideal", "candidate", "applicant",
    "minimum", "maximum", "at least", "or more", "or equivalent",
    "bachelor", "master", "degree", "phd", "computer science",
    "related field", "equivalent experience",
    "full-time", "part-time", "contract", "remote", "hybrid", "onsite",
    "salary", "compensation", "benefits", "health", "insurance",
}

# Known tech skill aliases mapping
SKILL_ALIASES = {
    "reactjs": "React",
    "react.js": "React",
    "react js": "React",
    "vuejs": "Vue.js",
    "vue js": "Vue.js",
    "angularjs": "Angular",
    "angular.js": "Angular",
    "angular js": "Angular",
    "node.js": "Node.js",
    "nodejs": "Node.js",
    "node js": "Node.js",
    "next.js": "Next.js",
    "nextjs": "Next.js",
    "nuxt.js": "Nuxt.js",
    "nuxtjs": "Nuxt.js",
    "express.js": "Express.js",
    "expressjs": "Express.js",
    "typescript": "TypeScript",
    "javascript": "JavaScript",
    "js": "JavaScript",
    "ts": "TypeScript",
    "python3": "Python",
    "python 3": "Python",
    "golang": "Go",
    "c++": "C++",
    "c#": "C#",
    "csharp": "C#",
    "c sharp": "C#",
    "postgres": "PostgreSQL",
    "postgresql": "PostgreSQL",
    "mongo": "MongoDB",
    "mongodb": "MongoDB",
    "dynamodb": "DynamoDB",
    "mysql": "MySQL",
    "mssql": "SQL Server",
    "ms sql": "SQL Server",
    "sql server": "SQL Server",
    "aws": "AWS",
    "amazon web services": "AWS",
    "gcp": "Google Cloud Platform",
    "google cloud": "Google Cloud Platform",
    "azure": "Microsoft Azure",
    "microsoft azure": "Microsoft Azure",
    "k8s": "Kubernetes",
    "kube": "Kubernetes",
    "kubernetes": "Kubernetes",
    "docker": "Docker",
    "terraform": "Terraform",
    "ci/cd": "CI/CD",
    "cicd": "CI/CD",
    "ci cd": "CI/CD",
    "ml": "Machine Learning",
    "machine learning": "Machine Learning",
    "deep learning": "Deep Learning",
    "dl": "Deep Learning",
    "nlp": "Natural Language Processing",
    "llm": "Large Language Models",
    "llms": "Large Language Models",
    "gen ai": "Generative AI",
    "genai": "Generative AI",
    "graphql": "GraphQL",
    "rest api": "REST API",
    "restful": "REST API",
    "redis": "Redis",
    "elasticsearch": "Elasticsearch",
    "kafka": "Apache Kafka",
    "spark": "Apache Spark",
    "airflow": "Apache Airflow",
    "pytorch": "PyTorch",
    "tensorflow": "TensorFlow",
    "scikit-learn": "scikit-learn",
    "sklearn": "scikit-learn",
    "pandas": "pandas",
    "numpy": "NumPy",
    "fastapi": "FastAPI",
    "flask": "Flask",
    "django": "Django",
    "spring boot": "Spring Boot",
    "springboot": "Spring Boot",
    "rails": "Ruby on Rails",
    "ruby on rails": "Ruby on Rails",
    "git": "Git",
    "github": "GitHub",
    "gitlab": "GitLab",
    "jira": "Jira",
    "linux": "Linux",
    "unix": "Linux/Unix",
    "agile": "Agile",
    "scrum": "Scrum",
}

# Fallback regex patterns for skill extraction when GLiNER is unavailable
TECH_PATTERNS = [
    # Programming languages
    r'\b(Python|Java|JavaScript|TypeScript|C\+\+|C#|Go|Rust|Ruby|PHP|Swift|Kotlin|Scala|R|Julia|Perl|Lua|Haskell|Elixir|Clojure|Dart)\b',
    # Frameworks and libraries
    r'\b(React|Angular|Vue\.?js|Next\.?js|Django|Flask|FastAPI|Spring\s*Boot|Express\.?js|Rails|Laravel|ASP\.NET|Svelte|Nuxt\.?js|Remix)\b',
    # Databases
    r'\b(PostgreSQL|MySQL|MongoDB|Redis|Elasticsearch|DynamoDB|Cassandra|Neo4j|SQLite|MariaDB|CouchDB|InfluxDB|TimescaleDB)\b',
    # Cloud and DevOps
    r'\b(AWS|GCP|Azure|Docker|Kubernetes|Terraform|Ansible|Jenkins|CircleCI|GitHub\s*Actions|GitLab\s*CI|ArgoCD|Helm)\b',
    # Data and ML
    r'\b(TensorFlow|PyTorch|scikit-learn|Pandas|NumPy|Spark|Kafka|Airflow|dbt|Snowflake|Databricks|MLflow|Kubeflow|Ray)\b',
    # Tools
    r'\b(Git|Linux|GraphQL|REST\s*API|gRPC|RabbitMQ|Celery|Nginx|Apache|Prometheus|Grafana|Datadog|Splunk|ELK)\b',
    # Concepts
    r'\b(CI/CD|microservices|distributed\s+systems|system\s+design|machine\s+learning|deep\s+learning|NLP|computer\s+vision|DevOps|MLOps|DataOps|SRE)\b',
]


class SkillExtractor:
    """Extracts technical skills from job descriptions using GLiNER or regex fallback."""
    
    def __init__(self, model_name: str = "urchade/gliner_base", use_gliner: bool = True):
        """
        Initialize the skill extractor.
        
        Args:
            model_name: GLiNER model name to load.
            use_gliner: Whether to attempt GLiNER loading (False = regex-only mode).
        """
        self.model = None
        self.use_gliner = use_gliner
        
        if use_gliner:
            try:
                from gliner import GLiNER
                logger.info(f"Loading GLiNER model: {model_name}...")
                self.model = GLiNER.from_pretrained(model_name)
                logger.info("GLiNER model loaded successfully")
            except ImportError:
                logger.warning("GLiNER not installed. Using regex fallback for skill extraction.")
                self.use_gliner = False
            except Exception as e:
                logger.warning(f"Failed to load GLiNER model: {e}. Using regex fallback.")
                self.use_gliner = False
    
    def extract_skills(self, text: str) -> list[dict]:
        """
        Extract technical skills from text.
        
        Returns list of dicts with keys: skill, category, confidence
        """
        if not text or len(text.strip()) < 10:
            return []
        
        raw_skills = []
        
        # Try GLiNER first
        if self.model is not None:
            raw_skills = self._extract_with_gliner(text)
        
        # Always supplement with regex patterns
        regex_skills = self._extract_with_regex(text)
        
        # Merge results
        seen_normalized = set()
        merged = []
        
        for skill in raw_skills + regex_skills:
            normalized = self._normalize_skill_name(skill["skill"])
            if normalized and normalized.lower() not in seen_normalized:
                seen_normalized.add(normalized.lower())
                skill["skill"] = normalized
                merged.append(skill)
        
        return merged
    
    def _extract_with_gliner(self, text: str) -> list[dict]:
        """Extract skills using GLiNER zero-shot NER."""
        skills = []
        try:
            # GLiNER has a max token limit, chunk if needed
            chunks = self._chunk_text(text, max_chars=1500)
            
            for chunk in chunks:
                entities = self.model.predict_entities(
                    chunk,
                    ENTITY_LABELS,
                    threshold=0.3
                )
                
                for entity in entities:
                    skill_text = entity.get("text", "").strip()
                    label = entity.get("label", "TOOL")
                    score = entity.get("score", 0.5)
                    
                    if self._is_valid_skill(skill_text):
                        category = self._label_to_category(label)
                        skills.append({
                            "skill": skill_text,
                            "category": category,
                            "confidence": round(score, 3),
                        })
        except Exception as e:
            logger.warning(f"GLiNER extraction error: {e}")
        
        return skills
    
    def _extract_with_regex(self, text: str) -> list[dict]:
        """Extract skills using regex patterns as fallback/supplement."""
        skills = []
        
        for pattern in TECH_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                match = match.strip()
                if self._is_valid_skill(match):
                    skills.append({
                        "skill": match,
                        "category": self._categorize_skill(match),
                        "confidence": 0.75,  # Regex match confidence
                    })
        
        return skills
    
    def _normalize_skill_name(self, name: str) -> str:
        """Normalize a skill name using the alias map."""
        if not name:
            return ""
        
        name = name.strip()
        name_lower = name.lower()
        
        # Check alias map
        if name_lower in SKILL_ALIASES:
            return SKILL_ALIASES[name_lower]
        
        return name
    
    def _is_valid_skill(self, text: str) -> bool:
        """Filter out generic/non-technical phrases."""
        if not text:
            return False
        
        text_lower = text.lower().strip()
        
        # Too short or too long
        if len(text_lower) < 2 or len(text_lower) > 50:
            return False
        
        # In generic filter list
        if text_lower in GENERIC_FILTERS:
            return False
        
        # Mostly numbers
        if re.match(r'^[\d\s\+\-\.]+$', text_lower):
            return False
        
        return True
    
    def _label_to_category(self, label: str) -> str:
        """Convert GLiNER label to our category system."""
        mapping = {
            "PROGRAMMING_LANGUAGE": "programming_language",
            "FRAMEWORK": "framework",
            "TOOL": "tool",
            "DATABASE": "database",
            "CLOUD_PLATFORM": "cloud",
            "CONCEPT": "concept",
            "MINIMUM_EXPERIENCE": "experience",
        }
        return mapping.get(label, "tool")
    
    def _categorize_skill(self, skill: str) -> str:
        """Categorize a regex-extracted skill."""
        skill_lower = skill.lower()
        
        languages = {"python", "java", "javascript", "typescript", "c++", "c#", "go", "rust", 
                     "ruby", "php", "swift", "kotlin", "scala", "r", "julia", "perl", "lua",
                     "haskell", "elixir", "clojure", "dart"}
        frameworks = {"react", "angular", "vue.js", "next.js", "django", "flask", "fastapi",
                      "spring boot", "express.js", "rails", "laravel", "asp.net", "svelte"}
        databases = {"postgresql", "mysql", "mongodb", "redis", "elasticsearch", "dynamodb",
                     "cassandra", "neo4j", "sqlite", "mariadb"}
        cloud = {"aws", "gcp", "azure", "docker", "kubernetes", "terraform", "ansible",
                "jenkins", "github actions", "gitlab ci"}
        
        if skill_lower in languages:
            return "programming_language"
        elif skill_lower in frameworks:
            return "framework"
        elif skill_lower in databases:
            return "database"
        elif skill_lower in cloud:
            return "cloud"
        else:
            return "tool"
    
    def _chunk_text(self, text: str, max_chars: int = 1500) -> list[str]:
        """Split text into chunks for GLiNER processing."""
        if len(text) <= max_chars:
            return [text]
        
        chunks = []
        sentences = re.split(r'[.\n]', text)
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 > max_chars:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text[:max_chars]]


# Module-level convenience function
_extractor = None

def get_extractor(use_gliner: bool = True) -> SkillExtractor:
    """Get or create the singleton skill extractor."""
    global _extractor
    if _extractor is None:
        _extractor = SkillExtractor(use_gliner=use_gliner)
    return _extractor


def extract_skills(text: str, use_gliner: bool = True) -> list[dict]:
    """Convenience function to extract skills from text."""
    extractor = get_extractor(use_gliner=use_gliner)
    return extractor.extract_skills(text)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    sample = """
    Requirements:
    - 5+ years experience with Python and JavaScript
    - Strong experience with React and Node.js
    - PostgreSQL and Redis for data storage
    - Docker and Kubernetes for deployment
    - Experience with AWS (EC2, S3, Lambda)
    - Understanding of CI/CD pipelines
    - Machine learning experience is a plus
    - Familiarity with TensorFlow or PyTorch
    """
    
    skills = extract_skills(sample, use_gliner=False)
    print(f"Extracted {len(skills)} skills:")
    for s in skills:
        print(f"  {s['skill']} ({s['category']}) - confidence: {s['confidence']}")
