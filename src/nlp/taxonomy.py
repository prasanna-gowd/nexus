"""
ESCO taxonomy mapper for normalizing extracted skills.

Maps raw extracted skill names to the ESCO (European Skills/Competences) 
taxonomy using sentence-transformer embeddings and cosine similarity.
"""
import os
import csv
import logging
import uuid
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Default similarity threshold for ESCO matching
DEFAULT_SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.82"))
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
ESCO_DATA_PATH = os.getenv("ESCO_DATA_PATH", "data/skills_en.csv")


class TaxonomyMapper:
    """Maps extracted skill names to ESCO canonical skills using embeddings."""
    
    def __init__(self, 
                 esco_path: str = ESCO_DATA_PATH,
                 model_name: str = EMBEDDING_MODEL_NAME,
                 threshold: float = DEFAULT_SIMILARITY_THRESHOLD):
        """
        Initialize the taxonomy mapper.
        
        Args:
            esco_path: Path to the ESCO skills CSV file.
            model_name: Sentence-transformer model name.
            threshold: Cosine similarity threshold for matching.
        """
        self.threshold = threshold
        self.esco_skills = []
        self.esco_embeddings = None
        self.model = None
        self.model_name = model_name
        self._loaded = False
        
        # Try to load ESCO data
        self._load_esco_data(esco_path)
    
    def _load_esco_data(self, esco_path: str) -> None:
        """Load the ESCO skills dataset."""
        if not Path(esco_path).exists():
            logger.warning(f"ESCO dataset not found at {esco_path}. "
                          "Taxonomy mapping will use skill names as-is. "
                          "Download from: https://esco.ec.europa.eu/en/use-esco/download")
            return
        
        try:
            with open(esco_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # ESCO CSV has columns like: conceptUri, preferredLabel, description
                    label = row.get('preferredLabel', row.get('preferred_label', '')).strip()
                    uri = row.get('conceptUri', row.get('concept_uri', '')).strip()
                    
                    if label:
                        self.esco_skills.append({
                            'label': label,
                            'uri': uri,
                        })
            
            logger.info(f"Loaded {len(self.esco_skills)} ESCO skills from {esco_path}")
        except Exception as e:
            logger.error(f"Failed to load ESCO data: {e}")
    
    def _load_model_and_embeddings(self) -> None:
        """Lazy-load the sentence transformer model and compute ESCO embeddings."""
        if self._loaded:
            return
        
        if not self.esco_skills:
            self._loaded = True
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            
            logger.info(f"Loading embedding model: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            
            # Compute embeddings for all ESCO skills
            labels = [s['label'] for s in self.esco_skills]
            logger.info(f"Computing embeddings for {len(labels)} ESCO skills...")
            self.esco_embeddings = self.model.encode(labels, show_progress_bar=False, 
                                                      normalize_embeddings=True)
            
            logger.info(f"ESCO embeddings computed: shape={self.esco_embeddings.shape}")
            self._loaded = True
            
        except ImportError:
            logger.warning("sentence-transformers not installed. Using direct name matching.")
            self._loaded = True
        except Exception as e:
            logger.error(f"Failed to load model/compute embeddings: {e}")
            self._loaded = True
    
    def map_skill(self, skill_name: str, category: str = "tool") -> dict:
        """
        Map a single skill name to the ESCO taxonomy.
        
        Args:
            skill_name: The raw extracted skill name.
            category: The skill category from the extractor.
        
        Returns:
            Dict with: skill_id, canonical_name, category, esco_uri, 
                       confidence, is_emerging
        """
        if not skill_name:
            return self._create_skill_entry(skill_name, category, is_emerging=True)
        
        # If no ESCO data, return as-is
        if not self.esco_skills:
            return self._create_skill_entry(skill_name, category, is_emerging=False)
        
        # Lazy load model
        self._load_model_and_embeddings()
        
        # If model failed to load, do exact string matching
        if self.model is None or self.esco_embeddings is None:
            return self._exact_match(skill_name, category)
        
        try:
            import numpy as np
            
            # Encode the query skill
            query_embedding = self.model.encode([skill_name], normalize_embeddings=True)
            
            # Compute cosine similarity
            similarities = np.dot(self.esco_embeddings, query_embedding.T).flatten()
            best_idx = int(np.argmax(similarities))
            best_score = float(similarities[best_idx])
            
            if best_score >= self.threshold:
                # Matched to ESCO taxonomy
                matched = self.esco_skills[best_idx]
                return self._create_skill_entry(
                    canonical_name=matched['label'],
                    category=category,
                    esco_uri=matched['uri'],
                    confidence=best_score,
                    is_emerging=False
                )
            else:
                # Below threshold - potential emerging skill
                return self._create_skill_entry(
                    canonical_name=skill_name,
                    category=category,
                    confidence=best_score,
                    is_emerging=True,
                    note=f"Best ESCO match: '{self.esco_skills[best_idx]['label']}' "
                         f"(similarity: {best_score:.3f})"
                )
                
        except Exception as e:
            logger.warning(f"Embedding-based matching failed for '{skill_name}': {e}")
            return self._create_skill_entry(skill_name, category, is_emerging=True)
    
    def _exact_match(self, skill_name: str, category: str) -> dict:
        """Fallback exact/substring matching against ESCO labels."""
        skill_lower = skill_name.lower()
        
        for esco_skill in self.esco_skills:
            if skill_lower == esco_skill['label'].lower():
                return self._create_skill_entry(
                    canonical_name=esco_skill['label'],
                    category=category,
                    esco_uri=esco_skill['uri'],
                    confidence=1.0,
                    is_emerging=False
                )
        
        return self._create_skill_entry(skill_name, category, is_emerging=True)
    
    def _create_skill_entry(self, canonical_name: str, category: str,
                           esco_uri: str = "", confidence: float = 0.0,
                           is_emerging: bool = False, note: str = "") -> dict:
        """Create a standardized skill entry dict."""
        skill_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, canonical_name.lower().strip()))
        
        entry = {
            "skill_id": skill_id,
            "canonical_name": canonical_name.strip(),
            "category": category,
            "esco_uri": esco_uri,
            "confidence": round(confidence, 4),
            "is_emerging": is_emerging,
        }
        
        if note:
            logger.debug(f"Skill '{canonical_name}': {note}")
        
        return entry


# Module-level singleton
_mapper = None

def get_mapper(**kwargs) -> TaxonomyMapper:
    """Get or create the singleton taxonomy mapper."""
    global _mapper
    if _mapper is None:
        _mapper = TaxonomyMapper(**kwargs)
    return _mapper


def map_skill(skill_name: str, category: str = "tool") -> dict:
    """Convenience function to map a single skill."""
    mapper = get_mapper()
    return mapper.map_skill(skill_name, category)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    test_skills = ["Python", "ReactJS", "kubernetes", "FastAPI", "LangChain"]
    mapper = TaxonomyMapper()
    
    print("Taxonomy mapping results:")
    for skill in test_skills:
        result = mapper.map_skill(skill, "tool")
        status = "EMERGING" if result["is_emerging"] else "MATCHED"
        print(f"  {skill} -> {result['canonical_name']} [{status}] "
              f"(confidence: {result['confidence']:.3f})")
