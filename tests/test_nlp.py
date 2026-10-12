"""
Unit & Integration Tests for Phase 2: NLP, Skill Extraction & Taxonomy Mapping.
"""
import pytest

from src.nlp.cleaner import clean_html, remove_boilerplate, clean_description
from src.nlp.extractor import SkillExtractor
from src.nlp.taxonomy import TaxonomyMapper


def test_clean_html():
    raw_html = "<div><h1>Job Title</h1><p>We are hiring <b>Python</b> devs.</p></div>"
    cleaned = clean_html(raw_html)
    assert "<" not in cleaned
    assert ">" not in cleaned
    assert "Job Title" in cleaned
    assert "Python" in cleaned


def test_remove_boilerplate():
    text = (
        "Core qualifications include Python and Docker.\n"
        "We are an equal opportunity employer and do not discriminate.\n"
        "Benefits include 401(k) matching and unlimited PTO."
    )
    cleaned = remove_boilerplate(text)
    assert "Python and Docker" in cleaned
    assert "equal opportunity" not in cleaned.lower()
    assert "401(k)" not in cleaned


def test_clean_description():
    sample = """
    <div>
      <h2>Software Engineer</h2>
      <p>Company introduction and overview.</p>
      <h3>Requirements</h3>
      <ul>
        <li>3+ years with Python and FastAPI</li>
        <li>PostgreSQL experience</li>
      </ul>
      <p>Equal Opportunity Employer. All applicants are considered regardless of race.</p>
    </div>
    """
    cleaned = clean_description(sample)
    assert "Python" in cleaned
    assert "FastAPI" in cleaned
    assert "Equal Opportunity" not in cleaned


def test_skill_extractor_regex_fallback():
    extractor = SkillExtractor(use_gliner=False)
    text = (
        "We need a backend developer with 4 years of experience in Python, FastAPI, and Docker. "
        "Familiarity with PostgreSQL and AWS is required. Knowledge of React is a plus."
    )
    extracted = extractor.extract_skills(text)
    skill_names = {s["skill"] for s in extracted}

    assert "Python" in skill_names
    assert "FastAPI" in skill_names
    assert "Docker" in skill_names
    assert "PostgreSQL" in skill_names
    assert "AWS" in skill_names
    assert "React" in skill_names


def test_alias_normalization():
    extractor = SkillExtractor(use_gliner=False)
    assert extractor._normalize_skill_name("ReactJS") == "React"
    assert extractor._normalize_skill_name("postgres") == "PostgreSQL"
    assert extractor._normalize_skill_name("golang") == "Go"
    assert extractor._normalize_skill_name("k8s") == "Kubernetes"


def test_generic_phrase_filtering():
    extractor = SkillExtractor(use_gliner=False)
    assert not extractor._is_valid_skill("team player")
    assert not extractor._is_valid_skill("communication")
    assert not extractor._is_valid_skill("years")
    assert extractor._is_valid_skill("Python")
    assert extractor._is_valid_skill("Docker")


def test_taxonomy_mapper_deterministic_id():
    mapper = TaxonomyMapper(threshold=0.82)
    entry1 = mapper.map_skill("Python", "programming_language")
    entry2 = mapper.map_skill("Python", "programming_language")

    assert entry1["skill_id"] == entry2["skill_id"], "Skill IDs must be deterministic"
    assert entry1["canonical_name"] == "Python"
