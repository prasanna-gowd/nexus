"""
Unit & Integration Tests for Phase 3: Database Migration, Embeddings & Market Analytics.
"""
from datetime import datetime
import pytest

from src.database.migrate import VectorEmbeddingGenerator, parse_date_safely
from src.analytics.trends import MarketAnalytics


def test_embedding_dimensions():
    embed_gen = VectorEmbeddingGenerator(dim=384)
    texts = [
        "Senior Python Engineer at Fintech startup",
        "Machine Learning Infrastructure with PyTorch and Kubernetes"
    ]
    vectors = embed_gen.encode(texts)
    assert len(vectors) == 2
    assert len(vectors[0]) == 384, "Embedding vectors must strictly have 384 dimensions"
    assert len(vectors[1]) == 384


def test_parse_date_safely():
    d1 = parse_date_safely("2026-04-12")
    assert isinstance(d1, datetime)
    assert d1.year == 2026

    d2 = parse_date_safely("2026-04-12T15:30:00Z")
    assert isinstance(d2, datetime)

    d3 = parse_date_safely(None)
    assert isinstance(d3, datetime), "None dates should safely fall back to current time"

    d4 = parse_date_safely("invalid-date-string")
    assert isinstance(d4, datetime)


def test_market_analytics_initialization():
    analytics = MarketAnalytics(use_postgres=False)
    # Even without PostgreSQL running, DuckDB fallback or graceful empty list must be returned
    top = analytics.get_top_skills(limit=5)
    assert isinstance(top, list)

    cooccur = analytics.get_skill_co_occurrences("Python", limit=5)
    assert isinstance(cooccur, list)

    velocity = analytics.get_skill_velocity()
    assert "status" in velocity
