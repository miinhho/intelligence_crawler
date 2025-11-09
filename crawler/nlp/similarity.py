"""Semantic similarity calculation utilities."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)

    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    similarity = dot_product / (norm1 * norm2)
    return float(similarity)


def calculate_topic_relevance(
    content_embedding: np.ndarray, topic_embedding: np.ndarray
) -> float:
    """Calculate how relevant content is to a topic"""
    return cosine_similarity(content_embedding, topic_embedding)


def calculate_link_relevance(
    page_embedding: np.ndarray, link_text_embedding: np.ndarray
) -> float:
    """Calculate how relevant a link is to current page"""
    similarity = cosine_similarity(page_embedding, link_text_embedding)
    # Normalize to 0-1 range (cosine similarity is -1 to 1)
    normalized = (similarity + 1) / 2
    return float(normalized)
