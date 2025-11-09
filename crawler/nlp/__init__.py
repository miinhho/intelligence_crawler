"""NLP module for text processing."""

from .embedder import TextEmbedder
from .processor import NLPProcessor
from .similarity import (
    calculate_link_relevance,
    calculate_topic_relevance,
    cosine_similarity,
)
from .summarizer import TextSummarizer

__all__ = [
    "NLPProcessor",
    "TextEmbedder",
    "TextSummarizer",
    "cosine_similarity",
    "calculate_topic_relevance",
    "calculate_link_relevance",
]
