"""NLP processing facade - orchestrates embedder, summarizer, and similarity."""

import logging
from typing import Any

import numpy as np
import torch

from .embedder import TextEmbedder
from .similarity import calculate_link_relevance, calculate_topic_relevance
from .summarizer import TextSummarizer

logger = logging.getLogger(__name__)


class NLPProcessor:
    """Orchestrates NLP operations: summarization, embedding, similarity"""

    def __init__(
        self,
        embedding_model: str = "jhgan/ko-sroberta-multitask",
        summarization_model: str | None = None,
        device: str | None = None,
        enable_cache: bool = True,
        cache_maxsize: int = 1000,
        cache_ttl: int = 3600,
    ):
        """
        Initialize NLP processor

        Args:
            embedding_model: Model for sentence embeddings (Korean SBERT)
            summarization_model: Model for summarization (optional, can be heavy)
            device: Device to use ('cuda', 'cpu', or None for auto)
            enable_cache: Enable embedding caching
            cache_maxsize: Maximum cache size
            cache_ttl: Cache TTL in seconds
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        logger.info(f"Using device: {self.device}")

        # Initialize components
        self.embedder = TextEmbedder(
            model_name=embedding_model,
            device=device,
            enable_cache=enable_cache,
            cache_maxsize=cache_maxsize,
            cache_ttl=cache_ttl,
        )

        self.summarizer = TextSummarizer(model_name=summarization_model, device=device)

    async def process_page(
        self, content: str, topic: str, title: str = ""
    ) -> dict[str, Any]:
        """
        Process page content: summarize and calculate topic relevance

        Args:
            content: Main text content
            topic: Target topic for relevance calculation
            title: Page title

        Returns:
            Dictionary with summary, embedding, and relevance score
        """
        if not content:
            return {
                "summary": title or "No content",
                "embedding": np.zeros(768),
                "topic_relevance": 0.0,
            }

        # Truncate very long content
        max_length = 5000
        if len(content) > max_length:
            content = content[:max_length]

        # Generate summary
        summary = self.summarizer.summarize(content, title)

        # Generate embedding for summary
        embedding = await self.embedder.embed_async(summary)

        # Calculate topic relevance
        topic_embedding = await self.embedder.embed_async(topic)
        relevance = calculate_topic_relevance(embedding, topic_embedding)

        return {
            "summary": summary,
            "embedding": embedding,
            "topic_relevance": float(relevance),
        }

    async def calculate_link_relevance(
        self, page_summary: str, anchor_text: str, context: str
    ) -> float:
        """
        Calculate semantic relevance of a link based on anchor text and context

        Args:
            page_summary: Summary of current page
            anchor_text: Link anchor text
            context: Text surrounding the link

        Returns:
            Relevance score (0-1)
        """
        if not anchor_text and not context:
            return 0.5  # Neutral score

        # Combine anchor text and context
        link_text = f"{anchor_text} {context}".strip()
        if not link_text:
            return 0.5

        # Limit length
        if len(link_text) > 500:
            link_text = link_text[:500]

        # Calculate similarity with page summary
        page_embedding = await self.embedder.embed_async(page_summary)
        link_embedding = await self.embedder.embed_async(link_text)

        return calculate_link_relevance(page_embedding, link_embedding)

    def calculate_page_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """Calculate similarity between two page embeddings"""
        from .similarity import cosine_similarity

        return cosine_similarity(embedding1, embedding2)

    def batch_embed(self, texts: list[str]) -> list[np.ndarray]:
        """Embed multiple texts in batch for efficiency"""
        return self.embedder.batch_embed(texts)

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics"""
        return self.embedder.get_cache_stats()

    def clear_cache(self) -> None:
        """Clear the embedding cache"""
        self.embedder.clear_cache()
