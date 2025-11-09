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

        # For auto topic discovery using embeddings
        self.page_embeddings: list[np.ndarray] = []
        self.page_summaries: list[str] = []
        self.page_titles: list[str] = []
        self.discovered_topic_embedding: np.ndarray | None = None
        self.discovered_topic_text: str | None = None
        self.outlier_threshold: float = 0.3  # Threshold to detect outliers

    def _detect_outliers(self, embeddings: list[np.ndarray]) -> list[int]:
        """
        Detect outlier pages using isolation from centroid

        Returns indices of non-outlier pages
        """
        if len(embeddings) < 3:
            return list(range(len(embeddings)))

        embeddings_matrix = np.vstack(embeddings)
        centroid = np.mean(embeddings_matrix, axis=0)

        # Calculate similarity of each page to centroid
        similarities = []
        for emb in embeddings:
            sim = np.dot(emb, centroid) / (
                np.linalg.norm(emb) * np.linalg.norm(centroid)
            )
            similarities.append(sim)

        # Filter out pages with very low similarity (outliers)
        median_sim = np.median(similarities)
        inlier_indices = [
            idx
            for idx, sim in enumerate(similarities)
            if sim >= median_sim * (1 - self.outlier_threshold)
        ]

        return inlier_indices

    async def discover_topic_from_embeddings(
        self, min_pages: int = 5
    ) -> tuple[np.ndarray, str] | None:
        """
        Automatically discover the main topic using embedding-based clustering

        Improved accuracy through:
        1. Outlier detection and removal
        2. Weighted centroid (emphasize high-quality pages)
        3. Multi-candidate selection with title consideration
        4. Iterative refinement

        Args:
            min_pages: Minimum number of pages needed to discover topic

        Returns:
            Tuple of (topic_embedding, topic_text) or None
        """
        if len(self.page_embeddings) < min_pages:
            return None

        # Step 1: Remove outliers for more robust centroid
        inlier_indices = self._detect_outliers(self.page_embeddings)

        if len(inlier_indices) < 3:
            # Too few pages after outlier removal, use all
            inlier_indices = list(range(len(self.page_embeddings)))

        inlier_embeddings = [self.page_embeddings[i] for i in inlier_indices]
        inlier_summaries = [self.page_summaries[i] for i in inlier_indices]
        inlier_titles = [self.page_titles[i] for i in inlier_indices]

        logger.info(
            f"Topic discovery: {len(inlier_embeddings)}/{len(self.page_embeddings)} pages after outlier removal"
        )

        # Step 2: Calculate weighted centroid
        # Weight by inverse of distance to initial centroid (iterative refinement)
        embeddings_matrix = np.vstack(inlier_embeddings)
        initial_centroid = np.mean(embeddings_matrix, axis=0)

        # Calculate weights based on similarity to initial centroid
        weights = []
        for emb in inlier_embeddings:
            sim = np.dot(emb, initial_centroid) / (
                np.linalg.norm(emb) * np.linalg.norm(initial_centroid)
            )
            # Higher similarity = higher weight (squared for emphasis)
            weights.append(sim**2)

        weights = np.array(weights)
        weights = weights / np.sum(weights)  # Normalize

        # Weighted centroid
        refined_centroid = np.average(embeddings_matrix, axis=0, weights=weights)

        # Step 3: Find top candidates closest to refined centroid
        similarities = []
        for idx, embedding in enumerate(inlier_embeddings):
            sim = np.dot(embedding, refined_centroid) / (
                np.linalg.norm(embedding) * np.linalg.norm(refined_centroid)
            )
            similarities.append((idx, sim))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Step 4: Select best candidate considering both similarity and title quality
        best_idx = similarities[0][0]
        best_sim = similarities[0][1]

        # Check if title is informative (not too short, has meaningful words)
        for idx, sim in similarities[:3]:  # Check top 3
            title = inlier_titles[idx]

            # Prefer pages with informative titles
            title_words = len(title.split())
            if title_words >= 3 and sim >= best_sim * 0.95:
                # Good title and similar enough
                best_idx = idx
                best_sim = sim
                break

        # Use combination of title and summary for topic text
        topic_title = inlier_titles[best_idx]
        topic_summary = inlier_summaries[best_idx]

        # Extract key phrases from summary for concise topic
        # Remove title prefix if summary starts with it
        if topic_summary.startswith(f"{topic_title}."):
            clean_summary = topic_summary[len(topic_title) + 1 :].strip()
        else:
            clean_summary = topic_summary

        # Limit to first 2-3 sentences for topic
        sentences = clean_summary.split(". ")
        key_sentences = ". ".join(sentences[:2]).strip()

        # Create concise topic description
        if len(topic_title) > 10 and topic_title not in key_sentences:
            topic_text = f"{topic_title}: {key_sentences}"
        else:
            topic_text = key_sentences

        # Limit topic text length to 150 chars for better focus
        if len(topic_text) > 150:
            topic_text = topic_text[:150].rsplit(" ", 1)[0] + "..."

        logger.info(
            f"Auto-discovered topic (similarity: {best_sim:.3f}, {len(inlier_embeddings)} pages):"
        )
        logger.info(f"  Title: {topic_title}")
        logger.info(f"  Topic: {topic_text}")

        return refined_centroid, topic_text

    async def process_page(
        self,
        content: str,
        topic: str | None = None,
        title: str = "",
        auto_discover: bool = False,
    ) -> dict[str, Any]:
        """
        Process page content: summarize and calculate topic relevance

        Args:
            content: Main text content
            topic: Target topic for relevance calculation (optional)
            title: Page title
            auto_discover: If True and topic is None, automatically discover topic

        Returns:
            Dictionary with summary, embedding, and relevance score
        """
        if not content:
            return {
                "summary": title or "No content",
                "embedding": np.zeros(768),
                "topic_relevance": 0.0,
                "auto_discovered_topic": None,
            }

        # Truncate very long content
        max_length = 5000
        if len(content) > max_length:
            content = content[:max_length]

        # Generate summary
        summary = self.summarizer.summarize(content, title)

        # Generate embedding for summary
        embedding = await self.embedder.embed_async(summary)

        # Initialize topic_embedding as None
        topic_embedding: np.ndarray | None = None

        # Store embedding and summary for topic discovery
        if auto_discover and topic is None:
            self.page_embeddings.append(embedding)
            self.page_summaries.append(summary)
            self.page_titles.append(title)

            # Try to discover topic after collecting enough pages
            # Also re-discover periodically to refine (every 5 pages after first discovery)
            should_discover = (
                self.discovered_topic_embedding is None
                and len(self.page_embeddings) >= 5
            ) or (
                self.discovered_topic_embedding is not None
                and len(self.page_embeddings) % 5 == 0
            )

            if should_discover:
                result = await self.discover_topic_from_embeddings()
                if result:
                    self.discovered_topic_embedding, self.discovered_topic_text = result
                    logger.info(
                        f"Topic {'discovered' if len(self.page_embeddings) == 5 else 'refined'} with {len(self.page_embeddings)} pages"
                    )

            # Use discovered topic embedding if available
            topic_embedding = self.discovered_topic_embedding

        # Calculate topic relevance
        if auto_discover and topic is None:
            if topic_embedding is None:
                # No topic discovered yet - use position-based scoring
                # Earlier pages get slightly higher initial scores as they guide discovery
                num_pages = len(self.page_embeddings)
                if num_pages <= 1:
                    relevance = 0.95  # First page - seed
                elif num_pages <= 3:
                    relevance = 0.90  # Early exploration
                else:
                    relevance = 0.85  # Pre-discovery collection
            else:
                # Compare with discovered topic centroid
                base_relevance = calculate_topic_relevance(embedding, topic_embedding)

                # Calculate confidence based on cluster cohesion
                # More pages = more confident = stricter filtering
                num_pages = len(self.page_embeddings)

                if num_pages <= 7:
                    # Still learning - moderate boost for borderline content
                    boost = 1.05
                elif num_pages <= 15:
                    # Gaining confidence - small boost
                    boost = 1.02
                else:
                    # Mature collection - no boost, trust the score
                    boost = 1.0

                relevance = min(1.0, base_relevance * boost)

                # Log if significantly different from base
                if abs(relevance - base_relevance) > 0.05:
                    logger.debug(
                        f"Relevance adjusted: {base_relevance:.3f} -> {relevance:.3f} (boost: {boost})"
                    )
        else:
            # User provided explicit topic
            if topic:
                topic_embedding = await self.embedder.embed_async(topic)
                relevance = calculate_topic_relevance(embedding, topic_embedding)
            else:
                relevance = 1.0

        return {
            "summary": summary,
            "embedding": embedding,
            "topic_relevance": float(relevance),
            "auto_discovered_topic": self.discovered_topic_text
            if auto_discover
            else None,
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
