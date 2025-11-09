"""Text embedding module using Sentence-BERT."""

import asyncio
import logging

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from ..core.cache import EmbeddingCache

logger = logging.getLogger(__name__)


class TextEmbedder:
    """Handles text embedding with caching"""

    def __init__(
        self,
        model_name: str = "jhgan/ko-sroberta-multitask",
        device: str | None = None,
        enable_cache: bool = True,
        cache_maxsize: int = 1000,
        cache_ttl: int = 3600,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        logger.info(f"Using device: {self.device}")

        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=self.device)

        self.enable_cache = enable_cache
        self.cache = (
            EmbeddingCache(max_size=cache_maxsize, ttl=cache_ttl)
            if enable_cache
            else None
        )
        if enable_cache:
            logger.info(
                f"Embedding cache enabled (max_size={cache_maxsize}, ttl={cache_ttl}s)"
            )

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for text with caching support"""
        if not text:
            return np.zeros(768)

        if self.enable_cache and self.cache:
            cached = self.cache.get(text)
            if cached is not None:
                logger.debug(f"Cache hit for text (length: {len(text)})")
                return cached

        embedding = self.model.encode(
            text, convert_to_numpy=True, show_progress_bar=False
        )

        if self.enable_cache and self.cache:
            self.cache.set(text, embedding)
            logger.debug(f"Cached embedding for text (length: {len(text)})")

        return embedding

    def batch_embed(self, texts: list[str]) -> list[np.ndarray]:
        """Embed multiple texts in batch for efficiency"""
        if not texts:
            return []

        embeddings = self.model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False, batch_size=32
        )
        return [emb for emb in embeddings]

    async def embed_async(self, text: str) -> np.ndarray:
        """Async wrapper for embedding"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.embed, text)

    async def batch_embed_async(self, texts: list[str]) -> list[np.ndarray]:
        """Async wrapper for batch embedding"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.batch_embed, texts)

    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        if self.enable_cache and self.cache:
            return {"enabled": True, **self.cache.stats()}
        return {"enabled": False, "size": 0, "max_size": 0, "ttl": 0}

    def clear_cache(self) -> None:
        """Clear the embedding cache"""
        if self.enable_cache and self.cache:
            self.cache.clear()
            logger.info("Embedding cache cleared")
