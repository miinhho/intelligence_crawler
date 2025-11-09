"""
Caching system for embeddings and other computed values
"""

import hashlib
import time
from typing import Optional, Any, Dict, Tuple
from functools import lru_cache
import numpy as np


class EmbeddingCache:
    """LRU cache for text embeddings"""

    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        """
        Initialize embedding cache

        Args:
            max_size: Maximum number of cached entries
            ttl: Time-to-live in seconds (0 = no expiry)
        """
        self.max_size = max_size
        self.ttl = ttl
        self._cache: Dict[str, Tuple[np.ndarray, float]] = {}
        self._access_times: Dict[str, float] = {}

    def _get_key(self, text: str) -> str:
        """Generate cache key from text"""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> Optional[np.ndarray]:
        """
        Get cached embedding

        Args:
            text: Input text

        Returns:
            Cached embedding or None if not found/expired
        """
        key = self._get_key(text)

        if key not in self._cache:
            return None

        embedding, timestamp = self._cache[key]

        # Check TTL
        if self.ttl > 0 and (time.time() - timestamp) > self.ttl:
            del self._cache[key]
            del self._access_times[key]
            return None

        # Update access time for LRU
        self._access_times[key] = time.time()

        return embedding

    def set(self, text: str, embedding: np.ndarray) -> None:
        """
        Cache embedding

        Args:
            text: Input text
            embedding: Computed embedding
        """
        key = self._get_key(text)

        # Evict LRU if cache is full
        if len(self._cache) >= self.max_size and key not in self._cache:
            self._evict_lru()

        self._cache[key] = (embedding.copy(), time.time())
        self._access_times[key] = time.time()

    def _evict_lru(self) -> None:
        """Evict least recently used entry"""
        if not self._access_times:
            return

        lru_key = min(self._access_times.items(), key=lambda x: x[1])[0]
        del self._cache[lru_key]
        del self._access_times[lru_key]

    def clear(self) -> None:
        """Clear all cached entries"""
        self._cache.clear()
        self._access_times.clear()

    def size(self) -> int:
        """Get current cache size"""
        return len(self._cache)

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "ttl": self.ttl,
            "oldest_entry": (
                min(self._cache.values(), key=lambda x: x[1])[1]
                if self._cache
                else None
            ),
        }


@lru_cache(maxsize=1000)
def cached_text_hash(text: str) -> str:
    """
    Cached text hashing

    Args:
        text: Input text

    Returns:
        SHA256 hash
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class SimpleCache:
    """Simple generic cache with LRU eviction"""

    def __init__(self, max_size: int = 100):
        """
        Initialize cache

        Args:
            max_size: Maximum number of cached entries
        """
        self.max_size = max_size
        self._cache: Dict[str, Any] = {}
        self._access_order: list[str] = []

    def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        if key in self._cache:
            # Move to end (most recently used)
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None

    def set(self, key: str, value: Any) -> None:
        """Set cached value"""
        if key in self._cache:
            # Update existing
            self._cache[key] = value
            self._access_order.remove(key)
            self._access_order.append(key)
        else:
            # Add new
            if len(self._cache) >= self.max_size:
                # Evict LRU
                lru_key = self._access_order.pop(0)
                del self._cache[lru_key]

            self._cache[key] = value
            self._access_order.append(key)

    def clear(self) -> None:
        """Clear cache"""
        self._cache.clear()
        self._access_order.clear()

    def size(self) -> int:
        """Get cache size"""
        return len(self._cache)
