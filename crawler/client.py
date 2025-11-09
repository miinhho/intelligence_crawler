"""
Convenience functions for the intelligent web crawler.
"""

import asyncio
from typing import Any

from .crawler_engine import CrawlConfig, IntelligentCrawler


def crawl(
    seed_urls: list[str],
    topic: str,
    max_pages: int = 100,
    max_depth: int = 3,
    request_delay: float = 1.0,
    max_concurrent_requests: int = 5,
    relevance_threshold: float = 0.5,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Convenience function to crawl websites with semantic filtering.

    Example:
        >>> from crawler import crawl
        >>> results = crawl(
        ...     seed_urls=["https://example.com"],
        ...     topic="artificial intelligence",
        ...     max_pages=50
        ... )

    Args:
        seed_urls: List of starting URLs
        topic: Topic keywords for semantic filtering
        max_pages: Maximum number of pages to crawl
        max_depth: Maximum crawl depth from seed URLs
        request_delay: Delay between requests to same domain (seconds)
        max_concurrent_requests: Maximum concurrent HTTP requests
        relevance_threshold: Minimum topic relevance (0-1) to include full content
        **kwargs: Additional arguments (embedding_model, device, enable_cache, etc.)

    Returns:
        Dictionary containing:
            - pages: List of crawled page data with metadata
            - links: List of discovered links with relevance scores
            - statistics: Crawl statistics (num_pages, avg_relevance, etc.)
            - graph: NetworkX graph of page relationships
    """
    config = CrawlConfig(
        max_pages=max_pages,
        max_depth=max_depth,
        request_delay=request_delay,
        max_concurrent_requests=max_concurrent_requests,
    )

    crawler = IntelligentCrawler(config)

    return asyncio.run(
        crawler.crawl(
            seed_urls=seed_urls,
            topic=topic,
            relevance_threshold=relevance_threshold,
            **kwargs,
        )
    )


async def crawl_async(
    seed_urls: list[str],
    topic: str,
    max_pages: int = 100,
    max_depth: int = 3,
    request_delay: float = 1.0,
    max_concurrent_requests: int = 5,
    relevance_threshold: float = 0.5,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Async convenience function to crawl websites with semantic filtering.

    Example:
        >>> from crawler import crawl_async
        >>> results = await crawl_async(
        ...     seed_urls=["https://example.com"],
        ...     topic="artificial intelligence",
        ...     max_pages=50
        ... )

    Args:
        seed_urls: List of starting URLs
        topic: Topic keywords for semantic filtering
        max_pages: Maximum number of pages to crawl
        max_depth: Maximum crawl depth from seed URLs
        request_delay: Delay between requests to same domain (seconds)
        max_concurrent_requests: Maximum concurrent HTTP requests
        relevance_threshold: Minimum topic relevance (0-1) to include full content
        **kwargs: Additional arguments (embedding_model, device, enable_cache, etc.)

    Returns:
        Dictionary containing crawl results
    """
    config = CrawlConfig(
        max_pages=max_pages,
        max_depth=max_depth,
        request_delay=request_delay,
        max_concurrent_requests=max_concurrent_requests,
    )

    crawler = IntelligentCrawler(config)

    return await crawler.crawl(
        seed_urls=seed_urls,
        topic=topic,
        relevance_threshold=relevance_threshold,
        **kwargs,
    )
