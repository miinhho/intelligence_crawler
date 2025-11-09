"""
Core crawler engine using established libraries.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import aiohttp
from aiohttp_retry import ExponentialRetry, RetryClient
from aiolimiter import AsyncLimiter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CrawlConfig:
    """Configuration for crawler behavior"""

    max_depth: int = 3
    max_pages: int = 100
    request_delay: float = 1.0
    timeout: int = 30
    user_agent: str = "IntelligentCrawler/1.0"
    respect_robots_txt: bool = True
    max_concurrent_requests: int = 5


@dataclass
class URLMetadata:
    """Metadata for a URL to be crawled"""

    url: str
    depth: int
    parent_url: Optional[str] = None
    anchor_text: Optional[str] = None
    context: Optional[str] = None
    priority: float = 1.0


class URLManager:
    """Manages URL queue, visited set, and URL filtering"""

    def __init__(self, config: CrawlConfig):
        self.config = config
        self.visited: set[str] = set()
        self.queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._url_metadata: dict[str, URLMetadata] = {}

    def add_url(self, metadata: URLMetadata) -> None:
        """Add URL to queue if not visited"""
        if metadata.url not in self.visited and metadata.url not in self._url_metadata:
            self._url_metadata[metadata.url] = metadata
            self.queue.put_nowait((-metadata.priority, metadata.url))

    async def get_next(self) -> Optional[URLMetadata]:
        """Get next URL to crawl"""
        try:
            _, url = await asyncio.wait_for(self.queue.get(), timeout=0.1)
            return self._url_metadata.get(url)
        except asyncio.TimeoutError:
            return None

    def mark_visited(self, url: str) -> None:
        """Mark URL as visited"""
        self.visited.add(url)

    def is_visited(self, url: str) -> bool:
        """Check if URL has been visited"""
        return url in self.visited

    def get_queue_size(self) -> int:
        """Get current queue size"""
        return self.queue.qsize()

    def normalize_url(self, url: str) -> str:
        """Normalize URL by removing fragments and trailing slashes"""
        parsed = urlparse(url)
        normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if parsed.query:
            normalized += f"?{parsed.query}"
        if normalized.endswith("/") and parsed.path != "/":
            normalized = normalized[:-1]
        return normalized

    def is_internal_link(self, base_url: str, link_url: str) -> bool:
        """Check if link is internal (same domain)"""
        base_domain = urlparse(base_url).netloc
        link_domain = urlparse(link_url).netloc
        return base_domain == link_domain


class IntelligentCrawler:
    """Main crawler orchestrator"""

    def __init__(
        self, config: Optional[CrawlConfig] = None, enable_profiling: bool = False
    ):
        self.config = config or CrawlConfig()
        self.url_manager = URLManager(self.config)
        self.pages_crawled = 0
        self.enable_profiling = enable_profiling

        # Libraries for HTTP requests
        self._session: aiohttp.ClientSession | None = None
        self._retry_client: RetryClient | None = None
        self._rate_limiters: dict[str, AsyncLimiter] = {}
        self._robots_cache: dict[str, RobotFileParser] = {}

        # Performance monitoring
        if enable_profiling:
            from .core.performance import PerformanceMonitor

            self.perf = PerformanceMonitor()
        else:
            self.perf = None

    def _get_rate_limiter(self, domain: str) -> AsyncLimiter:
        """Get or create rate limiter for domain"""
        if domain not in self._rate_limiters:
            max_rate = 1.0 / self.config.request_delay
            self._rate_limiters[domain] = AsyncLimiter(max_rate, 1.0)
        return self._rate_limiters[domain]

    async def _check_robots_txt(self, url: str) -> bool:
        """Check if URL is allowed by robots.txt"""
        if not self.config.respect_robots_txt:
            return True

        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

        if robots_url not in self._robots_cache:
            parser = RobotFileParser()
            parser.set_url(robots_url)
            try:
                if self._session is None:
                    return True

                async with self._session.get(
                    robots_url, timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        content = await response.text()
                        parser.parse(content.splitlines())
                    else:
                        parser.parse([])
            except Exception:
                parser.parse([])

            self._robots_cache[robots_url] = parser

        return self._robots_cache[robots_url].can_fetch(self.config.user_agent, url)

    async def _fetch_url(self, url: str) -> Optional[str]:
        """Fetch URL with retry, rate limiting, and robots.txt check"""
        if not await self._check_robots_txt(url):
            logger.info(f"Blocked by robots.txt: {url}")
            return None

        if self._retry_client is None:
            return None

        domain = urlparse(url).netloc
        limiter = self._get_rate_limiter(domain)

        async with limiter:
            try:
                async with self._retry_client.get(url) as response:
                    if response.status == 200:
                        content_type = response.headers.get("Content-Type", "")
                        if "text/html" in content_type:
                            return await response.text()
                        logger.info(f"Skipping non-HTML: {url}")
                    else:
                        logger.warning(f"HTTP {response.status}: {url}")
            except asyncio.TimeoutError:
                logger.warning(f"Timeout: {url}")
            except Exception as e:
                logger.error(f"Error fetching {url}: {type(e).__name__}")

            return None

    async def _process_single_url(
        self,
        url_metadata: URLMetadata,
        topic: str,
        content_extractor: Any,
        nlp_processor: Any,
        graph_manager: Any,
    ) -> Optional[dict[str, Any]]:
        """Process a single URL: fetch, extract, analyze, and add to graph"""
        url = url_metadata.url

        if self.url_manager.is_visited(url):
            return None

        if url_metadata.depth > self.config.max_depth:
            return None

        logger.info(f"Crawling [{url_metadata.depth}]: {url}")

        # Measure fetch time
        if self.perf:
            with self.perf.measure("fetch_url"):
                html_content = await self._fetch_url(url)
                self.perf.increment("urls_fetched")
        else:
            html_content = await self._fetch_url(url)

        if html_content is None:
            self.url_manager.mark_visited(url)
            return None

        # Measure extraction time
        if self.perf:
            with self.perf.measure("extract_content"):
                extracted = content_extractor.extract(html_content, url)
        else:
            extracted = content_extractor.extract(html_content, url)

        # Measure NLP processing time
        if self.perf:
            with self.perf.measure("nlp_processing"):
                nlp_result = await nlp_processor.process_page(
                    extracted["main_content"], topic, extracted["title"]
                )
                self.perf.increment("pages_processed")
        else:
            nlp_result = await nlp_processor.process_page(
                extracted["main_content"], topic, extracted["title"]
            )

        if nlp_result["topic_relevance"] < 0.5:
            logger.info(
                f"Low relevance ({nlp_result['topic_relevance']:.2f}), skipping: {url}"
            )
            self.url_manager.mark_visited(url)
            return None

        graph_manager.add_page(
            url=url,
            title=extracted["title"],
            summary=nlp_result["summary"],
            full_content=extracted["main_content"],
            embedding=nlp_result["embedding"],
            relevance=nlp_result["topic_relevance"],
            depth=url_metadata.depth,
        )

        self.url_manager.mark_visited(url)
        self.pages_crawled += 1

        return {
            "url": url,
            "url_metadata": url_metadata,
            "extracted": extracted,
            "nlp_result": nlp_result,
        }

    async def crawl(
        self,
        seed_urls: list[str],
        topic: str,
        relevance_threshold: float = 0.5,
        embedding_model: str = "jhgan/ko-sroberta-multitask",
        summarization_model: Optional[str] = None,
        device: Optional[str] = None,
        enable_cache: bool = True,
        cache_maxsize: int = 1000,
        cache_ttl: int = 3600,
    ) -> dict[str, Any]:
        """
        Main crawl loop with semantic filtering and parallel URL processing

        Args:
            seed_urls: Initial URLs to start crawling
            topic: Topic to filter pages by
            relevance_threshold: Minimum relevance to include full content
            embedding_model: Sentence transformer model for embeddings
            summarization_model: Model for text summarization (optional)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            enable_cache: Enable embedding cache for performance
            cache_maxsize: Maximum number of cached embeddings
            cache_ttl: Cache time-to-live in seconds

        Returns:
            Dictionary with crawl results
        """
        from .extraction import ContentExtractor
        from .graph import GraphManager
        from .nlp import NLPProcessor

        content_extractor = ContentExtractor()
        nlp_processor = NLPProcessor(
            embedding_model=embedding_model,
            summarization_model=summarization_model,
            device=device,
            enable_cache=enable_cache,
            cache_maxsize=cache_maxsize,
            cache_ttl=cache_ttl,
        )
        graph_manager = GraphManager()

        for url in seed_urls:
            normalized_url = self.url_manager.normalize_url(url)
            metadata = URLMetadata(url=normalized_url, depth=0, priority=1.0)
            self.url_manager.add_url(metadata)

        # Setup HTTP client with retry
        headers = {
            "User-Agent": self.config.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
        }
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        self._session = aiohttp.ClientSession(headers=headers, timeout=timeout)

        retry_options = ExponentialRetry(attempts=3, start_timeout=1)
        self._retry_client = RetryClient(self._session, retry_options=retry_options)

        try:
            while self.pages_crawled < self.config.max_pages:
                batch_urls: list[URLMetadata] = []
                for _ in range(
                    min(
                        self.config.max_concurrent_requests,
                        self.config.max_pages - self.pages_crawled,
                    )
                ):
                    next_url = await self.url_manager.get_next()
                    if next_url is None:
                        break
                    batch_urls.append(next_url)

                if not batch_urls:
                    if self.url_manager.get_queue_size() == 0:
                        logger.info("Queue is empty, crawl complete")
                        break
                    await asyncio.sleep(0.1)
                    continue

                tasks = [
                    self._process_single_url(
                        url_metadata,
                        topic,
                        content_extractor,
                        nlp_processor,
                        graph_manager,
                    )
                    for url_metadata in batch_urls
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for result in results:
                    if isinstance(result, Exception):
                        logger.error(f"Error processing URL: {result}")
                        continue

                    if result is None or not isinstance(result, dict):
                        continue

                    url: str = result["url"]
                    url_metadata: URLMetadata = result["url_metadata"]
                    extracted: dict[str, Any] = result["extracted"]
                    nlp_result: dict[str, Any] = result["nlp_result"]

                    links_to_process = [
                        link
                        for link in extracted["links"]
                        if not self.url_manager.is_visited(
                            self.url_manager.normalize_url(link["url"])
                        )
                    ]

                    if links_to_process:
                        link_relevance_tasks = [
                            nlp_processor.calculate_link_relevance(
                                nlp_result["summary"],
                                link["anchor_text"],
                                link["context"],
                            )
                            for link in links_to_process
                        ]
                        link_relevances = await asyncio.gather(*link_relevance_tasks)

                        for link, link_relevance in zip(
                            links_to_process, link_relevances
                        ):
                            link_url = self.url_manager.normalize_url(link["url"])

                            graph_manager.add_edge(
                                source=url,
                                target=link_url,
                                anchor_text=link["anchor_text"],
                                relevance=link_relevance,
                                is_internal=link["is_internal"],
                            )

                            if (
                                link_relevance > 0.4
                                and url_metadata.depth < self.config.max_depth
                            ):
                                new_metadata = URLMetadata(
                                    url=link_url,
                                    depth=url_metadata.depth + 1,
                                    parent_url=url,
                                    anchor_text=link["anchor_text"],
                                    context=link["context"],
                                    priority=link_relevance,
                                )
                                self.url_manager.add_url(new_metadata)

                logger.info(
                    f"Progress: {self.pages_crawled}/{self.config.max_pages} pages, "
                    f"Queue: {self.url_manager.get_queue_size()}"
                )
        finally:
            if self._retry_client:
                await self._retry_client.close()
            if self._session:
                await self._session.close()

        # Collect performance metrics
        results = graph_manager.get_results(relevance_threshold=relevance_threshold)

        if self.perf:
            # Record cache stats
            cache_stats = nlp_processor.get_cache_stats()
            self.perf.record_cache_stats("embedding_cache", cache_stats)

            # Add performance summary to results
            results["performance"] = self.perf.get_summary()
            self.perf.print_summary()

        return results
