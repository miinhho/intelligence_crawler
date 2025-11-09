"""Intelligent Web Crawler Package"""

from .client import crawl, crawl_async
from .core.config import (
    CacheSettings,
    CrawlSettings,
    LoggingSettings,
    NLPSettings,
    Settings,
    get_settings,
)
from .crawler_engine import CrawlConfig, IntelligentCrawler
from .extraction import ContentExtractor, LinkData
from .graph import GraphManager
from .nlp import NLPProcessor

__all__ = [
    # Public API
    "crawl",
    "crawl_async",
    "IntelligentCrawler",
    "CrawlConfig",
    # Components (advanced usage)
    "ContentExtractor",
    "GraphManager",
    "NLPProcessor",
    # Configuration
    "CacheSettings",
    "CrawlSettings",
    "LoggingSettings",
    "NLPSettings",
    "Settings",
    "get_settings",
    # Types
    "LinkData",
]
