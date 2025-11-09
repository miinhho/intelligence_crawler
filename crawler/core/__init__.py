"""Core utilities and configurations."""

from .cache import EmbeddingCache, SimpleCache, cached_text_hash
from .config import (
    CacheSettings,
    CrawlSettings,
    LoggingSettings,
    NLPSettings,
    Settings,
    get_settings,
    load_settings,
)
from .exceptions import (
    ConfigurationException,
    ContentExtractionException,
    CrawlerException,
    EmbeddingException,
    GraphException,
    HTMLParseException,
    HTTPException,
    ModelLoadException,
    NLPException,
    NetworkException,
    RobotsTxtException,
    SummarizationException,
    TimeoutException,
    ValidationException,
)
from .performance import PerformanceMonitor, async_timed, timed

__all__ = [
    # Cache
    "EmbeddingCache",
    "SimpleCache",
    "cached_text_hash",
    # Config
    "Settings",
    "CrawlSettings",
    "NLPSettings",
    "CacheSettings",
    "LoggingSettings",
    "get_settings",
    "load_settings",
    # Exceptions
    "CrawlerException",
    "NetworkException",
    "HTTPException",
    "TimeoutException",
    "RobotsTxtException",
    "ContentExtractionException",
    "HTMLParseException",
    "NLPException",
    "ModelLoadException",
    "EmbeddingException",
    "SummarizationException",
    "GraphException",
    "ConfigurationException",
    "ValidationException",
    # Performance
    "PerformanceMonitor",
    "timed",
    "async_timed",
]
