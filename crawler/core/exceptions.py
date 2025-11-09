"""
Custom exceptions for the intelligent crawler
"""


class CrawlerException(Exception):
    """Base exception for all crawler errors"""

    pass


class NetworkException(CrawlerException):
    """Network-related errors"""

    pass


class HTTPException(NetworkException):
    """HTTP request errors"""

    def __init__(self, status_code: int, url: str, message: str = ""):
        self.status_code = status_code
        self.url = url
        super().__init__(message or f"HTTP {status_code} for {url}")


class TimeoutException(NetworkException):
    """Request timeout errors"""

    def __init__(self, url: str, timeout: float):
        self.url = url
        self.timeout = timeout
        super().__init__(f"Timeout after {timeout}s for {url}")


class RobotsTxtException(NetworkException):
    """Robots.txt related errors"""

    pass


class ContentExtractionException(CrawlerException):
    """Content extraction errors"""

    pass


class HTMLParseException(ContentExtractionException):
    """HTML parsing errors"""

    pass


class NLPException(CrawlerException):
    """NLP processing errors"""

    pass


class ModelLoadException(NLPException):
    """Model loading errors"""

    pass


class EmbeddingException(NLPException):
    """Embedding generation errors"""

    pass


class SummarizationException(NLPException):
    """Text summarization errors"""

    pass


class GraphException(CrawlerException):
    """Graph management errors"""

    pass


class ConfigurationException(CrawlerException):
    """Configuration errors"""

    pass


class ValidationException(CrawlerException):
    """Data validation errors"""

    pass
