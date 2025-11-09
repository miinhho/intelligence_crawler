"""
Configuration management using Pydantic
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class CrawlSettings(BaseModel):
    """Crawling configuration"""

    max_depth: int = Field(default=3, ge=0, le=10, description="Maximum crawl depth")
    max_pages: int = Field(
        default=100, ge=1, le=10000, description="Maximum number of pages to crawl"
    )
    request_delay: float = Field(
        default=1.0, ge=0.0, le=10.0, description="Delay between requests (seconds)"
    )
    timeout: int = Field(
        default=30, ge=5, le=300, description="HTTP request timeout (seconds)"
    )
    max_concurrent_requests: int = Field(
        default=5, ge=1, le=50, description="Maximum concurrent requests"
    )
    respect_robots_txt: bool = Field(
        default=True, description="Respect robots.txt rules"
    )
    user_agent: str = Field(
        default="IntelligentCrawler/1.0",
        description="User-Agent header",
    )

    @field_validator("request_delay")
    @classmethod
    def validate_request_delay(cls, v: float) -> float:
        if v < 0.1 and v != 0:
            raise ValueError("request_delay should be at least 0.1 seconds or 0")
        return v


class NLPSettings(BaseModel):
    """NLP processing configuration"""

    embedding_model: str = Field(
        default="jhgan/ko-sroberta-multitask",
        description="Sentence transformer model name",
    )
    summarization_model: Optional[str] = Field(
        default=None, description="Summarization model name (optional)"
    )
    device: Optional[str] = Field(
        default=None, description="Device to use (cuda, cpu, or None for auto)"
    )
    batch_size: int = Field(
        default=32, ge=1, le=256, description="Batch size for embedding"
    )
    max_content_length: int = Field(
        default=5000, ge=100, le=50000, description="Maximum content length to process"
    )
    summary_max_sentences: int = Field(
        default=3, ge=1, le=10, description="Maximum sentences in extractive summary"
    )
    relevance_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum topic relevance threshold",
    )
    link_relevance_threshold: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Minimum link relevance threshold",
    )


class CacheSettings(BaseModel):
    """Caching configuration"""

    enable_embedding_cache: bool = Field(
        default=True, description="Enable embedding caching"
    )
    cache_max_size: int = Field(
        default=1000, ge=0, le=100000, description="Maximum cache entries"
    )
    cache_ttl: int = Field(
        default=3600, ge=0, description="Cache TTL in seconds (0 = no expiry)"
    )


class LoggingSettings(BaseModel):
    """Logging configuration"""

    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format",
    )
    file: Optional[str] = Field(default=None, description="Log file path")

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"level must be one of {valid_levels}")
        return v.upper()


class Settings(BaseSettings):
    """Main application settings"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    # Sub-configurations
    crawl: CrawlSettings = Field(default_factory=CrawlSettings)
    nlp: NLPSettings = Field(default_factory=NLPSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    # Global settings
    debug: bool = Field(default=False, description="Enable debug mode")
    use_uvloop: bool = Field(default=False, description="Use uvloop event loop")

    def model_post_init(self, __context) -> None:
        """Post-initialization validation"""
        # Ensure link_relevance_threshold <= relevance_threshold
        if self.nlp.link_relevance_threshold > self.nlp.relevance_threshold:
            self.nlp.link_relevance_threshold = self.nlp.relevance_threshold


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get global settings instance"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def load_settings(settings: Settings) -> None:
    """Load custom settings"""
    global _settings
    _settings = settings
