"""Content extraction facade - orchestrates content, metadata, and link extraction."""

import logging
from typing import TypedDict

from bs4 import BeautifulSoup

from .content import extract_main_content, extract_paragraphs
from .links import LinkData, extract_links
from .metadata import extract_description, extract_title

logger = logging.getLogger(__name__)


class ExtractedContent(TypedDict):
    """Type definition for extracted content"""

    title: str
    description: str
    main_content: str
    links: list[LinkData]
    url: str


class ContentExtractor:
    """Orchestrates extraction of content, metadata, and links from HTML"""

    def extract(self, html: str, base_url: str) -> ExtractedContent:
        """
        Extract all relevant information from HTML

        Args:
            html: Raw HTML content
            base_url: Base URL for resolving relative links

        Returns:
            Dictionary with extracted content, links, and metadata
        """
        soup = BeautifulSoup(html, "lxml")

        # Extract main content
        main_content = extract_main_content(html, base_url)

        # Extract metadata
        title = extract_title(soup)
        description = extract_description(soup)

        # Extract and analyze links
        links = extract_links(soup, base_url)

        return ExtractedContent(
            title=title,
            description=description,
            main_content=main_content,
            links=links,
            url=base_url,
        )

    def extract_paragraphs(self, html: str) -> list[str]:
        """Extract individual paragraphs for paragraph-level analysis"""
        return extract_paragraphs(html)
