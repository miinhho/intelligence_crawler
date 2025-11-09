"""Main content extraction using trafilatura."""

import logging
import re

import trafilatura
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def extract_main_content(html: str, url: str) -> str:
    """Extract main text content using trafilatura"""
    # Use trafilatura for main content extraction
    extracted = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True,
        no_fallback=False,
        url=url,
    )

    if extracted:
        # Clean up whitespace
        extracted = re.sub(r"\s+", " ", extracted).strip()
        return extracted

    # Fallback to basic extraction
    soup = BeautifulSoup(html, "lxml")
    # Remove script and style elements
    for script in soup(["script", "style", "nav", "footer", "header"]):
        script.decompose()

    text = soup.get_text()
    # Clean up whitespace
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = " ".join(chunk for chunk in chunks if chunk)
    return text


def extract_paragraphs(html: str) -> list[str]:
    """Extract individual paragraphs for paragraph-level analysis"""
    soup = BeautifulSoup(html, "lxml")

    # Remove unwanted elements
    for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
        element.decompose()

    paragraphs = []

    # Extract from paragraph tags
    for p in soup.find_all(["p", "div"]):
        text = p.get_text(strip=True)
        # Filter out very short paragraphs
        if len(text) > 50:  # Minimum length
            paragraphs.append(text)

    return paragraphs
