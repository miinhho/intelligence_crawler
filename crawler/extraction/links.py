"""Link extraction and analysis from HTML."""

import logging
from typing import TypedDict
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class LinkData(TypedDict):
    """Type definition for extracted link data"""

    url: str
    anchor_text: str
    context: str
    is_internal: bool


def extract_links(soup: BeautifulSoup, base_url: str) -> list[LinkData]:
    """Extract all links with anchor text and context"""
    links: list[LinkData] = []
    base_domain = urlparse(base_url).netloc

    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]

        # Ensure href is a string
        if not isinstance(href, str):
            href = str(href) if href else ""

        # Skip invalid links
        if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
            continue

        # Resolve relative URLs
        absolute_url = urljoin(base_url, href)

        # Skip non-HTTP(S) URLs
        parsed = urlparse(absolute_url)
        if parsed.scheme not in ("http", "https"):
            continue

        # Extract anchor text
        anchor_text = anchor.get_text(strip=True)
        if not anchor_text:
            # Try aria-label or title
            aria_label = anchor.get("aria-label")
            title_attr = anchor.get("title")
            anchor_text = (
                str(aria_label)
                if aria_label
                else (str(title_attr) if title_attr else "")
            )

        # Extract surrounding context
        context = extract_link_context(anchor)

        # Determine if internal or external
        link_domain = parsed.netloc
        is_internal = link_domain == base_domain

        link_data: LinkData = {
            "url": absolute_url,
            "anchor_text": anchor_text,
            "context": context,
            "is_internal": is_internal,
        }
        links.append(link_data)

    return links


def extract_link_context(anchor_tag, max_chars: int = 200) -> str:
    """Extract text context around a link"""
    # Get parent element
    parent = anchor_tag.parent
    if not parent:
        return ""

    # Get all text from parent
    parent_text = parent.get_text(strip=True)

    # Find anchor text position
    anchor_text = anchor_tag.get_text(strip=True)
    if not anchor_text or anchor_text not in parent_text:
        return parent_text[:max_chars]

    # Get context around anchor
    anchor_pos = parent_text.index(anchor_text)
    start = max(0, anchor_pos - max_chars // 2)
    end = min(len(parent_text), anchor_pos + len(anchor_text) + max_chars // 2)

    context = parent_text[start:end].strip()
    return context
