"""Metadata extraction from HTML (title, description, etc.)."""

import logging

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def extract_title(soup: BeautifulSoup) -> str:
    """Extract page title"""
    title: str = ""

    # Try og:title meta tag
    og_title = soup.find("meta", property="og:title")
    if og_title:
        content = og_title.get("content")
        if content and isinstance(content, str):
            title = content

    # Try title tag
    if not title:
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text()

    # Try h1
    if not title:
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text()

    if title:
        return title.strip()
    return "Untitled"


def extract_description(soup: BeautifulSoup) -> str:
    """Extract page description"""
    # Try meta description
    meta_desc = soup.find("meta", attrs={"name": "description"})
    if meta_desc:
        content = meta_desc.get("content")
        if content and isinstance(content, str):
            return content.strip()

    # Try og:description
    og_desc = soup.find("meta", property="og:description")
    if og_desc:
        content = og_desc.get("content")
        if content and isinstance(content, str):
            return content.strip()

    return ""
