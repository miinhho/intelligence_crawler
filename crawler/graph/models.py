"""Graph data models."""

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PageNode:
    """Represents a crawled page"""

    url: str
    title: str
    summary: str
    full_content: str
    embedding: np.ndarray
    relevance: float
    depth: int
    timestamp: float = field(default_factory=time.time)


@dataclass
class LinkEdge:
    """Represents a link between pages"""

    source: str
    target: str
    anchor_text: str
    relevance: float
    is_internal: bool
