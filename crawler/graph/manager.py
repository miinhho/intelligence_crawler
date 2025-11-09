"""Graph management facade - orchestrates storage, analytics, and export."""

import logging
from typing import Any

import numpy as np

from .analytics import (
    calculate_hits,
    calculate_pagerank,
    find_clusters,
    get_graph_statistics,
    get_related_pages,
    get_shortest_path,
    get_strongly_connected_components,
)
from .export import export_graph
from .models import LinkEdge, PageNode
from .storage import GraphStorage

logger = logging.getLogger(__name__)


class GraphManager:
    """Orchestrates graph storage, analytics, and export"""

    def __init__(self):
        self.storage = GraphStorage()

    def add_page(
        self,
        url: str,
        title: str,
        summary: str,
        full_content: str,
        embedding: np.ndarray,
        relevance: float,
        depth: int,
    ):
        """Add a page node to the graph"""
        self.storage.add_page(
            url, title, summary, full_content, embedding, relevance, depth
        )

    def add_edge(
        self,
        source: str,
        target: str,
        anchor_text: str,
        relevance: float,
        is_internal: bool,
    ):
        """Add a link edge to the graph"""
        self.storage.add_edge(source, target, anchor_text, relevance, is_internal)

    def get_page(self, url: str) -> PageNode | None:
        """Get page node by URL"""
        return self.storage.get_page(url)

    def get_outgoing_links(self, url: str) -> list[LinkEdge]:
        """Get all outgoing links from a page"""
        return self.storage.get_outgoing_links(url)

    def get_incoming_links(self, url: str) -> list[LinkEdge]:
        """Get all incoming links to a page"""
        return self.storage.get_incoming_links(url)

    def get_related_pages(self, url: str, top_k: int = 10) -> list[tuple]:
        """Get most semantically similar pages"""
        return get_related_pages(self.storage.pages, url, top_k)

    def get_page_rank(self) -> dict[str, float]:
        """Calculate PageRank for all pages"""
        return calculate_pagerank(self.storage.graph)

    def get_hub_and_authority_scores(self) -> tuple[dict[str, float], dict[str, float]]:
        """Calculate HITS algorithm scores"""
        return calculate_hits(self.storage.graph)

    def get_strongly_connected_components(self) -> list[set[str]]:
        """Find strongly connected components"""
        return get_strongly_connected_components(self.storage.graph)

    def get_shortest_path(self, source: str, target: str) -> list[str] | None:
        """Find shortest path between two pages"""
        return get_shortest_path(self.storage.graph, source, target)

    def get_statistics(self) -> dict[str, Any]:
        """Get graph statistics"""
        return get_graph_statistics(
            self.storage.graph, self.storage.pages, self.storage.edges
        )

    def get_results(self, relevance_threshold: float = 0.5) -> dict[str, Any]:
        """Get comprehensive crawl results"""
        pagerank = self.get_page_rank()

        pages_data = []
        for url, page in self.storage.pages.items():
            page_data = {
                "url": url,
                "title": page.title,
                "summary": page.summary,
                "topic_relevance": page.relevance,
                "depth": page.depth,
                "pagerank": pagerank.get(url, 0),
                "timestamp": page.timestamp,
            }

            if page.relevance >= relevance_threshold:
                page_data["full_content"] = page.full_content

            pages_data.append(page_data)

        pages_data.sort(key=lambda x: x["topic_relevance"], reverse=True)

        links_data = []
        for edge in self.storage.edges:
            link_data = {
                "source": edge.source,
                "target": edge.target,
                "anchor_text": edge.anchor_text,
                "relevance": edge.relevance,
                "is_internal": edge.is_internal,
            }
            links_data.append(link_data)

        stats = self.get_statistics()

        return {
            "pages": pages_data,
            "links": links_data,
            "statistics": stats,
            "graph": self.storage.graph,
        }

    def export_graph(self, format: str = "gexf") -> str:
        """Export graph to file format"""
        return export_graph(self.storage.graph, format)

    def get_subgraph_by_relevance(self, min_relevance: float = 0.5):
        """Get subgraph containing only high-relevance pages"""
        return self.storage.get_subgraph_by_relevance(min_relevance)

    def find_clusters(self) -> dict[str, int]:
        """Find communities/clusters in the graph"""
        return find_clusters(self.storage.graph)
