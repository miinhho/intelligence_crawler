"""Graph storage and basic operations."""

import logging

import networkx as nx

from .models import LinkEdge, PageNode

logger = logging.getLogger(__name__)


class GraphStorage:
    """Manages in-memory storage of pages and links"""

    def __init__(self):
        self.graph = nx.DiGraph()
        self.pages: dict[str, PageNode] = {}
        self.edges: list[LinkEdge] = []

    def add_page(
        self,
        url: str,
        title: str,
        summary: str,
        full_content: str,
        embedding,
        relevance: float,
        depth: int,
    ):
        """Add a page node to the graph"""
        page = PageNode(
            url=url,
            title=title,
            summary=summary,
            full_content=full_content,
            embedding=embedding,
            relevance=relevance,
            depth=depth,
        )

        self.pages[url] = page

        self.graph.add_node(
            url,
            title=title,
            summary=summary,
            full_content=full_content,
            relevance=relevance,
            depth=depth,
        )

        logger.debug(f"Added page: {url}")

    def add_edge(
        self,
        source: str,
        target: str,
        anchor_text: str,
        relevance: float,
        is_internal: bool,
    ):
        """Add a link edge to the graph"""
        edge = LinkEdge(
            source=source,
            target=target,
            anchor_text=anchor_text,
            relevance=relevance,
            is_internal=is_internal,
        )

        self.edges.append(edge)

        self.graph.add_edge(
            source,
            target,
            anchor_text=anchor_text,
            relevance=relevance,
            is_internal=is_internal,
        )

        logger.debug(f"Added edge: {source} -> {target}")

    def get_page(self, url: str) -> PageNode | None:
        """Get page node by URL"""
        return self.pages.get(url)

    def get_outgoing_links(self, url: str) -> list[LinkEdge]:
        """Get all outgoing links from a page"""
        return [edge for edge in self.edges if edge.source == url]

    def get_incoming_links(self, url: str) -> list[LinkEdge]:
        """Get all incoming links to a page"""
        return [edge for edge in self.edges if edge.target == url]

    def get_subgraph_by_relevance(self, min_relevance: float = 0.5) -> nx.DiGraph:
        """Get subgraph containing only high-relevance pages"""
        relevant_nodes = [
            url for url, page in self.pages.items() if page.relevance >= min_relevance
        ]
        subgraph = self.graph.subgraph(relevant_nodes)
        return nx.DiGraph(subgraph)
