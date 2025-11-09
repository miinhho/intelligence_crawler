"""Graph analytics and algorithms."""

import logging
from typing import Any

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity"""
    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)

    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))


def get_related_pages(pages: dict, source_url: str, top_k: int = 10) -> list[tuple]:
    """
    Get most semantically similar pages

    Args:
        pages: Dictionary of PageNode objects
        source_url: Source page URL
        top_k: Number of similar pages to return

    Returns:
        list of (url, similarity_score) tuples
    """
    source_page = pages.get(source_url)
    if not source_page:
        return []

    similarities = []
    source_embedding = source_page.embedding

    for other_url, other_page in pages.items():
        if other_url == source_url:
            continue

        similarity = cosine_similarity(source_embedding, other_page.embedding)
        similarities.append((other_url, float(similarity)))

    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


def calculate_pagerank(graph: nx.DiGraph) -> dict[str, float]:
    """Calculate PageRank for all pages"""
    if not graph.nodes():
        return {}
    return nx.pagerank(graph)


def calculate_hits(graph: nx.DiGraph) -> tuple[dict[str, float], dict[str, float]]:
    """Calculate HITS algorithm scores"""
    if not graph.nodes():
        return {}, {}
    hubs, authorities = nx.hits(graph)
    return hubs, authorities


def get_strongly_connected_components(graph: nx.DiGraph) -> list[set[str]]:
    """Find strongly connected components"""
    if not graph.nodes():
        return []
    return list(nx.strongly_connected_components(graph))


def get_shortest_path(graph: nx.DiGraph, source: str, target: str) -> list[str] | None:
    """Find shortest path between two pages"""
    if source not in graph or target not in graph:
        return None

    try:
        return nx.shortest_path(graph, source, target)
    except nx.NetworkXNoPath:
        return None


def find_clusters(graph: nx.DiGraph) -> dict[str, int]:
    """Find communities/clusters in the graph"""
    undirected = graph.to_undirected()
    from networkx.algorithms import community

    communities_generator = community.label_propagation_communities(undirected)
    communities = {}
    for i, comm in enumerate(communities_generator):
        for node in comm:
            communities[node] = i
    return communities


def get_graph_statistics(graph, pages: dict, edges: list) -> dict[str, Any]:
    """Get graph statistics"""
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()

    # Calculate avg_degree properly
    if num_nodes > 0:
        degrees = dict(graph.degree())
        avg_degree = sum(degrees.values()) / num_nodes
    else:
        avg_degree = 0

    stats = {
        "num_pages": num_nodes,
        "num_links": num_edges,
        "density": nx.density(graph) if num_nodes > 0 else 0,
        "avg_degree": avg_degree,
    }

    # Calculate average relevance
    if pages:
        avg_relevance = sum(p.relevance for p in pages.values()) / len(pages)
        stats["avg_topic_relevance"] = avg_relevance

    # Internal vs external links
    internal_links = sum(1 for e in edges if e.is_internal)
    external_links = len(edges) - internal_links
    stats["internal_links"] = internal_links
    stats["external_links"] = external_links

    return stats
