"""Graph export utilities."""

import logging

import networkx as nx

logger = logging.getLogger(__name__)


def export_graph(graph: nx.DiGraph, format: str = "gexf") -> str:
    """
    Export graph to file format

    Args:
        format: Export format ('gexf', 'graphml', 'gml')

    Returns:
        Serialized graph data
    """
    if format == "gexf":
        import io

        buffer = io.BytesIO()
        nx.write_gexf(graph, buffer)
        return buffer.getvalue().decode("utf-8")
    elif format == "graphml":
        import io

        buffer = io.BytesIO()
        nx.write_graphml(graph, buffer)
        return buffer.getvalue().decode("utf-8")
    elif format == "gml":
        import io

        buffer = io.StringIO()
        nx.write_gml(graph, buffer)
        return buffer.getvalue()
    else:
        raise ValueError(f"Unsupported format: {format}")
