"""
ChronoEye Infinity - Phase 6: Spatio-Temporal Graph Serializer
Provides JSON-compatible serialization and deserialization for the spatio-temporal traffic graph.
"""

import json
from typing import Dict, Any
from app.graph.graph_builder import SpatioTemporalGraphBuilder


class GraphSerializer:
    """
    Graph Serializer / Deserializer Engine.
    Converts NetworkX traffic graph structures to and from JSON.
    """

    @staticmethod
    def to_dict(builder: SpatioTemporalGraphBuilder) -> Dict[str, Any]:
        """
        Converts spatio-temporal traffic graph to JSON-compatible dictionary.
        """
        nodes_list = []
        for n, d in builder.graph.nodes(data=True):
            nodes_list.append({
                "id": n,
                "node_type": d.get("node_type"),
                "data": d.get("data", {}),
            })

        edges_list = []
        for u, v, k, d in builder.graph.edges(data=True, keys=True):
            edges_list.append({
                "source": u,
                "target": v,
                "key": k,
                "edge_type": d.get("edge_type"),
                "timestamp": d.get("timestamp", 0.0),
                "data": d.get("data", {}),
            })

        return {
            "version": "1.0",
            "node_count": len(nodes_list),
            "edge_count": len(edges_list),
            "nodes": nodes_list,
            "edges": edges_list,
        }

    @staticmethod
    def to_json(builder: SpatioTemporalGraphBuilder) -> str:
        """
        Converts spatio-temporal traffic graph to JSON string.
        """
        return json.dumps(GraphSerializer.to_dict(builder), indent=2)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> SpatioTemporalGraphBuilder:
        """
        Reconstructs SpatioTemporalGraphBuilder from dictionary representation.
        """
        builder = SpatioTemporalGraphBuilder()
        builder.graph.clear()

        for n in data.get("nodes", []):
            builder.graph.add_node(
                n["id"],
                node_type=n.get("node_type"),
                data=n.get("data", {}),
            )

        for e in data.get("edges", []):
            builder.graph.add_edge(
                e["source"],
                e["target"],
                key=e.get("key"),
                edge_type=e.get("edge_type"),
                timestamp=e.get("timestamp", 0.0),
                data=e.get("data", {}),
            )

        return builder

    @staticmethod
    def from_json(json_str: str) -> SpatioTemporalGraphBuilder:
        """
        Reconstructs SpatioTemporalGraphBuilder from JSON string.
        """
        data = json.loads(json_str)
        return GraphSerializer.from_dict(data)
