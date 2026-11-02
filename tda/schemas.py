from __future__ import annotations

from pydantic import BaseModel


class ClusterNode(BaseModel):
    cluster_id: int
    label: str
    size: int
    event_indices: list[int]
    persistence: float


class ClusterEdge(BaseModel):
    source: int
    target: int
    merge_distance: float


class TopologySnapshot(BaseModel):
    nodes: list[ClusterNode]
    edges: list[ClusterEdge]
    event_labels: list[int]
    h0_features: list[dict]
    h1_features: list[dict]
    betti: list[int]
    timestamp: str
