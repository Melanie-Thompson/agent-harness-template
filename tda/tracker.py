"""Topological Data Analysis for log event grouping.

Maintains a sliding window of log event embeddings and computes persistent
homology (H0 connected components, H1 loops) over a time-decayed distance
matrix.  Large clusters are pruned via MST skeletonization.

Ported from MCPHarness TopologyTracker, adapted for structured log events
instead of conversation messages.  Adds NetworkX MST trimming from the
Gemini DTCM design.
"""

from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

from logging_config import get_logger

_log = get_logger("tda.tracker")


def log_event_to_text(event: dict) -> str:
    parts = [
        event.get("level", ""),
        event.get("logger", ""),
        event.get("event", ""),
        event.get("function", ""),
    ]
    skip = {"level", "logger", "event", "function", "timestamp", "log_level", "params"}
    extras = [f"{k}={v}" for k, v in event.items() if k not in skip and v is not None]
    parts.extend(extras[:5])
    return " ".join(p for p in parts if p)


def label_cluster(events: list[dict]) -> str:
    prefixes: list[str] = []
    for e in events:
        event_name = e.get("event", "")
        prefix = event_name.split(".")[0] if "." in event_name else event_name
        logger = e.get("logger", "")
        prefixes.append(f"{logger}:{prefix}" if logger else prefix)
    most_common = Counter(prefixes).most_common(1)
    return most_common[0][0] if most_common else "unknown"


@dataclass(frozen=True)
class PersistenceFeature:
    dimension: int
    birth: float
    death: float

    @property
    def lifetime(self) -> float:
        if self.death == float("inf"):
            return 0.0
        return self.death - self.birth


@dataclass(frozen=True)
class TopologySignals:
    window_size: int
    h0_clusters: int
    h1_loops: int
    betti: list[int]
    h0_features: list[PersistenceFeature]
    h1_features: list[PersistenceFeature]
    clusters: list[list[int]]
    loop_indices: list[list[int]]
    distance_matrix: np.ndarray | None = field(default=None, repr=False)

    @staticmethod
    def _safe_float(v: float) -> float | None:
        if v == float("inf") or v == float("-inf") or v != v:
            return None
        return round(v, 4)

    def to_dict(self) -> dict:
        sf = self._safe_float
        return {
            "window_size": self.window_size,
            "h0_clusters": self.h0_clusters,
            "h1_loops": self.h1_loops,
            "betti": self.betti,
            "h0_features": [
                {"dim": f.dimension, "birth": sf(f.birth),
                 "death": sf(f.death), "lifetime": sf(f.lifetime)}
                for f in self.h0_features
            ],
            "h1_features": [
                {"dim": f.dimension, "birth": sf(f.birth),
                 "death": sf(f.death), "lifetime": sf(f.lifetime)}
                for f in self.h1_features
            ],
            "clusters": self.clusters,
            "loop_indices": self.loop_indices,
        }


_EMPTY_SIGNALS = TopologySignals(
    window_size=0, h0_clusters=0, h1_loops=0, betti=[],
    h0_features=[], h1_features=[], clusters=[], loop_indices=[],
)


class LogTopologyTracker:
    """Streaming TDA over a sliding window of log events."""

    def __init__(
        self,
        max_window: int = 200,
        time_decay: float = 0.05,
        max_scale: float = 1.5,
        min_persistence: float = 0.1,
        max_component_nodes: int = 15,
    ):
        self.max_window = max_window
        self.time_decay = time_decay
        self.max_scale = max_scale
        self.min_persistence = min_persistence
        self.max_component_nodes = max_component_nodes

        self._texts: deque[str] = deque(maxlen=max_window)
        self._embeddings: deque[np.ndarray] = deque(maxlen=max_window)
        self._timestamps: deque[float] = deque(maxlen=max_window)
        self._event_dicts: deque[dict] = deque(maxlen=max_window)

    @property
    def size(self) -> int:
        return len(self._texts)

    def add_event(self, event: dict, *, timestamp: float | None = None) -> None:
        if timestamp is None:
            timestamp = time.monotonic()

        text = log_event_to_text(event)
        from tda.embeddings import embed_texts
        emb = embed_texts([text])[0]

        self._texts.append(text)
        self._embeddings.append(emb)
        self._timestamps.append(timestamp)
        self._event_dicts.append(event)

    def _distance_matrix(self) -> np.ndarray:
        from sklearn.metrics.pairwise import cosine_distances

        emb = np.array(self._embeddings)
        ts = np.array(self._timestamps)
        semantic = cosine_distances(emb)

        if self.time_decay > 0:
            time_diff = np.abs(np.subtract.outer(ts, ts))
            decay = 1.0 + self.time_decay * time_diff
            return semantic * decay

        return semantic

    def analyze(self) -> TopologySignals:
        n = len(self._texts)
        if n < 3:
            return TopologySignals(
                window_size=n, h0_clusters=max(n, 0), h1_loops=0,
                betti=[n] if n else [], h0_features=[], h1_features=[],
                clusters=[[i] for i in range(n)], loop_indices=[],
            )

        dist = self._distance_matrix()

        try:
            h0, h1, _betti = self._gudhi_homology(dist)
        except ImportError:
            _log.debug("tda.gudhi_fallback", reason="gudhi not installed, using numpy fallback")
            h0, h1, _betti = self._numpy_homology(dist)

        clusters = self._extract_clusters(dist, h0)
        clusters = [self._skeletonize_cluster(c, dist) for c in clusters]
        loops = self._extract_loops(dist, h1)

        return TopologySignals(
            window_size=n,
            h0_clusters=len(clusters),
            h1_loops=len(loops),
            betti=[len(clusters), len(loops)],
            h0_features=h0,
            h1_features=h1,
            clusters=clusters,
            loop_indices=loops,
            distance_matrix=dist,
        )

    def _gudhi_homology(
        self, dist: np.ndarray,
    ) -> tuple[list[PersistenceFeature], list[PersistenceFeature], list[int]]:
        import gudhi

        rips = gudhi.RipsComplex(
            distance_matrix=dist.tolist(),
            max_edge_length=self.max_scale,
        )
        st = rips.create_simplex_tree(max_dimension=2)
        persistence = st.persistence()

        h0: list[PersistenceFeature] = []
        h1: list[PersistenceFeature] = []
        for dim, (birth, death) in persistence:
            feat = PersistenceFeature(dimension=dim, birth=birth, death=death)
            if feat.lifetime < self.min_persistence and death != float("inf"):
                continue
            if dim == 0:
                h0.append(feat)
            elif dim == 1:
                h1.append(feat)
        return h0, h1, list(st.betti_numbers())

    def _numpy_homology(
        self, dist: np.ndarray,
    ) -> tuple[list[PersistenceFeature], list[PersistenceFeature], list[int]]:
        n = dist.shape[0]

        def find(parent: list[int], x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        edges = sorted(
            (float(dist[i, j]), i, j)
            for i in range(n)
            for j in range(i + 1, n)
            if dist[i, j] <= self.max_scale
        )

        parent = list(range(n))
        h0: list[PersistenceFeature] = []
        cycle_edges: list[tuple[float, int, int]] = []
        for weight, i, j in edges:
            ri, rj = find(parent, i), find(parent, j)
            if ri == rj:
                cycle_edges.append((weight, i, j))
            else:
                parent[ri] = rj
                h0.append(PersistenceFeature(dimension=0, birth=0.0, death=weight))

        for _ in range(len({find(parent, i) for i in range(n)})):
            h0.append(
                PersistenceFeature(dimension=0, birth=0.0, death=float("inf"))
            )

        def fill_scale(i: int, j: int) -> float:
            best = float("inf")
            for k in range(n):
                if k in (i, j):
                    continue
                if dist[i, k] <= self.max_scale and dist[j, k] <= self.max_scale:
                    best = min(
                        best,
                        max(float(dist[i, j]), float(dist[i, k]), float(dist[j, k])),
                    )
            return best

        h1: list[PersistenceFeature] = [
            PersistenceFeature(dimension=1, birth=birth, death=fill_scale(i, j))
            for birth, i, j in cycle_edges
        ]

        h0 = [
            f for f in h0
            if f.lifetime >= self.min_persistence or f.death == float("inf")
        ]
        h1 = [
            f for f in h1
            if f.lifetime >= self.min_persistence or f.death == float("inf")
        ]
        return h0, h1, [len(h0), len(h1)]

    def _extract_clusters(
        self,
        dist: np.ndarray,
        h0_features: list[PersistenceFeature],
    ) -> list[list[int]]:
        n = dist.shape[0]
        if n == 0:
            return []

        deaths = [f.death for f in h0_features if f.death != float("inf")]
        threshold = float(np.median(deaths)) if deaths else float(np.median(dist))

        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(n):
            for j in range(i + 1, n):
                if dist[i, j] <= threshold:
                    union(i, j)

        groups: dict[int, list[int]] = {}
        for i in range(n):
            root = find(i)
            groups.setdefault(root, []).append(i)

        return sorted(groups.values(), key=lambda g: g[0])

    def _skeletonize_cluster(
        self, cluster_indices: list[int], dist: np.ndarray,
    ) -> list[int]:
        if len(cluster_indices) <= self.max_component_nodes:
            return cluster_indices

        import networkx as nx

        G = nx.Graph()
        for i in range(len(cluster_indices)):
            for j in range(i + 1, len(cluster_indices)):
                u, v = cluster_indices[i], cluster_indices[j]
                G.add_edge(u, v, weight=float(dist[u, v]))
        mst = nx.minimum_spanning_tree(G)
        centrality = nx.degree_centrality(mst)

        root = cluster_indices[0]
        latest = cluster_indices[-1]
        ranked = sorted(
            cluster_indices,
            key=lambda idx: (idx == latest, idx == root, centrality.get(idx, 0)),
            reverse=True,
        )
        return sorted(ranked[: self.max_component_nodes])

    def _extract_loops(
        self,
        dist: np.ndarray,
        h1_features: list[PersistenceFeature],
    ) -> list[list[int]]:
        n = dist.shape[0]
        if n < 3 or not h1_features:
            return []

        loops: list[list[int]] = []
        for feat in h1_features:
            best_tri: list[int] | None = None
            best_gap = float("inf")
            for i in range(n):
                for j in range(i + 1, n):
                    for k in range(j + 1, n):
                        max_edge = max(dist[i, j], dist[i, k], dist[j, k])
                        gap = abs(max_edge - feat.birth)
                        if gap < best_gap:
                            best_gap = gap
                            best_tri = [i, j, k]
            if best_tri is not None:
                loops.append(best_tri)
        return loops

    def get_cluster_labels(self, clusters: list[list[int]]) -> list[str]:
        event_list = list(self._event_dicts)
        labels: list[str] = []
        for cluster in clusters:
            cluster_events = [event_list[i] for i in cluster if i < len(event_list)]
            labels.append(label_cluster(cluster_events))
        return labels
