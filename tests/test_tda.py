"""Tests for tda module — topological log event analysis."""

import numpy as np

from tda.tracker import (
    LogTopologyTracker,
    PersistenceFeature,
    TopologySignals,
    label_cluster,
    log_event_to_text,
)
from tda.schemas import ClusterNode, TopologySnapshot
from tda.embeddings import embed_texts


def _make_event(event: str, logger: str = "test", level: str = "info", **extra):
    return {"event": event, "logger": logger, "level": level, **extra}


class TestLogEventToText:
    def test_basic_event(self):
        text = log_event_to_text(_make_event("invoke_agent.start", logger="ai.agent"))
        assert "ai.agent" in text
        assert "invoke_agent.start" in text

    def test_includes_extras(self):
        text = log_event_to_text(_make_event("agent.tool_result", tool="db_read", duration_ms=42.5))
        assert "tool=db_read" in text
        assert "duration_ms=42.5" in text

    def test_skips_timestamp_and_params(self):
        text = log_event_to_text({"event": "x", "timestamp": "2026-01-01", "params": {"a": 1}})
        assert "2026-01-01" not in text
        assert "params" not in text


class TestLabelCluster:
    def test_dominant_prefix(self):
        events = [
            _make_event("agent.llm_call"),
            _make_event("agent.tool_result"),
            _make_event("agent.llm_call"),
        ]
        lbl = label_cluster(events)
        assert "agent" in lbl

    def test_empty(self):
        assert label_cluster([]) == "unknown"


class TestPersistenceFeature:
    def test_lifetime(self):
        feat = PersistenceFeature(dimension=0, birth=0.1, death=0.5)
        assert abs(feat.lifetime - 0.4) < 1e-6

    def test_infinite_death(self):
        feat = PersistenceFeature(dimension=0, birth=0.1, death=float("inf"))
        assert feat.lifetime == 0.0


class TestEmbeddings:
    def test_embed_texts_returns_ndarray(self):
        result = embed_texts(["hello world"])
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 1
        assert result.ndim == 2

    def test_similar_texts_are_close(self):
        embs = embed_texts([
            "FastAPI backend routing configuration",
            "Configure FastAPI route handlers",
            "Grocery shopping list for dinner",
        ])
        from sklearn.metrics.pairwise import cosine_distances
        dists = cosine_distances(embs)
        assert dists[0, 1] < dists[0, 2]


class TestLogTopologyTracker:
    def test_needs_three_events(self):
        tracker = LogTopologyTracker()
        tracker.add_event(_make_event("a"), timestamp=0.0)
        signals = tracker.analyze()
        assert signals.window_size == 1
        assert signals.h0_clusters == 1
        assert signals.clusters == [[0]]

        tracker.add_event(_make_event("b"), timestamp=1.0)
        signals = tracker.analyze()
        assert signals.window_size == 2
        assert signals.h0_clusters == 2

    def test_same_topic_forms_one_cluster(self):
        tracker = LogTopologyTracker(min_persistence=0.05)
        tracker.add_event(
            _make_event("invoke_agent.start", logger="ai.agent"),
            timestamp=0.0,
        )
        tracker.add_event(
            _make_event("agent.llm_call", logger="ai.agent", iteration=1),
            timestamp=1.0,
        )
        tracker.add_event(
            _make_event("agent.tool_result", logger="ai.agent", tool="db_read"),
            timestamp=2.0,
        )

        signals = tracker.analyze()
        assert signals.window_size == 3
        assert len(signals.clusters) >= 1
        largest = max(signals.clusters, key=len)
        assert len(largest) >= 2

    def test_different_topics_form_separate_clusters(self):
        tracker = LogTopologyTracker(min_persistence=0.05)
        tracker.add_event(
            _make_event("invoke_agent.start", logger="ai.agent"),
            timestamp=0.0,
        )
        tracker.add_event(
            _make_event("agent.llm_call", logger="ai.agent"),
            timestamp=10.0,
        )
        tracker.add_event(
            _make_event("auth.session_invalid", logger="auth.permissions", session_id="abc"),
            timestamp=20.0,
        )
        tracker.add_event(
            _make_event("auth.permission_denied", logger="auth.permissions", user_id="u1"),
            timestamp=30.0,
        )

        signals = tracker.analyze()
        assert signals.h0_clusters >= 2

    def test_sliding_window(self):
        tracker = LogTopologyTracker(max_window=3)
        for i in range(5):
            tracker.add_event(_make_event(f"event_{i}"), timestamp=float(i))
        assert tracker.size == 3

    def test_to_dict(self):
        tracker = LogTopologyTracker()
        tracker.add_event(_make_event("a"), timestamp=0.0)
        tracker.add_event(_make_event("b"), timestamp=1.0)
        tracker.add_event(_make_event("c"), timestamp=2.0)

        signals = tracker.analyze()
        d = signals.to_dict()
        assert "window_size" in d
        assert "h0_clusters" in d
        assert "h1_loops" in d
        assert "betti" in d
        assert "clusters" in d
        assert isinstance(d["clusters"], list)

    def test_cluster_labels(self):
        tracker = LogTopologyTracker(min_persistence=0.05)
        tracker.add_event(
            _make_event("agent.llm_call", logger="ai.agent"),
            timestamp=0.0,
        )
        tracker.add_event(
            _make_event("agent.tool_result", logger="ai.agent"),
            timestamp=1.0,
        )
        tracker.add_event(
            _make_event("agent.output_eval", logger="ai.agent"),
            timestamp=2.0,
        )

        signals = tracker.analyze()
        labels = tracker.get_cluster_labels(signals.clusters)
        assert len(labels) == len(signals.clusters)
        assert all(isinstance(l, str) for l in labels)

    def test_skeletonize_preserves_small_clusters(self):
        tracker = LogTopologyTracker(max_component_nodes=5)
        for i in range(4):
            tracker.add_event(_make_event(f"agent.step_{i}", logger="ai.agent"), timestamp=float(i))

        signals = tracker.analyze()
        for cluster in signals.clusters:
            assert len(cluster) <= 5


class TestTopologySnapshot:
    def test_snapshot_serialization(self):
        snap = TopologySnapshot(
            nodes=[ClusterNode(cluster_id=0, label="test", size=3, event_indices=[0, 1, 2], persistence=0.5)],
            edges=[],
            event_labels=[0, 0, 0],
            h0_features=[{"dim": 0, "birth": 0.0, "death": 0.5, "lifetime": 0.5}],
            h1_features=[],
            betti=[1, 0],
            timestamp="2026-11-02T00:00:00",
        )
        d = snap.model_dump()
        assert d["nodes"][0]["label"] == "test"
        assert d["betti"] == [1, 0]
