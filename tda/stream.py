"""Real-time TDA streaming over log events."""

from __future__ import annotations

import asyncio
import time

from logging_config import get_logger
from tda.schemas import ClusterEdge, ClusterNode, TopologySnapshot
from tda.tracker import LogTopologyTracker

_log = get_logger("tda.stream")

_tda_stream: TDAStream | None = None


class TDAStream:
    def __init__(self, window_size: int = 200, interval: float = 2.0):
        self._tracker = LogTopologyTracker(max_window=window_size)
        self._interval = interval
        self._subscribers: set[asyncio.Queue] = set()
        self._task: asyncio.Task | None = None
        self._dirty = False
        self._current: TopologySnapshot | None = None

    def ingest(self, event: dict) -> None:
        self._tracker.add_event(event)
        self._dirty = True

    async def run(self) -> None:
        while True:
            await asyncio.sleep(self._interval)
            if not self._dirty or self._tracker.size < 3:
                continue
            self._dirty = False
            try:
                signals = self._tracker.analyze()
                snapshot = self._build_snapshot(signals)
                self._current = snapshot
                for q in list(self._subscribers):
                    try:
                        q.put_nowait(snapshot)
                    except asyncio.QueueFull:
                        self._subscribers.discard(q)
            except Exception as exc:
                _log.error("tda.analyze_error", error=str(exc))

    def _build_snapshot(self, signals) -> TopologySnapshot:
        labels = self._tracker.get_cluster_labels(signals.clusters)

        nodes: list[ClusterNode] = []
        for i, cluster in enumerate(signals.clusters):
            max_persistence = 0.0
            for f in signals.h0_features:
                if f.death != float("inf"):
                    max_persistence = max(max_persistence, f.lifetime)
            nodes.append(ClusterNode(
                cluster_id=i,
                label=labels[i] if i < len(labels) else "unknown",
                size=len(cluster),
                event_indices=cluster,
                persistence=round(max_persistence, 4),
            ))

        edges: list[ClusterEdge] = []
        if signals.distance_matrix is not None and len(signals.clusters) > 1:
            for i in range(len(signals.clusters)):
                for j in range(i + 1, len(signals.clusters)):
                    ci = signals.clusters[i]
                    cj = signals.clusters[j]
                    min_dist = float("inf")
                    for a in ci:
                        for b in cj:
                            d = float(signals.distance_matrix[a, b])
                            if d < min_dist:
                                min_dist = d
                    edges.append(ClusterEdge(
                        source=i, target=j, merge_distance=round(min_dist, 4),
                    ))

        event_labels = [-1] * signals.window_size
        for i, cluster in enumerate(signals.clusters):
            for idx in cluster:
                if idx < signals.window_size:
                    event_labels[idx] = i

        sig_dict = signals.to_dict()

        return TopologySnapshot(
            nodes=nodes,
            edges=edges,
            event_labels=event_labels,
            h0_features=sig_dict["h0_features"],
            h1_features=sig_dict["h1_features"],
            betti=sig_dict["betti"],
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )

    def subscribe(self, catchup: bool = True) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        if catchup and self._current is not None:
            try:
                q.put_nowait(self._current)
            except asyncio.QueueFull:
                pass
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    def get_current_snapshot(self) -> TopologySnapshot | None:
        return self._current

    async def start(self) -> None:
        self._task = asyncio.create_task(self.run())
        _log.info("tda.stream_started")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        _log.info("tda.stream_stopped")


async def init_tda_stream(window_size: int = 200, interval: float = 2.0) -> None:
    global _tda_stream
    _tda_stream = TDAStream(window_size=window_size, interval=interval)
    await _tda_stream.start()


def get_tda_stream() -> TDAStream | None:
    return _tda_stream


async def close_tda_stream() -> None:
    global _tda_stream
    if _tda_stream is not None:
        await _tda_stream.stop()
        _tda_stream = None
