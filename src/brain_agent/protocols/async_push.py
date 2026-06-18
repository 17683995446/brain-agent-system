"""ASYNC_PUSH protocol - fire and forget message passing.

Used by sensory pathways (60-200Hz).
Data flows from parent to children without waiting for acknowledgment.
"""
import time
from collections import deque
from typing import Optional, Dict
from ..types import BrainData
from ..node import BrainNode
from .base import BaseProtocol


class AsyncPushProtocol(BaseProtocol):
    """Asynchronous push protocol.

    Each node has a deque-based message queue.
    Sender pushes data and immediately returns.
    Receiver processes when ready.
    """

    def __init__(self):
        super().__init__("ASYNC_PUSH")
        self._queues: Dict[int, deque] = {}  # node_id -> message queue

    def _ensure_queue(self, node_id: int):
        if node_id not in self._queues:
            self._queues[node_id] = deque()

    def send(self, source: BrainNode, target: BrainNode, data: BrainData) -> None:
        start = time.perf_counter()
        self._ensure_queue(target.node_id)
        self._queues[target.node_id].append(data)
        self._message_count += 1
        self._total_latency += time.perf_counter() - start

    def receive(self, node: BrainNode) -> Optional[BrainData]:
        self._ensure_queue(node.node_id)
        if self._queues[node.node_id]:
            return self._queues[node.node_id].popleft()
        return None

    def pending_count(self, node_id: int) -> int:
        self._ensure_queue(node_id)
        return len(self._queues[node_id])
