"""ASYNC_PUSH protocol - fire and forget message passing.

Used by sensory pathways (60-200Hz).
Data flows from parent to children without waiting for acknowledgment.
"""
import logging
import time
from collections import deque
from typing import Optional, Dict
from ..types import BrainData
from ..node import BrainNode
from .base import BaseProtocol

logger = logging.getLogger(__name__)

# Maximum messages per node queue. Prevents unbounded memory growth.
MAX_PROTOCOL_QUEUE_SIZE = 5000


class AsyncPushProtocol(BaseProtocol):
    """Asynchronous push protocol.

    Each node has a deque-based message queue with a maximum size.
    Sender pushes data and immediately returns.
    Receiver processes when ready.
    """

    def __init__(self) -> None:
        super().__init__("ASYNC_PUSH")
        self._queues: Dict[int, deque] = {}  # node_id -> message queue

    def _ensure_queue(self, node_id: int) -> None:
        if node_id not in self._queues:
            self._queues[node_id] = deque(maxlen=MAX_PROTOCOL_QUEUE_SIZE)

    def send(self, source: BrainNode, target: BrainNode, data: BrainData) -> None:
        start = time.perf_counter()
        self._ensure_queue(target.node_id)
        try:
            self._queues[target.node_id].append(data)
        except MemoryError:
            logger.error(
                f"MemoryError in ASYNC_PUSH send to node {target.node_id} "
                f"(queue size: {len(self._queues[target.node_id])})"
            )
            self._error_count += 1
            raise
        self._message_count += 1
        self._total_latency += time.perf_counter() - start

    def receive(self, node: BrainNode) -> Optional[BrainData]:
        self._ensure_queue(node.node_id)
        if self._queues[node.node_id]:
            return self._queues[node.node_id].popleft()  # type: ignore[no-any-return]
        return None

    def pending_count(self, node_id: int) -> int:
        self._ensure_queue(node_id)
        return len(self._queues[node_id])
