"""PUB_SUB protocol - publish/subscribe messaging.

Used by modulation systems (1-10Hz).
One publisher broadcasts to multiple subscribers.
"""
import logging
import time
from collections import defaultdict, deque
from typing import Optional, Dict, List, Set
from ..types import BrainData
from ..node import BrainNode
from .base import BaseProtocol

logger = logging.getLogger(__name__)

# Maximum messages per node queue. Prevents unbounded memory growth.
MAX_PROTOCOL_QUEUE_SIZE = 5000


class PubSubProtocol(BaseProtocol):
    """Publish/Subscribe protocol.

    Nodes subscribe to topics.
    Publishers broadcast to all topic subscribers.
    """

    def __init__(self) -> None:
        super().__init__("PUB_SUB")
        self._subscribers: Dict[str, Set[int]] = defaultdict(set)
        self._queues: Dict[int, deque] = {}  # node_id -> message queue

    def subscribe(self, node: BrainNode, topic: str) -> None:
        self._subscribers[topic].add(node.node_id)
        if node.node_id not in self._queues:
            self._queues[node.node_id] = deque(maxlen=MAX_PROTOCOL_QUEUE_SIZE)

    def unsubscribe(self, node: BrainNode, topic: str) -> None:
        self._subscribers[topic].discard(node.node_id)

    def send(self, source: BrainNode, target: BrainNode, data: BrainData) -> None:
        """Direct send (used for tree propagation)."""
        start = time.perf_counter()
        if target.node_id not in self._queues:
            self._queues[target.node_id] = deque(maxlen=MAX_PROTOCOL_QUEUE_SIZE)
        try:
            self._queues[target.node_id].append(data)
        except MemoryError:
            logger.error(
                f"MemoryError in PUB_SUB send to node {target.node_id} "
                f"(queue size: {len(self._queues[target.node_id])})"
            )
            self._error_count += 1
            raise
        self._message_count += 1
        self._total_latency += time.perf_counter() - start

    def publish(self, publisher: BrainNode, topic: str, data: BrainData) -> int:
        """Publish data to all subscribers of a topic. Returns number of subscribers."""
        start = time.perf_counter()
        subscriber_ids = self._subscribers.get(topic, set())
        for sid in subscriber_ids:
            if sid not in self._queues:
                self._queues[sid] = deque(maxlen=MAX_PROTOCOL_QUEUE_SIZE)
            try:
                self._queues[sid].append(data)
            except MemoryError:
                logger.error(
                    f"MemoryError in PUB_SUB publish to node {sid} "
                    f"(queue size: {len(self._queues[sid])})"
                )
                self._error_count += 1
                raise
            self._message_count += 1
        self._total_latency += time.perf_counter() - start
        return len(subscriber_ids)

    def receive(self, node: BrainNode) -> Optional[BrainData]:
        if node.node_id in self._queues and self._queues[node.node_id]:
            return self._queues[node.node_id].popleft()  # type: ignore[no-any-return]
        return None

    def get_subscribers(self, topic: str) -> Set[int]:
        return self._subscribers.get(topic, set())

    def get_topics(self) -> List[str]:
        return list(self._subscribers.keys())
