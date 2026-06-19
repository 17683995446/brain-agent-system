"""SYNC_CALL protocol - synchronous request-response.

Used by memory and motor pathways (5-50Hz).
Sender waits for all children to complete processing.
"""
import logging
import time
from typing import Optional, List, Dict
from ..types import BrainData
from ..node import BrainNode
from .base import BaseProtocol

logger = logging.getLogger(__name__)


class SyncCallProtocol(BaseProtocol):
    """Synchronous call protocol.

    Direct method invocation with result collection.
    No queuing - immediate execution.
    """

    def __init__(self) -> None:
        super().__init__("SYNC_CALL")
        self._last_results: Dict[int, List[BrainData]] = {}

    def send(self, source: BrainNode, target: BrainNode, data: BrainData) -> None:
        start = time.perf_counter()
        try:
            result = target.process(data)
        except MemoryError:
            logger.error(f"MemoryError in SYNC_CALL to node {target.node_id}")
            self._error_count += 1
            raise
        self._last_results.setdefault(source.node_id, []).append(result)
        self._message_count += 1
        self._total_latency += time.perf_counter() - start

    def receive(self, node: BrainNode) -> Optional[BrainData]:
        return None  # SYNC_CALL doesn't use receive queue

    def call_and_collect(self, source: BrainNode, targets: List[BrainNode], data: BrainData) -> List[BrainData]:
        """Call all targets synchronously and collect results."""
        start = time.perf_counter()
        results: List[BrainData] = []
        for target in targets:
            try:
                result = target.process(data)
            except MemoryError:
                logger.error(f"MemoryError in SYNC_CALL to node {target.node_id}")
                self._error_count += 1
                raise
            results.append(result)
            self._message_count += 1
        self._total_latency += time.perf_counter() - start
        self._last_results[source.node_id] = results
        return results

    def get_last_results(self, node_id: int) -> List[BrainData]:
        return self._last_results.get(node_id, [])
