"""Base protocol class for inter-node communication."""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from ..types import BrainData
from ..node import BrainNode


class BaseProtocol(ABC):
    """Abstract base for communication protocols."""

    def __init__(self, name: str):
        self._name = name
        self._message_count = 0
        self._error_count = 0
        self._total_latency = 0.0

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def send(self, source: BrainNode, target: BrainNode, data: BrainData) -> None:
        """Send data from source to target node."""
        ...

    @abstractmethod
    def receive(self, node: BrainNode) -> Optional[BrainData]:
        """Receive data for a node."""
        ...

    def get_stats(self) -> Dict[str, Any]:
        return {
            "protocol": self._name,
            "message_count": self._message_count,
            "error_count": self._error_count,
            "avg_latency_ms": (self._total_latency / self._message_count * 1000) if self._message_count > 0 else 0.0,
        }
