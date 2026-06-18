"""Communication protocols for inter-node messaging.

Provides three protocol implementations:
- AsyncPushProtocol: fire-and-forget (sensory pathways, 60-200Hz)
- SyncCallProtocol: synchronous request-response (memory/motor, 5-50Hz)
- PubSubProtocol: publish/subscribe (modulation, 1-10Hz)
"""

from .base import BaseProtocol
from .async_push import AsyncPushProtocol
from .sync_call import SyncCallProtocol
from .pub_sub import PubSubProtocol

__all__ = [
    "BaseProtocol",
    "AsyncPushProtocol",
    "SyncCallProtocol",
    "PubSubProtocol",
]
