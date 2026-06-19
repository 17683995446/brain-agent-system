"""Request tracing for brain simulation.

Tracks data flow through the node tree.
"""

import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class TraceSpan:
    """A single span in a trace."""
    node_id: int
    node_name: str
    layer: int
    start_time: float
    end_time: float = 0.0
    duration_ms: float = 0.0
    children: List["TraceSpan"] = field(default_factory=list)
    data_type: str = ""
    error: Optional[str] = None

    def finish(self) -> None:
        self.end_time = time.perf_counter()
        self.duration_ms = (self.end_time - self.start_time) * 1000


class Tracer:
    """Traces data flow through the brain node tree."""

    def __init__(self) -> None:
        self._traces: List[Dict[str, Any]] = []
        self._span_stack: List[TraceSpan] = []
        self._enabled = True

    @property
    def _current_span(self) -> Optional[TraceSpan]:
        return self._span_stack[-1] if self._span_stack else None

    def start_span(self, node_id: int, node_name: str, layer: int, data_type: str = "") -> TraceSpan:
        if not self._enabled:
            return TraceSpan(node_id=node_id, node_name=node_name, layer=layer, start_time=0)
        span = TraceSpan(
            node_id=node_id,
            node_name=node_name,
            layer=layer,
            start_time=time.perf_counter(),
            data_type=data_type,
        )
        if self._current_span:
            self._current_span.children.append(span)
        else:
            self._traces.append({"root": span, "spans": []})
        self._span_stack.append(span)
        return span

    def finish_span(self, span: TraceSpan, error: Optional[str] = None) -> None:
        if not self._enabled:
            return
        span.finish()
        span.error = error
        # Pop back to parent
        if self._span_stack and self._span_stack[-1] == span:
            self._span_stack.pop()

    def get_traces(self) -> List[Dict[str, Any]]:
        return self._traces

    def get_latest_trace(self) -> Optional[Dict[str, Any]]:
        return self._traces[-1] if self._traces else None

    def format_trace(self, trace: Optional[Dict[str, Any]] = None) -> str:
        """Format a trace as a readable string."""
        trace = trace or self.get_latest_trace()
        if not trace:
            return "No traces"
        lines = []
        root = trace["root"]
        lines.append(f"Trace: {root.node_name} (L{root.layer:02d})")
        self._format_span(root, lines, indent=2)
        return "\n".join(lines)

    def _format_span(self, span: TraceSpan, lines: List[str], indent: int) -> None:
        prefix = " " * indent
        err = f" ERROR: {span.error}" if span.error else ""
        lines.append(f"{prefix}└─ {span.node_name} (L{span.layer:02d}) {span.duration_ms:.2f}ms{err}")
        for child in span.children:
            self._format_span(child, lines, indent + 2)

    def clear(self) -> None:
        self._traces.clear()
        self._span_stack.clear()
