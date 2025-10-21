"""
Monitoring Module - Metrics, Logging, and Tracing

This module provides comprehensive monitoring capabilities including:
- Metrics collection (counters, gauges, histograms, timers)
- Structured logging with context
- Distributed tracing with spans
- Performance monitoring
- Integration with OpenTelemetry (optional)

Copyright (c) 2025 Lionel TAGNE. All Rights Reserved.
This software is proprietary and confidential.
"""

import contextvars
import json
import time
import uuid
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union

from num_agents.core import Flow, Node, SharedStore
from num_agents.exceptions import NumAgentsException
from num_agents.logging_config import get_logger

logger = get_logger(__name__)


# ============================================================================
# Exceptions
# ============================================================================


class MonitoringException(NumAgentsException):
    """Base exception for monitoring errors."""

    pass


# ============================================================================
# Context Variables for Tracing
# ============================================================================

# Thread-local context for current trace and span
_current_trace_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "current_trace_id", default=None
)
_current_span_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "current_span_id", default=None
)


# ============================================================================
# Enums
# ============================================================================


class MetricType(str, Enum):
    """Types of metrics."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class SpanStatus(str, Enum):
    """Span status for tracing."""

    OK = "ok"
    ERROR = "error"
    UNSET = "unset"


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class Metric:
    """Represents a metric."""

    name: str
    type: MetricType
    value: Union[int, float]
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.type.value,
            "value": self.value,
            "labels": self.labels,
            "timestamp": self.timestamp,
            "description": self.description,
        }


@dataclass
class Span:
    """Represents a trace span."""

    span_id: str
    trace_id: str
    name: str
    start_time: float
    end_time: Optional[float] = None
    parent_span_id: Optional[str] = None
    status: SpanStatus = SpanStatus.UNSET
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def duration_ms(self) -> Optional[float]:
        """Get duration in milliseconds."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add an event to the span."""
        self.events.append(
            {"name": name, "timestamp": time.time(), "attributes": attributes or {}}
        )

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute."""
        self.attributes[key] = value

    def finish(self, status: SpanStatus = SpanStatus.OK, error: Optional[str] = None) -> None:
        """Finish the span."""
        self.end_time = time.time()
        self.status = status
        if error:
            self.error = error

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "name": self.name,
            "parent_span_id": self.parent_span_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "attributes": self.attributes,
            "events": self.events,
            "error": self.error,
        }


# ============================================================================
# Metrics Collector
# ============================================================================


class MetricsCollector:
    """
    Collects and manages metrics.

    Supports counters, gauges, histograms, and timers.
    """

    def __init__(self) -> None:
        """Initialize metrics collector."""
        self._metrics: Dict[str, List[Metric]] = defaultdict(list)
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)

    def counter(
        self,
        name: str,
        value: Union[int, float] = 1,
        labels: Optional[Dict[str, str]] = None,
        description: Optional[str] = None,
    ) -> None:
        """
        Increment a counter.

        Args:
            name: Counter name
            value: Value to add (default 1)
            labels: Optional labels
            description: Optional description
        """
        key = self._make_key(name, labels)
        self._counters[key] += value

        metric = Metric(
            name=name,
            type=MetricType.COUNTER,
            value=self._counters[key],
            labels=labels or {},
            description=description,
        )
        self._metrics[name].append(metric)

    def gauge(
        self,
        name: str,
        value: Union[int, float],
        labels: Optional[Dict[str, str]] = None,
        description: Optional[str] = None,
    ) -> None:
        """
        Set a gauge value.

        Args:
            name: Gauge name
            value: Current value
            labels: Optional labels
            description: Optional description
        """
        key = self._make_key(name, labels)
        self._gauges[key] = value

        metric = Metric(
            name=name,
            type=MetricType.GAUGE,
            value=value,
            labels=labels or {},
            description=description,
        )
        self._metrics[name].append(metric)

    def histogram(
        self,
        name: str,
        value: Union[int, float],
        labels: Optional[Dict[str, str]] = None,
        description: Optional[str] = None,
    ) -> None:
        """
        Record a histogram value.

        Args:
            name: Histogram name
            value: Value to record
            labels: Optional labels
            description: Optional description
        """
        key = self._make_key(name, labels)
        self._histograms[key].append(value)

        metric = Metric(
            name=name,
            type=MetricType.HISTOGRAM,
            value=value,
            labels=labels or {},
            description=description,
        )
        self._metrics[name].append(metric)

    @contextmanager
    def timer(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
        description: Optional[str] = None,
    ):
        """
        Time a block of code.

        Usage:
            with metrics.timer("operation_duration"):
                # code to time
                pass

        Args:
            name: Timer name
            labels: Optional labels
            description: Optional description
        """
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.histogram(
                name=name,
                value=duration * 1000,  # Convert to milliseconds
                labels=labels,
                description=description or "Duration in milliseconds",
            )

    def get_metric(self, name: str) -> List[Metric]:
        """Get all metrics with a given name."""
        return self._metrics.get(name, [])

    def get_counter_value(
        self, name: str, labels: Optional[Dict[str, str]] = None
    ) -> float:
        """Get current counter value."""
        key = self._make_key(name, labels)
        return self._counters.get(key, 0)

    def get_gauge_value(self, name: str, labels: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Get current gauge value."""
        key = self._make_key(name, labels)
        return self._gauges.get(key)

    def get_histogram_stats(
        self, name: str, labels: Optional[Dict[str, str]] = None
    ) -> Dict[str, float]:
        """
        Get histogram statistics.

        Returns:
            Dictionary with min, max, avg, p50, p95, p99
        """
        key = self._make_key(name, labels)
        values = self._histograms.get(key, [])

        if not values:
            return {}

        sorted_values = sorted(values)
        n = len(sorted_values)

        return {
            "count": n,
            "min": sorted_values[0],
            "max": sorted_values[-1],
            "avg": sum(sorted_values) / n,
            "p50": sorted_values[max(0, int(n * 0.5) - 1)],
            "p95": sorted_values[max(0, min(n - 1, int(n * 0.95)))],
            "p99": sorted_values[max(0, min(n - 1, int(n * 0.99)))],
        }

    def get_all_metrics(self) -> Dict[str, List[Metric]]:
        """Get all collected metrics."""
        return dict(self._metrics)

    def clear(self) -> None:
        """Clear all metrics."""
        self._metrics.clear()
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()

    def export_prometheus(self) -> str:
        """
        Export metrics in Prometheus format.

        Returns:
            String in Prometheus exposition format
        """
        lines = []

        # Export counters
        for key, value in self._counters.items():
            name, labels = self._parse_key(key)
            label_str = self._format_prometheus_labels(labels)
            lines.append(f"{name}{label_str} {value}")

        # Export gauges
        for key, value in self._gauges.items():
            name, labels = self._parse_key(key)
            label_str = self._format_prometheus_labels(labels)
            lines.append(f"{name}{label_str} {value}")

        return "\n".join(lines)

    def _make_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Create a unique key from name and labels."""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def _parse_key(self, key: str) -> tuple:
        """Parse a key back into name and labels."""
        if "{" not in key:
            return key, {}

        name, label_part = key.split("{", 1)
        label_part = label_part.rstrip("}")

        labels = {}
        if label_part:
            for pair in label_part.split(","):
                k, v = pair.split("=", 1)
                labels[k] = v

        return name, labels

    def _format_prometheus_labels(self, labels: Dict[str, str]) -> str:
        """Format labels for Prometheus."""
        if not labels:
            return ""
        label_pairs = [f'{k}="{v}"' for k, v in labels.items()]
        return "{" + ",".join(label_pairs) + "}"


# ============================================================================
# Tracer
# ============================================================================


class Tracer:
    """
    Distributed tracing implementation.

    Tracks execution spans across nodes and flows.
    """

    def __init__(self) -> None:
        """Initialize tracer."""
        self._spans: Dict[str, Span] = {}
        self._active_traces: Set[str] = set()

    def start_trace(self, name: str, trace_id: Optional[str] = None) -> str:
        """
        Start a new trace.

        Args:
            name: Trace name
            trace_id: Optional trace ID (auto-generated if not provided)

        Returns:
            Trace ID
        """
        if trace_id is None:
            trace_id = str(uuid.uuid4())

        self._active_traces.add(trace_id)
        _current_trace_id.set(trace_id)

        # Create root span
        self.start_span(name, trace_id=trace_id)

        return trace_id

    def start_span(
        self,
        name: str,
        trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Span:
        """
        Start a new span.

        Args:
            name: Span name
            trace_id: Trace ID (uses current if not provided)
            parent_span_id: Parent span ID (uses current if not provided)
            attributes: Optional span attributes

        Returns:
            Started span
        """
        if trace_id is None:
            trace_id = _current_trace_id.get()
            if trace_id is None:
                trace_id = self.start_trace(name)

        if parent_span_id is None:
            parent_span_id = _current_span_id.get()

        span = Span(
            span_id=str(uuid.uuid4()),
            trace_id=trace_id,
            name=name,
            start_time=time.time(),
            parent_span_id=parent_span_id,
            attributes=attributes or {},
        )

        self._spans[span.span_id] = span
        _current_span_id.set(span.span_id)

        return span

    def finish_span(
        self, span: Span, status: SpanStatus = SpanStatus.OK, error: Optional[str] = None
    ) -> None:
        """
        Finish a span.

        Args:
            span: Span to finish
            status: Span status
            error: Optional error message
        """
        span.finish(status=status, error=error)

        # Restore parent span as current
        if span.parent_span_id:
            _current_span_id.set(span.parent_span_id)
        else:
            _current_span_id.set(None)

    @contextmanager
    def span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """
        Context manager for creating a span.

        Usage:
            with tracer.span("operation"):
                # code to trace
                pass

        Args:
            name: Span name
            attributes: Optional span attributes
        """
        span = self.start_span(name, attributes=attributes)
        try:
            yield span
            self.finish_span(span, status=SpanStatus.OK)
        except Exception as e:
            self.finish_span(span, status=SpanStatus.ERROR, error=str(e))
            raise

    def get_span(self, span_id: str) -> Optional[Span]:
        """Get a span by ID."""
        return self._spans.get(span_id)

    def get_trace_spans(self, trace_id: str) -> List[Span]:
        """Get all spans for a trace."""
        return [s for s in self._spans.values() if s.trace_id == trace_id]

    def get_all_spans(self) -> List[Span]:
        """Get all spans."""
        return list(self._spans.values())

    def clear(self) -> None:
        """Clear all spans."""
        self._spans.clear()
        self._active_traces.clear()
        _current_trace_id.set(None)
        _current_span_id.set(None)

    def export_jaeger(self, trace_id: str) -> Dict[str, Any]:
        """
        Export trace in Jaeger format.

        Args:
            trace_id: Trace ID to export

        Returns:
            Jaeger-formatted trace
        """
        spans = self.get_trace_spans(trace_id)

        jaeger_spans = []
        for span in spans:
            jaeger_span = {
                "traceID": span.trace_id,
                "spanID": span.span_id,
                "operationName": span.name,
                "startTime": int(span.start_time * 1_000_000),  # microseconds
                "duration": int((span.duration_ms or 0) * 1000),  # microseconds
                "tags": [{"key": k, "value": v} for k, v in span.attributes.items()],
                "logs": [
                    {
                        "timestamp": int(event["timestamp"] * 1_000_000),
                        "fields": [
                            {"key": k, "value": v}
                            for k, v in event["attributes"].items()
                        ],
                    }
                    for event in span.events
                ],
            }

            if span.parent_span_id:
                jaeger_span["references"] = [
                    {"refType": "CHILD_OF", "spanID": span.parent_span_id}
                ]

            jaeger_spans.append(jaeger_span)

        return {"traceID": trace_id, "spans": jaeger_spans}


# ============================================================================
# Monitor - Main Monitoring Interface
# ============================================================================


class Monitor:
    """
    Main monitoring interface combining metrics, logging, and tracing.
    """

    def __init__(
        self,
        service_name: str = "num_agents",
        enable_metrics: bool = True,
        enable_tracing: bool = True,
        enable_logging: bool = True,
    ) -> None:
        """
        Initialize monitor.

        Args:
            service_name: Service name for monitoring
            enable_metrics: Enable metrics collection
            enable_tracing: Enable distributed tracing
            enable_logging: Enable structured logging
        """
        self.service_name = service_name
        self._enable_metrics = enable_metrics
        self._enable_tracing = enable_tracing
        self._enable_logging = enable_logging

        self.metrics = MetricsCollector() if enable_metrics else None
        self.tracer = Tracer() if enable_tracing else None
        self._logger = get_logger(__name__) if enable_logging else None

    def log(
        self,
        level: str,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log a structured message.

        Args:
            level: Log level (debug, info, warning, error, critical)
            message: Log message
            extra: Extra context data
        """
        if not self._enable_logging or not self._logger:
            return

        # Add trace context
        context = {
            "service": self.service_name,
            "trace_id": _current_trace_id.get(),
            "span_id": _current_span_id.get(),
        }

        if extra:
            context.update(extra)

        log_method = getattr(self._logger, level.lower(), self._logger.info)
        log_method(message, extra=context)

    def get_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        stats = {"service": self.service_name}

        if self.metrics:
            all_metrics = self.metrics.get_all_metrics()
            stats["metrics_count"] = sum(len(v) for v in all_metrics.values())
            stats["metric_names"] = list(all_metrics.keys())

        if self.tracer:
            all_spans = self.tracer.get_all_spans()
            stats["spans_count"] = len(all_spans)
            stats["active_traces"] = len(self.tracer._active_traces)

        return stats

    def clear(self) -> None:
        """Clear all monitoring data."""
        if self.metrics:
            self.metrics.clear()
        if self.tracer:
            self.tracer.clear()


# ============================================================================
# Monitoring Nodes for Flow Integration
# ============================================================================


class MonitoringNode(Node):
    """Node that monitors flow execution."""

    def __init__(
        self,
        monitor: Monitor,
        metric_prefix: str = "flow",
        name: Optional[str] = None,
        enable_logging: bool = True,
    ) -> None:
        """
        Initialize monitoring node.

        Args:
            monitor: Monitor instance
            metric_prefix: Prefix for metrics
            name: Optional node name
            enable_logging: Enable logging
        """
        super().__init__(name or "MonitoringNode", enable_logging=enable_logging)
        self.monitor = monitor
        self.metric_prefix = metric_prefix

    def exec(self, shared: SharedStore) -> Dict[str, Any]:
        """Record monitoring statistics."""
        stats = self.monitor.get_statistics()

        if self.monitor.metrics:
            self.monitor.metrics.gauge(
                f"{self.metric_prefix}.metrics_count", stats.get("metrics_count", 0)
            )
            self.monitor.metrics.gauge(
                f"{self.metric_prefix}.spans_count", stats.get("spans_count", 0)
            )

        return {"stats": stats}


# ============================================================================
# Global Singleton
# ============================================================================

# Global monitor instance
_global_monitor: Optional[Monitor] = None


def get_monitor(
    service_name: str = "num_agents",
    enable_metrics: bool = True,
    enable_tracing: bool = True,
    enable_logging: bool = True,
) -> Monitor:
    """
    Get or create global monitor instance.

    Args:
        service_name: Service name
        enable_metrics: Enable metrics
        enable_tracing: Enable tracing
        enable_logging: Enable logging

    Returns:
        Global monitor instance
    """
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = Monitor(
            service_name=service_name,
            enable_metrics=enable_metrics,
            enable_tracing=enable_tracing,
            enable_logging=enable_logging,
        )
    return _global_monitor
