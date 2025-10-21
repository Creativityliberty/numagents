"""
Tests for Monitoring module.

Copyright (c) 2025 Lionel TAGNE. All Rights Reserved.
This software is proprietary and confidential.
"""

import time
from typing import Any, Dict

import pytest

from num_agents.core import Flow, SharedStore
from num_agents.modules.monitoring import (
    Metric,
    MetricsCollector,
    MetricType,
    Monitor,
    MonitoringNode,
    Span,
    SpanStatus,
    Tracer,
    get_monitor,
)


class TestMetric:
    """Test Metric class."""

    def test_create_metric(self) -> None:
        """Test metric creation."""
        metric = Metric(
            name="test.counter",
            type=MetricType.COUNTER,
            value=10,
            labels={"env": "test"},
            description="Test counter",
        )

        assert metric.name == "test.counter"
        assert metric.type == MetricType.COUNTER
        assert metric.value == 10
        assert metric.labels == {"env": "test"}
        assert metric.description == "Test counter"
        assert metric.timestamp > 0

    def test_metric_to_dict(self) -> None:
        """Test metric serialization."""
        metric = Metric(
            name="test.gauge", type=MetricType.GAUGE, value=42.5, labels={"host": "server1"}
        )

        data = metric.to_dict()

        assert data["name"] == "test.gauge"
        assert data["type"] == "gauge"
        assert data["value"] == 42.5
        assert data["labels"] == {"host": "server1"}
        assert "timestamp" in data


class TestMetricsCollector:
    """Test MetricsCollector."""

    @pytest.fixture
    def collector(self) -> MetricsCollector:
        """Create a metrics collector."""
        return MetricsCollector()

    def test_counter(self, collector: MetricsCollector) -> None:
        """Test counter increment."""
        collector.counter("requests.total")
        collector.counter("requests.total")
        collector.counter("requests.total", value=3)

        assert collector.get_counter_value("requests.total") == 5

    def test_counter_with_labels(self, collector: MetricsCollector) -> None:
        """Test counter with labels."""
        collector.counter("http.requests", labels={"method": "GET", "status": "200"})
        collector.counter("http.requests", labels={"method": "GET", "status": "200"}, value=2)
        collector.counter("http.requests", labels={"method": "POST", "status": "201"})

        assert (
            collector.get_counter_value("http.requests", labels={"method": "GET", "status": "200"})
            == 3
        )
        assert (
            collector.get_counter_value("http.requests", labels={"method": "POST", "status": "201"})
            == 1
        )

    def test_gauge(self, collector: MetricsCollector) -> None:
        """Test gauge set."""
        collector.gauge("cpu.usage", 45.2)
        assert collector.get_gauge_value("cpu.usage") == 45.2

        collector.gauge("cpu.usage", 67.8)
        assert collector.get_gauge_value("cpu.usage") == 67.8

    def test_gauge_with_labels(self, collector: MetricsCollector) -> None:
        """Test gauge with labels."""
        collector.gauge("memory.usage", 1024, labels={"host": "server1"})
        collector.gauge("memory.usage", 2048, labels={"host": "server2"})

        assert collector.get_gauge_value("memory.usage", labels={"host": "server1"}) == 1024
        assert collector.get_gauge_value("memory.usage", labels={"host": "server2"}) == 2048

    def test_histogram(self, collector: MetricsCollector) -> None:
        """Test histogram recording."""
        values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        for value in values:
            collector.histogram("response.time", value)

        stats = collector.get_histogram_stats("response.time")

        assert stats["count"] == 10
        assert stats["min"] == 10
        assert stats["max"] == 100
        assert stats["avg"] == 55
        assert stats["p50"] == 50

    def test_timer(self, collector: MetricsCollector) -> None:
        """Test timer context manager."""
        with collector.timer("operation.duration"):
            time.sleep(0.01)  # Sleep for 10ms

        stats = collector.get_histogram_stats("operation.duration")

        assert stats["count"] == 1
        assert stats["min"] >= 10  # At least 10ms
        assert stats["max"] >= 10

    def test_get_metric(self, collector: MetricsCollector) -> None:
        """Test getting metrics by name."""
        collector.counter("test.metric")
        collector.counter("test.metric")

        metrics = collector.get_metric("test.metric")

        assert len(metrics) == 2
        assert all(m.name == "test.metric" for m in metrics)

    def test_get_all_metrics(self, collector: MetricsCollector) -> None:
        """Test getting all metrics."""
        collector.counter("metric1")
        collector.gauge("metric2", 10)
        collector.histogram("metric3", 20)

        all_metrics = collector.get_all_metrics()

        assert "metric1" in all_metrics
        assert "metric2" in all_metrics
        assert "metric3" in all_metrics

    def test_clear(self, collector: MetricsCollector) -> None:
        """Test clearing metrics."""
        collector.counter("test.counter")
        collector.gauge("test.gauge", 10)

        assert collector.get_counter_value("test.counter") == 1

        collector.clear()

        assert collector.get_counter_value("test.counter") == 0
        assert collector.get_gauge_value("test.gauge") is None

    def test_prometheus_export(self, collector: MetricsCollector) -> None:
        """Test Prometheus format export."""
        collector.counter("requests_total", labels={"method": "GET"})
        collector.gauge("memory_usage_bytes", 1024, labels={"host": "server1"})

        prometheus_output = collector.export_prometheus()

        assert "requests_total" in prometheus_output
        assert "memory_usage_bytes" in prometheus_output
        assert 'method="GET"' in prometheus_output
        assert 'host="server1"' in prometheus_output


class TestSpan:
    """Test Span class."""

    def test_create_span(self) -> None:
        """Test span creation."""
        span = Span(
            span_id="span-123",
            trace_id="trace-456",
            name="test.operation",
            start_time=time.time(),
        )

        assert span.span_id == "span-123"
        assert span.trace_id == "trace-456"
        assert span.name == "test.operation"
        assert span.status == SpanStatus.UNSET
        assert span.end_time is None
        assert span.duration_ms is None

    def test_span_finish(self) -> None:
        """Test finishing a span."""
        span = Span(
            span_id="span-123", trace_id="trace-456", name="test", start_time=time.time()
        )

        time.sleep(0.01)
        span.finish(status=SpanStatus.OK)

        assert span.end_time is not None
        assert span.status == SpanStatus.OK
        assert span.duration_ms is not None
        assert span.duration_ms >= 10  # At least 10ms

    def test_span_with_error(self) -> None:
        """Test span with error."""
        span = Span(
            span_id="span-123", trace_id="trace-456", name="test", start_time=time.time()
        )

        span.finish(status=SpanStatus.ERROR, error="Something went wrong")

        assert span.status == SpanStatus.ERROR
        assert span.error == "Something went wrong"

    def test_span_attributes(self) -> None:
        """Test span attributes."""
        span = Span(
            span_id="span-123",
            trace_id="trace-456",
            name="test",
            start_time=time.time(),
            attributes={"key1": "value1"},
        )

        assert span.attributes["key1"] == "value1"

        span.set_attribute("key2", "value2")
        assert span.attributes["key2"] == "value2"

    def test_span_events(self) -> None:
        """Test span events."""
        span = Span(
            span_id="span-123", trace_id="trace-456", name="test", start_time=time.time()
        )

        span.add_event("checkpoint", attributes={"status": "processing"})
        span.add_event("completion")

        assert len(span.events) == 2
        assert span.events[0]["name"] == "checkpoint"
        assert span.events[0]["attributes"]["status"] == "processing"

    def test_span_to_dict(self) -> None:
        """Test span serialization."""
        span = Span(
            span_id="span-123",
            trace_id="trace-456",
            name="test",
            start_time=time.time(),
            attributes={"key": "value"},
        )

        span.finish(status=SpanStatus.OK)

        data = span.to_dict()

        assert data["span_id"] == "span-123"
        assert data["trace_id"] == "trace-456"
        assert data["name"] == "test"
        assert data["status"] == "ok"
        assert data["attributes"] == {"key": "value"}
        assert data["duration_ms"] is not None


class TestTracer:
    """Test Tracer."""

    @pytest.fixture
    def tracer(self) -> Tracer:
        """Create a tracer."""
        return Tracer()

    def test_start_trace(self, tracer: Tracer) -> None:
        """Test starting a trace."""
        trace_id = tracer.start_trace("test.trace")

        assert trace_id is not None
        assert trace_id in tracer._active_traces

    def test_start_span(self, tracer: Tracer) -> None:
        """Test starting a span."""
        trace_id = tracer.start_trace("test.trace")
        span = tracer.start_span("test.span", trace_id=trace_id)

        assert span.trace_id == trace_id
        assert span.name == "test.span"
        assert span.span_id is not None

    def test_finish_span(self, tracer: Tracer) -> None:
        """Test finishing a span."""
        trace_id = tracer.start_trace("test.trace")
        span = tracer.start_span("test.span", trace_id=trace_id)

        time.sleep(0.01)
        tracer.finish_span(span, status=SpanStatus.OK)

        assert span.end_time is not None
        assert span.status == SpanStatus.OK

    def test_span_context_manager(self, tracer: Tracer) -> None:
        """Test span context manager."""
        trace_id = tracer.start_trace("test.trace")

        with tracer.span("operation") as span:
            assert span.name == "operation"
            assert span.end_time is None

        assert span.end_time is not None
        assert span.status == SpanStatus.OK

    def test_span_context_manager_with_error(self, tracer: Tracer) -> None:
        """Test span context manager with error."""
        trace_id = tracer.start_trace("test.trace")

        with pytest.raises(ValueError):
            with tracer.span("operation") as span:
                raise ValueError("Test error")

        # Span should still be finished with error status
        assert span.status == SpanStatus.ERROR
        assert span.error == "Test error"

    def test_nested_spans(self, tracer: Tracer) -> None:
        """Test nested spans."""
        trace_id = tracer.start_trace("test.trace")

        with tracer.span("parent") as parent_span:
            with tracer.span("child") as child_span:
                assert child_span.parent_span_id == parent_span.span_id

    def test_get_span(self, tracer: Tracer) -> None:
        """Test getting a span by ID."""
        trace_id = tracer.start_trace("test.trace")
        span = tracer.start_span("test.span", trace_id=trace_id)

        retrieved_span = tracer.get_span(span.span_id)

        assert retrieved_span == span

    def test_get_trace_spans(self, tracer: Tracer) -> None:
        """Test getting all spans for a trace."""
        trace_id = tracer.start_trace("test.trace")

        span1 = tracer.start_span("span1", trace_id=trace_id)
        tracer.finish_span(span1)

        span2 = tracer.start_span("span2", trace_id=trace_id)
        tracer.finish_span(span2)

        trace_spans = tracer.get_trace_spans(trace_id)

        # Should have 3 spans: root + span1 + span2
        assert len(trace_spans) >= 3

    def test_clear(self, tracer: Tracer) -> None:
        """Test clearing tracer."""
        trace_id = tracer.start_trace("test.trace")
        span = tracer.start_span("test.span", trace_id=trace_id)

        assert len(tracer.get_all_spans()) > 0

        tracer.clear()

        assert len(tracer.get_all_spans()) == 0
        assert len(tracer._active_traces) == 0

    def test_jaeger_export(self, tracer: Tracer) -> None:
        """Test Jaeger format export."""
        trace_id = tracer.start_trace("test.trace")

        with tracer.span("operation") as span:
            span.set_attribute("key", "value")
            span.add_event("checkpoint")

        jaeger_trace = tracer.export_jaeger(trace_id)

        assert jaeger_trace["traceID"] == trace_id
        assert "spans" in jaeger_trace
        assert len(jaeger_trace["spans"]) >= 1


class TestMonitor:
    """Test Monitor."""

    @pytest.fixture
    def monitor(self) -> Monitor:
        """Create a monitor."""
        return Monitor(service_name="test_service")

    def test_create_monitor(self, monitor: Monitor) -> None:
        """Test monitor creation."""
        assert monitor.service_name == "test_service"
        assert monitor.metrics is not None
        assert monitor.tracer is not None

    def test_monitor_disabled_features(self) -> None:
        """Test monitor with disabled features."""
        monitor = Monitor(
            service_name="test", enable_metrics=False, enable_tracing=False
        )

        assert monitor.metrics is None
        assert monitor.tracer is None

    def test_monitor_metrics(self, monitor: Monitor) -> None:
        """Test monitor metrics integration."""
        monitor.metrics.counter("test.counter")
        monitor.metrics.gauge("test.gauge", 10)

        assert monitor.metrics.get_counter_value("test.counter") == 1
        assert monitor.metrics.get_gauge_value("test.gauge") == 10

    def test_monitor_tracing(self, monitor: Monitor) -> None:
        """Test monitor tracing integration."""
        trace_id = monitor.tracer.start_trace("test.trace")

        with monitor.tracer.span("operation"):
            pass

        spans = monitor.tracer.get_trace_spans(trace_id)
        assert len(spans) >= 1

    def test_monitor_log(self, monitor: Monitor) -> None:
        """Test monitor logging."""
        # Should not raise
        monitor.log("info", "Test message", extra={"key": "value"})

    def test_get_statistics(self, monitor: Monitor) -> None:
        """Test getting statistics."""
        monitor.metrics.counter("test")
        trace_id = monitor.tracer.start_trace("test")

        stats = monitor.get_statistics()

        assert stats["service"] == "test_service"
        assert stats["metrics_count"] >= 1
        assert stats["active_traces"] >= 1

    def test_clear(self, monitor: Monitor) -> None:
        """Test clearing monitor."""
        monitor.metrics.counter("test")
        trace_id = monitor.tracer.start_trace("test")

        monitor.clear()

        assert len(monitor.metrics.get_all_metrics()) == 0
        assert len(monitor.tracer.get_all_spans()) == 0


class TestMonitoringNode:
    """Test MonitoringNode."""

    def test_monitoring_node(self) -> None:
        """Test monitoring node in flow."""
        monitor = Monitor(service_name="test")
        monitor.metrics.counter("test.metric")

        node = MonitoringNode(monitor=monitor)
        shared = SharedStore()

        result = node.exec(shared)

        assert "stats" in result
        assert result["stats"]["service"] == "test"

    def test_monitoring_node_in_flow(self) -> None:
        """Test monitoring node in complete flow."""
        monitor = Monitor(service_name="test")

        # Add some metrics
        monitor.metrics.counter("flow.executions")
        monitor.metrics.gauge("flow.active", 1)

        node = MonitoringNode(monitor=monitor, metric_prefix="test_flow")

        flow = Flow(nodes=[node])
        result = flow.execute()

        # Check that monitoring metrics were recorded
        monitoring_result = result.get("MonitoringNode", {})
        assert "stats" in monitoring_result


class TestGetMonitor:
    """Test get_monitor singleton."""

    def test_get_monitor(self) -> None:
        """Test getting global monitor."""
        monitor1 = get_monitor()
        monitor2 = get_monitor()

        # Should be the same instance
        assert monitor1 is monitor2

    def test_get_monitor_service_name(self) -> None:
        """Test monitor service name."""
        monitor = get_monitor(service_name="my_service")

        # Note: This might not change the service name if monitor was already created
        assert monitor.service_name is not None


class TestIntegration:
    """Integration tests for monitoring."""

    def test_full_monitoring_workflow(self) -> None:
        """Test complete monitoring workflow."""
        monitor = Monitor(service_name="integration_test")

        # Start trace
        trace_id = monitor.tracer.start_trace("workflow")

        # Record metrics
        with monitor.tracer.span("database_query") as span:
            span.set_attribute("query", "SELECT * FROM users")

            with monitor.metrics.timer("db.query.duration"):
                time.sleep(0.01)

            monitor.metrics.counter("db.queries.total", labels={"table": "users"})

        # More operations
        with monitor.tracer.span("cache_lookup"):
            monitor.metrics.counter("cache.hits")

        # Get statistics
        stats = monitor.get_statistics()

        assert stats["spans_count"] >= 3  # workflow + database_query + cache_lookup
        assert stats["metrics_count"] >= 2

    def test_monitor_with_errors(self) -> None:
        """Test monitoring with errors."""
        monitor = Monitor(service_name="error_test")

        trace_id = monitor.tracer.start_trace("error_workflow")

        try:
            with monitor.tracer.span("failing_operation") as span:
                span.set_attribute("attempt", 1)
                raise ValueError("Simulated error")
        except ValueError:
            pass

        # Check that span was marked as error
        spans = monitor.tracer.get_trace_spans(trace_id)
        error_spans = [s for s in spans if s.status == SpanStatus.ERROR]

        assert len(error_spans) >= 1
        assert any(s.error == "Simulated error" for s in error_spans)
