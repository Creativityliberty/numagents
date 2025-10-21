# Monitoring Module - Complete Guide

**Version:** 0.1.0
**Module:** `num_agents.modules.monitoring`

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Metrics](#metrics)
4. [Distributed Tracing](#distributed-tracing)
5. [Structured Logging](#structured-logging)
6. [Flow Integration](#flow-integration)
7. [Export Formats](#export-formats)
8. [Best Practices](#best-practices)
9. [API Reference](#api-reference)

---

## Overview

The Monitoring module provides comprehensive observability for Nüm Agents applications through:

- **Metrics Collection**: Counters, gauges, histograms, and timers
- **Distributed Tracing**: Track execution across nodes and flows with spans
- **Structured Logging**: Contextual logging with trace correlation
- **Prometheus Export**: Native Prometheus format support
- **Jaeger Export**: Distributed tracing in Jaeger format
- **Flow Integration**: Automatic monitoring with MonitoringNode

### When to Use

- **Production Monitoring**: Track application performance and health
- **Performance Analysis**: Identify bottlenecks with timers and histograms
- **Debugging**: Trace request flows across complex agent workflows
- **SLA Monitoring**: Track metrics and latencies
- **Capacity Planning**: Analyze resource usage patterns

---

## Quick Start

### Installation

The monitoring module is included in `num-agents`:

```bash
pip install num-agents
```

### Basic Example

```python
from num_agents.modules.monitoring import Monitor

# Create monitor
monitor = Monitor(service_name="my_agent")

# Track metrics
monitor.metrics.counter("requests.total")
monitor.metrics.gauge("memory.usage", 1024)

with monitor.metrics.timer("operation.duration"):
    # Your code here
    pass

# Distributed tracing
trace_id = monitor.tracer.start_trace("user_request")

with monitor.tracer.span("database_query") as span:
    span.set_attribute("query", "SELECT * FROM users")
    # Execute query

# Get statistics
stats = monitor.get_statistics()
print(f"Metrics: {stats['metrics_count']}")
print(f"Spans: {stats['spans_count']}")
```

---

## Metrics

### Metric Types

#### 1. Counters

Monotonically increasing values (e.g., requests, errors).

```python
from num_agents.modules.monitoring import MetricsCollector

metrics = MetricsCollector()

# Basic counter
metrics.counter("api.requests.total")

# Counter with custom increment
metrics.counter("bytes.sent", value=1024)

# Counter with labels
metrics.counter(
    "http.requests",
    labels={"method": "GET", "status": "200"}
)

# Get counter value
count = metrics.get_counter_value("api.requests.total")
```

#### 2. Gauges

Current value that can go up or down (e.g., memory usage, queue size).

```python
# Set gauge value
metrics.gauge("memory.usage.bytes", 2048)
metrics.gauge("queue.size", 42)

# Gauge with labels
metrics.gauge(
    "cpu.usage.percent",
    50.5,
    labels={"host": "server1", "core": "0"}
)

# Get gauge value
usage = metrics.get_gauge_value("memory.usage.bytes")
```

#### 3. Histograms

Distribution of values (e.g., response times, request sizes).

```python
# Record histogram values
for response_time in [10, 15, 20, 25, 30]:
    metrics.histogram("api.response.time.ms", response_time)

# Get histogram statistics
stats = metrics.get_histogram_stats("api.response.time.ms")
print(f"Min: {stats['min']}")
print(f"Max: {stats['max']}")
print(f"Avg: {stats['avg']}")
print(f"p50: {stats['p50']}")
print(f"p95: {stats['p95']}")
print(f"p99: {stats['p99']}")
```

#### 4. Timers

Measure duration of operations (context manager).

```python
# Time a block of code
with metrics.timer("database.query.duration"):
    # Execute database query
    result = db.execute("SELECT * FROM users")

# Timer with labels
with metrics.timer("cache.lookup", labels={"cache": "redis"}):
    value = cache.get("key")

# Multiple timings
for i in range(100):
    with metrics.timer("process.item"):
        process(item)

# Get timing statistics
stats = metrics.get_histogram_stats("process.item")
print(f"Average duration: {stats['avg']:.2f}ms")
```

### Labels

Labels add dimensions to metrics:

```python
# HTTP requests by method and status
metrics.counter("http.requests", labels={"method": "GET", "status": "200"})
metrics.counter("http.requests", labels={"method": "POST", "status": "201"})

# Get specific label combination
get_200_count = metrics.get_counter_value(
    "http.requests",
    labels={"method": "GET", "status": "200"}
)
```

---

## Distributed Tracing

### Traces and Spans

A **trace** represents a complete workflow, while **spans** represent individual operations.

```python
from num_agents.modules.monitoring import Tracer

tracer = Tracer()

# Start a trace
trace_id = tracer.start_trace("user_registration")

# Create spans
with tracer.span("validate_input") as span:
    span.set_attribute("email", "user@example.com")
    # Validation logic

with tracer.span("save_to_database") as span:
    span.set_attribute("table", "users")
    # Database save

with tracer.span("send_email") as span:
    span.set_attribute("template", "welcome")
    span.add_event("email_queued")
    # Send email

# Get all spans for trace
spans = tracer.get_trace_spans(trace_id)
print(f"Total spans: {len(spans)}")
```

### Nested Spans

Create hierarchical spans for complex operations:

```python
tracer = Tracer()
trace_id = tracer.start_trace("api_request")

with tracer.span("handle_request") as parent:
    parent.set_attribute("path", "/api/users")

    with tracer.span("authenticate") as auth_span:
        auth_span.set_attribute("user_id", "123")
        # Authentication logic

    with tracer.span("fetch_data") as data_span:
        data_span.set_attribute("query", "SELECT * FROM users")

        with tracer.span("query_cache") as cache_span:
            # Check cache
            pass

        with tracer.span("query_database") as db_span:
            # Query database
            pass
```

### Span Attributes and Events

Add context to spans:

```python
with tracer.span("payment_processing") as span:
    # Set attributes
    span.set_attribute("amount", 99.99)
    span.set_attribute("currency", "USD")
    span.set_attribute("payment_method", "credit_card")

    # Add events
    span.add_event("validation_started")
    span.add_event("payment_gateway_called", attributes={
        "gateway": "stripe",
        "request_id": "req_123"
    })
    span.add_event("payment_confirmed")
```

### Error Handling

Track errors in spans:

```python
try:
    with tracer.span("risky_operation") as span:
        span.set_attribute("attempt", 1)
        # Operation that might fail
        raise ValueError("Something went wrong")
except Exception as e:
    # Span automatically marked as ERROR with error message
    pass
```

---

## Structured Logging

### Basic Logging

```python
from num_agents.modules.monitoring import Monitor

monitor = Monitor(service_name="my_service")

# Log with levels
monitor.log("info", "Application started")
monitor.log("warning", "High memory usage detected")
monitor.log("error", "Failed to connect to database")

# Log with extra context
monitor.log("info", "User logged in", extra={
    "user_id": "123",
    "ip_address": "192.168.1.1"
})
```

### Trace-Correlated Logging

Logs automatically include trace context:

```python
monitor = Monitor(service_name="api_service")

trace_id = monitor.tracer.start_trace("user_request")

with monitor.tracer.span("process_request") as span:
    # Logs will include trace_id and span_id
    monitor.log("info", "Processing user request", extra={
        "user_id": "456",
        "action": "create_order"
    })
```

---

## Flow Integration

### MonitoringNode

Automatically collect metrics in flows:

```python
from num_agents.core import Flow, Node
from num_agents.modules.monitoring import Monitor, MonitoringNode

monitor = Monitor(service_name="my_flow")

# Add some metrics
monitor.metrics.counter("flow.executions")
monitor.metrics.gauge("flow.active_nodes", 3)

# Create monitoring node
monitoring_node = MonitoringNode(
    monitor=monitor,
    metric_prefix="flow"
)

# Add to flow
flow = Flow(nodes=[monitoring_node])

# Execute
result = flow.execute()

# Check monitoring stats
stats = result["MonitoringNode"]["stats"]
print(f"Metrics count: {stats['metrics_count']}")
```

### Custom Node Monitoring

Monitor custom nodes:

```python
from num_agents.core import Node, SharedStore
from num_agents.modules.monitoring import get_monitor

class DataProcessingNode(Node):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.monitor = get_monitor()

    def exec(self, shared: SharedStore):
        # Start span for this node
        with self.monitor.tracer.span(f"node.{self.name}") as span:
            # Track execution
            self.monitor.metrics.counter("nodes.executed", labels={
                "node": self.name
            })

            # Time the processing
            with self.monitor.metrics.timer("node.duration", labels={
                "node": self.name
            }):
                # Your processing logic
                result = self.process_data(shared)

            span.set_attribute("records_processed", len(result))

            return {"result": result}
```

---

## Export Formats

### Prometheus Export

Export metrics in Prometheus format:

```python
from num_agents.modules.monitoring import MetricsCollector

metrics = MetricsCollector()

# Collect some metrics
metrics.counter("requests_total", labels={"method": "GET"})
metrics.gauge("memory_usage_bytes", 1024, labels={"host": "server1"})

# Export to Prometheus format
prometheus_text = metrics.export_prometheus()
print(prometheus_text)

# Output:
# requests_total{method="GET"} 1
# memory_usage_bytes{host="server1"} 1024
```

Expose metrics endpoint:

```python
from flask import Flask, Response

app = Flask(__name__)
metrics = MetricsCollector()

@app.route("/metrics")
def metrics_endpoint():
    return Response(
        metrics.export_prometheus(),
        mimetype="text/plain"
    )
```

### Jaeger Export

Export traces in Jaeger format:

```python
from num_agents.modules.monitoring import Tracer
import json

tracer = Tracer()

# Create trace
trace_id = tracer.start_trace("api_request")
with tracer.span("operation1"):
    pass
with tracer.span("operation2"):
    pass

# Export to Jaeger format
jaeger_trace = tracer.export_jaeger(trace_id)
print(json.dumps(jaeger_trace, indent=2))
```

---

## Best Practices

### 1. Metric Naming

Use clear, hierarchical names:

```python
# Good
metrics.counter("http.requests.total")
metrics.gauge("database.connections.active")
metrics.histogram("api.response.time.ms")

# Avoid
metrics.counter("requests")  # Too generic
metrics.gauge("db_conn")  # Unclear abbreviation
```

### 2. Label Cardinality

Keep label combinations reasonable:

```python
# Good: Limited cardinality
metrics.counter("requests", labels={
    "method": "GET",  # ~10 values
    "status": "200"   # ~20 values
})

# Avoid: High cardinality
metrics.counter("requests", labels={
    "user_id": "12345",  # Thousands of values
    "timestamp": "2025-01-21T10:30:00"  # Unbounded
})
```

### 3. Span Naming

Use descriptive, consistent span names:

```python
# Good
with tracer.span("database.query.users"):
    pass
with tracer.span("cache.lookup.redis"):
    pass
with tracer.span("external.api.payment_gateway"):
    pass

# Avoid
with tracer.span("db"):  # Too generic
    pass
with tracer.span("DoStuffWithCache"):  # Inconsistent casing
    pass
```

### 4. Error Handling

Always handle errors in monitored code:

```python
try:
    with tracer.span("critical_operation") as span:
        with metrics.timer("operation.duration"):
            result = perform_operation()

        # Success metrics
        metrics.counter("operation.success")
        span.set_attribute("result_count", len(result))

except Exception as e:
    # Error metrics
    metrics.counter("operation.errors", labels={
        "error_type": type(e).__name__
    })
    # Span is automatically marked as ERROR
    raise
```

### 5. Resource Cleanup

Clear monitoring data when needed:

```python
# In tests
def test_something():
    monitor = Monitor(service_name="test")

    # Test code
    monitor.metrics.counter("test.counter")

    # Cleanup
    monitor.clear()

# In production (periodic cleanup)
import schedule

def cleanup_old_metrics():
    monitor = get_monitor()
    # Keep last hour of data, clear older
    monitor.clear()

schedule.every().hour.do(cleanup_old_metrics)
```

### 6. Sampling for High-Traffic

Sample traces in high-traffic scenarios:

```python
import random

monitor = Monitor(service_name="high_traffic_api")

# Sample 10% of requests
if random.random() < 0.1:
    trace_id = monitor.tracer.start_trace("api_request")
    with monitor.tracer.span("handle_request"):
        # Process request with tracing
        pass
else:
    # Process without tracing (just metrics)
    monitor.metrics.counter("requests.total")
```

---

## API Reference

### Monitor

```python
Monitor(
    service_name: str = "num_agents",
    enable_metrics: bool = True,
    enable_tracing: bool = True,
    enable_logging: bool = True
)
```

Main monitoring interface.

**Attributes:**
- `metrics`: MetricsCollector instance
- `tracer`: Tracer instance

**Methods:**
- `log(level, message, extra)`: Log structured message
- `get_statistics()`: Get monitoring statistics
- `clear()`: Clear all monitoring data

### MetricsCollector

**Methods:**

- `counter(name, value=1, labels=None, description=None)`: Increment counter
- `gauge(name, value, labels=None, description=None)`: Set gauge value
- `histogram(name, value, labels=None, description=None)`: Record histogram value
- `timer(name, labels=None, description=None)`: Time code block (context manager)
- `get_counter_value(name, labels=None)`: Get counter value
- `get_gauge_value(name, labels=None)`: Get gauge value
- `get_histogram_stats(name, labels=None)`: Get histogram statistics
- `get_metric(name)`: Get all metrics with name
- `get_all_metrics()`: Get all metrics
- `clear()`: Clear all metrics
- `export_prometheus()`: Export in Prometheus format

### Tracer

**Methods:**

- `start_trace(name, trace_id=None)`: Start new trace
- `start_span(name, trace_id=None, parent_span_id=None, attributes=None)`: Start span
- `finish_span(span, status=SpanStatus.OK, error=None)`: Finish span
- `span(name, attributes=None)`: Span context manager
- `get_span(span_id)`: Get span by ID
- `get_trace_spans(trace_id)`: Get all spans for trace
- `get_all_spans()`: Get all spans
- `clear()`: Clear all spans
- `export_jaeger(trace_id)`: Export trace in Jaeger format

### Span

**Attributes:**
- `span_id`: Unique span ID
- `trace_id`: Parent trace ID
- `name`: Span name
- `start_time`: Start timestamp
- `end_time`: End timestamp (or None)
- `status`: SpanStatus (OK, ERROR, UNSET)
- `attributes`: Dictionary of attributes
- `events`: List of events
- `error`: Error message (if any)

**Methods:**
- `set_attribute(key, value)`: Set attribute
- `add_event(name, attributes=None)`: Add event
- `finish(status=SpanStatus.OK, error=None)`: Finish span
- `to_dict()`: Serialize to dictionary

**Properties:**
- `duration_ms`: Duration in milliseconds

### Metric

**Attributes:**
- `name`: Metric name
- `type`: MetricType (COUNTER, GAUGE, HISTOGRAM, TIMER)
- `value`: Metric value
- `labels`: Dictionary of labels
- `timestamp`: Creation timestamp
- `description`: Optional description

**Methods:**
- `to_dict()`: Serialize to dictionary

### MonitoringNode

```python
MonitoringNode(
    monitor: Monitor,
    metric_prefix: str = "flow",
    name: Optional[str] = None,
    enable_logging: bool = True
)
```

Node for flow monitoring integration.

### get_monitor()

```python
get_monitor(
    service_name: str = "num_agents",
    enable_metrics: bool = True,
    enable_tracing: bool = True,
    enable_logging: bool = True
) -> Monitor
```

Get or create global monitor singleton.

---

## Advanced Topics

### OpenTelemetry Integration

The monitoring module is designed to work with OpenTelemetry:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

# Setup OpenTelemetry
trace.set_tracer_provider(TracerProvider())
otel_tracer = trace.get_tracer(__name__)

# Use with Nüm Agents monitoring
monitor = Monitor(service_name="my_service")

# Export Nüm spans to OpenTelemetry
for span in monitor.tracer.get_all_spans():
    with otel_tracer.start_as_current_span(span.name) as otel_span:
        for key, value in span.attributes.items():
            otel_span.set_attribute(key, value)
```

### Custom Exporters

Create custom metric exporters:

```python
class CloudWatchExporter:
    def __init__(self, metrics_collector):
        self.metrics = metrics_collector

    def export(self):
        import boto3
        cloudwatch = boto3.client('cloudwatch')

        for name, metric_list in self.metrics.get_all_metrics().items():
            for metric in metric_list:
                cloudwatch.put_metric_data(
                    Namespace='NumAgents',
                    MetricData=[{
                        'MetricName': metric.name,
                        'Value': metric.value,
                        'Unit': 'Count',
                        'Dimensions': [
                            {'Name': k, 'Value': v}
                            for k, v in metric.labels.items()
                        ]
                    }]
                )

# Usage
metrics = MetricsCollector()
metrics.counter("requests.total")

exporter = CloudWatchExporter(metrics)
exporter.export()
```

### Performance Monitoring Dashboard

Combine metrics and tracing for dashboards:

```python
from num_agents.modules.monitoring import Monitor

class PerformanceDashboard:
    def __init__(self, monitor: Monitor):
        self.monitor = monitor

    def get_dashboard_data(self):
        stats = self.monitor.get_statistics()

        # Get key metrics
        request_count = self.monitor.metrics.get_counter_value("requests.total")
        error_count = self.monitor.metrics.get_counter_value("errors.total")

        # Get latency percentiles
        latency_stats = self.monitor.metrics.get_histogram_stats("request.duration")

        # Get active traces
        active_traces = stats.get("active_traces", 0)

        return {
            "requests": {
                "total": request_count,
                "errors": error_count,
                "error_rate": error_count / request_count if request_count > 0 else 0
            },
            "latency": {
                "p50": latency_stats.get("p50", 0),
                "p95": latency_stats.get("p95", 0),
                "p99": latency_stats.get("p99", 0)
            },
            "tracing": {
                "active_traces": active_traces,
                "total_spans": stats.get("spans_count", 0)
            }
        }
```

---

## Copyright

Copyright (c) 2025 Lionel TAGNE. All Rights Reserved.
This software is proprietary and confidential.
