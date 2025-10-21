"""
Monitoring Module Demo - Metrics, Logs, and Distributed Tracing

This demo showcases the comprehensive monitoring capabilities including:
- Metrics collection (counters, gauges, histograms, timers)
- Distributed tracing with spans
- Structured logging
- Prometheus and Jaeger export
- Flow integration

Copyright (c) 2025 Lionel TAGNE. All Rights Reserved.
This software is proprietary and confidential.
"""

import json
import random
import time
from typing import Dict, Any

from num_agents.core import Flow, Node, SharedStore
from num_agents.modules.monitoring import (
    Monitor,
    MetricsCollector,
    Tracer,
    MonitoringNode,
    SpanStatus,
    get_monitor,
)


def demo_1_basic_metrics():
    """Demo 1: Basic metrics collection."""
    print("\n" + "=" * 70)
    print("Demo 1: Basic Metrics Collection")
    print("=" * 70)

    metrics = MetricsCollector()

    # Counters
    print("\n📊 Counters:")
    print("  Simulating API requests...")
    for i in range(10):
        method = random.choice(["GET", "POST", "PUT"])
        status = random.choice(["200", "201", "404", "500"])

        metrics.counter(
            "http.requests.total",
            labels={"method": method, "status": status}
        )

    print(f"  Total GET requests: {metrics.get_counter_value('http.requests.total', labels={'method': 'GET', 'status': '200'})}")
    print(f"  Total POST requests: {metrics.get_counter_value('http.requests.total', labels={'method': 'POST', 'status': '201'})}")

    # Gauges
    print("\n📈 Gauges:")
    print("  Simulating resource monitoring...")
    for i in range(5):
        cpu_usage = random.uniform(20, 80)
        memory_mb = random.randint(512, 2048)

        metrics.gauge("system.cpu.percent", cpu_usage)
        metrics.gauge("system.memory.mb", memory_mb)
        time.sleep(0.1)

    print(f"  Current CPU: {metrics.get_gauge_value('system.cpu.percent'):.1f}%")
    print(f"  Current Memory: {metrics.get_gauge_value('system.memory.mb')} MB")

    # Histograms
    print("\n📊 Histograms:")
    print("  Recording response times...")
    response_times = [random.uniform(10, 200) for _ in range(100)]
    for rt in response_times:
        metrics.histogram("api.response.time.ms", rt)

    stats = metrics.get_histogram_stats("api.response.time.ms")
    print(f"  Min: {stats['min']:.2f}ms")
    print(f"  Max: {stats['max']:.2f}ms")
    print(f"  Avg: {stats['avg']:.2f}ms")
    print(f"  p50: {stats['p50']:.2f}ms")
    print(f"  p95: {stats['p95']:.2f}ms")
    print(f"  p99: {stats['p99']:.2f}ms")

    # Timers
    print("\n⏱️  Timers:")
    print("  Timing operations...")

    with metrics.timer("database.query.duration"):
        time.sleep(0.05)  # Simulate DB query

    with metrics.timer("cache.lookup.duration"):
        time.sleep(0.01)  # Simulate cache lookup

    db_stats = metrics.get_histogram_stats("database.query.duration")
    cache_stats = metrics.get_histogram_stats("cache.lookup.duration")

    print(f"  Database query: {db_stats['avg']:.2f}ms")
    print(f"  Cache lookup: {cache_stats['avg']:.2f}ms")


def demo_2_distributed_tracing():
    """Demo 2: Distributed tracing with spans."""
    print("\n" + "=" * 70)
    print("Demo 2: Distributed Tracing")
    print("=" * 70)

    tracer = Tracer()

    print("\n🔍 Creating a traced workflow...")

    # Start trace
    trace_id = tracer.start_trace("user_registration")
    print(f"  Trace ID: {trace_id[:8]}...")

    # Span 1: Validate input
    with tracer.span("validate_input") as span:
        span.set_attribute("email", "user@example.com")
        span.set_attribute("username", "john_doe")
        span.add_event("validation_started")
        time.sleep(0.01)
        span.add_event("validation_completed")

    # Span 2: Check existing user
    with tracer.span("check_existing_user") as span:
        span.set_attribute("query", "SELECT * FROM users WHERE email = ?")
        time.sleep(0.02)
        span.set_attribute("user_exists", False)

    # Span 3: Create user
    with tracer.span("create_user") as span:
        span.set_attribute("user_id", "usr_12345")
        span.add_event("database_insert")
        time.sleep(0.03)
        span.add_event("user_created")

    # Span 4: Send welcome email
    with tracer.span("send_welcome_email") as span:
        span.set_attribute("template", "welcome_email")
        span.set_attribute("recipient", "user@example.com")
        time.sleep(0.01)

    # Display trace results
    print("\n📋 Trace Summary:")
    spans = tracer.get_trace_spans(trace_id)
    print(f"  Total spans: {len(spans)}")

    for span in spans:
        if span.duration_ms:
            print(f"    - {span.name}: {span.duration_ms:.2f}ms ({span.status.value})")


def demo_3_nested_spans():
    """Demo 3: Nested spans and hierarchical tracing."""
    print("\n" + "=" * 70)
    print("Demo 3: Nested Spans (Hierarchical Tracing)")
    print("=" * 70)

    tracer = Tracer()

    print("\n🌳 Creating nested span hierarchy...")

    trace_id = tracer.start_trace("e_commerce_checkout")

    with tracer.span("checkout_process") as checkout:
        checkout.set_attribute("cart_total", 99.99)

        # Level 1: Inventory check
        with tracer.span("inventory_check") as inventory:
            inventory.set_attribute("items_count", 3)

            # Level 2: Individual item checks
            for item_id in ["item_1", "item_2", "item_3"]:
                with tracer.span(f"check_item_{item_id}") as item_span:
                    item_span.set_attribute("item_id", item_id)
                    time.sleep(0.005)

        # Level 1: Payment processing
        with tracer.span("payment_processing") as payment:
            payment.set_attribute("method", "credit_card")

            # Level 2: Validate payment
            with tracer.span("validate_payment") as validate:
                validate.set_attribute("card_last_4", "1234")
                time.sleep(0.01)

            # Level 2: Charge payment
            with tracer.span("charge_payment") as charge:
                charge.set_attribute("amount", 99.99)
                charge.add_event("payment_gateway_called")
                time.sleep(0.02)
                charge.add_event("payment_confirmed")

        # Level 1: Order creation
        with tracer.span("create_order") as order:
            order.set_attribute("order_id", "ord_789")
            time.sleep(0.015)

    # Visualize hierarchy
    print("\n📊 Span Hierarchy:")
    spans = tracer.get_trace_spans(trace_id)

    def print_span_tree(span_id, indent=0):
        span = tracer.get_span(span_id)
        if span and span.duration_ms:
            prefix = "  " * indent + ("└─ " if indent > 0 else "")
            print(f"{prefix}{span.name} ({span.duration_ms:.2f}ms)")

            # Find children
            children = [s for s in spans if s.parent_span_id == span_id]
            for child in children:
                print_span_tree(child.span_id, indent + 1)

    # Find root span
    root_spans = [s for s in spans if s.parent_span_id is None]
    for root in root_spans:
        print_span_tree(root.span_id)


def demo_4_error_handling():
    """Demo 4: Error handling and tracing."""
    print("\n" + "=" * 70)
    print("Demo 4: Error Handling and Tracing")
    print("=" * 70)

    monitor = Monitor(service_name="error_demo")

    print("\n⚠️  Simulating operations with errors...")

    trace_id = monitor.tracer.start_trace("error_workflow")

    # Successful operation
    try:
        with monitor.tracer.span("successful_operation") as span:
            span.set_attribute("attempt", 1)
            monitor.metrics.counter("operations.total")
            time.sleep(0.01)
            monitor.metrics.counter("operations.success")
            print("  ✅ Operation 1: Success")

    except Exception as e:
        monitor.metrics.counter("operations.errors")

    # Failed operation
    try:
        with monitor.tracer.span("failing_operation") as span:
            span.set_attribute("attempt", 1)
            monitor.metrics.counter("operations.total")
            # Simulate error
            raise ValueError("Database connection failed")

    except Exception as e:
        monitor.metrics.counter("operations.errors", labels={"error_type": "ValueError"})
        print(f"  ❌ Operation 2: Failed - {e}")

    # Retry logic with monitoring
    print("\n🔄 Retry with monitoring...")
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            with monitor.tracer.span(f"retry_operation_attempt_{attempt}") as span:
                span.set_attribute("attempt", attempt)
                monitor.metrics.counter("retry.attempts")

                # Simulate success on 3rd attempt
                if attempt < 3:
                    raise ConnectionError("Service unavailable")

                monitor.metrics.counter("retry.success")
                print(f"  ✅ Retry attempt {attempt}: Success")
                break

        except Exception as e:
            monitor.metrics.counter("retry.failures")
            print(f"  ❌ Retry attempt {attempt}: {e}")
            if attempt < max_retries:
                time.sleep(0.01)

    # Show error statistics
    print("\n📊 Error Statistics:")
    total = monitor.metrics.get_counter_value("operations.total")
    errors = monitor.metrics.get_counter_value("operations.errors")
    print(f"  Total operations: {total}")
    print(f"  Failed operations: {errors}")
    print(f"  Error rate: {(errors/total*100):.1f}%")

    # Show span statuses
    spans = monitor.tracer.get_trace_spans(trace_id)
    error_spans = [s for s in spans if s.status == SpanStatus.ERROR]
    print(f"\n  Spans with errors: {len(error_spans)}")
    for span in error_spans:
        print(f"    - {span.name}: {span.error}")


def demo_5_prometheus_export():
    """Demo 5: Prometheus export format."""
    print("\n" + "=" * 70)
    print("Demo 5: Prometheus Export")
    print("=" * 70)

    metrics = MetricsCollector()

    print("\n📊 Collecting metrics for Prometheus...")

    # Collect various metrics
    metrics.counter("requests_total", labels={"method": "GET", "status": "200"}, value=150)
    metrics.counter("requests_total", labels={"method": "POST", "status": "201"}, value=45)
    metrics.counter("requests_total", labels={"method": "GET", "status": "404"}, value=12)

    metrics.gauge("memory_usage_bytes", 1024 * 1024 * 512, labels={"host": "server1"})
    metrics.gauge("memory_usage_bytes", 1024 * 1024 * 768, labels={"host": "server2"})

    metrics.gauge("active_connections", 42, labels={"pool": "database"})
    metrics.gauge("active_connections", 15, labels={"pool": "cache"})

    # Export to Prometheus format
    print("\n📤 Prometheus Export Format:")
    print("-" * 50)
    prometheus_output = metrics.export_prometheus()
    print(prometheus_output)
    print("-" * 50)

    print("\n💡 This format can be scraped by Prometheus at /metrics endpoint")


def demo_6_jaeger_export():
    """Demo 6: Jaeger export format."""
    print("\n" + "=" * 70)
    print("Demo 6: Jaeger Export")
    print("=" * 70)

    tracer = Tracer()

    print("\n🔍 Creating trace for Jaeger export...")

    # Create a sample trace
    trace_id = tracer.start_trace("api_request")

    with tracer.span("authenticate_user") as span:
        span.set_attribute("user_id", "usr_123")
        span.set_attribute("auth_method", "jwt")
        time.sleep(0.01)

    with tracer.span("fetch_user_data") as span:
        span.set_attribute("database", "postgresql")
        span.add_event("query_started")
        time.sleep(0.02)
        span.add_event("query_completed", attributes={"rows": 1})

    with tracer.span("render_response") as span:
        span.set_attribute("format", "json")
        time.sleep(0.005)

    # Export to Jaeger format
    print("\n📤 Jaeger Export Format (JSON):")
    print("-" * 50)
    jaeger_trace = tracer.export_jaeger(trace_id)
    print(json.dumps(jaeger_trace, indent=2)[:500] + "...")
    print("-" * 50)

    print("\n💡 This format can be sent to Jaeger collector for visualization")


def demo_7_flow_integration():
    """Demo 7: Monitoring integration with flows."""
    print("\n" + "=" * 70)
    print("Demo 7: Flow Integration")
    print("=" * 70)

    monitor = Monitor(service_name="flow_demo")

    print("\n🔄 Creating monitored flow...")

    # Create custom monitored node
    class DataProcessorNode(Node):
        def exec(self, shared: SharedStore) -> Dict[str, Any]:
            # Get monitor
            mon = get_monitor()

            # Track execution
            mon.metrics.counter("nodes.executed", labels={"node": self.name})

            # Time the processing
            with mon.metrics.timer("node.duration", labels={"node": self.name}):
                # Simulate processing
                time.sleep(0.02)
                result = {"processed": 100, "errors": 0}

            return result

    # Create flow with monitoring
    processor = DataProcessorNode(name="DataProcessor")
    monitoring = MonitoringNode(monitor=monitor, metric_prefix="demo_flow")

    flow = Flow(nodes=[processor, monitoring])

    # Execute flow
    print("  Executing flow...")
    result = flow.execute()

    # Show monitoring results
    print("\n📊 Flow Monitoring Results:")
    mon_result = result.get("MonitoringNode", {})
    stats = mon_result.get("stats", {})

    print(f"  Service: {stats.get('service')}")
    print(f"  Metrics collected: {stats.get('metrics_count')}")
    print(f"  Spans created: {stats.get('spans_count')}")

    # Show node metrics
    processor_duration = monitor.metrics.get_histogram_stats(
        "node.duration",
        labels={"node": "DataProcessor"}
    )
    if processor_duration:
        print(f"  Node duration: {processor_duration.get('avg', 0):.2f}ms")


def demo_8_complete_workflow():
    """Demo 8: Complete monitoring workflow."""
    print("\n" + "=" * 70)
    print("Demo 8: Complete Monitoring Workflow")
    print("=" * 70)

    monitor = Monitor(service_name="complete_demo")

    print("\n🚀 Running complete monitored workflow...")

    # Start trace
    trace_id = monitor.tracer.start_trace("user_order_processing")

    try:
        # Step 1: Receive order
        with monitor.tracer.span("receive_order") as span:
            span.set_attribute("order_id", "ord_12345")
            span.set_attribute("items_count", 3)

            with monitor.metrics.timer("order.receive.duration"):
                time.sleep(0.01)

            monitor.metrics.counter("orders.received")
            monitor.log("info", "Order received", extra={"order_id": "ord_12345"})

        # Step 2: Validate order
        with monitor.tracer.span("validate_order") as span:
            span.add_event("validation_started")

            # Check inventory
            with monitor.tracer.span("check_inventory") as inv_span:
                with monitor.metrics.timer("inventory.check.duration"):
                    time.sleep(0.015)
                    inv_span.set_attribute("items_available", True)

            # Check payment
            with monitor.tracer.span("validate_payment") as pay_span:
                with monitor.metrics.timer("payment.validate.duration"):
                    time.sleep(0.02)
                    pay_span.set_attribute("payment_valid", True)

            span.add_event("validation_completed")
            monitor.metrics.counter("orders.validated")

        # Step 3: Process payment
        with monitor.tracer.span("process_payment") as span:
            span.set_attribute("amount", 299.99)
            span.set_attribute("currency", "USD")

            with monitor.metrics.timer("payment.process.duration"):
                time.sleep(0.025)
                span.add_event("payment_charged")

            monitor.metrics.counter("payments.processed")
            monitor.metrics.histogram("payment.amount", 299.99)

        # Step 4: Create shipment
        with monitor.tracer.span("create_shipment") as span:
            span.set_attribute("carrier", "FedEx")
            span.set_attribute("tracking_number", "1Z999AA1")

            with monitor.metrics.timer("shipment.create.duration"):
                time.sleep(0.01)

            monitor.metrics.counter("shipments.created")

        # Step 5: Send notification
        with monitor.tracer.span("send_notification") as span:
            span.set_attribute("channel", "email")
            span.set_attribute("template", "order_confirmation")

            with monitor.metrics.timer("notification.send.duration"):
                time.sleep(0.008)

            monitor.metrics.counter("notifications.sent", labels={"type": "email"})

        monitor.metrics.counter("orders.completed")
        print("  ✅ Order processing completed successfully")

    except Exception as e:
        monitor.metrics.counter("orders.failed")
        monitor.log("error", f"Order processing failed: {e}")
        print(f"  ❌ Order processing failed: {e}")

    # Display comprehensive statistics
    print("\n📊 Complete Workflow Statistics:")
    print("-" * 50)

    # Metrics summary
    print("\n  Metrics:")
    print(f"    Orders received: {monitor.metrics.get_counter_value('orders.received')}")
    print(f"    Orders completed: {monitor.metrics.get_counter_value('orders.completed')}")
    print(f"    Payments processed: {monitor.metrics.get_counter_value('payments.processed')}")
    print(f"    Notifications sent: {monitor.metrics.get_counter_value('notifications.sent')}")

    # Timing statistics
    print("\n  Timings:")
    for metric_name in ["order.receive.duration", "payment.process.duration", "shipment.create.duration"]:
        stats = monitor.metrics.get_histogram_stats(metric_name)
        if stats:
            print(f"    {metric_name}: {stats.get('avg', 0):.2f}ms")

    # Trace statistics
    print("\n  Tracing:")
    spans = monitor.tracer.get_trace_spans(trace_id)
    total_duration = sum(s.duration_ms for s in spans if s.duration_ms)
    print(f"    Total spans: {len(spans)}")
    print(f"    Total duration: {total_duration:.2f}ms")

    successful_spans = [s for s in spans if s.status == SpanStatus.OK]
    print(f"    Successful spans: {len(successful_spans)}")


def main():
    """Run all demos."""
    print("=" * 70)
    print("MONITORING MODULE - COMPREHENSIVE DEMO")
    print("=" * 70)
    print("\nThis demo showcases:")
    print("  1. Basic metrics collection (counters, gauges, histograms, timers)")
    print("  2. Distributed tracing with spans")
    print("  3. Nested spans and hierarchical tracing")
    print("  4. Error handling and monitoring")
    print("  5. Prometheus export format")
    print("  6. Jaeger export format")
    print("  7. Flow integration with MonitoringNode")
    print("  8. Complete monitoring workflow")

    demos = [
        demo_1_basic_metrics,
        demo_2_distributed_tracing,
        demo_3_nested_spans,
        demo_4_error_handling,
        demo_5_prometheus_export,
        demo_6_jaeger_export,
        demo_7_flow_integration,
        demo_8_complete_workflow,
    ]

    for demo in demos:
        demo()
        input("\n\n▶️  Press Enter to continue to next demo...")

    print("\n" + "=" * 70)
    print("All demos completed!")
    print("=" * 70)
    print("\nFor more information, see:")
    print("  - Documentation: docs/monitoring_guide.md")
    print("  - Tests: tests/test_monitoring.py")
    print("  - Module: num_agents/modules/monitoring.py")


if __name__ == "__main__":
    main()
