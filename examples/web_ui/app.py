"""
Nüm Agents Web UI - Flow Visualization and Debugging

A web-based dashboard for visualizing and debugging agent flows:
- Flow visualization with node graph
- Real-time metrics dashboard
- Distributed tracing viewer
- Goal and task management
- Interactive flow execution

Copyright (c) 2025 Lionel TAGNE. All Rights Reserved.
This software is proprietary and confidential.
"""

import json
from datetime import datetime
from typing import Any, Dict, List

from flask import Flask, render_template, request, jsonify

from num_agents.core import Flow, Node, SharedStore
from num_agents.modules.monitoring import get_monitor, Monitor
from num_agents.modules.structure_agent_ia import GoalManager, TaskManager, Status
from num_agents.serialization import FlowSerializer

app = Flask(__name__)

# Initialize global components
monitor = get_monitor(service_name="web_ui")
goal_manager = GoalManager()
task_manager = TaskManager()

# Store flows
flows: Dict[str, Flow] = {}
flow_results: Dict[str, List[Dict]] = {}


# ============================================================================
# Routes
# ============================================================================


@app.route("/")
def index():
    """Main dashboard."""
    return render_template("dashboard.html")


@app.route("/api/stats")
def get_stats():
    """Get overall statistics."""
    stats = monitor.get_statistics()

    # Add goal/task stats
    goal_stats = goal_manager.get_statistics()
    task_stats = task_manager.get_statistics()

    return jsonify({
        "monitoring": stats,
        "goals": goal_stats,
        "tasks": task_stats,
        "flows": {
            "total": len(flows),
            "executions": sum(len(results) for results in flow_results.values())
        }
    })


@app.route("/api/metrics")
def get_metrics():
    """Get current metrics."""
    all_metrics = monitor.metrics.get_all_metrics()

    # Convert metrics to JSON-friendly format
    metrics_data = {}
    for name, metric_list in all_metrics.items():
        metrics_data[name] = [m.to_dict() for m in metric_list]

    return jsonify(metrics_data)


@app.route("/api/metrics/prometheus")
def get_prometheus_metrics():
    """Get Prometheus formatted metrics."""
    prometheus_text = monitor.metrics.export_prometheus()
    return prometheus_text, 200, {'Content-Type': 'text/plain; charset=utf-8'}


@app.route("/api/traces")
def get_traces():
    """Get all traces."""
    traces = []
    for trace_id in monitor.tracer._active_traces:
        spans = monitor.tracer.get_trace_spans(trace_id)
        if spans:
            traces.append({
                "trace_id": trace_id,
                "span_count": len(spans),
                "start_time": min(s.start_time for s in spans),
                "duration_ms": sum(s.duration_ms or 0 for s in spans)
            })

    return jsonify(traces)


@app.route("/api/traces/<trace_id>")
def get_trace_detail(trace_id):
    """Get trace details."""
    spans = monitor.tracer.get_trace_spans(trace_id)

    return jsonify({
        "trace_id": trace_id,
        "spans": [s.to_dict() for s in spans]
    })


@app.route("/api/traces/<trace_id>/jaeger")
def get_trace_jaeger(trace_id):
    """Get trace in Jaeger format."""
    jaeger_data = monitor.tracer.export_jaeger(trace_id)
    return jsonify(jaeger_data)


@app.route("/api/goals")
def get_goals():
    """Get all goals."""
    goals = goal_manager.list_all()
    return jsonify({
        "goals": [g.to_dict() for g in goals],
        "stats": goal_manager.get_statistics()
    })


@app.route("/api/goals", methods=["POST"])
def create_goal():
    """Create a new goal."""
    data = request.json
    from num_agents.modules.structure_agent_ia import Priority

    goal_id = goal_manager.add_goal(
        description=data.get("description", ""),
        priority=Priority(data.get("priority", "medium")),
        metadata=data.get("metadata", {})
    )

    return jsonify({"goal_id": goal_id})


@app.route("/api/tasks")
def get_tasks():
    """Get all tasks."""
    tasks = task_manager.list_all()
    return jsonify({
        "tasks": [t.to_dict() for t in tasks],
        "stats": task_manager.get_statistics()
    })


@app.route("/api/tasks", methods=["POST"])
def create_task():
    """Create a new task."""
    data = request.json
    from num_agents.modules.structure_agent_ia import Priority

    task_id = task_manager.add_task(
        description=data.get("description", ""),
        goal_id=data.get("goal_id"),
        priority=Priority(data.get("priority", "medium")),
        dependencies=data.get("dependencies", [])
    )

    return jsonify({"task_id": task_id})


@app.route("/api/flows")
def get_flows():
    """Get all flows."""
    flows_data = []
    for flow_id, flow in flows.items():
        flows_data.append({
            "id": flow_id,
            "name": getattr(flow, 'name', flow_id),
            "node_count": len(flow.nodes),
            "executions": len(flow_results.get(flow_id, []))
        })

    return jsonify(flows_data)


@app.route("/api/flows/<flow_id>")
def get_flow_detail(flow_id):
    """Get flow details."""
    flow = flows.get(flow_id)
    if not flow:
        return jsonify({"error": "Flow not found"}), 404

    # Serialize flow
    flow_data = FlowSerializer.serialize_flow(flow)

    # Add execution history
    flow_data["executions"] = flow_results.get(flow_id, [])

    return jsonify(flow_data)


@app.route("/api/flows/<flow_id>/execute", methods=["POST"])
def execute_flow(flow_id):
    """Execute a flow."""
    flow = flows.get(flow_id)
    if not flow:
        return jsonify({"error": "Flow not found"}), 404

    # Get initial data
    initial_data = request.json.get("initial_data", {})

    # Start trace
    trace_id = monitor.tracer.start_trace(f"flow_{flow_id}_execution")

    with monitor.tracer.span("flow_execution") as span:
        span.set_attribute("flow_id", flow_id)

        # Execute flow
        try:
            result = flow.execute(initial_data=initial_data)

            # Store result
            if flow_id not in flow_results:
                flow_results[flow_id] = []

            flow_results[flow_id].append({
                "timestamp": datetime.now().isoformat(),
                "trace_id": trace_id,
                "result": result,
                "success": True
            })

            monitor.metrics.counter("flows.executions.success", labels={"flow_id": flow_id})

            return jsonify({
                "success": True,
                "trace_id": trace_id,
                "result": result
            })

        except Exception as e:
            monitor.metrics.counter("flows.executions.failed", labels={"flow_id": flow_id})

            return jsonify({
                "success": False,
                "error": str(e),
                "trace_id": trace_id
            }), 500


@app.route("/api/flows/register", methods=["POST"])
def register_flow():
    """Register a new flow."""
    data = request.json

    flow_id = data.get("id", f"flow_{len(flows) + 1}")

    # Create simple flow from node list
    # This is a simplified version - in production, use FlowDeserializer
    nodes = []
    for node_data in data.get("nodes", []):
        # Create a simple test node
        class TestNode(Node):
            def exec(self, shared):
                return {"message": "Node executed"}

        node = TestNode(name=node_data.get("name", "TestNode"))
        nodes.append(node)

    flow = Flow(nodes=nodes)
    flows[flow_id] = flow

    return jsonify({"flow_id": flow_id})


# ============================================================================
# Main
# ============================================================================


def create_sample_flow():
    """Create a sample flow for demonstration."""
    class WelcomeNode(Node):
        def exec(self, shared):
            return {"message": "Welcome to Nüm Agents!"}

    class ProcessNode(Node):
        def exec(self, shared):
            data = shared.get("data", [])
            return {"processed": len(data), "data": data}

    flow = Flow(nodes=[WelcomeNode(), ProcessNode()])
    flows["sample_flow"] = flow
    flow_results["sample_flow"] = []


if __name__ == "__main__":
    print("=" * 70)
    print("Nüm Agents Web UI")
    print("=" * 70)
    print("\nStarting dashboard...")

    # Create sample flow
    create_sample_flow()

    print("\n✅ Dashboard ready!")
    print("\n🌐 Open http://localhost:5000 in your browser")
    print("\nAvailable endpoints:")
    print("  /                     - Main dashboard")
    print("  /api/stats            - Overall statistics")
    print("  /api/metrics          - Metrics data")
    print("  /api/metrics/prometheus - Prometheus format")
    print("  /api/traces           - All traces")
    print("  /api/goals            - Goals management")
    print("  /api/tasks            - Tasks management")
    print("  /api/flows            - Flows management")
    print("\n" + "=" * 70 + "\n")

    app.run(debug=True, host="0.0.0.0", port=5000)
