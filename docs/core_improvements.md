# Core Improvements - Phase 1.2

This document describes the improvements made to the core architecture of Nüm Agents SDK in Phase 1.2.

## Overview

Phase 1.2 focused on making the core components more robust, observable, and flexible through:
- Enhanced error handling with custom exceptions
- Structured logging throughout the SDK
- Hooks for lifecycle events
- Conditional execution support
- Flow serialization/deserialization
- Runtime validation

---

## 1. Custom Exception System

All SDK exceptions now inherit from `NumAgentsException` and provide detailed error information.

### Available Exceptions

```python
from num_agents.exceptions import (
    # Flow exceptions
    FlowConfigurationError,  # Flow is misconfigured
    FlowExecutionError,      # Flow execution failed

    # Node exceptions
    NodeExecutionError,      # Node execution failed
    NodeNotImplementedError, # Node.exec() not implemented

    # Store exceptions
    SharedStoreKeyError,     # Required key missing

    # Serialization exceptions
    SerializationError,      # Serialization failed
    DeserializationError,    # Deserialization failed
)
```

### Exception Details

All exceptions support a `details` parameter for additional context:

```python
try:
    result = node.exec(shared)
except NodeExecutionError as e:
    print(e.message)      # Error message
    print(e.details)      # Additional details dict
    print(e.node_name)    # Name of failed node
    print(e.node_id)      # ID of failed node
```

---

## 2. Structured Logging

The SDK now includes comprehensive logging capabilities.

### Enabling Logging

```python
from num_agents import Flow, Node, configure_logging
import logging

# Configure logging for entire SDK
configure_logging(level=logging.INFO)

# Or enable per-component
flow = Flow(name="MyFlow", enable_logging=True)
node = MyNode(name="MyNode", enable_logging=True)
store = SharedStore(enable_logging=True)
```

### Log Levels

- **DEBUG**: Detailed execution traces
- **INFO**: Important execution events
- **WARNING**: Recoverable errors, retries
- **ERROR**: Execution failures

### Example Output

```
2025-01-15 10:30:00 - num_agents.core - INFO - Starting flow execution: MyFlow
2025-01-15 10:30:00 - num_agents.core - DEBUG - Executing node: ProcessNode
2025-01-15 10:30:01 - num_agents.core - INFO - Node ProcessNode completed in 0.543s
2025-01-15 10:30:01 - num_agents.core - INFO - Flow 'MyFlow' completed in 1.234s (3 nodes executed)
```

---

## 3. Enhanced SharedStore

The `SharedStore` now has additional methods for better data management.

### New Methods

```python
from num_agents import SharedStore

store = SharedStore()

# Get required value (raises exception if missing)
value = store.get_required("key")  # Raises SharedStoreKeyError if not found

# Update with multiple values
store.update({"key1": "value1", "key2": "value2"})

# Get all data as dict (copy)
data = store.to_dict()

# Check store size
size = len(store)
```

### Example Usage

```python
class MyNode(Node):
    def exec(self, shared: SharedStore) -> Dict[str, Any]:
        # Require certain keys to exist
        try:
            user_id = shared.get_required("user_id")
            config = shared.get_required("config")
        except SharedStoreKeyError as e:
            logger.error(f"Missing required key: {e.key}")
            raise

        # Process with required data
        result = process(user_id, config)

        # Update multiple values
        shared.update({
            "result": result,
            "processed_at": time.time(),
            "status": "completed"
        })

        return {"success": True}
```

---

## 4. Node Improvements

### Retry Mechanism

Nodes can now automatically retry on failure with exponential backoff:

```python
class UnstableAPINode(Node):
    def __init__(self):
        super().__init__(
            name="APICall",
            retry_count=3,  # Retry up to 3 times
            enable_logging=True
        )

    def exec(self, shared: SharedStore) -> Dict[str, Any]:
        response = call_external_api()
        return {"data": response}
```

Retry behavior:
- Attempt 1: immediate
- Attempt 2: wait 1 second
- Attempt 3: wait 2 seconds
- Attempt 4: wait 4 seconds

### Execution Hooks

Add custom logic before/after node execution:

```python
node = ProcessNode()

# Before execution
def before_hook(shared: SharedStore) -> None:
    shared.set("start_time", time.time())
    logger.info("Starting processing...")

node.add_before_hook(before_hook)

# After successful execution
def after_hook(shared: SharedStore, result: Dict[str, Any]) -> None:
    duration = time.time() - shared.get("start_time")
    logger.info(f"Processing completed in {duration:.2f}s")
    shared.set("processing_duration", duration)

node.add_after_hook(after_hook)

# On error
def error_hook(shared: SharedStore, error: Exception) -> None:
    logger.error(f"Processing failed: {str(error)}")
    shared.set("error_type", type(error).__name__)

node.add_error_hook(error_hook)
```

---

## 5. Flow Improvements

### Validation

Flows are now validated before execution:

```python
flow = Flow(name="MyFlow")
flow.add_node(node1)
flow.add_node(node2)
flow.set_start(node1)

# Explicit validation
try:
    flow.validate()
except FlowConfigurationError as e:
    print(f"Flow configuration error: {e}")

# Automatic validation on execute()
results = flow.execute()  # Validates first
```

Validation checks:
- Start node is defined
- No cycles in flow graph
- All nodes are reachable

### Fail-Fast vs Continue-on-Error

Control error handling behavior:

```python
# Fail-fast (default): stop on first error
flow = Flow(fail_fast=True)

# Continue-on-error: execute all possible nodes
flow = Flow(fail_fast=False)
results = flow.execute()
if "_errors" in results:
    print("Some nodes failed:", results["_errors"])
```

### Initial Data

Pass initial data to the flow:

```python
flow = Flow(nodes=[node1, node2, node3])

initial_data = {
    "user_id": "12345",
    "config": {"mode": "production"},
    "timestamp": time.time()
}

results = flow.execute(initial_data=initial_data)
```

### Execution Metadata

Track flow execution history:

```python
flow.execute()

# Get execution metrics
print(f"Last execution time: {flow.last_execution_time:.3f}s")

# Review execution history
for execution in flow.execution_history:
    print(f"Timestamp: {execution['timestamp']}")
    print(f"Duration: {execution['duration']:.3f}s")
    print(f"Nodes executed: {execution['nodes_executed']}")
    print(f"Success: {execution['success']}")
    print(f"Errors: {execution['errors']}")
```

### Flow Hooks

Add hooks at the flow level:

```python
flow = Flow(name="DataPipeline")

# Before flow execution
def before_flow(shared: SharedStore) -> None:
    shared.set("pipeline_start", time.time())
    logger.info("Starting data pipeline")

flow.add_before_hook(before_flow)

# After successful flow execution
def after_flow(shared: SharedStore, results: Dict[str, Any]) -> None:
    duration = time.time() - shared.get("pipeline_start")
    logger.info(f"Pipeline completed in {duration:.2f}s")

    # Save results to database
    save_results(results)

flow.add_after_hook(after_flow)

# On flow error
def on_error(shared: SharedStore, error: Exception) -> None:
    logger.error(f"Pipeline failed: {str(error)}")
    alert_team(error)

flow.add_error_hook(on_error)
```

---

## 6. Conditional Execution

The new `ConditionalNode` enables conditional branching in flows:

```python
from num_agents import ConditionalNode, Node, Flow, SharedStore

# Define condition
def check_threshold(shared: SharedStore) -> bool:
    return shared.get("value", 0) > 100

# Define true/false paths
high_value_node = HighValueProcessing()
low_value_node = LowValueProcessing()

# Create conditional node
conditional = ConditionalNode(
    condition=check_threshold,
    true_node=high_value_node,
    false_node=low_value_node,
    name="ValueCheck"
)

# Use in flow
flow = Flow()
flow.add_node(data_loader)
flow.add_node(conditional)
flow.add_transition(data_loader, conditional)
flow.set_start(data_loader)

results = flow.execute()
```

### Condition Result

The condition result is stored in the shared store:

```python
# Access condition result
condition_result = shared.get("ValueCheck_condition_result")
```

---

## 7. Flow Serialization

Flows can now be serialized to JSON for storage and transfer:

```python
from num_agents import Flow, FlowSerializer, FlowDeserializer

# Create a flow
flow = Flow(name="MyFlow", enable_logging=True)
flow.add_node(node1)
flow.add_node(node2)
flow.set_start(node1)

# Serialize to JSON
json_str = FlowSerializer.flow_to_json(flow, indent=2)

# Save to file
FlowSerializer.flow_to_file(flow, "my_flow.json")

# Deserialize from JSON
deserializer = FlowDeserializer(node_registry={
    "MyNode": MyNode,
    "ProcessNode": ProcessNode
})

flow = deserializer.flow_from_json(json_str)
# Or from file
flow = deserializer.flow_from_file("my_flow.json")
```

### Serialization Format

```json
{
  "name": "MyFlow",
  "start_node_id": "abc123...",
  "config": {
    "enable_logging": true,
    "fail_fast": true
  },
  "nodes": [
    {
      "id": "abc123...",
      "name": "Node1",
      "class": "MyNode",
      "module": "my_module",
      "next_node_ids": ["def456..."],
      "config": {
        "enable_logging": false,
        "retry_count": 0,
        "timeout": null
      }
    }
  ],
  "execution_history": []
}
```

---

## 8. Complete Example

Here's a complete example using all new features:

```python
from num_agents import (
    Node, Flow, SharedStore, ConditionalNode,
    FlowSerializer, configure_logging
)
import logging

# Configure logging
configure_logging(level=logging.INFO)

# Define custom nodes
class DataLoader(Node):
    def exec(self, shared: SharedStore) -> Dict[str, Any]:
        data = load_data()
        shared.set("data", data)
        shared.set("record_count", len(data))
        return {"loaded": len(data)}

class DataValidator(Node):
    def __init__(self):
        super().__init__(
            name="Validator",
            retry_count=2,  # Retry validation
            enable_logging=True
        )

    def exec(self, shared: SharedStore) -> Dict[str, Any]:
        data = shared.get_required("data")
        is_valid = validate(data)
        shared.set("is_valid", is_valid)
        return {"valid": is_valid}

class DataProcessor(Node):
    def exec(self, shared: SharedStore) -> Dict[str, Any]:
        data = shared.get_required("data")
        result = process(data)
        shared.set("result", result)
        return {"processed": True}

# Build flow with conditional
loader = DataLoader()
validator = DataValidator()
processor = DataProcessor()

# Conditional based on validation
def is_valid(shared: SharedStore) -> bool:
    return shared.get("is_valid", False)

conditional = ConditionalNode(
    condition=is_valid,
    true_node=processor,
    name="ValidityCheck"
)

# Create flow
flow = Flow(
    name="DataPipeline",
    enable_logging=True,
    fail_fast=True
)

flow.add_node(loader)
flow.add_node(validator)
flow.add_node(conditional)

flow.add_transition(loader, validator)
flow.add_transition(validator, conditional)
flow.set_start(loader)

# Add hooks
def before_pipeline(shared: SharedStore) -> None:
    shared.set("start_time", time.time())

def after_pipeline(shared: SharedStore, results: Dict[str, Any]) -> None:
    duration = time.time() - shared.get("start_time")
    print(f"Pipeline completed in {duration:.2f}s")

flow.add_before_hook(before_pipeline)
flow.add_after_hook(after_pipeline)

# Execute
try:
    results = flow.execute()
    print("Success:", results)
except FlowExecutionError as e:
    print(f"Pipeline failed: {e}")

# Save flow configuration
FlowSerializer.flow_to_file(flow, "pipeline.json")
```

---

## Best Practices

1. **Always enable logging in production**
   ```python
   flow = Flow(enable_logging=True)
   ```

2. **Use retry for unstable operations**
   ```python
   api_node = APINode(retry_count=3)
   ```

3. **Use hooks for cross-cutting concerns**
   - Metrics collection
   - Audit logging
   - Error notification

4. **Validate flows before deployment**
   ```python
   flow.validate()  # Catch config errors early
   ```

5. **Use fail_fast=False for data pipelines**
   ```python
   # Process all records, collect errors
   flow = Flow(fail_fast=False)
   ```

6. **Serialize flows for version control**
   ```python
   FlowSerializer.flow_to_file(flow, "flows/v1.0.0.json")
   ```

---

## Migration Guide

### Updating Existing Code

If you have existing flows, here's how to adopt the new features:

**Before:**
```python
flow = Flow(nodes=[node1, node2])
flow.execute()
```

**After (recommended):**
```python
flow = Flow(
    nodes=[node1, node2],
    name="MyFlow",
    enable_logging=True,
    fail_fast=True
)

try:
    results = flow.execute(initial_data={"config": config})
except FlowExecutionError as e:
    logger.error(f"Flow failed: {e}")
    handle_error(e)
```

### Backward Compatibility

All new features are opt-in. Existing code will continue to work without changes.

---

## Performance Considerations

- **Logging**: Minimal overhead when disabled (default)
- **Hooks**: ~1-2ms overhead per hook
- **Retry**: Exponential backoff adds delay only on failures
- **Validation**: One-time cost at flow start (~1ms for typical flows)
- **Serialization**: Fast for flows <1000 nodes

---

## Next Steps

Phase 1.3 will add:
- CI/CD pipeline setup
- Code coverage tracking
- Pre-commit hooks
- Automated testing

See [Phase 1.3 Documentation](phase_1_3.md) for details.
