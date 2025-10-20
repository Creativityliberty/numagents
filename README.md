# Nüm Agents SDK

A dynamic agent orchestration framework for building modular, scalable AI agent systems.

## Overview

Nüm Agents is a powerful SDK for creating, managing, and deploying AI agents with a universe-based modular architecture. It enables developers to quickly scaffold agent systems by declaring which functional universes they need, rather than manually coding each component.

## Key Features

- **Universe-Based Architecture**: Activate functional modules by simply declaring which universes your agent needs
- **Dynamic Agent Generation**: Automatically scaffold complete agent code from simple YAML specifications
- **Logical Graph Analysis**: Visualize agent component dependencies and interactions
- **Meta-Orchestration**: Built-in supervision capabilities to validate and improve agent designs
- **Extensible Framework**: Easily add custom universes, modules, and node types
- **Robust Error Handling**: Custom exception system with detailed error information
- **Structured Logging**: Comprehensive logging throughout the SDK for debugging and monitoring
- **Execution Hooks**: Add custom logic before/after node and flow execution
- **Conditional Execution**: Support for conditional branching in agent flows
- **Flow Serialization**: Save and load flow configurations as JSON
- **Retry Mechanism**: Automatic retry with exponential backoff for unstable operations

## Quick Start

```bash
# Install the SDK
pip install num-agents

# Generate an agent from a specification
num-agents generate --spec agent.yaml --catalog univers_catalog.yaml
```

## Agent Specification Example

```yaml
agent:
  name: "ExampleAgent"
  description: "An example agent built with Nüm Agents SDK"
  univers:
    - PocketFlowCore
    - StructureAgentIA
    - KnowledgeLayer
  protocol: N2A
  llm: gpt-4o
  memory: true
  eventbus: true
  scheduler: true
  metrics: true
  tracing: true
```

## Project Structure

The Nüm Agents SDK follows a modular architecture:

```text
num-agents-sdk/
├── num_agents/
│   ├── composer/       # Agent generation and composition
│   ├── graph/          # Logical graph generation
│   ├── orchestrator/   # Meta-orchestration and validation
│   ├── univers/        # Universe catalog management
│   └── utils/          # Utility functions
├── config/             # Configuration files
├── examples/           # Example agent specifications
├── tests/              # Test suite
└── docs/               # Documentation
```

## Advanced Features

### Execution Hooks and Error Handling

```python
from num_agents import Node, Flow, SharedStore, configure_logging
import logging

# Enable logging
configure_logging(level=logging.INFO)

# Create node with retry
class APINode(Node):
    def __init__(self):
        super().__init__(
            name="APICall",
            retry_count=3,  # Retry up to 3 times
            enable_logging=True
        )

    def exec(self, shared: SharedStore):
        response = call_external_api()
        return {"data": response}

# Add hooks
node = APINode()

node.add_before_hook(lambda shared: shared.set("start_time", time.time()))
node.add_after_hook(lambda shared, result:
    shared.set("duration", time.time() - shared.get("start_time")))
node.add_error_hook(lambda shared, error:
    logger.error(f"API call failed: {error}"))

# Execute in flow
flow = Flow(name="DataPipeline", enable_logging=True)
flow.add_node(node)
flow.set_start(node)

results = flow.execute(initial_data={"config": config})
```

### Conditional Execution

```python
from num_agents import ConditionalNode

# Define condition
def check_value(shared: SharedStore) -> bool:
    return shared.get("value", 0) > 100

# Create conditional node
conditional = ConditionalNode(
    condition=check_value,
    true_node=high_value_processor,
    false_node=low_value_processor
)
```

For more examples, see the [Core Improvements Guide](docs/core_improvements.md).

## Development Status

Nüm Agents SDK is currently in alpha (v0.1.0). We're actively developing core features and welcome contributions.

## License

Proprietary - All Rights Reserved

Copyright (c) 2025 Lionel TAGNE. Unauthorized copying, modification, distribution, or use of this software is strictly prohibited.
