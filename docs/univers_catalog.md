# Universe Catalog Format

The Nüm Agents SDK uses a universe-based architecture to organize functional modules. This document provides a comprehensive reference for the `univers_catalog.yaml` format.

## Basic Structure

A universe catalog file (`univers_catalog.yaml`) follows this basic structure:

```yaml
univers_catalog:
  Universe1:
    description: "Description of Universe1"
    modules:
      - Module1
      - Module2
  Universe2:
    description: "Description of Universe2"
    modules:
      - Module3
      - Module4
```

## Universe Definition

Each universe in the catalog is defined with a unique name and contains a description and a list of modules.

### Universe Name

The name of the universe should be descriptive and indicate the functional domain it covers. By convention, universe names use PascalCase (e.g., `PocketFlowCore`, `KnowledgeLayer`).

### Universe Description

A concise description of the universe's purpose and the types of functionality its modules provide.

### Universe Modules

A list of module names that belong to the universe. These modules are typically implemented as node classes in the agent's flow. By convention, module names use PascalCase and often end with "Node" (e.g., `ManagerGoalNode`, `MemoryRecallNode`).

## Example Universe Catalog

Here's an example of a universe catalog with several universes:

```yaml
univers_catalog:
  PocketFlowCore:
    description: "Core components for building agent flows"
    modules:
      - Flow
      - Node
      - Transition
      - SharedStore
  StructureAgentIA:
    description: "Structural components for agent intelligence"
    modules:
      - ManagerGoalNode
      - ManagerTaskNode
      - MCPNode
      - InputParserNode
  KnowledgeLayer:
    description: "Knowledge management components"
    modules:
      - KnowledgeStoreNode
      - MemoryRecallNode
      - MemoryStoreNode
  LearningLoop:
    description: "Components for continuous learning"
    modules:
      - ActiveLearningNode
      - ModelFineTunerNode
  QualityAssurance:
    description: "Quality assurance components"
    modules:
      - EvaluatorNode
      - FeedbackCollectorNode
  FallbackEscalade:
    description: "Error handling and escalation components"
    modules:
      - FallbackNodeAdvanced
      - EscalationManagerNode
  Monitoring:
    description: "Monitoring and observability components"
    modules:
      - MetricsCollectorNode
      - TracingNode
      - LoggingNode
```

## Common Universes and Their Modules

### PocketFlowCore

The core universe that provides the fundamental components for building agent flows.

- `Flow`: The main flow container that orchestrates the execution of nodes
- `Node`: The base class for all nodes in the flow
- `Transition`: Handles transitions between nodes
- `SharedStore`: Provides shared storage for data between nodes

### StructureAgentIA

Provides structural components for agent intelligence.

- `ManagerGoalNode`: Manages high-level agent goals
- `ManagerTaskNode`: Breaks down goals into manageable tasks
- `MCPNode`: Handles Model Context Protocol interactions
- `InputParserNode`: Parses and validates user inputs

### KnowledgeLayer

Provides components for knowledge management.

- `KnowledgeStoreNode`: Stores and retrieves knowledge
- `MemoryRecallNode`: Recalls information from memory
- `MemoryStoreNode`: Stores information in memory

### LearningLoop

Provides components for continuous learning.

- `ActiveLearningNode`: Identifies areas for improvement
- `ModelFineTunerNode`: Fine-tunes models based on feedback

### QualityAssurance

Provides components for quality assurance.

- `EvaluatorNode`: Evaluates agent outputs
- `FeedbackCollectorNode`: Collects feedback from users

### FallbackEscalade

Provides components for error handling and escalation.

- `FallbackNodeAdvanced`: Handles errors and provides fallback responses
- `EscalationManagerNode`: Escalates issues to human operators when necessary

### Monitoring

Provides components for monitoring and observability.

- `MetricsCollectorNode`: Collects performance metrics
- `TracingNode`: Provides distributed tracing
- `LoggingNode`: Handles logging

## Extending the Universe Catalog

You can extend the universe catalog by adding your own universes and modules. This allows you to create custom functionality for your agents.

### Creating a Custom Universe

To create a custom universe, add a new entry to the universe catalog:

```yaml
univers_catalog:
  # Existing universes...
  
  MyCustomUniverse:
    description: "My custom functionality"
    modules:
      - MyCustomNode1
      - MyCustomNode2
```

### Implementing Custom Modules

After defining your custom universe and modules in the catalog, you'll need to implement the corresponding node classes. These should be placed in your project's `nodes/` directory.

For example, for `MyCustomNode1`, you would create a file `nodes/my_custom_node1.py`:

```python
from num_agents.core import Node, SharedStore
from typing import Any, Dict

class MyCustomNode1(Node):
    """
    My custom node implementation.
    
    This node provides custom functionality for my agent.
    """
    
    def __init__(self) -> None:
        """Initialize the node."""
        super().__init__("MyCustomNode1")
    
    def exec(self, shared: SharedStore) -> Dict[str, Any]:
        """
        Execute the node's processing logic.
        
        Args:
            shared: The shared store for accessing and storing data
            
        Returns:
            A dictionary containing the results of the node's execution
        """
        # Custom implementation here
        return {"status": "success"}
```

## Using the Universe Catalog

When generating an agent with the Nüm Agents CLI, you can specify the path to your universe catalog:

```bash
num-agents generate --spec agent.yaml --catalog my_univers_catalog.yaml
```

If you don't specify a catalog, the default catalog will be used.

## Best Practices

1. **Modular Design**: Keep universes focused on specific functional domains
2. **Clear Naming**: Use descriptive names for universes and modules
3. **Documentation**: Include clear descriptions for each universe
4. **Consistency**: Follow naming conventions (PascalCase for universes and modules)
5. **Reusability**: Design modules to be reusable across different agents
