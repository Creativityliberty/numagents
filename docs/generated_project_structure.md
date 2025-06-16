# Generated Agent Project Structure

When you use the Nüm Agents SDK to generate an agent, it creates a complete project structure with all the necessary files to run your agent. This document explains the structure and purpose of each generated file.

## Overview

The generated agent project follows this structure:

```
agent_name/
├── main.py                # Entry point for running the agent
├── flow.py                # Definition of the agent's flow
├── shared_store.py        # Shared store for the agent
├── nodes/                 # Directory containing node implementations
│   ├── node1.py           # Implementation of Node1
│   ├── node2.py           # Implementation of Node2
│   └── ...
├── agent.yaml             # The original agent specification
├── logical_graph.mmd      # Mermaid flowchart of the agent's structure
├── logical_graph.md       # Markdown representation of the logical graph
├── audit_report.json      # Report on the agent's design consistency
└── README.md              # Basic documentation for the agent
```

## File Descriptions

### `main.py`

The entry point for running the agent. This file initializes the flow and shared store, then executes the flow.

Example:

```python
"""
Main entry point for the ExampleAgent agent.

This module provides the main function for running the agent.
"""

from flow import create_flow
from shared_store import create_shared_store


def main() -> None:
    """Run the agent."""
    # Create the flow
    flow = create_flow()
    
    # Create the shared store
    shared = create_shared_store()
    flow.shared = shared
    
    # Execute the flow
    results = flow.execute()
    
    # Print the results
    print("Flow execution completed.")
    print("Results:")
    for node_name, node_results in results.items():
        print(f"  {node_name}: {node_results}")


if __name__ == "__main__":
    main()
```

### `flow.py`

Defines the agent's flow by instantiating and sequencing the nodes. This file contains a `create_flow()` function that returns a `Flow` object with all the nodes from the specified universes.

Example:

```python
"""
Flow definition for the agent.

This module provides the create_flow function, which creates the agent's flow.
"""

from num_agents.core import Flow

from nodes.manager_goal_node import ManagerGoalNode
from nodes.tool_adapter_node import ToolAdapterNode
from nodes.memory_recall_node import MemoryRecallNode
from nodes.memory_store_node import MemoryStoreNode


def create_flow() -> Flow:
    """
    Create the agent's flow.
    
    Returns:
        The agent's flow
    """
    return Flow([
        ManagerGoalNode(),
        ToolAdapterNode(),
        MemoryRecallNode(),
        MemoryStoreNode()
    ])
```

### `shared_store.py`

Provides a factory function for creating the agent's shared store. The shared store is used to share data between nodes in the flow.

Example:

```python
"""
Shared store for the agent.

This module provides a factory function for creating the agent's shared store.
"""

from num_agents.core import SharedStore


def create_shared_store() -> SharedStore:
    """
    Create the agent's shared store.
    
    Returns:
        The agent's shared store
    """
    shared = SharedStore()
    
    # Initialize any shared data here
    
    return shared
```

### `nodes/` Directory

Contains Python files for each node in the agent's flow. Each file defines a node class that extends the base `Node` class from the Nüm Agents SDK.

Example node file (`nodes/manager_goal_node.py`):

```python
"""
ManagerGoalNode implementation.

This module provides the ManagerGoalNode class, which is a node in the agent flow.
"""

from typing import Any, Dict

from num_agents.core import Node, SharedStore


class ManagerGoalNode(Node):
    """
    ManagerGoalNode implementation.
    
    This node is responsible for managing high-level agent goals.
    """
    
    def __init__(self) -> None:
        """Initialize the node."""
        super().__init__("ManagerGoalNode")
    
    def exec(self, shared: SharedStore) -> Dict[str, Any]:
        """
        Execute the node's processing logic.
        
        Args:
            shared: The shared store for accessing and storing data
            
        Returns:
            A dictionary containing the results of the node's execution
        """
        # TODO: Implement the node's logic
        return {"status": "success"}
```

### `agent.yaml`

A copy of the original agent specification used to generate the agent. This file is included for reference and documentation purposes.

### `logical_graph.mmd`

A Mermaid flowchart representation of the agent's logical structure. This file can be rendered using Mermaid-compatible tools to visualize the agent's flow.

Example:

```
flowchart TD

ManagerGoalNode["ManagerGoalNode\n(Manages high-level agent goals)"]
ToolAdapterNode["ToolAdapterNode\n(Adapts tools for agent use)"]
MemoryRecallNode["MemoryRecallNode\n(Recalls information from memory)"]
MemoryStoreNode["MemoryStoreNode\n(Stores information in memory)"]

ManagerGoalNode --> ToolAdapterNode
ToolAdapterNode --> MemoryRecallNode
MemoryRecallNode --> MemoryStoreNode
```

### `logical_graph.md`

A Markdown representation of the agent's logical structure. This file includes the Mermaid flowchart as well as additional information about the nodes and transitions.

### `audit_report.json`

A JSON report on the agent's design consistency and suggestions for improvement. This report is generated by the Meta-Orchestrator component of the Nüm Agents SDK.

Example:

```json
{
  "validation": {
    "agent_name": "ExampleAgent",
    "status": "valid",
    "issues": [],
    "suggestions": [
      "Consider adding 'EscalationManager' when using 'ActiveLearningNode' for better supervision."
    ],
    "completeness": "100.0%",
    "declared_modules": [
      "ManagerGoalNode",
      "ToolAdapterNode",
      "MemoryRecallNode",
      "MemoryStoreNode"
    ],
    "graph_nodes": [
      "ManagerGoalNode",
      "ToolAdapterNode",
      "MemoryRecallNode",
      "MemoryStoreNode"
    ]
  }
}
```

### `README.md`

A basic README file for the agent, including its name, description, structure, and usage instructions.

## Customizing the Generated Project

After generating the agent project, you can customize it to fit your specific needs:

1. **Implement Node Logic**: Fill in the `exec()` methods of each node with your custom logic
2. **Add Custom Nodes**: Create new node files in the `nodes/` directory
3. **Modify the Flow**: Update the `create_flow()` function in `flow.py` to change the sequence of nodes
4. **Initialize Shared Data**: Add initialization code to the `create_shared_store()` function in `shared_store.py`

## Running the Generated Agent

To run the generated agent, simply execute the `main.py` file:

```bash
cd agent_name
python main.py
```

## Regenerating the Logical Graph

If you make changes to the agent's flow, you can regenerate the logical graph using the Nüm Agents CLI:

```bash
num-agents graph --agent-dir agent_name
```

## Auditing the Agent

You can also re-audit the agent to check for consistency and get suggestions for improvement:

```bash
num-agents audit --agent-dir agent_name
```
