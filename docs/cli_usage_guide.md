# CLI Usage Guide

The Nüm Agents SDK provides a powerful command-line interface (CLI) for generating, analyzing, and auditing agent projects. This guide explains how to use the CLI effectively.

## Installation

Before using the CLI, make sure you have installed the Nüm Agents SDK:

```bash
pip install num-agents
```

## Basic Commands

The Nüm Agents CLI provides several commands:

- `generate`: Generate an agent scaffold from a specification
- `audit`: Analyze an agent and generate an audit report
- `graph`: Generate a logical graph for an agent

## Generating an Agent

The `generate` command creates a complete agent scaffold based on an agent specification.

### Basic Usage

```bash
num-agents generate agent.yaml
```

This will generate an agent scaffold in a directory named after the agent, using the default universe catalog.

### Specifying a Universe Catalog

You can specify a custom universe catalog:

```bash
num-agents generate agent.yaml --univers-catalog my_univers_catalog.yaml
```

### Specifying an Output Directory

You can specify a custom output directory:

```bash
num-agents generate agent.yaml --output-dir my_agent_dir
```

### Skipping Graph or Audit

You can skip generating the logical graph or audit report:

```bash
num-agents generate agent.yaml --skip-graph
num-agents generate agent.yaml --skip-audit
```

### Full Example

```bash
num-agents generate examples/agent.yaml --univers-catalog config/univers_catalog.yaml --output-dir my_agent_dir
```

## Auditing an Agent

The `audit` command analyzes an agent and generates an audit report.

### Basic Usage

```bash
num-agents audit my_agent_dir
```

This will analyze the agent in the specified directory and generate an audit report.

### Specifying an Agent Specification

You can specify a custom agent specification:

```bash
num-agents audit my_agent_dir --agent-spec my_agent.yaml
```

### Specifying a Universe Catalog

You can specify a custom universe catalog:

```bash
num-agents audit my_agent_dir --univers-catalog my_univers_catalog.yaml
```

### Specifying an Output Path

You can specify a custom output path for the audit report:

```bash
num-agents audit my_agent_dir --output-path my_audit_report.json
```

### Full Example

```bash
num-agents audit my_agent_dir --agent-spec my_agent.yaml --univers-catalog my_univers_catalog.yaml --output-path my_audit_report.json
```

## Generating a Logical Graph

The `graph` command generates a logical graph for an agent.

### Basic Usage

```bash
num-agents graph my_agent_dir
```

This will generate a logical graph for the agent in the specified directory.

### Specifying Output Paths

You can specify custom output paths for the Mermaid flowchart and Markdown representation:

```bash
num-agents graph my_agent_dir --output-mermaid my_graph.mmd --output-markdown my_graph.md
```

### Full Example

```bash
num-agents graph my_agent_dir --output-mermaid my_graph.mmd --output-markdown my_graph.md
```

## Step-by-Step: Creating and Running an Agent

Here's a complete walkthrough of creating and running an agent with the Nüm Agents SDK:

### Step 1: Create an Agent Specification

Create a file named `agent.yaml` with your agent specification:

```yaml
agent:
  name: "MyFirstAgent"
  description: "A simple agent built with Nüm Agents SDK"
  univers:
    - PocketFlowCore
    - StructureAgentIA
    - KnowledgeLayer
  protocol: "standard"
  llm:
    provider: "openai"
    model: "gpt-4"
  memory:
    type: "simple"
```

### Step 2: Generate the Agent Scaffold

Use the `generate` command to create the agent scaffold:

```bash
num-agents generate agent.yaml
```

This will create a directory named `my_first_agent` with all the necessary files.

### Step 3: Implement Node Logic

Open the generated node files in the `nodes/` directory and implement the `exec()` methods with your custom logic:

```python
def exec(self, shared: SharedStore) -> Dict[str, Any]:
    """
    Execute the node's processing logic.
    
    Args:
        shared: The shared store for accessing and storing data
        
    Returns:
        A dictionary containing the results of the node's execution
    """
    # Get user input from the shared store
    user_input = shared.get("user_input", "")
    
    # Process the input
    response = f"You said: {user_input}"
    
    # Store the response in the shared store
    shared.set("response", response)
    
    return {"status": "success", "response": response}
```

### Step 4: Initialize Shared Data

Open the `shared_store.py` file and initialize any shared data:

```python
def create_shared_store() -> SharedStore:
    """
    Create the agent's shared store.
    
    Returns:
        The agent's shared store
    """
    shared = SharedStore()
    
    # Initialize shared data
    shared.set("user_input", "Hello, agent!")
    shared.set("response", "")
    
    return shared
```

### Step 5: Run the Agent

Run the agent by executing the `main.py` file:

```bash
cd my_first_agent
python main.py
```

### Step 6: Analyze the Results

After running the agent, you can analyze the results:

- Check the console output for the results of each node's execution
- Review the `logical_graph.mmd` file to visualize the agent's structure
- Review the `audit_report.json` file for design consistency and suggestions

### Step 7: Iterate and Improve

Based on the results and audit report, you can iterate and improve your agent:

1. Modify the node implementations
2. Update the flow sequence
3. Add or remove nodes
4. Regenerate the logical graph and audit report

## Advanced Usage

### Using Environment Variables

You can use environment variables to configure the Nüm Agents CLI:

```bash
export NUM_AGENTS_UNIVERS_CATALOG=/path/to/univers_catalog.yaml
num-agents generate agent.yaml
```

### Creating a Configuration File

You can create a configuration file to store common settings:

```yaml
# num_agents_config.yaml
univers_catalog: /path/to/univers_catalog.yaml
output_dir: /path/to/output
```

And then use it with the CLI:

```bash
num-agents --config num_agents_config.yaml generate agent.yaml
```

## Troubleshooting

### Common Issues

1. **Missing Universe**: If you get an error about a missing universe, make sure it's defined in your universe catalog.
2. **Missing Module**: If you get an error about a missing module, make sure it's defined in the appropriate universe.
3. **Import Error**: If you get an import error when running your agent, make sure all dependencies are installed.

### Getting Help

You can get help for any command by adding the `--help` flag:

```bash
num-agents --help
num-agents generate --help
num-agents audit --help
num-agents graph --help
```
