# Agent Specification Format

The Nüm Agents SDK uses a YAML-based specification format to define agents. This document provides a comprehensive reference for the `agent.yaml` specification format.

## Basic Structure

An agent specification file (`agent.yaml`) follows this basic structure:

```yaml
agent:
  name: "AgentName"
  description: "Agent description"
  univers:
    - Universe1
    - Universe2
  protocol: "standard"
  llm:
    provider: "openai"
    model: "gpt-4"
  memory:
    type: "simple"
  # Additional configuration options
```

## Required Fields

### `agent.name`

The name of the agent. This should be a concise, descriptive name that identifies the agent's purpose.

Example:
```yaml
agent:
  name: "CustomerSupportAgent"
```

### `agent.description`

A detailed description of the agent's purpose, capabilities, and intended use cases. This helps with documentation and understanding the agent's role.

Example:
```yaml
agent:
  description: "An agent that handles customer support inquiries, routes them to appropriate departments, and provides initial responses."
```

### `agent.univers`

A list of universes that the agent will use. Each universe provides a set of functional modules that the agent can leverage. The universes must be defined in the universe catalog.

Example:
```yaml
agent:
  univers:
    - PocketFlowCore
    - StructureAgentIA
    - KnowledgeLayer
```

### `agent.protocol`

The communication protocol that the agent will use. This defines how the agent interacts with other components in the system.

Example:
```yaml
agent:
  protocol: "standard"
```

Common protocol values:
- `standard`: The default protocol for most agents
- `N2A`: Node-to-Agent protocol for complex agent systems
- `A2A`: Agent-to-Agent protocol for multi-agent systems

## Optional Fields

### `agent.llm`

Configuration for the Language Model that the agent will use. This can be a simple string or a detailed configuration object.

Simple example:
```yaml
agent:
  llm: "gpt-4"
```

Detailed example:
```yaml
agent:
  llm:
    provider: "openai"
    model: "gpt-4"
    temperature: 0.7
    max_tokens: 1000
    system_prompt: "You are a helpful assistant."
```

### `agent.memory`

Configuration for the agent's memory system. This defines how the agent stores and retrieves information.

Simple example:
```yaml
agent:
  memory: true
```

Detailed example:
```yaml
agent:
  memory:
    type: "vector"
    storage: "pinecone"
    embedding_model: "text-embedding-ada-002"
    namespace: "agent_memory"
```

### `agent.eventbus`

Configuration for the agent's event bus system. This defines how the agent handles events and messages.

Example:
```yaml
agent:
  eventbus:
    enabled: true
    type: "redis"
    channel: "agent_events"
```

### `agent.scheduler`

Configuration for the agent's scheduler system. This defines how the agent schedules and executes tasks.

Example:
```yaml
agent:
  scheduler:
    enabled: true
    type: "cron"
    timezone: "UTC"
```

### `agent.metrics`

Configuration for the agent's metrics collection. This defines how the agent collects and reports performance metrics.

Example:
```yaml
agent:
  metrics:
    enabled: true
    provider: "prometheus"
    endpoint: "http://localhost:9090"
```

### `agent.tracing`

Configuration for the agent's tracing system. This defines how the agent traces execution for debugging and monitoring.

Example:
```yaml
agent:
  tracing:
    enabled: true
    provider: "jaeger"
    endpoint: "http://localhost:16686"
```

## Complete Example

Here's a complete example of an agent specification:

```yaml
agent:
  name: "CustomerSupportAgent"
  description: "An agent that handles customer support inquiries, routes them to appropriate departments, and provides initial responses."
  univers:
    - PocketFlowCore
    - StructureAgentIA
    - KnowledgeLayer
    - LearningLoop
  protocol: "N2A"
  llm:
    provider: "openai"
    model: "gpt-4"
    temperature: 0.7
    max_tokens: 1000
    system_prompt: "You are a helpful customer support assistant."
  memory:
    type: "vector"
    storage: "pinecone"
    embedding_model: "text-embedding-ada-002"
    namespace: "customer_support_memory"
  eventbus:
    enabled: true
    type: "redis"
    channel: "customer_support_events"
  scheduler:
    enabled: true
    type: "cron"
    timezone: "UTC"
  metrics:
    enabled: true
    provider: "prometheus"
    endpoint: "http://localhost:9090"
  tracing:
    enabled: true
    provider: "jaeger"
    endpoint: "http://localhost:16686"
```

## Next Steps

After creating your agent specification, you can use the Nüm Agents CLI to generate your agent scaffold:

```bash
num-agents generate --spec agent.yaml --catalog univers_catalog.yaml
```

This will create a directory with all the necessary files to run your agent, including:
- `main.py`: The entry point for your agent
- `flow.py`: The flow definition for your agent
- `nodes/`: Directory containing the node implementations
- `shared_store.py`: The shared store for your agent
- `logical_graph.mmd`: A Mermaid flowchart of your agent's logical structure
- `audit_report.json`: A report on your agent's design consistency and suggestions for improvement
