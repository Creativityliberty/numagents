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

```
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

## Development Status

Nüm Agents SDK is currently in alpha (v0.1.0). We're actively developing core features and welcome contributions.

## License

MIT License
