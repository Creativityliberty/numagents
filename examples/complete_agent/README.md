# Complete Intelligent Agent - End-to-End Example

This is a complete, production-ready example of an intelligent agent built with Nüm Agents SDK.

## Features

- **LLM Integration**: OpenAI GPT for reasoning and responses
- **Knowledge Layer**: Semantic memory with vector embeddings
- **Goal & Task Management**: Automatic planning and execution
- **Monitoring**: Full metrics and distributed tracing
- **Resilience**: Retry, circuit breaker, and timeout handling

## Prerequisites

```bash
# Install Nüm Agents SDK
pip install num-agents

# Install OpenAI (required for LLM and embeddings)
pip install openai

# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

## Quick Start

```bash
# Run the interactive agent
python intelligent_agent.py
```

## Usage

The agent supports natural language conversations:

```
You: Hello! Can you help me plan a project?
Agent: Of course! I'd be happy to help you plan a project. What kind of project are you working on?

You: I want to build a web application
Agent: Great! Building a web application involves several steps. Let me help you organize this...
[Plan created] [Memories: 2]

You: stats
📊 Agent Statistics:
  Goals: 3 total, 2 active
  Tasks: 8 total, 5 ready
  Memories: 15 stored
  Metrics: 47 collected
```

## Architecture

```
User Input
    ↓
Input Parser (parse intent, entities)
    ↓
Memory Recall (retrieve relevant context)
    ↓
LLM Reasoning (GPT generates response)
    ↓
Memory Storage (store interaction)
    ↓
Goal/Task Planning (create actionable plan)
    ↓
Response
```

## Components

### 1. LLMReasoningNode

- Integrates with OpenAI GPT
- Includes retry and circuit breaker
- Tracks metrics (tokens, latency)
- Timeout protection (30s default)

### 2. Knowledge Layer

- Stores conversation history
- Semantic search for relevant memories
- Vector embeddings via OpenAI
- Persistent memory across sessions

### 3. Structure Agent IA

- Parses user intent
- Creates goals and tasks automatically
- Manages dependencies
- Tracks progress

### 4. Monitoring

- Metrics: API calls, tokens, latency
- Distributed tracing across nodes
- Real-time statistics

### 5. Resilience

- Retry failed API calls (3 attempts)
- Circuit breaker for service failures
- Graceful degradation

## Customization

### Change LLM Model

```python
agent = IntelligentAgent(
    model="gpt-4",  # or "gpt-3.5-turbo-16k"
    openai_api_key="your-key"
)
```

### Disable Components

```python
agent = IntelligentAgent(
    enable_monitoring=False,  # Disable metrics
    enable_memory=False       # Disable knowledge layer
)
```

### Add Custom Nodes

```python
class CustomAnalysisNode(ResilientNode):
    def execute(self, shared: SharedStore):
        data = shared.get("data")
        # Your custom logic
        return {"analysis": result}

# Add to flow
agent.flow.nodes.append(CustomAnalysisNode())
```

## API Integration

Use the agent programmatically:

```python
from complete_agent.intelligent_agent import IntelligentAgent

# Initialize
agent = IntelligentAgent(openai_api_key="your-key")

# Chat
response = agent.chat("Tell me about Python")
print(response["response"])
print(f"Tokens used: {response['tokens_used']}")

# Get statistics
stats = agent.get_stats()
print(f"Memories stored: {stats['memory']['total_memories']}")
```

## Production Deployment

### Environment Variables

```bash
export OPENAI_API_KEY="sk-..."
export NUMAGENTS_LOG_LEVEL="INFO"
export NUMAGENTS_MONITORING_ENABLED="true"
```

### Monitoring Integration

The agent exposes metrics compatible with Prometheus:

```python
# Get Prometheus metrics
metrics = agent.monitor.metrics.export_prometheus()
```

### Distributed Tracing

Export traces to Jaeger:

```python
# Get trace in Jaeger format
trace_id = agent.flow.shared.get("trace_id")
jaeger_trace = agent.monitor.tracer.export_jaeger(trace_id)
```

## Troubleshooting

### OpenAI API Errors

```
Error: OpenAI API key not found
```

Set your API key:
```bash
export OPENAI_API_KEY="your-key"
```

### Memory Errors

```
Error: OpenAI library not installed
```

Install dependencies:
```bash
pip install openai
```

### Rate Limiting

The agent includes automatic retry with exponential backoff for rate limits.

## Performance

Typical response times:
- Memory recall: 50-100ms
- LLM inference (GPT-3.5): 1-3s
- LLM inference (GPT-4): 3-10s
- Goal planning: <50ms
- Total: 1-10s depending on LLM model

Memory usage:
- Base agent: ~100MB
- With 1000 memories: ~200MB
- Per conversation: ~5MB

## License

Copyright (c) 2025 Lionel TAGNE. All Rights Reserved.
This software is proprietary and confidential.
