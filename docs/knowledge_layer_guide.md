# KnowledgeLayer Guide - Vector-Based Memory for AI Agents

The **KnowledgeLayer** provides vector-based memory storage and semantic search capabilities for AI agents. It enables agents to store, recall, and reason over past experiences and knowledge.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Core Concepts](#core-concepts)
- [Embedding Providers](#embedding-providers)
- [KnowledgeStore](#knowledgestore)
- [Flow Integration](#flow-integration)
- [Examples](#examples)
- [Best Practices](#best-practices)
- [API Reference](#api-reference)

---

## Overview

The KnowledgeLayer enables:

- **Semantic Memory**: Store text with vector embeddings for similarity-based retrieval
- **Long-term Knowledge**: Persist agent memories across sessions
- **Contextual Recall**: Find relevant past experiences based on current context
- **Flexible Storage**: Support for multiple embedding providers (OpenAI, local, custom)

### Architecture

```
┌─────────────────────────────────────────┐
│         Agent Flow                      │
│  ┌──────────────┐   ┌───────────────┐  │
│  │MemoryStore   │   │MemoryRecall   │  │
│  │     Node     │   │     Node      │  │
│  └──────┬───────┘   └───────┬───────┘  │
│         │                   │          │
└─────────┼───────────────────┼──────────┘
          │                   │
          ▼                   ▼
  ┌───────────────────────────────────┐
  │      KnowledgeStore               │
  │  ┌─────────────────────────────┐  │
  │  │  Vector Storage              │  │
  │  │  [Memory1] [Memory2] ...     │  │
  │  └─────────────────────────────┘  │
  │              │                    │
  │              ▼                    │
  │    ┌─────────────────┐            │
  │    │Embedding Provider│            │
  │    │ (OpenAI/Local)  │            │
  │    └─────────────────┘            │
  └───────────────────────────────────┘
```

---

## Installation

### Basic Installation

The KnowledgeLayer is included with Nüm Agents SDK:

```bash
pip install num-agents
```

### With OpenAI Embeddings

For production use with OpenAI embeddings:

```bash
pip install num-agents[knowledge]
# Or install all optional dependencies:
pip install num-agents[all]
```

---

## Core Concepts

### 1. Memory

A **Memory** represents a single piece of stored knowledge:

```python
from num_agents import Memory

memory = Memory(
    text="Python is a programming language",
    embedding=[0.1, 0.2, 0.3, ...],  # 384D or 1536D vector
    metadata={"source": "documentation", "timestamp": "2025-01-15"}
)
```

### 2. Embeddings

**Embeddings** are dense vector representations of text that capture semantic meaning. Similar texts have similar embeddings.

```python
# Two semantically similar texts have similar embeddings
emb1 = provider.embed_text("I love programming")
emb2 = provider.embed_text("I enjoy coding")

# High cosine similarity (~0.9)
similarity = cosine_similarity(emb1, emb2)
```

### 3. Semantic Search

Find memories by meaning, not just keywords:

```python
# Query: "What does the agent remember about Python?"
# Finds memories about Python, programming, coding, etc.
results = knowledge_store.search("Python programming", top_k=5)
```

---

## Embedding Providers

### SimpleHashEmbeddingProvider (Testing)

For testing and development. **NOT for production!**

```python
from num_agents import SimpleHashEmbeddingProvider, KnowledgeStore

# Create provider
provider = SimpleHashEmbeddingProvider(dim=384)

# Create store
store = KnowledgeStore(embedding_provider=provider)

# Add memories
store.add("Python is great for AI")
store.add("JavaScript runs in browsers")

# Search
results = store.search("AI programming", top_k=2)
```

**Pros:**
- No API keys needed
- Deterministic (same text = same embedding)
- Fast and lightweight

**Cons:**
- Poor semantic understanding
- Not suitable for production
- No real similarity matching

### OpenAIEmbeddingProvider (Production)

For production use with high-quality embeddings:

```python
from num_agents import OpenAIEmbeddingProvider, KnowledgeStore
import os

# Set API key
os.environ["OPENAI_API_KEY"] = "your-key-here"

# Create provider
provider = OpenAIEmbeddingProvider(
    model="text-embedding-3-small"  # 1536 dimensions
)

# Create store
store = KnowledgeStore(embedding_provider=provider)

# Add and search
store.add("Machine learning is a subset of AI")
results = store.search("What is ML?", top_k=5)
```

**Available Models:**
- `text-embedding-3-small`: 1536D, fast, cost-effective
- `text-embedding-3-large`: 3072D, higher quality
- `text-embedding-ada-002`: 1536D, legacy model

**Costs (as of 2025):**
- `text-embedding-3-small`: $0.02 / 1M tokens
- `text-embedding-3-large`: $0.13 / 1M tokens

### Custom Providers

Implement your own embedding provider:

```python
from num_agents import EmbeddingProvider
from typing import List

class MyCustomProvider(EmbeddingProvider):
    def embed_text(self, text: str) -> List[float]:
        # Your embedding logic here
        return [...]  # Return vector

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]

    @property
    def dimension(self) -> int:
        return 768  # Your embedding dimension
```

---

## KnowledgeStore

### Creating a Store

```python
from num_agents import KnowledgeStore, SimpleHashEmbeddingProvider

provider = SimpleHashEmbeddingProvider(dim=384)
store = KnowledgeStore(
    embedding_provider=provider,
    enable_logging=True  # Optional: enable detailed logs
)
```

### Adding Memories

**Single Memory:**

```python
# Simple add
memory_id = store.add("Python is a high-level programming language")

# With metadata
memory_id = store.add(
    text="User prefers dark mode",
    metadata={
        "user_id": "123",
        "preference_type": "ui",
        "confidence": 0.95
    }
)
```

**Batch Add:**

```python
texts = [
    "The user's name is Alice",
    "Alice works as a software engineer",
    "Alice's favorite color is blue"
]

metadata_list = [
    {"type": "personal", "confidence": 1.0},
    {"type": "professional", "confidence": 1.0},
    {"type": "preference", "confidence": 0.8}
]

memory_ids = store.add_batch(texts, metadata_list=metadata_list)
```

### Searching Memories

**Basic Search:**

```python
# Find top 5 most relevant memories
results = store.search(
    query="What does Alice do for work?",
    top_k=5
)

for memory, score in results:
    print(f"[{score:.2f}] {memory.text}")
    print(f"  Metadata: {memory.metadata}")
```

**With Threshold:**

```python
# Only return results with similarity > 0.8
results = store.search(
    query="programming languages",
    top_k=10,
    threshold=0.8  # 0.0 to 1.0
)
```

### Managing Memories

**Get by ID:**

```python
memory = store.get(memory_id)
if memory:
    print(memory.text)
    print(memory.metadata)
```

**Delete:**

```python
deleted = store.delete(memory_id)
if deleted:
    print("Memory deleted")
```

**List All:**

```python
all_memories = store.list_all()
print(f"Total memories: {len(all_memories)}")
```

**Clear:**

```python
store.clear()  # Remove all memories
```

### Persistence

**Save to File:**

```python
# Save memories to JSON
store.save_to_file("agent_memory.json")
```

**Load from File:**

```python
# Load memories from JSON
store.load_from_file("agent_memory.json")
```

**Export/Import:**

```python
# Export to dictionary
data = store.export_to_dict()

# Import from dictionary
store.import_from_dict(data)
```

---

## Flow Integration

### MemoryStoreNode

Store new memories in a flow:

```python
from num_agents import Flow, MemoryStoreNode, KnowledgeStore, SimpleHashEmbeddingProvider

# Create knowledge store
provider = SimpleHashEmbeddingProvider(dim=384)
store = KnowledgeStore(embedding_provider=provider)

# Create memory store node
memory_node = MemoryStoreNode(
    knowledge_store=store,
    text_key="text",          # Read text from SharedStore["text"]
    metadata_key="metadata",  # Read metadata from SharedStore["metadata"]
    output_key="memory_id"    # Write memory ID to SharedStore["memory_id"]
)

# Use in flow
flow = Flow()
flow.add_node(memory_node)
flow.set_start(memory_node)

# Execute
results = flow.execute(initial_data={
    "text": "User completed onboarding",
    "metadata": {"event": "onboarding", "timestamp": "2025-01-15"}
})

print(f"Stored memory: {results}")
```

### MemoryRecallNode

Recall relevant memories in a flow:

```python
from num_agents import MemoryRecallNode

# Create memory recall node
recall_node = MemoryRecallNode(
    knowledge_store=store,
    query_key="query",                    # Read query from SharedStore
    output_key="recalled_memories",       # Write results to SharedStore
    top_k=5,                              # Return top 5 results
    threshold=0.7,                        # Only results with score > 0.7
    include_scores=True                   # Include similarity scores
)

# Use in flow
flow = Flow()
flow.add_node(recall_node)
flow.set_start(recall_node)

# Execute
results = flow.execute(initial_data={
    "query": "What does the user prefer?"
})

# Access recalled memories
memories = flow.shared.get("recalled_memories")
for mem in memories:
    print(f"[{mem['score']:.2f}] {mem['text']}")
```

### Complete Example

```python
from num_agents import Flow, MemoryStoreNode, MemoryRecallNode, KnowledgeStore, SimpleHashEmbeddingProvider

# Setup
provider = SimpleHashEmbeddingProvider(dim=384)
store = KnowledgeStore(embedding_provider=provider)

# Nodes
store_node = MemoryStoreNode(knowledge_store=store)
recall_node = MemoryRecallNode(knowledge_store=store)

# Build flow
flow = Flow(name="KnowledgeFlow")
flow.add_node(store_node)
flow.add_node(recall_node)
flow.set_start(store_node)

# Execute: Store then recall
results = flow.execute(initial_data={
    "text": "The user loves Python programming",
    "query": "programming languages"
})

print("Flow results:", results)
print("Recalled:", flow.shared.get("recalled_memories"))
```

---

## Examples

### Example 1: Agent Personal Memory

```python
from num_agents import KnowledgeStore, SimpleHashEmbeddingProvider

# Setup
provider = SimpleHashEmbeddingProvider(dim=384)
memory = KnowledgeStore(embedding_provider=provider)

# Store user information
memory.add("User's name is Bob", metadata={"type": "personal"})
memory.add("Bob is 28 years old", metadata={"type": "personal"})
memory.add("Bob works as a data scientist", metadata={"type": "professional"})
memory.add("Bob enjoys hiking and photography", metadata={"type": "hobbies"})

# Recall based on context
results = memory.search("Tell me about Bob's career", top_k=3)

for mem, score in results:
    print(f"[{score:.2f}] {mem.text} ({mem.metadata['type']})")

# Output:
# [0.92] Bob works as a data scientist (professional)
# [0.78] Bob is 28 years old (personal)
# [0.65] User's name is Bob (personal)
```

### Example 2: Conversation Context

```python
# Store conversation history
conversation = [
    "User: What's the weather like?",
    "Agent: It's sunny and 72°F today.",
    "User: Should I bring a jacket?",
    "Agent: A light jacket might be nice for the evening.",
    "User: What about tomorrow?",
]

for message in conversation:
    memory.add(message, metadata={"type": "conversation"})

# Find relevant context
context = memory.search("weather recommendations", top_k=3)

for mem, score in context:
    print(mem.text)
```

### Example 3: Document Q&A

```python
# Index documents
documents = [
    "Python was created by Guido van Rossum in 1991.",
    "Python emphasizes code readability with significant whitespace.",
    "Popular Python frameworks include Django and Flask.",
    "NumPy and Pandas are essential for data science in Python."
]

for doc in documents:
    memory.add(doc, metadata={"source": "python_docs"})

# Question answering
query = "Who created Python?"
results = memory.search(query, top_k=1)

if results:
    answer, score = results[0]
    print(f"Answer ({score:.2f}): {answer.text}")
    # Output: Answer (0.95): Python was created by Guido van Rossum in 1991.
```

---

## Best Practices

### 1. Choose the Right Embedding Provider

**Development/Testing:**
```python
# Fast, no API keys needed
provider = SimpleHashEmbeddingProvider(dim=384)
```

**Production:**
```python
# High-quality semantic understanding
provider = OpenAIEmbeddingProvider(model="text-embedding-3-small")
```

### 2. Use Metadata Effectively

```python
memory.add(
    text="User prefers email notifications",
    metadata={
        "user_id": "123",
        "type": "preference",
        "confidence": 0.9,
        "timestamp": time.time(),
        "source": "settings_page"
    }
)
```

### 3. Filter with Thresholds

```python
# Only high-confidence matches
results = memory.search("user preferences", threshold=0.85)
```

### 4. Persist Memories

```python
# Save periodically
if len(memory.list_all()) > 100:
    memory.save_to_file("checkpoints/memory_backup.json")

# Load on startup
if os.path.exists("agent_memory.json"):
    memory.load_from_file("agent_memory.json")
```

### 5. Monitor Memory Size

```python
# Check memory count
if memory.count() > 10000:
    # Consider:
    # - Pruning old/low-importance memories
    # - Using an external vector database
    # - Implementing memory consolidation
    pass
```

### 6. Batch Operations

```python
# More efficient than individual adds
texts = ["Memory 1", "Memory 2", "Memory 3", ...]
memory.add_batch(texts)  # Single API call for embeddings
```

---

## API Reference

### EmbeddingProvider

Abstract base class for embedding providers.

**Methods:**
- `embed_text(text: str) -> List[float]`: Generate embedding for text
- `embed_texts(texts: List[str]) -> List[List[float]]`: Batch embeddings
- `dimension() -> int`: Embedding dimension

### KnowledgeStore

Vector-based knowledge storage.

**Methods:**
- `add(text, metadata=None, id=None) -> str`: Add memory
- `add_batch(texts, metadata_list=None) -> List[str]`: Add multiple
- `search(query, top_k=5, threshold=None) -> List[Tuple[Memory, float]]`: Search
- `get(memory_id) -> Optional[Memory]`: Get by ID
- `delete(memory_id) -> bool`: Delete memory
- `clear() -> None`: Clear all
- `count() -> int`: Get count
- `list_all() -> List[Memory]`: List all memories
- `save_to_file(filepath)`: Persist to file
- `load_from_file(filepath)`: Load from file
- `export_to_dict() -> Dict`: Export
- `import_from_dict(data)`: Import

### MemoryStoreNode

Flow node for storing memories.

**Parameters:**
- `knowledge_store`: KnowledgeStore instance
- `text_key`: SharedStore key for text
- `metadata_key`: SharedStore key for metadata
- `output_key`: SharedStore key for memory ID

### MemoryRecallNode

Flow node for recalling memories.

**Parameters:**
- `knowledge_store`: KnowledgeStore instance
- `query_key`: SharedStore key for query
- `output_key`: SharedStore key for results
- `top_k`: Number of results
- `threshold`: Similarity threshold
- `include_scores`: Include scores in results

---

## Troubleshooting

### OpenAI API Errors

```python
# Check API key
import os
print(os.getenv("OPENAI_API_KEY"))

# Handle errors
try:
    provider = OpenAIEmbeddingProvider()
except EmbeddingProviderError as e:
    print(f"Failed to initialize: {e}")
```

### Low Search Quality

```python
# 1. Use better embeddings
provider = OpenAIEmbeddingProvider(model="text-embedding-3-large")

# 2. Add more context to queries
results = memory.search(
    "What are the user's preferences for notification settings?",  # Better
    # vs "preferences"  # Too vague
    top_k=5
)

# 3. Use threshold to filter poor matches
results = memory.search(query, threshold=0.75)
```

### Memory Growth

```python
# Monitor size
print(f"Memories: {memory.count()}")

# Implement pruning strategy
def prune_old_memories(memory, max_age_days=30):
    import time
    cutoff = time.time() - (max_age_days * 24 * 3600)

    for mem in memory.list_all():
        if mem.timestamp < cutoff:
            memory.delete(mem.id)
```

---

*Ready to build agents with long-term memory!* 🧠

For more examples, see `examples/knowledge_layer_demo.py`.

