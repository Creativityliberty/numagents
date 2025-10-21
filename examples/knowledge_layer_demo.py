"""
KnowledgeLayer Demo - Intelligent Agent with Long-term Memory

This example demonstrates how to build an agent with semantic memory
using the KnowledgeLayer module.

Copyright (c) 2025 Lionel TAGNE. All Rights Reserved.
This software is proprietary and confidential.
"""

from num_agents import (
    Flow,
    Node,
    SharedStore,
    KnowledgeStore,
    MemoryStoreNode,
    MemoryRecallNode,
    SimpleHashEmbeddingProvider,
    configure_logging,
)
from typing import Any, Dict
import logging

# Enable logging to see what's happening
configure_logging(level=logging.INFO)


def demo_basic_memory():
    """Demo 1: Basic memory storage and retrieval."""
    print("\n" + "=" * 70)
    print("DEMO 1: Basic Memory Storage and Retrieval")
    print("=" * 70)

    # Create knowledge store with simple hash embeddings
    provider = SimpleHashEmbeddingProvider(dim=384)
    memory = KnowledgeStore(embedding_provider=provider, enable_logging=True)

    # Store some facts
    print("\n📝 Storing facts about the user...")
    memory.add("The user's name is Alice", metadata={"type": "personal"})
    memory.add("Alice is a software engineer", metadata={"type": "professional"})
    memory.add("Alice loves Python programming", metadata={"type": "interest"})
    memory.add("Alice's favorite color is blue", metadata={"type": "preference"})
    memory.add("Alice works at TechCorp", metadata={"type": "professional"})

    print(f"✅ Stored {memory.count()} memories")

    # Search for relevant memories
    print("\n🔍 Searching: 'What does Alice do for work?'")
    results = memory.search("What does Alice do for work?", top_k=3)

    for i, (mem, score) in enumerate(results, 1):
        print(f"\n  Result {i} (similarity: {score:.2f})")
        print(f"  📄 {mem.text}")
        print(f"  🏷️  Type: {mem.metadata.get('type', 'N/A')}")


def demo_conversation_memory():
    """Demo 2: Conversation context memory."""
    print("\n" + "=" * 70)
    print("DEMO 2: Conversation Context Memory")
    print("=" * 70)

    provider = SimpleHashEmbeddingProvider(dim=384)
    memory = KnowledgeStore(embedding_provider=provider)

    # Simulate a conversation
    conversation = [
        ("user", "What's the weather like today?"),
        ("agent", "It's sunny and 75°F today."),
        ("user", "Should I bring an umbrella?"),
        ("agent", "No need for an umbrella today, it's clear!"),
        ("user", "What about a jacket?"),
        ("agent", "A light jacket would be good for the evening."),
    ]

    print("\n💬 Storing conversation history...")
    for speaker, message in conversation:
        memory.add(
            f"{speaker}: {message}", metadata={"speaker": speaker, "type": "conversation"}
        )

    # Find relevant context
    print("\n🔍 User asks: 'Did we talk about the weather?'")
    results = memory.search("weather", top_k=3)

    print("\n📋 Relevant conversation context:")
    for mem, score in results:
        print(f"  [{score:.2f}] {mem.text}")


def demo_flow_integration():
    """Demo 3: Integration with Flow using nodes."""
    print("\n" + "=" * 70)
    print("DEMO 3: KnowledgeLayer in Agent Flow")
    print("=" * 70)

    # Setup knowledge store
    provider = SimpleHashEmbeddingProvider(dim=384)
    knowledge_store = KnowledgeStore(embedding_provider=provider)

    # Pre-populate with some knowledge
    print("\n📚 Pre-populating knowledge base...")
    knowledge_store.add("Python is a high-level programming language")
    knowledge_store.add("JavaScript is used for web development")
    knowledge_store.add("SQL is for database queries")
    knowledge_store.add("Git is a version control system")

    # Create a custom processing node
    class QuestionNode(Node):
        """Node that prepares a question for memory search."""

        def exec(self, shared: SharedStore) -> Dict[str, Any]:
            question = shared.get_required("question")
            # Store the question as query for recall node
            shared.set("query", question)
            return {"question_processed": True, "question": question}

    # Create a response node
    class ResponseNode(Node):
        """Node that formats the response using recalled memories."""

        def exec(self, shared: SharedStore) -> Dict[str, Any]:
            question = shared.get("question")
            recalled = shared.get("recalled_memories", [])

            if recalled:
                # Use the most relevant memory
                best_match = recalled[0]
                response = f"Based on my knowledge: {best_match['text']}"
                confidence = best_match["score"]
            else:
                response = "I don't have enough information to answer that."
                confidence = 0.0

            shared.set("response", response)
            shared.set("confidence", confidence)

            return {"response": response, "confidence": confidence}

    # Build the flow
    question_node = QuestionNode(name="QuestionProcessor")
    recall_node = MemoryRecallNode(
        knowledge_store=knowledge_store, top_k=3, name="MemoryRecall"
    )
    response_node = ResponseNode(name="ResponseGenerator")

    flow = Flow(
        nodes=[question_node, recall_node, response_node], name="Q&A Agent", enable_logging=True
    )

    # Ask questions
    questions = [
        "What is Python?",
        "Tell me about version control",
        "What's used for databases?",
    ]

    print("\n❓ Asking questions to the agent:\n")
    for question in questions:
        print(f"Q: {question}")

        results = flow.execute(initial_data={"question": question})
        flow.reset()  # Reset for next question

        response = results["ResponseGenerator"]["response"]
        confidence = results["ResponseGenerator"]["confidence"]

        print(f"A: {response}")
        print(f"   (confidence: {confidence:.2f})\n")


def demo_metadata_filtering():
    """Demo 4: Advanced search with metadata."""
    print("\n" + "=" * 70)
    print("DEMO 4: Metadata-based Organization")
    print("=" * 70)

    provider = SimpleHashEmbeddingProvider(dim=384)
    memory = KnowledgeStore(embedding_provider=provider)

    # Store memories with different categories
    print("\n📂 Organizing memories by category...")

    memories = [
        ("Completed project Alpha", {"category": "work", "status": "done", "priority": "high"}),
        ("Meeting with team tomorrow", {"category": "work", "status": "pending", "priority": "high"}),
        ("Buy groceries", {"category": "personal", "status": "pending", "priority": "medium"}),
        ("Call dentist", {"category": "personal", "status": "pending", "priority": "low"}),
        ("Finished reading Python book", {"category": "learning", "status": "done", "priority": "medium"}),
        ("Learn about AI agents", {"category": "learning", "status": "pending", "priority": "high"}),
    ]

    for text, metadata in memories:
        memory.add(text, metadata=metadata)

    # Search and display by category
    print("\n🔍 Searching for work-related tasks...")
    results = memory.search("work tasks", top_k=5)

    print("\n📋 Work-related memories:")
    for mem, score in results:
        if mem.metadata.get("category") == "work":
            status = mem.metadata.get("status")
            priority = mem.metadata.get("priority")
            print(f"  [{priority}] {mem.text} ({status})")


def demo_persistence():
    """Demo 5: Saving and loading memories."""
    print("\n" + "=" * 70)
    print("DEMO 5: Memory Persistence")
    print("=" * 70)

    provider = SimpleHashEmbeddingProvider(dim=384)

    # Create and populate store
    print("\n💾 Creating memory store...")
    memory1 = KnowledgeStore(embedding_provider=provider)
    memory1.add("Important fact 1")
    memory1.add("Important fact 2")
    memory1.add("Important fact 3")
    print(f"✅ Created store with {memory1.count()} memories")

    # Save to file
    import tempfile
    import os

    temp_file = tempfile.mktemp(suffix=".json")
    print(f"\n💾 Saving to: {temp_file}")
    memory1.save_to_file(temp_file)

    # Create new store and load
    print("\n📂 Loading memories into new store...")
    memory2 = KnowledgeStore(embedding_provider=provider)
    memory2.load_from_file(temp_file)
    print(f"✅ Loaded {memory2.count()} memories")

    # Verify
    print("\n✓ Verifying memories match...")
    original_memories = sorted([m.text for m in memory1.list_all()])
    loaded_memories = sorted([m.text for m in memory2.list_all()])

    if original_memories == loaded_memories:
        print("✅ All memories preserved correctly!")
    else:
        print("❌ Memories don't match!")

    # Cleanup
    os.unlink(temp_file)


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("🧠 KnowledgeLayer Demo - Semantic Memory for AI Agents")
    print("=" * 70)
    print("\nThis demo shows how to use the KnowledgeLayer to build")
    print("agents with long-term semantic memory capabilities.")

    try:
        demo_basic_memory()
        demo_conversation_memory()
        demo_flow_integration()
        demo_metadata_filtering()
        demo_persistence()

        print("\n" + "=" * 70)
        print("✅ All demos completed successfully!")
        print("=" * 70)
        print("\n💡 Next steps:")
        print("   1. Try with OpenAI embeddings for better semantic understanding")
        print("   2. Build your own agent with persistent memory")
        print("   3. Explore the docs/knowledge_layer_guide.md for more examples")
        print("\n")

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
