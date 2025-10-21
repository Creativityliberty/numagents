"""
Tests for KnowledgeLayer module.

Copyright (c) 2025 Lionel TAGNE. All Rights Reserved.
This software is proprietary and confidential.
"""

import json
import os
import tempfile
from typing import Any, Dict

import pytest

from num_agents.core import Flow, SharedStore
from num_agents.modules.knowledge_layer import (
    EmbeddingProvider,
    KnowledgeStore,
    Memory,
    MemoryRecallNode,
    MemoryStoreNode,
    OpenAIEmbeddingProvider,
    SimpleHashEmbeddingProvider,
    VectorStoreError,
)


class TestSimpleHashEmbeddingProvider:
    """Test SimpleHashEmbeddingProvider."""

    def test_embed_text(self) -> None:
        """Test single text embedding."""
        provider = SimpleHashEmbeddingProvider(dim=384)
        embedding = provider.embed_text("Hello world")

        assert len(embedding) == 384
        assert all(isinstance(x, float) for x in embedding)

        # Check normalization (magnitude should be ~1)
        magnitude = sum(x * x for x in embedding) ** 0.5
        assert abs(magnitude - 1.0) < 0.01

    def test_embed_texts_batch(self) -> None:
        """Test batch text embedding."""
        provider = SimpleHashEmbeddingProvider(dim=384)
        texts = ["Hello", "World", "Test"]
        embeddings = provider.embed_texts(texts)

        assert len(embeddings) == 3
        assert all(len(emb) == 384 for emb in embeddings)

    def test_deterministic(self) -> None:
        """Test that embeddings are deterministic."""
        provider = SimpleHashEmbeddingProvider(dim=384)
        text = "Test text"

        emb1 = provider.embed_text(text)
        emb2 = provider.embed_text(text)

        assert emb1 == emb2

    def test_different_texts_different_embeddings(self) -> None:
        """Test that different texts have different embeddings."""
        provider = SimpleHashEmbeddingProvider(dim=384)

        emb1 = provider.embed_text("Hello")
        emb2 = provider.embed_text("World")

        assert emb1 != emb2

    def test_dimension_property(self) -> None:
        """Test dimension property."""
        provider = SimpleHashEmbeddingProvider(dim=256)
        assert provider.dimension == 256


class TestMemory:
    """Test Memory class."""

    def test_create_memory(self) -> None:
        """Test memory creation."""
        embedding = [0.1, 0.2, 0.3]
        metadata = {"source": "test"}

        memory = Memory(text="Test", embedding=embedding, metadata=metadata)

        assert memory.text == "Test"
        assert memory.embedding == embedding
        assert memory.metadata == metadata
        assert memory.id is not None
        assert memory.timestamp > 0

    def test_memory_to_dict(self) -> None:
        """Test memory serialization."""
        memory = Memory(
            text="Test", embedding=[0.1, 0.2], metadata={"key": "value"}, id="test-id"
        )

        data = memory.to_dict()

        assert data["id"] == "test-id"
        assert data["text"] == "Test"
        assert data["embedding"] == [0.1, 0.2]
        assert data["metadata"] == {"key": "value"}
        assert "timestamp" in data

    def test_memory_from_dict(self) -> None:
        """Test memory deserialization."""
        data = {
            "id": "test-id",
            "text": "Test",
            "embedding": [0.1, 0.2],
            "metadata": {"key": "value"},
            "timestamp": 123456.0,
        }

        memory = Memory.from_dict(data)

        assert memory.id == "test-id"
        assert memory.text == "Test"
        assert memory.embedding == [0.1, 0.2]
        assert memory.metadata == {"key": "value"}
        assert memory.timestamp == 123456.0


class TestKnowledgeStore:
    """Test KnowledgeStore."""

    @pytest.fixture
    def knowledge_store(self) -> KnowledgeStore:
        """Create a knowledge store with hash embeddings."""
        provider = SimpleHashEmbeddingProvider(dim=384)
        return KnowledgeStore(embedding_provider=provider)

    def test_add_memory(self, knowledge_store: KnowledgeStore) -> None:
        """Test adding a memory."""
        memory_id = knowledge_store.add("This is a test")

        assert memory_id is not None
        assert knowledge_store.count() == 1

    def test_add_memory_with_metadata(self, knowledge_store: KnowledgeStore) -> None:
        """Test adding memory with metadata."""
        metadata = {"source": "test", "priority": "high"}
        memory_id = knowledge_store.add("Test text", metadata=metadata)

        memory = knowledge_store.get(memory_id)
        assert memory is not None
        assert memory.metadata == metadata

    def test_add_batch(self, knowledge_store: KnowledgeStore) -> None:
        """Test batch adding memories."""
        texts = ["First", "Second", "Third"]
        metadata_list = [{"index": i} for i in range(3)]

        ids = knowledge_store.add_batch(texts, metadata_list=metadata_list)

        assert len(ids) == 3
        assert knowledge_store.count() == 3

        for i, memory_id in enumerate(ids):
            memory = knowledge_store.get(memory_id)
            assert memory is not None
            assert memory.text == texts[i]
            assert memory.metadata == metadata_list[i]

    def test_search(self, knowledge_store: KnowledgeStore) -> None:
        """Test semantic search."""
        # Add some memories
        knowledge_store.add("Python is a programming language")
        knowledge_store.add("JavaScript is also a language")
        knowledge_store.add("Cats are animals")

        # Search for programming-related content
        results = knowledge_store.search("programming", top_k=2)

        assert len(results) <= 2
        assert all(isinstance(mem, Memory) for mem, _ in results)
        assert all(isinstance(score, float) for _, score in results)

        # Results should be sorted by similarity (descending)
        if len(results) > 1:
            assert results[0][1] >= results[1][1]

    def test_search_with_threshold(self, knowledge_store: KnowledgeStore) -> None:
        """Test search with similarity threshold."""
        knowledge_store.add("Exact match test")

        # Exact match should have very high similarity
        results = knowledge_store.search("Exact match test", threshold=0.95)

        assert len(results) >= 1

    def test_get_memory(self, knowledge_store: KnowledgeStore) -> None:
        """Test getting a memory by ID."""
        text = "Test memory"
        memory_id = knowledge_store.add(text)

        memory = knowledge_store.get(memory_id)

        assert memory is not None
        assert memory.id == memory_id
        assert memory.text == text

    def test_get_nonexistent_memory(self, knowledge_store: KnowledgeStore) -> None:
        """Test getting a memory that doesn't exist."""
        memory = knowledge_store.get("nonexistent-id")
        assert memory is None

    def test_delete_memory(self, knowledge_store: KnowledgeStore) -> None:
        """Test deleting a memory."""
        memory_id = knowledge_store.add("To be deleted")
        assert knowledge_store.count() == 1

        deleted = knowledge_store.delete(memory_id)

        assert deleted is True
        assert knowledge_store.count() == 0
        assert knowledge_store.get(memory_id) is None

    def test_delete_nonexistent(self, knowledge_store: KnowledgeStore) -> None:
        """Test deleting a memory that doesn't exist."""
        deleted = knowledge_store.delete("nonexistent-id")
        assert deleted is False

    def test_clear(self, knowledge_store: KnowledgeStore) -> None:
        """Test clearing all memories."""
        knowledge_store.add("Memory 1")
        knowledge_store.add("Memory 2")
        assert knowledge_store.count() == 2

        knowledge_store.clear()

        assert knowledge_store.count() == 0

    def test_list_all(self, knowledge_store: KnowledgeStore) -> None:
        """Test listing all memories."""
        knowledge_store.add("First")
        knowledge_store.add("Second")
        knowledge_store.add("Third")

        memories = knowledge_store.list_all()

        assert len(memories) == 3
        assert all(isinstance(mem, Memory) for mem in memories)

    def test_export_import(self, knowledge_store: KnowledgeStore) -> None:
        """Test export and import."""
        # Add memories
        knowledge_store.add("Test 1", metadata={"index": 1})
        knowledge_store.add("Test 2", metadata={"index": 2})

        # Export
        data = knowledge_store.export_to_dict()

        assert "memories" in data
        assert "dimension" in data
        assert "count" in data
        assert len(data["memories"]) == 2

        # Create new store and import
        provider = SimpleHashEmbeddingProvider(dim=384)
        new_store = KnowledgeStore(embedding_provider=provider)
        new_store.import_from_dict(data)

        assert new_store.count() == 2

    def test_save_load_file(self, knowledge_store: KnowledgeStore) -> None:
        """Test saving and loading from file."""
        # Add memories
        knowledge_store.add("File test 1")
        knowledge_store.add("File test 2")

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_path = f.name

        try:
            knowledge_store.save_to_file(temp_path)

            # Create new store and load
            provider = SimpleHashEmbeddingProvider(dim=384)
            new_store = KnowledgeStore(embedding_provider=provider)
            new_store.load_from_file(temp_path)

            assert new_store.count() == 2

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_cosine_similarity(self) -> None:
        """Test cosine similarity calculation."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]

        # Same vectors should have similarity 1.0 (normalized to 0.5-1 range)
        similarity = KnowledgeStore._cosine_similarity(vec1, vec2)
        assert abs(similarity - 1.0) < 0.01

        # Orthogonal vectors
        vec3 = [0.0, 1.0, 0.0]
        similarity = KnowledgeStore._cosine_similarity(vec1, vec3)
        assert abs(similarity - 0.5) < 0.01  # Normalized to 0.5 for orthogonal


class TestMemoryStoreNode:
    """Test MemoryStoreNode."""

    def test_store_memory(self) -> None:
        """Test storing memory via node."""
        provider = SimpleHashEmbeddingProvider(dim=384)
        knowledge_store = KnowledgeStore(embedding_provider=provider)
        node = MemoryStoreNode(knowledge_store=knowledge_store)

        shared = SharedStore()
        shared.set("text", "This is a test memory")

        result = node.exec(shared)

        assert result["stored"] is True
        assert "memory_id" in result
        assert knowledge_store.count() == 1

        # Check memory ID is stored in shared store
        memory_id = shared.get("memory_id")
        assert memory_id is not None

    def test_store_with_metadata(self) -> None:
        """Test storing memory with metadata."""
        provider = SimpleHashEmbeddingProvider(dim=384)
        knowledge_store = KnowledgeStore(embedding_provider=provider)
        node = MemoryStoreNode(
            knowledge_store=knowledge_store, metadata_key="meta"
        )

        shared = SharedStore()
        shared.set("text", "Test")
        shared.set("meta", {"source": "test"})

        result = node.exec(shared)

        memory = knowledge_store.get(result["memory_id"])
        assert memory is not None
        assert memory.metadata == {"source": "test"}

    def test_custom_keys(self) -> None:
        """Test custom key names."""
        provider = SimpleHashEmbeddingProvider(dim=384)
        knowledge_store = KnowledgeStore(embedding_provider=provider)
        node = MemoryStoreNode(
            knowledge_store=knowledge_store,
            text_key="my_text",
            output_key="my_id",
        )

        shared = SharedStore()
        shared.set("my_text", "Custom key test")

        node.exec(shared)

        assert shared.has("my_id")


class TestMemoryRecallNode:
    """Test MemoryRecallNode."""

    def test_recall_memory(self) -> None:
        """Test recalling memories."""
        provider = SimpleHashEmbeddingProvider(dim=384)
        knowledge_store = KnowledgeStore(embedding_provider=provider)

        # Add some memories
        knowledge_store.add("Python programming")
        knowledge_store.add("JavaScript coding")
        knowledge_store.add("Cooking recipes")

        node = MemoryRecallNode(knowledge_store=knowledge_store, top_k=2)

        shared = SharedStore()
        shared.set("query", "programming")

        result = node.exec(shared)

        assert result["results_count"] <= 2

        # Check results stored in shared store
        recalled = shared.get("recalled_memories")
        assert recalled is not None
        assert isinstance(recalled, list)
        assert all("text" in mem for mem in recalled)
        assert all("score" in mem for mem in recalled)

    def test_recall_with_threshold(self) -> None:
        """Test recall with similarity threshold."""
        provider = SimpleHashEmbeddingProvider(dim=384)
        knowledge_store = KnowledgeStore(embedding_provider=provider)
        knowledge_store.add("Exact match")

        node = MemoryRecallNode(
            knowledge_store=knowledge_store, threshold=0.95, top_k=10
        )

        shared = SharedStore()
        shared.set("query", "Exact match")

        result = node.exec(shared)

        # Should find at least the exact match
        assert result["results_count"] >= 1

    def test_no_scores(self) -> None:
        """Test recall without scores."""
        provider = SimpleHashEmbeddingProvider(dim=384)
        knowledge_store = KnowledgeStore(embedding_provider=provider)
        knowledge_store.add("Test")

        node = MemoryRecallNode(
            knowledge_store=knowledge_store, include_scores=False
        )

        shared = SharedStore()
        shared.set("query", "Test")

        node.exec(shared)

        recalled = shared.get("recalled_memories")
        assert all("score" not in mem for mem in recalled)

    def test_in_flow(self) -> None:
        """Test MemoryStoreNode and MemoryRecallNode in a flow."""
        provider = SimpleHashEmbeddingProvider(dim=384)
        knowledge_store = KnowledgeStore(embedding_provider=provider)

        store_node = MemoryStoreNode(knowledge_store=knowledge_store)
        recall_node = MemoryRecallNode(knowledge_store=knowledge_store)

        flow = Flow(nodes=[store_node, recall_node])

        # Execute flow
        initial_data = {"text": "AI and machine learning", "query": "machine"}
        results = flow.execute(initial_data=initial_data)

        # Check both nodes executed
        assert "MemoryStoreNode" in results
        assert "MemoryRecallNode" in results

        # Check recall found the stored memory
        recalled = flow.shared.get("recalled_memories")
        assert len(recalled) >= 1
