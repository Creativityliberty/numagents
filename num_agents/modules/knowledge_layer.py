"""
KnowledgeLayer - Memory and Knowledge Management with Vector Storage

This module provides vector-based memory storage and retrieval for AI agents.

Copyright (c) 2025 Lionel TAGNE. All Rights Reserved.
This software is proprietary and confidential.
"""

import json
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid
import hashlib

from num_agents.core import Node, SharedStore
from num_agents.exceptions import NumAgentsException
from num_agents.logging_config import get_logger

logger = get_logger(__name__)


class KnowledgeLayerException(NumAgentsException):
    """Base exception for KnowledgeLayer errors."""

    pass


class EmbeddingProviderError(KnowledgeLayerException):
    """Exception raised when embedding provider fails."""

    pass


class VectorStoreError(KnowledgeLayerException):
    """Exception raised when vector store operations fail."""

    pass


# ============================================================================
# Embedding Providers
# ============================================================================


class EmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers.

    Embedding providers convert text into vector representations.
    """

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector
        """
        pass

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get the dimension of the embeddings."""
        pass


class SimpleHashEmbeddingProvider(EmbeddingProvider):
    """
    Simple hash-based embedding provider for testing.

    WARNING: This is NOT suitable for production! It uses a deterministic
    hash function to create fake embeddings. Use OpenAI, Cohere, or
    sentence-transformers for real applications.
    """

    def __init__(self, dim: int = 384) -> None:
        """
        Initialize the hash embedding provider.

        Args:
            dim: Dimension of the embedding vectors
        """
        self._dim = dim

    def embed_text(self, text: str) -> List[float]:
        """Generate a deterministic hash-based embedding."""
        # Create a deterministic hash
        hash_bytes = hashlib.sha256(text.encode()).digest()

        # Convert to floats in range [-1, 1]
        embedding = []
        for i in range(self._dim):
            byte_idx = i % len(hash_bytes)
            value = (hash_bytes[byte_idx] / 255.0) * 2 - 1
            embedding.append(value)

        # Normalize the vector
        magnitude = sum(x * x for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]

        return embedding

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        return [self.embed_text(text) for text in texts]

    @property
    def dimension(self) -> int:
        """Get the dimension of embeddings."""
        return self._dim


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    OpenAI embedding provider using text-embedding-3-small or similar models.

    Requires: openai package
    """

    def __init__(
        self, api_key: Optional[str] = None, model: str = "text-embedding-3-small"
    ) -> None:
        """
        Initialize OpenAI embedding provider.

        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: OpenAI embedding model to use
        """
        try:
            import openai  # type: ignore
        except ImportError:
            raise EmbeddingProviderError(
                "OpenAI package not installed. Install with: pip install openai"
            )

        self.model = model
        self.client = openai.OpenAI(api_key=api_key) if api_key else openai.OpenAI()

        # Get dimension by making a test call
        try:
            test_response = self.client.embeddings.create(input="test", model=self.model)
            self._dim = len(test_response.data[0].embedding)
        except Exception as e:
            raise EmbeddingProviderError(f"Failed to initialize OpenAI provider: {e}")

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding using OpenAI API."""
        try:
            response = self.client.embeddings.create(input=text, model=self.model)
            return response.data[0].embedding
        except Exception as e:
            raise EmbeddingProviderError(f"Failed to generate embedding: {e}")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        try:
            response = self.client.embeddings.create(input=texts, model=self.model)
            return [item.embedding for item in response.data]
        except Exception as e:
            raise EmbeddingProviderError(f"Failed to generate embeddings: {e}")

    @property
    def dimension(self) -> int:
        """Get the dimension of embeddings."""
        return self._dim


# ============================================================================
# Vector Store
# ============================================================================


class Memory:
    """Represents a single memory with text, embedding, and metadata."""

    def __init__(
        self,
        text: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None,
    ) -> None:
        """
        Initialize a memory.

        Args:
            text: The memory text
            embedding: Vector embedding of the text
            metadata: Optional metadata dictionary
            id: Optional custom ID (auto-generated if not provided)
        """
        self.id = id or str(uuid.uuid4())
        self.text = text
        self.embedding = embedding
        self.metadata = metadata or {}
        self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to dictionary."""
        return {
            "id": self.id,
            "text": self.text,
            "embedding": self.embedding,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Memory":
        """Create memory from dictionary."""
        memory = cls(
            text=data["text"],
            embedding=data["embedding"],
            metadata=data.get("metadata", {}),
            id=data.get("id"),
        )
        memory.timestamp = data.get("timestamp", time.time())
        return memory


class KnowledgeStore:
    """
    Vector-based knowledge store for semantic memory.

    Stores text with embeddings and enables semantic search.
    """

    def __init__(
        self, embedding_provider: EmbeddingProvider, enable_logging: bool = False
    ) -> None:
        """
        Initialize the knowledge store.

        Args:
            embedding_provider: Provider for generating embeddings
            enable_logging: Enable detailed logging
        """
        self.embedding_provider = embedding_provider
        self._memories: Dict[str, Memory] = {}
        self._enable_logging = enable_logging
        self._logger = get_logger(__name__) if enable_logging else None

    def add(
        self, text: str, metadata: Optional[Dict[str, Any]] = None, id: Optional[str] = None
    ) -> str:
        """
        Add a memory to the store.

        Args:
            text: Text to store
            metadata: Optional metadata
            id: Optional custom ID

        Returns:
            ID of the stored memory
        """
        try:
            # Generate embedding
            embedding = self.embedding_provider.embed_text(text)

            # Create memory
            memory = Memory(text=text, embedding=embedding, metadata=metadata, id=id)

            # Store
            self._memories[memory.id] = memory

            if self._enable_logging and self._logger:
                self._logger.debug(f"Added memory: {memory.id[:8]}... (len={len(text)})")

            return memory.id

        except Exception as e:
            raise VectorStoreError(f"Failed to add memory: {e}")

    def add_batch(
        self, texts: List[str], metadata_list: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        Add multiple memories in batch.

        Args:
            texts: List of texts to store
            metadata_list: Optional list of metadata dicts (same length as texts)

        Returns:
            List of memory IDs
        """
        try:
            # Generate embeddings in batch
            embeddings = self.embedding_provider.embed_texts(texts)

            # Create memories
            ids = []
            for i, (text, embedding) in enumerate(zip(texts, embeddings)):
                metadata = metadata_list[i] if metadata_list else None
                memory = Memory(text=text, embedding=embedding, metadata=metadata)
                self._memories[memory.id] = memory
                ids.append(memory.id)

            if self._enable_logging and self._logger:
                self._logger.info(f"Added {len(ids)} memories in batch")

            return ids

        except Exception as e:
            raise VectorStoreError(f"Failed to add batch: {e}")

    def search(
        self, query: str, top_k: int = 5, threshold: Optional[float] = None
    ) -> List[Tuple[Memory, float]]:
        """
        Semantic search for similar memories.

        Args:
            query: Search query
            top_k: Number of results to return
            threshold: Optional similarity threshold (0-1)

        Returns:
            List of (Memory, similarity_score) tuples, sorted by similarity
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_provider.embed_text(query)

            # Calculate similarities
            results = []
            for memory in self._memories.values():
                similarity = self._cosine_similarity(query_embedding, memory.embedding)

                if threshold is None or similarity >= threshold:
                    results.append((memory, similarity))

            # Sort by similarity (descending)
            results.sort(key=lambda x: x[1], reverse=True)

            # Return top_k
            top_results = results[:top_k]

            if self._enable_logging and self._logger:
                self._logger.debug(
                    f"Search returned {len(top_results)} results (query: '{query[:50]}...')"
                )

            return top_results

        except Exception as e:
            raise VectorStoreError(f"Search failed: {e}")

    def get(self, memory_id: str) -> Optional[Memory]:
        """
        Get a memory by ID.

        Args:
            memory_id: ID of the memory

        Returns:
            Memory object or None if not found
        """
        return self._memories.get(memory_id)

    def delete(self, memory_id: str) -> bool:
        """
        Delete a memory by ID.

        Args:
            memory_id: ID of the memory

        Returns:
            True if deleted, False if not found
        """
        if memory_id in self._memories:
            del self._memories[memory_id]
            return True
        return False

    def clear(self) -> None:
        """Clear all memories."""
        self._memories.clear()

    def count(self) -> int:
        """Get the number of memories stored."""
        return len(self._memories)

    def list_all(self) -> List[Memory]:
        """Get all memories."""
        return list(self._memories.values())

    def export_to_dict(self) -> Dict[str, Any]:
        """Export knowledge store to dictionary."""
        return {
            "memories": [mem.to_dict() for mem in self._memories.values()],
            "dimension": self.embedding_provider.dimension,
            "count": len(self._memories),
        }

    def import_from_dict(self, data: Dict[str, Any]) -> None:
        """Import knowledge store from dictionary."""
        self._memories.clear()
        for mem_data in data.get("memories", []):
            memory = Memory.from_dict(mem_data)
            self._memories[memory.id] = memory

    def save_to_file(self, filepath: str) -> None:
        """Save knowledge store to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.export_to_dict(), f, indent=2)

        if self._enable_logging and self._logger:
            self._logger.info(f"Saved {self.count()} memories to {filepath}")

    def load_from_file(self, filepath: str) -> None:
        """Load knowledge store from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)

        self.import_from_dict(data)

        if self._enable_logging and self._logger:
            self._logger.info(f"Loaded {self.count()} memories from {filepath}")

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Similarity score between 0 and 1
        """
        # Dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))

        # Magnitudes
        mag1 = sum(a * a for a in vec1) ** 0.5
        mag2 = sum(b * b for b in vec2) ** 0.5

        # Avoid division by zero
        if mag1 == 0 or mag2 == 0:
            return 0.0

        # Cosine similarity
        similarity = dot_product / (mag1 * mag2)

        # Normalize to 0-1 range (cosine similarity is -1 to 1)
        return (similarity + 1) / 2


# ============================================================================
# Nodes for Flow Integration
# ============================================================================


class MemoryStoreNode(Node):
    """
    Node that stores text in knowledge memory.

    Reads text from SharedStore and adds it to the KnowledgeStore.
    """

    def __init__(
        self,
        knowledge_store: KnowledgeStore,
        text_key: str = "text",
        metadata_key: Optional[str] = None,
        output_key: str = "memory_id",
        name: Optional[str] = None,
        enable_logging: bool = False,
    ) -> None:
        """
        Initialize MemoryStoreNode.

        Args:
            knowledge_store: KnowledgeStore instance to use
            text_key: Key in SharedStore to read text from
            metadata_key: Optional key in SharedStore to read metadata from
            output_key: Key in SharedStore to write memory ID to
            name: Optional node name
            enable_logging: Enable detailed logging
        """
        super().__init__(name or "MemoryStoreNode", enable_logging=enable_logging)
        self.knowledge_store = knowledge_store
        self.text_key = text_key
        self.metadata_key = metadata_key
        self.output_key = output_key

    def exec(self, shared: SharedStore) -> Dict[str, Any]:
        """
        Execute memory storage.

        Args:
            shared: SharedStore instance

        Returns:
            Dictionary with execution results
        """
        # Get text from shared store
        text = shared.get_required(self.text_key)

        # Get metadata if specified
        metadata = None
        if self.metadata_key:
            metadata = shared.get(self.metadata_key)

        # Store in knowledge store
        memory_id = self.knowledge_store.add(text=text, metadata=metadata)

        # Store memory ID in shared store
        shared.set(self.output_key, memory_id)

        return {"memory_id": memory_id, "text_length": len(text), "stored": True}


class MemoryRecallNode(Node):
    """
    Node that recalls memories using semantic search.

    Searches the KnowledgeStore for relevant memories and stores results.
    """

    def __init__(
        self,
        knowledge_store: KnowledgeStore,
        query_key: str = "query",
        output_key: str = "recalled_memories",
        top_k: int = 5,
        threshold: Optional[float] = None,
        include_scores: bool = True,
        name: Optional[str] = None,
        enable_logging: bool = False,
    ) -> None:
        """
        Initialize MemoryRecallNode.

        Args:
            knowledge_store: KnowledgeStore instance to search
            query_key: Key in SharedStore to read search query from
            output_key: Key in SharedStore to write results to
            top_k: Number of results to return
            threshold: Optional similarity threshold
            include_scores: Include similarity scores in results
            name: Optional node name
            enable_logging: Enable detailed logging
        """
        super().__init__(name or "MemoryRecallNode", enable_logging=enable_logging)
        self.knowledge_store = knowledge_store
        self.query_key = query_key
        self.output_key = output_key
        self.top_k = top_k
        self.threshold = threshold
        self.include_scores = include_scores

    def exec(self, shared: SharedStore) -> Dict[str, Any]:
        """
        Execute memory recall.

        Args:
            shared: SharedStore instance

        Returns:
            Dictionary with execution results
        """
        # Get query from shared store
        query = shared.get_required(self.query_key)

        # Search knowledge store
        results = self.knowledge_store.search(
            query=query, top_k=self.top_k, threshold=self.threshold
        )

        # Format results
        if self.include_scores:
            formatted_results = [
                {"id": mem.id, "text": mem.text, "metadata": mem.metadata, "score": score}
                for mem, score in results
            ]
        else:
            formatted_results = [
                {"id": mem.id, "text": mem.text, "metadata": mem.metadata}
                for mem, _ in results
            ]

        # Store results in shared store
        shared.set(self.output_key, formatted_results)

        return {
            "query": query,
            "results_count": len(results),
            "top_score": results[0][1] if results else 0.0,
        }
