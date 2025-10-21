"""
StateLayer - State Management and Persistence for AI Agents

This module provides finite state machines, persistent state, checkpointing,
and state transition management for complex agents.

Copyright (c) 2025 Lionel TAGNE. All Rights Reserved.
This software is proprietary and confidential.
"""

import json
import pickle
import time
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import uuid

from num_agents.core import Node, SharedStore
from num_agents.exceptions import NumAgentsException
from num_agents.logging_config import get_logger

logger = get_logger(__name__)


# ============================================================================
# Exceptions
# ============================================================================


class StateLayerException(NumAgentsException):
    """Base exception for StateLayer errors."""

    pass


class StateTransitionError(StateLayerException):
    """Exception raised when state transition fails."""

    pass


class CheckpointError(StateLayerException):
    """Exception raised when checkpoint operations fail."""

    pass


class StateValidationError(StateLayerException):
    """Exception raised when state validation fails."""

    pass


class StatePersistenceError(StateLayerException):
    """Exception raised when state persistence fails."""

    pass


# ============================================================================
# Data Classes
# ============================================================================


class State:
    """Represents a single state in a state machine."""

    def __init__(
        self,
        name: str,
        data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        on_enter: Optional[Callable] = None,
        on_exit: Optional[Callable] = None,
    ) -> None:
        """
        Initialize a state.

        Args:
            name: State name (unique identifier)
            data: State-specific data
            metadata: Optional metadata (tags, description, etc.)
            on_enter: Optional callback when entering this state
            on_exit: Optional callback when exiting this state
        """
        self.name = name
        self.data = data or {}
        self.metadata = metadata or {}
        self.on_enter = on_enter
        self.on_exit = on_exit
        self.entry_count = 0
        self.created_at = time.time()
        self.last_entered_at: Optional[float] = None

    def enter(self) -> None:
        """Called when entering this state."""
        self.entry_count += 1
        self.last_entered_at = time.time()
        if self.on_enter:
            self.on_enter(self)

    def exit(self) -> None:
        """Called when exiting this state."""
        if self.on_exit:
            self.on_exit(self)

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "name": self.name,
            "data": self.data,
            "metadata": self.metadata,
            "entry_count": self.entry_count,
            "created_at": self.created_at,
            "last_entered_at": self.last_entered_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "State":
        """Create state from dictionary."""
        state = cls(
            name=data["name"],
            data=data.get("data", {}),
            metadata=data.get("metadata", {}),
        )
        state.entry_count = data.get("entry_count", 0)
        state.created_at = data.get("created_at", time.time())
        state.last_entered_at = data.get("last_entered_at")
        return state


class StateTransition:
    """Represents a transition between states."""

    def __init__(
        self,
        from_state: str,
        to_state: str,
        condition: Optional[Callable[[Dict[str, Any]], bool]] = None,
        action: Optional[Callable[[Dict[str, Any]], None]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize a state transition.

        Args:
            from_state: Source state name
            to_state: Target state name
            condition: Optional condition function (returns True if transition allowed)
            action: Optional action to execute during transition
            metadata: Optional metadata (label, priority, etc.)
        """
        self.id = str(uuid.uuid4())
        self.from_state = from_state
        self.to_state = to_state
        self.condition = condition
        self.action = action
        self.metadata = metadata or {}
        self.execution_count = 0
        self.created_at = time.time()

    def can_transition(self, context: Dict[str, Any]) -> bool:
        """
        Check if transition is allowed.

        Args:
            context: Context data for condition evaluation

        Returns:
            True if transition is allowed
        """
        if self.condition is None:
            return True
        return self.condition(context)

    def execute(self, context: Dict[str, Any]) -> None:
        """
        Execute transition action.

        Args:
            context: Context data for action execution
        """
        self.execution_count += 1
        if self.action:
            self.action(context)

    def to_dict(self) -> Dict[str, Any]:
        """Convert transition to dictionary."""
        return {
            "id": self.id,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "metadata": self.metadata,
            "execution_count": self.execution_count,
            "created_at": self.created_at,
        }


class Checkpoint:
    """Represents a state checkpoint (snapshot)."""

    def __init__(
        self,
        state_name: str,
        state_data: Dict[str, Any],
        context: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None,
    ) -> None:
        """
        Initialize a checkpoint.

        Args:
            state_name: Name of the state
            state_data: State data at checkpoint
            context: Full context at checkpoint
            metadata: Optional metadata (label, auto_created, etc.)
            id: Optional custom ID
        """
        self.id = id or str(uuid.uuid4())
        self.state_name = state_name
        self.state_data = state_data.copy()
        self.context = context.copy()
        self.metadata = metadata or {}
        self.created_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert checkpoint to dictionary."""
        return {
            "id": self.id,
            "state_name": self.state_name,
            "state_data": self.state_data,
            "context": self.context,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Checkpoint":
        """Create checkpoint from dictionary."""
        checkpoint = cls(
            state_name=data["state_name"],
            state_data=data["state_data"],
            context=data["context"],
            metadata=data.get("metadata", {}),
            id=data.get("id"),
        )
        checkpoint.created_at = data.get("created_at", time.time())
        return checkpoint


# ============================================================================
# State Backend (Abstract)
# ============================================================================


class StateBackend(ABC):
    """
    Abstract base class for state persistence backends.

    Backends handle saving and loading state to/from storage.
    """

    @abstractmethod
    def save(self, key: str, data: Dict[str, Any]) -> None:
        """
        Save state data.

        Args:
            key: Unique identifier for the state
            data: State data to save
        """
        pass

    @abstractmethod
    def load(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Load state data.

        Args:
            key: Unique identifier for the state

        Returns:
            State data or None if not found
        """
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """
        Delete state data.

        Args:
            key: Unique identifier for the state

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    def list_keys(self) -> List[str]:
        """
        List all state keys.

        Returns:
            List of state keys
        """
        pass


# ============================================================================
# Concrete Backend Implementations
# ============================================================================


class InMemoryBackend(StateBackend):
    """In-memory state storage (not persistent across restarts)."""

    def __init__(self) -> None:
        """Initialize in-memory backend."""
        self._storage: Dict[str, Dict[str, Any]] = {}

    def save(self, key: str, data: Dict[str, Any]) -> None:
        """Save to memory."""
        self._storage[key] = data.copy()

    def load(self, key: str) -> Optional[Dict[str, Any]]:
        """Load from memory."""
        return self._storage.get(key)

    def delete(self, key: str) -> bool:
        """Delete from memory."""
        if key in self._storage:
            del self._storage[key]
            return True
        return False

    def list_keys(self) -> List[str]:
        """List all keys."""
        return list(self._storage.keys())

    def clear(self) -> None:
        """Clear all stored data."""
        self._storage.clear()


class FileBackend(StateBackend):
    """File-based state storage using JSON."""

    def __init__(self, directory: Union[str, Path]) -> None:
        """
        Initialize file backend.

        Args:
            directory: Directory to store state files
        """
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _get_filepath(self, key: str) -> Path:
        """Get filepath for a state key."""
        # Sanitize key to be filesystem-safe
        safe_key = key.replace("/", "_").replace("\\", "_")
        return self.directory / f"{safe_key}.json"

    def save(self, key: str, data: Dict[str, Any]) -> None:
        """Save to JSON file."""
        try:
            filepath = self._get_filepath(key)
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            raise StatePersistenceError(f"Failed to save state '{key}': {e}") from e

    def load(self, key: str) -> Optional[Dict[str, Any]]:
        """Load from JSON file."""
        try:
            filepath = self._get_filepath(key)
            if not filepath.exists():
                return None

            with open(filepath, "r") as f:
                return json.load(f)
        except Exception as e:
            raise StatePersistenceError(f"Failed to load state '{key}': {e}") from e

    def delete(self, key: str) -> bool:
        """Delete JSON file."""
        try:
            filepath = self._get_filepath(key)
            if filepath.exists():
                filepath.unlink()
                return True
            return False
        except Exception as e:
            raise StatePersistenceError(f"Failed to delete state '{key}': {e}") from e

    def list_keys(self) -> List[str]:
        """List all state keys."""
        keys = []
        for filepath in self.directory.glob("*.json"):
            key = filepath.stem
            keys.append(key)
        return keys


class PickleBackend(StateBackend):
    """File-based state storage using Pickle (supports any Python object)."""

    def __init__(self, directory: Union[str, Path]) -> None:
        """
        Initialize pickle backend.

        Args:
            directory: Directory to store state files
        """
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _get_filepath(self, key: str) -> Path:
        """Get filepath for a state key."""
        safe_key = key.replace("/", "_").replace("\\", "_")
        return self.directory / f"{safe_key}.pkl"

    def save(self, key: str, data: Dict[str, Any]) -> None:
        """Save to pickle file."""
        try:
            filepath = self._get_filepath(key)
            with open(filepath, "wb") as f:
                pickle.dump(data, f)
        except Exception as e:
            raise StatePersistenceError(f"Failed to save state '{key}': {e}") from e

    def load(self, key: str) -> Optional[Dict[str, Any]]:
        """Load from pickle file."""
        try:
            filepath = self._get_filepath(key)
            if not filepath.exists():
                return None

            with open(filepath, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            raise StatePersistenceError(f"Failed to load state '{key}': {e}") from e

    def delete(self, key: str) -> bool:
        """Delete pickle file."""
        try:
            filepath = self._get_filepath(key)
            if filepath.exists():
                filepath.unlink()
                return True
            return False
        except Exception as e:
            raise StatePersistenceError(f"Failed to delete state '{key}': {e}") from e

    def list_keys(self) -> List[str]:
        """List all state keys."""
        keys = []
        for filepath in self.directory.glob("*.pkl"):
            key = filepath.stem
            keys.append(key)
        return keys


# ============================================================================
# State Machine
# ============================================================================


class StateMachine:
    """
    Finite State Machine for agent state management.

    Manages states, transitions, and context.
    """

    def __init__(
        self,
        initial_state: str,
        states: Optional[List[State]] = None,
        transitions: Optional[List[StateTransition]] = None,
        enable_logging: bool = False,
    ) -> None:
        """
        Initialize state machine.

        Args:
            initial_state: Name of initial state
            states: Optional list of states
            transitions: Optional list of transitions
            enable_logging: Enable detailed logging
        """
        self.initial_state = initial_state
        self._states: Dict[str, State] = {}
        self._transitions: Dict[str, List[StateTransition]] = {}
        self._current_state: Optional[State] = None
        self._context: Dict[str, Any] = {}
        self._history: List[Tuple[str, str, float]] = []  # (from, to, timestamp)
        self._enable_logging = enable_logging
        self._logger = get_logger(__name__) if enable_logging else None

        # Add states
        if states:
            for state in states:
                self.add_state(state)

        # Add transitions
        if transitions:
            for transition in transitions:
                self.add_transition(transition)

    def add_state(self, state: State) -> None:
        """
        Add a state to the machine.

        Args:
            state: State to add
        """
        self._states[state.name] = state

        if self._enable_logging and self._logger:
            self._logger.debug(f"Added state: {state.name}")

    def add_transition(self, transition: StateTransition) -> None:
        """
        Add a transition to the machine.

        Args:
            transition: Transition to add
        """
        if transition.from_state not in self._transitions:
            self._transitions[transition.from_state] = []

        self._transitions[transition.from_state].append(transition)

        if self._enable_logging and self._logger:
            self._logger.debug(
                f"Added transition: {transition.from_state} -> {transition.to_state}"
            )

    def start(self) -> None:
        """Start the state machine (enter initial state)."""
        if self.initial_state not in self._states:
            raise StateValidationError(
                f"Initial state '{self.initial_state}' not found in states"
            )

        self._current_state = self._states[self.initial_state]
        self._current_state.enter()

        if self._enable_logging and self._logger:
            self._logger.info(f"Started state machine in state: {self.initial_state}")

    def transition_to(self, target_state: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Transition to a target state.

        Args:
            target_state: Name of target state
            context: Optional context for transition condition/action

        Returns:
            True if transition successful, False otherwise

        Raises:
            StateTransitionError: If transition fails
        """
        if self._current_state is None:
            raise StateTransitionError("State machine not started. Call start() first.")

        if target_state not in self._states:
            raise StateTransitionError(f"Target state '{target_state}' not found")

        # Get available transitions
        available_transitions = self._transitions.get(self._current_state.name, [])

        # Find matching transition
        matching_transition = None
        for transition in available_transitions:
            if transition.to_state == target_state:
                # Check condition
                transition_context = {**self._context, **(context or {})}
                if transition.can_transition(transition_context):
                    matching_transition = transition
                    break

        if matching_transition is None:
            raise StateTransitionError(
                f"No valid transition from '{self._current_state.name}' to '{target_state}'"
            )

        # Execute transition
        from_state = self._current_state.name
        transition_context = {**self._context, **(context or {})}

        # Exit current state
        self._current_state.exit()

        # Execute transition action
        matching_transition.execute(transition_context)

        # Enter new state
        self._current_state = self._states[target_state]
        self._current_state.enter()

        # Record in history
        self._history.append((from_state, target_state, time.time()))

        if self._enable_logging and self._logger:
            self._logger.info(f"Transitioned: {from_state} -> {target_state}")

        return True

    def get_available_transitions(self) -> List[str]:
        """
        Get list of states that can be transitioned to from current state.

        Returns:
            List of target state names
        """
        if self._current_state is None:
            return []

        transitions = self._transitions.get(self._current_state.name, [])
        return [t.to_state for t in transitions]

    def get_current_state(self) -> Optional[str]:
        """Get name of current state."""
        return self._current_state.name if self._current_state else None

    def get_context(self) -> Dict[str, Any]:
        """Get current context."""
        return self._context.copy()

    def set_context(self, key: str, value: Any) -> None:
        """Set context value."""
        self._context[key] = value

    def update_context(self, updates: Dict[str, Any]) -> None:
        """Update multiple context values."""
        self._context.update(updates)

    def get_history(self) -> List[Tuple[str, str, float]]:
        """Get transition history."""
        return self._history.copy()

    def reset(self) -> None:
        """Reset to initial state and clear context/history."""
        self._context.clear()
        self._history.clear()
        self._current_state = None
        self.start()

    def to_dict(self) -> Dict[str, Any]:
        """Export state machine to dictionary."""
        return {
            "initial_state": self.initial_state,
            "current_state": self.get_current_state(),
            "context": self._context,
            "history": self._history,
            "states": {name: state.to_dict() for name, state in self._states.items()},
        }


# ============================================================================
# State Manager
# ============================================================================


class StateManager:
    """
    High-level state management with persistence.

    Combines StateMachine with StateBackend for persistent state.
    """

    def __init__(
        self,
        state_machine: StateMachine,
        backend: Optional[StateBackend] = None,
        auto_save: bool = True,
        enable_logging: bool = False,
    ) -> None:
        """
        Initialize state manager.

        Args:
            state_machine: StateMachine to manage
            backend: Optional persistence backend (default: InMemoryBackend)
            auto_save: Automatically save state after transitions
            enable_logging: Enable detailed logging
        """
        self.state_machine = state_machine
        self.backend = backend or InMemoryBackend()
        self.auto_save = auto_save
        self._enable_logging = enable_logging
        self._logger = get_logger(__name__) if enable_logging else None
        self._state_id = str(uuid.uuid4())

    def start(self) -> None:
        """Start state machine and load persisted state if available."""
        # Try to load existing state
        persisted = self.backend.load(self._state_id)

        if persisted:
            # Restore from persisted state
            self.state_machine._context = persisted.get("context", {})
            self.state_machine._history = persisted.get("history", [])

            current_state = persisted.get("current_state")
            if current_state and current_state in self.state_machine._states:
                self.state_machine._current_state = self.state_machine._states[
                    current_state
                ]

                if self._enable_logging and self._logger:
                    self._logger.info(
                        f"Loaded persisted state: {current_state} (id={self._state_id[:8]}...)"
                    )
            else:
                self.state_machine.start()
        else:
            self.state_machine.start()

    def transition_to(
        self, target_state: str, context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Transition to target state (with auto-save if enabled).

        Args:
            target_state: Name of target state
            context: Optional context for transition

        Returns:
            True if successful
        """
        result = self.state_machine.transition_to(target_state, context)

        if result and self.auto_save:
            self.save()

        return result

    def save(self) -> None:
        """Save current state to backend."""
        state_data = self.state_machine.to_dict()
        self.backend.save(self._state_id, state_data)

        if self._enable_logging and self._logger:
            self._logger.debug(f"Saved state (id={self._state_id[:8]}...)")

    def load(self) -> None:
        """Load state from backend."""
        persisted = self.backend.load(self._state_id)
        if persisted:
            self.state_machine._context = persisted.get("context", {})
            self.state_machine._history = persisted.get("history", [])

            current_state = persisted.get("current_state")
            if current_state and current_state in self.state_machine._states:
                self.state_machine._current_state = self.state_machine._states[
                    current_state
                ]

    def delete(self) -> bool:
        """Delete persisted state."""
        return self.backend.delete(self._state_id)


# ============================================================================
# Checkpoint Manager
# ============================================================================


class CheckpointManager:
    """
    Manages state checkpoints (snapshots).

    Enables saving and restoring state at specific points.
    """

    def __init__(
        self,
        backend: Optional[StateBackend] = None,
        max_checkpoints: Optional[int] = None,
        enable_logging: bool = False,
    ) -> None:
        """
        Initialize checkpoint manager.

        Args:
            backend: Optional persistence backend (default: InMemoryBackend)
            max_checkpoints: Maximum number of checkpoints to keep (None = unlimited)
            enable_logging: Enable detailed logging
        """
        self.backend = backend or InMemoryBackend()
        self.max_checkpoints = max_checkpoints
        self._checkpoints: List[Checkpoint] = []
        self._enable_logging = enable_logging
        self._logger = get_logger(__name__) if enable_logging else None

    def create_checkpoint(
        self,
        state_machine: StateMachine,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Checkpoint:
        """
        Create a checkpoint from current state machine state.

        Args:
            state_machine: StateMachine to checkpoint
            metadata: Optional metadata

        Returns:
            Created Checkpoint
        """
        current_state = state_machine.get_current_state()
        if current_state is None:
            raise CheckpointError("Cannot checkpoint: state machine not started")

        state_data = state_machine._states[current_state].to_dict()
        context = state_machine.get_context()

        checkpoint = Checkpoint(
            state_name=current_state,
            state_data=state_data,
            context=context,
            metadata=metadata,
        )

        # Add to list
        self._checkpoints.append(checkpoint)

        # Enforce max_checkpoints
        if self.max_checkpoints and len(self._checkpoints) > self.max_checkpoints:
            removed = self._checkpoints.pop(0)
            self.backend.delete(f"checkpoint_{removed.id}")

        # Persist
        self.backend.save(f"checkpoint_{checkpoint.id}", checkpoint.to_dict())

        if self._enable_logging and self._logger:
            self._logger.info(
                f"Created checkpoint: {checkpoint.id[:8]}... (state={current_state})"
            )

        return checkpoint

    def restore_checkpoint(
        self, checkpoint_id: str, state_machine: StateMachine
    ) -> None:
        """
        Restore state machine from a checkpoint.

        Args:
            checkpoint_id: ID of checkpoint to restore
            state_machine: StateMachine to restore to

        Raises:
            CheckpointError: If checkpoint not found
        """
        # Load checkpoint
        checkpoint_data = self.backend.load(f"checkpoint_{checkpoint_id}")
        if checkpoint_data is None:
            raise CheckpointError(f"Checkpoint '{checkpoint_id}' not found")

        checkpoint = Checkpoint.from_dict(checkpoint_data)

        # Restore state machine
        state_machine._context = checkpoint.context.copy()

        if checkpoint.state_name in state_machine._states:
            state_machine._current_state = state_machine._states[checkpoint.state_name]
        else:
            raise CheckpointError(
                f"State '{checkpoint.state_name}' not found in state machine"
            )

        if self._enable_logging and self._logger:
            self._logger.info(
                f"Restored checkpoint: {checkpoint_id[:8]}... (state={checkpoint.state_name})"
            )

    def list_checkpoints(self) -> List[Checkpoint]:
        """Get list of all checkpoints."""
        return self._checkpoints.copy()

    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Delete a checkpoint.

        Args:
            checkpoint_id: ID of checkpoint to delete

        Returns:
            True if deleted, False if not found
        """
        # Remove from list
        self._checkpoints = [c for c in self._checkpoints if c.id != checkpoint_id]

        # Remove from backend
        return self.backend.delete(f"checkpoint_{checkpoint_id}")

    def clear_all(self) -> None:
        """Clear all checkpoints."""
        for checkpoint in self._checkpoints:
            self.backend.delete(f"checkpoint_{checkpoint.id}")
        self._checkpoints.clear()


# ============================================================================
# Nodes for Flow Integration
# ============================================================================


class StateTransitionNode(Node):
    """
    Node that performs a state transition.

    Reads target state from SharedStore and performs transition.
    """

    def __init__(
        self,
        state_manager: StateManager,
        target_state_key: str = "target_state",
        context_key: Optional[str] = None,
        output_key: str = "transition_result",
        name: Optional[str] = None,
        enable_logging: bool = False,
    ) -> None:
        """
        Initialize StateTransitionNode.

        Args:
            state_manager: StateManager instance
            target_state_key: Key in SharedStore to read target state from
            context_key: Optional key to read transition context from
            output_key: Key in SharedStore to write result to
            name: Optional node name
            enable_logging: Enable detailed logging
        """
        super().__init__(name or "StateTransition", enable_logging=enable_logging)
        self.state_manager = state_manager
        self.target_state_key = target_state_key
        self.context_key = context_key
        self.output_key = output_key

    def exec(self, shared: SharedStore) -> Dict[str, Any]:
        """
        Execute state transition.

        Args:
            shared: SharedStore instance

        Returns:
            Dictionary with execution results
        """
        # Get target state
        target_state = shared.get_required(self.target_state_key)

        # Get optional context
        context = None
        if self.context_key:
            context = shared.get(self.context_key)

        # Perform transition
        current_state = self.state_manager.state_machine.get_current_state()
        success = self.state_manager.transition_to(target_state, context)

        result = {
            "success": success,
            "from_state": current_state,
            "to_state": target_state,
            "timestamp": time.time(),
        }

        # Store result
        shared.set(self.output_key, result)

        return result


class CheckpointNode(Node):
    """
    Node that creates a state checkpoint.

    Creates a checkpoint of current state machine state.
    """

    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        state_manager: StateManager,
        metadata_key: Optional[str] = None,
        output_key: str = "checkpoint_id",
        name: Optional[str] = None,
        enable_logging: bool = False,
    ) -> None:
        """
        Initialize CheckpointNode.

        Args:
            checkpoint_manager: CheckpointManager instance
            state_manager: StateManager instance
            metadata_key: Optional key to read checkpoint metadata from
            output_key: Key in SharedStore to write checkpoint ID to
            name: Optional node name
            enable_logging: Enable detailed logging
        """
        super().__init__(name or "Checkpoint", enable_logging=enable_logging)
        self.checkpoint_manager = checkpoint_manager
        self.state_manager = state_manager
        self.metadata_key = metadata_key
        self.output_key = output_key

    def exec(self, shared: SharedStore) -> Dict[str, Any]:
        """
        Create checkpoint.

        Args:
            shared: SharedStore instance

        Returns:
            Dictionary with execution results
        """
        # Get optional metadata
        metadata = None
        if self.metadata_key:
            metadata = shared.get(self.metadata_key)

        # Create checkpoint
        checkpoint = self.checkpoint_manager.create_checkpoint(
            self.state_manager.state_machine, metadata
        )

        # Store checkpoint ID
        shared.set(self.output_key, checkpoint.id)

        return {
            "checkpoint_id": checkpoint.id,
            "state": checkpoint.state_name,
            "timestamp": checkpoint.created_at,
        }
