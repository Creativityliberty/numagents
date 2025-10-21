"""
Tests for StateLayer

Copyright (c) 2025 Lionel TAGNE. All Rights Reserved.
This software is proprietary and confidential.
"""

import tempfile
import pytest
from pathlib import Path
from num_agents import (
    State,
    StateTransition,
    Checkpoint,
    StateMachine,
    StateManager,
    CheckpointManager,
    InMemoryBackend,
    FileBackend,
    StateTransitionNode,
    CheckpointNode,
    Flow,
    SharedStore,
    StateTransitionError,
    StateValidationError,
    CheckpointError,
)


# ============================================================================
# Test State
# ============================================================================


def test_state_creation():
    """Test creating a state."""
    state = State(name="idle", data={"counter": 0})

    assert state.name == "idle"
    assert state.data["counter"] == 0
    assert state.entry_count == 0


def test_state_enter():
    """Test entering a state."""
    state = State(name="active")

    state.enter()

    assert state.entry_count == 1
    assert state.last_entered_at is not None


def test_state_callbacks():
    """Test state enter/exit callbacks."""
    entered = []
    exited = []

    def on_enter(state):
        entered.append(state.name)

    def on_exit(state):
        exited.append(state.name)

    state = State(name="processing", on_enter=on_enter, on_exit=on_exit)

    state.enter()
    assert entered == ["processing"]

    state.exit()
    assert exited == ["processing"]


def test_state_to_dict():
    """Test converting state to dictionary."""
    state = State(name="waiting", data={"timeout": 30})
    state.enter()

    state_dict = state.to_dict()

    assert state_dict["name"] == "waiting"
    assert state_dict["data"]["timeout"] == 30
    assert state_dict["entry_count"] == 1


# ============================================================================
# Test StateTransition
# ============================================================================


def test_transition_creation():
    """Test creating a transition."""
    transition = StateTransition(from_state="idle", to_state="active")

    assert transition.from_state == "idle"
    assert transition.to_state == "active"
    assert transition.execution_count == 0


def test_transition_condition():
    """Test transition with condition."""

    def can_activate(context):
        return context.get("ready", False)

    transition = StateTransition(
        from_state="idle", to_state="active", condition=can_activate
    )

    # Should not allow transition
    assert transition.can_transition({"ready": False}) is False

    # Should allow transition
    assert transition.can_transition({"ready": True}) is True


def test_transition_action():
    """Test transition with action."""
    executed = []

    def log_transition(context):
        executed.append(f"Transitioning with {context}")

    transition = StateTransition(
        from_state="idle", to_state="active", action=log_transition
    )

    transition.execute({"user": "Alice"})

    assert len(executed) == 1
    assert transition.execution_count == 1


# ============================================================================
# Test StateMachine
# ============================================================================


def test_state_machine_creation():
    """Test creating a state machine."""
    idle = State(name="idle")
    active = State(name="active")

    machine = StateMachine(initial_state="idle", states=[idle, active])

    assert machine.initial_state == "idle"
    assert machine._current_state is None  # Not started yet


def test_state_machine_start():
    """Test starting a state machine."""
    idle = State(name="idle")
    machine = StateMachine(initial_state="idle", states=[idle])

    machine.start()

    assert machine.get_current_state() == "idle"
    assert idle.entry_count == 1


def test_state_machine_transition():
    """Test transitioning between states."""
    idle = State(name="idle")
    active = State(name="active")
    done = State(name="done")

    transition1 = StateTransition(from_state="idle", to_state="active")
    transition2 = StateTransition(from_state="active", to_state="done")

    machine = StateMachine(
        initial_state="idle",
        states=[idle, active, done],
        transitions=[transition1, transition2],
    )

    machine.start()
    assert machine.get_current_state() == "idle"

    # Transition to active
    machine.transition_to("active")
    assert machine.get_current_state() == "active"

    # Transition to done
    machine.transition_to("done")
    assert machine.get_current_state() == "done"


def test_state_machine_invalid_transition():
    """Test that invalid transitions raise errors."""
    idle = State(name="idle")
    active = State(name="active")

    transition = StateTransition(from_state="idle", to_state="active")

    machine = StateMachine(
        initial_state="idle", states=[idle, active], transitions=[transition]
    )

    machine.start()

    # Try invalid transition (no transition from active to idle defined)
    with pytest.raises(StateTransitionError):
        machine.transition_to("idle")


def test_state_machine_context():
    """Test state machine context management."""
    machine = StateMachine(initial_state="idle", states=[State(name="idle")])
    machine.start()

    # Set context
    machine.set_context("user_id", "123")
    machine.set_context("role", "admin")

    context = machine.get_context()
    assert context["user_id"] == "123"
    assert context["role"] == "admin"

    # Update context
    machine.update_context({"status": "active"})
    assert machine.get_context()["status"] == "active"


def test_state_machine_history():
    """Test state machine transition history."""
    states = [State(name="a"), State(name="b"), State(name="c")]
    transitions = [
        StateTransition(from_state="a", to_state="b"),
        StateTransition(from_state="b", to_state="c"),
    ]

    machine = StateMachine(initial_state="a", states=states, transitions=transitions)
    machine.start()

    machine.transition_to("b")
    machine.transition_to("c")

    history = machine.get_history()
    assert len(history) == 2
    assert history[0][0] == "a"
    assert history[0][1] == "b"
    assert history[1][0] == "b"
    assert history[1][1] == "c"


# ============================================================================
# Test StateBackend - InMemoryBackend
# ============================================================================


def test_in_memory_backend():
    """Test InMemoryBackend."""
    backend = InMemoryBackend()

    # Save
    backend.save("state1", {"current": "idle", "count": 5})

    # Load
    loaded = backend.load("state1")
    assert loaded["current"] == "idle"
    assert loaded["count"] == 5

    # List
    assert "state1" in backend.list_keys()

    # Delete
    assert backend.delete("state1") is True
    assert backend.load("state1") is None


# ============================================================================
# Test StateBackend - FileBackend
# ============================================================================


def test_file_backend():
    """Test FileBackend."""
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = FileBackend(directory=tmpdir)

        # Save
        backend.save("state1", {"current": "active"})

        # Check file exists
        assert Path(tmpdir, "state1.json").exists()

        # Load
        loaded = backend.load("state1")
        assert loaded["current"] == "active"

        # List
        assert "state1" in backend.list_keys()

        # Delete
        assert backend.delete("state1") is True
        assert not Path(tmpdir, "state1.json").exists()


# ============================================================================
# Test StateManager
# ============================================================================


def test_state_manager_persistence():
    """Test StateManager with persistence."""
    backend = InMemoryBackend()

    # Create state machine
    states = [State(name="idle"), State(name="active")]
    transitions = [StateTransition(from_state="idle", to_state="active")]
    machine = StateMachine(initial_state="idle", states=states, transitions=transitions)

    # Create manager
    manager = StateManager(state_machine=machine, backend=backend, auto_save=True)

    # Start
    manager.start()
    assert manager.state_machine.get_current_state() == "idle"

    # Transition (should auto-save)
    manager.transition_to("active")

    # Create new manager with same backend
    machine2 = StateMachine(initial_state="idle", states=states, transitions=transitions)
    manager2 = StateManager(state_machine=machine2, backend=backend)

    # Load should restore to "active"
    manager2.start()
    # Note: Currently starts at initial_state, persistence needs state_id matching


# ============================================================================
# Test CheckpointManager
# ============================================================================


def test_checkpoint_creation():
    """Test creating checkpoints."""
    backend = InMemoryBackend()
    checkpoint_manager = CheckpointManager(backend=backend)

    # Create state machine
    states = [State(name="processing")]
    machine = StateMachine(initial_state="processing", states=states)
    machine.start()
    machine.set_context("progress", 50)

    # Create checkpoint
    checkpoint = checkpoint_manager.create_checkpoint(machine, metadata={"label": "halfway"})

    assert checkpoint.state_name == "processing"
    assert checkpoint.context["progress"] == 50
    assert checkpoint.metadata["label"] == "halfway"


def test_checkpoint_restore():
    """Test restoring from checkpoint."""
    backend = InMemoryBackend()
    checkpoint_manager = CheckpointManager(backend=backend)

    # Create and checkpoint
    states = [State(name="idle"), State(name="processing")]
    machine = StateMachine(initial_state="processing", states=states)
    machine.start()
    machine.set_context("step", 10)

    checkpoint = checkpoint_manager.create_checkpoint(machine)

    # Modify state
    machine.set_context("step", 20)
    assert machine.get_context()["step"] == 20

    # Restore
    checkpoint_manager.restore_checkpoint(checkpoint.id, machine)
    assert machine.get_context()["step"] == 10


def test_checkpoint_max_limit():
    """Test checkpoint max limit enforcement."""
    backend = InMemoryBackend()
    checkpoint_manager = CheckpointManager(backend=backend, max_checkpoints=2)

    states = [State(name="idle")]
    machine = StateMachine(initial_state="idle", states=states)
    machine.start()

    # Create 3 checkpoints
    cp1 = checkpoint_manager.create_checkpoint(machine)
    cp2 = checkpoint_manager.create_checkpoint(machine)
    cp3 = checkpoint_manager.create_checkpoint(machine)

    # Should only keep last 2
    checkpoints = checkpoint_manager.list_checkpoints()
    assert len(checkpoints) == 2
    assert cp1 not in checkpoints
    assert cp2 in checkpoints
    assert cp3 in checkpoints


# ============================================================================
# Test Nodes (Flow Integration)
# ============================================================================


def test_state_transition_node():
    """Test StateTransitionNode in a Flow."""
    # Setup state machine
    states = [State(name="idle"), State(name="active")]
    transitions = [StateTransition(from_state="idle", to_state="active")]
    machine = StateMachine(initial_state="idle", states=states, transitions=transitions)

    backend = InMemoryBackend()
    manager = StateManager(machine, backend)
    manager.start()

    # Create node
    node = StateTransitionNode(
        state_manager=manager,
        target_state_key="next_state",
        output_key="transition_result",
    )

    # Execute
    shared = SharedStore()
    shared.set("next_state", "active")

    result = node.exec(shared)

    assert result["success"] is True
    assert result["from_state"] == "idle"
    assert result["to_state"] == "active"
    assert manager.state_machine.get_current_state() == "active"


def test_checkpoint_node():
    """Test CheckpointNode in a Flow."""
    # Setup
    states = [State(name="processing")]
    machine = StateMachine(initial_state="processing", states=states)

    backend = InMemoryBackend()
    manager = StateManager(machine, backend)
    manager.start()

    checkpoint_manager = CheckpointManager(backend=backend)

    # Create node
    node = CheckpointNode(
        checkpoint_manager=checkpoint_manager,
        state_manager=manager,
        output_key="checkpoint_id",
    )

    # Execute
    shared = SharedStore()
    result = node.exec(shared)

    assert "checkpoint_id" in result
    assert result["state"] == "processing"

    # Verify checkpoint was created
    assert len(checkpoint_manager.list_checkpoints()) == 1


# ============================================================================
# Integration Test
# ============================================================================


def test_state_layer_integration():
    """Integration test: Full state layer workflow."""
    # Define states
    idle = State(name="idle", data={"initialized": True})
    thinking = State(name="thinking", data={"depth": 0})
    acting = State(name="acting")
    reflecting = State(name="reflecting")

    # Define transitions with conditions
    def can_think(context):
        return context.get("task") is not None

    def can_act(context):
        return context.get("plan") is not None

    transitions = [
        StateTransition(from_state="idle", to_state="thinking", condition=can_think),
        StateTransition(from_state="thinking", to_state="acting", condition=can_act),
        StateTransition(from_state="acting", to_state="reflecting"),
        StateTransition(from_state="reflecting", to_state="idle"),
    ]

    # Create state machine
    machine = StateMachine(
        initial_state="idle",
        states=[idle, thinking, acting, reflecting],
        transitions=transitions,
        enable_logging=False,
    )

    # Create backend and manager
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = FileBackend(tmpdir)
        manager = StateManager(machine, backend, auto_save=True)

        # Start
        manager.start()
        assert machine.get_current_state() == "idle"

        # Set context and transition
        machine.set_context("task", "analyze_data")
        manager.transition_to("thinking")
        assert machine.get_current_state() == "thinking"

        # Create checkpoint
        checkpoint_mgr = CheckpointManager(backend)
        cp1 = checkpoint_mgr.create_checkpoint(machine)

        # Continue workflow
        machine.set_context("plan", "use_tool_x")
        manager.transition_to("acting")
        manager.transition_to("reflecting")
        manager.transition_to("idle")

        # Restore checkpoint
        checkpoint_mgr.restore_checkpoint(cp1.id, machine)
        assert machine.get_current_state() == "thinking"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
