"""
StateLayer Demo - Demonstrating State Management capabilities

This example shows how to:
1. Create finite state machines
2. Manage state transitions
3. Persist state across sessions
4. Create and restore checkpoints
5. Integrate state management with Flows

Copyright (c) 2025 Lionel TAGNE. All Rights Reserved.
This software is proprietary and confidential.
"""

import tempfile
from pathlib import Path
from num_agents import (
    State,
    StateTransition,
    StateMachine,
    StateManager,
    CheckpointManager,
    InMemoryBackend,
    FileBackend,
    StateTransitionNode,
    CheckpointNode,
    Flow,
    Node,
    SharedStore,
)


# ============================================================================
# Example 1: Basic State Machine
# ============================================================================


def basic_state_machine():
    """Demonstrate basic state machine creation and transitions."""
    print("=" * 60)
    print("Example 1: Basic State Machine")
    print("=" * 60)

    # Define states
    idle = State(name="idle", data={"initialized": True})
    thinking = State(name="thinking", data={"depth": 0})
    acting = State(name="acting", data={"tool": None})
    done = State(name="done", data={"success": False})

    # Define transitions
    transitions = [
        StateTransition(from_state="idle", to_state="thinking"),
        StateTransition(from_state="thinking", to_state="acting"),
        StateTransition(from_state="acting", to_state="done"),
        StateTransition(from_state="done", to_state="idle"),  # Loop back
    ]

    # Create state machine
    machine = StateMachine(
        initial_state="idle", states=[idle, thinking, acting, done], transitions=transitions
    )

    # Start
    machine.start()
    print(f"\n🎬 Started in state: {machine.get_current_state()}")

    # Perform transitions
    print("\n🔄 Performing transitions...")
    machine.transition_to("thinking")
    print(f"  → {machine.get_current_state()}")

    machine.transition_to("acting")
    print(f"  → {machine.get_current_state()}")

    machine.transition_to("done")
    print(f"  → {machine.get_current_state()}")

    # View history
    print("\n📜 Transition History:")
    for from_state, to_state, timestamp in machine.get_history():
        print(f"  {from_state} → {to_state}")


# ============================================================================
# Example 2: State Machine with Conditions
# ============================================================================


def state_machine_with_conditions():
    """Demonstrate state transitions with conditions."""
    print("\n" + "=" * 60)
    print("Example 2: State Machine with Conditions")
    print("=" * 60)

    # Define states
    states = [
        State(name="waiting"),
        State(name="processing"),
        State(name="completed"),
        State(name="failed"),
    ]

    # Define conditional transitions
    def has_valid_input(context):
        return context.get("input_valid", False)

    def processing_successful(context):
        return context.get("success", False)

    transitions = [
        StateTransition(
            from_state="waiting", to_state="processing", condition=has_valid_input
        ),
        StateTransition(
            from_state="processing",
            to_state="completed",
            condition=processing_successful,
        ),
        StateTransition(
            from_state="processing",
            to_state="failed",
            condition=lambda ctx: not processing_successful(ctx),
        ),
    ]

    # Create machine
    machine = StateMachine(
        initial_state="waiting", states=states, transitions=transitions
    )
    machine.start()

    print(f"\n📍 Current state: {machine.get_current_state()}")

    # Try transition without valid input
    print("\n❌ Attempting transition without valid input...")
    try:
        machine.transition_to("processing", context={"input_valid": False})
    except Exception as e:
        print(f"  Error: {type(e).__name__}")

    # Set valid input and transition
    print("\n✅ Setting valid input and transitioning...")
    machine.set_context("input_valid", True)
    machine.transition_to("processing", context={"input_valid": True})
    print(f"  → {machine.get_current_state()}")

    # Simulate successful processing
    machine.set_context("success", True)
    machine.transition_to("completed", context={"success": True})
    print(f"  → {machine.get_current_state()}")


# ============================================================================
# Example 3: State Persistence
# ============================================================================


def state_persistence_demo():
    """Demonstrate persisting state across sessions."""
    print("\n" + "=" * 60)
    print("Example 3: State Persistence")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        backend = FileBackend(directory=tmpdir)

        # Create state machine
        states = [State(name="idle"), State(name="active"), State(name="paused")]
        transitions = [
            StateTransition(from_state="idle", to_state="active"),
            StateTransition(from_state="active", to_state="paused"),
            StateTransition(from_state="paused", to_state="active"),
        ]

        machine = StateMachine(
            initial_state="idle", states=states, transitions=transitions
        )

        # Create manager with persistence
        manager = StateManager(machine, backend=backend, auto_save=True)

        print("\n💾 Session 1: Creating and saving state...")
        manager.start()
        machine.set_context("user_id", "user123")
        machine.set_context("task_count", 5)

        manager.transition_to("active")
        print(f"  State: {machine.get_current_state()}")
        print(f"  Context: {machine.get_context()}")

        # Save state
        manager.save()
        print("  ✅ State saved to disk")

        # Simulate new session - create new manager with same backend
        print("\n🔄 Session 2: Loading previous state...")
        machine2 = StateMachine(
            initial_state="idle", states=states, transitions=transitions
        )
        manager2 = StateManager(machine2, backend=backend)

        # Note: Current implementation starts at initial_state
        # In production, you'd load the saved state in start()
        manager2.load()
        print(f"  Loaded context: {machine2.get_context()}")


# ============================================================================
# Example 4: Checkpoints
# ============================================================================


def checkpoints_demo():
    """Demonstrate checkpoint creation and restoration."""
    print("\n" + "=" * 60)
    print("Example 4: Checkpoints (Save Points)")
    print("=" * 60)

    backend = InMemoryBackend()
    checkpoint_mgr = CheckpointManager(backend=backend, max_checkpoints=5)

    # Create state machine
    states = [State(name="level1"), State(name="level2"), State(name="level3")]
    transitions = [
        StateTransition(from_state="level1", to_state="level2"),
        StateTransition(from_state="level2", to_state="level3"),
    ]

    machine = StateMachine(
        initial_state="level1", states=states, transitions=transitions
    )
    machine.start()

    # Progress through levels
    print("\n🎮 Gaming Scenario:")

    # Level 1 - create checkpoint
    machine.set_context("score", 100)
    machine.set_context("health", 100)
    cp1 = checkpoint_mgr.create_checkpoint(machine, metadata={"label": "Level 1 Start"})
    print(f"💾 Checkpoint 1: {cp1.state_name}, score={cp1.context['score']}")

    # Level 2
    machine.transition_to("level2")
    machine.set_context("score", 250)
    machine.set_context("health", 75)
    cp2 = checkpoint_mgr.create_checkpoint(machine, metadata={"label": "Level 2 Start"})
    print(f"💾 Checkpoint 2: {cp2.state_name}, score={cp2.context['score']}")

    # Level 3 - things go bad
    machine.transition_to("level3")
    machine.set_context("score", 300)
    machine.set_context("health", 10)
    print(f"\n❌ Current state: {machine.get_current_state()}, health=10 (low!)")

    # Restore to checkpoint 2
    print(f"\n🔄 Restoring to Checkpoint 2...")
    checkpoint_mgr.restore_checkpoint(cp2.id, machine)
    print(
        f"✅ Restored: {machine.get_current_state()}, health={machine.get_context()['health']}"
    )

    # List all checkpoints
    print("\n📋 Available Checkpoints:")
    for cp in checkpoint_mgr.list_checkpoints():
        print(f"  - {cp.metadata.get('label', cp.id)}: {cp.state_name}")


# ============================================================================
# Example 5: Agent Lifecycle State Machine
# ============================================================================


def agent_lifecycle_demo():
    """Demonstrate a real-world agent lifecycle state machine."""
    print("\n" + "=" * 60)
    print("Example 5: Agent Lifecycle State Machine")
    print("=" * 60)

    # Define agent states with callbacks
    def on_enter_thinking(state):
        print("  🧠 Entering thinking state...")

    def on_exit_thinking(state):
        print("  🧠 Exiting thinking state...")

    def on_enter_acting(state):
        print("  🎬 Entering acting state...")

    # Define states
    states = [
        State(name="idle"),
        State(name="thinking", on_enter=on_enter_thinking, on_exit=on_exit_thinking),
        State(name="acting", on_enter=on_enter_acting),
        State(name="reflecting"),
        State(name="learning"),
    ]

    # Define agent workflow transitions
    def has_task(context):
        return context.get("task") is not None

    def has_plan(context):
        return context.get("plan") is not None

    def action_completed(context):
        return context.get("action_done", False)

    transitions = [
        StateTransition(from_state="idle", to_state="thinking", condition=has_task),
        StateTransition(from_state="thinking", to_state="acting", condition=has_plan),
        StateTransition(
            from_state="acting", to_state="reflecting", condition=action_completed
        ),
        StateTransition(from_state="reflecting", to_state="learning"),
        StateTransition(from_state="learning", to_state="idle"),
    ]

    # Create machine
    machine = StateMachine(
        initial_state="idle", states=states, transitions=transitions
    )

    print("\n🤖 Agent Lifecycle:")
    machine.start()
    print(f"1. {machine.get_current_state()}")

    # Receive task
    machine.set_context("task", "Analyze data")
    machine.transition_to("thinking", context={"task": "Analyze data"})
    print(f"2. {machine.get_current_state()}")

    # Create plan
    machine.set_context("plan", "Use tool X")
    machine.transition_to("acting", context={"plan": "Use tool X"})
    print(f"3. {machine.get_current_state()}")

    # Complete action
    machine.set_context("action_done", True)
    machine.transition_to("reflecting", context={"action_done": True})
    print(f"4. {machine.get_current_state()}")

    # Learn and return to idle
    machine.transition_to("learning")
    print(f"5. {machine.get_current_state()}")

    machine.transition_to("idle")
    print(f"6. {machine.get_current_state()} (ready for next task)")


# ============================================================================
# Example 6: Flow Integration
# ============================================================================


class TaskInputNode(Node):
    """Node that provides a task."""

    def exec(self, shared: SharedStore) -> dict:
        shared.set("task", "Process user data")
        shared.set("target_state", "thinking")
        print("\n📥 Task received: Process user data")
        return {"task_received": True}


class PlanningNode(Node):
    """Node that creates a plan."""

    def exec(self, shared: SharedStore) -> dict:
        shared.set("plan", "Use data_processor tool")
        shared.set("target_state", "acting")
        print("🧠 Created plan: Use data_processor tool")
        return {"plan_created": True}


class ActionNode(Node):
    """Node that executes action."""

    def exec(self, shared: SharedStore) -> dict:
        shared.set("action_result", "Data processed successfully")
        shared.set("target_state", "done")
        print("🎬 Action executed: Data processed")
        return {"action_done": True}


def flow_integration_demo():
    """Demonstrate integrating state management with Flows."""
    print("\n" + "=" * 60)
    print("Example 6: Flow Integration")
    print("=" * 60)

    # Setup state machine
    states = [State(name="idle"), State(name="thinking"), State(name="acting"), State(name="done")]
    transitions = [
        StateTransition(from_state="idle", to_state="thinking"),
        StateTransition(from_state="thinking", to_state="acting"),
        StateTransition(from_state="acting", to_state="done"),
    ]

    machine = StateMachine(
        initial_state="idle", states=states, transitions=transitions
    )
    backend = InMemoryBackend()
    manager = StateManager(machine, backend)
    manager.start()

    checkpoint_mgr = CheckpointManager(backend)

    # Build flow
    flow = Flow(name="StatefulAgentFlow")

    # Add nodes
    task_node = TaskInputNode(name="TaskInput")
    transition1 = StateTransitionNode(
        manager, target_state_key="target_state", name="TransitionToThinking"
    )
    planning_node = PlanningNode(name="Planning")
    transition2 = StateTransitionNode(
        manager, target_state_key="target_state", name="TransitionToActing"
    )
    checkpoint = CheckpointNode(
        checkpoint_mgr, manager, name="SaveCheckpoint"
    )
    action_node = ActionNode(name="Action")
    transition3 = StateTransitionNode(
        manager, target_state_key="target_state", name="TransitionToDone"
    )

    flow.add_node(task_node)
    flow.add_node(transition1)
    flow.add_node(planning_node)
    flow.add_node(transition2)
    flow.add_node(checkpoint)
    flow.add_node(action_node)
    flow.add_node(transition3)

    flow.set_start(task_node)

    # Execute
    print("\n🚀 Executing stateful flow...")
    results = flow.execute()

    print(f"\n✅ Final state: {machine.get_current_state()}")
    print(f"📋 Checkpoints created: {len(checkpoint_mgr.list_checkpoints())}")


# ============================================================================
# Main Demo
# ============================================================================


def main():
    """Run all demonstrations."""
    print("\n")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║      StateLayer Demo - Agent State Management             ║")
    print("╚════════════════════════════════════════════════════════════╝")

    # Run examples
    basic_state_machine()
    state_machine_with_conditions()
    state_persistence_demo()
    checkpoints_demo()
    agent_lifecycle_demo()
    flow_integration_demo()

    print("\n" + "=" * 60)
    print("✨ Demo Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("  1. State machines model agent workflows")
    print("  2. Conditions control state transitions")
    print("  3. State persists across sessions")
    print("  4. Checkpoints enable save/restore")
    print("  5. Seamless Flow integration")
    print("\n")


if __name__ == "__main__":
    main()
