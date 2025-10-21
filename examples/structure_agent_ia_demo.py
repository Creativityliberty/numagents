"""
StructureAgentIA Demo - Goal and Task Management

This demo showcases the capabilities of the StructureAgentIA module
for managing goals, tasks, dependencies, and agent planning.

Copyright (c) 2025 Lionel TAGNE. All Rights Reserved.
This software is proprietary and confidential.
"""

from typing import Any, Dict

from num_agents.core import Flow, SharedStore
from num_agents.modules.structure_agent_ia import (
    Goal,
    GoalManager,
    GoalPlannerNode,
    InputParserNode,
    Priority,
    Status,
    Task,
    TaskExecutorNode,
    TaskManager,
)


def demo_1_basic_goals():
    """Demo 1: Basic goal management."""
    print("\n" + "=" * 70)
    print("Demo 1: Basic Goal Management")
    print("=" * 70)

    # Create goal manager
    goal_manager = GoalManager(enable_logging=False)

    # Add goals with different priorities
    print("\n📝 Adding goals...")
    goal1_id = goal_manager.add_goal(
        "Build authentication system", priority=Priority.CRITICAL
    )
    goal2_id = goal_manager.add_goal("Implement user dashboard", priority=Priority.HIGH)
    goal3_id = goal_manager.add_goal("Write documentation", priority=Priority.MEDIUM)
    goal4_id = goal_manager.add_goal("Update UI styling", priority=Priority.LOW)

    print(f"✅ Added {goal_manager.count()} goals")

    # Display goals
    print("\n📋 All goals:")
    for goal in goal_manager.list_all():
        print(f"  - [{goal.priority.value.upper():8}] {goal.description}")

    # Update progress
    print("\n📊 Simulating progress...")
    goal_manager.update_goal_progress(goal1_id, 0.8)
    goal_manager.update_goal_progress(goal2_id, 0.5)
    goal_manager.update_goal_progress(goal3_id, 0.2)

    # Update statuses
    goal_manager.update_goal_status(goal1_id, Status.IN_PROGRESS)
    goal_manager.update_goal_status(goal2_id, Status.IN_PROGRESS)

    # Get statistics
    print("\n📈 Statistics:")
    stats = goal_manager.get_statistics()
    print(f"  Total goals: {stats['total']}")
    print(f"  In progress: {stats['in_progress']}")
    print(f"  Completed: {stats['completed']}")
    print(f"  Completion rate: {stats['completion_rate']:.1%}")
    print(f"  Average progress: {stats['average_progress']:.1%}")

    # Filter by priority
    print("\n🔥 Critical and high priority goals:")
    for priority in [Priority.CRITICAL, Priority.HIGH]:
        goals = goal_manager.get_goals_by_priority(priority)
        for goal in goals:
            print(f"  - [{priority.value.upper()}] {goal.description} ({goal.progress:.0%})")


def demo_2_goal_hierarchies():
    """Demo 2: Hierarchical goals (parent-child relationships)."""
    print("\n" + "=" * 70)
    print("Demo 2: Goal Hierarchies")
    print("=" * 70)

    goal_manager = GoalManager()

    # Create project goal
    print("\n🎯 Creating project goal hierarchy...")
    project_id = goal_manager.add_goal(
        "Launch E-commerce Platform", priority=Priority.CRITICAL
    )

    # Create main components
    backend_id = goal_manager.add_goal(
        "Complete Backend Development",
        priority=Priority.HIGH,
        parent_id=project_id,
    )

    frontend_id = goal_manager.add_goal(
        "Complete Frontend Development",
        priority=Priority.HIGH,
        parent_id=project_id,
    )

    devops_id = goal_manager.add_goal(
        "Setup DevOps Pipeline",
        priority=Priority.MEDIUM,
        parent_id=project_id,
    )

    # Backend sub-goals
    api_id = goal_manager.add_goal(
        "Build REST API",
        priority=Priority.HIGH,
        parent_id=backend_id,
    )

    db_id = goal_manager.add_goal(
        "Design Database Schema",
        priority=Priority.CRITICAL,
        parent_id=backend_id,
    )

    # Frontend sub-goals
    ui_id = goal_manager.add_goal(
        "Implement UI Components",
        priority=Priority.HIGH,
        parent_id=frontend_id,
    )

    # Display hierarchy
    print("\n📊 Goal Hierarchy:")
    root_goals = goal_manager.get_root_goals()
    for root in root_goals:
        print(f"\n🎯 {root.description}")
        sub_goals = goal_manager.get_sub_goals(root.id)
        for sub in sub_goals:
            print(f"  ├─ {sub.description}")
            sub_sub_goals = goal_manager.get_sub_goals(sub.id)
            for sub_sub in sub_sub_goals:
                print(f"  │  └─ {sub_sub.description}")

    # Simulate progress
    print("\n📊 Updating progress...")
    goal_manager.update_goal_progress(db_id, 1.0)  # Completed
    goal_manager.update_goal_progress(api_id, 0.6)
    goal_manager.update_goal_progress(ui_id, 0.4)

    print("\n✅ Progress updated:")
    for sub in goal_manager.get_sub_goals(backend_id):
        print(f"  - {sub.description}: {sub.progress:.0%}")


def demo_3_task_dependencies():
    """Demo 3: Task dependencies and scheduling."""
    print("\n" + "=" * 70)
    print("Demo 3: Task Dependencies and Scheduling")
    print("=" * 70)

    task_manager = TaskManager(enable_logging=False)

    # Create a data processing pipeline
    print("\n🔧 Creating data processing pipeline...")

    task1_id = task_manager.add_task(
        "Download data from API",
        priority=Priority.HIGH,
        metadata={"duration_min": 5},
    )

    task2_id = task_manager.add_task(
        "Clean and validate data",
        priority=Priority.HIGH,
        dependencies=[task1_id],
        metadata={"duration_min": 10},
    )

    task3_id = task_manager.add_task(
        "Transform data format",
        priority=Priority.HIGH,
        dependencies=[task2_id],
        metadata={"duration_min": 8},
    )

    task4_id = task_manager.add_task(
        "Generate analytics",
        priority=Priority.MEDIUM,
        dependencies=[task3_id],
        metadata={"duration_min": 15},
    )

    task5_id = task_manager.add_task(
        "Save to database",
        priority=Priority.HIGH,
        dependencies=[task3_id],
        metadata={"duration_min": 5},
    )

    task6_id = task_manager.add_task(
        "Send notification",
        priority=Priority.LOW,
        dependencies=[task4_id, task5_id],
        metadata={"duration_min": 2},
    )

    print(f"✅ Created {task_manager.count()} tasks")

    # Show execution order
    print("\n📋 Task Execution Simulation:")
    print("-" * 50)

    step = 1
    while True:
        # Get next task
        next_task = task_manager.get_next_task()
        if not next_task:
            break

        # Show task
        duration = next_task.metadata.get("duration_min", 0)
        print(f"Step {step}: {next_task.description} (~{duration}min)")

        # Mark as completed
        task_manager.update_task_status(next_task.id, Status.COMPLETED)
        step += 1

    # Show blocked tasks during execution
    print("\n🔒 Demonstrating blocked tasks:")
    task_manager.clear()

    # Recreate with one intentionally incomplete
    task1_id = task_manager.add_task("Task A")
    task2_id = task_manager.add_task("Task B", dependencies=[task1_id])
    task3_id = task_manager.add_task("Task C", dependencies=[task1_id])

    blocked = task_manager.get_blocked_tasks()
    print(f"  Blocked tasks: {len(blocked)}")
    for task in blocked:
        deps = task.dependencies
        print(f"    - {task.description} (waiting for {len(deps)} task(s))")


def demo_4_input_parsing_and_planning():
    """Demo 4: Input parsing and automatic planning."""
    print("\n" + "=" * 70)
    print("Demo 4: Input Parsing and Planning")
    print("=" * 70)

    goal_manager = GoalManager()
    task_manager = TaskManager()

    # Create nodes
    parser = InputParserNode()
    planner = GoalPlannerNode(goal_manager=goal_manager, task_manager=task_manager)

    # Create flow
    flow = Flow(nodes=[parser, planner])

    # Test different inputs
    test_inputs = [
        "Create a new feature for user authentication",
        "Create critical priority task for database backup",
        "Add high priority goal to implement search functionality",
    ]

    print("\n🤖 Testing natural language input parsing...")
    print("-" * 50)

    for user_input in test_inputs:
        print(f"\n💬 User: '{user_input}'")

        # Execute flow
        result = flow.execute(initial_data={"user_input": user_input})

        # Get parsed data
        parsed = flow.shared.get("parsed_input")
        print(f"   Intent: {parsed['intent']}")
        print(f"   Keywords: {', '.join(parsed['keywords'][:5])}")

        # Get plan
        plan = flow.shared.get("plan")
        if "goal_id" in plan:
            goal = goal_manager.get_goal(plan["goal_id"])
            print(f"   ✅ Created goal: {goal.description}")
            print(f"   ✅ Created {len(plan['task_ids'])} task(s)")

    # Show created goals and tasks
    print(f"\n📊 Summary:")
    print(f"  Total goals created: {goal_manager.count()}")
    print(f"  Total tasks created: {task_manager.count()}")


def demo_5_task_execution():
    """Demo 5: Task execution with custom executor."""
    print("\n" + "=" * 70)
    print("Demo 5: Task Execution with Custom Executor")
    print("=" * 70)

    task_manager = TaskManager()

    # Create tasks
    task1_id = task_manager.add_task("Send welcome email")
    task2_id = task_manager.add_task("Update user profile")
    task3_id = task_manager.add_task("Generate analytics report")

    print(f"\n📋 Created {task_manager.count()} tasks")

    # Custom executor function
    def custom_executor(task: Task, shared: SharedStore) -> Dict[str, Any]:
        """Simulate task execution."""
        print(f"   ⚙️  Executing: {task.description}...")

        # Simulate work based on task description
        if "email" in task.description.lower():
            return {
                "status": "sent",
                "recipient": "user@example.com",
                "message_id": "msg-12345",
            }
        elif "profile" in task.description.lower():
            return {"status": "updated", "fields_changed": ["name", "avatar"]}
        elif "analytics" in task.description.lower():
            return {
                "status": "generated",
                "metrics": {"users": 1250, "revenue": 50000},
            }
        else:
            return {"status": "completed"}

    # Create executor node
    executor_node = TaskExecutorNode(task_manager=task_manager, executor=custom_executor)

    # Execute tasks
    print("\n🚀 Executing tasks...")
    print("-" * 50)

    for task_id in [task1_id, task2_id, task3_id]:
        # Create shared store
        shared = SharedStore()
        shared.set("task_id", task_id)

        # Execute
        result = executor_node.exec(shared)

        # Show result
        task = task_manager.get_task(task_id)
        print(f"\n✅ {task.description}")
        print(f"   Status: {task.status.value}")
        print(f"   Result: {task.result}")

    # Show statistics
    print("\n📊 Execution Statistics:")
    stats = task_manager.get_statistics()
    print(f"  Total tasks: {stats['total']}")
    print(f"  Completed: {stats['completed']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Completion rate: {stats['completion_rate']:.0%}")


def demo_6_persistence():
    """Demo 6: Saving and loading goals/tasks."""
    print("\n" + "=" * 70)
    print("Demo 6: Persistence - Save and Load")
    print("=" * 70)

    import json
    import tempfile

    # Create and populate managers
    goal_manager = GoalManager()
    task_manager = TaskManager()

    print("\n📝 Creating sample data...")

    # Add goals
    goal1_id = goal_manager.add_goal("Complete Phase 1", priority=Priority.HIGH)
    goal2_id = goal_manager.add_goal("Complete Phase 2", priority=Priority.MEDIUM)

    # Update progress
    goal_manager.update_goal_progress(goal1_id, 0.75)
    goal_manager.update_goal_progress(goal2_id, 0.25)

    # Add tasks
    task1_id = task_manager.add_task("Task A", goal_id=goal1_id, priority=Priority.HIGH)
    task2_id = task_manager.add_task(
        "Task B", goal_id=goal1_id, dependencies=[task1_id]
    )

    print(f"  Created {goal_manager.count()} goals")
    print(f"  Created {task_manager.count()} tasks")

    # Save to temporary files
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        goals_file = f.name
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        tasks_file = f.name

    print(f"\n💾 Saving to files...")
    goal_manager.save_to_file(goals_file)
    task_manager.export_to_dict()  # Export to dict
    print(f"  Goals saved to: {goals_file}")
    print(f"  Tasks exported to dict")

    # Load into new managers
    print(f"\n📂 Loading from files...")
    new_goal_manager = GoalManager()
    new_goal_manager.load_from_file(goals_file)

    print(f"  ✅ Loaded {new_goal_manager.count()} goals")

    # Show loaded data
    print("\n📋 Loaded goals:")
    for goal in new_goal_manager.list_all():
        print(f"  - {goal.description}: {goal.progress:.0%} complete")

    # Show export format
    print("\n📄 Export format (JSON):")
    data = new_goal_manager.export_to_dict()
    print(json.dumps(data, indent=2)[:300] + "...")

    # Cleanup
    import os

    os.unlink(goals_file)


def demo_7_complete_workflow():
    """Demo 7: Complete workflow - parsing, planning, and execution."""
    print("\n" + "=" * 70)
    print("Demo 7: Complete Agent Workflow")
    print("=" * 70)

    goal_manager = GoalManager()
    task_manager = TaskManager()

    # Custom executor
    def workflow_executor(task: Task, shared: SharedStore) -> Dict[str, Any]:
        """Execute task based on description."""
        description = task.description.lower()

        if "authentication" in description:
            return {
                "status": "implemented",
                "features": ["login", "logout", "password_reset"],
                "tests_passed": True,
            }
        else:
            return {"status": "completed"}

    # Create workflow nodes
    parser = InputParserNode()
    planner = GoalPlannerNode(goal_manager, task_manager)

    # Build flow
    flow = Flow(nodes=[parser, planner])

    print("\n🤖 Starting agent workflow...")
    print("-" * 50)

    # User input
    user_input = "Create a critical priority feature for user authentication"
    print(f"\n💬 User request: '{user_input}'")

    # Parse and plan
    print("\n1️⃣  Parsing input...")
    result = flow.execute(initial_data={"user_input": user_input})

    parsed = flow.shared.get("parsed_input")
    print(f"   Intent: {parsed['intent']}")
    print(f"   Keywords: {', '.join(parsed['keywords'])}")

    plan = flow.shared.get("plan")
    print(f"\n2️⃣  Created plan:")
    print(f"   Goals: {len([plan.get('goal_id')]) if plan.get('goal_id') else 0}")
    print(f"   Tasks: {len(plan.get('task_ids', []))}")

    # Execute tasks
    print(f"\n3️⃣  Executing tasks...")
    executor = TaskExecutorNode(task_manager, executor=workflow_executor)

    for task_id in plan.get("task_ids", []):
        shared = SharedStore()
        shared.set("task_id", task_id)

        result = executor.exec(shared)
        task = task_manager.get_task(task_id)

        print(f"   ✅ {task.description}")
        if task.result:
            print(f"      Result: {task.result}")

    # Show final statistics
    print(f"\n📊 Final Statistics:")
    goal_stats = goal_manager.get_statistics()
    task_stats = task_manager.get_statistics()

    print(f"   Goals: {goal_stats['total']} total, {goal_stats['completed']} completed")
    print(f"   Tasks: {task_stats['total']} total, {task_stats['completed']} completed")

    print("\n✨ Workflow completed successfully!")


def main():
    """Run all demos."""
    print("=" * 70)
    print("STRUCTUREAGENT IA - GOAL AND TASK MANAGEMENT DEMO")
    print("=" * 70)
    print("\nThis demo showcases:")
    print("  1. Basic goal management")
    print("  2. Hierarchical goals (parent-child)")
    print("  3. Task dependencies and scheduling")
    print("  4. Natural language input parsing and planning")
    print("  5. Task execution with custom executors")
    print("  6. Data persistence (save/load)")
    print("  7. Complete agent workflow")

    demos = [
        demo_1_basic_goals,
        demo_2_goal_hierarchies,
        demo_3_task_dependencies,
        demo_4_input_parsing_and_planning,
        demo_5_task_execution,
        demo_6_persistence,
        demo_7_complete_workflow,
    ]

    for demo in demos:
        demo()
        input("\n\n▶️  Press Enter to continue to next demo...")

    print("\n" + "=" * 70)
    print("All demos completed!")
    print("=" * 70)
    print("\nFor more information, see:")
    print("  - Documentation: docs/structure_agent_ia_guide.md")
    print("  - Tests: tests/test_structure_agent_ia.py")
    print("  - Module: num_agents/modules/structure_agent_ia.py")


if __name__ == "__main__":
    main()
