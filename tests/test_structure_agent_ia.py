"""
Tests for StructureAgentIA module.

Copyright (c) 2025 Lionel TAGNE. All Rights Reserved.
This software is proprietary and confidential.
"""

import json
import os
import tempfile
from typing import Any, Dict

import pytest

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


class TestPriority:
    """Test Priority enum."""

    def test_priority_values(self) -> None:
        """Test priority enum values."""
        assert Priority.LOW == "low"
        assert Priority.MEDIUM == "medium"
        assert Priority.HIGH == "high"
        assert Priority.CRITICAL == "critical"

    def test_priority_comparison(self) -> None:
        """Test priority comparisons."""
        priorities = [Priority.LOW, Priority.MEDIUM, Priority.HIGH, Priority.CRITICAL]
        assert len(priorities) == 4


class TestStatus:
    """Test Status enum."""

    def test_status_values(self) -> None:
        """Test status enum values."""
        assert Status.PENDING == "pending"
        assert Status.IN_PROGRESS == "in_progress"
        assert Status.COMPLETED == "completed"
        assert Status.FAILED == "failed"
        assert Status.BLOCKED == "blocked"
        assert Status.CANCELLED == "cancelled"


class TestGoal:
    """Test Goal class."""

    def test_create_goal(self) -> None:
        """Test goal creation."""
        goal = Goal(description="Test goal", priority=Priority.HIGH)

        assert goal.description == "Test goal"
        assert goal.priority == Priority.HIGH
        assert goal.status == Status.PENDING
        assert goal.progress == 0.0
        assert goal.id is not None
        assert goal.parent_id is None
        assert goal.metadata == {}

    def test_create_goal_with_parent(self) -> None:
        """Test creating a sub-goal."""
        parent_goal = Goal(description="Parent")
        child_goal = Goal(description="Child", parent_id=parent_goal.id)

        assert child_goal.parent_id == parent_goal.id

    def test_create_goal_with_metadata(self) -> None:
        """Test goal creation with metadata."""
        metadata = {"category": "development", "team": "backend"}
        goal = Goal(description="Test", metadata=metadata)

        assert goal.metadata == metadata

    def test_update_status(self) -> None:
        """Test updating goal status."""
        goal = Goal(description="Test")
        assert goal.status == Status.PENDING

        goal.status = Status.IN_PROGRESS
        assert goal.status == Status.IN_PROGRESS

        goal.status = Status.COMPLETED
        assert goal.status == Status.COMPLETED

    def test_update_progress(self) -> None:
        """Test updating goal progress."""
        goal = Goal(description="Test")
        assert goal.progress == 0.0

        goal.progress = 0.5
        assert goal.progress == 0.5

        goal.progress = 1.0
        assert goal.progress == 1.0

    def test_goal_serialization(self) -> None:
        """Test goal to_dict and from_dict."""
        goal = Goal(
            description="Test goal",
            priority=Priority.HIGH,
            metadata={"key": "value"},
            id="test-id",
        )
        goal.status = Status.IN_PROGRESS
        goal.progress = 0.7

        data = goal.to_dict()

        assert data["id"] == "test-id"
        assert data["description"] == "Test goal"
        assert data["priority"] == "high"
        assert data["status"] == "in_progress"
        assert data["progress"] == 0.7
        assert data["metadata"] == {"key": "value"}

        # Test from_dict
        restored_goal = Goal.from_dict(data)
        assert restored_goal.id == goal.id
        assert restored_goal.description == goal.description
        assert restored_goal.priority == goal.priority
        assert restored_goal.status == goal.status
        assert restored_goal.progress == goal.progress


class TestTask:
    """Test Task class."""

    def test_create_task(self) -> None:
        """Test task creation."""
        task = Task(description="Test task", priority=Priority.MEDIUM)

        assert task.description == "Test task"
        assert task.priority == Priority.MEDIUM
        assert task.status == Status.PENDING
        assert task.goal_id is None
        assert task.dependencies == []
        assert task.result is None
        assert task.error is None

    def test_create_task_with_goal(self) -> None:
        """Test creating task associated with a goal."""
        goal_id = "goal-123"
        task = Task(description="Test", goal_id=goal_id)

        assert task.goal_id == goal_id

    def test_create_task_with_dependencies(self) -> None:
        """Test creating task with dependencies."""
        task = Task(description="Test", dependencies=["task-1", "task-2"])

        assert len(task.dependencies) == 2
        assert "task-1" in task.dependencies
        assert "task-2" in task.dependencies

    def test_update_status(self) -> None:
        """Test updating task status."""
        task = Task(description="Test")

        task.status = Status.IN_PROGRESS
        assert task.status == Status.IN_PROGRESS

        task.status = Status.COMPLETED
        assert task.status == Status.COMPLETED

    def test_task_with_result(self) -> None:
        """Test task with execution result."""
        task = Task(description="Test")
        task.result = {"output": "success", "data": [1, 2, 3]}

        assert task.result["output"] == "success"
        assert task.result["data"] == [1, 2, 3]

    def test_task_with_error(self) -> None:
        """Test task with error."""
        task = Task(description="Test")
        task.error = "Something went wrong"
        task.status = Status.FAILED

        assert task.error == "Something went wrong"
        assert task.status == Status.FAILED

    def test_task_serialization(self) -> None:
        """Test task to_dict and from_dict."""
        task = Task(
            description="Test task",
            goal_id="goal-123",
            priority=Priority.HIGH,
            dependencies=["task-1"],
            id="test-id",
        )
        task.status = Status.COMPLETED
        task.result = {"success": True}

        data = task.to_dict()

        assert data["id"] == "test-id"
        assert data["description"] == "Test task"
        assert data["goal_id"] == "goal-123"
        assert data["priority"] == "high"
        assert data["status"] == "completed"
        assert data["dependencies"] == ["task-1"]
        assert data["result"] == {"success": True}

        # Test from_dict
        restored_task = Task.from_dict(data)
        assert restored_task.id == task.id
        assert restored_task.description == task.description
        assert restored_task.goal_id == task.goal_id
        assert restored_task.priority == task.priority
        assert restored_task.status == task.status
        assert restored_task.dependencies == task.dependencies


class TestGoalManager:
    """Test GoalManager."""

    @pytest.fixture
    def goal_manager(self) -> GoalManager:
        """Create a goal manager."""
        return GoalManager()

    def test_add_goal(self, goal_manager: GoalManager) -> None:
        """Test adding a goal."""
        goal_id = goal_manager.add_goal("Test goal", priority=Priority.HIGH)

        assert goal_id is not None
        assert goal_manager.count() == 1

        goal = goal_manager.get_goal(goal_id)
        assert goal is not None
        assert goal.description == "Test goal"
        assert goal.priority == Priority.HIGH

    def test_add_goal_with_metadata(self, goal_manager: GoalManager) -> None:
        """Test adding goal with metadata."""
        metadata = {"category": "test"}
        goal_id = goal_manager.add_goal("Test", metadata=metadata)

        goal = goal_manager.get_goal(goal_id)
        assert goal is not None
        assert goal.metadata == metadata

    def test_add_sub_goal(self, goal_manager: GoalManager) -> None:
        """Test adding a sub-goal."""
        parent_id = goal_manager.add_goal("Parent goal")
        child_id = goal_manager.add_goal("Child goal", parent_id=parent_id)

        child = goal_manager.get_goal(child_id)
        assert child is not None
        assert child.parent_id == parent_id

        # Test get_sub_goals
        sub_goals = goal_manager.get_sub_goals(parent_id)
        assert len(sub_goals) == 1
        assert sub_goals[0].id == child_id

    def test_update_goal_status(self, goal_manager: GoalManager) -> None:
        """Test updating goal status."""
        goal_id = goal_manager.add_goal("Test")

        goal_manager.update_goal_status(goal_id, Status.IN_PROGRESS)
        goal = goal_manager.get_goal(goal_id)
        assert goal is not None
        assert goal.status == Status.IN_PROGRESS

    def test_update_goal_progress(self, goal_manager: GoalManager) -> None:
        """Test updating goal progress."""
        goal_id = goal_manager.add_goal("Test")

        goal_manager.update_goal_progress(goal_id, 0.5)
        goal = goal_manager.get_goal(goal_id)
        assert goal is not None
        assert goal.progress == 0.5

    def test_delete_goal(self, goal_manager: GoalManager) -> None:
        """Test deleting a goal."""
        goal_id = goal_manager.add_goal("Test")
        assert goal_manager.count() == 1

        deleted = goal_manager.delete_goal(goal_id)
        assert deleted is True
        assert goal_manager.count() == 0
        assert goal_manager.get_goal(goal_id) is None

    def test_get_goals_by_status(self, goal_manager: GoalManager) -> None:
        """Test filtering goals by status."""
        goal_manager.add_goal("Goal 1", priority=Priority.HIGH)
        goal_id_2 = goal_manager.add_goal("Goal 2", priority=Priority.LOW)
        goal_manager.update_goal_status(goal_id_2, Status.COMPLETED)

        pending_goals = goal_manager.get_goals_by_status(Status.PENDING)
        assert len(pending_goals) == 1
        assert pending_goals[0].description == "Goal 1"

        completed_goals = goal_manager.get_goals_by_status(Status.COMPLETED)
        assert len(completed_goals) == 1
        assert completed_goals[0].description == "Goal 2"

    def test_get_goals_by_priority(self, goal_manager: GoalManager) -> None:
        """Test filtering goals by priority."""
        goal_manager.add_goal("Low priority", priority=Priority.LOW)
        goal_manager.add_goal("High priority", priority=Priority.HIGH)
        goal_manager.add_goal("Critical priority", priority=Priority.CRITICAL)

        high_goals = goal_manager.get_goals_by_priority(Priority.HIGH)
        assert len(high_goals) == 1
        assert high_goals[0].description == "High priority"

    def test_get_root_goals(self, goal_manager: GoalManager) -> None:
        """Test getting root goals (no parent)."""
        root_id_1 = goal_manager.add_goal("Root 1")
        root_id_2 = goal_manager.add_goal("Root 2")
        goal_manager.add_goal("Child", parent_id=root_id_1)

        root_goals = goal_manager.get_root_goals()
        assert len(root_goals) == 2
        assert all(g.parent_id is None for g in root_goals)

    def test_list_all_goals(self, goal_manager: GoalManager) -> None:
        """Test listing all goals."""
        goal_manager.add_goal("Goal 1")
        goal_manager.add_goal("Goal 2")
        goal_manager.add_goal("Goal 3")

        all_goals = goal_manager.list_all()
        assert len(all_goals) == 3

    def test_clear_goals(self, goal_manager: GoalManager) -> None:
        """Test clearing all goals."""
        goal_manager.add_goal("Goal 1")
        goal_manager.add_goal("Goal 2")
        assert goal_manager.count() == 2

        goal_manager.clear()
        assert goal_manager.count() == 0

    def test_get_statistics(self, goal_manager: GoalManager) -> None:
        """Test getting goal statistics."""
        goal_manager.add_goal("Goal 1", priority=Priority.HIGH)
        goal_id_2 = goal_manager.add_goal("Goal 2", priority=Priority.LOW)
        goal_manager.update_goal_status(goal_id_2, Status.COMPLETED)

        stats = goal_manager.get_statistics()

        assert stats["total"] == 2
        assert stats["by_status"][Status.PENDING] == 1
        assert stats["by_status"][Status.COMPLETED] == 1
        assert stats["by_priority"][Priority.HIGH] == 1
        assert stats["by_priority"][Priority.LOW] == 1
        assert stats["completion_rate"] == 0.5

    def test_export_import(self, goal_manager: GoalManager) -> None:
        """Test export and import."""
        goal_manager.add_goal("Goal 1", priority=Priority.HIGH, metadata={"key": "value"})
        goal_manager.add_goal("Goal 2", priority=Priority.LOW)

        # Export
        data = goal_manager.export_to_dict()
        assert "goals" in data
        assert len(data["goals"]) == 2

        # Import
        new_manager = GoalManager()
        new_manager.import_from_dict(data)
        assert new_manager.count() == 2

    def test_save_load_file(self, goal_manager: GoalManager) -> None:
        """Test saving and loading from file."""
        goal_manager.add_goal("Goal 1")
        goal_manager.add_goal("Goal 2")

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_path = f.name

        try:
            goal_manager.save_to_file(temp_path)

            new_manager = GoalManager()
            new_manager.load_from_file(temp_path)
            assert new_manager.count() == 2

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestTaskManager:
    """Test TaskManager."""

    @pytest.fixture
    def task_manager(self) -> TaskManager:
        """Create a task manager."""
        return TaskManager()

    def test_add_task(self, task_manager: TaskManager) -> None:
        """Test adding a task."""
        task_id = task_manager.add_task("Test task", priority=Priority.HIGH)

        assert task_id is not None
        assert task_manager.count() == 1

        task = task_manager.get_task(task_id)
        assert task is not None
        assert task.description == "Test task"
        assert task.priority == Priority.HIGH

    def test_add_task_with_dependencies(self, task_manager: TaskManager) -> None:
        """Test adding task with dependencies."""
        task1_id = task_manager.add_task("Task 1")
        task2_id = task_manager.add_task("Task 2", dependencies=[task1_id])

        task2 = task_manager.get_task(task2_id)
        assert task2 is not None
        assert task1_id in task2.dependencies

    def test_add_task_with_goal(self, task_manager: TaskManager) -> None:
        """Test adding task with goal association."""
        goal_id = "goal-123"
        task_id = task_manager.add_task("Test", goal_id=goal_id)

        task = task_manager.get_task(task_id)
        assert task is not None
        assert task.goal_id == goal_id

    def test_update_task_status(self, task_manager: TaskManager) -> None:
        """Test updating task status."""
        task_id = task_manager.add_task("Test")

        task_manager.update_task_status(task_id, Status.IN_PROGRESS)
        task = task_manager.get_task(task_id)
        assert task is not None
        assert task.status == Status.IN_PROGRESS

    def test_update_task_result(self, task_manager: TaskManager) -> None:
        """Test updating task result."""
        task_id = task_manager.add_task("Test")
        result = {"output": "success"}

        task_manager.update_task_result(task_id, result)
        task = task_manager.get_task(task_id)
        assert task is not None
        assert task.result == result

    def test_is_task_ready(self, task_manager: TaskManager) -> None:
        """Test checking if task is ready to execute."""
        task1_id = task_manager.add_task("Task 1")
        task2_id = task_manager.add_task("Task 2", dependencies=[task1_id])

        # Task 2 should not be ready (Task 1 is pending)
        assert task_manager.is_task_ready(task2_id) is False

        # Complete Task 1
        task_manager.update_task_status(task1_id, Status.COMPLETED)

        # Now Task 2 should be ready
        assert task_manager.is_task_ready(task2_id) is True

    def test_get_next_task(self, task_manager: TaskManager) -> None:
        """Test getting next task to execute."""
        # Add tasks with different priorities
        task_low_id = task_manager.add_task("Low priority", priority=Priority.LOW)
        task_high_id = task_manager.add_task("High priority", priority=Priority.HIGH)
        task_critical_id = task_manager.add_task("Critical priority", priority=Priority.CRITICAL)

        # Should return highest priority ready task
        next_task = task_manager.get_next_task()
        assert next_task is not None
        assert next_task.priority == Priority.CRITICAL

        # Mark critical as in progress
        task_manager.update_task_status(task_critical_id, Status.IN_PROGRESS)

        # Should return high priority now
        next_task = task_manager.get_next_task()
        assert next_task is not None
        assert next_task.priority == Priority.HIGH

    def test_get_next_task_with_dependencies(self, task_manager: TaskManager) -> None:
        """Test next task respects dependencies."""
        task1_id = task_manager.add_task("Task 1", priority=Priority.LOW)
        task2_id = task_manager.add_task(
            "Task 2", priority=Priority.HIGH, dependencies=[task1_id]
        )

        # Should get Task 1 first (Task 2 is blocked)
        next_task = task_manager.get_next_task()
        assert next_task is not None
        assert next_task.id == task1_id

    def test_get_blocked_tasks(self, task_manager: TaskManager) -> None:
        """Test getting blocked tasks."""
        task1_id = task_manager.add_task("Task 1")
        task2_id = task_manager.add_task("Task 2", dependencies=[task1_id])

        blocked = task_manager.get_blocked_tasks()
        assert len(blocked) == 1
        assert blocked[0].id == task2_id

    def test_get_tasks_by_status(self, task_manager: TaskManager) -> None:
        """Test filtering tasks by status."""
        task_manager.add_task("Task 1")
        task2_id = task_manager.add_task("Task 2")
        task_manager.update_task_status(task2_id, Status.COMPLETED)

        pending = task_manager.get_tasks_by_status(Status.PENDING)
        completed = task_manager.get_tasks_by_status(Status.COMPLETED)

        assert len(pending) == 1
        assert len(completed) == 1

    def test_get_tasks_by_goal(self, task_manager: TaskManager) -> None:
        """Test filtering tasks by goal."""
        goal_id = "goal-123"
        task_manager.add_task("Task 1", goal_id=goal_id)
        task_manager.add_task("Task 2", goal_id=goal_id)
        task_manager.add_task("Task 3", goal_id="other-goal")

        tasks = task_manager.get_tasks_by_goal(goal_id)
        assert len(tasks) == 2

    def test_delete_task(self, task_manager: TaskManager) -> None:
        """Test deleting a task."""
        task_id = task_manager.add_task("Test")
        assert task_manager.count() == 1

        deleted = task_manager.delete_task(task_id)
        assert deleted is True
        assert task_manager.count() == 0

    def test_clear_tasks(self, task_manager: TaskManager) -> None:
        """Test clearing all tasks."""
        task_manager.add_task("Task 1")
        task_manager.add_task("Task 2")
        assert task_manager.count() == 2

        task_manager.clear()
        assert task_manager.count() == 0

    def test_get_statistics(self, task_manager: TaskManager) -> None:
        """Test getting task statistics."""
        task_manager.add_task("Task 1", priority=Priority.HIGH)
        task2_id = task_manager.add_task("Task 2", priority=Priority.LOW)
        task_manager.update_task_status(task2_id, Status.COMPLETED)

        stats = task_manager.get_statistics()

        assert stats["total"] == 2
        assert stats["by_status"][Status.PENDING] == 1
        assert stats["by_status"][Status.COMPLETED] == 1
        assert stats["completion_rate"] == 0.5

    def test_export_import(self, task_manager: TaskManager) -> None:
        """Test export and import."""
        task_manager.add_task("Task 1", priority=Priority.HIGH)
        task_manager.add_task("Task 2", priority=Priority.LOW)

        data = task_manager.export_to_dict()
        assert "tasks" in data
        assert len(data["tasks"]) == 2

        new_manager = TaskManager()
        new_manager.import_from_dict(data)
        assert new_manager.count() == 2


class TestInputParserNode:
    """Test InputParserNode."""

    def test_parse_simple_input(self) -> None:
        """Test parsing simple user input."""
        node = InputParserNode()
        shared = SharedStore()
        shared.set("user_input", "Create a new feature for user authentication")

        result = node.exec(shared)

        parsed = shared.get("parsed_input")
        assert parsed is not None
        assert "intent" in parsed
        assert "entities" in parsed
        assert "keywords" in parsed

    def test_custom_keys(self) -> None:
        """Test custom input/output keys."""
        node = InputParserNode(input_key="my_input", output_key="my_output")
        shared = SharedStore()
        shared.set("my_input", "Test input")

        node.exec(shared)

        assert shared.has("my_output")


class TestGoalPlannerNode:
    """Test GoalPlannerNode."""

    def test_create_plan(self) -> None:
        """Test creating a plan from parsed input."""
        goal_manager = GoalManager()
        task_manager = TaskManager()
        node = GoalPlannerNode(goal_manager=goal_manager, task_manager=task_manager)

        shared = SharedStore()
        shared.set(
            "parsed_input",
            {
                "intent": "create_feature",
                "entities": ["authentication", "user"],
                "keywords": ["create", "feature", "authentication"],
            },
        )

        result = node.exec(shared)

        assert result["goals_created"] >= 1
        assert result["tasks_created"] >= 1
        assert goal_manager.count() >= 1
        assert task_manager.count() >= 1

    def test_custom_keys(self) -> None:
        """Test custom input/output keys."""
        goal_manager = GoalManager()
        task_manager = TaskManager()
        node = GoalPlannerNode(
            goal_manager=goal_manager,
            task_manager=task_manager,
            input_key="my_parsed",
            output_key="my_plan",
        )

        shared = SharedStore()
        shared.set(
            "my_parsed",
            {"intent": "test", "entities": ["test"], "keywords": ["test"]},
        )

        node.exec(shared)

        assert shared.has("my_plan")


class TestTaskExecutorNode:
    """Test TaskExecutorNode."""

    def test_execute_task(self) -> None:
        """Test executing a task."""
        task_manager = TaskManager()
        task_id = task_manager.add_task("Test task")

        def executor(task: Task, shared: SharedStore) -> Dict[str, Any]:
            return {"success": True, "output": "Task executed"}

        node = TaskExecutorNode(task_manager=task_manager, executor=executor)
        shared = SharedStore()
        shared.set("task_id", task_id)

        result = node.exec(shared)

        assert result["executed"] is True
        assert result["status"] == Status.COMPLETED

        # Check task was updated
        task = task_manager.get_task(task_id)
        assert task is not None
        assert task.status == Status.COMPLETED
        assert task.result == {"success": True, "output": "Task executed"}

    def test_execute_task_failure(self) -> None:
        """Test executing a task that fails."""
        task_manager = TaskManager()
        task_id = task_manager.add_task("Test task")

        def executor(task: Task, shared: SharedStore) -> Dict[str, Any]:
            raise ValueError("Task failed")

        node = TaskExecutorNode(task_manager=task_manager, executor=executor)
        shared = SharedStore()
        shared.set("task_id", task_id)

        result = node.exec(shared)

        assert result["executed"] is False
        assert result["status"] == Status.FAILED

        # Check task was marked as failed
        task = task_manager.get_task(task_id)
        assert task is not None
        assert task.status == Status.FAILED
        assert task.error is not None

    def test_in_flow(self) -> None:
        """Test nodes in a complete flow."""
        goal_manager = GoalManager()
        task_manager = TaskManager()

        # Create nodes
        parser_node = InputParserNode()
        planner_node = GoalPlannerNode(
            goal_manager=goal_manager, task_manager=task_manager
        )

        flow = Flow(nodes=[parser_node, planner_node])

        # Execute flow
        initial_data = {"user_input": "Create a feature for user authentication"}
        results = flow.execute(initial_data=initial_data)

        # Check execution
        assert "InputParserNode" in results
        assert "GoalPlannerNode" in results
        assert goal_manager.count() >= 1
        assert task_manager.count() >= 1
