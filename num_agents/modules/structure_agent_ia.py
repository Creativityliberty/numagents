"""
StructureAgentIA - Agent Structure and Intelligence

This module provides goal and task management for intelligent agents,
including planning, execution tracking, and hierarchical organization.

Copyright (c) 2025 Lionel TAGNE. All Rights Reserved.
This software is proprietary and confidential.
"""

import json
import time
import uuid
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from num_agents.core import Node, SharedStore
from num_agents.exceptions import NumAgentsException
from num_agents.logging_config import get_logger

logger = get_logger(__name__)


class StructureAgentException(NumAgentsException):
    """Base exception for StructureAgentIA errors."""

    pass


class GoalError(StructureAgentException):
    """Exception raised when goal operations fail."""

    pass


class TaskError(StructureAgentException):
    """Exception raised when task operations fail."""

    pass


# ============================================================================
# Enums
# ============================================================================


class Priority(str, Enum):
    """Priority levels for goals and tasks."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    def __lt__(self, other: "Priority") -> bool:
        """Enable priority comparison."""
        priority_order = {
            Priority.LOW: 0,
            Priority.MEDIUM: 1,
            Priority.HIGH: 2,
            Priority.CRITICAL: 3,
        }
        return priority_order[self] < priority_order[other]

    def __le__(self, other: "Priority") -> bool:
        """Enable priority comparison."""
        return self < other or self == other


class Status(str, Enum):
    """Status for goals and tasks."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


# ============================================================================
# Goal Class
# ============================================================================


class Goal:
    """
    Represents a high-level objective for an agent.

    Goals can have sub-goals and associated tasks.
    """

    def __init__(
        self,
        description: str,
        priority: Priority = Priority.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None,
        parent_id: Optional[str] = None,
    ) -> None:
        """
        Initialize a goal.

        Args:
            description: Goal description
            priority: Goal priority level
            metadata: Optional metadata dictionary
            id: Optional custom ID (auto-generated if not provided)
            parent_id: Optional ID of parent goal (for hierarchies)
        """
        self.id = id or str(uuid.uuid4())
        self.description = description
        self.priority = priority
        self.status = Status.PENDING
        self.metadata = metadata or {}
        self.parent_id = parent_id
        self.created_at = time.time()
        self.updated_at = self.created_at
        self.completed_at: Optional[float] = None
        self.progress: float = 0.0  # 0.0 to 1.0

    def update_status(self, status: Status) -> None:
        """Update goal status."""
        self.status = status
        self.updated_at = time.time()
        if status == Status.COMPLETED:
            self.completed_at = time.time()
            self.progress = 1.0

    def update_progress(self, progress: float) -> None:
        """
        Update goal progress.

        Args:
            progress: Progress value between 0.0 and 1.0
        """
        self.progress = max(0.0, min(1.0, progress))
        self.updated_at = time.time()

        # Auto-complete if progress reaches 100%
        if self.progress >= 1.0 and self.status != Status.COMPLETED:
            self.update_status(Status.COMPLETED)

    def to_dict(self) -> Dict[str, Any]:
        """Convert goal to dictionary."""
        return {
            "id": self.id,
            "description": self.description,
            "priority": self.priority.value,
            "status": self.status.value,
            "metadata": self.metadata,
            "parent_id": self.parent_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "progress": self.progress,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Goal":
        """Create goal from dictionary."""
        goal = cls(
            description=data["description"],
            priority=Priority(data["priority"]),
            metadata=data.get("metadata", {}),
            id=data.get("id"),
            parent_id=data.get("parent_id"),
        )
        goal.status = Status(data.get("status", "pending"))
        goal.created_at = data.get("created_at", time.time())
        goal.updated_at = data.get("updated_at", goal.created_at)
        goal.completed_at = data.get("completed_at")
        goal.progress = data.get("progress", 0.0)
        return goal

    def __repr__(self) -> str:
        """String representation."""
        return f"Goal(id='{self.id[:8]}...', desc='{self.description[:30]}...', status={self.status.value})"


# ============================================================================
# Task Class
# ============================================================================


class Task:
    """
    Represents a concrete action to be performed.

    Tasks can have dependencies and are associated with goals.
    """

    def __init__(
        self,
        description: str,
        goal_id: Optional[str] = None,
        priority: Priority = Priority.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None,
        dependencies: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize a task.

        Args:
            description: Task description
            goal_id: Optional ID of associated goal
            priority: Task priority level
            metadata: Optional metadata dictionary
            id: Optional custom ID (auto-generated if not provided)
            dependencies: Optional list of task IDs this task depends on
        """
        self.id = id or str(uuid.uuid4())
        self.description = description
        self.goal_id = goal_id
        self.priority = priority
        self.status = Status.PENDING
        self.metadata = metadata or {}
        self.dependencies = dependencies or []
        self.created_at = time.time()
        self.updated_at = self.created_at
        self.completed_at: Optional[float] = None
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None

    def update_status(self, status: Status) -> None:
        """Update task status."""
        self.status = status
        self.updated_at = time.time()
        if status == Status.COMPLETED:
            self.completed_at = time.time()

    def set_result(self, result: Dict[str, Any]) -> None:
        """Set task execution result."""
        self.result = result
        self.updated_at = time.time()

    def add_dependency(self, task_id: str) -> None:
        """Add a task dependency."""
        if task_id not in self.dependencies:
            self.dependencies.append(task_id)

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            "id": self.id,
            "description": self.description,
            "goal_id": self.goal_id,
            "priority": self.priority.value,
            "status": self.status.value,
            "metadata": self.metadata,
            "dependencies": self.dependencies,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "result": self.result,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """Create task from dictionary."""
        task = cls(
            description=data["description"],
            goal_id=data.get("goal_id"),
            priority=Priority(data["priority"]),
            metadata=data.get("metadata", {}),
            id=data.get("id"),
            dependencies=data.get("dependencies", []),
        )
        task.status = Status(data.get("status", "pending"))
        task.created_at = data.get("created_at", time.time())
        task.updated_at = data.get("updated_at", task.created_at)
        task.completed_at = data.get("completed_at")
        task.result = data.get("result")
        task.error = data.get("error")
        return task

    def __repr__(self) -> str:
        """String representation."""
        return f"Task(id='{self.id[:8]}...', desc='{self.description[:30]}...', status={self.status.value})"


# ============================================================================
# GoalManager
# ============================================================================


class GoalManager:
    """
    Manages goals with hierarchies, priorities, and progress tracking.
    """

    def __init__(self, enable_logging: bool = False) -> None:
        """
        Initialize the goal manager.

        Args:
            enable_logging: Enable detailed logging
        """
        self._goals: Dict[str, Goal] = {}
        self._enable_logging = enable_logging
        self._logger = get_logger(__name__) if enable_logging else None

    def add_goal(
        self,
        description: str,
        priority: Priority = Priority.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None,
        parent_id: Optional[str] = None,
    ) -> str:
        """
        Add a new goal.

        Args:
            description: Goal description
            priority: Priority level
            metadata: Optional metadata
            parent_id: Optional parent goal ID for hierarchies

        Returns:
            Goal ID
        """
        goal = Goal(
            description=description,
            priority=priority,
            metadata=metadata,
            parent_id=parent_id,
        )

        self._goals[goal.id] = goal

        if self._enable_logging and self._logger:
            self._logger.info(f"Added goal: {goal.id[:8]}... - {description[:50]}")

        return goal.id

    def get_goal(self, goal_id: str) -> Optional[Goal]:
        """Get a goal by ID."""
        return self._goals.get(goal_id)

    def update_goal_status(self, goal_id: str, status: Status) -> None:
        """Update goal status."""
        goal = self._goals.get(goal_id)
        if not goal:
            raise GoalError(f"Goal not found: {goal_id}")

        goal.update_status(status)

        if self._enable_logging and self._logger:
            self._logger.info(f"Goal {goal_id[:8]}... status: {status.value}")

    def update_goal_progress(self, goal_id: str, progress: float) -> None:
        """Update goal progress (0.0 to 1.0)."""
        goal = self._goals.get(goal_id)
        if not goal:
            raise GoalError(f"Goal not found: {goal_id}")

        goal.update_progress(progress)

        if self._enable_logging and self._logger:
            self._logger.debug(f"Goal {goal_id[:8]}... progress: {progress:.1%}")

    def get_active_goals(self) -> List[Goal]:
        """Get all active (non-completed, non-cancelled) goals."""
        return [
            g
            for g in self._goals.values()
            if g.status not in (Status.COMPLETED, Status.CANCELLED, Status.FAILED)
        ]

    def get_goals_by_status(self, status: Status) -> List[Goal]:
        """Get goals with specific status."""
        return [g for g in self._goals.values() if g.status == status]

    def get_goals_by_priority(self, priority: Priority) -> List[Goal]:
        """Get goals with specific priority."""
        return [g for g in self._goals.values() if g.priority == priority]

    def get_sub_goals(self, parent_id: str) -> List[Goal]:
        """Get all sub-goals of a parent goal."""
        return [g for g in self._goals.values() if g.parent_id == parent_id]

    def get_root_goals(self) -> List[Goal]:
        """Get all root goals (no parent)."""
        return [g for g in self._goals.values() if g.parent_id is None]

    def list_all(self) -> List[Goal]:
        """Get all goals."""
        return list(self._goals.values())

    def delete_goal(self, goal_id: str) -> bool:
        """Delete a goal."""
        if goal_id in self._goals:
            del self._goals[goal_id]
            return True
        return False

    def clear(self) -> None:
        """Clear all goals."""
        self._goals.clear()

    def count(self) -> int:
        """Get number of goals."""
        return len(self._goals)

    def get_statistics(self) -> Dict[str, Any]:
        """Get goal statistics."""
        goals = list(self._goals.values())
        completed_count = len([g for g in goals if g.status == Status.COMPLETED])

        return {
            "total": len(goals),
            "pending": len([g for g in goals if g.status == Status.PENDING]),
            "in_progress": len([g for g in goals if g.status == Status.IN_PROGRESS]),
            "completed": completed_count,
            "failed": len([g for g in goals if g.status == Status.FAILED]),
            "blocked": len([g for g in goals if g.status == Status.BLOCKED]),
            "by_status": {
                Status.PENDING: len([g for g in goals if g.status == Status.PENDING]),
                Status.IN_PROGRESS: len([g for g in goals if g.status == Status.IN_PROGRESS]),
                Status.COMPLETED: completed_count,
                Status.FAILED: len([g for g in goals if g.status == Status.FAILED]),
                Status.BLOCKED: len([g for g in goals if g.status == Status.BLOCKED]),
                Status.CANCELLED: len([g for g in goals if g.status == Status.CANCELLED]),
            },
            "by_priority": {
                Priority.LOW: len([g for g in goals if g.priority == Priority.LOW]),
                Priority.MEDIUM: len([g for g in goals if g.priority == Priority.MEDIUM]),
                Priority.HIGH: len([g for g in goals if g.priority == Priority.HIGH]),
                Priority.CRITICAL: len([g for g in goals if g.priority == Priority.CRITICAL]),
            },
            "completion_rate": completed_count / len(goals) if goals else 0.0,
            "average_progress": (
                sum(g.progress for g in goals) / len(goals) if goals else 0.0
            ),
        }

    def export_to_dict(self) -> Dict[str, Any]:
        """Export all goals to dictionary."""
        return {
            "goals": [goal.to_dict() for goal in self._goals.values()],
            "count": len(self._goals),
        }

    def import_from_dict(self, data: Dict[str, Any]) -> None:
        """Import goals from dictionary."""
        self._goals.clear()
        for goal_data in data.get("goals", []):
            goal = Goal.from_dict(goal_data)
            self._goals[goal.id] = goal

    def save_to_file(self, filepath: str) -> None:
        """Save goals to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.export_to_dict(), f, indent=2)

        if self._enable_logging and self._logger:
            self._logger.info(f"Saved {self.count()} goals to {filepath}")

    def load_from_file(self, filepath: str) -> None:
        """Load goals from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)

        self.import_from_dict(data)

        if self._enable_logging and self._logger:
            self._logger.info(f"Loaded {self.count()} goals from {filepath}")


# ============================================================================
# TaskManager
# ============================================================================


class TaskManager:
    """
    Manages tasks with dependencies, scheduling, and execution tracking.
    """

    def __init__(self, enable_logging: bool = False) -> None:
        """
        Initialize the task manager.

        Args:
            enable_logging: Enable detailed logging
        """
        self._tasks: Dict[str, Task] = {}
        self._enable_logging = enable_logging
        self._logger = get_logger(__name__) if enable_logging else None

    def add_task(
        self,
        description: str,
        goal_id: Optional[str] = None,
        priority: Priority = Priority.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[str]] = None,
    ) -> str:
        """
        Add a new task.

        Args:
            description: Task description
            goal_id: Optional associated goal ID
            priority: Priority level
            metadata: Optional metadata
            dependencies: Optional list of task IDs this depends on

        Returns:
            Task ID
        """
        task = Task(
            description=description,
            goal_id=goal_id,
            priority=priority,
            metadata=metadata,
            dependencies=dependencies,
        )

        self._tasks[task.id] = task

        if self._enable_logging and self._logger:
            self._logger.info(f"Added task: {task.id[:8]}... - {description[:50]}")

        return task.id

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    def update_task_status(self, task_id: str, status: Status) -> None:
        """Update task status."""
        task = self._tasks.get(task_id)
        if not task:
            raise TaskError(f"Task not found: {task_id}")

        task.update_status(status)

        if self._enable_logging and self._logger:
            self._logger.info(f"Task {task_id[:8]}... status: {status.value}")

    def set_task_result(self, task_id: str, result: Dict[str, Any]) -> None:
        """Set task execution result."""
        task = self._tasks.get(task_id)
        if not task:
            raise TaskError(f"Task not found: {task_id}")

        task.set_result(result)

    def update_task_result(self, task_id: str, result: Dict[str, Any]) -> None:
        """Update task execution result (alias for set_task_result)."""
        self.set_task_result(task_id, result)

    def add_task_dependency(self, task_id: str, depends_on: str) -> None:
        """Add a dependency to a task."""
        task = self._tasks.get(task_id)
        if not task:
            raise TaskError(f"Task not found: {task_id}")

        task.add_dependency(depends_on)

    def is_task_ready(self, task_id: str) -> bool:
        """
        Check if a task is ready to execute (all dependencies completed).

        Args:
            task_id: Task ID

        Returns:
            True if task is ready, False otherwise
        """
        task = self._tasks.get(task_id)
        if not task:
            return False

        # Check if all dependencies are completed
        for dep_id in task.dependencies:
            dep_task = self._tasks.get(dep_id)
            if not dep_task or dep_task.status != Status.COMPLETED:
                return False

        return True

    def get_ready_tasks(self) -> List[Task]:
        """Get all tasks that are ready to execute."""
        return [
            task
            for task in self._tasks.values()
            if task.status == Status.PENDING and self.is_task_ready(task.id)
        ]

    def get_next_task(self) -> Optional[Task]:
        """
        Get the next task to execute (highest priority, ready task).

        Returns:
            Next task to execute or None if no tasks are ready
        """
        ready_tasks = self.get_ready_tasks()
        if not ready_tasks:
            return None

        # Sort by priority (highest first)
        ready_tasks.sort(key=lambda t: t.priority, reverse=True)
        return ready_tasks[0]

    def get_tasks_by_status(self, status: Status) -> List[Task]:
        """Get tasks with specific status."""
        return [t for t in self._tasks.values() if t.status == status]

    def get_tasks_by_goal(self, goal_id: str) -> List[Task]:
        """Get all tasks associated with a goal."""
        return [t for t in self._tasks.values() if t.goal_id == goal_id]

    def get_blocked_tasks(self) -> List[Task]:
        """Get tasks that are blocked by uncompleted dependencies."""
        return [
            task
            for task in self._tasks.values()
            if task.status == Status.PENDING and not self.is_task_ready(task.id)
        ]

    def list_all(self) -> List[Task]:
        """Get all tasks."""
        return list(self._tasks.values())

    def delete_task(self, task_id: str) -> bool:
        """Delete a task."""
        if task_id in self._tasks:
            del self._tasks[task_id]
            return True
        return False

    def clear(self) -> None:
        """Clear all tasks."""
        self._tasks.clear()

    def count(self) -> int:
        """Get number of tasks."""
        return len(self._tasks)

    def get_statistics(self) -> Dict[str, Any]:
        """Get task statistics."""
        tasks = list(self._tasks.values())
        ready_tasks = self.get_ready_tasks()
        blocked_tasks = self.get_blocked_tasks()
        completed_count = len([t for t in tasks if t.status == Status.COMPLETED])

        return {
            "total": len(tasks),
            "pending": len([t for t in tasks if t.status == Status.PENDING]),
            "in_progress": len([t for t in tasks if t.status == Status.IN_PROGRESS]),
            "completed": completed_count,
            "failed": len([t for t in tasks if t.status == Status.FAILED]),
            "ready": len(ready_tasks),
            "blocked": len(blocked_tasks),
            "by_status": {
                Status.PENDING: len([t for t in tasks if t.status == Status.PENDING]),
                Status.IN_PROGRESS: len([t for t in tasks if t.status == Status.IN_PROGRESS]),
                Status.COMPLETED: completed_count,
                Status.FAILED: len([t for t in tasks if t.status == Status.FAILED]),
                Status.BLOCKED: len([t for t in tasks if t.status == Status.BLOCKED]),
                Status.CANCELLED: len([t for t in tasks if t.status == Status.CANCELLED]),
            },
            "by_priority": {
                Priority.LOW: len([t for t in tasks if t.priority == Priority.LOW]),
                Priority.MEDIUM: len([t for t in tasks if t.priority == Priority.MEDIUM]),
                Priority.HIGH: len([t for t in tasks if t.priority == Priority.HIGH]),
                Priority.CRITICAL: len([t for t in tasks if t.priority == Priority.CRITICAL]),
            },
            "completion_rate": completed_count / len(tasks) if tasks else 0.0,
        }

    def export_to_dict(self) -> Dict[str, Any]:
        """Export all tasks to dictionary."""
        return {
            "tasks": [task.to_dict() for task in self._tasks.values()],
            "count": len(self._tasks),
        }

    def import_from_dict(self, data: Dict[str, Any]) -> None:
        """Import tasks from dictionary."""
        self._tasks.clear()
        for task_data in data.get("tasks", []):
            task = Task.from_dict(task_data)
            self._tasks[task.id] = task


# ============================================================================
# Nodes for Flow Integration
# ============================================================================


class InputParserNode(Node):
    """
    Node that parses user input and extracts intent, entities, and context.

    This is a simple pattern-based parser. For production use, consider
    integrating with NLU services like Rasa, DialogFlow, or LLMs.
    """

    def __init__(
        self,
        input_key: str = "user_input",
        output_key: str = "parsed_input",
        name: Optional[str] = None,
        enable_logging: bool = False,
    ) -> None:
        """
        Initialize InputParserNode.

        Args:
            input_key: SharedStore key to read user input from
            output_key: SharedStore key to write parsed result to
            name: Optional node name
            enable_logging: Enable detailed logging
        """
        super().__init__(name or "InputParserNode", enable_logging=enable_logging)
        self.input_key = input_key
        self.output_key = output_key

    def exec(self, shared: SharedStore) -> Dict[str, Any]:
        """
        Parse user input.

        Args:
            shared: SharedStore instance

        Returns:
            Dictionary with parsing results
        """
        user_input = shared.get_required(self.input_key)

        # Simple pattern-based parsing
        parsed = {
            "original": user_input,
            "intent": self._extract_intent(user_input),
            "entities": self._extract_entities(user_input),
            "keywords": self._extract_keywords(user_input),
        }

        shared.set(self.output_key, parsed)

        return {"parsed": True, "intent": parsed["intent"]}

    def _extract_intent(self, text: str) -> str:
        """Extract intent from text using simple rules."""
        text_lower = text.lower()

        # Simple intent detection
        if any(word in text_lower for word in ["create", "add", "make", "new"]):
            return "create"
        elif any(word in text_lower for word in ["delete", "remove", "cancel"]):
            return "delete"
        elif any(word in text_lower for word in ["update", "change", "modify", "edit"]):
            return "update"
        elif any(word in text_lower for word in ["find", "search", "look", "show", "list"]):
            return "query"
        elif any(word in text_lower for word in ["help", "how", "what", "?"]):
            return "help"
        else:
            return "unknown"

    def _extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract entities from text (simple implementation)."""
        entities = {}

        # Extract priority
        text_lower = text.lower()
        if "critical" in text_lower or "urgent" in text_lower:
            entities["priority"] = Priority.CRITICAL.value
        elif "high" in text_lower or "important" in text_lower:
            entities["priority"] = Priority.HIGH.value
        elif "low" in text_lower:
            entities["priority"] = Priority.LOW.value

        return entities

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        # Simple keyword extraction (remove common words)
        stop_words = {
            "a",
            "an",
            "the",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
        }

        words = text.lower().split()
        keywords = [w for w in words if w not in stop_words and len(w) > 2]

        return keywords[:10]  # Limit to 10 keywords


class GoalPlannerNode(Node):
    """
    Node that creates goals and tasks from parsed input.
    """

    def __init__(
        self,
        goal_manager: GoalManager,
        task_manager: TaskManager,
        input_key: str = "parsed_input",
        output_key: str = "plan",
        name: Optional[str] = None,
        enable_logging: bool = False,
    ) -> None:
        """
        Initialize GoalPlannerNode.

        Args:
            goal_manager: GoalManager instance
            task_manager: TaskManager instance
            input_key: SharedStore key to read parsed input from
            output_key: SharedStore key to write plan to
            name: Optional node name
            enable_logging: Enable detailed logging
        """
        super().__init__(name or "GoalPlannerNode", enable_logging=enable_logging)
        self.goal_manager = goal_manager
        self.task_manager = task_manager
        self.input_key = input_key
        self.output_key = output_key

    def exec(self, shared: SharedStore) -> Dict[str, Any]:
        """
        Create a plan (goals and tasks) from parsed input.

        Args:
            shared: SharedStore instance

        Returns:
            Dictionary with plan results
        """
        parsed = shared.get_required(self.input_key)
        intent = parsed.get("intent", "unknown")
        entities = parsed.get("entities", {})

        goals_created = 0
        tasks_created = 0

        # Create goal based on intent
        if "create" in intent:
            goal_desc = f"Create: {parsed.get('original', ' '.join(parsed.get('keywords', [])))}"
            # Handle entities as dict or list
            if isinstance(entities, dict):
                priority_str = entities.get("priority", Priority.MEDIUM.value)
            else:
                priority_str = Priority.MEDIUM.value
            priority = Priority(priority_str)

            goal_id = self.goal_manager.add_goal(goal_desc, priority=priority)
            goals_created = 1

            # Create associated tasks
            task_id = self.task_manager.add_task(
                description=f"Execute: {parsed.get('original', ' '.join(parsed.get('keywords', [])))}",
                goal_id=goal_id,
                priority=priority,
            )
            tasks_created = 1

            plan = {
                "goal_id": goal_id,
                "task_ids": [task_id],
                "intent": intent,
                "priority": priority.value,
            }
        else:
            # For other intents, create a simple plan
            plan = {"intent": intent, "action": "handle_" + intent}

        shared.set(self.output_key, plan)

        return {
            "plan_created": True,
            "intent": intent,
            "goals_created": goals_created,
            "tasks_created": tasks_created,
        }


class TaskExecutorNode(Node):
    """
    Node that executes tasks from a task manager.
    """

    def __init__(
        self,
        task_manager: TaskManager,
        task_key: str = "task_id",
        result_key: str = "task_result",
        executor: Optional[Callable[[Task, SharedStore], Dict[str, Any]]] = None,
        name: Optional[str] = None,
        enable_logging: bool = False,
    ) -> None:
        """
        Initialize TaskExecutorNode.

        Args:
            task_manager: TaskManager instance
            task_key: SharedStore key to read task ID from
            result_key: SharedStore key to write result to
            executor: Optional custom executor function
            name: Optional node name
            enable_logging: Enable detailed logging
        """
        super().__init__(name or "TaskExecutorNode", enable_logging=enable_logging)
        self.task_manager = task_manager
        self.task_key = task_key
        self.result_key = result_key
        self.executor = executor or self._default_executor

    def _default_executor(self, task: Task, shared: SharedStore) -> Dict[str, Any]:
        """Default task executor (placeholder)."""
        return {
            "status": "executed",
            "task_id": task.id,
            "description": task.description,
            "message": "Task executed successfully (default executor)",
        }

    def exec(self, shared: SharedStore) -> Dict[str, Any]:
        """
        Execute a task.

        Args:
            shared: SharedStore instance

        Returns:
            Dictionary with execution results
        """
        task_id = shared.get_required(self.task_key)

        task = self.task_manager.get_task(task_id)
        if not task:
            raise TaskError(f"Task not found: {task_id}")

        # Check if task is ready
        if not self.task_manager.is_task_ready(task_id):
            raise TaskError(f"Task {task_id} is not ready (dependencies not met)")

        # Update status to in_progress
        self.task_manager.update_task_status(task_id, Status.IN_PROGRESS)

        try:
            # Execute task
            result = self.executor(task, shared)

            # Update task with result
            self.task_manager.set_task_result(task_id, result)
            self.task_manager.update_task_status(task_id, Status.COMPLETED)

            # Store result in shared store
            shared.set(self.result_key, result)

            return {"executed": True, "task_id": task_id, "status": Status.COMPLETED}

        except Exception as e:
            # Mark task as failed and store error
            task.error = str(e)
            self.task_manager.update_task_status(task_id, Status.FAILED)
            return {"executed": False, "task_id": task_id, "status": Status.FAILED, "error": str(e)}
