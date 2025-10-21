# StructureAgentIA - Goal and Task Management Guide

**Version:** 0.1.0
**Module:** `num_agents.modules.structure_agent_ia`

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
4. [Components](#components)
5. [Usage Examples](#usage-examples)
6. [Flow Integration](#flow-integration)
7. [Best Practices](#best-practices)
8. [API Reference](#api-reference)

---

## Overview

StructureAgentIA provides **goal and task management** capabilities for intelligent agents. It enables:

- **Hierarchical goal management** with parent-child relationships
- **Task dependency tracking** and automatic scheduling
- **Priority-based execution** for critical tasks
- **Progress monitoring** and statistics
- **Persistence** via JSON serialization
- **Flow integration** with specialized nodes

### When to Use

- Building agents that need to **plan and execute** multi-step workflows
- Managing **complex goals** with sub-goals and dependencies
- Implementing **task scheduling** with priority queues
- Tracking **execution progress** and statistics
- Creating **autonomous agents** that can plan their own actions

---

## Quick Start

### Installation

StructureAgentIA is included in the core `num-agents` package:

```bash
pip install num-agents
```

### Basic Example

```python
from num_agents.modules.structure_agent_ia import (
    GoalManager,
    TaskManager,
    Priority,
    Status,
)

# Create managers
goal_manager = GoalManager()
task_manager = TaskManager()

# Add a goal
goal_id = goal_manager.add_goal(
    "Build user authentication system",
    priority=Priority.HIGH
)

# Add tasks for the goal
task1_id = task_manager.add_task(
    "Design database schema",
    goal_id=goal_id,
    priority=Priority.HIGH
)

task2_id = task_manager.add_task(
    "Implement login API",
    goal_id=goal_id,
    priority=Priority.HIGH,
    dependencies=[task1_id]  # Depends on task 1
)

# Execute tasks in order
next_task = task_manager.get_next_task()
print(f"Next task: {next_task.description}")

# Update task status
task_manager.update_task_status(task1_id, Status.COMPLETED)

# Now task 2 is ready
next_task = task_manager.get_next_task()
print(f"Next task: {next_task.description}")
```

---

## Core Concepts

### 1. Goals

**Goals** represent high-level objectives that an agent wants to achieve.

- Can have **sub-goals** (parent-child hierarchy)
- Track **progress** from 0.0 to 1.0
- Have **priority** levels (LOW, MEDIUM, HIGH, CRITICAL)
- Have **status** (PENDING, IN_PROGRESS, COMPLETED, FAILED, BLOCKED, CANCELLED)

### 2. Tasks

**Tasks** represent concrete actions to be performed.

- Can be associated with **goals**
- Support **dependencies** (task A must complete before task B)
- Have **priority** levels for scheduling
- Store **results** and **errors** from execution

### 3. Priority Levels

```python
class Priority:
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
```

Priorities support comparison:
```python
Priority.CRITICAL > Priority.HIGH  # True
```

### 4. Status States

```python
class Status:
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"
```

---

## Components

### GoalManager

Manages goals with hierarchies, priorities, and progress tracking.

```python
from num_agents.modules.structure_agent_ia import GoalManager, Priority

goal_manager = GoalManager(enable_logging=True)

# Add goals
parent_id = goal_manager.add_goal(
    "Build web application",
    priority=Priority.HIGH
)

child_id = goal_manager.add_goal(
    "Implement authentication",
    priority=Priority.CRITICAL,
    parent_id=parent_id
)

# Query goals
active_goals = goal_manager.get_active_goals()
high_priority = goal_manager.get_goals_by_priority(Priority.HIGH)
root_goals = goal_manager.get_root_goals()

# Update goals
goal_manager.update_goal_progress(child_id, 0.5)
goal_manager.update_goal_status(child_id, Status.IN_PROGRESS)

# Get statistics
stats = goal_manager.get_statistics()
print(f"Completion rate: {stats['completion_rate']:.1%}")
```

### TaskManager

Manages tasks with dependencies, scheduling, and execution tracking.

```python
from num_agents.modules.structure_agent_ia import TaskManager, Priority

task_manager = TaskManager(enable_logging=True)

# Add tasks with dependencies
task1_id = task_manager.add_task(
    "Fetch data from API",
    priority=Priority.HIGH
)

task2_id = task_manager.add_task(
    "Process data",
    priority=Priority.HIGH,
    dependencies=[task1_id]
)

task3_id = task_manager.add_task(
    "Save to database",
    priority=Priority.MEDIUM,
    dependencies=[task2_id]
)

# Check if task is ready
if task_manager.is_task_ready(task1_id):
    print("Task 1 is ready to execute")

# Get next task to execute
next_task = task_manager.get_next_task()
print(f"Execute: {next_task.description}")

# Execute and update
task_manager.update_task_status(task1_id, Status.IN_PROGRESS)
result = {"data": [1, 2, 3]}
task_manager.update_task_result(task1_id, result)
task_manager.update_task_status(task1_id, Status.COMPLETED)

# Get blocked tasks
blocked = task_manager.get_blocked_tasks()
```

### Goal and Task Classes

```python
from num_agents.modules.structure_agent_ia import Goal, Task, Priority

# Create a goal
goal = Goal(
    description="Complete project",
    priority=Priority.HIGH,
    metadata={"team": "backend", "sprint": "Q1"}
)

# Update progress
goal.update_progress(0.3)

# Serialize
goal_dict = goal.to_dict()

# Restore from dict
restored_goal = Goal.from_dict(goal_dict)

# Create a task
task = Task(
    description="Write unit tests",
    goal_id=goal.id,
    priority=Priority.HIGH,
    dependencies=[]
)

# Set result
task.set_result({"tests_passed": 42, "coverage": 0.85})
```

---

## Usage Examples

### Example 1: Simple Task Queue

```python
from num_agents.modules.structure_agent_ia import TaskManager, Priority, Status

task_manager = TaskManager()

# Add tasks with different priorities
task_manager.add_task("Low priority task", priority=Priority.LOW)
task_manager.add_task("High priority task", priority=Priority.HIGH)
task_manager.add_task("Critical task", priority=Priority.CRITICAL)

# Execute tasks by priority
while True:
    next_task = task_manager.get_next_task()
    if not next_task:
        break

    print(f"Executing: {next_task.description}")
    task_manager.update_task_status(next_task.id, Status.COMPLETED)
```

### Example 2: Task Dependencies

```python
from num_agents.modules.structure_agent_ia import TaskManager, Status

task_manager = TaskManager()

# Build a pipeline
task1 = task_manager.add_task("Download file")
task2 = task_manager.add_task("Parse file", dependencies=[task1])
task3 = task_manager.add_task("Validate data", dependencies=[task2])
task4 = task_manager.add_task("Save results", dependencies=[task3])

# Execute in order
while True:
    next_task = task_manager.get_next_task()
    if not next_task:
        break

    print(f"Executing: {next_task.description}")
    # Simulate execution
    task_manager.update_task_status(next_task.id, Status.COMPLETED)
```

### Example 3: Goal Hierarchies

```python
from num_agents.modules.structure_agent_ia import GoalManager, Priority

goal_manager = GoalManager()

# Create goal hierarchy
project_id = goal_manager.add_goal(
    "Launch product",
    priority=Priority.CRITICAL
)

# Sub-goals
backend_id = goal_manager.add_goal(
    "Complete backend",
    priority=Priority.HIGH,
    parent_id=project_id
)

frontend_id = goal_manager.add_goal(
    "Complete frontend",
    priority=Priority.HIGH,
    parent_id=project_id
)

# Sub-sub-goals
api_id = goal_manager.add_goal(
    "Build REST API",
    priority=Priority.HIGH,
    parent_id=backend_id
)

db_id = goal_manager.add_goal(
    "Design database",
    priority=Priority.CRITICAL,
    parent_id=backend_id
)

# Query hierarchy
root_goals = goal_manager.get_root_goals()
backend_subgoals = goal_manager.get_sub_goals(backend_id)
```

### Example 4: Progress Tracking

```python
from num_agents.modules.structure_agent_ia import GoalManager, Priority, Status

goal_manager = GoalManager()

# Add goals
goals = [
    goal_manager.add_goal(f"Goal {i}", priority=Priority.MEDIUM)
    for i in range(5)
]

# Simulate progress
goal_manager.update_goal_progress(goals[0], 1.0)  # 100% complete
goal_manager.update_goal_progress(goals[1], 0.75)
goal_manager.update_goal_progress(goals[2], 0.5)
goal_manager.update_goal_progress(goals[3], 0.25)

# Get statistics
stats = goal_manager.get_statistics()
print(f"Total goals: {stats['total']}")
print(f"Completed: {stats['completed']}")
print(f"Completion rate: {stats['completion_rate']:.1%}")
print(f"Average progress: {stats['average_progress']:.1%}")
```

### Example 5: Persistence

```python
from num_agents.modules.structure_agent_ia import GoalManager, TaskManager

# Save goals
goal_manager = GoalManager()
goal_manager.add_goal("Goal 1")
goal_manager.add_goal("Goal 2")
goal_manager.save_to_file("goals.json")

# Load goals
new_manager = GoalManager()
new_manager.load_from_file("goals.json")
print(f"Loaded {new_manager.count()} goals")

# Or use export/import
data = goal_manager.export_to_dict()
new_manager.import_from_dict(data)
```

---

## Flow Integration

StructureAgentIA provides three nodes for integration with Nüm Agents flows:

### 1. InputParserNode

Parses user input to extract intent, entities, and keywords.

```python
from num_agents.core import Flow, SharedStore
from num_agents.modules.structure_agent_ia import InputParserNode

parser = InputParserNode()

flow = Flow(nodes=[parser])

result = flow.execute(initial_data={
    "user_input": "Create a high priority task for user authentication"
})

parsed = flow.shared.get("parsed_input")
print(f"Intent: {parsed['intent']}")
print(f"Entities: {parsed['entities']}")
print(f"Keywords: {parsed['keywords']}")
```

### 2. GoalPlannerNode

Creates goals and tasks from parsed input.

```python
from num_agents.core import Flow
from num_agents.modules.structure_agent_ia import (
    GoalManager,
    TaskManager,
    InputParserNode,
    GoalPlannerNode,
)

goal_manager = GoalManager()
task_manager = TaskManager()

parser = InputParserNode()
planner = GoalPlannerNode(
    goal_manager=goal_manager,
    task_manager=task_manager
)

flow = Flow(nodes=[parser, planner])

result = flow.execute(initial_data={
    "user_input": "Create a new feature for user authentication"
})

plan = flow.shared.get("plan")
print(f"Created goal: {plan['goal_id']}")
print(f"Created tasks: {plan['task_ids']}")
```

### 3. TaskExecutorNode

Executes tasks with custom executor functions.

```python
from num_agents.core import Flow, SharedStore
from num_agents.modules.structure_agent_ia import (
    TaskManager,
    Task,
    TaskExecutorNode,
)

def my_executor(task: Task, shared: SharedStore):
    """Custom task executor."""
    print(f"Executing task: {task.description}")
    return {"status": "success", "message": "Task completed"}

task_manager = TaskManager()
task_id = task_manager.add_task("Send email notification")

executor_node = TaskExecutorNode(
    task_manager=task_manager,
    executor=my_executor
)

flow = Flow(nodes=[executor_node])

result = flow.execute(initial_data={"task_id": task_id})
print(f"Executed: {result['TaskExecutorNode']['executed']}")
```

### Complete Flow Example

```python
from num_agents.core import Flow
from num_agents.modules.structure_agent_ia import (
    GoalManager,
    TaskManager,
    InputParserNode,
    GoalPlannerNode,
    TaskExecutorNode,
    Task,
)

# Create managers
goal_manager = GoalManager()
task_manager = TaskManager()

# Create custom executor
def execute_task(task: Task, shared):
    print(f"Executing: {task.description}")
    return {"status": "completed"}

# Build flow
parser = InputParserNode()
planner = GoalPlannerNode(goal_manager, task_manager)
executor = TaskExecutorNode(task_manager, executor=execute_task)

flow = Flow(nodes=[parser, planner])

# Execute
result = flow.execute(initial_data={
    "user_input": "Create a high priority feature for authentication"
})

# Get created tasks
plan = flow.shared.get("plan")
for task_id in plan.get("task_ids", []):
    task = task_manager.get_task(task_id)
    print(f"Task created: {task.description}")
```

---

## Best Practices

### 1. Goal Design

- Keep goals **specific and measurable**
- Use **hierarchies** for complex objectives
- Set appropriate **priorities** based on business value
- Track **progress** regularly

```python
# Good
goal_manager.add_goal(
    "Increase API response time by 50%",
    priority=Priority.HIGH,
    metadata={"deadline": "2025-02-01", "owner": "backend-team"}
)

# Less specific
goal_manager.add_goal("Make API faster")
```

### 2. Task Dependencies

- Define dependencies **explicitly**
- Avoid **circular dependencies**
- Use `is_task_ready()` before execution
- Handle **failed dependencies** appropriately

```python
# Check if task is ready
if task_manager.is_task_ready(task_id):
    # Execute task
    task_manager.update_task_status(task_id, Status.IN_PROGRESS)
else:
    # Handle blocked task
    print("Task is blocked by dependencies")
```

### 3. Priority Management

- Reserve **CRITICAL** for urgent, blocking work
- Use **HIGH** for important but not urgent
- **MEDIUM** for regular work
- **LOW** for nice-to-have improvements

```python
# Priority examples
task_manager.add_task("Production bug fix", priority=Priority.CRITICAL)
task_manager.add_task("New feature", priority=Priority.HIGH)
task_manager.add_task("Code refactoring", priority=Priority.MEDIUM)
task_manager.add_task("Update docs", priority=Priority.LOW)
```

### 4. Error Handling

- Always check for **None** returns
- Store **error messages** in tasks
- Mark failed tasks with **Status.FAILED**
- Update task error field

```python
try:
    # Execute task
    result = execute_my_task(task)
    task_manager.update_task_result(task_id, result)
    task_manager.update_task_status(task_id, Status.COMPLETED)
except Exception as e:
    # Handle failure
    task = task_manager.get_task(task_id)
    task.error = str(e)
    task_manager.update_task_status(task_id, Status.FAILED)
```

### 5. Statistics and Monitoring

- Use **get_statistics()** to monitor progress
- Track **completion rates** over time
- Monitor **blocked tasks** to identify issues
- Check **average progress** for goals

```python
# Regular monitoring
stats = goal_manager.get_statistics()
if stats['completion_rate'] < 0.5:
    print("Warning: Low completion rate")

blocked = task_manager.get_blocked_tasks()
if len(blocked) > 5:
    print(f"Warning: {len(blocked)} blocked tasks")
```

### 6. Persistence

- **Save frequently** to avoid data loss
- Use **versioned filenames** for backups
- **Validate** loaded data
- Consider **compression** for large datasets

```python
import datetime

# Save with timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
goal_manager.save_to_file(f"goals_{timestamp}.json")

# Load with error handling
try:
    goal_manager.load_from_file("goals.json")
except FileNotFoundError:
    print("No saved goals found")
```

---

## API Reference

### Priority Enum

```python
class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
```

**Methods:**
- `__lt__(other)`: Less than comparison
- `__le__(other)`: Less than or equal comparison

### Status Enum

```python
class Status(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"
```

### Goal Class

```python
Goal(
    description: str,
    priority: Priority = Priority.MEDIUM,
    metadata: Optional[Dict[str, Any]] = None,
    id: Optional[str] = None,
    parent_id: Optional[str] = None
)
```

**Attributes:**
- `id`: Unique identifier
- `description`: Goal description
- `priority`: Priority level
- `status`: Current status
- `progress`: Progress (0.0 to 1.0)
- `metadata`: Custom metadata
- `parent_id`: Parent goal ID
- `created_at`: Creation timestamp
- `updated_at`: Last update timestamp
- `completed_at`: Completion timestamp

**Methods:**
- `update_status(status: Status)`: Update goal status
- `update_progress(progress: float)`: Update progress (0.0 to 1.0)
- `to_dict()`: Serialize to dictionary
- `from_dict(data: Dict)`: Create from dictionary

### Task Class

```python
Task(
    description: str,
    goal_id: Optional[str] = None,
    priority: Priority = Priority.MEDIUM,
    metadata: Optional[Dict[str, Any]] = None,
    id: Optional[str] = None,
    dependencies: Optional[List[str]] = None
)
```

**Attributes:**
- `id`: Unique identifier
- `description`: Task description
- `goal_id`: Associated goal ID
- `priority`: Priority level
- `status`: Current status
- `dependencies`: List of task IDs this depends on
- `result`: Execution result
- `error`: Error message if failed
- `metadata`: Custom metadata
- `created_at`: Creation timestamp
- `updated_at`: Last update timestamp
- `completed_at`: Completion timestamp

**Methods:**
- `update_status(status: Status)`: Update task status
- `set_result(result: Dict)`: Set execution result
- `add_dependency(task_id: str)`: Add dependency
- `to_dict()`: Serialize to dictionary
- `from_dict(data: Dict)`: Create from dictionary

### GoalManager Class

```python
GoalManager(enable_logging: bool = False)
```

**Methods:**

- `add_goal(description, priority, metadata, parent_id) -> str`: Add a new goal
- `get_goal(goal_id: str) -> Optional[Goal]`: Get goal by ID
- `update_goal_status(goal_id: str, status: Status)`: Update status
- `update_goal_progress(goal_id: str, progress: float)`: Update progress
- `get_active_goals() -> List[Goal]`: Get active goals
- `get_goals_by_status(status: Status) -> List[Goal]`: Filter by status
- `get_goals_by_priority(priority: Priority) -> List[Goal]`: Filter by priority
- `get_sub_goals(parent_id: str) -> List[Goal]`: Get sub-goals
- `get_root_goals() -> List[Goal]`: Get root goals
- `list_all() -> List[Goal]`: Get all goals
- `delete_goal(goal_id: str) -> bool`: Delete goal
- `clear()`: Clear all goals
- `count() -> int`: Get goal count
- `get_statistics() -> Dict`: Get statistics
- `export_to_dict() -> Dict`: Export to dictionary
- `import_from_dict(data: Dict)`: Import from dictionary
- `save_to_file(filepath: str)`: Save to JSON file
- `load_from_file(filepath: str)`: Load from JSON file

### TaskManager Class

```python
TaskManager(enable_logging: bool = False)
```

**Methods:**

- `add_task(description, goal_id, priority, metadata, dependencies) -> str`: Add task
- `get_task(task_id: str) -> Optional[Task]`: Get task by ID
- `update_task_status(task_id: str, status: Status)`: Update status
- `update_task_result(task_id: str, result: Dict)`: Update result
- `add_task_dependency(task_id: str, depends_on: str)`: Add dependency
- `is_task_ready(task_id: str) -> bool`: Check if ready
- `get_ready_tasks() -> List[Task]`: Get ready tasks
- `get_next_task() -> Optional[Task]`: Get next task to execute
- `get_tasks_by_status(status: Status) -> List[Task]`: Filter by status
- `get_tasks_by_goal(goal_id: str) -> List[Task]`: Get tasks for goal
- `get_blocked_tasks() -> List[Task]`: Get blocked tasks
- `list_all() -> List[Task]`: Get all tasks
- `delete_task(task_id: str) -> bool`: Delete task
- `clear()`: Clear all tasks
- `count() -> int`: Get task count
- `get_statistics() -> Dict`: Get statistics
- `export_to_dict() -> Dict`: Export to dictionary
- `import_from_dict(data: Dict)`: Import from dictionary

### InputParserNode

```python
InputParserNode(
    input_key: str = "user_input",
    output_key: str = "parsed_input",
    name: Optional[str] = None,
    enable_logging: bool = False
)
```

Parses user input and extracts intent, entities, and keywords.

**Output format:**
```python
{
    "original": "user input text",
    "intent": "create|delete|update|query|help|unknown",
    "entities": {"priority": "high"},
    "keywords": ["list", "of", "keywords"]
}
```

### GoalPlannerNode

```python
GoalPlannerNode(
    goal_manager: GoalManager,
    task_manager: TaskManager,
    input_key: str = "parsed_input",
    output_key: str = "plan",
    name: Optional[str] = None,
    enable_logging: bool = False
)
```

Creates goals and tasks from parsed input.

**Output format:**
```python
{
    "plan_created": True,
    "intent": "create",
    "goals_created": 1,
    "tasks_created": 1
}
```

### TaskExecutorNode

```python
TaskExecutorNode(
    task_manager: TaskManager,
    task_key: str = "task_id",
    result_key: str = "task_result",
    executor: Optional[Callable[[Task, SharedStore], Dict]] = None,
    name: Optional[str] = None,
    enable_logging: bool = False
)
```

Executes tasks with custom executor function.

**Executor signature:**
```python
def executor(task: Task, shared: SharedStore) -> Dict[str, Any]:
    # Execute task
    return {"status": "success"}
```

**Output format:**
```python
{
    "executed": True,
    "task_id": "task-id",
    "status": Status.COMPLETED
}
```

---

## Advanced Topics

### Custom Intent Detection

For production use, integrate with NLU services:

```python
import openai

class LLMInputParser(InputParserNode):
    def _extract_intent(self, text: str) -> str:
        # Use OpenAI for intent classification
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "user",
                "content": f"Classify intent: {text}"
            }]
        )
        return response.choices[0].message.content
```

### Task Scheduling

Implement custom scheduling logic:

```python
class CustomTaskManager(TaskManager):
    def get_next_task(self):
        # Custom logic: prioritize tasks with deadlines
        ready_tasks = self.get_ready_tasks()
        if not ready_tasks:
            return None

        # Sort by deadline in metadata
        ready_tasks.sort(
            key=lambda t: t.metadata.get("deadline", "9999-12-31")
        )
        return ready_tasks[0]
```

### Progress Calculation

Auto-calculate goal progress from sub-goals:

```python
def calculate_goal_progress(goal_manager, goal_id):
    sub_goals = goal_manager.get_sub_goals(goal_id)
    if not sub_goals:
        return goal_manager.get_goal(goal_id).progress

    total_progress = sum(g.progress for g in sub_goals)
    avg_progress = total_progress / len(sub_goals)

    goal_manager.update_goal_progress(goal_id, avg_progress)
    return avg_progress
```

---

## Copyright

Copyright (c) 2025 Lionel TAGNE. All Rights Reserved.
This software is proprietary and confidential.
