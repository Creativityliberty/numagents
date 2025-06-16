"""
Tests for the NumAgentsComposer class.
"""

import os
import shutil
import tempfile
from pathlib import Path

import pytest
import yaml

from num_agents.composer.composer import NumAgentsComposer


@pytest.fixture
def sample_agent_spec():
    """Create a sample agent specification for testing."""
    return {
        "agent": {
            "name": "Test Agent",
            "description": "A test agent for unit testing",
            "univers": ["core", "memory"],
            "protocol": "standard",
            "llm": {
                "provider": "openai",
                "model": "gpt-4"
            },
            "memory": {
                "type": "simple"
            }
        }
    }


@pytest.fixture
def sample_univers_catalog():
    """Create a sample universe catalog for testing."""
    return {
        "univers_catalog": {
            "core": {
                "description": "Core universe with essential modules",
                "modules": ["ManagerGoalNode", "ToolAdapterNode"]
            },
            "memory": {
                "description": "Memory-related modules",
                "modules": ["MemoryRecallNode", "MemoryStoreNode"]
            },
            "advanced": {
                "description": "Advanced modules",
                "modules": ["ActiveLearningNode", "FallbackNodeAdvanced"]
            }
        }
    }


@pytest.fixture
def agent_spec_file(sample_agent_spec):
    """Create a temporary agent specification file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(sample_agent_spec, f)
        spec_path = f.name
    
    yield spec_path
    
    # Clean up
    os.unlink(spec_path)


@pytest.fixture
def univers_catalog_file(sample_univers_catalog):
    """Create a temporary universe catalog file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(sample_univers_catalog, f)
        catalog_path = f.name
    
    yield catalog_path
    
    # Clean up
    os.unlink(catalog_path)


@pytest.fixture
def output_dir():
    """Create a temporary output directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


def test_resolve_modules(agent_spec_file, univers_catalog_file):
    """Test resolving modules from the agent specification."""
    composer = NumAgentsComposer(agent_spec_file, univers_catalog_file)
    
    modules = composer.resolve_modules()
    
    assert modules == {"ManagerGoalNode", "ToolAdapterNode", "MemoryRecallNode", "MemoryStoreNode"}


def test_generate_scaffold(agent_spec_file, univers_catalog_file, output_dir):
    """Test generating an agent scaffold."""
    composer = NumAgentsComposer(agent_spec_file, univers_catalog_file, output_dir)
    
    scaffold_dir = composer.generate_scaffold()
    
    assert scaffold_dir == output_dir
    
    # Check that the expected files and directories were created
    assert os.path.exists(os.path.join(scaffold_dir, "nodes"))
    assert os.path.exists(os.path.join(scaffold_dir, "flow.py"))
    assert os.path.exists(os.path.join(scaffold_dir, "shared_store.py"))
    assert os.path.exists(os.path.join(scaffold_dir, "main.py"))
    assert os.path.exists(os.path.join(scaffold_dir, "agent.yaml"))
    assert os.path.exists(os.path.join(scaffold_dir, "README.md"))
    
    # Check that the node files were created
    assert os.path.exists(os.path.join(scaffold_dir, "nodes", "manager_goal_node.py"))
    assert os.path.exists(os.path.join(scaffold_dir, "nodes", "tool_adapter_node.py"))
    assert os.path.exists(os.path.join(scaffold_dir, "nodes", "memory_recall_node.py"))
    assert os.path.exists(os.path.join(scaffold_dir, "nodes", "memory_store_node.py"))


def test_to_snake_case():
    """Test converting names to snake_case."""
    assert NumAgentsComposer._to_snake_case("ManagerGoalNode") == "manager_goal_node"
    assert NumAgentsComposer._to_snake_case("ToolAdapterNode") == "tool_adapter_node"
    assert NumAgentsComposer._to_snake_case("MemoryRecallNode") == "memory_recall_node"
    assert NumAgentsComposer._to_snake_case("FallbackNodeAdvanced") == "fallback_node_advanced"
