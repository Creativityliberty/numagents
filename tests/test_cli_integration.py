"""
Integration tests for the Nüm Agents CLI.
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
import yaml

from num_agents.cli import generate


@pytest.fixture
def sample_agent_spec():
    """Create a sample agent specification for testing."""
    return {
        "agent": {
            "name": "Integration Test Agent",
            "description": "An agent for integration testing",
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


def test_generate_command_integration(agent_spec_file, univers_catalog_file, output_dir):
    """Test the generate command end-to-end."""
    # Call the generate function directly
    generate(
        agent_spec=agent_spec_file,
        univers_catalog=univers_catalog_file,
        output_dir=output_dir,
        skip_graph=False,
        skip_audit=False
    )
    
    # Check that the expected files and directories were created
    assert os.path.exists(os.path.join(output_dir, "nodes"))
    assert os.path.exists(os.path.join(output_dir, "flow.py"))
    assert os.path.exists(os.path.join(output_dir, "shared_store.py"))
    assert os.path.exists(os.path.join(output_dir, "main.py"))
    assert os.path.exists(os.path.join(output_dir, "agent.yaml"))
    assert os.path.exists(os.path.join(output_dir, "README.md"))
    assert os.path.exists(os.path.join(output_dir, "logical_graph.mmd"))
    assert os.path.exists(os.path.join(output_dir, "logical_graph.md"))
    assert os.path.exists(os.path.join(output_dir, "audit_report.json"))
    
    # Check that the node files were created
    assert os.path.exists(os.path.join(output_dir, "nodes", "manager_goal_node.py"))
    assert os.path.exists(os.path.join(output_dir, "nodes", "tool_adapter_node.py"))
    assert os.path.exists(os.path.join(output_dir, "nodes", "memory_recall_node.py"))
    assert os.path.exists(os.path.join(output_dir, "nodes", "memory_store_node.py"))
    
    # Check that the flow.py file contains the expected content
    with open(os.path.join(output_dir, "flow.py"), "r") as f:
        flow_content = f.read()
        assert "from nodes.manager_goal_node import ManagerGoalNode" in flow_content
        assert "from nodes.tool_adapter_node import ToolAdapterNode" in flow_content
        assert "from nodes.memory_recall_node import MemoryRecallNode" in flow_content
        assert "from nodes.memory_store_node import MemoryStoreNode" in flow_content
        assert "ManagerGoalNode()" in flow_content
        assert "ToolAdapterNode()" in flow_content
        assert "MemoryRecallNode()" in flow_content
        assert "MemoryStoreNode()" in flow_content
    
    # Check that the logical_graph.mmd file contains the expected content
    with open(os.path.join(output_dir, "logical_graph.mmd"), "r") as f:
        graph_content = f.read()
        assert "flowchart TD" in graph_content
        assert "ManagerGoalNode" in graph_content
        assert "ToolAdapterNode" in graph_content
        assert "MemoryRecallNode" in graph_content
        assert "MemoryStoreNode" in graph_content
    
    # Check that the audit_report.json file contains the expected content
    with open(os.path.join(output_dir, "audit_report.json"), "r") as f:
        import json
        report = json.load(f)
        assert "validation" in report
        assert "agent_name" in report["validation"]
        assert report["validation"]["agent_name"] == "Integration Test Agent"
        assert "status" in report["validation"]
        assert "declared_modules" in report["validation"]
        assert "graph_nodes" in report["validation"]
