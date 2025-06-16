"""
Tests for the LogicalGraphBuilder class.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from num_agents.core import Flow, Node
from num_agents.graph.logical_graph import LogicalGraphBuilder


class MockNode(Node):
    """Mock node for testing."""
    
    def __init__(self, name, next_nodes=None):
        """Initialize the mock node."""
        super().__init__(name)
        self._next_nodes = next_nodes or []
    
    def get_next_nodes(self):
        """Get the next nodes."""
        return self._next_nodes
    
    def exec(self, shared):
        """Execute the node."""
        return {"status": "success"}


@pytest.fixture
def sample_flow():
    """Create a sample flow for testing."""
    node_c = MockNode("NodeC")
    node_b = MockNode("NodeB", [node_c])
    node_a = MockNode("NodeA", [node_b])
    
    return Flow([node_a, node_b, node_c])


def test_analyze_flow(sample_flow):
    """Test analyzing a flow."""
    builder = LogicalGraphBuilder(flow=sample_flow)
    
    builder.analyze_flow()
    
    # Check that the nodes were extracted
    assert len(builder._nodes) == 3
    assert builder._nodes[0].name == "NodeA"
    assert builder._nodes[1].name == "NodeB"
    assert builder._nodes[2].name == "NodeC"
    
    # Check that the edges were built
    assert len(builder._edges) == 2
    assert ("NodeA", "NodeB") in builder._edges
    assert ("NodeB", "NodeC") in builder._edges


def test_generate_mermaid(sample_flow):
    """Test generating a Mermaid flowchart."""
    builder = LogicalGraphBuilder(flow=sample_flow)
    
    mermaid = builder.generate_mermaid()
    
    # Check that the Mermaid flowchart contains the expected elements
    assert "flowchart TD" in mermaid
    assert 'NodeA["NodeA' in mermaid
    assert 'NodeB["NodeB' in mermaid
    assert 'NodeC["NodeC' in mermaid
    assert "NodeA --> NodeB" in mermaid
    assert "NodeB --> NodeC" in mermaid


def test_export_mermaid(sample_flow):
    """Test exporting a Mermaid flowchart to a file."""
    builder = LogicalGraphBuilder(flow=sample_flow)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "logical_graph.mmd")
        builder.export_mermaid(output_path)
        
        # Check that the file was created
        assert os.path.exists(output_path)
        
        # Check that the file contains the expected content
        with open(output_path, "r") as f:
            content = f.read()
            assert "flowchart TD" in content
            assert "NodeA --> NodeB" in content
            assert "NodeB --> NodeC" in content


def test_generate_markdown(sample_flow):
    """Test generating a Markdown representation."""
    builder = LogicalGraphBuilder(flow=sample_flow)
    
    markdown = builder.generate_markdown()
    
    # Check that the Markdown contains the expected elements
    assert "# Logical Graph" in markdown
    assert "## Flow Diagram" in markdown
    assert "```mermaid" in markdown
    assert "flowchart TD" in markdown
    assert "```" in markdown
    assert "## Nodes" in markdown
    assert "### NodeA" in markdown
    assert "### NodeB" in markdown
    assert "### NodeC" in markdown
    assert "## Transitions" in markdown
    assert "- NodeA → NodeB" in markdown
    assert "- NodeB → NodeC" in markdown


def test_export_markdown(sample_flow):
    """Test exporting a Markdown representation to a file."""
    builder = LogicalGraphBuilder(flow=sample_flow)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "logical_graph.md")
        builder.export_markdown(output_path)
        
        # Check that the file was created
        assert os.path.exists(output_path)
        
        # Check that the file contains the expected content
        with open(output_path, "r") as f:
            content = f.read()
            assert "# Logical Graph" in content
            assert "## Flow Diagram" in content
            assert "```mermaid" in content
            assert "flowchart TD" in content
            assert "```" in content


def test_extract_node_description():
    """Test extracting a description from a node."""
    # Test with a node that has a docstring
    node_with_docstring = MockNode("NodeWithDocstring")
    node_with_docstring.__doc__ = "This is a test node.\nWith multiple lines."
    
    description = LogicalGraphBuilder._extract_node_description(node_with_docstring)
    
    assert description == "This is a test node."
    
    # Test with a node that doesn't have a docstring
    node_without_docstring = MockNode("NodeWithoutDocstring")
    node_without_docstring.__doc__ = None
    
    description = LogicalGraphBuilder._extract_node_description(node_without_docstring)
    
    assert description == "MockNode"
