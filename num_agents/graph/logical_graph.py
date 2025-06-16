"""
Logical graph generation for the Nüm Agents SDK.

This module provides functionality for generating logical graphs of agent flows,
visualizing dependencies and relationships between nodes.
"""

import os
from typing import Any, Dict, List, Optional, Set, Tuple

from num_agents.core import Flow, Node


class LogicalGraphBuilder:
    """
    Builder for logical graphs of agent flows.
    
    This class is responsible for analyzing a flow and generating a logical graph
    that visualizes the dependencies and relationships between nodes.
    """
    
    def __init__(self, flow: Optional[Flow] = None, flow_path: Optional[str] = None) -> None:
        """
        Initialize the logical graph builder.
        
        Args:
            flow: Optional Flow object to analyze
            flow_path: Optional path to a flow.py file to analyze
        """
        self.flow = flow
        self.flow_path = flow_path
        self._nodes: List[Node] = []
        self._edges: List[Tuple[str, str]] = []
        
        # If a flow is provided, extract its nodes
        if flow:
            self._nodes = flow.get_nodes()
    
    def analyze_flow(self) -> None:
        """
        Analyze the flow and build the graph.
        
        This method extracts the nodes and edges from the flow.
        """
        if not self._nodes and self.flow:
            self._nodes = self.flow.get_nodes()
        
        # Build the edges based on the node transitions
        self._edges = []
        for node in self._nodes:
            next_nodes = node.get_next_nodes()
            for next_node in next_nodes:
                self._edges.append((node.name, next_node.name))
    
    def generate_mermaid(self) -> str:
        """
        Generate a Mermaid flowchart representation of the graph.
        
        Returns:
            A string containing the Mermaid flowchart
        """
        if not self._edges:
            self.analyze_flow()
        
        # Start with the flowchart header
        mermaid = "flowchart TD\n\n"
        
        # Add node definitions with descriptions
        for node in self._nodes:
            # Extract a description from the node's docstring or class name
            description = self._extract_node_description(node)
            mermaid += f'{node.name}["{node.name}\\n({description})"]\n'
        
        mermaid += "\n"
        
        # Add the edges
        for source, target in self._edges:
            mermaid += f"{source} --> {target}\n"
        
        return mermaid
    
    def export_mermaid(self, output_path: str) -> None:
        """
        Export the Mermaid flowchart to a file.
        
        Args:
            output_path: Path to write the Mermaid flowchart to
        """
        mermaid = self.generate_mermaid()
        
        # Create directory if it doesn't exist
        directory = os.path.dirname(output_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        with open(output_path, "w") as f:
            f.write(mermaid)
    
    def generate_markdown(self) -> str:
        """
        Generate a Markdown representation of the graph.
        
        Returns:
            A string containing the Markdown representation
        """
        if not self._edges:
            self.analyze_flow()
        
        markdown = f"# Logical Graph\n\n"
        
        # Add a section for the Mermaid diagram
        markdown += "## Flow Diagram\n\n"
        markdown += "```mermaid\n"
        markdown += self.generate_mermaid()
        markdown += "```\n\n"
        
        # Add a section for the nodes
        markdown += "## Nodes\n\n"
        for node in self._nodes:
            description = self._extract_node_description(node)
            markdown += f"### {node.name}\n\n"
            markdown += f"{description}\n\n"
        
        # Add a section for the edges
        markdown += "## Transitions\n\n"
        for source, target in self._edges:
            markdown += f"- {source} → {target}\n"
        
        return markdown
    
    def export_markdown(self, output_path: str) -> None:
        """
        Export the Markdown representation to a file.
        
        Args:
            output_path: Path to write the Markdown to
        """
        markdown = self.generate_markdown()
        
        # Create directory if it doesn't exist
        directory = os.path.dirname(output_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        with open(output_path, "w") as f:
            f.write(markdown)
    
    @staticmethod
    def _extract_node_description(node: Node) -> str:
        """
        Extract a description from a node.
        
        Args:
            node: The node to extract a description from
            
        Returns:
            A description of the node
        """
        # Try to get the docstring
        if node.__doc__:
            # Extract the first line of the docstring
            docstring = node.__doc__.strip().split("\n")[0]
            return docstring
        
        # If no docstring, use the class name
        return node.__class__.__name__
    
    @classmethod
    def from_flow_file(cls, flow_path: str) -> "LogicalGraphBuilder":
        """
        Create a LogicalGraphBuilder from a flow.py file.
        
        Args:
            flow_path: Path to the flow.py file
            
        Returns:
            A LogicalGraphBuilder instance
            
        Raises:
            ImportError: If the flow.py file can't be imported
            AttributeError: If the flow.py file doesn't have a create_flow function
        """
        # Get the directory containing the flow.py file
        flow_dir = os.path.dirname(os.path.abspath(flow_path))
        
        # Add the directory to the Python path
        import sys
        sys.path.insert(0, flow_dir)
        
        try:
            # Import the create_flow function
            from flow import create_flow
            
            # Create the flow
            flow = create_flow()
            
            # Create the LogicalGraphBuilder
            return cls(flow=flow)
        finally:
            # Remove the directory from the Python path
            sys.path.pop(0)


def generate_logical_graph(
    agent_dir: str,
    output_mermaid: Optional[str] = None,
    output_markdown: Optional[str] = None
) -> Tuple[str, str]:
    """
    Generate a logical graph for an agent.
    
    Args:
        agent_dir: Path to the agent directory
        output_mermaid: Optional path to write the Mermaid flowchart to
        output_markdown: Optional path to write the Markdown to
        
    Returns:
        A tuple of (mermaid_path, markdown_path)
    """
    # Get the path to the flow.py file
    flow_path = os.path.join(agent_dir, "flow.py")
    
    # Create the LogicalGraphBuilder
    builder = LogicalGraphBuilder.from_flow_file(flow_path)
    
    # Analyze the flow
    builder.analyze_flow()
    
    # Set default output paths if not provided
    if not output_mermaid:
        output_mermaid = os.path.join(agent_dir, "logical_graph.mmd")
    
    if not output_markdown:
        output_markdown = os.path.join(agent_dir, "logical_graph.md")
    
    # Export the Mermaid flowchart
    builder.export_mermaid(output_mermaid)
    
    # Export the Markdown
    builder.export_markdown(output_markdown)
    
    return output_mermaid, output_markdown
