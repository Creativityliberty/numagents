"""
Meta-Orchestrator implementation for the Nüm Agents SDK.

This module provides the MetaOrchestrator class, which is responsible for
validating agent designs, checking for consistency, and providing suggestions
for improvements.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from num_agents.univers.univers_catalog_loader import UniversCatalogLoader
from num_agents.utils.file_io import AgentSpecLoader, read_yaml


class ConsistencyChecker:
    """
    Checker for agent design consistency.
    
    This class is responsible for checking the consistency of an agent design,
    comparing the declared modules with the nodes present in the flow.
    """
    
    def __init__(
        self,
        agent_spec: Dict[str, Any],
        univers_catalog: Dict[str, Any],
        graph_nodes: Set[str]
    ) -> None:
        """
        Initialize the consistency checker.
        
        Args:
            agent_spec: The agent specification
            univers_catalog: The universe catalog
            graph_nodes: The set of node names in the logical graph
        """
        self.agent_spec = agent_spec
        self.univers_catalog = univers_catalog
        self.graph_nodes = graph_nodes
    
    def check(self) -> Dict[str, Any]:
        """
        Check the consistency of the agent design.
        
        Returns:
            A dictionary containing the results of the consistency check
        """
        # Get the declared universes
        universes = self.agent_spec["agent"]["univers"]
        
        # Resolve the declared modules
        declared_modules = set()
        for universe in universes:
            universe_modules = self.univers_catalog.get("univers_catalog", {}).get(universe, {}).get("modules", [])
            declared_modules.update(universe_modules)
        
        # Find missing modules (nodes in the graph but not declared)
        missing_modules = self.graph_nodes - declared_modules
        
        # Find unused modules (declared but not in the graph)
        unused_modules = declared_modules - self.graph_nodes
        
        return {
            "declared_modules": list(declared_modules),
            "graph_nodes": list(self.graph_nodes),
            "missing_modules": list(missing_modules),
            "unused_modules": list(unused_modules),
            "is_consistent": len(missing_modules) == 0
        }


class SuggestionEngine:
    """
    Engine for suggesting improvements to agent designs.
    
    This class is responsible for analyzing an agent design and suggesting
    improvements based on best practices and common patterns.
    """
    
    def __init__(
        self,
        agent_spec: Dict[str, Any],
        univers_catalog: Dict[str, Any],
        consistency_results: Dict[str, Any]
    ) -> None:
        """
        Initialize the suggestion engine.
        
        Args:
            agent_spec: The agent specification
            univers_catalog: The universe catalog
            consistency_results: The results of the consistency check
        """
        self.agent_spec = agent_spec
        self.univers_catalog = univers_catalog
        self.consistency_results = consistency_results
    
    def suggest(self) -> List[str]:
        """
        Generate suggestions for improving the agent design.
        
        Returns:
            A list of suggestion strings
        """
        suggestions = []
        
        # Get the active modules
        active_modules = set(self.consistency_results["graph_nodes"])
        
        # Suggest adding missing modules
        for module in self.consistency_results["missing_modules"]:
            suggestions.append(f"Add '{module}' to the agent specification (it's used in the flow but not declared).")
        
        # Suggest removing unused modules
        for module in self.consistency_results["unused_modules"]:
            suggestions.append(f"Consider removing '{module}' from the agent specification (it's declared but not used in the flow).")
        
        # Suggest common module combinations
        if "ActiveLearningNode" in active_modules and "EscalationManager" not in active_modules:
            suggestions.append("Consider adding 'EscalationManager' when using 'ActiveLearningNode' for better supervision.")
        
        if "EventBus" in active_modules and "SchedulerNode" not in active_modules:
            suggestions.append("Consider adding 'SchedulerNode' when using 'EventBus' for event scheduling.")
        
        if "FallbackNodeAdvanced" in active_modules and "EscalationManager" not in active_modules:
            suggestions.append("Consider adding 'EscalationManager' when using 'FallbackNodeAdvanced' for error handling.")
        
        if "MemoryRecallNode" in active_modules and "MemoryStoreNode" not in active_modules:
            suggestions.append("Consider adding 'MemoryStoreNode' when using 'MemoryRecallNode' for complete memory management.")
        
        return suggestions


class ReportBuilder:
    """
    Builder for agent audit reports.
    
    This class is responsible for building audit reports for agent designs,
    summarizing the results of consistency checks and suggestions.
    """
    
    def __init__(
        self,
        agent_spec: Dict[str, Any],
        consistency_results: Dict[str, Any],
        suggestions: List[str]
    ) -> None:
        """
        Initialize the report builder.
        
        Args:
            agent_spec: The agent specification
            consistency_results: The results of the consistency check
            suggestions: The list of suggestions
        """
        self.agent_spec = agent_spec
        self.consistency_results = consistency_results
        self.suggestions = suggestions
    
    def build(self) -> Dict[str, Any]:
        """
        Build the audit report.
        
        Returns:
            A dictionary containing the audit report
        """
        # Calculate completeness percentage
        total_modules = len(self.consistency_results["declared_modules"])
        if total_modules > 0:
            unused_modules = len(self.consistency_results["unused_modules"])
            completeness = 100 - (unused_modules * 100 / total_modules)
        else:
            completeness = 0
        
        return {
            "validation": {
                "agent_name": self.agent_spec["agent"]["name"],
                "status": "valid" if self.consistency_results["is_consistent"] else "invalid",
                "issues": [
                    {"type": "missing_module", "module": module}
                    for module in self.consistency_results["missing_modules"]
                ] + [
                    {"type": "unused_module", "module": module}
                    for module in self.consistency_results["unused_modules"]
                ],
                "suggestions": self.suggestions,
                "completeness": f"{completeness:.1f}%",
                "declared_modules": self.consistency_results["declared_modules"],
                "graph_nodes": self.consistency_results["graph_nodes"]
            }
        }


class MetaOrchestrator:
    """
    Orchestrator for validating and supervising agent designs.
    
    This class is responsible for coordinating the validation and supervision
    of agent designs, using the ConsistencyChecker, SuggestionEngine, and
    ReportBuilder.
    """
    
    def __init__(
        self,
        agent_dir: str,
        agent_spec_path: Optional[str] = None,
        univers_catalog_path: Optional[str] = None
    ) -> None:
        """
        Initialize the meta-orchestrator.
        
        Args:
            agent_dir: Path to the agent directory
            agent_spec_path: Optional path to the agent specification YAML file.
                            If not provided, it will be looked for in the agent directory.
            univers_catalog_path: Optional path to the universe catalog YAML file.
                                 If not provided, the default path will be used.
        """
        self.agent_dir = agent_dir
        
        # Set the agent specification path
        if agent_spec_path:
            self.agent_spec_path = agent_spec_path
        else:
            self.agent_spec_path = os.path.join(agent_dir, "agent.yaml")
        
        # Load the agent specification
        self.agent_spec_loader = AgentSpecLoader(self.agent_spec_path)
        self.agent_spec = self.agent_spec_loader.load()
        
        # Load the universe catalog
        self.univers_catalog_loader = UniversCatalogLoader(univers_catalog_path)
        self.univers_catalog = self.univers_catalog_loader.load()
        
        # Set the logical graph path
        self.logical_graph_path = os.path.join(agent_dir, "logical_graph.mmd")
    
    def analyze(self) -> Dict[str, Any]:
        """
        Analyze the agent design.
        
        This method coordinates the validation and supervision of the agent design,
        using the ConsistencyChecker, SuggestionEngine, and ReportBuilder.
        
        Returns:
            A dictionary containing the audit report
        """
        # Extract node names from the logical graph
        graph_nodes = self._extract_nodes_from_graph()
        
        # Check consistency
        checker = ConsistencyChecker(self.agent_spec, self.univers_catalog, graph_nodes)
        consistency_results = checker.check()
        
        # Generate suggestions
        suggestion_engine = SuggestionEngine(self.agent_spec, self.univers_catalog, consistency_results)
        suggestions = suggestion_engine.suggest()
        
        # Build the report
        report_builder = ReportBuilder(self.agent_spec, consistency_results, suggestions)
        report = report_builder.build()
        
        return report
    
    def export_report(self, output_path: Optional[str] = None) -> str:
        """
        Export the audit report to a file.
        
        Args:
            output_path: Optional path to write the report to.
                        If not provided, it will be written to audit_report.json
                        in the agent directory.
        
        Returns:
            The path to the exported report
        """
        # Generate the report
        report = self.analyze()
        
        # Set the output path
        if not output_path:
            output_path = os.path.join(self.agent_dir, "audit_report.json")
        
        # Create directory if it doesn't exist
        directory = os.path.dirname(output_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        # Write the report
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        
        return output_path
    
    def _extract_nodes_from_graph(self) -> Set[str]:
        """
        Extract node names from the logical graph.
        
        Returns:
            A set of node names
        """
        if not os.path.exists(self.logical_graph_path):
            return set()
        
        nodes = set()
        with open(self.logical_graph_path, "r") as f:
            for line in f:
                # Look for node definitions (e.g., "NodeName["NodeName\n(description)"]")
                if "[" in line and "]" in line:
                    node_name = line.split("[")[0].strip()
                    nodes.add(node_name)
                
                # Look for edges (e.g., "NodeA --> NodeB")
                elif "-->" in line:
                    parts = line.strip().split("-->")
                    if len(parts) == 2:
                        nodes.add(parts[0].strip())
                        nodes.add(parts[1].strip())
        
        return nodes


def analyze_agent(
    agent_dir: str,
    agent_spec_path: Optional[str] = None,
    univers_catalog_path: Optional[str] = None,
    output_path: Optional[str] = None
) -> str:
    """
    Analyze an agent and generate an audit report.
    
    Args:
        agent_dir: Path to the agent directory
        agent_spec_path: Optional path to the agent specification YAML file
        univers_catalog_path: Optional path to the universe catalog YAML file
        output_path: Optional path to write the report to
        
    Returns:
        The path to the exported report
    """
    orchestrator = MetaOrchestrator(agent_dir, agent_spec_path, univers_catalog_path)
    return orchestrator.export_report(output_path)
