"""
Composer implementation for the Nüm Agents SDK.

This module provides the NumAgentsComposer class, which is responsible for
generating agent scaffolds based on agent specifications and universe catalogs.
"""

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from num_agents.univers.univers_catalog_loader import UniversCatalogLoader
from num_agents.utils.file_io import AgentSpecLoader, read_yaml, write_yaml


class NumAgentsComposer:
    """
    Composer for generating agent scaffolds.
    
    This class is responsible for generating agent scaffolds based on
    agent specifications and universe catalogs.
    """
    
    def __init__(
        self,
        agent_spec_path: str,
        univers_catalog_path: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> None:
        """
        Initialize the composer.
        
        Args:
            agent_spec_path: Path to the agent specification YAML file
            univers_catalog_path: Optional path to the universe catalog YAML file.
                                 If not provided, the default path will be used.
            output_dir: Optional output directory for the generated scaffold.
                       If not provided, a directory will be created based on the agent name.
        """
        self.agent_spec_path = agent_spec_path
        self.univers_catalog_path = univers_catalog_path
        
        # Load the agent specification
        self.agent_spec_loader = AgentSpecLoader(agent_spec_path)
        self.agent_spec = self.agent_spec_loader.load()
        
        # Set the output directory
        if output_dir:
            self.output_dir = output_dir
        else:
            # Create a directory based on the agent name
            agent_name = self.agent_spec_loader.get_agent_name()
            # Convert to snake_case
            agent_dir_name = agent_name.lower().replace(" ", "_")
            self.output_dir = os.path.join(os.getcwd(), agent_dir_name)
        
        # Load the universe catalog
        self.univers_catalog_loader = UniversCatalogLoader(univers_catalog_path)
        self.univers_catalog = self.univers_catalog_loader.load()
    
    def resolve_modules(self) -> Set[str]:
        """
        Resolve all modules required by the agent based on its universes.
        
        Returns:
            A set of module names
        """
        universes = self.agent_spec_loader.get_agent_universes()
        return self.univers_catalog_loader.resolve_modules(universes)
    
    def generate_scaffold(self) -> str:
        """
        Generate the agent scaffold.
        
        This method creates the directory structure and files for the agent
        based on the agent specification and resolved modules.
        
        Returns:
            The path to the generated scaffold
        """
        # Create the output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Create the nodes directory
        nodes_dir = os.path.join(self.output_dir, "nodes")
        if not os.path.exists(nodes_dir):
            os.makedirs(nodes_dir)
        
        # Resolve the modules
        modules = self.resolve_modules()
        
        # Generate the node files
        for module in modules:
            self._generate_node_file(module)
        
        # Generate the flow.py file
        self._generate_flow_file(modules)
        
        # Generate the shared_store.py file
        self._generate_shared_store_file()
        
        # Generate the main.py file
        self._generate_main_file()
        
        # Copy the agent.yaml file
        shutil.copy(self.agent_spec_path, os.path.join(self.output_dir, "agent.yaml"))
        
        # Generate a basic README.md
        self._generate_readme()
        
        return self.output_dir
    
    def _generate_node_file(self, module_name: str) -> None:
        """
        Generate a node file for a module.
        
        Args:
            module_name: The name of the module
        """
        # Convert module name to snake_case for the file name
        file_name = self._to_snake_case(module_name)
        if not file_name.endswith("_node") and module_name.endswith("Node"):
            file_name += "_node"
        
        file_path = os.path.join(self.output_dir, "nodes", f"{file_name}.py")
        
        # Generate the node class content
        content = f'''"""
{module_name} implementation.

This module provides the {module_name} class, which is a node in the agent flow.
"""

from typing import Any, Dict

from num_agents.core import Node, SharedStore


class {module_name}(Node):
    """
    {module_name} implementation.
    
    This node is responsible for [DESCRIPTION].
    """
    
    def __init__(self) -> None:
        """Initialize the node."""
        super().__init__("{module_name}")
    
    def exec(self, shared: SharedStore) -> Dict[str, Any]:
        """
        Execute the node's processing logic.
        
        Args:
            shared: The shared store for accessing and storing data
            
        Returns:
            A dictionary containing the results of the node's execution
        """
        # TODO: Implement the node's logic
        return {{"status": "success"}}
'''
        
        # Write the file
        with open(file_path, "w") as f:
            f.write(content)
    
    def _generate_flow_file(self, modules: Set[str]) -> None:
        """
        Generate the flow.py file.
        
        Args:
            modules: The set of module names
        """
        file_path = os.path.join(self.output_dir, "flow.py")
        
        # Generate the imports
        imports = []
        for module in modules:
            # Convert module name to snake_case for the import
            module_file = self._to_snake_case(module)
            if not module_file.endswith("_node") and module.endswith("Node"):
                module_file += "_node"
            
            imports.append(f"from nodes.{module_file} import {module}")
        
        imports_str = "\n".join(imports)
        
        # Generate the flow creation function
        modules_list = ", ".join([f"{module}()" for module in modules])
        
        content = f'''"""
Flow definition for the agent.

This module provides the create_flow function, which creates the agent's flow.
"""

from num_agents.core import Flow

{imports_str}


def create_flow() -> Flow:
    """
    Create the agent's flow.
    
    Returns:
        The agent's flow
    """
    return Flow([
        {modules_list}
    ])
'''
        
        # Write the file
        with open(file_path, "w") as f:
            f.write(content)
    
    def _generate_shared_store_file(self) -> None:
        """Generate the shared_store.py file."""
        file_path = os.path.join(self.output_dir, "shared_store.py")
        
        content = '''"""
Shared store for the agent.

This module provides a factory function for creating the agent's shared store.
"""

from num_agents.core import SharedStore


def create_shared_store() -> SharedStore:
    """
    Create the agent's shared store.
    
    Returns:
        The agent's shared store
    """
    shared = SharedStore()
    
    # Initialize any shared data here
    
    return shared
'''
        
        # Write the file
        with open(file_path, "w") as f:
            f.write(content)
    
    def _generate_main_file(self) -> None:
        """Generate the main.py file."""
        file_path = os.path.join(self.output_dir, "main.py")
        
        agent_name = self.agent_spec_loader.get_agent_name()
        
        content = f'''"""
Main entry point for the {agent_name} agent.

This module provides the main function for running the agent.
"""

from flow import create_flow
from shared_store import create_shared_store


def main() -> None:
    """Run the agent."""
    # Create the flow
    flow = create_flow()
    
    # Create the shared store
    shared = create_shared_store()
    flow.shared = shared
    
    # Execute the flow
    results = flow.execute()
    
    # Print the results
    print("Flow execution completed.")
    print("Results:")
    for node_name, node_results in results.items():
        print(f"  {node_name}: {node_results}")


if __name__ == "__main__":
    main()
'''
        
        # Write the file
        with open(file_path, "w") as f:
            f.write(content)
    
    def _generate_readme(self) -> None:
        """Generate a basic README.md file."""
        file_path = os.path.join(self.output_dir, "README.md")
        
        agent_name = self.agent_spec_loader.get_agent_name()
        agent_description = self.agent_spec_loader.get_agent_description() or "No description provided."
        
        content = f'''# {agent_name}

{agent_description}

## Overview

This agent was generated using the Nüm Agents SDK.

## Structure

- `main.py`: Main entry point for running the agent
- `flow.py`: Definition of the agent's flow
- `shared_store.py`: Factory for creating the agent's shared store
- `nodes/`: Directory containing the agent's node implementations
- `agent.yaml`: Agent specification

## Usage

```bash
# Run the agent
python main.py
```
'''
        
        # Write the file
        with open(file_path, "w") as f:
            f.write(content)
    
    @staticmethod
    def _to_snake_case(name: str) -> str:
        """
        Convert a name to snake_case.
        
        Args:
            name: The name to convert
            
        Returns:
            The name in snake_case
        """
        result = ""
        for i, char in enumerate(name):
            if char.isupper() and i > 0:
                result += "_"
            result += char.lower()
        return result
