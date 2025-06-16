"""
Universe catalog loader for the Nüm Agents SDK.

This module provides functionality for loading and parsing the universe catalog,
which defines the available universes and their associated modules.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Any

import yaml


class UniversCatalogLoader:
    """
    Loader for the universe catalog.
    
    This class is responsible for loading and parsing the universe catalog YAML file,
    which defines the available universes and their associated modules.
    """
    
    def __init__(self, catalog_path: Optional[str] = None) -> None:
        """
        Initialize the universe catalog loader.
        
        Args:
            catalog_path: Optional path to the universe catalog YAML file.
                          If not provided, the default path will be used.
        """
        self.catalog_path = catalog_path or self._get_default_catalog_path()
        self._catalog: Dict[str, Any] = {}
    
    @staticmethod
    def _get_default_catalog_path() -> str:
        """
        Get the default path to the universe catalog YAML file.
        
        Returns:
            The default path to the universe catalog YAML file
        """
        # Try to find the catalog in the config directory relative to the package
        package_dir = Path(__file__).parent.parent.parent
        default_path = os.path.join(package_dir, "config", "univers_catalog.yaml")
        
        if os.path.exists(default_path):
            return default_path
        
        # If not found, check if it's in the current working directory
        cwd_path = os.path.join(os.getcwd(), "config", "univers_catalog.yaml")
        if os.path.exists(cwd_path):
            return cwd_path
        
        # If still not found, return the default path anyway (it will fail when loaded)
        return default_path
    
    def load(self) -> Dict[str, Any]:
        """
        Load the universe catalog from the YAML file.
        
        Returns:
            The parsed universe catalog as a dictionary
            
        Raises:
            FileNotFoundError: If the catalog file doesn't exist
            yaml.YAMLError: If the catalog file is not valid YAML
        """
        if not os.path.exists(self.catalog_path):
            raise FileNotFoundError(f"Universe catalog not found at {self.catalog_path}")
        
        with open(self.catalog_path, "r") as f:
            self._catalog = yaml.safe_load(f)
        
        return self._catalog
    
    def get_universes(self) -> List[str]:
        """
        Get the list of available universes.
        
        Returns:
            A list of universe names
        """
        if not self._catalog:
            self.load()
        
        return list(self._catalog.get("univers_catalog", {}).keys())
    
    def get_modules_for_universe(self, universe: str) -> List[str]:
        """
        Get the list of modules for a specific universe.
        
        Args:
            universe: The name of the universe
            
        Returns:
            A list of module names for the specified universe
            
        Raises:
            KeyError: If the universe doesn't exist in the catalog
        """
        if not self._catalog:
            self.load()
        
        universe_data = self._catalog.get("univers_catalog", {}).get(universe)
        if not universe_data:
            raise KeyError(f"Universe '{universe}' not found in catalog")
        
        return universe_data.get("modules", [])
    
    def resolve_modules(self, universes: List[str]) -> Set[str]:
        """
        Resolve all modules for a list of universes.
        
        Args:
            universes: A list of universe names
            
        Returns:
            A set of all module names from the specified universes
        """
        if not self._catalog:
            self.load()
        
        modules = set()
        for universe in universes:
            try:
                universe_modules = self.get_modules_for_universe(universe)
                modules.update(universe_modules)
            except KeyError:
                # Skip universes that don't exist
                continue
        
        return modules
