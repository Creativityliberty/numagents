"""
Tests for the UniversCatalogLoader class.
"""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from num_agents.univers.univers_catalog_loader import UniversCatalogLoader


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
def univers_catalog_file(sample_univers_catalog):
    """Create a temporary universe catalog file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(sample_univers_catalog, f)
        catalog_path = f.name
    
    yield catalog_path
    
    # Clean up
    os.unlink(catalog_path)


def test_load_univers_catalog(univers_catalog_file, sample_univers_catalog):
    """Test loading a universe catalog from a file."""
    loader = UniversCatalogLoader(univers_catalog_file)
    catalog = loader.load()
    
    assert catalog == sample_univers_catalog
    assert "univers_catalog" in catalog
    assert "core" in catalog["univers_catalog"]
    assert "memory" in catalog["univers_catalog"]
    assert "advanced" in catalog["univers_catalog"]


def test_resolve_modules_single_universe(univers_catalog_file):
    """Test resolving modules from a single universe."""
    loader = UniversCatalogLoader(univers_catalog_file)
    loader.load()
    
    modules = loader.resolve_modules(["core"])
    
    assert modules == {"ManagerGoalNode", "ToolAdapterNode"}


def test_resolve_modules_multiple_universes(univers_catalog_file):
    """Test resolving modules from multiple universes."""
    loader = UniversCatalogLoader(univers_catalog_file)
    loader.load()
    
    modules = loader.resolve_modules(["core", "memory"])
    
    assert modules == {"ManagerGoalNode", "ToolAdapterNode", "MemoryRecallNode", "MemoryStoreNode"}


def test_resolve_modules_nonexistent_universe(univers_catalog_file):
    """Test resolving modules from a nonexistent universe."""
    loader = UniversCatalogLoader(univers_catalog_file)
    loader.load()
    
    modules = loader.resolve_modules(["nonexistent"])
    
    assert modules == set()


def test_resolve_modules_empty_universes(univers_catalog_file):
    """Test resolving modules from an empty list of universes."""
    loader = UniversCatalogLoader(univers_catalog_file)
    loader.load()
    
    modules = loader.resolve_modules([])
    
    assert modules == set()
