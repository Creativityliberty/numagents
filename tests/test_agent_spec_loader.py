"""
Tests for the AgentSpecLoader class.
"""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from num_agents.utils.file_io import AgentSpecLoader


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
def agent_spec_file(sample_agent_spec):
    """Create a temporary agent specification file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(sample_agent_spec, f)
        spec_path = f.name
    
    yield spec_path
    
    # Clean up
    os.unlink(spec_path)


def test_load_agent_spec(agent_spec_file, sample_agent_spec):
    """Test loading an agent specification from a file."""
    loader = AgentSpecLoader(agent_spec_file)
    spec = loader.load()
    
    assert spec == sample_agent_spec
    assert "agent" in spec
    assert "name" in spec["agent"]
    assert "description" in spec["agent"]
    assert "univers" in spec["agent"]


def test_get_agent_name(agent_spec_file):
    """Test getting the agent name from the specification."""
    loader = AgentSpecLoader(agent_spec_file)
    loader.load()
    
    name = loader.get_agent_name()
    
    assert name == "Test Agent"


def test_get_agent_description(agent_spec_file):
    """Test getting the agent description from the specification."""
    loader = AgentSpecLoader(agent_spec_file)
    loader.load()
    
    description = loader.get_agent_description()
    
    assert description == "A test agent for unit testing"


def test_get_agent_universes(agent_spec_file):
    """Test getting the agent universes from the specification."""
    loader = AgentSpecLoader(agent_spec_file)
    loader.load()
    
    universes = loader.get_agent_universes()
    
    assert universes == ["core", "memory"]


def test_get_agent_protocol(agent_spec_file):
    """Test getting the agent protocol from the specification."""
    loader = AgentSpecLoader(agent_spec_file)
    loader.load()
    
    protocol = loader.get_agent_protocol()
    
    assert protocol == "standard"


def test_get_agent_llm(agent_spec_file):
    """Test getting the agent LLM configuration from the specification."""
    loader = AgentSpecLoader(agent_spec_file)
    loader.load()
    
    llm = loader.get_agent_llm()
    
    assert llm == {
        "provider": "openai",
        "model": "gpt-4"
    }


def test_get_agent_memory(agent_spec_file):
    """Test getting the agent memory configuration from the specification."""
    loader = AgentSpecLoader(agent_spec_file)
    loader.load()
    
    memory = loader.get_agent_memory()
    
    assert memory == {
        "type": "simple"
    }
