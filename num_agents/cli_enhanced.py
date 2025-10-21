#!/usr/bin/env python3
"""
NumAgents Enhanced CLI - Extended Command Line Interface

Provides enhanced commands for creating, running, testing, and managing agents
with all 4 layers (Tool, State, Security, Morphic).

This complements the existing CLI with more interactive and user-friendly commands.

Copyright (c) 2025 Lionel TAGNE. All Rights Reserved.
This software is proprietary and confidential.
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any
import yaml

# Try to import rich for beautiful output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.prompt import Prompt, Confirm
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

__version__ = "0.2.0"

# ============================================================================
# Configuration
# ============================================================================

class EnhancedCLIConfig:
    """Manage enhanced CLI configuration."""

    def __init__(self):
        self.config_dir = Path.home() / ".numagents"
        self.config_file = self.config_dir / "config_enhanced.yaml"
        self.agents_dir = self.config_dir / "agents_enhanced"
        self.logs_dir = self.config_dir / "logs"

        # Create directories
        for dir_path in [self.config_dir, self.agents_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load or create configuration."""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return yaml.safe_load(f) or {}

        default_config = {
            "version": __version__,
            "agents_dir": str(self.agents_dir),
            "logs_dir": str(self.logs_dir),
            "default_layers": {
                "tool": True,
                "state": True,
                "security": True,
                "morphic": True
            },
            "templates": {
                "simple": "Basic agent with minimal setup",
                "morphic": "Goal-oriented agent with RRLA",
                "full": "Full-featured agent with all 4 layers",
                "conversational": "Conversational agent with dialogue management"
            }
        }

        self._save_config(default_config)
        return default_config

    def _save_config(self, config: Dict[str, Any]) -> None:
        """Save configuration."""
        with open(self.config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

# ============================================================================
# Output System
# ============================================================================

class EnhancedOutput:
    """Enhanced output with Rich support."""

    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None

    def banner(self):
        """Show banner."""
        banner_text = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║           🤖  NumAgents SDK - Enhanced CLI  🚀              ║
║                                                              ║
║  AI Agent Development Toolkit with Full Layer Support       ║
║  ToolLayer • StateLayer • SecurityLayer • MorphicLayer      ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
        """
        if self.console:
            self.console.print(banner_text, style="cyan bold")
        else:
            print(banner_text)

    def success(self, text: str):
        """Success message."""
        msg = f"✅ {text}"
        if self.console:
            self.console.print(msg, style="green")
        else:
            print(msg)

    def error(self, text: str):
        """Error message."""
        msg = f"❌ {text}"
        if self.console:
            self.console.print(msg, style="red bold")
        else:
            print(msg)

    def info(self, text: str):
        """Info message."""
        msg = f"ℹ️  {text}"
        if self.console:
            self.console.print(msg, style="blue")
        else:
            print(msg)

    def warning(self, text: str):
        """Warning message."""
        msg = f"⚠️  {text}"
        if self.console:
            self.console.print(msg, style="yellow")
        else:
            print(msg)

    def header(self, text: str):
        """Section header."""
        if self.console:
            self.console.print(f"\n{text}", style="bold cyan")
            self.console.print("═" * len(text), style="cyan")
        else:
            print(f"\n{text}")
            print("=" * len(text))

    def code(self, code: str, language: str = "python"):
        """Display code."""
        if self.console:
            from rich.syntax import Syntax
            syntax = Syntax(code, language, theme="monokai", line_numbers=True)
            self.console.print(syntax)
        else:
            print(code)

    def table(self, data: list, columns: list):
        """Display table."""
        if self.console:
            table = Table(show_header=True, header_style="bold magenta")
            for col in columns:
                table.add_column(col)
            for row in data:
                table.add_row(*[str(row.get(col, "")) for col in columns])
            self.console.print(table)
        else:
            print("\n" + " | ".join(columns))
            print("-" * 70)
            for row in data:
                print(" | ".join([str(row.get(col, ""))[:20] for col in columns]))

output = EnhancedOutput()

# ============================================================================
# Interactive Agent Builder
# ============================================================================

def interactive_create():
    """Interactive agent creation wizard."""
    output.banner()
    output.header("🎨 Interactive Agent Creator")

    if not RICH_AVAILABLE:
        output.warning("Install 'rich' for better experience: pip install rich")

    # Get agent name
    if RICH_AVAILABLE:
        agent_name = Prompt.ask("Agent name", default="my_agent")
    else:
        agent_name = input("Agent name (my_agent): ").strip() or "my_agent"

    # Select template
    output.info("\nAvailable templates:")
    templates = {
        "1": ("simple", "Basic agent - minimal setup"),
        "2": ("morphic", "Morphic agent - goal-oriented with RRLA"),
        "3": ("full", "Full agent - all 4 layers"),
        "4": ("custom", "Custom - choose layers")
    }

    for key, (name, desc) in templates.items():
        print(f"  {key}. {name:15} - {desc}")

    if RICH_AVAILABLE:
        choice = Prompt.ask("Choose template", choices=["1", "2", "3", "4"], default="1")
    else:
        choice = input("Choose template (1): ").strip() or "1"

    template_name = templates[choice][0]

    # Select layers (if custom)
    layers = None
    if template_name == "custom":
        output.info("\nSelect layers to include:")
        layers = []

        layer_options = [
            ("tool", "ToolLayer - Action/tool management"),
            ("state", "StateLayer - State machines & persistence"),
            ("security", "SecurityLayer - Auth, sanitization, audit"),
            ("morphic", "MorphicLayer - Goal-oriented reasoning")
        ]

        for layer_id, layer_desc in layer_options:
            if RICH_AVAILABLE:
                include = Confirm.ask(f"  Include {layer_desc}?", default=True)
            else:
                include = input(f"  Include {layer_desc}? (Y/n): ").lower() != 'n'

            if include:
                layers.append(layer_id)

    # Create agent
    output.header(f"Creating agent: {agent_name}")

    config = EnhancedCLIConfig()
    agent_dir = config.agents_dir / agent_name
    agent_dir.mkdir(parents=True, exist_ok=True)

    # Generate code based on template
    code = generate_agent_code(agent_name, template_name, layers)
    agent_file = agent_dir / f"{agent_name}.py"
    agent_file.write_text(code)

    # Save agent config
    agent_config = {
        "name": agent_name,
        "template": template_name,
        "layers": layers or get_default_layers(template_name),
        "created_at": str(agent_file.stat().st_mtime)
    }

    with open(agent_dir / "config.yaml", 'w') as f:
        yaml.dump(agent_config, f)

    output.success(f"Agent '{agent_name}' created!")
    output.info(f"Location: {agent_file}")
    output.info(f"Template: {template_name}")

    # Show next steps
    output.header("📝 Next Steps")
    print(f"1. View code:   cat {agent_file}")
    print(f"2. Edit code:   nano {agent_file}")
    print(f"3. Run agent:   numagent-enhanced run {agent_name}")
    print(f"4. Test agent:  numagent-enhanced test {agent_name}")

def get_default_layers(template: str) -> list:
    """Get default layers for template."""
    defaults = {
        "simple": [],
        "morphic": ["morphic"],
        "full": ["tool", "state", "security", "morphic"],
        "custom": []
    }
    return defaults.get(template, [])

def generate_agent_code(name: str, template: str, layers: Optional[list]) -> str:
    """Generate agent code based on template."""

    if template == "simple":
        return f'''"""
Simple AI Agent: {name}

Created with NumAgents Enhanced CLI
"""

from num_agents import Flow, Node, SharedStore


class {name.title().replace('_', '')}Agent(Node):
    """Simple agent implementation."""

    def __init__(self):
        super().__init__(name="{name}")

    def exec(self, shared: SharedStore) -> dict:
        """Execute agent logic."""
        task = shared.get("task", "No task")

        # Your logic here
        result = f"Processed: {{task}}"

        shared.set("result", result)
        return {{"status": "success", "result": result}}


def create_agent():
    """Create and return agent flow."""
    flow = Flow(name="{name}")
    agent = {name.title().replace('_', '')}Agent()
    flow.add_node(agent)
    flow.set_start(agent)
    return flow


if __name__ == "__main__":
    flow = create_agent()
    result = flow.execute(initial_data={{"task": "Hello World"}})
    print(f"Result: {{result}}")
'''

    elif template == "morphic":
        return f'''"""
Morphic Reasoning Agent: {name}

Uses MorphicLayer for goal-oriented reasoning with RRLA
"""

from num_agents import (
    Flow, Node, SharedStore,
    ObjetReactif, ObjectifG, FluxPhi,
    RRLA, Memoire, NUMTEMA_PERSONA
)


class {name.title().replace('_', '')}Agent:
    """Morphic reasoning agent."""

    def __init__(self):
        self.name = "{name}"
        self.persona = NUMTEMA_PERSONA
        self.memoire = Memoire(nom=f"{{self.name}}_memory")
        self.rrla = RRLA(memoire=self.memoire, enable_logging=True)

    def execute(self, task: str, context: dict = None):
        """Execute task with morphic reasoning."""

        # 1. Reason about the task
        cor = self.rrla.raisonner(
            probleme=task,
            contexte=context or {{}},
            mode="complet"
        )

        print(f"🧠 Reasoning completed with {{len(cor.get_etapes())}} steps")
        print(f"📊 Confidence: {{cor.to_dict()['confiance_moyenne']:.0%}}")

        # 2. Execute based on reasoning
        # Your implementation here

        return {{
            "reasoning_id": cor.id,
            "confidence": cor.to_dict()["confiance_moyenne"],
            "result": "Task completed"
        }}


def create_agent():
    """Create morphic agent."""
    return {name.title().replace('_', '')}Agent()


if __name__ == "__main__":
    agent = create_agent()
    result = agent.execute("Analyze this problem")
    print(f"Result: {{result}}")
'''

    else:  # full template
        return f'''"""
Full-Featured Agent: {name}

Uses all 4 layers for production-ready agent
"""

from num_agents import (
    # Core
    Flow, Node, SharedStore,
    # Tool Layer
    ToolRegistry, ToolExecutor, PythonFunctionTool,
    # State Layer
    StateMachine, State, StateTransition, StateManager, InMemoryBackend,
    # Security Layer
    SecurityManager, APIKeyAuthenticator, RegexSanitizer, AuditLogger,
    # Morphic Layer
    ObjetReactif, ObjectifG, FluxPhi, RRLA, Memoire, NUMTEMA_PERSONA
)


class {name.title().replace('_', '')}Agent:
    """Full-featured production-ready agent."""

    def __init__(self):
        self.name = "{name}"
        self.setup_tools()
        self.setup_state()
        self.setup_security()
        self.setup_morphic()

    def setup_tools(self):
        """Setup Tool Layer."""
        self.registry = ToolRegistry()

        # Register tools
        def example_tool(text: str) -> str:
            return f"Processed: {{text}}"

        self.registry.register_function("process", example_tool)
        self.executor = ToolExecutor(self.registry)

    def setup_state(self):
        """Setup State Layer."""
        states = [
            State(name="idle"),
            State(name="processing"),
            State(name="done")
        ]

        transitions = [
            StateTransition(from_state="idle", to_state="processing"),
            StateTransition(from_state="processing", to_state="done")
        ]

        machine = StateMachine(
            initial_state="idle",
            states=states,
            transitions=transitions
        )

        self.state_manager = StateManager(machine, InMemoryBackend())
        self.state_manager.start()

    def setup_security(self):
        """Setup Security Layer."""
        auth = APIKeyAuthenticator()
        auth.add_key("demo_key", user_id="user1", roles=["admin"])

        self.security = SecurityManager(
            authenticator=auth,
            sanitizer=RegexSanitizer(),
            audit_logger=AuditLogger()
        )

    def setup_morphic(self):
        """Setup Morphic Layer."""
        self.memoire = Memoire(nom=f"{{self.name}}_memory")
        self.rrla = RRLA(memoire=self.memoire)
        self.persona = NUMTEMA_PERSONA

    def execute(self, task: dict, credentials: dict):
        """Execute task with full layer support."""

        # 1. Authenticate
        auth_result = self.security.authenticate_request(credentials)
        if not auth_result["authenticated"]:
            return {{"error": "Authentication failed"}}

        # 2. State transition
        self.state_manager.transition_to("processing")

        # 3. Reason about task
        cor = self.rrla.raisonner(
            probleme=task.get("description", "Unknown task"),
            mode="rapide"
        )

        # 4. Execute with tools
        tool_result = self.executor.execute(
            "process",
            text=task.get("data", "")
        )

        # 5. Complete
        self.state_manager.transition_to("done")

        return {{
            "status": "success",
            "reasoning": cor.id,
            "tool_result": tool_result,
            "final_state": self.state_manager.state_machine.get_current_state()
        }}


def create_agent():
    """Create full-featured agent."""
    return {name.title().replace('_', '')}Agent()


if __name__ == "__main__":
    agent = create_agent()
    result = agent.execute(
        task={{"description": "Process data", "data": "Hello"}},
        credentials={{"api_key": "demo_key"}}
    )
    print(f"Result: {{result}}")
'''

# ============================================================================
# CLI Commands
# ============================================================================

def cmd_status(args):
    """Show system status."""
    output.banner()
    output.header("🔧 System Status")

    # Check SDK
    try:
        import num_agents
        sdk_version = num_agents.__version__
        output.success(f"SDK Version: {sdk_version}")
    except:
        output.error("SDK not found")
        return

    # Check layers
    output.header("📦 Available Layers")

    layers = [
        ("ToolLayer", "_TOOL_LAYER_AVAILABLE"),
        ("StateLayer", "_STATE_LAYER_AVAILABLE"),
        ("SecurityLayer", "_SECURITY_LAYER_AVAILABLE"),
        ("MorphicLayer", "_MORPHIC_LAYER_AVAILABLE"),
    ]

    layer_data = []
    for layer_name, var_name in layers:
        try:
            available = getattr(num_agents, var_name, False)
            status = "✅ Available" if available else "❌ Not Available"
            layer_data.append({"Layer": layer_name, "Status": status})
        except:
            layer_data.append({"Layer": layer_name, "Status": "❌ Not Available"})

    output.table(layer_data, ["Layer", "Status"])

    # Show config
    config = EnhancedCLIConfig()
    output.header("⚙️  Configuration")
    output.info(f"Agents directory: {config.agents_dir}")
    output.info(f"Logs directory: {config.logs_dir}")
    output.info(f"Config file: {config.config_file}")

def cmd_wizard(args):
    """Launch interactive wizard."""
    interactive_create()

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="numagent-enhanced",
        description="NumAgents Enhanced CLI - Full-Featured Agent Development",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Wizard command (interactive)
    wizard_parser = subparsers.add_parser('wizard', aliases=['w'],
                                          help='Interactive agent creation wizard')

    # Status command
    status_parser = subparsers.add_parser('status', aliases=['s'],
                                          help='Show system status')

    args = parser.parse_args()

    if args.command in ['wizard', 'w']:
        cmd_wizard(args)
    elif args.command in ['status', 's']:
        cmd_status(args)
    else:
        output.banner()
        output.info("Available commands:")
        output.info("  wizard (w)  - Interactive agent creation")
        output.info("  status (s)  - Show system status")
        output.info("\nFor help: numagent-enhanced --help")

if __name__ == '__main__':
    main()
