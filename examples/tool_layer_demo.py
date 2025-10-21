"""
ToolLayer Demo - Demonstrating Tool/Action capabilities

This example shows how to:
1. Register Python functions as tools
2. Execute tools through a registry
3. Chain tools together
4. Integrate tools with Flows

Copyright (c) 2025 Lionel TAGNE. All Rights Reserved.
This software is proprietary and confidential.
"""

from num_agents import (
    ToolRegistry,
    ToolExecutor,
    PythonFunctionTool,
    ChainedTool,
    ToolExecuteNode,
    Flow,
    Node,
    SharedStore,
)


# ============================================================================
# Example 1: Basic Tool Registration and Execution
# ============================================================================


def search_web(query: str) -> dict:
    """Simulate web search."""
    # In real usage, this would call a search API
    return {
        "query": query,
        "results": [
            f"Result 1 for {query}",
            f"Result 2 for {query}",
            f"Result 3 for {query}",
        ],
    }


def send_email(to: str, subject: str, body: str) -> dict:
    """Simulate sending email."""
    # In real usage, this would send an actual email
    return {"status": "sent", "to": to, "subject": subject}


def query_database(table: str, conditions: str) -> dict:
    """Simulate database query."""
    # In real usage, this would query a real database
    return {
        "table": table,
        "conditions": conditions,
        "rows": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
    }


def basic_tool_usage():
    """Demonstrate basic tool registration and execution."""
    print("=" * 60)
    print("Example 1: Basic Tool Usage")
    print("=" * 60)

    # Create registry
    registry = ToolRegistry(enable_logging=True)

    # Register tools
    registry.register_function("search_web", search_web)
    registry.register_function("send_email", send_email)
    registry.register_function("query_db", query_database)

    print(f"\n✅ Registered {registry.count()} tools")
    print(f"📋 Available tools: {registry.list_tools()}")

    # Create executor
    executor = ToolExecutor(registry, enable_logging=True)

    # Execute tools
    print("\n🔧 Executing 'search_web' tool...")
    result1 = executor.execute("search_web", query="Python AI agents")
    print(f"Result: {result1['result']}")

    print("\n📧 Executing 'send_email' tool...")
    result2 = executor.execute(
        "send_email", to="user@example.com", subject="Hello", body="Test message"
    )
    print(f"Status: {result2['result']['status']}")

    # View execution history
    print("\n📊 Execution History:")
    for entry in executor.get_history():
        print(
            f"  - {entry['tool_name']}: {entry['result'] if entry['success'] else entry['error']}"
        )


# ============================================================================
# Example 2: Custom Tools
# ============================================================================


def analyze_sentiment(text: str) -> str:
    """Simple sentiment analysis."""
    positive_words = ["good", "great", "excellent", "amazing", "love"]
    negative_words = ["bad", "terrible", "awful", "hate", "poor"]

    text_lower = text.lower()
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)

    if pos_count > neg_count:
        return "positive"
    elif neg_count > pos_count:
        return "negative"
    else:
        return "neutral"


def extract_keywords(text: str, top_n: int = 5) -> list:
    """Simple keyword extraction."""
    # Very basic implementation
    words = text.lower().split()
    # Filter out common words
    stopwords = {"the", "a", "an", "and", "or", "but", "in", "on", "at"}
    keywords = [w for w in words if w not in stopwords and len(w) > 3]
    return list(set(keywords))[:top_n]


def custom_tools_demo():
    """Demonstrate creating custom tools."""
    print("\n" + "=" * 60)
    print("Example 2: Custom Tools")
    print("=" * 60)

    registry = ToolRegistry()

    # Register with custom descriptions
    registry.register_function(
        "sentiment",
        analyze_sentiment,
        description="Analyze the sentiment of text (positive/negative/neutral)",
    )

    registry.register_function(
        "keywords",
        extract_keywords,
        description="Extract important keywords from text",
    )

    executor = ToolExecutor(registry)

    # Test sentiment analysis
    text = "This is a great and amazing product! I love it!"
    print(f"\n📝 Analyzing: '{text}'")

    sentiment = executor.execute("sentiment", text=text)
    print(f"😊 Sentiment: {sentiment['result']}")

    keywords = executor.execute("keywords", text=text, top_n=3)
    print(f"🔑 Keywords: {keywords['result']}")


# ============================================================================
# Example 3: Tool Chaining
# ============================================================================


def fetch_data(source: str) -> dict:
    """Fetch data from a source."""
    return {"source": source, "data": "Raw data from " + source}


def process_data(input: dict) -> dict:
    """Process fetched data."""
    return {"processed": True, "data": input.get("data", "").upper()}


def save_data(input: dict) -> dict:
    """Save processed data."""
    return {"saved": True, "location": "/data/output.json"}


def tool_chaining_demo():
    """Demonstrate chaining tools together."""
    print("\n" + "=" * 60)
    print("Example 3: Tool Chaining")
    print("=" * 60)

    # Create individual tools
    fetch_tool = PythonFunctionTool("fetch", fetch_data)
    process_tool = PythonFunctionTool("process", process_data)
    save_tool = PythonFunctionTool("save", save_data)

    # Create a chain
    pipeline = ChainedTool(
        name="data_pipeline",
        tools=[fetch_tool, process_tool, save_tool],
        description="Fetch -> Process -> Save pipeline",
    )

    print("\n🔗 Created tool chain: fetch → process → save")

    # Execute chain
    result = pipeline.execute(source="api.example.com")
    print(f"\n✅ Pipeline result: {result}")


# ============================================================================
# Example 4: Flow Integration
# ============================================================================


class UserInputNode(Node):
    """Node that simulates user input."""

    def exec(self, shared: SharedStore) -> dict:
        user_query = "What is the weather in Paris?"
        shared.set("user_query", user_query)
        print(f"\n👤 User: {user_query}")
        return {"query_received": True}


class ResponseNode(Node):
    """Node that formats the final response."""

    def exec(self, shared: SharedStore) -> dict:
        tool_result = shared.get("tool_result")
        search_results = tool_result.get("result", {}).get("results", [])

        response = f"Here's what I found:\n"
        for i, result in enumerate(search_results, 1):
            response += f"  {i}. {result}\n"

        shared.set("response", response)
        print(f"\n🤖 Agent: {response}")
        return {"response_generated": True}


def flow_integration_demo():
    """Demonstrate integrating tools with Flows."""
    print("\n" + "=" * 60)
    print("Example 4: Flow Integration - Agent with Tools")
    print("=" * 60)

    # Setup tools
    registry = ToolRegistry()
    registry.register_function("search", search_web)

    executor = ToolExecutor(registry)

    # Build flow
    flow = Flow(name="AgentWithTools")

    # Nodes
    input_node = UserInputNode(name="UserInput")
    tool_node = ToolExecuteNode(
        executor=executor,
        tool_name="search",
        input_key="user_query",
        output_key="tool_result",
        name="SearchTool",
    )
    response_node = ResponseNode(name="ResponseGenerator")

    # Add to flow
    flow.add_node(input_node)
    flow.add_node(tool_node)
    flow.add_node(response_node)

    # Set start
    flow.set_start(input_node)

    # Execute
    print("\n🚀 Executing agent flow...")
    results = flow.execute()

    print("\n📊 Flow Results:")
    for node_name, node_result in results.items():
        if isinstance(node_result, dict):
            print(f"  {node_name}: {list(node_result.keys())}")


# ============================================================================
# Example 5: Advanced - Tool Discovery and Search
# ============================================================================


def advanced_tool_usage():
    """Demonstrate advanced tool features."""
    print("\n" + "=" * 60)
    print("Example 5: Advanced - Tool Discovery")
    print("=" * 60)

    registry = ToolRegistry()

    # Register many tools
    tools = {
        "math_add": lambda a, b: a + b,
        "math_multiply": lambda a, b: a * b,
        "text_uppercase": lambda s: s.upper(),
        "text_lowercase": lambda s: s.lower(),
        "data_filter": lambda items, key: [i for i in items if key in str(i)],
    }

    for name, func in tools.items():
        registry.register_function(name, func)

    print(f"\n📚 Total tools registered: {registry.count()}")

    # Search for tools
    print("\n🔍 Searching for 'math' tools:")
    math_tools = registry.search("math")
    for tool in math_tools:
        print(f"  - {tool.name}: {tool.description}")

    print("\n🔍 Searching for 'text' tools:")
    text_tools = registry.search("text")
    for tool in text_tools:
        print(f"  - {tool.name}: {tool.description}")

    # Export registry
    registry_data = registry.to_dict()
    print(f"\n📦 Registry can be exported: {len(registry_data['tools'])} tools")


# ============================================================================
# Main Demo
# ============================================================================


def main():
    """Run all demonstrations."""
    print("\n")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║         ToolLayer Demo - Agent Tool Management            ║")
    print("╚════════════════════════════════════════════════════════════╝")

    # Run examples
    basic_tool_usage()
    custom_tools_demo()
    tool_chaining_demo()
    flow_integration_demo()
    advanced_tool_usage()

    print("\n" + "=" * 60)
    print("✨ Demo Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("  1. Tools wrap Python functions for agent use")
    print("  2. ToolRegistry manages tool catalog")
    print("  3. ToolExecutor handles execution with safety")
    print("  4. Tools integrate seamlessly with Flows")
    print("  5. Support for tool chaining and discovery")
    print("\n")


if __name__ == "__main__":
    main()
