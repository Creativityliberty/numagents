"""
Complete Intelligent Agent - End-to-End Example

This is a complete example showing all Nüm Agents SDK features:
- LLM integration (OpenAI GPT)
- KnowledgeLayer for semantic memory
- StructureAgentIA for goal and task management
- Monitoring with metrics and tracing
- Resilience with retry and circuit breaker
- Flow orchestration

Copyright (c) 2025 Lionel TAGNE. All Rights Reserved.
This software is proprietary and confidential.
"""

import os
from typing import Any, Dict, List, Optional

from num_agents.core import Flow, Node, SharedStore
from num_agents.modules.knowledge_layer import (
    KnowledgeStore,
    Memory,
    MemoryStoreNode,
    MemoryRecallNode,
    OpenAIEmbeddingProvider,
)
from num_agents.modules.structure_agent_ia import (
    GoalManager,
    TaskManager,
    InputParserNode,
    GoalPlannerNode,
    Priority,
)
from num_agents.modules.monitoring import Monitor, get_monitor
from num_agents.modules.resilience import (
    RetryConfig,
    CircuitBreakerConfig,
    ResilientNode,
)


# ============================================================================
# LLM Integration Node
# ============================================================================


class LLMReasoningNode(ResilientNode):
    """
    Node that uses LLM for reasoning and decision making.

    Uses OpenAI GPT models with resilience patterns.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 500,
        **kwargs
    ) -> None:
        """
        Initialize LLM reasoning node.

        Args:
            api_key: OpenAI API key
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            **kwargs: Additional arguments for ResilientNode
        """
        # Configure resilience
        retry_config = RetryConfig(max_attempts=3, initial_delay=1.0)
        circuit_breaker_config = CircuitBreakerConfig(failure_threshold=5, timeout=60.0)

        super().__init__(
            name="LLMReasoning",
            retry_config=retry_config,
            circuit_breaker_config=circuit_breaker_config,
            timeout_seconds=30.0,
            **kwargs
        )

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize monitor
        self.monitor = get_monitor()

    def execute(self, shared: SharedStore) -> Dict[str, Any]:
        """Execute LLM reasoning with monitoring and resilience."""
        try:
            import openai
        except ImportError:
            return {
                "response": "OpenAI library not installed. Install with: pip install openai",
                "error": True
            }

        # Get input
        user_input = shared.get("user_input", "")
        context = shared.get("context", "")
        memories = shared.get("recalled_memories", [])

        # Build prompt with context
        prompt = self._build_prompt(user_input, context, memories)

        # Track with monitoring
        trace_id = shared.get("trace_id")
        if not trace_id:
            trace_id = self.monitor.tracer.start_trace("llm_reasoning")
            shared.set("trace_id", trace_id)

        with self.monitor.tracer.span("llm_api_call") as span:
            span.set_attribute("model", self.model)
            span.set_attribute("temperature", self.temperature)

            # Time the API call
            with self.monitor.metrics.timer("llm.api.duration"):
                # Call OpenAI API
                client = openai.OpenAI(api_key=self.api_key)

                response = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful AI assistant with access to memory and task management."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )

                # Track metrics
                self.monitor.metrics.counter("llm.requests.total")
                self.monitor.metrics.histogram("llm.tokens.used", response.usage.total_tokens)

                # Extract response
                llm_response = response.choices[0].message.content

                span.set_attribute("tokens_used", response.usage.total_tokens)
                span.set_attribute("response_length", len(llm_response))

        return {
            "llm_response": llm_response,
            "tokens_used": response.usage.total_tokens,
            "model": self.model
        }

    def _build_prompt(self, user_input: str, context: str, memories: List[Memory]) -> str:
        """Build prompt with context and memories."""
        prompt_parts = []

        if context:
            prompt_parts.append(f"Context: {context}")

        if memories:
            prompt_parts.append("\nRelevant memories:")
            for i, mem in enumerate(memories[:3], 1):
                prompt_parts.append(f"{i}. {mem.content}")

        prompt_parts.append(f"\nUser: {user_input}")
        prompt_parts.append("\nAssistant:")

        return "\n".join(prompt_parts)


# ============================================================================
# Intelligent Agent
# ============================================================================


class IntelligentAgent:
    """
    Complete intelligent agent with LLM, memory, goals, and monitoring.

    This agent can:
    - Understand natural language input
    - Remember past conversations (KnowledgeLayer)
    - Plan and execute goals/tasks (StructureAgentIA)
    - Reason with LLM (OpenAI GPT)
    - Monitor performance (Monitoring)
    - Handle failures gracefully (Resilience)
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        enable_monitoring: bool = True,
        enable_memory: bool = True,
    ) -> None:
        """
        Initialize intelligent agent.

        Args:
            openai_api_key: OpenAI API key
            model: LLM model to use
            enable_monitoring: Enable monitoring
            enable_memory: Enable knowledge memory
        """
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.model = model

        # Initialize components
        self.monitor = Monitor(service_name="intelligent_agent") if enable_monitoring else None

        # Knowledge Layer for memory
        if enable_memory and self.api_key:
            embedding_provider = OpenAIEmbeddingProvider(api_key=self.api_key)
            self.knowledge_store = KnowledgeStore(
                embedding_provider=embedding_provider,
                dimension=1536  # OpenAI embedding dimension
            )
        else:
            self.knowledge_store = None

        # Goal and Task management
        self.goal_manager = GoalManager()
        self.task_manager = TaskManager()

        # Build flow
        self.flow = self._build_flow()

    def _build_flow(self) -> Flow:
        """Build the agent's processing flow."""
        nodes = []

        # 1. Parse input
        parser = InputParserNode(
            input_key="user_input",
            output_key="parsed_input"
        )
        nodes.append(parser)

        # 2. Recall relevant memories (if enabled)
        if self.knowledge_store:
            recall = MemoryRecallNode(
                knowledge_store=self.knowledge_store,
                query_key="user_input",
                output_key="recalled_memories",
                top_k=3
            )
            nodes.append(recall)

        # 3. LLM reasoning
        llm = LLMReasoningNode(
            api_key=self.api_key,
            model=self.model
        )
        nodes.append(llm)

        # 4. Store interaction in memory (if enabled)
        if self.knowledge_store:
            store = MemoryStoreNode(
                knowledge_store=self.knowledge_store,
                content_key="user_input",
                metadata_key="llm_response"
            )
            nodes.append(store)

        # 5. Plan goals/tasks if needed
        planner = GoalPlannerNode(
            goal_manager=self.goal_manager,
            task_manager=self.task_manager,
            input_key="parsed_input",
            output_key="plan"
        )
        nodes.append(planner)

        return Flow(nodes=nodes)

    def chat(self, user_input: str, context: str = "") -> Dict[str, Any]:
        """
        Process user input and generate response.

        Args:
            user_input: User's input message
            context: Optional context

        Returns:
            Agent's response with metadata
        """
        # Start monitoring trace
        if self.monitor:
            trace_id = self.monitor.tracer.start_trace("agent_chat")
            self.monitor.metrics.counter("agent.chats.total")

        # Execute flow
        with self.monitor.tracer.span("flow_execution") if self.monitor else nullcontext():
            result = self.flow.execute(initial_data={
                "user_input": user_input,
                "context": context,
                "trace_id": trace_id if self.monitor else None
            })

        # Extract response
        llm_result = result.get("LLMReasoning", {})
        response = llm_result.get("llm_response", "I couldn't process that request.")

        # Build response
        response_data = {
            "response": response,
            "tokens_used": llm_result.get("tokens_used", 0),
            "model": llm_result.get("model", self.model),
        }

        # Add memory info if enabled
        if self.knowledge_store:
            memories = result.get("MemoryRecall", {}).get("memories", [])
            response_data["memories_recalled"] = len(memories)

        # Add plan info if created
        plan = result.get("GoalPlannerNode", {}).get("plan_created")
        if plan:
            response_data["plan_created"] = True

        return response_data

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        stats = {
            "goals": {
                "total": self.goal_manager.count(),
                "active": len(self.goal_manager.get_active_goals())
            },
            "tasks": {
                "total": self.task_manager.count(),
                "ready": len(self.task_manager.get_ready_tasks())
            }
        }

        if self.knowledge_store:
            stats["memory"] = {
                "total_memories": len(self.knowledge_store.memories)
            }

        if self.monitor:
            monitor_stats = self.monitor.get_statistics()
            stats["monitoring"] = monitor_stats

        return stats


# ============================================================================
# Context manager for null context
# ============================================================================


from contextlib import contextmanager

@contextmanager
def nullcontext():
    """Null context manager."""
    yield


# ============================================================================
# Main - Interactive Demo
# ============================================================================


def main():
    """Run interactive demo of intelligent agent."""
    print("=" * 70)
    print("INTELLIGENT AGENT - Complete End-to-End Demo")
    print("=" * 70)

    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n⚠️  Warning: OPENAI_API_KEY not found in environment")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        print("\nRunning in limited mode (no LLM or memory)...\n")

    # Create agent
    print("\n🤖 Initializing Intelligent Agent...")
    agent = IntelligentAgent(
        openai_api_key=api_key,
        model="gpt-3.5-turbo",
        enable_monitoring=True,
        enable_memory=bool(api_key)
    )
    print("✅ Agent initialized!")

    # Interactive loop
    print("\n" + "=" * 70)
    print("Chat with the agent (type 'quit' to exit, 'stats' for statistics)")
    print("=" * 70 + "\n")

    conversation_history = []

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() == 'quit':
                break

            if user_input.lower() == 'stats':
                stats = agent.get_stats()
                print("\n📊 Agent Statistics:")
                print(f"  Goals: {stats['goals']['total']} total, {stats['goals']['active']} active")
                print(f"  Tasks: {stats['tasks']['total']} total, {stats['tasks']['ready']} ready")
                if 'memory' in stats:
                    print(f"  Memories: {stats['memory']['total_memories']} stored")
                if 'monitoring' in stats:
                    print(f"  Metrics: {stats['monitoring'].get('metrics_count', 0)} collected")
                print()
                continue

            # Get response
            response_data = agent.chat(user_input)

            # Display response
            print(f"\nAgent: {response_data['response']}")

            # Show metadata
            if response_data.get('tokens_used'):
                print(f"  [Tokens: {response_data['tokens_used']}]", end="")
            if response_data.get('memories_recalled'):
                print(f" [Memories: {response_data['memories_recalled']}]", end="")
            if response_data.get('plan_created'):
                print(f" [Plan created]", end="")
            print("\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")

    # Final stats
    print("\n" + "=" * 70)
    print("Session Summary")
    print("=" * 70)
    stats = agent.get_stats()
    print(f"Goals created: {stats['goals']['total']}")
    print(f"Tasks created: {stats['tasks']['total']}")
    if 'memory' in stats:
        print(f"Memories stored: {stats['memory']['total_memories']}")


if __name__ == "__main__":
    main()
