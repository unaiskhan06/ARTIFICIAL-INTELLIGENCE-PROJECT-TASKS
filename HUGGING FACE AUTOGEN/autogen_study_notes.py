"""
Part 3: AutoGen Basics - Study Notes
=====================================

AutoGen is Microsoft's open-source framework for building AI agent systems.
This file documents the key concepts studied from the official docs:
https://microsoft.github.io/autogen/stable/

--- Core Architecture ---

AutoGen v0.4+ has three layers:

1. autogen-core
   - Event-driven runtime for multi-agent systems
   - Defines protocols: Agent, Runtime, Message
   - Handles messaging, serialization, cancellation

2. autogen-agentchat  (built on core)
   - High-level API for conversational agents
   - AssistantAgent, Team patterns, termination conditions
   - This is what most developers use

3. autogen-ext
   - Extensions for external services
   - Model clients (OpenAI, Azure, Anthropic, Ollama, etc.)
   - Code executors, MCP workbench

--- Key Components ---

AssistantAgent:
  - An agent powered by an LLM model client
  - Has a system_message that defines its role/persona
  - Processes messages and generates responses

Teams (Group Chat patterns):
  - RoundRobinGroupChat: agents take turns in order (simplest)
  - SelectorGroupChat: an LLM picks the next speaker
  - Swarm: agents hand off to each other via HandoffMessage
  - MagenticOneGroupChat: generalist multi-agent for complex tasks

Termination Conditions:
  - TextMentionTermination: stops when a keyword appears (e.g. "APPROVE")
  - MaxMessageTermination: stops after N messages
  - ExternalTermination: stops via external signal
  - Conditions can be combined with & (AND) or | (OR)

Model Clients:
  - OpenAIChatCompletionClient: works with OpenAI AND any OpenAI-compatible API
    (Groq, Together, Gemini, local servers, etc.)
  - Set base_url to point to a compatible endpoint
  - AnthropicChatCompletionClient, OllamaChatCompletionClient, etc.

--- How a Multi-Agent System Works ---

1. Create a model client (e.g., OpenAIChatCompletionClient with Groq's base_url)
2. Create AssistantAgent instances with different system_messages (roles)
3. Create a Team (e.g., RoundRobinGroupChat) with the agents
4. Set a termination condition (e.g., TextMentionTermination("APPROVE"))
5. Call team.run(task="...") or team.run_stream(task="...")
6. Agents take turns, each seeing all previous messages
7. When termination condition is met, a TaskResult is returned

--- Research Agent Pattern (what we build in Part 4) ---

  User Task -> Research Agent -> Analysis Agent -> Review Agent -> Final Agent
       |           |               |               |              |
       |     gathers info     analyzes findings   reviews quality  writes final answer
       |
  Reviewer says "APPROVE" when satisfied -> system stops

This is the "reflection" pattern extended to a pipeline:
  - Each agent specializes in one step
  - The reviewer provides feedback
  - If not approved, the cycle repeats (round-robin)
  - When approved, the final agent produces the answer
"""
