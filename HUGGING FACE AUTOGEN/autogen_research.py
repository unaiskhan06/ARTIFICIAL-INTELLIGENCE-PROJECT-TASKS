"""
Part 4: Multi-Agent Research System with AutoGen
=================================================
A multi-agent communication system for research purposes, built with
Microsoft AutoGen and powered by the Groq API.

Agent Pipeline:
  Research Agent  -> gathers key facts and information on the topic
  Analysis Agent  -> synthesizes and analyzes the research findings
  Review Agent    -> reviews quality, provides feedback, says APPROVE when satisfied
  Final Agent     -> produces the polished final research summary

Flow:
  1. User provides a research topic
  2. Agents collaborate in a RoundRobinGroupChat
  3. Research -> Analysis -> Review (feedback loop until APPROVE)
  4. Final Agent writes the definitive answer
  5. System terminates when "FINAL_ANSWER_COMPLETE" is detected

Setup:
  - Groq API key in .env file (GROQ_API_KEY=...)
  - pip install autogen-agentchat autogen-ext[openai] python-dotenv

Usage:
  python autogen_research.py --topic "The impact of AI on healthcare"
  python autogen_research.py --topic "Renewable energy trends in 2024" --model openai/gpt-oss-120b
"""

import asyncio
import argparse
import os
import sys
from pathlib import Path

# Fix Windows console encoding for Unicode output from agents.
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.base import TaskResult
from autogen_ext.models.openai import OpenAIChatCompletionClient


# Groq's OpenAI-compatible endpoint.
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "openai/gpt-oss-20b"


def get_groq_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key and key != "your-key-here":
        return key
    env_path = Path(__file__).with_name(".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("GROQ_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                if key and key != "your-key-here":
                    return key
    print("No GROQ_API_KEY found. Put it in the .env file.")
    sys.exit(1)


def create_model_client(model: str) -> OpenAIChatCompletionClient:
    """Create a Groq-backed model client (OpenAI-compatible)."""
    return OpenAIChatCompletionClient(
        model=model,
        api_key=get_groq_key(),
        base_url=GROQ_BASE_URL,
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": "unknown",
            "structured_output": False,
        },
    )


def create_research_team(model: str) -> RoundRobinGroupChat:
    """Build the 4-agent research team."""
    client = create_model_client(model)

    # --- Agent 1: Research Agent ---
    # Gathers facts, key points, and background info on the topic.
    # Kept concise to stay within Groq free-tier token limits.
    research_agent = AssistantAgent(
        name="ResearchAgent",
        model_client=client,
        system_message=(
            "You are a Research Agent. Gather key facts about the topic. "
            "Keep your response under 200 words. Use 5-7 bullet points "
            "with the most important findings. Be factual and concise. "
            "Do NOT make up facts."
        ),
    )

    # --- Agent 2: Analysis Agent ---
    # Synthesizes research into insights and patterns.
    analysis_agent = AssistantAgent(
        name="AnalysisAgent",
        model_client=client,
        system_message=(
            "You are an Analysis Agent. Analyze the research findings. "
            "Keep your response under 200 words. Identify 3-4 key themes "
            "and their implications. Be analytical and concise."
        ),
    )

    # --- Agent 3: Review Agent ---
    # Reviews quality and provides feedback. Says APPROVE when satisfied.
    review_agent = AssistantAgent(
        name="ReviewAgent",
        model_client=client,
        system_message=(
            "You are a Review Agent. Review the research and analysis. "
            "Keep your response under 100 words. "
            "If the work is adequate, respond with exactly 'APPROVE'. "
            "Otherwise, briefly list what needs improvement."
        ),
    )

    # --- Agent 4: Final Answer Agent ---
    # Produces the polished final summary when review is approved.
    final_agent = AssistantAgent(
        name="FinalAgent",
        model_client=client,
        system_message=(
            "You are the Final Answer Agent. Write a polished research "
            "summary based on the team's work. Keep it under 300 words. "
            "Include: a title, key findings, and conclusions. "
            "End with the exact phrase 'FINAL_ANSWER_COMPLETE' on the last line."
        ),
    )

    # --- Termination: stop when final answer is complete ---
    # Max 12 messages as a safety net (3 rounds of 4 agents).
    termination = TextMentionTermination("FINAL_ANSWER_COMPLETE") | \
                  TextMentionTermination("APPROVE") | \
                  MaxMessageTermination(12)

    # --- Team: agents take turns in round-robin order ---
    team = RoundRobinGroupChat(
        participants=[research_agent, analysis_agent, review_agent, final_agent],
        termination_condition=termination,
    )

    return team


async def run_research(topic: str, model: str):
    """Run the multi-agent research system on a topic."""
    print("=" * 65)
    print("  Multi-Agent Research System (AutoGen + Groq)")
    print("=" * 65)
    print(f"  Topic:  {topic}")
    print(f"  Model:  {model}")
    print(f"  Agents: ResearchAgent -> AnalysisAgent -> ReviewAgent -> FinalAgent")
    print("=" * 65)
    print("\n  Agents are collaborating... (streaming output below)\n")

    team = create_research_team(model)

    # Stream agent messages live in the terminal (encoding-safe).
    # Includes retry logic for Groq free-tier rate limits (8000 TPM).
    max_retries = 3
    result = None

    for attempt in range(max_retries):
        try:
            async for message in team.run_stream(task=topic):
                if isinstance(message, TaskResult):
                    result = message
                else:
                    source = getattr(message, "source", "?")
                    content = getattr(message, "content", str(message))
                    print(f"\n--- {source} ---")
                    print(content)
            break  # Success - exit retry loop.
        except RuntimeError as e:
            if "rate_limit" in str(e).lower() or "413" in str(e):
                wait = (attempt + 1) * 20
                print(f"\n  [Rate limit hit. Waiting {wait}s before retry "
                      f"({attempt + 1}/{max_retries})...]")
                await asyncio.sleep(wait)
                # Reset team for a fresh run.
                await team.reset()
            else:
                raise
    else:
        print("\n  Max retries reached. Try again later or upgrade your Groq tier.")
        return None

    if result is None:
        print("No result returned.")
        return None

    print("\n" + "=" * 65)
    print("  Research Complete!")
    print("=" * 65)
    print(f"  Total messages exchanged: {len(result.messages)}")
    print(f"  Stop reason: {result.stop_reason}")
    print("=" * 65)

    # Print the final agent's output.
    final_messages = [
        m for m in result.messages
        if hasattr(m, "source") and m.source == "FinalAgent"
    ]
    if final_messages:
        print("\n--- FINAL RESEARCH REPORT ---\n")
        # Get the last FinalAgent message (the polished version).
        report = final_messages[-1].content
        # Remove the termination marker from the display.
        report = report.replace("FINAL_ANSWER_COMPLETE", "").strip()
        print(report)
        print("\n--- END OF REPORT ---\n")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Research System (AutoGen + Groq)"
    )
    parser.add_argument("--topic", required=True,
                        help="Research topic for the agents to investigate")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Groq model to use (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    asyncio.run(run_research(args.topic, args.model))


if __name__ == "__main__":
    main()
