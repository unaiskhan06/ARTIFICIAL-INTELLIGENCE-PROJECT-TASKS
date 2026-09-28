"""
Task 5 - FastAPI Backend: Multimodal AutoGen Agent
===================================================
FastAPI server that runs a multimodal multi-agent system using AutoGen.
Handles both text and image inputs.

Architecture:
  - Image is described using HuggingFace's BLIP model locally (image -> text)
  - The description + user query is sent to an AutoGen agent team:
      VisionAgent  -> interprets the image description
      ResearchAgent -> gathers relevant knowledge
      AnalysisAgent -> analyzes and synthesizes
      FinalAgent    -> produces the final answer
  - Powered by Groq API (OpenAI-compatible)

Run:
  python api.py          (starts on port 8000)
"""

import asyncio
import os
import sys
import io
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Image captioning - BLIP model loaded locally
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# AutoGen for multi-agent system
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from autogen_agentchat.base import TaskResult
from autogen_ext.models.openai import OpenAIChatCompletionClient

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "openai/gpt-oss-20b"
BLIP_MODEL = "Salesforce/blip-image-captioning-base"

app = FastAPI(title="Multimodal AutoGen Agent API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Load BLIP model once at startup (image -> text caption)
# --------------------------------------------------------------------------- #
print("Loading BLIP image captioning model...")
_blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL)
_blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL)
print("BLIP model loaded.\n")


def describe_image(image_bytes: bytes) -> str:
    """Generate a text caption from an image using local BLIP model."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = _blip_processor(img, return_tensors="pt")
        output = _blip_model.generate(**inputs, max_new_tokens=50)
        caption = _blip_processor.decode(output[0], skip_special_tokens=True)
        return caption.strip()
    except Exception as e:
        return f"(Could not analyze image: {e})"


# --------------------------------------------------------------------------- #
# Groq / AutoGen helpers
# --------------------------------------------------------------------------- #
def get_groq_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key and key != "your-key-here":
        return key
    # Try .env file directly
    env_path = Path(__file__).with_name(".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("GROQ_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                if key and key != "your-key-here":
                    return key
    sys.exit("No GROQ_API_KEY in .env")


def create_model_client() -> OpenAIChatCompletionClient:
    return OpenAIChatCompletionClient(
        model=GROQ_MODEL,
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


def create_agent_team() -> RoundRobinGroupChat:
    """Build the multimodal multi-agent team."""
    client = create_model_client()

    vision_agent = AssistantAgent(
        name="VisionAgent",
        model_client=client,
        system_message=(
            "You are a Vision Agent. You receive an image description "
            "and the user's question. Explain what's in the image in detail. "
            "Keep it under 150 words."
        ),
    )

    research_agent = AssistantAgent(
        name="ResearchAgent",
        model_client=client,
        system_message=(
            "You are a Research Agent. Based on the image description and "
            "user question, provide relevant factual knowledge. "
            "Keep it under 150 words."
        ),
    )

    analysis_agent = AssistantAgent(
        name="AnalysisAgent",
        model_client=client,
        system_message=(
            "You are an Analysis Agent. Synthesize the vision and research "
            "into key insights. Keep it under 150 words."
        ),
    )

    final_agent = AssistantAgent(
        name="FinalAgent",
        model_client=client,
        system_message=(
            "You are the Final Agent. Write a clear, polished answer to the "
            "user's question based on all the team's work. Keep it under 250 words. "
            "End with 'FINAL_ANSWER_COMPLETE' on the last line."
        ),
    )

    termination = TextMentionTermination("FINAL_ANSWER_COMPLETE") | \
                  MaxMessageTermination(10)

    return RoundRobinGroupChat(
        participants=[vision_agent, research_agent, analysis_agent, final_agent],
        termination_condition=termination,
    )


# --------------------------------------------------------------------------- #
# API Routes
# --------------------------------------------------------------------------- #
class TextQuery(BaseModel):
    query: str


@app.get("/")
def root():
    return {"status": "ok", "service": "Multimodal AutoGen Agent API"}


@app.post("/api/text")
async def handle_text(data: TextQuery):
    """Process a text-only query through the agent team."""
    team = create_agent_team()
    messages = []
    stop = ""

    async for msg in team.run_stream(task=data.query):
        if isinstance(msg, TaskResult):
            stop = msg.stop_reason
        else:
            messages.append({
                "agent": getattr(msg, "source", "?"),
                "content": getattr(msg, "content", str(msg)),
            })

    return {"messages": messages, "stop_reason": stop}


@app.post("/api/multimodal")
async def handle_multimodal(
    query: str = Form(...),
    image: UploadFile = File(None),
):
    """Process a text + optional image query through the agent team."""
    image_description = ""
    if image:
        img_bytes = await image.read()
        image_description = describe_image(img_bytes)

    # Build the combined task for the agents.
    if image_description:
        task = (
            f"User question: {query}\n\n"
            f"Image description (from BLIP vision model): {image_description}"
        )
    else:
        task = query

    team = create_agent_team()
    messages = []
    stop = ""

    async for msg in team.run_stream(task=task):
        if isinstance(msg, TaskResult):
            stop = msg.stop_reason
        else:
            messages.append({
                "agent": getattr(msg, "source", "?"),
                "content": getattr(msg, "content", str(msg)),
            })

    return {
        "messages": messages,
        "image_description": image_description,
        "stop_reason": stop,
    }


if __name__ == "__main__":
    print("\n  FastAPI backend running at http://127.0.0.1:8000")
    print("  Docs at http://127.0.0.1:8000/docs\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)
