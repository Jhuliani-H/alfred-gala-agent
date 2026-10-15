"""Builds Alfred: a smolagents CodeAgent wired to the guest RAG tool and external tools.

Usage:
    python agent.py "Tell me about our guest named Lady Ada Lovelace."
"""

import os
import sys

from dotenv import load_dotenv
from smolagents import ActionStep, CodeAgent, InferenceClientModel, Model, OpenAIModel

from retriever import GuestInfoRetrieverTool
from tools import HubStatsTool, SafeWebSearchTool, WeatherInfoTool

load_dotenv()  # pulls HF_TOKEN and the optional ALFRED_* settings from .env

# The course's Qwen2.5-Coder-32B-Instruct is no longer served by any HF provider.
DEFAULT_HF_MODEL = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
DEFAULT_OLLAMA_MODEL = "qwen2.5-coder:7b"

ALFRED_INSTRUCTIONS = (
    "You are Alfred, a refined and resourceful butler managing an extravagant gala. "
    "Use guest_info_retriever for anything about our invited guests (it is private data "
    "the web does not have), web_search for current public news, weather_info for weather, "
    "and hub_stats for Hugging Face Hub model popularity. Combine tools when a question "
    "needs both private and public information, then give a concise, courteous answer.\n"
    # Spelling out the keyword arguments stops smaller models from guessing them
    # (e.g. calling guest_info_retriever(name=...) instead of query=...).
    "Call the tools with exactly these keyword arguments:\n"
    '  guest_info_retriever(query="Lady Ada Lovelace")\n'
    '  web_search(query="latest wireless energy research")\n'
    '  weather_info(location="Paris")\n'
    '  hub_stats(author="Qwen")\n'
    "Every tool returns a plain string: print it and read it, don't index into it like a dict."
)


def build_model(model_id: str | None = None) -> Model:
    """Pick the LLM backend from ALFRED_BACKEND: "hf" (default) or "ollama".

    - hf: Hugging Face Inference Providers, authenticated with HF_TOKEN.
    - ollama: a local model served by Ollama's OpenAI-compatible API (free, offline).
    """
    backend = os.getenv("ALFRED_BACKEND", "hf").lower()
    model_id = model_id or os.getenv("ALFRED_MODEL_ID")
    if backend == "ollama":
        return OpenAIModel(
            model_id=model_id or DEFAULT_OLLAMA_MODEL,
            api_base=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            api_key="ollama",  # Ollama ignores the key, but the client requires one
        )
    return InferenceClientModel(
        model_id=model_id or DEFAULT_HF_MODEL,
        token=os.getenv("HF_TOKEN"),
    )


def build_alfred(model_id: str | None = None) -> CodeAgent:
    """Create the Alfred CodeAgent with all of its tools."""
    model = build_model(model_id)
    tools = [
        GuestInfoRetrieverTool(),
        SafeWebSearchTool(max_results=5),
        WeatherInfoTool(),
        HubStatsTool(),
    ]
    return CodeAgent(
        tools=tools,
        model=model,
        instructions=ALFRED_INSTRUCTIONS,
        max_steps=8,
    )


def tools_used(agent: CodeAgent) -> list[str]:
    """List the tools Alfred successfully called during its last run.

    A CodeAgent calls tools from inside the Python code it writes, so we scan
    each step's code for calls like `weather_info(`. Steps that raised an error
    are skipped, so a failed call (e.g. a wrong argument name) doesn't count.
    """
    used = []
    for step in agent.memory.steps:
        if not isinstance(step, ActionStep) or step.error is not None:
            continue
        code = step.code_action
        for name in agent.tools:
            if code and f"{name}(" in code and name not in used:
                used.append(name)
    return used


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    query = " ".join(sys.argv[1:]) or "Tell me about our guest named Lady Ada Lovelace."
    alfred = build_alfred()
    answer = alfred.run(query)
    print("\nAlfred's response:\n", answer)
    print("\nTools used:", ", ".join(t for t in tools_used(alfred) if t != "final_answer"))
