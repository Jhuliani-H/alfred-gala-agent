"""Test Alfred on the GAIA benchmark from Unit 4 of the Hugging Face Agents Course.

The course's scoring API serves 20 level-1 GAIA questions and grades answers by
exact match (ignoring case and surrounding spaces):
https://huggingface.co/learn/agents-course/unit4/hands-on

Usage:
    python gaia.py                  # Alfred with his GAIA tools answers all 20 questions
    python gaia.py --tools gala     # stock Alfred with only his gala tools (baseline)
    python gaia.py --limit 3        # only the first 3 questions
    python gaia.py --submit --username YOUR_HF_NAME --agent-code https://github.com/YOU/REPO/tree/main

Answers are saved to gaia_results/ (one file per tool set and model) after
every question, so an interrupted run picks up where it stopped.
Submitting puts the score on the public leaderboard.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download
from smolagents import ActionStep, CodeAgent, FinalAnswerStep, Tool

from agent import build_alfred, build_model, tools_used
from gaia_tools import VisitWebpageTool, YouTubeTranscriptTool, read_attachment
from tools import SafeWebSearchTool

API_URL = "https://agents-course-unit4-scoring.hf.space"
FILES_DIR = Path("gaia_files")
RESULTS_DIR = Path("gaia_results")

# Added to every question: GAIA grades by exact match, so format matters as much as facts.
ANSWER_RULES = (
    "Finish by calling final_answer with ONLY the answer, no explanation. It is graded by exact match:\n"
    "- a number: digits only, no thousands separators, no units unless the question asks for them\n"
    "- a word or name: as few words as possible, no articles, no abbreviations\n"
    "- a list: items separated by a comma and a space, in the order the question asks for\n"
    "Follow any extra format the question gives (e.g. 'first name only', 'two decimal places')."
)

GAIA_INSTRUCTIONS = (
    "You are Alfred, a meticulous butler who researches hard questions for his master. "
    "Use web_search to find pages, visit_webpage to read them (pass focus keywords for long "
    "pages such as Wikipedia), and youtube_transcript for what is said in a YouTube video "
    "whose youtube.com link is in the question. Attached files are already pasted into the "
    "question. Solve puzzles, logic and maths by writing Python code.\n"
    "Call the tools with exactly these keyword arguments:\n"
    '  web_search(query="Eiffel Tower height")\n'
    '  visit_webpage(url="https://en.wikipedia.org/wiki/Eiffel_Tower", focus="height metres")\n'
    '  youtube_transcript(url="<the youtube.com link from the question>")\n'
    "Every tool returns a plain string: print it and read it, don't index into it like a dict. "
    "Never call final_answer in the same code block as a tool: print the tool's output, "
    "read it, and answer in a later step."
)


class ReadBeforeAnswering:
    """Makes Alfred read tool results before he answers.

    Small models often write one code block that searches and then calls
    final_answer straight away, answering from output they never saw. This
    tracks tool calls and rejects an answer given in the same step as a call
    whose output the model hasn't read yet; the model then sees the printed
    output and the rejection, and tries again.
    """

    def __init__(self, tools: list[Tool]):
        self.read = set()  # tool calls whose output the model has seen in an earlier step
        self.unread = set()  # new tool calls made in the current step
        for tool in tools:
            tool.forward = self._tracked(tool.name, tool.forward)

    def _tracked(self, name: str, forward):
        def wrapper(*args, **kwargs):
            call = (name, repr(args), repr(sorted(kwargs.items())))
            if call not in self.read:
                self.unread.add(call)
            return forward(*args, **kwargs)

        return wrapper

    def check(self, final_answer, memory, agent=None) -> bool:
        if self.unread:
            raise ValueError(
                "final_answer was called in the same code block as a tool, before its "
                "output was read. Read the output above, then call final_answer on its own."
            )
        return True

    def end_step(self, step, agent=None) -> None:
        self.read |= self.unread
        self.unread.clear()

    def end_run(self, step, agent=None) -> None:
        self.read.clear()
        self.unread.clear()


def build_gaia_alfred(model_id: str | None = None) -> CodeAgent:
    """Alfred with research tools instead of his gala tools."""
    tools = [SafeWebSearchTool(max_results=5), VisitWebpageTool(), YouTubeTranscriptTool()]
    guard = ReadBeforeAnswering(tools)
    return CodeAgent(
        tools=tools,
        model=build_model(model_id),
        instructions=GAIA_INSTRUCTIONS,
        additional_authorized_imports=["pandas", "numpy"],
        max_steps=10,
        final_answer_checks=[guard.check],
        step_callbacks={ActionStep: guard.end_step, FinalAnswerStep: guard.end_run},
    )


def fetch_questions() -> list[dict]:
    response = requests.get(f"{API_URL}/questions", timeout=60)
    response.raise_for_status()
    return response.json()


def download_attachment(task: dict) -> Path | None:
    """Fetch the file attached to a question, or None if it can't be had.

    The scoring API's /files endpoint is the official source but can return 404.
    The same files are in the gated gaia-benchmark/GAIA dataset, which works
    once HF_TOKEN belongs to an account that accepted the dataset's terms.
    """
    name = task["file_name"]
    path = FILES_DIR / name
    if path.exists():
        return path
    response = requests.get(f"{API_URL}/files/{task['task_id']}", timeout=60)
    if response.ok:
        FILES_DIR.mkdir(exist_ok=True)
        path.write_bytes(response.content)
        return path
    try:
        return Path(
            hf_hub_download(
                "gaia-benchmark/GAIA",
                f"2023/validation/{name}",
                repo_type="dataset",
                token=os.getenv("HF_TOKEN"),
            )
        )
    except Exception:
        return None


def build_task(task: dict) -> str:
    """The question, plus its attachment's contents and the answer rules."""
    parts = [task["question"]]
    name = task.get("file_name")
    if name:
        path = download_attachment(task)
        if path is None:
            parts.append(f"The question refers to an attached file '{name}', but it could not be downloaded.")
        else:
            try:
                content = read_attachment(path)
            except Exception as e:
                content = f"(The file could not be read: {type(e).__name__}: {e})"
            parts.append(f"Attached file '{name}' (saved at {path.resolve()}):\n{content}")
    parts.append(ANSWER_RULES)
    return "\n\n".join(parts)


def clean_answer(answer) -> str:
    """Tidy a final answer for exact-match grading."""
    if isinstance(answer, (list, tuple, set)):
        answer = ", ".join(str(a) for a in answer)
    elif isinstance(answer, float) and answer.is_integer():
        answer = int(answer)
    text = str(answer).strip()
    # After max steps the model answers in plain text, sometimes as "final_answer('x')".
    text = re.sub(r"^final[_ ]answer\s*\((.*)\)$", r"\1", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"^final[_ ]answer\s*[:\-]?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)  # 1,234 -> 1234
    text = re.sub(r"\s*,\s*", ", ", text)  # a,b -> a, b
    return re.sub(r"^[\"'\s]+|[\"'.\s]+$", "", text)  # surrounding quotes, trailing period


POLISH_PROMPT = """You check answers for a quiz that is graded by exact string match.

Question:
{question}

Proposed answer: {answer}

Rewrite the proposed answer so it follows the format the question asks for (e.g. "first name \
only", "city name", "two decimal places", "comma separated", "alphabetical order") and these rules:
- a number: digits only, no thousands separators, no units unless the question asks for them
- a word or name: no articles, no abbreviations (spell abbreviated words out)
- a list: items separated by a comma and a space
Never change the facts in the answer or add new information. If it already follows the format, \
repeat it unchanged. Reply with the answer only."""


def polish_answer(agent: CodeAgent, question: str, answer: str) -> str:
    """One extra LLM call that fixes the answer's format (not its content).

    Small models often find the right fact but ignore format instructions
    like 'first name only' or 'without abbreviations', which exact match punishes.
    """
    if not answer:
        return answer
    prompt = POLISH_PROMPT.format(question=question, answer=answer)
    try:
        reply = agent.model.generate([{"role": "user", "content": [{"type": "text", "text": prompt}]}])
    except Exception:
        return answer
    polished = clean_answer(reply.content or "")
    # A long reply means the model explained instead of reformatting, and a list
    # that gained or lost items had its content changed: keep the original then.
    if not polished or len(polished) > 2 * len(answer) + 20 or polished.count(",") != answer.count(","):
        return answer
    return polished


def answer_questions(questions: list[dict], agent: CodeAgent, results_path: Path, polish: bool = True) -> dict:
    """Run Alfred on each question not yet in results_path, saving after each one."""
    results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else {}
    for n, task in enumerate(questions, 1):
        if task["task_id"] in results:
            continue
        print(f"\n===== [{n}/{len(questions)}] {task['question'][:120]}", flush=True)
        start = time.time()
        error = None
        try:
            raw = agent.run(build_task(task))  # run() resets memory: questions don't leak into each other
        except Exception as e:
            raw, error = "", f"{type(e).__name__}: {e}"
        answer = clean_answer(raw)
        results[task["task_id"]] = {
            "question": task["question"],
            "file_name": task.get("file_name") or None,
            "submitted_answer": polish_answer(agent, task["question"], answer) if polish else answer,
            "raw_answer": str(raw),
            "tools_used": [t for t in tools_used(agent) if t != "final_answer"],
            "steps": sum(isinstance(s, ActionStep) for s in agent.memory.steps),
            "seconds": round(time.time() - start),
            "model": agent.model.model_id,
            "error": error,
        }
        results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"===== Answer: {results[task['task_id']]['submitted_answer']!r}", flush=True)
    return results


def print_summary(results: dict) -> None:
    print(f"\n{'Answer':<32} {'Steps':>5} {'Time':>6}  Tools | Question")
    for r in results.values():
        tools = ", ".join(r["tools_used"]) or "-"
        print(f"{r['submitted_answer'][:32]:<32} {r['steps']:>5} {r['seconds']:>5}s  {tools} | {r['question'][:50]!r}")


def submit(results: dict, username: str, agent_code: str) -> None:
    payload = {
        "username": username.strip(),
        "agent_code": agent_code,
        "answers": [{"task_id": tid, "submitted_answer": r["submitted_answer"]} for tid, r in results.items()],
    }
    response = requests.post(f"{API_URL}/submit", json=payload, timeout=120)
    if not response.ok:
        sys.exit(f"Submission failed ({response.status_code}): {response.text}")
    data = response.json()
    print(f"\nScore: {data['score']}% ({data['correct_count']} correct)\n{data['message']}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tools", choices=["gaia", "gala"], default="gaia", help="Alfred's GAIA research tools, or his stock gala tools")
    parser.add_argument("--model", help="override the model id for the chosen backend")
    parser.add_argument("--limit", type=int, help="only answer the first N questions")
    parser.add_argument("--submit", action="store_true", help="submit the saved answers to the leaderboard")
    parser.add_argument("--username", help="your Hugging Face username (needed with --submit)")
    parser.add_argument("--agent-code", help="public link to your code, e.g. .../tree/main (needed with --submit)")
    args = parser.parse_args()
    if args.submit and not (args.username and args.agent_code):
        parser.error("--submit needs --username and --agent-code")

    questions = fetch_questions()[: args.limit]
    agent = build_gaia_alfred(args.model) if args.tools == "gaia" else build_alfred(args.model)
    RESULTS_DIR.mkdir(exist_ok=True)
    model_name = re.sub(r"[^\w.-]", "_", agent.model.model_id)  # e.g. qwen3:8b -> qwen3_8b
    results_path = RESULTS_DIR / f"answers_{args.tools}_{model_name}.json"
    # The stock gala baseline stays as plain as possible: no format pass.
    results = answer_questions(questions, agent, results_path, polish=args.tools == "gaia")
    print_summary(results)
    if args.submit:
        submit(results, args.username, args.agent_code)
