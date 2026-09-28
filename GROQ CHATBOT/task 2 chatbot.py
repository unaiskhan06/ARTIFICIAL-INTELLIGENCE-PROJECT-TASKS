"""
Groq Chatbot - Terminal Edition
===============================
A clean, interactive terminal chatbot powered by the Groq API.

Features:
  - Streaming responses (tokens appear as they're generated).
  - Switch between Groq models on the fly.
  - Customizable system prompt.
  - Conversation history with /save to export a transcript.
  - Live token-usage + latency stats per reply.
  - Slash commands: /help /model /system /clear /save /history /exit

Setup:
  1. pip install groq python-dotenv
  2. Copy .env.example to .env and paste your Groq API key
     (get one free at https://console.groq.com/keys)
  3. python chatbot.py
"""

import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # .env is optional; env var still works.

try:
    from groq import Groq
except ImportError:
    sys.exit("Missing dependency: run  pip install groq python-dotenv")



# ANSI colors (no external dependency).
 
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    MAGENTA = "\033[35m"
    BLUE = "\033[34m"


# Enable ANSI on Windows.
if sys.platform == "win32":
    os.system("")



# Available Groq chat models (subset; user can switch at runtime).

MODELS = {
    "1": ("openai/gpt-oss-20b", "GPT-OSS 20B (fast, balanced)"),
    "2": ("openai/gpt-oss-120b", "GPT-OSS 120B (largest, most capable)"),
    "3": ("qwen/qwen3.8-27b", "Qwen 3.8 27B"),
    "4": ("groq/compound-mini", "Groq Compound Mini (agentic)"),
    "5": ("allam-2-7b", "Allam 2 7B (Arabic-tuned)"),
}

DEFAULT_SYSTEM = (
    "You are a helpful, concise assistant. Answer clearly and to the point."
)



# Chatbot core.
 
class GroqChatbot:
    def __init__(self, api_key: str, model: str, system: str):
        self.client = Groq(api_key=api_key)
        self.model = model
        self.system = system
        self.history: list[dict] = [
            {"role": "system", "content": system}
        ]
        self.total_tokens = 0

    #  core call  #
    def stream_reply(self, user_text: str):
        """Yield token chunks as they arrive; return usage stats at the end."""
        self.history.append({"role": "user", "content": user_text})
        start = time.time()
        collected = []

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=self.history,
            stream=True,
            temperature=0.7,
        )

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                piece = chunk.choices[0].delta.content
                collected.append(piece)
                yield piece

        elapsed = time.time() - start
        full = "".join(collected)
        self.history.append({"role": "assistant", "content": full})
        # Estimate tokens from text length (no usage in stream for this SDK).
        est_tokens = max(1, len(full) // 4)
        self.total_tokens += est_tokens
        yield ("__STATS__", est_tokens, elapsed)

    # helpers #
    def clear(self):
        self.history = [{"role": "system", "content": self.system}]
        self.total_tokens = 0

    def set_system(self, text: str):
        self.system = text
        self.clear()

    def save_transcript(self, path: str):
        Path(path).write_text(
            json.dumps(self.history, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )



# Terminal UI.#

def banner():
    print(f"\n{C.BOLD}{C.MAGENTA}=== Groq Chatbot ==={C.RESET}")
    print(f"{C.DIM}Type /help for commands, /exit to quit.{C.RESET}\n")


def prompt_user() -> str:
    return input(f"{C.BOLD}{C.GREEN}you> {C.RESET}").strip()


def print_help():
    cmds = [
        ("/help", "show this help"),
        ("/model", "list and switch Groq models"),
        ("/system <text>", "set a new system prompt (resets history)"),
        ("/clear", "clear conversation history"),
        ("/history", "show message count + token usage"),
        ("/save <file>", "save transcript to JSON file"),
        ("/exit", "quit the chatbot"),
    ]
    print(f"\n{C.BOLD}Commands:{C.RESET}")
    for cmd, desc in cmds:
        print(f"  {C.CYAN}{cmd:<16}{C.RESET} {desc}")
    print()


def choose_model(current: str) -> str:
    print(f"\n{C.BOLD}Available models:{C.RESET}")
    for key, (mid, label) in MODELS.items():
        marker = " *" if mid == current else ""
        print(f"  [{key}] {label}{marker}")
    choice = input(f"{C.DIM}select number (Enter to keep current): {C.RESET}").strip()
    if choice in MODELS:
        print(f"{C.GREEN}Switched to {MODELS[choice][1]}{C.RESET}\n")
        return MODELS[choice][0]
    return current


def get_api_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key:
        return key
    # Try .env file directly as a fallback.
    env_path = Path(__file__).with_name(".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("GROQ_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                if key and key != "your-key-here":
                    return key
    print(f"{C.RED}No GROQ_API_KEY found.{C.RESET}")
    print(f"  1. Get a free key at {C.CYAN}https://console.groq.com/keys{C.RESET}")
    print(f"  2. Copy .env.example to .env and paste your key there.")
    print(f"  3. Or set it in your shell:  set GROQ_API_KEY=your-groq-api-key-here{C.RESET}")
    sys.exit(1)


def main():
    api_key = get_api_key()
    model = MODELS["1"][0]
    bot = GroqChatbot(api_key, model, DEFAULT_SYSTEM)

    banner()
    print(f"{C.DIM}Model: {model}{C.RESET}\n")

    while True:
        try:
            text = prompt_user()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{C.DIM}Bye!{C.RESET}")
            break

        if not text:
            continue

        #  slash commands  #
        if text.startswith("/"):
            cmd, *rest = text[1:].split(maxsplit=1)
            arg = rest[0] if rest else ""

            if cmd == "exit":
                print(f"{C.DIM}Bye!{C.RESET}")
                break
            elif cmd == "help":
                print_help()
            elif cmd == "model":
                bot.model = choose_model(bot.model)
            elif cmd == "system":
                if not arg:
                    print(f"{C.YELLOW}Usage: /system <prompt text>{C.RESET}")
                else:
                    bot.set_system(arg)
                    print(f"{C.GREEN}System prompt updated, history cleared.{C.RESET}\n")
            elif cmd == "clear":
                bot.clear()
                print(f"{C.GREEN}History cleared.{C.RESET}\n")
            elif cmd == "history":
                msgs = len(bot.history) - 1  # exclude system
                print(f"{C.CYAN}messages: {msgs}  total tokens: {bot.total_tokens}{C.RESET}\n")
            elif cmd == "save":
                if not arg:
                    print(f"{C.YELLOW}Usage: /save <filename>{C.RESET}")
                else:
                    fname = arg if arg.endswith(".json") else arg + ".json"
                    bot.save_transcript(fname)
                    print(f"{C.GREEN}Saved transcript to {fname}{C.RESET}\n")
            else:
                print(f"{C.RED}Unknown command: /{cmd}. Try /help.{C.RESET}\n")
            continue

        #  chat turn  #
        try:
            print(f"{C.BOLD}{C.BLUE}assistant> {C.RESET}", end="", flush=True)
            stats = None
            for item in bot.stream_reply(text):
                if isinstance(item, tuple) and item[0] == "__STATS__":
                    stats = item
                else:
                    print(item, end="", flush=True)
            print()  # newline after streamed reply
            if stats:
                _, est_tokens, elapsed = stats
                print(
                    f"{C.DIM}  [~{est_tokens} tokens, "
                    f"{elapsed:.2f}s, model: {bot.model}]{C.RESET}"
                )
            print()
        except KeyboardInterrupt:
            print(f"\n{C.YELLOW}(interrupted){C.RESET}\n")
        except Exception as e:
            print(f"\n{C.RED}Error: {e}{C.RESET}\n")


if __name__ == "__main__":
    main()
