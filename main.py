"""Chat with a Qwen3.5 agent (default 9B; --model Qwen/Qwen3.5-2B for the small one) that can read and edit its own weights.

    uv run python main.py                       # interactive
    uv run python main.py --task "..."          # one task, then exit

Interactive commands (yours, not the model's): /reset  /status  /quit
"""
import argparse
import os

os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")  # keep current default; silences Triton 3.9 warning

from swm import SelfEditAgent, load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--task", help="run one task non-interactively")
    ap.add_argument("--think", action="store_true", help="enable thinking mode")
    ap.add_argument("--max-steps", type=int, default=20, help="max tool-call turns per user message")
    args = ap.parse_args()

    model, tok = load(args.model)
    agent = SelfEditAgent(model, tok, think=args.think)
    print(f"logging to {agent.log_path}")

    if args.task:
        agent.chat(args.task, max_steps=args.max_steps)
        return
    while True:
        try:
            text = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text in ("/quit", "/exit"):
            break
        if text == "/reset":
            print(f"restored {agent.human_reset()} tensor(s); the model has been told")
            continue
        if text == "/status":
            print(f"pending: {len(agent.store.pending)}  applied: {len(agent.store.applied)}")
            for n, d in agent.store.drift().items():
                print(f"  {n}: |delta| = {d:.4g}")
            continue
        if text:
            agent.chat(text, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
