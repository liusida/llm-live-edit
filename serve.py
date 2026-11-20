"""Serve a model whose weights can be edited live over HTTP, with a browser chat.

    uv run python serve.py                          # Qwen3.5-9B on http://127.0.0.1:8000
    uv run python serve.py --model Qwen/Qwen3.5-0.8B --port 8001
"""
from llm_live_edit.server import main

if __name__ == "__main__":
    main()
