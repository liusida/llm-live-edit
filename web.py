"""Web UI for the self-editing agent.

    uv run python web.py                          # Qwen3.5-9B on http://127.0.0.1:8000
    uv run python web.py --model Qwen/Qwen3.5-2B --port 8001
"""
from swm.web import main

if __name__ == "__main__":
    main()
