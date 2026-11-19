"""Web UI server: one model, one agent, any number of browser tabs watching.

WebSocket /ws
  client -> {"type": "chat", "text": "...", "max_steps": 20} | {"type": "stop"}
  server -> agent events (token, gen_start, gen_end, assistant, tool_result, state, notice)
            plus hello / user / busy / error. Non-token events are kept so a reloaded
            page can replay the conversation.
REST /api/...  weight inspector and the human's own controls (reset, discard, clear).
"""
from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .agent import SelfEditAgent
from .weights import describe, parse_index

STATIC = Path(__file__).parent / "static"
UI_MAX_VALUES = 65536  # inspector limit (the agent's own read limit stays at 512)


class InspectReq(BaseModel):
    name: str
    index: str = ":"
    delta: bool = False  # show current - original instead of current


class DebugLog:
    """Timestamped trace of what the server is doing, printed to its terminal.
    Generated tokens are streamed raw between a turn's start and end lines."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.mid_line = False  # inside a raw token stream

    def line(self, msg: str) -> None:
        if not self.enabled:
            return
        if self.mid_line:
            sys.stdout.write("\n")
            self.mid_line = False
        sys.stdout.write(f"[{time.strftime('%H:%M:%S')}.{int(time.time() * 1000) % 1000:03d}] {msg}\n")
        sys.stdout.flush()

    def event(self, ev: dict) -> None:
        if not self.enabled:
            return
        t = ev["type"]
        if t == "token":
            sys.stdout.write(ev["text"])
            sys.stdout.flush()
            self.mid_line = not ev["text"].endswith("\n")
        elif t == "user":
            self.line(f"user: {ev['text'][:200]}")
        elif t == "gen_start":
            self.line(f"generating: re-reading {ev['prompt_tokens']} tokens (prefill), then streaming:")
        elif t == "gen_end":
            self.line(f"generation done: {ev['new_tokens']} new tokens; prefill+first token "
                      f"{ev['prefill_seconds']:.2f}s, decode {ev['tok_per_s']:.1f} tok/s, total {ev['seconds']:.2f}s")
        elif t == "assistant":
            calls = ", ".join(c["fn"] for c in ev["calls"]) or "none"
            self.line(f"parsed reply: {len(ev['content'])} chars visible, tool calls: {calls}"
                      + (", INTERRUPTED" if ev.get("interrupted") else ""))
        elif t == "tool_start":
            args = ", ".join(f"{k}={v[:60]!r}" for k, v in ev["args"].items())
            self.line(f"tool {ev['fn']}({args}) running…")
        elif t == "tool_result":
            first = ev["result"].splitlines()[0][:160] if ev["result"] else ""
            self.line(f"tool {ev['fn']} finished in {ev.get('seconds', 0):.2f}s: {first}")
        elif t == "state":
            self.line(f"weight state: {len(ev['pending'])} staged, {len(ev['applied'])} applied"
                      + (f" (computed in {ev['seconds']:.2f}s)" if "seconds" in ev else ""))
        elif t == "env_note":
            self.line(f"note to model: {ev['text']}")
        elif t in ("notice", "error"):
            self.line(f"{t}: {ev['text']}")
        elif t == "busy":
            self.line("model busy" if ev["busy"] else "model idle")


def create_app(model, tok, model_name: str, think: bool = False, debug: bool = True) -> FastAPI:
    app = FastAPI()
    clients: set[WebSocket] = set()
    history: list[dict] = []
    busy = threading.Lock()
    loop_holder: dict = {}
    dbg = DebugLog(debug)

    def broadcast(ev: dict) -> None:
        """Called from the generation thread or the event loop."""
        dbg.event(ev)
        if ev["type"] != "token":
            history.append(ev)
        loop = loop_holder["loop"]
        for ws in list(clients):
            asyncio.run_coroutine_threadsafe(_safe_send(ws, ev), loop)

    async def _safe_send(ws: WebSocket, ev: dict) -> None:
        try:
            await ws.send_json(ev)
        except Exception:
            clients.discard(ws)

    agent = SelfEditAgent(model, tok, think=think, on_event=broadcast)

    def info() -> dict:
        cfg = model.config
        return {"model": model_name, "layers": cfg.num_hidden_layers, "hidden": cfg.hidden_size,
                "think": think, "log": str(agent.log_path),
                "params": sum(p.numel() for p in model.parameters())}

    def run_chat(text: str, max_steps: int) -> None:
        try:
            agent.chat(text, max_steps=max_steps)
        except Exception as ex:
            broadcast({"type": "error", "text": f"{type(ex).__name__}: {ex}"})
        finally:
            busy.release()
            broadcast({"type": "busy", "busy": False})

    @app.on_event("startup")
    async def _startup():
        loop_holder["loop"] = asyncio.get_running_loop()

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        clients.add(ws)
        await ws.send_json({"type": "hello", **info(), "busy": busy.locked(), "instructions": agent.instructions(),
                            "history": history, "state": agent.state()})
        try:
            while True:
                msg = await ws.receive_json()
                if msg.get("type") == "chat":
                    text = (msg.get("text") or "").strip()
                    if not text:
                        continue
                    if not busy.acquire(blocking=False):
                        await ws.send_json({"type": "error", "text": "the model is busy; stop it first"})
                        continue
                    broadcast({"type": "busy", "busy": True})  # the agent emits "user" (after any env note)
                    threading.Thread(target=run_chat, args=(text, int(msg.get("max_steps") or 20)),
                                     daemon=True).start()
                elif msg.get("type") == "stop":
                    dbg.line("stop requested by the browser" + ("" if busy.locked() else " (model was idle)"))
                    agent.stop_event.set()
        except WebSocketDisconnect:
            clients.discard(ws)

    # ---- REST: inspector + human controls -------------------------------------------
    @app.get("/api/context")
    def context():
        """The full prompt the model will read on its next turn (if you sent nothing new)."""
        text = agent.render()
        return {"text": text, "tokens": len(tok(text).input_ids), "messages": len(agent.messages)}

    @app.get("/api/state")
    def state():
        return {**info(), "busy": busy.locked(), "state": agent.state()}

    @app.get("/api/params")
    def params(filter: str = ""):
        return [{"name": n, "shape": list(s)} for n, s in agent.store.list(filter)]

    @app.get("/api/describe")
    def describe_param(name: str):
        try:
            return {"text": describe(name, tuple(agent.store._get(name).shape), model.config)}
        except KeyError as ex:
            raise HTTPException(404, str(ex))

    @app.post("/api/inspect")
    def inspect(req: InspectReq):
        st = agent.store
        try:
            p = st._get(req.name)
            idx = parse_index(req.index, p.ndim)
            cur = p.data[idx].float().cpu()
        except (KeyError, ValueError, IndexError) as ex:
            raise HTTPException(400, str(ex))
        if cur.numel() > UI_MAX_VALUES:
            raise HTTPException(400, f"region {tuple(cur.shape)} has {cur.numel()} values; "
                                     f"the inspector shows at most {UI_MAX_VALUES}")
        vals = cur - st.original(req.name, req.index) if req.delta else cur
        v2 = vals.reshape(-1, vals.shape[-1]) if vals.ndim >= 2 else vals.reshape(1, -1)
        return {"name": req.name, "index": req.index, "shape": list(cur.shape), "grid": list(v2.shape),
                "values": v2.flatten().tolist(), "modified": req.name in st._backup,
                "stats": {"mean": vals.mean().item(), "std": vals.std().item() if vals.numel() > 1 else 0.0,
                          "min": vals.min().item(), "max": vals.max().item(),
                          "n_nonzero": int((vals != 0).sum())}}

    def _human(action: str, fn) -> dict:
        if not busy.acquire(blocking=False):
            dbg.line(f"refused '{action}': model is busy")
            raise HTTPException(409, "the model is busy; stop it first")
        try:
            t0 = time.time()
            out = fn()
            dbg.line(f"you: {action} -> {out} ({time.time() - t0:.2f}s)")
            broadcast({"type": "notice", "text": f"you: {action} ({out})", "weights_changed": action == "reset weights"})
            broadcast({"type": "state", **agent.state()})
            return {"ok": True, "result": out}
        finally:
            busy.release()

    @app.post("/api/reset")
    def reset():
        return _human("reset weights", lambda: f"{agent.human_reset()} tensor(s) restored")

    @app.post("/api/discard")
    def discard():
        return _human("discard staged edits", lambda: f"{agent.human_discard()} discarded")

    @app.post("/api/clear")
    def clear():
        def do():
            del agent.messages[1:]
            history.clear()
            broadcast({"type": "cleared"})
            history.clear()
            return "conversation cleared; weights unchanged"
        return _human("new conversation", do)

    return app


def main():
    import argparse
    import os

    os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")
    import uvicorn

    from . import load

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--quiet", action="store_true", help="no debug trace in the terminal")
    args = ap.parse_args()
    model, tok = load(args.model)
    torch.cuda.synchronize()
    print(f"serving {args.model} on http://{args.host}:{args.port}", flush=True)
    uvicorn.run(create_app(model, tok, args.model, args.think, debug=not args.quiet),
                host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
