"""LLM Live Edit server: one model in memory, an HTTP API that edits its weights, and a
browser chat that runs on whatever the weights are right now.

HTTP /api/...  read, edit, transform, apply, reset weights; probe the model (generate, next_token)
WebSocket /ws  the browser chat
  client -> {"type": "chat", "text": "..."} | {"type": "stop"}
  server -> token, gen_start, gen_end, assistant, user, api, state, notice, busy, error, hello.
            Non-token events are kept so a reloaded page can replay the conversation.
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

from .expr import ExprError, summarize
from .session import ChatSession
from .weights import describe, parse_index, release_memory

STATIC = Path(__file__).parent / "static"
MAX_VALUES = 65536  # largest region /api/read and /api/inspect return


class OnReq(BaseModel):
    on: bool


class EditReq(BaseModel):
    name: str
    index: str = ":"
    values: str | float | list[float]  # one per element, or a single value to broadcast
    mode: str = "set"                  # set | add | scale
    apply: bool = True                 # False: only stage it
    note: str | None = None            # shown in the browser next to the edit


class TransformReq(BaseModel):
    name: str
    index: str = ":"
    expression: str                    # x = the region's current value; see llm_live_edit/expr.py
    apply: bool = True
    note: str | None = None


class ComputeReq(BaseModel):
    expression: str
    max_values: int = 64


class ReadReq(BaseModel):
    name: str
    index: str = ":"


class GenerateReq(BaseModel):
    prompt: str
    max_new_tokens: int = 200
    chat: bool = True                  # wrap in the chat template (no system prompt unless given)
    think: bool = False
    system: str | None = None
    temperature: float = 0.0           # 0 = greedy


class NextTokenReq(BaseModel):
    prompt: str
    top_k: int = 10
    chat: bool = False                 # False: raw text continuation, e.g. "The capital of France is"
    think: bool = False
    targets: list[str] | None = None   # report these strings' first-token probability and rank


class InspectReq(BaseModel):
    name: str
    index: str = ":"
    delta: bool = False  # show current - original instead of current


class DebugLog:
    """Timestamped trace of what the server is doing, printed to its terminal.
    Generated tokens are streamed raw between a reply's start and end lines."""

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
            self.line(f"generating{' (thinking on)' if ev.get('think') else ''}: re-reading "
                      f"{ev['prompt_tokens']} tokens (prefill), then streaming:")
        elif t == "gen_end":
            self.line(f"generation done: {ev['new_tokens']} new tokens; prefill+first token "
                      f"{ev['prefill_seconds']:.2f}s, decode {ev['tok_per_s']:.1f} tok/s, total {ev['seconds']:.2f}s")
        elif t == "assistant":
            self.line(f"reply: {len(ev['content'])} chars" + (", INTERRUPTED" if ev.get("interrupted") else ""))
        elif t == "state":
            self.line(f"weight state: {len(ev['pending'])} staged, {len(ev['applied'])} applied")
        elif t in ("notice", "error"):
            self.line(f"{t}: {ev['text']}")
        elif t == "busy":
            self.line("model busy" if ev["busy"] else "model idle")


def create_app(model, tok, model_name: str, think: bool = False, debug: bool = True,
               max_new_tokens: int = 32768, system: str | None = None) -> FastAPI:
    app = FastAPI(title="LLM Live Edit")
    clients: set[WebSocket] = set()
    history: list[dict] = []
    busy = threading.Lock()  # one thing touches the model at a time: a reply, an edit or a probe
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

    session = ChatSession(model, tok, think=think, on_event=broadcast, max_new_tokens=max_new_tokens,
                          system=system)
    store = session.store

    def info() -> dict:
        cfg = model.config
        return {"model": model_name, "layers": cfg.num_hidden_layers, "hidden": cfg.hidden_size,
                "think": session.think, "system": session.system, "log": str(session.log_path),
                "params": sum(p.numel() for p in model.parameters())}

    def run_chat(text: str) -> None:
        try:
            session.chat(text)
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
        await ws.send_json({"type": "hello", **info(), "busy": busy.locked(), "history": history,
                            "state": session.state()})
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
                    broadcast({"type": "busy", "busy": True})
                    threading.Thread(target=run_chat, args=(text,), daemon=True).start()
                elif msg.get("type") == "stop":
                    dbg.line("stop requested by the browser" + ("" if busy.locked() else " (model was idle)"))
                    session.stop_event.set()
        except WebSocketDisconnect:
            clients.discard(ws)

    # ---- every call that touches the model goes through here ----------------------------
    def _locked(action: str, fn, event: dict | None = None, quiet: bool = False):
        """Run fn while holding the model. Refused (409) while a reply is being generated.
        Bad input -> 400. Unless quiet, the action shows in the browser timeline."""
        if not busy.acquire(blocking=False):
            dbg.line(f"refused '{action}': model is busy")
            raise HTTPException(409, "the model is busy generating; stop it first")
        try:
            t0 = time.time()
            try:
                out = fn()
            except HTTPException:
                raise
            except (ValueError, KeyError, IndexError, ExprError, RuntimeError) as ex:
                dbg.line(f"{action} -> error: {ex}")
                raise HTTPException(400, str(ex))
            dbg.line(f"{action} ({time.time() - t0:.2f}s)")
            if not quiet:
                broadcast(event or {"type": "api", "text": action})
                broadcast({"type": "state", **session.state()})
            return out
        finally:
            release_memory()
            busy.release()

    # ---- weights: look ---------------------------------------------------------------------
    @app.get("/api/state")
    def state():
        return {**info(), "busy": busy.locked(), "state": session.state()}

    @app.get("/api/params")
    def params(filter: str = ""):
        return [{"name": n, "shape": list(s)} for n, s in store.list(filter)]

    @app.get("/api/describe")
    def describe_param(name: str):
        try:
            return {"text": describe(name, tuple(store._get(name).shape), model.config)}
        except KeyError as ex:
            raise HTTPException(404, str(ex))

    @app.post("/api/read")
    def api_read(req: ReadReq):
        def do():
            p = store._get(req.name)
            r = p.data[parse_index(req.index, p.ndim)]
            if r.numel() > MAX_VALUES:
                raise ValueError(f"region has {r.numel()} values; read at most {MAX_VALUES} (use /api/compute for stats)")
            return {"shape": list(r.shape), "values": r.float().flatten().tolist()}
        return _locked(f"read {req.name}[{req.index}]", do, quiet=True)

    @app.post("/api/compute")
    def api_compute(req: ComputeReq):
        """Read-only matrix math over the weights."""
        def do():
            r = store.compute(req.expression)
            out = {"summary": summarize(r, max_values=req.max_values)}
            if isinstance(r, torch.Tensor):
                out["shape"] = list(r.shape)
                if r.numel() == 1:
                    out["value"] = r.item()
                elif r.numel() <= 4096:
                    out["values"] = r.float().flatten().tolist()
            else:
                out["value"] = r
            return out
        return _locked(f"compute {req.expression}", do, quiet=True)

    @app.post("/api/inspect")
    def inspect(req: InspectReq):
        """For the browser's heatmap: a region as a 2-D grid, optionally as its change from the original."""
        try:
            p = store._get(req.name)
            cur = p.data[parse_index(req.index, p.ndim)].float().cpu()
        except (KeyError, ValueError, IndexError) as ex:
            raise HTTPException(400, str(ex))
        if cur.numel() > MAX_VALUES:
            raise HTTPException(400, f"region {tuple(cur.shape)} has {cur.numel()} values; "
                                     f"the inspector shows at most {MAX_VALUES}")
        vals = cur - store.original(req.name, req.index) if req.delta else cur
        v2 = vals.reshape(-1, vals.shape[-1]) if vals.ndim >= 2 else vals.reshape(1, -1)
        return {"name": req.name, "index": req.index, "shape": list(cur.shape), "grid": list(v2.shape),
                "values": v2.flatten().tolist(), "modified": req.name in store._backup,
                "stats": {"mean": vals.mean().item(), "std": vals.std().item() if vals.numel() > 1 else 0.0,
                          "min": vals.min().item(), "max": vals.max().item(),
                          "n_nonzero": int((vals != 0).sum())}}

    # ---- weights: change -------------------------------------------------------------------
    def _stage_and_maybe_apply(stage, apply: bool, label: str, note: str | None):
        def do():
            if apply and store.pending:
                raise HTTPException(409, f"{len(store.pending)} edit(s) are already staged; "
                                         "apply or discard them first")
            e, preview = stage()
            if not apply:
                return {"staged": True, "applied": False, "preview": preview}
            try:
                rep = store.apply()[0]
            except RuntimeError:
                store.pending.remove(e)
                raise
            return {"staged": True, "applied": True, "preview": preview, "n": rep["n"],
                    "n_changed": rep["n_changed"], "max_abs_change": rep["max_abs_change"],
                    "stored_first_values": rep["stored_preview"].tolist()}
        action = ("" if apply else "staged ") + label
        return _locked(action, do, {"type": "api", "text": action, "note": note, "weights_changed": apply})

    @app.post("/api/edit")
    def api_edit(req: EditReq):
        """Set / add / scale a region with explicit numbers."""
        vals = req.values if not isinstance(req.values, list) else ",".join(map(str, req.values))
        def stage():
            e = store.write(req.name, req.index, vals, req.mode)
            return e, f"{req.mode} {e.n} value(s)"
        return _stage_and_maybe_apply(stage, req.apply, f"{req.mode} {req.name}[{req.index}]", req.note)

    @app.post("/api/transform")
    def api_transform(req: TransformReq):
        """Set a region to an expression of x (its current value) and other regions W(...)."""
        return _stage_and_maybe_apply(lambda: store.transform(req.name, req.index, req.expression),
                                      req.apply, f"{req.name}[{req.index}] = {req.expression}", req.note)

    @app.post("/api/apply")
    def api_apply():
        def do():
            if not store.pending:
                raise ValueError("nothing is staged")
            return [{k: v for k, v in r.items() if k != "stored_preview"} for r in store.apply()]
        return _locked("apply staged edits", do, {"type": "api", "text": "apply staged edits", "weights_changed": True})

    @app.post("/api/discard")
    def api_discard():
        def do():
            return {"discarded": store.discard()}
        return _locked("discard staged edits", do)

    @app.post("/api/reset")
    def api_reset():
        def do():
            return {"restored": store.reset()}
        return _locked("reset weights", do, {"type": "api", "text": "reset weights", "weights_restored": True})

    # ---- probes: not part of the conversation ------------------------------------------------
    @app.post("/api/generate")
    def api_generate(req: GenerateReq):
        """One-off generation with the current weights."""
        return _locked(f"generate ({req.max_new_tokens} max tokens)", lambda: session.generate_once(
            req.prompt, req.max_new_tokens, req.chat, req.think, req.system, req.temperature), quiet=True)

    @app.post("/api/next_token")
    def api_next_token(req: NextTokenReq):
        """Next-token probabilities after a prompt, with the current weights."""
        return _locked("next_token", lambda: session.next_token(req.prompt, req.top_k, req.chat, req.think,
                                                                 req.targets), quiet=True)

    # ---- the browser chat --------------------------------------------------------------------
    @app.get("/api/context")
    def context():
        """The full prompt the model will read on its next reply (if you sent nothing new)."""
        text = session.render()
        return {"text": text, "tokens": len(tok(text).input_ids), "messages": len(session.messages)}

    @app.post("/api/think")
    def set_think(req: OnReq):
        """Turn Qwen's thinking mode on/off. Takes effect at the model's next generation."""
        session.think = req.on
        dbg.line(f"thinking mode {'on' if req.on else 'off'}"
                 + (" (from the next generation; the current one is unchanged)" if busy.locked() else ""))
        broadcast({"type": "think", "on": req.on})
        return {"ok": True, "think": req.on}

    @app.post("/api/clear")
    def clear():
        def do():
            session.messages.clear()
            history.clear()
            broadcast({"type": "cleared"})
            history.clear()
            return {"cleared": True}
        return _locked("new conversation", do, quiet=True)

    return app


def main():
    import argparse
    import os

    os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")
    import uvicorn

    from . import load

    ap = argparse.ArgumentParser(description="Serve a model whose weights can be edited live over HTTP.")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--think", action="store_true", help="start with thinking mode on")
    ap.add_argument("--system", help="system prompt for the browser chat (default: none)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--quiet", action="store_true", help="no debug trace in the terminal")
    ap.add_argument("--max-new-tokens", type=int, default=32768,
                    help="cap per reply, thinking included (default 32768; use Stop/Esc for runaway replies)")
    args = ap.parse_args()
    model, tok = load(args.model)
    torch.cuda.synchronize()
    print(f"serving {args.model} on http://{args.host}:{args.port}", flush=True)
    uvicorn.run(create_app(model, tok, args.model, args.think, debug=not args.quiet,
                           max_new_tokens=args.max_new_tokens, system=args.system),
                host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
