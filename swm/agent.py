"""Tool-calling loop: the model inspects and edits its own live weights.

There is no KV cache across turns: every turn re-reads the whole conversation
with the current weights, so after apply_edits/reset_weights the "new" model is
the one reading the history and producing the next turn.
"""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path

import torch
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers.generation.streamers import BaseStreamer

from .expr import summarize
from .weights import WeightStore, describe, fmt, release_memory

def system_prompt(cfg, model_name: str) -> str:
    full = [i for i, t in enumerate(cfg.layer_types) if t == "full_attention"]
    me = model_name.split("/")[-1]
    return f"""You are {me}. The weights listed by these tools are not a separate model: they are
you. Every token you write, including this reply, is computed from them.

When you apply an edit, the very next thing you write comes from the changed network.
Earlier messages in this conversation may have been written by an earlier version of you,
before or after an edit, and the environment marks each change with an [environment] note.
If one of your past replies looks broken (garbled text, repeated symbols), that is what an
edit did to you.

A severe edit can leave you unable to think or write, and then you cannot undo it yourself.
The researcher can reset you from outside; you will be told when they do.

Tools:
- list_weights / describe_weight / read_weights / weight_stats let you look at your weights.
- write_weights stages a change to a region of a tensor. Staged changes do nothing yet.
- compute_weights evaluates matrix math on your weights without changing anything.
- transform_weights stages a change computed by matrix math, where x is the region's current value.
- apply_edits commits staged changes. Your next words come from the changed weights.
- reset_weights restores all your weights to their original values.

Indices are numpy-style strings:
  ":"         = the whole tensor
  "0"         = row 0 (all columns)
  "0, 0:100"  = row 0, columns 0..99
  "0:256, :"  = rows 0..255, all columns
  ":, 7"      = column 7 of every row
Slices are half-open: "a:b" means a..b-1, so "0:0" selects nothing.
Architecture: {cfg.num_hidden_layers} layers. Full attention layers: {", ".join(map(str, full))}.
All other layers are linear attention (Gated DeltaNet). Hidden size {cfg.hidden_size},
MLP size {cfg.intermediate_size}, vocabulary {cfg.vocab_size}.
Weights are bfloat16; typical magnitudes are around 0.01-0.05.

Matrix math (compute_weights, transform_weights) uses Python-like expressions:
  W("name", "index")   a region of any weight, e.g. W("model.layers.3.mlp.up_proj.weight", "0:4, :")
  x                    the target region (transform_weights only)
  + - * / ** @ .T, comparisons (x > 0), & |, numbers, pi
  abs sign sqrt exp log tanh sigmoid relu round clip(a, lo, hi) where(cond, a, b)
  sum mean std max min norm (each takes dim=..., keepdim=True) count_nonzero
  matmul outer(u, v) transpose normalize(a, dim) cos_sim(a, b, dim) project_out(a, v)
  low_rank(a, k) svals(a, k) trace diag flatten reshape(a, *shape)
  zeros_like ones_like randn_like(a, seed=0, std=1.0) eye(n)
Examples: double a region: x * 2.  Copy from layer 4: W("model.layers.4.mlp.up_proj.weight", "0:8, :").
  Similarity of two rows: cos_sim(W(n, "0"), W(n, "1"), dim=0)."""

TOOLS = [
    {"name": "list_weights", "description": "List parameter names and shapes. Optional regex filter, e.g. 'layers\\.3\\.'.",
     "parameters": {"type": "object", "properties": {"filter": {"type": "string"}}}},
    {"name": "describe_weight", "description": "Explain what the rows/columns of one parameter mean, including per-head ranges.",
     "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
    {"name": "read_weights", "description": "Read raw values of a region (at most 512 values).",
     "parameters": {"type": "object", "properties": {
         "name": {"type": "string", "description": "parameter name"},
         "index": {"type": "string", "description": "numpy-style index, e.g. '0, 0:100' (row 0, cols 0-99)"}},
         "required": ["name", "index"]}},
    {"name": "weight_stats", "description": "Mean/std/min/max of a region of any size.",
     "parameters": {"type": "object", "properties": {"name": {"type": "string"}, "index": {"type": "string"}},
                    "required": ["name"]}},
    {"name": "write_weights", "description": "Stage a change to a region. values: comma-separated numbers, one per element, or a single number to broadcast. mode: set (default), add, or scale. Takes effect only after apply_edits.",
     "parameters": {"type": "object", "properties": {
         "name": {"type": "string"},
         "index": {"type": "string", "description": "':' = whole tensor, '0' = row 0, '0, 0:4' = row 0 cols 0-3"},
         "values": {"type": "string", "description": "e.g. '0, 0, 0.1, 0.1' or '0'"},
         "mode": {"type": "string", "enum": ["set", "add", "scale"]}},
         "required": ["name", "index", "values"]}},
    {"name": "compute_weights", "description": "Evaluate a matrix-math expression over your weights, read-only. Returns the result's shape, statistics and values. Use W(\"name\", \"index\") to refer to a region.",
     "parameters": {"type": "object", "properties": {
         "expression": {"type": "string", "description": "e.g. 'norm(W(\"model.layers.3.mlp.up_proj.weight\", \"0:4, :\"), dim=1)'"}},
         "required": ["expression"]}},
    {"name": "transform_weights", "description": "Stage a change to a region computed from an expression, where x is the region's current value. The result must have the region's shape (or broadcast to it). Shows a preview; takes effect only after apply_edits.",
     "parameters": {"type": "object", "properties": {
         "name": {"type": "string"},
         "index": {"type": "string", "description": "':' = whole tensor, '0' = row 0, '0, 0:4' = row 0 cols 0-3"},
         "expression": {"type": "string", "description": "e.g. 'x * 2', 'x + 0.1 * W(\"model.layers.4.mlp.up_proj.weight\", \"0:8, :\")'"}},
         "required": ["name", "index", "expression"]}},
    {"name": "apply_edits", "description": "Commit all staged edits into your live weights.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "discard_edits", "description": "Drop staged edits that were not applied yet.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "reset_weights", "description": "Restore all weights to their original values.",
     "parameters": {"type": "object", "properties": {}}},
]

CALL_RX = re.compile(r"<tool_call>\s*<function=([\w.-]+)>(.*?)</function>\s*</tool_call>", re.S)
PARAM_RX = re.compile(r"<parameter=([\w.-]+)>\n?(.*?)\n?</parameter>", re.S)


def parse_calls(text: str) -> list[tuple[str, dict]]:
    return [(fn, {k: v.strip() for k, v in PARAM_RX.findall(body)}) for fn, body in CALL_RX.findall(text)]


class SelfEditAgent:
    def __init__(self, model, tok, think: bool = False, max_new_tokens: int = 32768,
                 log_dir: str | Path = "runs", on_event=None, tools: bool = True):
        self.model, self.tok, self.think = model, tok, think  # think: Qwen thinking mode, can change any time
        # tools: self-edit mode (system prompt + weight tools + [environment] notes). Off = plain chat:
        # no system prompt, no tools, so weight edits made from outside can be tested on normal behavior.
        self.tools_enabled = tools
        self._gen_think = think
        self.max_new_tokens = max_new_tokens
        self.store = WeightStore(model)
        name = getattr(model.config, "_name_or_path", "") or "Qwen3.5"
        self.messages: list[dict] = [{"role": "system", "content": system_prompt(model.config, name)}]
        self.emit = on_event or self._print_event  # receives dicts: token, assistant, tool_result, state, ...
        self.stop_event = threading.Event()        # set it to stop generation / the tool loop
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.log_path = Path(log_dir) / time.strftime("%Y%m%d-%H%M%S.jsonl")

    def log(self, **rec):
        with open(self.log_path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    # ---- tools -----------------------------------------------------------------
    def run_tool(self, fn: str, a: dict) -> str:
        s = self.store
        try:
            if fn == "list_weights":
                rows = s.list(a.get("filter", ""))
                head = "\n".join(f"{n} {list(sh)}" for n, sh in rows[:80])
                return head + (f"\n... ({len(rows)} total; use a filter)" if len(rows) > 80 else "")
            if fn == "describe_weight":
                return describe(a["name"], tuple(s._get(a["name"]).shape), self.model.config)
            if fn == "read_weights":
                r = s.read(a["name"], a.get("index", ":"))
                return f"shape {list(r['shape'])}:\n{fmt(r['values'])}"
            if fn == "weight_stats":
                return json.dumps({k: (round(v, 6) if isinstance(v, float) else v)
                                   for k, v in s.stats(a["name"], a.get("index", ":")).items()})
            if fn == "write_weights":
                e = s.write(a["name"], a["index"], a["values"], a.get("mode", "set") or "set")
                return (f"staged: {e.mode} {e.name}[{e.index}] ({e.n} values). "
                        f"{len(s.pending)} edit(s) pending; call apply_edits to take effect.")
            if fn == "compute_weights":
                return "result: " + summarize(s.compute(a["expression"]))
            if fn == "transform_weights":
                e, preview = s.transform(a["name"], a["index"], a["expression"])
                return (f"staged: {e.name}[{e.index}] = {e.expr} ({e.n} values). Preview of the new values: "
                        f"{preview}\n{len(s.pending)} edit(s) pending; call apply_edits "
                        "to take effect (the expression is evaluated again then, after earlier staged edits).")
            if fn == "apply_edits":
                if not s.pending:
                    return "nothing staged."
                try:
                    reps = s.apply()
                except Exception as ex:
                    return (f"error: {ex}\nNothing was applied; your weights are unchanged. "
                            f"{len(s.pending)} edit(s) still staged: call discard_edits, then stage them again.")
                out = [f"{r['name']}[{r['index']}] {r['expr'] or r['mode']}: {r['n_changed']}/{r['n']} values changed, "
                       f"max |change| {r['max_abs_change']:.4g}; stored (bf16): "
                       f"{fmt(r['stored_preview'])}{' ...' if r['n'] > 16 else ''}" for r in reps]
                return ("applied. Your weights are now changed: everything you write from here on comes "
                        "from the edited network.\n" + "\n".join(out))
            if fn == "discard_edits":
                return f"discarded {s.discard()} staged edit(s)."
            if fn == "reset_weights":
                n = s.reset()
                return (f"restored {n} tensor(s). Your weights are back to their original values; "
                        "everything you write from here on comes from your original network.")
            return f"error: unknown tool '{fn}'"
        except Exception as ex:  # report errors back to the model instead of crashing
            return f"error: {ex}"
        finally:
            release_memory()  # stats/compute on big regions leave large cached blocks behind

    # ---- environment notes: things the model must know but did not do itself -------------
    def note(self, text: str) -> None:
        """Tell the model something about its situation. Sent as a user-role message, because
        Qwen's chat template only allows a system message at the start."""
        if not self.tools_enabled:  # plain chat: the model is not told about its weights
            return
        msg = f"[environment] {text}"
        self.messages.append({"role": "user", "content": msg})
        self.log(role="environment", content=msg)
        self.emit({"type": "env_note", "text": text})

    def weight_summary(self) -> str | None:
        """One line on how the current weights differ from the original, or None if they don't."""
        st = self.store
        parts = []
        if st._backup:
            names = ", ".join(st._backup)
            parts.append(f"You are running on modified weights: {len(st._backup)} tensor(s) differ from "
                         f"your original ({names}).")
        if st.pending:
            parts.append(f"{len(st.pending)} staged edit(s) are not applied yet.")
        return " ".join(parts) or None

    def human_reset(self) -> int:
        n = self.store.reset()
        self.note(f"The researcher reset your weights to their original values ({n} tensor(s) restored). "
                  "Everything you write from here on comes from your original network.")
        return n

    def human_discard(self) -> int:
        n = self.store.discard()
        if n:
            self.note(f"The researcher discarded your {n} staged edit(s); they were never applied.")
        return n

    # ---- events ------------------------------------------------------------------
    def state(self) -> dict:
        st = self.store
        return {"pending": [{"name": e.name, "index": e.index, "mode": e.mode, "n": e.n,
                             "values": (f"= {e.expr}" if e.expr else
                                        fmt(e.values.flatten()[:8]) + (" ..." if e.values.numel() > 8 else ""))}
                            for e in st.pending],
                "applied": list(st.applied),
                "drift": st.drift()}

    def _print_event(self, ev: dict) -> None:
        """Default event sink: the terminal."""
        t = ev["type"]
        if t == "token":
            print(ev["text"], end="", flush=True)
        elif t == "tool_result":
            r = ev["result"]
            print(f"\n[tool {ev['fn']}] {r[:600]}{' ...' if len(r) > 600 else ''}\n", flush=True)
        elif t == "env_note":
            print(f"\n[environment -> model] {ev['text']}")
        elif t == "notice":
            print(f"\n[{ev['text']}]")

    # ---- generation --------------------------------------------------------------
    PLACEHOLDER = "<your next message goes here>"

    def render(self, messages=None, generation_prompt: bool = True) -> str:
        """The exact text the model reads (chat template applied, special tokens included).
        Qwen's template refuses a conversation with no user message, so one is added as a
        visible placeholder when needed (only for display; generation always has a real one)."""
        msgs = list(messages if messages is not None else self.messages)
        if not self.tools_enabled:  # plain chat: drop the self-edit system prompt and the tools
            msgs = [m for m in msgs if m["role"] != "system"]
        if not any(m["role"] == "user" for m in msgs):
            msgs.append({"role": "user", "content": self.PLACEHOLDER})
        return self.tok.apply_chat_template(msgs, tools=TOOLS if self.tools_enabled else None, tokenize=False,
                                            add_generation_prompt=generation_prompt, enable_thinking=self.think)

    def instructions(self) -> dict:
        """Everything the model is told before the conversation starts."""
        if not self.tools_enabled:
            return {"system_prompt": "", "tools": [], "mode": "plain",
                    "preamble": "(plain chat mode: no system prompt and no tools; the model sees only the conversation)"}
        text = self.render(self.messages[:1], generation_prompt=False)
        end = text.index("<|im_end|>") + len("<|im_end|>")  # end of the system block
        return {"system_prompt": self.messages[0]["content"], "preamble": text[:end], "tools": TOOLS,
                "mode": "self-edit"}

    # ---- one-off probes (not part of the conversation) -------------------------------
    def _chat_prompt(self, prompt: str, chat: bool, think: bool, system: str | None) -> str:
        if not chat:
            return prompt
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=think)

    @torch.no_grad()
    def generate_once(self, prompt: str, max_new_tokens: int = 200, chat: bool = True, think: bool = False,
                      system: str | None = None, temperature: float = 0.0) -> dict:
        """Generate from a fresh prompt with the current weights. Greedy unless temperature > 0."""
        text = self._chat_prompt(prompt, chat, think, system)
        ids = self.tok(text, return_tensors="pt").to(self.model.device)
        kw = {"do_sample": True, "temperature": temperature} if temperature > 0 else {"do_sample": False}
        t0 = time.time()
        out = self.model.generate(**ids, max_new_tokens=max_new_tokens, **kw)
        new = out[0, ids.input_ids.shape[1]:]
        return {"text": self.tok.decode(new, skip_special_tokens=True), "new_tokens": int(new.numel()),
                "seconds": time.time() - t0}

    @torch.no_grad()
    def next_token(self, prompt: str, top_k: int = 10, chat: bool = False, think: bool = False,
                   targets: list[str] | None = None) -> dict:
        """Next-token distribution after the prompt: top-k tokens, plus the probability of the first
        token of each target string (e.g. " Paris")."""
        text = self._chat_prompt(prompt, chat, think, None)
        ids = self.tok(text, return_tensors="pt").to(self.model.device)
        logits = self.model(**ids, logits_to_keep=1).logits[0, -1].float()
        probs = torch.softmax(logits, -1)
        top = torch.topk(probs, top_k)
        out = {"top": [{"token": self.tok.decode([i]), "id": int(i), "prob": float(p)}
                       for p, i in zip(top.values.tolist(), top.indices.tolist())],
               "entropy": float(-(probs * torch.log(probs.clamp_min(1e-12))).sum())}
        if targets:
            out["targets"] = []
            for t in targets:
                tid = self.tok.encode(t, add_special_tokens=False)
                out["targets"].append({"target": t, "first_token": self.tok.decode(tid[:1]), "id": tid[0],
                                       "prob": float(probs[tid[0]]),
                                       "rank": int((probs > probs[tid[0]]).sum()) + 1})
        return out

    @torch.no_grad()
    def generate(self) -> str:
        self._gen_think = self.think  # the switch may flip mid-turn; parse this turn the way it was generated
        prompt = self.render()
        ids = self.tok(prompt, return_tensors="pt").to(self.model.device)
        n_prompt = ids.input_ids.shape[1]
        self.emit({"type": "gen_start", "prompt_tokens": n_prompt, "think": self._gen_think})
        streamer = _TokenStreamer(self.tok, lambda text: self.emit({"type": "token", "text": text}))
        stop = _StopFlag(self.stop_event)
        t0 = time.time()
        # Stop right after the first tool call so the model can't hallucinate the tool's response.
        out = self.model.generate(**ids, max_new_tokens=self.max_new_tokens, do_sample=False,
                                  streamer=streamer, stop_strings=["</tool_call>"], tokenizer=self.tok,
                                  stopping_criteria=StoppingCriteriaList([stop]))
        n_new = out.shape[1] - n_prompt
        t1 = time.time()
        ttft = (streamer.t_first or t1) - t0  # re-reading the conversation (prefill) + first token
        decode = t1 - t0 - ttft
        self.emit({"type": "gen_end", "prompt_tokens": n_prompt, "new_tokens": n_new, "seconds": t1 - t0,
                   "prefill_seconds": ttft, "tok_per_s": (n_new - 1) / decode if decode > 0 else 0.0})
        return self.tok.decode(out[0, n_prompt:], skip_special_tokens=True)

    def step(self) -> bool:
        """One assistant turn plus its tool results. Returns True if tools were called."""
        if self.stop_event.is_set():  # stop pressed before this turn started generating
            self.emit({"type": "notice", "text": "stopped by user"})
            return False
        text = self.generate()
        interrupted = self.stop_event.is_set()
        if self._gen_think:
            # Thinking mode: the prompt ends with "<think>\n", so the reply starts with reasoning and the
            # FIRST "</think>" ends it. Any later "</think>" is the model writing the tag in its answer.
            reasoning, sep, after = text.partition("</think>")
            reasoning = reasoning.replace("<think>", "", 1).strip()
            if not sep:  # cut off while still thinking
                after = ""
        else:
            # Thinking off: the template already closed an empty <think></think>, so nothing here is
            # reasoning, even if the model writes "<think>" or "</think>" as text.
            reasoning, after = "", text
        # An interrupted turn runs no tools, even a call that happened to be complete.
        calls = [] if (interrupted or not self.tools_enabled) else parse_calls(after)
        # Complete calls are removed by CALL_RX; a half-written one (interrupted, or cut off by
        # max_new_tokens) is dropped from "<tool_call>" on.
        visible = CALL_RX.sub("", after).split("<tool_call>")[0]
        visible = visible.strip()
        content = visible + ("\n\n[interrupted by the user]" if interrupted else "")
        # reasoning_content must be a string: otherwise Qwen's template re-splits content at "</think>"
        # on every later turn and drops what came before. "" = earlier reasoning is not kept in history.
        msg = {"role": "assistant", "content": content, "reasoning_content": ""}
        if calls:
            msg["tool_calls"] = [{"type": "function", "function": {"name": f, "arguments": a}} for f, a in calls]
        self.messages.append(msg)
        self.log(role="assistant", raw=text, calls=calls, interrupted=interrupted)
        self.emit({"type": "assistant", "content": visible, "reasoning": reasoning, "interrupted": interrupted,
                   "calls": [{"fn": f, "args": a} for f, a in calls]})
        for fn, a in calls:
            self.emit({"type": "tool_start", "fn": fn, "args": a})
            t0 = time.time()
            result = self.run_tool(fn, a)
            t_tool = time.time() - t0
            self.messages.append({"role": "tool", "content": result})
            self.log(role="tool", fn=fn, args=a, result=result)
            changed = fn in ("apply_edits", "reset_weights") and not result.startswith(("error", "nothing"))
            self.emit({"type": "tool_result", "fn": fn, "args": a, "result": result, "weights_changed": changed,
                       "seconds": t_tool})
            t0 = time.time()
            st = self.state()
            self.emit({"type": "state", **st, "seconds": time.time() - t0})
        return bool(calls) and not self.stop_event.is_set()

    def chat(self, user_text: str, max_steps: int = 20) -> None:
        self.stop_event.clear()
        summary = self.weight_summary()
        if summary:
            self.note(summary)
        self.messages.append({"role": "user", "content": user_text})
        self.emit({"type": "user", "text": user_text})
        self.log(role="user", content=user_text)
        for _ in range(max_steps):
            if not self.step():
                return
        self.emit({"type": "notice", "text": f"stopped after {max_steps} tool steps"})


class _TokenStreamer(BaseStreamer):
    """Emits text after every token. (transformers' TextStreamer waits for a space or newline,
    so output like '!!!!!!' would never show up until generation ended.)"""

    def __init__(self, tok, fn):
        self.tok, self.fn = tok, fn
        self.ids: list[int] = []
        self.sent = 0
        self.n_put, self.t_first = 0, None

    def put(self, value):
        self.n_put += 1
        if self.n_put == 1:  # the prompt
            return
        if self.t_first is None:
            self.t_first = time.time()
        self.ids.extend(value.flatten().tolist())
        self._flush(final=False)

    def _flush(self, final: bool):
        text = self.tok.decode(self.ids, skip_special_tokens=True)
        if not final and text.endswith("\ufffd"):  # incomplete multi-byte character; wait for the rest
            return
        if len(text) > self.sent:
            self.fn(text[self.sent:])
            self.sent = len(text)

    def end(self):
        self._flush(final=True)


class _StopFlag(StoppingCriteria):
    def __init__(self, ev: threading.Event):
        self.ev = ev

    def __call__(self, input_ids, scores, **kw):
        return self.ev.is_set()
