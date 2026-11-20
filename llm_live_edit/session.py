"""A chat with the live model, plus one-off probes (generate, next-token).

There is no KV cache across turns: every turn re-reads the whole conversation with the
current weights, so after an edit the edited model is the one reading the history and
writing the next reply. The model is not told about edits.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import torch
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers.generation.streamers import BaseStreamer

from .weights import WeightStore, fmt


class ChatSession:
    def __init__(self, model, tok, think: bool = False, max_new_tokens: int = 32768,
                 system: str | None = None, log_dir: str | Path = "runs", on_event=None):
        self.model, self.tok = model, tok
        self.think = think                  # Qwen thinking mode; can change any time
        self._gen_think = think             # the mode the current/last reply was generated in
        self.max_new_tokens = max_new_tokens
        self.system = system or None        # optional system prompt (none by default)
        self.store = WeightStore(model)
        self.messages: list[dict] = []
        self.emit = on_event or (lambda ev: None)  # receives dicts: token, assistant, gen_start, ...
        self.stop_event = threading.Event()        # set it to stop generation
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.log_path = Path(log_dir) / time.strftime("%Y%m%d-%H%M%S.jsonl")

    def log(self, **rec):
        with open(self.log_path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    def state(self) -> dict:
        st = self.store
        return {"pending": [{"name": e.name, "index": e.index, "mode": e.mode, "n": e.n,
                             "values": (f"= {e.expr}" if e.expr else
                                        fmt(e.values.flatten()[:8]) + (" ..." if e.values.numel() > 8 else ""))}
                            for e in st.pending],
                "applied": list(st.applied),
                "drift": st.drift()}

    # ---- prompts -----------------------------------------------------------------
    PLACEHOLDER = "<your next message goes here>"

    def render(self, generation_prompt: bool = True) -> str:
        """The exact text the model reads (chat template applied, special tokens included).
        Qwen's template refuses a conversation with no user message, so one is added as a
        visible placeholder when needed (only for display; generation always has a real one)."""
        msgs = ([{"role": "system", "content": self.system}] if self.system else []) + self.messages
        if not any(m["role"] == "user" for m in msgs):
            msgs.append({"role": "user", "content": self.PLACEHOLDER})
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=generation_prompt,
                                            enable_thinking=self.think)

    def _chat_prompt(self, prompt: str, chat: bool, think: bool, system: str | None) -> str:
        if not chat:
            return prompt
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=think)

    # ---- one-off probes (not part of the conversation) -------------------------------
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

    # ---- the conversation ------------------------------------------------------------
    @torch.no_grad()
    def generate(self) -> str:
        self._gen_think = self.think  # the switch may flip mid-reply; parse it the way it was generated
        prompt = self.render()
        ids = self.tok(prompt, return_tensors="pt").to(self.model.device)
        n_prompt = ids.input_ids.shape[1]
        self.emit({"type": "gen_start", "prompt_tokens": n_prompt, "think": self._gen_think})
        streamer = _TokenStreamer(self.tok, lambda text: self.emit({"type": "token", "text": text}))
        t0 = time.time()
        out = self.model.generate(**ids, max_new_tokens=self.max_new_tokens, do_sample=False, streamer=streamer,
                                  stopping_criteria=StoppingCriteriaList([_StopFlag(self.stop_event)]))
        n_new = out.shape[1] - n_prompt
        t1 = time.time()
        ttft = (streamer.t_first or t1) - t0  # re-reading the conversation (prefill) + first token
        decode = t1 - t0 - ttft
        self.emit({"type": "gen_end", "prompt_tokens": n_prompt, "new_tokens": n_new, "seconds": t1 - t0,
                   "prefill_seconds": ttft, "tok_per_s": (n_new - 1) / decode if decode > 0 else 0.0})
        return self.tok.decode(out[0, n_prompt:], skip_special_tokens=True)

    def reply(self) -> None:
        """Generate one assistant reply and add it to the conversation."""
        if self.stop_event.is_set():  # stop pressed before generation started
            self.emit({"type": "notice", "text": "stopped by user"})
            return
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
        visible = after.strip()
        content = visible + ("\n\n[interrupted by the user]" if interrupted else "")
        # reasoning_content must be a string: otherwise Qwen's template re-splits content at "</think>"
        # on every later turn and drops what came before. "" = earlier reasoning is not kept in history.
        self.messages.append({"role": "assistant", "content": content, "reasoning_content": ""})
        self.log(role="assistant", raw=text, interrupted=interrupted)
        self.emit({"type": "assistant", "content": visible, "reasoning": reasoning, "interrupted": interrupted})

    def chat(self, user_text: str) -> None:
        self.stop_event.clear()
        self.messages.append({"role": "user", "content": user_text})
        self.emit({"type": "user", "text": user_text})
        self.log(role="user", content=user_text)
        self.reply()


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
        if not final and text.endswith("�"):  # incomplete multi-byte character; wait for the rest
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
