"""Read, stage, apply and reset edits to a live model's weights (in memory).

Workflow:
    store = WeightStore(model)
    store.read("model.layers.3.self_attn.q_proj.weight", "0, 0:100")
    store.write("model.layers.3.self_attn.q_proj.weight", "0, 0:4", "0, 0, 0.1, 0.1")
    store.apply()     # staged edits go into the live tensors; next forward pass uses them
    store.reset()     # every touched tensor restored bit-exactly from its backup

Indices use numpy-style basic indexing written as a string ("3", "0:100",
"5, 0:100", ":, 7"). Values are written into bf16 tensors, so they get rounded;
`apply()` reports what was actually stored.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch

from .expr import ExprError, evaluate, summarize

MAX_READ = 512  # largest number of values one WeightStore.read() may return (the HTTP API allows more)


def parse_index(spec: str, ndim: int) -> tuple:
    """'5, 0:100' -> (5, slice(0, 100)). Only ints and slices; no eval."""
    spec = spec.strip().strip("[]()").strip()
    if spec in ("", ":"):
        return (slice(None),)
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) > ndim:
        raise ValueError(f"index '{spec}' has {len(parts)} dims but tensor has {ndim}")
    out = []
    for p in parts:
        if ":" in p:
            bits = p.split(":")
            if len(bits) > 3:
                raise ValueError(f"bad slice '{p}'")
            nums = [int(b) if b.strip() else None for b in bits]
            out.append(slice(*nums))
        else:
            out.append(int(p))
    return tuple(out)


def parse_values(values) -> torch.Tensor:
    """'0, 0, 0.1' or [0, 0, 0.1] or 0.5 -> float32 1-D tensor."""
    if isinstance(values, (int, float)):
        return torch.tensor([float(values)])
    if isinstance(values, str):
        toks = [t for t in re.split(r"[,\s\[\]]+", values) if t]
        return torch.tensor([float(t) for t in toks])
    return torch.tensor([float(v) for v in values])


def fmt(vals: torch.Tensor) -> str:
    return ", ".join(f"{v:.4g}" for v in vals.flatten().tolist())


CHUNK_ELEMENTS = 1 << 24  # ~16M values (64 MB as float32) per chunk when applying edits


def _row_chunks(t: torch.Tensor):
    """Slices along dim 0 covering t in pieces of at most ~CHUNK_ELEMENTS values."""
    if t.ndim == 0:
        yield Ellipsis
        return
    per_row = max(1, t[0].numel()) if t.shape[0] else 1
    step = max(1, CHUNK_ELEMENTS // per_row)
    for i in range(0, t.shape[0], step):
        yield slice(i, i + step)


def _first_values(t: torch.Tensor, k: int) -> torch.Tensor:
    """The first k values of t in row-major order, without copying all of t."""
    if t.ndim <= 1:
        return t.reshape(-1)[:k]
    out, n = [], 0
    for j in range(t.shape[0]):
        row = _first_values(t[j], k - n)
        out.append(row)
        n += row.numel()
        if n >= k:
            break
    return torch.cat(out)[:k]


def release_memory() -> None:
    """Give PyTorch's cached GPU memory back to the system. On GB10 GPU memory *is* system RAM,
    so memory PyTorch keeps cached after a big edit is RAM nothing else can use."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@dataclass
class Edit:
    name: str
    index: str
    values: torch.Tensor  # float32: shaped like the target region, or a single value to broadcast
    mode: str = "set"  # set | add | scale | expr
    n: int = 0  # number of elements in the target region
    expr: str | None = None  # mode "expr": evaluated at apply time with x = the current region


@dataclass
class WeightStore:
    model: torch.nn.Module
    pending: list[Edit] = field(default_factory=list)
    applied: list[dict] = field(default_factory=list)
    # Original copies of modified tensors, kept on the GPU. On GB10 CPU and GPU share one physical
    # memory, so a CPU copy saves nothing and copying 1 GB to the CPU side takes ~13 s.
    _backup: dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self):
        self.params = dict(self.model.named_parameters())

    # ---- explore -------------------------------------------------------------
    def list(self, pattern: str = "") -> list[tuple[str, tuple]]:
        rx = re.compile(pattern) if pattern else None
        return [(n, tuple(p.shape)) for n, p in self.params.items() if rx is None or rx.search(n)]

    def _get(self, name: str) -> torch.Tensor:
        if name not in self.params:
            raise KeyError(f"no parameter named '{name}' (use list_weights to see names)")
        return self.params[name]

    def read(self, name: str, index: str = ":") -> dict:
        p = self._get(name)
        region = p.data[parse_index(index, p.ndim)]
        if region.numel() > MAX_READ:
            raise ValueError(
                f"region {tuple(region.shape)} has {region.numel()} values; max per read is {MAX_READ}. "
                "Use a narrower index, or stats() for summaries."
            )
        return {"name": name, "index": index, "shape": tuple(region.shape),
                "values": region.float().cpu()}

    def original(self, name: str, index: str = ":") -> torch.Tensor:
        """The region as it was before any applied edit (float32, CPU)."""
        p = self._get(name)
        src = self._backup.get(name, p.data)
        return src[parse_index(index, p.ndim)].float().cpu()

    def stats(self, name: str, index: str = ":") -> dict:
        p = self._get(name)
        r = p.data[parse_index(index, p.ndim)].float()
        return {"name": name, "index": index, "shape": tuple(r.shape),
                "mean": r.mean().item(), "std": r.std().item() if r.numel() > 1 else 0.0,
                "min": r.min().item(), "max": r.max().item(), "abs_mean": r.abs().mean().item()}

    # ---- modify --------------------------------------------------------------
    def write(self, name: str, index: str, values, mode: str = "set") -> Edit:
        """Stage an edit. Nothing changes in the model until apply()."""
        if mode not in ("set", "add", "scale"):
            raise ValueError("mode must be set, add or scale")
        p = self._get(name)
        target_shape = p.data[parse_index(index, p.ndim)].shape
        n = target_shape.numel()
        if n == 0:
            raise ValueError(f"index '{index}' selects no elements of {name} {tuple(p.shape)}; "
                             "use ':' for a whole tensor, e.g. ':' or ':, :'")
        v = parse_values(values)
        if v.numel() == 1:
            v = v.reshape(())  # broadcast at apply time; don't materialize n copies
        elif v.numel() == n:
            v = v.reshape(target_shape)
        else:
            raise ValueError(f"region {tuple(target_shape)} needs {n} values (or 1 to broadcast); got {v.numel()}")
        if not torch.isfinite(v).all():
            raise ValueError("values must be finite")
        e = Edit(name, index, v, mode, n)
        self.pending.append(e)
        return e

    # ---- matrix math -------------------------------------------------------------
    @torch.no_grad()
    def compute(self, expression: str):
        """Evaluate an expression read-only (see llm_live_edit.expr). Returns a tensor or a number."""
        return evaluate(expression, self)

    @torch.no_grad()
    def _eval_for_region(self, expression: str, old: torch.Tensor) -> torch.Tensor:
        r = evaluate(expression, self, x=old)
        if not isinstance(r, torch.Tensor):
            r = torch.tensor(float(r), device=old.device)
        if r.dtype == torch.bool:
            raise ExprError("the expression gives a boolean mask; wrap it, e.g. where(x > 0, x, 0)")
        try:
            ok = torch.broadcast_shapes(r.shape, old.shape) == old.shape
        except RuntimeError:
            ok = False
        if not ok:
            raise ExprError(f"result shape {list(r.shape)} does not fit the target region {list(old.shape)}")
        r = r.float().expand_as(old)
        if not torch.isfinite(r).all():
            raise ExprError("result contains inf/nan")
        return r

    @torch.no_grad()
    def transform(self, name: str, index: str, expression: str) -> tuple[Edit, str]:
        """Stage an edit computed from an expression, where x is the region's current value.
        It is checked (and previewed) now, but evaluated again when applied, in order, so it
        sees the effect of edits applied before it in the same batch. Returns the staged edit
        and a text summary (shape, statistics, first values) of the new values."""
        p = self._get(name)
        old = p.data[parse_index(index, p.ndim)].float()
        if old.numel() == 0:
            raise ValueError(f"index '{index}' selects no elements of {name} {tuple(p.shape)}")
        preview = summarize(self._eval_for_region(expression, old), max_values=16)
        e = Edit(name, index, torch.tensor(0.0), "expr", old.numel(), expression)
        self.pending.append(e)
        del old
        release_memory()
        return e, preview

    @torch.no_grad()
    def apply(self) -> list[dict]:
        """Commit all staged edits into the live tensors, in order. All-or-nothing:
        if any edit fails, the edits already done in this call are undone, nothing
        is applied, and the staged edits stay pending (fix with discard())."""
        reports, undo, new_backups = [], [], []
        try:
            for i, e in enumerate(self.pending):
                try:
                    p = self.params[e.name]
                    idx = parse_index(e.index, p.ndim)
                    if e.name not in self._backup:
                        self._backup[e.name] = p.data.detach().clone()
                        new_backups.append(e.name)
                    region = p.data[idx]          # a view: writing to it writes the parameter
                    old_raw = region.clone()      # same dtype as the weights; used to undo
                    undo.append((p, idx, old_raw))
                    # An expression may need the whole region at once (norm(x), low_rank(x), ...),
                    # so it is evaluated in full; everything else is done in row chunks below.
                    new_full = self._eval_for_region(e.expr, old_raw.float()) if e.mode == "expr" else None
                    v = None if new_full is not None else e.values.to(p.device)
                    n_changed, max_change = 0, 0.0
                    for sl in _row_chunks(region):
                        o = old_raw[sl].float()
                        if new_full is not None:
                            nw = new_full[sl]
                        else:
                            vc = v if v.ndim == 0 else v[sl]
                            nw = {"set": vc.expand_as(o), "add": o + vc, "scale": o * vc}[e.mode]
                        if not torch.isfinite(nw).all():
                            raise ValueError("result would contain inf/nan")
                        region[sl] = nw.to(p.dtype)
                        d = (region[sl].float() - o).abs()
                        n_changed += int((d != 0).sum())
                        max_change = max(max_change, d.max().item())
                        del o, nw, d
                    del new_full
                    reports.append({"name": e.name, "index": e.index, "mode": e.mode, "expr": e.expr,
                                    "n": region.numel(), "n_changed": n_changed, "max_abs_change": max_change,
                                    "stored_preview": _first_values(region, 16).float().cpu()})
                except Exception as ex:
                    raise RuntimeError(f"edit #{i + 1} ({e.mode} {e.name}[{e.index}]) failed: {ex}") from ex
        except Exception:
            for p, idx, old_raw in reversed(undo):
                p.data[idx] = old_raw
            for name in new_backups:
                del self._backup[name]
            raise
        finally:
            del undo
            release_memory()
        self.applied.extend({k: v for k, v in r.items() if k != "stored_preview"} for r in reports)
        self.pending.clear()
        return reports

    def discard(self) -> int:
        n = len(self.pending)
        self.pending.clear()
        return n

    @torch.no_grad()
    def reset(self) -> int:
        """Restore every modified tensor to its original values (bit-exact) and drop staged edits."""
        n = len(self._backup)
        for name, orig in self._backup.items():
            self.params[name].data.copy_(orig)
        self._backup.clear()
        self.applied.clear()
        self.pending.clear()
        return n

    @torch.no_grad()
    def drift(self) -> dict[str, float]:
        """L2 norm of (current - original) for each modified tensor.
        Computed in chunks to bound temporary memory."""
        out = {}
        for n, orig in self._backup.items():
            cur = self.params[n].data
            c2, o2 = cur.reshape(cur.shape[0], -1), orig.reshape(orig.shape[0], -1)
            step = max(1, (64 << 20) // max(1, c2.shape[1]))  # ~64M values per chunk
            sq = 0.0
            for i in range(0, c2.shape[0], step):
                d = c2[i:i + step].float() - o2[i:i + step].float()
                sq += d.pow(2).sum().item()
            out[n] = sq ** 0.5
        return out


# ---- human-readable layout notes for Qwen3.5 ------------------------------------
def describe(name: str, shape: tuple, cfg) -> str:
    """Explain what rows/columns of a Qwen3.5 parameter mean, with per-head ranges."""
    m = re.match(r"model\.layers\.(\d+)\.(.+)\.weight$", name)
    lines = [f"{name}: shape {shape}, stored as bfloat16. Linear weights are [out_features, in_features]: "
             "row i produces output feature i, column j reads input feature j."]
    tied = getattr(cfg, "tie_word_embeddings", False)
    if name == "model.embed_tokens.weight":
        lines.append("Token embeddings: row t is the input vector for token id t."
                     + (" Tied with the output head, so editing a row also changes how likely token t "
                        "is to be produced." if tied else " Not tied: the output side is lm_head.weight."))
    if name == "lm_head.weight":
        lines.append("Output head: the logit for token t is row t dotted with the final hidden state, "
                     "so scaling row t changes how likely token t is to be produced.")
    if name == "model.norm.weight":
        lines.append("Final RMSNorm scale applied to the residual stream before the output head "
                     "(effective scale is 1 + weight).")
    if not m:
        return "\n".join(lines)
    layer, sub = int(m.group(1)), m.group(2)
    kind = cfg.layer_types[layer]
    lines.append(f"Layer {layer} is a {kind} layer.")
    hd = cfg.head_dim
    if sub == "self_attn.q_proj":
        lines.append(f"{cfg.num_attention_heads} query heads, each owning {2*hd} rows: for head h, rows "
                     f"[h*{2*hd}, h*{2*hd}+{hd}) produce the query and rows [h*{2*hd}+{hd}, (h+1)*{2*hd}) "
                     "produce a sigmoid output gate. E.g. head 0 query = rows 0:256, gate = rows 256:512.")
    elif sub in ("self_attn.k_proj", "self_attn.v_proj"):
        lines.append(f"{cfg.num_key_value_heads} KV heads (each shared by "
                     f"{cfg.num_attention_heads // cfg.num_key_value_heads} query heads), head h = rows "
                     f"[h*{hd}, (h+1)*{hd}).")
    elif sub == "self_attn.o_proj":
        lines.append(f"Maps concatenated head outputs back to the residual stream: head h reads columns "
                     f"[h*{hd}, (h+1)*{hd}).")
    elif sub == "linear_attn.in_proj_qkv":
        kd, vd = cfg.linear_key_head_dim, cfg.linear_value_head_dim
        nh, nv = cfg.linear_num_key_heads, cfg.linear_num_value_heads
        lines.append(f"Gated DeltaNet projection, rows split into query [0, {kd*nh}), key [{kd*nh}, {2*kd*nh}), "
                     f"value [{2*kd*nh}, {2*kd*nh + vd*nv}). {nh} query/key heads: head h's query = rows "
                     f"[h*{kd}, (h+1)*{kd}), key = rows [{kd*nh}+h*{kd}, ...). {nv} value heads: value head h "
                     f"= rows [{2*kd*nh}+h*{vd}, ...).")
    elif sub == "linear_attn.in_proj_z":
        lines.append(f"Output gate for the {cfg.linear_num_value_heads} linear-attention heads, head h = rows "
                     f"[h*{cfg.linear_value_head_dim}, (h+1)*{cfg.linear_value_head_dim}).")
    elif sub == "linear_attn.out_proj":
        lines.append(f"Maps linear-attention head outputs to the residual stream: head h reads columns "
                     f"[h*{cfg.linear_value_head_dim}, (h+1)*{cfg.linear_value_head_dim}).")
    elif sub in ("mlp.gate_proj", "mlp.up_proj"):
        lines.append(f"MLP input side: row i belongs to hidden neuron i (of {cfg.intermediate_size}). "
                     "neuron = silu(gate_row . x) * (up_row . x).")
    elif sub == "mlp.down_proj":
        lines.append("MLP output side: column i is what hidden neuron i writes into the residual stream.")
    elif sub.endswith("layernorm"):
        lines.append("RMSNorm scale per residual dimension (Qwen3.5 uses 1 + weight, so 0 means unchanged).")
    return "\n".join(lines)
