"""A small, safe math language over weight regions, for /api/compute and /api/transform.

Syntax is a subset of Python expressions, parsed with `ast` and checked node by node;
nothing is passed to eval(). Values are float32 tensors on the model's device.

    x                       the target region (only in a transform)
    W("name", "index")      a region of any parameter (index defaults to ":")
    2 * x, x + 0.1 * W(...), a @ b, a.T, -x, x ** 2, x / norm(x, dim=1, keepdim=True)
    x > 0, where(x > 0, x, 0), (x > 0) & (x < 1)
    functions: see FUNCS below
"""
from __future__ import annotations

import ast
import math

import torch

MAX_ELEMENTS = 1 << 31  # refuse intermediate results larger than this


def _t(v, dev):
    return v if isinstance(v, torch.Tensor) else torch.tensor(float(v), device=dev)


def _dim_kw(kw):
    return {k: (tuple(v) if isinstance(v, list) else v) for k, v in kw.items()}


def _reduce(fn):
    def f(a, dim=None, keepdim=False):
        a = _t(a, None)
        return fn(a) if dim is None else fn(a, dim=dim, keepdim=keepdim)
    return f


def _randn_like(a, seed=0, std=1.0):
    g = torch.Generator(device=a.device).manual_seed(int(seed))
    return torch.randn(a.shape, generator=g, device=a.device, dtype=a.dtype) * std


def _normalize(a, dim=-1):
    return a / a.norm(dim=dim, keepdim=True).clamp_min(1e-12)


def _cos_sim(a, b, dim=-1):
    return torch.nn.functional.cosine_similarity(a, b, dim=dim)


def _project_out(a, v):
    """Remove the component of a's rows (last dim) along direction v."""
    v = v.flatten()
    v = v / v.norm().clamp_min(1e-12)
    return a - (a @ v).unsqueeze(-1) * v


def _low_rank(a, k):
    """Best rank-k approximation of a 2-D tensor (truncated SVD)."""
    u, s, vh = torch.linalg.svd(a, full_matrices=False)
    k = int(k)
    return (u[:, :k] * s[:k]) @ vh[:k]


def _svals(a, k=None):
    s = torch.linalg.svdvals(a)
    return s if k is None else s[: int(k)]


FUNCS = {
    # elementwise
    "abs": torch.abs, "sign": torch.sign, "sqrt": torch.sqrt, "exp": torch.exp, "log": torch.log,
    "tanh": torch.tanh, "sigmoid": torch.sigmoid, "relu": torch.relu,
    "clip": lambda a, lo=None, hi=None: torch.clamp(a, lo, hi),
    "where": lambda c, a, b: torch.where(c, _t(a, c.device), _t(b, c.device)),
    "round": torch.round,
    # reductions (dim=None reduces everything)
    "sum": _reduce(torch.sum), "mean": _reduce(torch.mean), "std": _reduce(torch.std),
    "max": _reduce(torch.amax), "min": _reduce(torch.amin),
    "norm": lambda a, dim=None, keepdim=False: torch.linalg.vector_norm(a, dim=dim, keepdim=keepdim),
    "count_nonzero": lambda a: torch.count_nonzero(a).float(),
    # linear algebra
    "matmul": torch.matmul, "outer": lambda u, v: torch.outer(u.flatten(), v.flatten()),
    "transpose": lambda a: a.T, "normalize": _normalize, "cos_sim": _cos_sim,
    "project_out": _project_out, "low_rank": _low_rank, "svals": _svals,
    "trace": torch.trace, "diag": torch.diag,
    # shape
    "flatten": torch.flatten, "reshape": lambda a, *shape: a.reshape(*shape),
    # constructors
    "zeros_like": torch.zeros_like, "ones_like": torch.ones_like, "randn_like": _randn_like,
    "eye": None,  # filled per call (needs device)
}
CONSTS = {"pi": math.pi, "e": math.e}
_ELEMENTWISE = {"abs", "sign", "sqrt", "exp", "log", "tanh", "sigmoid", "relu", "round"}

_BINOPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
           ast.Div: lambda a, b: a / b, ast.Pow: lambda a, b: a ** b, ast.MatMult: lambda a, b: a @ b,
           ast.Mod: lambda a, b: a % b, ast.BitAnd: lambda a, b: a & b, ast.BitOr: lambda a, b: a | b}
_CMPS = {ast.Lt: lambda a, b: a < b, ast.LtE: lambda a, b: a <= b, ast.Gt: lambda a, b: a > b,
         ast.GtE: lambda a, b: a >= b, ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b}


class ExprError(ValueError):
    pass


def evaluate(expr: str, store, x: torch.Tensor | None = None):
    """Evaluate `expr` against `store` (a WeightStore). Returns a tensor or a Python number."""
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as ex:
        raise ExprError(f"syntax error: {ex.msg}") from None
    dev = next(iter(store.params.values())).device

    def region(name, index=":"):
        from .weights import parse_index  # local import: weights imports this module
        p = store._get(name)
        return p.data[parse_index(index, p.ndim)].float()

    def check(v):
        if isinstance(v, torch.Tensor) and v.numel() > MAX_ELEMENTS:
            raise ExprError(f"intermediate result too large ({v.numel()} elements)")
        return v

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant):
            if isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
                return n.value
            if isinstance(n.value, str):
                return n.value  # only meaningful as an argument to W()
            raise ExprError(f"unsupported constant {n.value!r}")
        if isinstance(n, ast.Name):
            if n.id == "x":
                if x is None:
                    raise ExprError("'x' (the target region) only exists in a transform")
                return x
            if n.id in CONSTS:
                return CONSTS[n.id]
            raise ExprError(f"unknown name '{n.id}'")
        if isinstance(n, ast.BinOp) and type(n.op) in _BINOPS:
            return check(_BINOPS[type(n.op)](ev(n.left), ev(n.right)))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd, ast.Invert)):
            v = ev(n.operand)
            return -v if isinstance(n.op, ast.USub) else (~v if isinstance(n.op, ast.Invert) else v)
        if isinstance(n, ast.Compare) and len(n.ops) == 1 and type(n.ops[0]) in _CMPS:
            return _CMPS[type(n.ops[0])](ev(n.left), ev(n.comparators[0]))
        if isinstance(n, ast.Attribute):
            if n.attr == "T":
                return ev(n.value).T
            raise ExprError(f"attribute '.{n.attr}' is not allowed (only .T)")
        if isinstance(n, (ast.List, ast.Tuple)):
            return [ev(e) for e in n.elts]
        if isinstance(n, ast.Call):
            if not isinstance(n.func, ast.Name):
                raise ExprError("only plain function calls are allowed, e.g. norm(x)")
            fname = n.func.id
            args = [ev(a) for a in n.args]
            kw = {k.arg: ev(k.value) for k in n.keywords if k.arg}
            if fname == "W":
                if not args or not isinstance(args[0], str):
                    raise ExprError('W needs a parameter name: W("model.layers.0.mlp.up_proj.weight", "0:4, :")')
                return check(region(*args, **kw))
            if fname == "eye":
                return torch.eye(int(args[0]), device=dev)
            if fname not in FUNCS:
                raise ExprError(f"unknown function '{fname}'. Available: W, {', '.join(sorted(FUNCS))}")
            if fname in _ELEMENTWISE:  # these need tensor inputs; allow e.g. sqrt(2)
                args = [_t(a, dev) if isinstance(a, (int, float)) else a for a in args]
            try:
                return check(FUNCS[fname](*args, **_dim_kw(kw)))
            except (RuntimeError, TypeError, ValueError, IndexError) as ex:
                raise ExprError(f"{fname}(): {ex}") from None
        raise ExprError(f"'{ast.unparse(n)}' is not allowed")

    try:
        out = ev(tree)
    except ExprError:
        raise
    except (RuntimeError, TypeError, ValueError, IndexError, KeyError, ZeroDivisionError) as ex:
        raise ExprError(str(ex).split("\n")[0]) from None
    finally:
        # `ev` calls itself, so its closure refers to itself: a reference cycle that would keep this
        # frame (and `x`, possibly gigabytes of GPU memory) alive until the garbage collector runs.
        ev = None  # noqa: F841
    if isinstance(out, str):
        raise ExprError("the expression evaluates to a string, not a number or tensor")
    if isinstance(out, list):
        raise ExprError("the expression evaluates to a list; use it only as a function argument")
    return out


def summarize(v, max_values: int = 64) -> str:
    """Short text description of an evaluation result."""
    if not isinstance(v, torch.Tensor):
        return f"{v:.6g}"
    if v.dtype == torch.bool:
        return f"boolean tensor, shape {list(v.shape)}, {int(v.sum())} true of {v.numel()}"
    if v.numel() == 1:
        return f"{v.item():.6g}"
    f = v.float()
    stats = (f"shape {list(v.shape)}; mean {f.mean().item():.4g}, std {f.std().item():.4g}, "
             f"min {f.min().item():.4g}, max {f.max().item():.4g}, norm {f.norm().item():.4g}")
    vals = f.flatten()
    shown = ", ".join(f"{t:.4g}" for t in vals[:max_values].tolist())
    more = f" ... ({vals.numel() - max_values} more)" if vals.numel() > max_values else ""
    label = "values" if vals.numel() <= max_values else f"first {max_values} values"
    return f"{stats}\n{label}: {shown}{more}"
