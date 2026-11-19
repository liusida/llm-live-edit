"""Command-line client for a running web.py server: edit and probe the model's weights from outside.

    export SWM_URL=http://127.0.0.1:8001          # default http://127.0.0.1:8000

    python edit.py state
    python edit.py params "layers\\.3\\."
    python edit.py describe model.layers.3.self_attn.q_proj.weight
    python edit.py read model.layers.3.self_attn.q_proj.weight "0, 0:8"
    python edit.py compute 'norm(W("model.layers.3.mlp.up_proj.weight", "0:4, :"), dim=1)'
    python edit.py edit model.layers.3.self_attn.q_proj.weight "0, 0:4" "0, 0, 0.1, 0.1"
    python edit.py edit model.layers.3.self_attn.q_proj.weight "0:256, :" 2 --mode scale
    python edit.py transform model.layers.5.mlp.up_proj.weight "0:8, :" 'x * 2' --note "double 8 neurons"
    python edit.py apply | discard | reset
    python edit.py generate "What is the capital of France?"
    python edit.py next "The capital of France is" --target " Paris" --target " Lyon"
    python edit.py tools off          # plain chat (no self-edit prompt/tools)
    python edit.py think on

Uses only the standard library, so it runs with any python3.
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request


def call(base: str, method: str, path: str, body: dict | None = None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base.rstrip("/") + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as ex:
        try:
            detail = json.loads(ex.read()).get("detail")
        except Exception:
            detail = ex.reason
        sys.exit(f"error {ex.code}: {detail}")
    except urllib.error.URLError as ex:
        sys.exit(f"cannot reach {base}: {ex.reason} (is web.py running? set SWM_URL or --url)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.environ.get("SWM_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--json", action="store_true", help="print raw JSON responses")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("state")
    p = sub.add_parser("params"); p.add_argument("filter", nargs="?", default="")
    p = sub.add_parser("describe"); p.add_argument("name")
    p = sub.add_parser("read"); p.add_argument("name"); p.add_argument("index", nargs="?", default=":")
    p = sub.add_parser("compute"); p.add_argument("expression"); p.add_argument("--max-values", type=int, default=64)
    for c in ("edit", "transform"):
        p = sub.add_parser(c)
        p.add_argument("name"); p.add_argument("index")
        p.add_argument("values" if c == "edit" else "expression")
        if c == "edit":
            p.add_argument("--mode", default="set", choices=["set", "add", "scale"])
        p.add_argument("--stage-only", action="store_true", help="stage without applying")
        p.add_argument("--note", help="shown in the browser next to the edit")
    for c in ("apply", "discard", "reset", "clear"):
        sub.add_parser(c)
    p = sub.add_parser("generate"); p.add_argument("prompt")
    p.add_argument("--max", type=int, default=200); p.add_argument("--raw", action="store_true", help="no chat template")
    p.add_argument("--think", action="store_true"); p.add_argument("--system"); p.add_argument("--temp", type=float, default=0.0)
    p = sub.add_parser("next"); p.add_argument("prompt"); p.add_argument("--top", type=int, default=10)
    p.add_argument("--chat", action="store_true", help="wrap in the chat template")
    p.add_argument("--target", action="append", help="report this string's first-token probability")
    for c in ("tools", "think"):
        p = sub.add_parser(c); p.add_argument("onoff", choices=["on", "off"])
    a = ap.parse_args()
    u = a.url

    if a.cmd == "state":
        r = call(u, "GET", "/api/state")
    elif a.cmd == "params":
        r = call(u, "GET", "/api/params?filter=" + urllib.parse.quote(a.filter))
        if not a.json:
            for p in r:
                print(p["name"], p["shape"])
            return
    elif a.cmd == "describe":
        r = call(u, "GET", "/api/describe?name=" + urllib.parse.quote(a.name))
        if not a.json:
            print(r["text"]); return
    elif a.cmd == "read":
        r = call(u, "POST", "/api/read", {"name": a.name, "index": a.index})
        if not a.json:
            print(f"shape {r['shape']}"); print(", ".join(f"{v:.4g}" for v in r["values"])); return
    elif a.cmd == "compute":
        r = call(u, "POST", "/api/compute", {"expression": a.expression, "max_values": a.max_values})
        if not a.json:
            print(r["summary"]); return
    elif a.cmd == "edit":
        r = call(u, "POST", "/api/edit", {"name": a.name, "index": a.index, "values": a.values, "mode": a.mode,
                                          "apply": not a.stage_only, "note": a.note})
    elif a.cmd == "transform":
        r = call(u, "POST", "/api/transform", {"name": a.name, "index": a.index, "expression": a.expression,
                                               "apply": not a.stage_only, "note": a.note})
    elif a.cmd in ("apply", "discard", "reset", "clear"):
        r = call(u, "POST", f"/api/{a.cmd}")
    elif a.cmd == "generate":
        r = call(u, "POST", "/api/generate", {"prompt": a.prompt, "max_new_tokens": a.max, "chat": not a.raw,
                                              "think": a.think, "system": a.system, "temperature": a.temp})
        if not a.json:
            print(r["text"]); print(f"[{r['new_tokens']} tokens, {r['seconds']:.1f}s]", file=sys.stderr); return
    elif a.cmd == "next":
        r = call(u, "POST", "/api/next_token", {"prompt": a.prompt, "top_k": a.top, "chat": a.chat,
                                                "targets": a.target})
        if not a.json:
            for t in r["top"]:
                print(f"{t['prob']:8.4f}  {t['token']!r}")
            for t in r.get("targets", []):
                print(f"target {t['target']!r} (first token {t['first_token']!r}): p={t['prob']:.4g}, rank {t['rank']}")
            print(f"entropy {r['entropy']:.3f} nats"); return
    elif a.cmd in ("tools", "think"):
        r = call(u, "POST", f"/api/{a.cmd}", {"on": a.onoff == "on"})
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
