# LLM Live Edit

Edit a running LLM's weights over HTTP, then chat with the edited model in the browser.

The server keeps one model (Qwen3.5, default 9B) in GPU memory. An HTTP API reads and edits
any weight region in place, with matrix math and a bit-exact reset. A browser chat runs on
whatever the weights are at that moment, so you see an edit's effect on the next reply. Every
edit is marked in the conversation. The model itself is never told.

## Setup

Developed on an NVIDIA GB10 with CUDA 13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

`causal-conv1d` (a faster kernel for Qwen3.5's linear-attention layers) is compiled from
source during `uv sync`. It needs the CUDA toolkit (`nvcc`) and can take ~15 minutes.

## Run

```bash
uv run python serve.py                                  # http://127.0.0.1:8000, Qwen3.5-9B
uv run python serve.py --model Qwen/Qwen3.5-0.8B --port 8001
uv run python serve.py --think --system "You are terse."
uv run pytest -q                                        # tests (load Qwen3.5-2B, ~30 s)
```

Open the address in a browser to chat. Edit from another terminal, a script or another agent.

## Editing from the command line (`edit.py`)

`edit.py` uses only the standard library, so it runs with any `python3`.

```bash
export LLM_LIVE_EDIT_URL=http://127.0.0.1:8000
python edit.py next "The capital of France is" --target " Paris"      # next-token probabilities
python edit.py transform model.embed_tokens.weight 11751 'x * -1' --note "flip the ' Paris' row"
python edit.py edit model.layers.3.self_attn.q_proj.weight "0:256, :" 2 --mode scale
python edit.py swap cat dog                                           # exchange two words' token rows
python edit.py compute 'cos_sim(W("model.layers.4.mlp.up_proj.weight", "0"), W("model.layers.4.mlp.up_proj.weight", "1"), dim=0)'
python edit.py generate "What is the capital of France?"              # one-off, not in the chat
python edit.py reset
```

## HTTP API

| Endpoint | Body | Does |
|---|---|---|
| `GET /api/state`, `/api/params?filter=`, `/api/describe?name=` | | model info, weight state, parameter list, what a tensor's rows/columns mean |
| `POST /api/read` | `name, index` | raw values (≤ 65,536) |
| `POST /api/compute` | `expression` | read-only matrix math |
| `POST /api/edit` | `name, index, values, mode, apply=true, note` | set / add / scale |
| `POST /api/transform` | `name, index, expression, apply=true, note` | region = expression of `x` |
| `POST /api/apply`, `/api/discard`, `/api/reset` | | staged edits; restore originals |
| `POST /api/generate` | `prompt, max_new_tokens, chat, think, system, temperature` | one-off generation |
| `POST /api/next_token` | `prompt, top_k, chat, targets` | next-token top-k and target probabilities |
| `POST /api/think` | `on` | thinking mode for the browser chat |

Calls are refused (409) while the model is generating a reply; bad input returns 400. The API
has no authentication: keep the default `--host 127.0.0.1`.

Indices are numpy-style strings: `":"` (whole tensor), `"0"` (row 0), `"0, 0:100"`,
`"0:256, :"`, `":, 7"`. Slices are half-open.

### Matrix math

`compute` and `transform` use a small, safe expression language (`llm_live_edit/expr.py`: parsed
with `ast` and checked node by node, never `eval`'d). `W("name", "index")` is any region; in a
transform, `x` is the target region's current value.

```
x * 2                                              double a region
W("model.layers.4.mlp.up_proj.weight", "0:8, :")   copy from another tensor
project_out(x, W("model.embed_tokens.weight", "7993"))   remove a direction from every row
low_rank(x, 64)                                    best rank-64 approximation
```

Operators `+ - * / ** @ .T`, comparisons and `& |`, plus `abs sqrt exp log tanh sigmoid relu clip
where sum mean std max min norm matmul outer normalize cos_sim project_out low_rank svals trace diag
flatten reshape zeros_like ones_like randn_like eye`. Staged expressions are evaluated when
applied, in order, so later ones see earlier edits.

## Python API

```python
from llm_live_edit import load, WeightStore
model, tok = load("Qwen/Qwen3.5-2B")
s = WeightStore(model)
s.read("model.layers.3.self_attn.q_proj.weight", "0, 0:100")                   # <= 512 values
s.write("model.layers.3.self_attn.q_proj.weight", "0, 0:4", "0, 0, 0.1, 0.1")  # staged
s.transform("model.layers.5.mlp.up_proj.weight", "0:8, :", "x * 2")             # staged
s.apply()   # live weights change; the next forward pass uses them
s.reset()   # restore every touched tensor bit-exactly
```

Also `compute()`, `stats()`, `discard()`, `drift()`, and `mode="add"|"scale"` for `write()`.

## Web UI

- Streaming chat with Markdown; Stop button or Esc; typing while it generates offers "Interrupt & send".
- Every API edit appears in the conversation with a divider: replies below it come from the edited model.
- Sidebar: staged and applied edits, distance from the original weights, and an inspector that
  draws any region (or its change from the original) as a heatmap.
- "What the model reads": the exact prompt for the next reply.
- The server terminal prints a timestamped trace (`--quiet` to turn it off). Chats are logged to `runs/*.jsonl`.

## Notes

- No KV cache is kept across turns: every reply re-reads the whole conversation with the current weights.
- Weights are bf16, so written values are rounded (0.1 → 0.1001). `apply` reports what was stored.
- Originals are backed up on the GPU the first time a tensor is touched, so reset is fast and exact.
- Qwen3.5-0.8B and 2B tie `model.embed_tokens.weight` to the output head; 9B has a separate `lm_head.weight`.
