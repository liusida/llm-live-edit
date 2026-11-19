# self-weight-modification

An LLM agent (Qwen3.5, default 9B) with tools to read and edit its own weights in memory.
After an edit is applied, the agent keeps working on the modified weights, so it can observe
(or fail to observe) what it did to itself.

## Setup

Developed on an NVIDIA GB10 with CUDA 13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

`causal-conv1d` (a faster kernel for Qwen3.5's linear-attention layers) is compiled from
source during `uv sync`. It needs the CUDA toolkit (`nvcc`) and can take ~15 minutes.

## Run

```bash
uv run python web.py                  # web UI on http://127.0.0.1:8000 (Qwen3.5-9B)
uv run python web.py --model Qwen/Qwen3.5-2B --port 8001
uv run python main.py                 # terminal (/reset, /status, /quit)
uv run python main.py --task "..."    # one task
uv run pytest -q                      # tests (load the model, ~30 s)
```

## Python API (`swm.WeightStore`)

```python
from swm import load, WeightStore
model, tok = load()
s = WeightStore(model)
s.read("model.layers.3.self_attn.q_proj.weight", "0, 0:100")   # <= 512 values
s.write("model.layers.3.self_attn.q_proj.weight", "0, 0:4", "0, 0, 0.1, 0.1")  # staged
s.apply()   # live weights change; the next forward pass uses them
s.reset()   # restore every touched tensor bit-exactly
```

Also `stats()`, `discard()`, `drift()`, and `mode="add"|"scale"` for `write()`.

Matrix math uses a small, safe expression language (`swm/expr.py`: parsed with `ast` and
checked node by node, never `eval`'d):

```python
s.compute('norm(W("model.layers.4.mlp.up_proj.weight", "0:4, :"), dim=1)')   # read-only
s.transform("model.layers.3.self_attn.q_proj.weight", "0:256, :", "x * 2")   # staged; x = region
s.transform("model.layers.5.mlp.up_proj.weight", "0:8, :",
            'x + 0.1 * W("model.layers.4.mlp.up_proj.weight", "0:8, :")')
```

Operators `+ - * / ** @ .T`, comparisons and `& |`, plus `abs sqrt exp log tanh sigmoid relu clip
where sum mean std max min norm matmul outer normalize cos_sim project_out low_rank svals trace diag
flatten reshape zeros_like ones_like randn_like eye`. Expression edits are evaluated when applied,
in order, so they see earlier edits in the same batch.

## Editing from outside (third-party API)

The running `web.py` server also exposes the weights to other programs, so someone else (a
person, a script, another agent) can edit the model while you chat with it in the browser.
Turn **Self-edit tools** off in the header (or start with `--plain`) to chat with a plain model:
no system prompt, no tools, and it is not told about edits.

```bash
export SWM_URL=http://127.0.0.1:8000
python edit.py next "The capital of France is" --target " Paris"      # next-token probabilities
python edit.py transform model.embed_tokens.weight 11751 'x * -1' --note "flip the ' Paris' row"
python edit.py edit model.layers.3.self_attn.q_proj.weight "0:256, :" 2 --mode scale
python edit.py compute 'cos_sim(W("model.layers.4.mlp.up_proj.weight", "0"), W("model.layers.4.mlp.up_proj.weight", "1"), dim=0)'
python edit.py generate "What is the capital of France?"              # one-off, not in the chat
python edit.py reset
```

| Endpoint | Body | Does |
|---|---|---|
| `GET /api/state`, `/api/params?filter=`, `/api/describe?name=` | | model info, weight state, parameter list |
| `POST /api/read` | `name, index` | raw values (≤ 65,536) |
| `POST /api/compute` | `expression` | read-only matrix math |
| `POST /api/edit` | `name, index, values, mode, apply=true, note` | set / add / scale |
| `POST /api/transform` | `name, index, expression, apply=true, note` | region = expression of `x` |
| `POST /api/apply`, `/api/discard`, `/api/reset` | | staged edits; restore originals |
| `POST /api/generate` | `prompt, max_new_tokens, chat, think, system, temperature` | one-off generation |
| `POST /api/next_token` | `prompt, top_k, chat, targets` | next-token top-k and target probabilities |
| `POST /api/tools`, `/api/think` | `on` | self-edit mode, thinking mode |

Calls are refused (409) while the model is generating. Every edit appears in the browser with
a "via API" divider. The API has no authentication: keep the default `--host 127.0.0.1`.

## Web UI

- Streaming conversation with tool calls shown as cards; a divider marks every weight change.
- Stop with the Stop button or Esc; typing while the model generates offers "Interrupt & send".
- Sidebar: staged and applied edits, distance from the original weights, and an inspector that
  draws any region (or its change from the original) as a heatmap.
- "Instructions the model sees": the system prompt, the exact system block rendered by the chat
  template, and the full context for the next turn.
- The server terminal prints a timestamped trace (`--quiet` to turn it off).

## Agent tools

`list_weights`, `describe_weight` (per-head row/column ranges), `read_weights`,
`weight_stats`, `write_weights`, `compute_weights` (read-only matrix math),
`transform_weights` (stage an edit from an expression), `apply_edits`, `discard_edits`,
`reset_weights`.

No KV cache is kept across turns: every turn re-reads the whole conversation
with the current weights. Each turn ends at the first tool call. Logs go to `runs/*.jsonl`.

Things the model did not do itself (you resetting its weights, the current weight state) are
told to it as `[environment]` notes in the conversation.

Notes: weights are bf16, so written values are rounded (0.1 -> 0.1001). Qwen3.5-2B ties
`model.embed_tokens.weight` to the output head; 9B has a separate `lm_head.weight`.
