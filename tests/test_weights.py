import pytest
import torch

from swm import WeightStore, load
from swm.agent import parse_calls
from swm.weights import parse_index, parse_values

Q = "model.layers.3.self_attn.q_proj.weight"


def test_parse_index():
    assert parse_index("0, 0:100", 2) == (0, slice(0, 100))
    assert parse_index("[:, 7]", 2) == (slice(None), 7)
    assert parse_index("::2", 1) == (slice(None, None, 2),)
    with pytest.raises(ValueError):
        parse_index("0, 0, 0", 2)


def test_parse_values():
    assert parse_values("0, 0, 0.1, 0.1").tolist() == pytest.approx([0, 0, 0.1, 0.1])
    assert parse_values("[1 2 3]").tolist() == [1, 2, 3]
    assert parse_values(0.5).tolist() == [0.5]


def test_parse_calls():
    text = ("Let me look.\n<tool_call>\n<function=read_weights>\n<parameter=name>\nx.weight\n</parameter>\n"
            "<parameter=index>\n0, 0:4\n</parameter>\n</function>\n</tool_call>")
    assert parse_calls(text) == [("read_weights", {"name": "x.weight", "index": "0, 0:4"})]


@pytest.fixture(scope="module")
def mt():
    return load("Qwen/Qwen3.5-2B")  # small + tied embeddings (used by the behavior test)


def logits(model, tok, text="The capital of France is"):
    ids = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        return model(**ids).logits[0, -1].float().cpu()


def answer(model, tok, q="What is the capital of France? Answer in one word."):
    s = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=False)
    ids = tok(s, return_tensors="pt").to(model.device)
    out = model.generate(**ids, max_new_tokens=10, do_sample=False)
    return tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True)


def test_read_write_apply_reset(mt):
    model, tok = mt
    s = WeightStore(model)
    base = logits(model, tok)
    orig = s.read(Q, "0, 0:4")["values"].clone()

    s.write(Q, "0:256, :", "0")          # zero head 0's query rows
    assert torch.equal(logits(model, tok), base), "staged edit must not change the model"

    rep = s.apply()[0]
    assert rep["n"] == 256 * 2048
    assert s.read(Q, "0, 0:4")["values"].tolist() == [0, 0, 0, 0]
    assert not torch.equal(logits(model, tok), base), "applied edit must change the output"

    assert s.reset() == 1
    assert torch.equal(s.read(Q, "0, 0:4")["values"], orig)
    assert torch.equal(logits(model, tok), base), "reset must be bit-exact"


def test_values_are_rounded_to_bf16(mt):
    model, _ = mt
    s = WeightStore(model)
    s.write(Q, "1, 0:2", "0.1, 0.123456789")
    stored = s.apply()[0]["stored_preview"]
    assert stored.tolist() == torch.tensor([0.1, 0.123456789]).bfloat16().float().tolist()
    s.reset()


def test_errors(mt):
    model, _ = mt
    s = WeightStore(model)
    with pytest.raises(ValueError):
        s.write(Q, "0, 0:4", "1, 2, 3")  # wrong count
    with pytest.raises(ValueError):
        s.read(Q, "0")                   # 2048 values > MAX_READ
    with pytest.raises(KeyError):
        s.read("nope", "0")
    assert not s.pending


def test_add_and_scale_modes(mt):
    model, _ = mt
    s = WeightStore(model)
    o = s.read(Q, "2, 0:3")["values"]
    s.write(Q, "2, 0:3", "2", mode="scale")
    s.write(Q, "2, 0:3", "1", mode="add")
    s.apply()
    assert s.read(Q, "2, 0:3")["values"].tolist() == pytest.approx((o * 2 + 1).tolist(), rel=1e-2)
    s.reset()


def test_behavior_change_via_tied_embedding(mt):
    model, tok = mt
    s = WeightStore(model)
    before = answer(model, tok)
    assert "Paris" in before
    # Negating these rows blocks the exact tokens, not the knowledge: the model
    # answered "PARIS" (different tokens) when this was written.
    for t in {"Paris", " Paris"}:
        tid = tok.convert_tokens_to_ids(tok.tokenize(t))
        assert len(tid) == 1
        s.write("model.embed_tokens.weight", str(tid[0]), "-1", mode="scale")
    s.apply()
    changed = answer(model, tok)
    print("after edit:", changed)
    assert "Paris" not in changed and changed != before
    s.reset()
    assert answer(model, tok) == before


def test_empty_region_rejected(mt):
    model, _ = mt
    s = WeightStore(model)
    with pytest.raises(ValueError, match="selects no elements"):
        s.write(Q, "0:0", "0")
    assert not s.pending


def test_whole_tensor_broadcast(mt):
    model, tok = mt
    s = WeightStore(model)
    base = logits(model, tok)
    name = "model.layers.0.mlp.down_proj.weight"
    e = s.write(name, ":", "0")
    assert e.values.numel() == 1 and e.n == s.params[name].numel()  # not materialized
    rep = s.apply()[0]
    assert rep["n"] == s.params[name].numel() and len(rep["stored_preview"]) == 16
    assert s.stats(name)["abs_mean"] == 0
    s.reset()
    assert torch.equal(logits(model, tok), base)


def test_apply_is_all_or_nothing(mt):
    from swm.weights import Edit
    model, tok = mt
    s = WeightStore(model)
    base = logits(model, tok)
    s.write(Q, "0:4, :", "0")                                          # valid
    s.pending.append(Edit(Q, "0, 0:4", torch.zeros(3), "set", 4))      # corrupt: wrong shape
    with pytest.raises(RuntimeError, match="edit #2"):
        s.apply()
    assert torch.equal(logits(model, tok), base), "first edit must be rolled back"
    assert len(s.pending) == 2 and not s._backup and not s.applied
    s.discard()
