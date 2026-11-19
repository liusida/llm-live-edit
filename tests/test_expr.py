import pytest
import torch

from swm import SelfEditAgent, WeightStore
from swm.expr import ExprError

Q = "model.layers.3.self_attn.q_proj.weight"
UP4 = "model.layers.4.mlp.up_proj.weight"
UP5 = "model.layers.5.mlp.up_proj.weight"


@pytest.fixture
def s(mt):
    st = WeightStore(mt[0])
    yield st
    st.reset()


def region(s, name, index):
    return s.read(name, index)["values"]


def test_double_a_region(s):
    old = region(s, Q, "0:2, 0:4")
    s.transform(Q, "0:2, 0:4", "x * 2")
    s.apply()
    assert torch.equal(region(s, Q, "0:2, 0:4"), old * 2)  # doubling is exact in bf16
    assert torch.equal(region(s, Q, "2, 0:4"), s.original(Q, "2, 0:4"))  # neighbours untouched


def test_copy_from_another_tensor(s):
    s.transform(UP5, "0:2, 0:8", f'W("{UP4}", "0:2, 0:8")')
    s.apply()
    assert torch.equal(region(s, UP5, "0:2, 0:8"), region(s, UP4, "0:2, 0:8"))


def test_staged_expressions_apply_in_order(s):
    old = region(s, Q, "0, 0:4")
    s.transform(Q, "0, 0:4", "x * 2")
    s.transform(Q, "0, 0:4", "x + 1")
    s.apply()
    expect = (old * 2 + 1).bfloat16().float()
    assert torch.equal(region(s, Q, "0, 0:4"), expect)


def test_compute_matches_torch(s):
    w = s.params[UP4].data[0:4].float()
    assert torch.allclose(s.compute(f'norm(W("{UP4}", "0:4, :"), dim=1)'), w.norm(dim=1))
    assert s.compute(f'cos_sim(W("{UP4}", "0"), W("{UP4}", "0"), dim=0)').item() == pytest.approx(1, abs=1e-5)
    assert s.compute(f'svals(W("{UP4}", "0:16, 0:16"), 3)').shape == (3,)
    assert s.compute("2 + 3 * 4") == 14


def test_shape_mismatch_rejected_at_staging(s):
    with pytest.raises(ExprError, match="does not fit"):
        s.transform(Q, "0:2, 0:4", f'W("{UP4}", "0:3, 0:4")')
    assert not s.pending


@pytest.mark.parametrize("bad", [
    '__import__("os")', "x.__class__", "lambda: 1", 'open("f")', "[i for i in x]", "x[0]",
    "exec('1')", "x.cpu()", "y + 1",
])
def test_unsafe_or_unknown_rejected(s, bad):
    with pytest.raises(ExprError):
        s.transform(Q, "0, 0:4", bad)
    assert not s.pending


def test_failure_at_apply_rolls_back(s, mt):
    model, tok = mt
    before = model.model.layers[3].self_attn.q_proj.weight.detach().clone()
    s.transform(Q, "0, 0:1", "x * 0")
    s.transform(Q, "1, 0:4", f'x / W("{Q}", "0, 0:1")')  # fine when staged; divides by 0 once #1 applies
    with pytest.raises(RuntimeError, match="edit #2"):
        s.apply()
    assert torch.equal(model.model.layers[3].self_attn.q_proj.weight, before)
    s.discard()


def test_agent_tools(mt):
    model, tok = mt
    a = SelfEditAgent(model, tok, on_event=lambda ev: None)
    out = a.run_tool("compute_weights", {"expression": f'mean(W("{Q}", "0:8, :"))'})
    assert out.startswith("result: ")
    out = a.run_tool("transform_weights", {"name": Q, "index": "0, 0:4", "expression": "x * 2"})
    assert out.startswith("staged:") and "Preview" in out
    assert a.state()["pending"][0]["values"] == "= x * 2"
    out = a.run_tool("apply_edits", {})
    assert "x * 2: 4/4 values changed" in out or "x * 2:" in out
    out = a.run_tool("transform_weights", {"name": Q, "index": "0, 0:4", "expression": "import os"})
    assert out.startswith("error:")
    a.store.reset()
