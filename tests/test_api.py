import pytest
from fastapi.testclient import TestClient

from swm.web import create_app

E = "model.embed_tokens.weight"
Q = "model.layers.3.self_attn.q_proj.weight"


@pytest.fixture(scope="module")
def client(mt):
    model, tok = mt
    with TestClient(create_app(model, tok, "Qwen/Qwen3.5-2B", debug=False, tools=False)) as c:
        yield c
        c.post("/api/reset")


def paris(c):
    r = c.post("/api/next_token", json={"prompt": "The capital of France is", "targets": [" Paris"]}).json()
    return r["targets"][0]


def test_edit_changes_probabilities_and_reset_restores(client):
    before = paris(client)
    assert before["rank"] == 1
    r = client.post("/api/transform", json={"name": E, "index": str(before["id"]), "expression": "x * -1"})
    assert r.status_code == 200 and r.json()["applied"] and r.json()["n_changed"] > 0
    after = paris(client)
    assert after["prob"] < before["prob"] * 1e-6
    state = client.get("/api/state").json()["state"]
    assert state["applied"][0]["by"] == "api" and state["applied"][0]["expr"] == "x * -1"
    client.post("/api/reset")
    assert paris(client)["prob"] == pytest.approx(before["prob"], rel=1e-6)


def test_edit_with_values_and_stage_only(client):
    r = client.post("/api/edit", json={"name": Q, "index": "0, 0:4", "values": [0, 0, 0.1, 0.1], "apply": False})
    assert r.json() == {"staged": True, "applied": False, "preview": "set 4 value(s)"}
    assert client.post("/api/read", json={"name": Q, "index": "0, 0:2"}).json()["values"] != [0, 0]
    client.post("/api/apply")
    assert client.post("/api/read", json={"name": Q, "index": "0, 0:4"}).json()["values"][:2] == [0, 0]
    client.post("/api/reset")


def test_errors_are_400_and_leave_nothing_staged(client):
    assert client.post("/api/transform", json={"name": Q, "index": "0", "expression": "x +"}).status_code == 400
    assert client.post("/api/transform", json={"name": Q, "index": "0", "expression": "__import__('os')"}).status_code == 400
    assert client.post("/api/edit", json={"name": "nope", "index": "0", "values": 1}).status_code == 400
    assert client.get("/api/state").json()["state"]["pending"] == []


def test_compute_and_generate(client):
    r = client.post("/api/compute", json={"expression": f'norm(W("{Q}", "0:4, :"), dim=1)'}).json()
    assert r["shape"] == [4] and len(r["values"]) == 4
    assert client.post("/api/compute", json={"expression": "1 + 2"}).json()["value"] == 3
    g = client.post("/api/generate", json={"prompt": "Say hi.", "max_new_tokens": 8}).json()
    assert g["new_tokens"] > 0 and isinstance(g["text"], str)


def test_tools_switch(client):
    assert client.post("/api/tools", json={"on": True}).json()["tools"] is True
    assert client.get("/api/state").json()["tools"] is True
    client.post("/api/tools", json={"on": False})
