from swm import SelfEditAgent

REPLY = "I can't output the raw tags `<think>` or `</think>`. But here is the rest of my answer."


def run_step(mt, text, think):
    model, tok = mt
    events = []
    a = SelfEditAgent(model, tok, on_event=events.append, tools=False, think=think)
    a.messages.append({"role": "user", "content": "say <think>"})
    a._gen_think = think
    a.generate = lambda: text  # no model call: we only test how the reply is parsed and stored
    a.step()
    asst = [e for e in events if e["type"] == "assistant"][0]
    return a, asst


def test_tags_in_answer_are_text_when_thinking_is_off(mt):
    a, ev = run_step(mt, REPLY, think=False)
    assert ev["reasoning"] == "" and ev["content"] == REPLY
    a.messages.append({"role": "user", "content": "next"})
    assert REPLY in a.render()  # the template must not re-split the stored reply


def test_first_close_tag_ends_reasoning_when_thinking_is_on(mt):
    a, ev = run_step(mt, "Let me think.\n</think>\n\n" + REPLY, think=True)
    assert ev["reasoning"] == "Let me think." and ev["content"] == REPLY
    a.messages.append({"role": "user", "content": "next"})
    assert REPLY in a.render()


def test_cut_off_while_thinking(mt):
    _, ev = run_step(mt, "Still thinking about it", think=True)
    assert ev["reasoning"] == "Still thinking about it" and ev["content"] == ""
