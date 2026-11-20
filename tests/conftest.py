import pytest

from llm_live_edit import load


@pytest.fixture(scope="session")
def mt():
    return load("Qwen/Qwen3.5-2B")  # small + tied embeddings (used by the behavior test)
