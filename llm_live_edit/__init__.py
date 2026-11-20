from .weights import WeightStore, describe
from .session import ChatSession


def load(model_id: str = "Qwen/Qwen3.5-9B"):
    import torch
    from gb10_load_llm import from_pretrained_to_cuda
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = from_pretrained_to_cuda(AutoModelForCausalLM, model_id, device="cuda", dtype=torch.bfloat16)
    model.eval().requires_grad_(False)
    return model, tok


__all__ = ["WeightStore", "ChatSession", "describe", "load"]
