"""Serving by OpenAI-compatible endpoint (vLLM multi-LoRA, Ollama). Apps call the endpoint, never model code."""

__all__ = ["EndpointClient", "EndpointError", "Reply", "vllm_command"]


def __getattr__(name):  # lazy, so `python -m genesis.serving.endpoint` runs without a double import
    if name in __all__:
        from . import endpoint
        return getattr(endpoint, name)
    raise AttributeError(name)
