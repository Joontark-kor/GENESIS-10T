"""Export a dense GENESIS checkpoint to a Hugging Face Llama folder.

    python -m genesis.export_hf --checkpoint checkpoints/dense-100m/model.pt \
        --output exports/dense-100m-hf

The folder loads with ``transformers.AutoModelForCausalLM`` and
``AutoTokenizer``, so standard harnesses (lm-evaluation-harness, HRET, vLLM)
can evaluate or serve it. The architecture maps one-to-one onto Llama:
RMSNorm pre-norm, rotate-half RoPE, GQA, SwiGLU, tied embeddings optional.
``--verify`` (default on) runs both models on random tokens and fails if the
logits differ.

Not exportable yet: MoE checkpoints (n_experts > 1), learned positions,
LayerNorm, the byte tokenizer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import torch

from .config import ModelConfig
from .chat import HF_CHAT_TEMPLATE
from .tokenizer import EOS_TOKEN

PAD_TOKEN = "<|pad|>"


def check_exportable(config: ModelConfig, tokenizer_info: dict) -> None:
    problems = []
    if config.is_moe:
        problems.append("MoE checkpoints are not exportable yet (dense only)")
    if config.position != "rope":
        problems.append("learned positions have no Llama equivalent")
    if config.norm != "rmsnorm":
        problems.append("LayerNorm models have no Llama equivalent")
    if tokenizer_info.get("type") != "bpe":
        problems.append("the byte tokenizer cannot be exported; train a BPE tokenizer")
    if problems:
        raise ValueError("; ".join(problems))


def llama_config(config: ModelConfig, eos_id: int, pad_id: int | None) -> dict:
    return {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": config.vocab_size,
        "hidden_size": config.d_model,
        "intermediate_size": config.expert_hidden_dim,
        "num_hidden_layers": config.n_layers,
        "num_attention_heads": config.n_heads,
        "num_key_value_heads": config.kv_heads,
        "head_dim": config.head_dim,
        "max_position_embeddings": config.context_length,
        "rms_norm_eps": config.norm_eps,
        "rope_theta": config.rope_theta,
        "hidden_act": "silu",
        "attention_bias": config.attention_bias,
        "mlp_bias": False,
        "tie_word_embeddings": config.tie_embeddings,
        "bos_token_id": None,
        "eos_token_id": eos_id,
        "pad_token_id": pad_id,
        "torch_dtype": "float32",
        "initializer_range": 0.02,
    }


def convert_state_dict(state: dict[str, torch.Tensor], config: ModelConfig) -> dict[str, torch.Tensor]:
    q_width = config.n_heads * config.head_dim
    kv_width = config.kv_heads * config.head_dim
    out = {"model.embed_tokens.weight": state["token_embedding.weight"],
           "model.norm.weight": state["final_norm.weight"]}
    if not config.tie_embeddings:
        out["lm_head.weight"] = state["lm_head.weight"]
    for i in range(config.n_layers):
        src, dst = f"blocks.{i}.", f"model.layers.{i}."
        q, k, v = state[src + "attention.qkv.weight"].split((q_width, kv_width, kv_width), dim=0)
        out[dst + "self_attn.q_proj.weight"] = q
        out[dst + "self_attn.k_proj.weight"] = k
        out[dst + "self_attn.v_proj.weight"] = v
        out[dst + "self_attn.o_proj.weight"] = state[src + "attention.proj.weight"]
        if config.attention_bias:
            qb, kb, vb = state[src + "attention.qkv.bias"].split((q_width, kv_width, kv_width), dim=0)
            out[dst + "self_attn.q_proj.bias"] = qb
            out[dst + "self_attn.k_proj.bias"] = kb
            out[dst + "self_attn.v_proj.bias"] = vb
            out[dst + "self_attn.o_proj.bias"] = state[src + "attention.proj.bias"]
        gate, up = state[src + "ffn.gate_up.weight"].chunk(2, dim=0)
        out[dst + "mlp.gate_proj.weight"] = gate
        out[dst + "mlp.up_proj.weight"] = up
        out[dst + "mlp.down_proj.weight"] = state[src + "ffn.down.weight"]
        out[dst + "input_layernorm.weight"] = state[src + "attention_norm.weight"]
        out[dst + "post_attention_layernorm.weight"] = state[src + "ffn_norm.weight"]
    return {k: v.detach().to(torch.float32).contiguous().clone() for k, v in out.items()}


def export(checkpoint_path: Path, output: Path, verify: bool = True) -> Path:
    from safetensors.torch import save_file

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = ModelConfig.from_dict(checkpoint["config"])
    tok_info = checkpoint.get("tokenizer") or {}
    check_exportable(config, tok_info)
    tokenizer_path = Path(tok_info["spec"])
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"tokenizer {tokenizer_path} recorded in the checkpoint is missing")

    output.mkdir(parents=True, exist_ok=True)
    from tokenizers import Tokenizer as HFTokenizer
    raw_tok = HFTokenizer.from_file(str(tokenizer_path))
    eos_id = raw_tok.token_to_id(EOS_TOKEN)
    pad_id = raw_tok.token_to_id(PAD_TOKEN)

    (output / "config.json").write_text(json.dumps(llama_config(config, eos_id, pad_id), indent=2), encoding="utf-8")
    (output / "generation_config.json").write_text(json.dumps(
        {"eos_token_id": eos_id, "pad_token_id": pad_id, "do_sample": True, "temperature": 0.8,
         "top_k": 40}, indent=2), encoding="utf-8")
    save_file(convert_state_dict(checkpoint["model_state"], config), str(output / "model.safetensors"),
              metadata={"format": "pt"})
    shutil.copyfile(tokenizer_path, output / "tokenizer.json")
    (output / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "PreTrainedTokenizerFast",
        "eos_token": EOS_TOKEN, "pad_token": PAD_TOKEN, "bos_token": None, "unk_token": None,
        "model_max_length": config.context_length,
        "add_bos_token": False, "add_eos_token": False, "clean_up_tokenization_spaces": False,
        "chat_template": HF_CHAT_TEMPLATE,
    }, indent=2), encoding="utf-8")
    (output / "special_tokens_map.json").write_text(json.dumps(
        {"eos_token": EOS_TOKEN, "pad_token": PAD_TOKEN}, indent=2), encoding="utf-8")
    (output / "genesis_export.json").write_text(json.dumps({
        "source_checkpoint": str(checkpoint_path), "step": checkpoint.get("step"),
        "parameters": checkpoint.get("parameters"), "tokenizer_sha256": tok_info.get("sha256"),
        "genesis_config": checkpoint["config"], "val": checkpoint.get("val"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    if verify:
        max_diff = verify_export(checkpoint_path, output)
        print(f"verified: max |logit difference| = {max_diff:.2e}")
    return output


@torch.no_grad()
def verify_export(checkpoint_path: Path, output: Path, atol: float = 1e-3) -> float:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .generate import load_model

    ours, tokenizer, _ = load_model(checkpoint_path, "cpu")
    theirs = AutoModelForCausalLM.from_pretrained(output, torch_dtype=torch.float32).eval()
    hf_tok = AutoTokenizer.from_pretrained(output)
    sample = "대한민국의 수도는 서울이며, 2026년 인구는 약 930만 명이다. Hello 123!"
    if hf_tok(sample, add_special_tokens=False)["input_ids"] != tokenizer.encode(sample):
        raise AssertionError("exported tokenizer encodes differently from the original")
    generator = torch.Generator().manual_seed(0)
    length = min(32, ours.config.context_length)
    ids = torch.randint(0, tokenizer.vocab_size, (2, length), generator=generator)
    diff = (ours(ids)["logits"] - theirs(ids).logits).abs().max().item()
    if diff > atol:
        raise AssertionError(f"exported model logits differ by {diff:.3e} (> {atol})")
    return diff


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--no-verify", action="store_true")
    args = p.parse_args()
    out = export(args.checkpoint, args.output, verify=not args.no_verify)
    print(f"exported to {out}")


if __name__ == "__main__":
    main()
