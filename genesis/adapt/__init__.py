"""Adapting an open base model: LoRA / QLoRA fine-tuning on chat data (assistant-only loss).

    genesis.adapt.chatdata  chat examples -> token ids with loss only on assistant replies
    genesis.adapt.finetune  LoRA / QLoRA training CLI that writes the adapter and run.json
"""
