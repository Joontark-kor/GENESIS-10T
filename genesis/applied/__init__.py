"""GENESIS-Applied: specialise open LLMs for Korean/Japanese corporate finance.

Pipeline::

    genesis.applied.dart            OpenDART client: company list + financial statements (cached)
    genesis.applied.build_fin_tasks statements -> extraction / ratio tasks with computed answers
    genesis.applied.finetune        LoRA / QLoRA fine-tuning on chat data (assistant-only loss)
    genesis.applied.evaluate_tasks  accuracy, latency and cost of a base or tuned model
"""
