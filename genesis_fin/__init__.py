"""genesis-fin: Korean listed-company financial statements on top of the GENESIS core.

    genesis_fin.dart         OpenDART client: company list + financial statements (cached)
    genesis_fin.build_tasks  statements -> extraction / ratio tasks with computed answers

Training and evaluation use the core: ``genesis.adapt.finetune`` and
``genesis.harness.evaluate`` (numeric scorer). This package depends on
``genesis``; ``genesis`` never imports this package.
"""
