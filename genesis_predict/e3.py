"""E3: does the physics engine help a small model? (GPU, research track)

Same model, same physics dev set, two conditions (greedy, thinking off):

* ``direct`` - solve, last line ``정답: 숫자``.
* ``spec``   - ``physics_set.PROMPT``: write ``시나리오: {...}`` and ``정답: 숫자``; the engine recomputes from
  the model's own spec. Policy ``engine``: if the spec solves, answer with the engine value, else keep the
  model's number.

Pre-registered criteria (docs/13), on the model's OWN spec:
* planted-error detection >= 80% and false block of correct claims <= 2%;
* secondary: accuracy of the ``engine`` policy vs ``direct`` over seeds (ranges must not overlap).

    python -m genesis_predict.e3 --model Qwen/Qwen3-4B --seeds 0,1,2 --out runs/e3
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from . import physics, physics_set

REPORT_KIND = "genesis_predict.e3/v1"
DIRECT = "다음 물리 문제를 풀어라. 풀이는 짧게 쓰고, 마지막 줄은 반드시 '정답: 숫자' 형식으로 써라(단위는 쓰지 말 것).\n문제: "
NUM_RE = re.compile(r"정답\s*[:：]\s*([-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?)")
UNSUP_RE = re.compile(r"시나리오\s*[:：]\s*미지원")


def parse_answer(text: str) -> float | None:
    m = NUM_RE.findall(text or "")
    if not m:
        return None
    try:
        return float(m[-1].replace(",", ""))
    except ValueError:
        return None


def normalize_spec(spec: dict | None) -> dict | None:
    """Light, rule-based cleanup of unit spellings a model commonly writes (not a second model)."""
    if not isinstance(spec, dict):
        return None
    fix = {"°C": "degC", "℃": "degC", "Ω": "ohm", "μ": "u", "µ": "u", "²": "**2", "³": "**3", "°": " deg"}

    def clean(v):
        s = str(v)
        for a, b in fix.items():
            s = s.replace(a, b)
        return s.strip()
    out = dict(spec)
    out["params"] = {k: clean(v) for k, v in (spec.get("params") or {}).items()} if isinstance(spec.get("params"), dict) else {}
    if out.get("unit"):
        out["unit"] = clean(out["unit"])
    return out


class Generator:
    def __init__(self, model_id: str, max_new_tokens: int = 384):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from genesis.precision import describe, inference_dtype

        self.torch = torch
        use_cuda = torch.cuda.is_available()
        self.precision = describe(torch, use_cuda)
        self.tok = AutoTokenizer.from_pretrained(model_id)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        kw = {"dtype": inference_dtype(torch, use_cuda)}
        if use_cuda:
            kw["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kw).eval()
        self.device = next(self.model.parameters()).device
        self.max_new_tokens = max_new_tokens
        self.hardware = torch.cuda.get_device_name() if use_cuda else "cpu"

    def __call__(self, prompts: list[str], batch_size: int = 16) -> list[str]:
        from genesis.adapt.chatdata import render

        outs = []
        for i in range(0, len(prompts), batch_size):
            chunk = [render(self.tok, [{"role": "user", "content": p}], add_generation_prompt=True,
                            template_kwargs={"enable_thinking": False}) for p in prompts[i:i + batch_size]]
            enc = self.tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
            with self.torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                          pad_token_id=self.tok.pad_token_id)
            for ids in gen[:, enc["input_ids"].shape[1]:]:
                outs.append(self.tok.decode(ids, skip_special_tokens=True))
            print(f"  {min(i + batch_size, len(prompts))}/{len(prompts)}", flush=True)
        return outs


def score(items: list[dict], direct_out: list[str], spec_out: list[str]) -> dict:
    """All numbers for one seed. Pure function of the texts, so it is unit-tested without a GPU."""
    d_outputs, s_outputs, e_outputs = {}, {}, {}
    det = {"wrong_n": 0, "detected": 0, "correct_n": 0, "blocked": 0, "unverifiable": 0}
    rows = []
    for it, dtext, stext in zip(items, direct_out, spec_out):
        unsup = bool(UNSUP_RE.search(stext or ""))
        spec = None if unsup else normalize_spec(physics.extract_spec(stext))
        d_ans, s_ans = parse_answer(dtext), parse_answer(stext)
        eng_val = None
        if spec:
            st, val = physics.solve(spec)
            if st == "ok":
                eng_val = float(val.magnitude)
        d_outputs[it["id"]] = {"answer": d_ans}
        s_outputs[it["id"]] = {"spec": spec, "answer": s_ans, "unsupported": unsup}
        e_outputs[it["id"]] = {"spec": spec, "answer": eng_val if eng_val is not None else s_ans, "unsupported": unsup}
        if it["status"] == "supported":
            for c in it["claims"]:
                v = physics.verify(spec, c["value"])["status"] if spec else "bad_spec"
                if c["error"] is None:
                    det["correct_n"] += 1
                    det["blocked"] += v == "mismatch"
                else:
                    det["wrong_n"] += 1
                    det["detected"] += v == "mismatch"
                det["unverifiable"] += v not in ("match", "mismatch")
        rows.append({"id": it["id"], "status": it["status"], "gold": it.get("gold"), "direct": d_ans, "model": s_ans,
                     "engine": eng_val, "spec": spec, "unsupported": unsup,
                     "direct_tail": (dtext or "")[-160:], "spec_tail": (stext or "")[-260:]})
    out = {"direct": physics_set.score_model(items, d_outputs), "spec_model": physics_set.score_model(items, s_outputs),
           "spec_engine": physics_set.score_model(items, e_outputs),
           "detection": det["detected"] / max(det["wrong_n"], 1), "false_block": det["blocked"] / max(det["correct_n"], 1),
           "unverifiable_claims": det["unverifiable"] / max(det["wrong_n"] + det["correct_n"], 1)}
    return {"summary": out, "rows": rows}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="Qwen/Qwen3-4B")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    gen = Generator(a.model)
    per_seed = {}
    for seed in [int(s) for s in a.seeds.split(",")]:
        items = physics_set.make(seed=seed)[: a.limit]
        t0 = time.time()
        d = gen([DIRECT + it["question"] for it in items], a.batch_size)
        s = gen([physics_set.PROMPT + it["question"] for it in items], a.batch_size)
        rep = score(items, d, s)
        rep["seconds"] = round(time.time() - t0, 1)
        (a.out / f"e3_seed{seed}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
        per_seed[seed] = rep["summary"]
        print(f"seed {seed}:", json.dumps(rep["summary"], ensure_ascii=False), flush=True)
    keys = [("direct", "accuracy"), ("spec_model", "accuracy"), ("spec_engine", "accuracy"), ("spec_engine", "spec_accuracy"),
            ("spec_model", "engine_rescue"), ("spec_model", "engine_harm"), ("spec_engine", "unsupported_recall")]
    table = {f"{a_}.{b}": [round(per_seed[s][a_][b], 4) for s in per_seed] for a_, b in keys}
    table["detection"] = [round(per_seed[s]["detection"], 4) for s in per_seed]
    table["false_block"] = [round(per_seed[s]["false_block"], 4) for s in per_seed]
    final = {"kind": REPORT_KIND, "model": a.model, "hardware": gen.hardware, "precision": gen.precision,
             "seeds": list(per_seed), "by_seed": table,
             "pass_detection": min(table["detection"]) >= 0.8 and max(table["false_block"]) <= 0.02,
             "engine_beats_direct": min(table["spec_engine.accuracy"]) > max(table["direct.accuracy"])}
    (a.out / "e3_summary.json").write_text(json.dumps(final, ensure_ascii=False, indent=1), encoding="utf-8")
    print("E3", json.dumps(final, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
