"""A tiny OpenAI-compatible server on top of transformers, for trying an app against a real model.

Not for production (one request at a time, no batching): use vLLM for that (``genesis.serving.endpoint vllm``).
It exists so ``EndpointClient`` / ``CalcClient`` can be checked over real HTTP with a real model on a machine
where vLLM or Ollama is not installed (a Colab T4, a laptop), with nothing downloaded except the model.

    python -m genesis.serving.mini_server --model Qwen/Qwen3-4B --port 8000
"""
from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

# generate(messages, max_tokens, temperature, template_kwargs) -> (text, prompt_tokens, completion_tokens)
Generate = Callable[[list, int, float, dict], tuple[str, int, int]]  # template kwargs may hold continue_final_message


def make_handler(model_name: str, generate: Generate):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: dict) -> None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802 (http.server API)
            if self.path.rstrip("/") == "/v1/models":
                self._send(200, {"object": "list", "data": [{"id": model_name, "object": "model"}]})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if self.path.rstrip("/") != "/v1/chat/completions":
                return self._send(404, {"error": "not found"})
            try:
                payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                messages = payload["messages"]
            except (ValueError, KeyError):
                return self._send(400, {"error": "expected JSON with messages"})
            text, prompt_tokens, completion_tokens = generate(
                messages, int(payload.get("max_tokens", 256)), float(payload.get("temperature", 0.0)),
                {**(payload.get("chat_template_kwargs") or {}),
                 **{k: payload[k] for k in ("continue_final_message", "add_generation_prompt") if k in payload}})
            self._send(200, {
                "id": f"chatcmpl-{int(time.time() * 1000)}", "object": "chat.completion", "model": model_name,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                          "total_tokens": prompt_tokens + completion_tokens}})

        def log_message(self, *args):  # keep notebook output quiet
            pass

    return Handler


def transformers_generate(model_id: str) -> Generate:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32  # T4 has no bf16
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype,
                                                 device_map="auto" if torch.cuda.is_available() else None)

    def generate(messages, max_tokens, temperature, template_kwargs):
        template_kwargs = {"add_generation_prompt": True, **template_kwargs}
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, **template_kwargs)
        encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
        options = {"do_sample": True, "temperature": temperature} if temperature > 0 else {"do_sample": False}
        with torch.no_grad():
            out = model.generate(**encoded, max_new_tokens=max_tokens, **options)
        new = out[0][encoded["input_ids"].shape[1]:]
        return tokenizer.decode(new, skip_special_tokens=True), int(encoded["input_ids"].shape[1]), int(new.shape[0])

    return generate


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1")
    args = p.parse_args(argv)
    handler = make_handler(args.model, transformers_generate(args.model))
    print(f"serving {args.model} on http://{args.host}:{args.port}/v1", flush=True)
    ThreadingHTTPServer((args.host, args.port), handler).serve_forever()


if __name__ == "__main__":
    main()
