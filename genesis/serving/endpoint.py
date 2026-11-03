"""Serving by endpoint: apps talk to GENESIS only through an OpenAI-compatible API.

One base model serves several domain adapters at once (vLLM multi-LoRA).
Apps never import GENESIS model code; they send chat requests to the endpoint
and pick an adapter by its served name. Swapping the base model therefore
changes nothing in the apps.

    # print the vLLM command for a base model plus named adapters
    python -m genesis.serving.endpoint vllm --base Qwen/Qwen3-4B \
        --adapter fin=runs/applied/fin-v1/adapter --port 8000

    # check a running server and list what it serves
    python -m genesis.serving.endpoint models --url http://localhost:8000/v1

    # one question to a served adapter
    python -m genesis.serving.endpoint chat --url http://localhost:8000/v1 --model fin "부채비율은?"

Ollama (``http://localhost:11434/v1``) speaks the same API for local tests.
Only the standard library is used, so an app can copy this file as-is.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import os
import shlex
import urllib.error
import urllib.request


class EndpointError(RuntimeError):
    pass


@dataclass
class Reply:
    text: str
    completion_tokens: int
    prompt_tokens: int
    finish_reason: str | None
    raw: dict = field(repr=False, default_factory=dict)


class EndpointClient:
    def __init__(self, base_url: str, api_key: str | None = None, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("GENESIS_API_KEY")
        self.timeout = timeout

    def _request(self, path: str, payload: dict | None = None) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(self.base_url + path, data=data, headers=headers,
                                         method="POST" if payload is not None else "GET")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", "replace")[:500]
            raise EndpointError(f"{error.code} from {self.base_url}{path}: {body}") from error
        except urllib.error.URLError as error:
            raise EndpointError(f"cannot reach {self.base_url}{path}: {error.reason}") from error

    def models(self) -> list[str]:
        return [m["id"] for m in self._request("/models").get("data", [])]

    def chat(self, model: str, messages: list[dict], max_tokens: int = 256, temperature: float = 0.0,
             template_kwargs: dict | None = None, **extra) -> Reply:
        payload = {"model": model, "messages": messages, "max_tokens": max_tokens,
                   "temperature": temperature, **extra}
        if template_kwargs:
            payload["chat_template_kwargs"] = template_kwargs  # vLLM extension; other servers ignore it
        body = self._request("/chat/completions", payload)
        try:
            choice = body["choices"][0]
            text = choice["message"].get("content") or ""
        except (KeyError, IndexError, TypeError) as error:
            raise EndpointError(f"unexpected response: {str(body)[:300]}") from error
        usage = body.get("usage") or {}
        return Reply(text=text, completion_tokens=int(usage.get("completion_tokens", 0)),
                     prompt_tokens=int(usage.get("prompt_tokens", 0)),
                     finish_reason=choice.get("finish_reason"), raw=body)


def vllm_command(base: str, adapters: dict[str, str] | None = None, port: int = 8000,
                 max_lora_rank: int = 16, max_model_len: int | None = None,
                 served_name: str | None = None) -> list[str]:
    """``vllm serve`` arguments for one base model and named LoRA adapters."""
    cmd = ["vllm", "serve", base, "--port", str(port)]
    if served_name:
        cmd += ["--served-model-name", served_name]
    if max_model_len:
        cmd += ["--max-model-len", str(max_model_len)]
    if adapters:
        cmd += ["--enable-lora", "--max-loras", str(len(adapters)), "--max-lora-rank", str(max_lora_rank),
                "--lora-modules", *[f"{name}={path}" for name, path in adapters.items()]]
    return cmd


def _parse_adapters(values: list[str]) -> dict[str, str]:
    adapters = {}
    for value in values:
        name, sep, path = value.partition("=")
        if not sep or not name or not path:
            raise SystemExit(f"--adapter expects name=path, got {value!r}")
        adapters[name] = path
    return adapters


def main(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    v = sub.add_parser("vllm", help="print the vLLM serve command")
    v.add_argument("--base", required=True)
    v.add_argument("--adapter", action="append", default=[], help="name=path, repeatable")
    v.add_argument("--port", type=int, default=8000)
    v.add_argument("--max-lora-rank", type=int, default=16)
    v.add_argument("--max-model-len", type=int, default=None)
    v.add_argument("--served-name", default=None)
    m = sub.add_parser("models", help="list served models")
    m.add_argument("--url", required=True)
    c = sub.add_parser("chat", help="ask one question")
    c.add_argument("--url", required=True)
    c.add_argument("--model", required=True)
    c.add_argument("--max-tokens", type=int, default=256)
    c.add_argument("question")
    args = p.parse_args(argv)
    if args.command == "vllm":
        cmd = vllm_command(args.base, _parse_adapters(args.adapter), args.port, args.max_lora_rank,
                           args.max_model_len, args.served_name)
        print(shlex.join(cmd))
        return cmd
    client = EndpointClient(args.url)
    if args.command == "models":
        names = client.models()
        print("\n".join(names))
        return names
    reply = client.chat(args.model, [{"role": "user", "content": args.question}], max_tokens=args.max_tokens)
    print(reply.text)
    return reply


if __name__ == "__main__":
    main()
