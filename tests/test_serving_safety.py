import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading

import pytest

from genesis.harness.evaluate import contamination, main as eval_main
from genesis.harness.scorers import exact, numeric, resolve
from genesis.safety import Action, AllowAll, SafetyCheck, Verdict, guarded_chat, load_safety_check
from genesis.serving.endpoint import EndpointClient, EndpointError, main as endpoint_main, vllm_command


class _FakeOpenAI(BaseHTTPRequestHandler):
    requests: list = []

    def log_message(self, *args):
        pass

    def _send(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/v1/models":
            self._send(200, {"data": [{"id": "Qwen/Qwen3-4B"}, {"id": "fin"}]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append({"body": body, "auth": self.headers.get("Authorization")})
        if body["model"] == "missing":
            return self._send(404, {"error": {"message": "model not found"}})
        question = body["messages"][-1]["content"]
        self._send(200, {"choices": [{"message": {"role": "assistant", "content": f"계산 결과\n정답: {len(question)}"},
                                      "finish_reason": "stop"}],
                         "usage": {"prompt_tokens": 7, "completion_tokens": 5}})


@pytest.fixture()
def server():
    _FakeOpenAI.requests = []
    httpd = HTTPServer(("127.0.0.1", 0), _FakeOpenAI)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}/v1"
    httpd.shutdown()


def test_endpoint_client(server):
    client = EndpointClient(server, api_key="test-key")
    assert client.models() == ["Qwen/Qwen3-4B", "fin"]
    reply = client.chat("fin", [{"role": "user", "content": "abc"}], template_kwargs={"enable_thinking": False})
    assert reply.text.endswith("정답: 3") and reply.completion_tokens == 5 and reply.finish_reason == "stop"
    sent = _FakeOpenAI.requests[-1]
    assert sent["auth"] == "Bearer test-key" and sent["body"]["temperature"] == 0.0
    assert sent["body"]["chat_template_kwargs"] == {"enable_thinking": False}
    with pytest.raises(EndpointError, match="404"):
        client.chat("missing", [{"role": "user", "content": "x"}])
    with pytest.raises(EndpointError, match="cannot reach"):
        EndpointClient("http://127.0.0.1:9/v1", timeout=2).models()


def test_vllm_command_and_cli():
    cmd = vllm_command("Qwen/Qwen3-4B", {"fin": "runs/fin/adapter", "other": "runs/x"}, port=8001)
    assert cmd[:3] == ["vllm", "serve", "Qwen/Qwen3-4B"] and "--enable-lora" in cmd
    assert cmd[cmd.index("--max-loras") + 1] == "2"
    assert cmd[-2:] == ["fin=runs/fin/adapter", "other=runs/x"]
    assert "--enable-lora" not in vllm_command("Qwen/Qwen3-4B")
    printed = endpoint_main(["vllm", "--base", "Qwen/Qwen3-4B", "--adapter", "fin=runs/fin/adapter"])
    assert printed[-1] == "fin=runs/fin/adapter"
    with pytest.raises(SystemExit):
        endpoint_main(["vllm", "--base", "b", "--adapter", "no-equals-sign"])


def test_harness_evaluates_a_served_model(server, tmp_path):
    data = tmp_path / "eval.jsonl"
    items = [{"id": "a", "kind": "len", "messages": [{"role": "user", "content": "abcd"}], "gold": 4, "tolerance": 0},
             {"id": "b", "kind": "len", "messages": [{"role": "user", "content": "xy"}], "gold": 3, "tolerance": 0},
             {"id": "c", "kind": "word", "scorer": "exact", "messages": [{"role": "user", "content": "q"}],
              "gold": "1"}]
    data.write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in items), encoding="utf-8")
    train = tmp_path / "train.jsonl"
    train.write_text(json.dumps({"messages": [{"role": "user", "content": "abcd"},
                                              {"role": "assistant", "content": "4"}]}), encoding="utf-8")
    report = eval_main(["run", "--endpoint", server, "--model", "fin", "--data", str(data), "--name", "served",
                        "--train-data", str(train), "--gpu-hour-price", "1.0", "--out", str(tmp_path / "r.json")])
    assert [r["correct"] for r in report["rows"]] == [True, False, True]
    assert report["by_kind"]["len"]["accuracy"] == 0.5 and report["hardware"].startswith("endpoint:")
    assert report["speed"]["generated_tokens"] == 15
    assert report["contamination"]["overlap_ids"] == ["a"]
    with pytest.raises(SystemExit):
        eval_main(["run", "--endpoint", server, "--model", "fin", "--adapter", "x", "--data", str(data), "--name", "n"])


def test_scorers():
    assert numeric("정답: 1,234.5", {"gold": 1234.5}) == (1234.5, True)
    assert numeric("no number", {"gold": 1})[1] is False
    assert exact("Answer:  Seoul ", {"gold": "seoul"}) == ("Seoul", True)
    assert resolve("genesis.harness.scorers:exact") is exact
    with pytest.raises(ValueError):
        resolve("nope")


def test_contamination_without_training_data():
    assert contamination([{"messages": [{"role": "user", "content": "x"}]}], []) == {"checked": False}


class BlockWord:
    name = "test-block-word"

    def check_input(self, messages):
        if "blocked" in messages[-1]["content"]:
            return Verdict(Action.BLOCK, ("matched test word",), replacement="cannot help with that")
        return Verdict()

    def check_output(self, messages, reply):
        if "secret" in reply:
            return Verdict(Action.REPLACE, ("reply rule",), replacement="[removed]")
        return Verdict()


def test_safety_interface(monkeypatch):
    calls = []

    def ask(messages):
        calls.append(messages)
        return "the secret is 42" if "tell" in messages[-1]["content"] else "fine"

    check = BlockWord()
    assert isinstance(check, SafetyCheck)
    blocked = guarded_chat(check, ask, [{"role": "user", "content": "blocked request"}])
    assert blocked.text == "cannot help with that" and not blocked.called_model and blocked.action is Action.BLOCK
    assert calls == []
    replaced = guarded_chat(check, ask, [{"role": "user", "content": "tell me"}])
    assert replaced.text == "[removed]" and replaced.action is Action.REPLACE
    ok = guarded_chat(check, ask, [{"role": "user", "content": "hello"}])
    assert ok.text == "fine" and ok.action is Action.ALLOW and ok.check_name == "test-block-word"

    monkeypatch.delenv("GENESIS_SAFETY_CHECK", raising=False)
    default = load_safety_check()
    assert isinstance(default, AllowAll)
    assert "no safety check configured" in guarded_chat(default, ask, [{"role": "user", "content": "x"}]).input_verdict.reasons
    monkeypatch.setenv("GENESIS_SAFETY_CHECK", "tests.test_serving_safety:BlockWord")
    assert load_safety_check().name == "test-block-word"
    with pytest.raises(ValueError):
        load_safety_check("no_colon")
    with pytest.raises(TypeError):
        load_safety_check("tests.test_serving_safety:_FakeOpenAI")
