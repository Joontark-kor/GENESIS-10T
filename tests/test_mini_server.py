import threading
from http.server import ThreadingHTTPServer

from genesis.serving.calc import CalcClient
from genesis.serving.endpoint import EndpointClient
from genesis.serving.mini_server import make_handler


def serve(generate):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler("stub", generate))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, EndpointClient(f"http://127.0.0.1:{server.server_address[1]}/v1")


def test_endpoint_client_and_calc_client_work_over_real_http():
    seen = {}

    def generate(messages, max_tokens, temperature, template_kwargs):
        seen["kwargs"] = template_kwargs
        return "계산식: 12000 * (1 - 15%)", 11, 6

    server, client = serve(generate)
    try:
        assert client.models() == ["stub"]
        reply = client.chat("stub", [{"role": "user", "content": "q"}], template_kwargs={"enable_thinking": False})
        assert reply.text.startswith("계산식") and reply.completion_tokens == 6
        assert seen["kwargs"] == {"enable_thinking": False}
        result = CalcClient(client, "stub").ask([{"role": "user", "content": "정가 12,000원, 15% 할인가는?"}])
        assert result.answer == "10200" and result.formula == "12000 * (1 - 15%)"
    finally:
        server.shutdown()
