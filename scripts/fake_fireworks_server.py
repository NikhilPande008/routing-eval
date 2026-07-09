#!/usr/bin/env python3
"""Minimal OpenAI-compatible /chat/completions stub server (stdlib only).

Stands in for Fireworks so scripts/conformance_smoke.sh can prove the
container's real HTTP + I/O contract end-to-end without a live API key. Not a
model -- echoes a canned answer derived from the prompt so the smoke test can
assert the request actually reached the server.

Run:  python3 scripts/fake_fireworks_server.py [port]   (default 8811)
"""
from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        if not self.path.endswith("/chat/completions"):
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        prompt = (body.get("messages") or [{}])[-1].get("content", "")
        model = body.get("model", "")
        resp = {
            "choices": [{"index": 0, "finish_reason": "stop",
                        "message": {"role": "assistant",
                                    "content": f"stub-answer:[{model}]{prompt[:60]}"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        payload = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt: str, *args) -> None:
        pass  # keep smoke-test output quiet


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8811
    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"fake fireworks server on :{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
