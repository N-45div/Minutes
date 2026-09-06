"""Serve the parent's screen locally, on one origin with the runtime.

The runtime's own server (``python app.py``, port 8080) sets no CORS headers,
so a page served from anywhere else cannot call it from a browser. This
server puts both on one origin: it serves ``site/`` as static files and
forwards ``POST /api`` to ``http://127.0.0.1:8080/invocations`` unchanged.

Run both, in two terminals, from the repository root::

    python app.py                       # the runtime, on 127.0.0.1:8080
    python scripts/dev_web.py           # this, on http://127.0.0.1:8000

then open http://127.0.0.1:8000/#/case/maya-demo. No key is needed locally
(the ``x-minutes-key`` header is passed through and ignored by the runtime).
``--port`` and ``--runtime`` change the defaults::

    python scripts/dev_web.py --port 8001 --runtime http://127.0.0.1:8080/invocations

Standard library only. Nothing here is part of the deployed product: the
deployed page talks to a proxy that adds the key check and the AgentCore
call; this file is the same-origin stand-in for it on a laptop.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import sys
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

SITE = Path(__file__).resolve().parent.parent / "site"
SESSION_HEADER = "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id"


def session_id_for(payload: dict) -> str:
    """A stable runtime session per case, so the local server's logs line up by case.

    The runtime keys the caseworker by ``case_id`` itself; this header only
    names the session in the SDK's own logging. AgentCore requires at least
    33 characters, so the id is padded with a digest of itself.
    """
    case = str(payload.get("case_id") or "maya-demo")
    return f"minutes-web-{case}-{hashlib.sha256(case.encode()).hexdigest()[:24]}"


class Handler(SimpleHTTPRequestHandler):
    runtime: tuple[str, int, str] = ("127.0.0.1", 8080, "/invocations")

    def do_POST(self) -> None:  # noqa: N802 - http.server's naming
        if urlsplit(self.path).path != "/api":
            self.send_error(HTTPStatus.NOT_FOUND, "only POST /api is forwarded")
            return
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(body or b"{}")
        except ValueError:
            self._json(HTTPStatus.BAD_REQUEST, {"status": "error", "error": "the request body is not JSON"})
            return

        host, port, path = self.runtime
        headers = {
            "Content-Type": "application/json",
            SESSION_HEADER: session_id_for(payload if isinstance(payload, dict) else {}),
        }
        try:
            conn = http.client.HTTPConnection(host, port, timeout=600)
            conn.request("POST", path, body=body, headers=headers)
            upstream = conn.getresponse()
            data = upstream.read()
            status = upstream.status
            content_type = upstream.getheader("Content-Type") or "application/json"
            conn.close()
        except OSError as exc:
            self._json(
                HTTPStatus.BAD_GATEWAY,
                {
                    "status": "error",
                    "error": f"the runtime at {host}:{port} did not answer ({exc}). "
                    "Start it with `python app.py` in another terminal.",
                },
            )
            return
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        if urlsplit(self.path).path == "/":
            self.path = "/index.html"
        super().do_GET()

    def end_headers(self) -> None:
        # A page under development must never be served stale.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _json(self, status: HTTPStatus, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8000, help="port to serve on (default 8000)")
    parser.add_argument("--host", default="127.0.0.1", help="interface to bind (default 127.0.0.1)")
    parser.add_argument(
        "--runtime",
        default="http://127.0.0.1:8080/invocations",
        help="the runtime's invocations URL (default http://127.0.0.1:8080/invocations)",
    )
    args = parser.parse_args()

    parts = urlsplit(args.runtime)
    Handler.runtime = (parts.hostname or "127.0.0.1", parts.port or 80, parts.path or "/invocations")
    handler = partial(Handler, directory=str(SITE))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving {SITE} on http://{args.host}:{args.port}/ and forwarding POST /api to {args.runtime}")
    print(f"Open http://{args.host}:{args.port}/#/case/maya-demo")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
