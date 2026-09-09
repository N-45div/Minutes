"""The one URL a parent opens: the app, and the door to the runtime behind it.

Minutes runs on Amazon Bedrock AgentCore Runtime, which accepts SigV4-signed
calls and nothing else. A browser has no AWS credentials to sign with, and it
must never be handed any, so the parent's screen cannot talk to the runtime
directly. This file is the thing in between: a single Lambda behind a Lambda
Function URL that serves the screen itself at ``/`` and forwards ``POST /api``
to the runtime under the Lambda's own role — a role allowed to invoke exactly
this runtime and to do nothing else.

One URL is the whole application. There is no bucket, no CDN, no second
origin: the HTML, the script and the API share one host, so the browser makes
same-origin calls and the CORS surface stays as small as the Function URL's
own configuration. Any path that is not an asset comes back as ``index.html``
so the screen's hash router resolves a deep link from a cold start.

What stands between a stranger and a family's case is a demo key sent in the
``x-minutes-key`` header and compared in constant time, plus a case id nobody
can guess. That is right for a demo of a synthetic case and wrong for real
children's records; the README says so in as many words.

Nothing here is imported by the rest of the project. It has to run on the
Lambda Python 3.12 runtime with the ``boto3`` that runtime ships and no other
dependency, which is why it is one file and why the deployer zips it with the
``site/`` directory rather than installing anything.
"""

from __future__ import annotations

import base64
import binascii
import hmac
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

# The runtime a browser may reach, where it is, and the key that opens the
# door. All three come from the function's environment, which the deployer
# sets; none is written down here.
ENV_RUNTIME_ARN = "MINUTES_RUNTIME_ARN"
ENV_REGION = "MINUTES_REGION"
ENV_DEMO_KEY = "MINUTES_DEMO_KEY"
DEFAULT_REGION = "us-east-1"

KEY_HEADER = "x-minutes-key"
API_PATH = "/api"
ASSETS_PREFIX = "/assets/"
INDEX = "index.html"

# A payload IS a document upload now: a parent sends the IEP as the PDF the
# district emailed them, and a service log as a photograph of the page.
#
# The arithmetic, from measurements rather than from the docs. Base64 adds a
# third, so the runtime's own 3.5 MB per-file limit is about 4.67 MB of body.
# A Lambda Function URL rejects the whole event above 6,291,456 bytes -- hard,
# not raisable, and its refusal is an opaque 413 the parent cannot act on. So
# the cap sits between the two: high enough that every file the runtime would
# accept gets through and is refused by name, low enough to leave more than a
# megabyte of headroom under the ceiling that answers with nothing.
MAX_BODY_BYTES = 5_000_000

# Everything the bundle may serve, by extension, and how to label it. A file
# with any other extension is not served even if it is in the zip.
CONTENT_TYPES: dict[str, str] = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".json": "application/json; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".woff2": "font/woff2",
}
BINARY_EXTENSIONS = frozenset({".png", ".ico", ".woff2"})
# The screen is redeployed in place at one URL, so the browser has to ask
# each time whether the html, script or styles changed. Images and fonts
# change by getting a new name.
NO_CACHE_EXTENSIONS = frozenset({".html", ".js", ".css"})
CACHE_CONTROL_STATIC = "public, max-age=3600"

# A runtimeSessionId must be at least 33 characters and look like an
# identifier; the same rule the scheduler's session ids are held to.
SESSION_ID_MIN = 33
SESSION_ID_MAX = 100
SESSION_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{32,99}$")
SESSION_PREFIX = "web-"

# How a failure on the far side of the proxy is reported to the browser. The
# runtime's own errors come back as data with status 200, so what is mapped
# here is only what the SDK raised before a reply existed: the browser's
# request was malformed, the account is throttled, or the proxy is pointed
# at something it cannot reach — which is nobody's fault in the browser.
UPSTREAM_STATUS: dict[str, int] = {
    "ValidationException": 400,
    "ThrottlingException": 429,
    "ServiceQuotaExceededException": 429,
    "AccessDeniedException": 502,
    "UnauthorizedException": 502,
    "ResourceNotFoundException": 502,
    "RuntimeClientError": 502,
    "InternalServerException": 502,
    "ServiceUnavailableException": 503,
}
DEFAULT_UPSTREAM_STATUS = 502

# An invocation is not idempotent — a wake writes to the case — so the SDK
# must not retry one on its own. The read timeout sits under the function's
# own 60 s so a slow runtime comes back as this proxy's JSON, not as the
# Function URL's bare 502 after the function was cut off.
CONNECT_TIMEOUT_SECONDS = 5
READ_TIMEOUT_SECONDS = 55
CLIENT_CONFIG = Config(
    connect_timeout=CONNECT_TIMEOUT_SECONDS,
    read_timeout=READ_TIMEOUT_SECONDS,
    retries={"total_max_attempts": 1},
)


def _find_site() -> Path:
    """Where the screen's files are: beside this file in the zip, one up in the repo."""
    here = Path(__file__).resolve().parent
    for candidate in (here / "site", here.parent / "site"):
        if candidate.is_dir():
            return candidate
    return here / "site"


SITE_DIR = _find_site()

_clients: dict[str, Any] = {}


def _runtime_client(region: str) -> Any:
    """One client per region for the life of the container: a cold start signs
    once, and every request after that reuses the connection."""
    client = _clients.get(region)
    if client is None:
        client = boto3.client("bedrock-agentcore", region_name=region, config=CLIENT_CONFIG)
        _clients[region] = client
    return client


# --- the request ---------------------------------------------------------


def method_of(event: dict[str, Any]) -> str:
    """The HTTP method, from payload format 2.0 or the older shape if that is what came."""
    http = (event.get("requestContext") or {}).get("http") or {}
    return str(http.get("method") or event.get("httpMethod") or "GET").upper()


def path_of(event: dict[str, Any]) -> str:
    """The request path, decoded, from payload format 2.0 or the older shape."""
    raw = event.get("rawPath") or event.get("path") or "/"
    return unquote(str(raw))


def headers_of(event: dict[str, Any]) -> dict[str, str]:
    """Headers with lower-cased names. Function URLs lower-case them already;
    this makes that a fact rather than a hope."""
    return {str(name).lower(): str(value) for name, value in (event.get("headers") or {}).items()}


def body_of(event: dict[str, Any]) -> bytes:
    """The request body as bytes, base64-decoded when the Function URL says so."""
    body = event.get("body")
    if body is None:
        return b""
    if isinstance(body, bytes):
        return body
    text = str(body)
    if event.get("isBase64Encoded"):
        try:
            return base64.b64decode(text, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("body is marked base64 and is not") from exc
    return text.encode("utf-8")


# --- the responses -------------------------------------------------------


def _response(
    status: int,
    body: bytes | str,
    content_type: str,
    *,
    cache_control: str | None = None,
    extra: dict[str, str] | None = None,
    binary: bool = False,
) -> dict[str, Any]:
    headers = {"content-type": content_type, "x-content-type-options": "nosniff"}
    if cache_control:
        headers["cache-control"] = cache_control
    if extra:
        headers.update(extra)
    if binary:
        raw = body if isinstance(body, bytes) else body.encode("utf-8")
        return {
            "statusCode": status,
            "headers": headers,
            "body": base64.b64encode(raw).decode("ascii"),
            "isBase64Encoded": True,
        }
    text = body.decode("utf-8") if isinstance(body, bytes) else body
    return {"statusCode": status, "headers": headers, "body": text, "isBase64Encoded": False}


def error(status: int, message: str) -> dict[str, Any]:
    """An error the browser can read: the same shape the runtime uses for its own."""
    return _response(
        status,
        json.dumps({"status": "error", "error": message}),
        CONTENT_TYPES[".json"],
        cache_control="no-store",
    )


# --- the key -------------------------------------------------------------


def key_is_valid(headers: dict[str, str], expected: str | None) -> bool:
    """Whether the request carries the demo key. Compared in constant time, so
    the time a wrong guess takes says nothing about how wrong it was."""
    if not expected:
        return False
    given = headers.get(KEY_HEADER, "")
    return hmac.compare_digest(given.encode("utf-8"), expected.encode("utf-8"))


# --- the session id ------------------------------------------------------


def session_id_for(case_id: Any) -> str:
    """A stable runtimeSessionId for a case, so an approval answered later
    lands on the same runtime session as the wake that raised it when that
    session is still warm. The case itself lives in S3 under its id, so a
    session that has gone cold loses nothing; this only spares the round
    trip. A payload with no case id gets a fresh session each time."""
    if case_id is None or str(case_id) == "":
        return f"{SESSION_PREFIX}{uuid.uuid4().hex}"
    stem = re.sub(r"[^A-Za-z0-9_-]+", "-", str(case_id)).strip("-")
    if not stem:
        return f"{SESSION_PREFIX}{uuid.uuid4().hex}"
    session = f"{SESSION_PREFIX}{stem}-".ljust(SESSION_ID_MIN, "0")[:SESSION_ID_MAX]
    if not SESSION_ID_PATTERN.fullmatch(session):
        raise ValueError(f"case_id {case_id!r} does not make a legal session id")
    return session


# --- the API -------------------------------------------------------------


def _upstream_error(exc: ClientError) -> dict[str, Any]:
    info = exc.response.get("Error", {}) if isinstance(exc.response, dict) else {}
    code = str(info.get("Code") or type(exc).__name__)
    message = str(info.get("Message") or "the runtime call failed")
    return error(UPSTREAM_STATUS.get(code, DEFAULT_UPSTREAM_STATUS), f"{code}: {message}")


def handle_api(event: dict[str, Any]) -> dict[str, Any]:
    """``POST /api``: check the key, forward the payload, hand back the reply."""
    if method_of(event) != "POST":
        return error(405, "POST /api is the only API route")

    expected = os.environ.get(ENV_DEMO_KEY)
    if not expected:
        return error(503, f"{ENV_DEMO_KEY} is not configured on this function")
    if not key_is_valid(headers_of(event), expected):
        return error(401, f"missing or wrong {KEY_HEADER}")

    runtime_arn = os.environ.get(ENV_RUNTIME_ARN)
    if not runtime_arn:
        return error(503, f"{ENV_RUNTIME_ARN} is not configured on this function")
    region = os.environ.get(ENV_REGION) or DEFAULT_REGION

    try:
        raw = body_of(event)
    except ValueError as exc:
        return error(400, str(exc))
    if len(raw) > MAX_BODY_BYTES:
        return error(413, f"that request is over {MAX_BODY_BYTES / 1_000_000:.0f} MB, which is more than Minutes accepts in one go")
    try:
        payload = json.loads(raw.decode("utf-8") or "null")
    except (UnicodeDecodeError, json.JSONDecodeError):
        return error(400, "body must be a JSON object")
    if not isinstance(payload, dict):
        return error(400, "body must be a JSON object")

    try:
        session = session_id_for(payload.get("case_id"))
    except ValueError as exc:
        return error(400, str(exc))

    try:
        reply = _runtime_client(region).invoke_agent_runtime(
            agentRuntimeArn=runtime_arn,
            runtimeSessionId=session,
            contentType="application/json",
            accept="application/json",
            payload=json.dumps(payload).encode("utf-8"),
        )
        body = reply["response"].read()
    except ClientError as exc:
        return _upstream_error(exc)
    except BotoCoreError as exc:
        # A connection or read timeout: the runtime never answered, so there
        # is no reply to relay and no traceback worth showing a browser.
        name = type(exc).__name__
        return error(504 if "Timeout" in name else 502, f"{name}: the runtime did not answer")
    except Exception as exc:  # the proxy boundary: report the kind, not the stack
        return error(502, f"{type(exc).__name__}: the runtime call failed")

    return _response(
        200,
        body,
        "application/json",
        cache_control="no-store",
        extra={"x-minutes-session": session},
    )


# --- the screen ----------------------------------------------------------


def resolve_asset(path: str) -> Path | None:
    """The bundled file a path names, or None when it names none.

    The path is walked segment by segment against the bundle and nothing
    else: a ``..`` anywhere, an empty or dotted segment, a backslash, or an
    extension this does not serve all resolve to None rather than to a file.
    The resolved path is then checked to be inside the bundle once more, in
    case the filesystem had an opinion the string check did not."""
    if not path.startswith("/"):
        return None
    segments = path[1:].split("/")
    if not segments or any(
        segment in {"", ".", ".."} or segment.startswith(".") or "\\" in segment or "\0" in segment
        for segment in segments
    ):
        return None
    if Path(segments[-1]).suffix.lower() not in CONTENT_TYPES:
        return None
    root = SITE_DIR.resolve()
    candidate = root.joinpath(*segments)
    try:
        resolved = candidate.resolve()
    except OSError:
        return None
    if resolved == root or not resolved.is_relative_to(root):
        return None
    if not resolved.is_file():
        return None
    return resolved


def serve_file(file: Path) -> dict[str, Any]:
    """One bundled file with its content type and the right cache rule."""
    suffix = file.suffix.lower()
    cache = "no-cache" if suffix in NO_CACHE_EXTENSIONS else CACHE_CONTROL_STATIC
    return _response(
        200,
        file.read_bytes(),
        CONTENT_TYPES[suffix],
        cache_control=cache,
        binary=suffix in BINARY_EXTENSIONS,
    )


def serve_index() -> dict[str, Any]:
    """The screen. Every path the router owns comes back as this, status 200,
    so a deep link shared from one parent to another opens where it points."""
    index = resolve_asset(f"/{INDEX}")
    if index is None:
        return error(404, f"{INDEX} is not in the bundle")
    return serve_file(index)


def handle_site(event: dict[str, Any]) -> dict[str, Any]:
    method = method_of(event)
    if method not in {"GET", "HEAD"}:
        return error(405, f"{method} is not served here")
    path = path_of(event)
    if path == API_PATH or path.startswith(API_PATH + "/"):
        return error(404, "no such API route")
    if path.startswith(ASSETS_PREFIX):
        # A missing asset is a missing asset. Answering it with the screen
        # would hand a script tag a page of HTML and a font tag a mystery.
        file = resolve_asset(path)
        return serve_file(file) if file else error(404, "no such asset")
    if path != "/":
        file = resolve_asset(path)
        if file:
            return serve_file(file)
    response = serve_index()
    if method == "HEAD":
        response["body"] = ""
    return response


# --- the entry point -----------------------------------------------------


def lambda_handler(event: dict[str, Any], context: Any = None) -> dict[str, Any]:
    """Route one Function URL request. The proxy never raises to the browser."""
    method = method_of(event)
    if method == "OPTIONS":
        # A preflight is answered by the Function URL from its own CORS
        # configuration and normally never reaches here. If one does, an
        # empty 204 lets that configuration be the only CORS policy there is.
        return {"statusCode": 204, "headers": {}, "body": "", "isBase64Encoded": False}
    path = path_of(event)
    if path == API_PATH:
        return handle_api(event)
    return handle_site(event)
