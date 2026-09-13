"""The proxy between a browser and the runtime, checked without either.

What goes wrong at a Function URL is what a test can hold still: a body that
arrives base64-encoded when the browser did not send it that way, a path
that reaches for a file outside the bundle, a key comparison that leaks how
wrong a guess was through how long it took, a boto error that would have
shown a family a stack trace. Every request here is a hand-built payload
format 2.0 event, and the runtime is a stub that records what it was asked.

No test here builds a boto3 client against AWS: the client factory is
replaced for the whole module, and forgetting to replace it is a failure.
"""

from __future__ import annotations

import ast
import base64
import io
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from botocore.exceptions import ClientError, EndpointConnectionError, ReadTimeoutError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web"))

import lambda_function as lf  # noqa: E402  (the path above is what makes it importable)

RUNTIME = "arn:aws:bedrock-agentcore:us-east-1:111122223333:runtime/minutes-AbCdEf1234"
KEY = "correct-horse-battery-staple"


class StubRuntime:
    """The bedrock-agentcore client the proxy sees: records the call, returns a body."""

    def __init__(self, body: bytes = b'{"status": "done"}', raises: Exception | None = None) -> None:
        self.body = body
        self.raises = raises
        self.calls: list[dict[str, Any]] = []

    def invoke_agent_runtime(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self.raises:
            raise self.raises
        return {"response": io.BytesIO(self.body), "statusCode": 200}


@pytest.fixture
def site(tmp_path: Path, monkeypatch) -> Path:
    """A bundle like the one the deployer zips: an index, an asset, a font."""
    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text("<!doctype html><title>Minutes</title>", encoding="utf-8")
    (tmp_path / "statement.html").write_text("<title>statement</title>", encoding="utf-8")
    (tmp_path / "assets" / "app.js").write_text("console.log('minutes')", encoding="utf-8")
    (tmp_path / "assets" / "styles.css").write_text("body{}", encoding="utf-8")
    (tmp_path / "assets" / "mark.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00")
    (tmp_path / "assets" / "notes.py").write_text("print('not served')", encoding="utf-8")
    (tmp_path / ".secret").write_text("nope", encoding="utf-8")
    monkeypatch.setattr(lf, "SITE_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def runtime(monkeypatch) -> StubRuntime:
    stub = StubRuntime()
    monkeypatch.setattr(lf, "_clients", {})
    monkeypatch.setattr(lf.boto3, "client", lambda *a, **k: stub)
    return stub


@pytest.fixture
def configured(monkeypatch, runtime) -> StubRuntime:
    monkeypatch.setenv("MINUTES_DEMO_KEY", KEY)
    monkeypatch.setenv("MINUTES_RUNTIME_ARN", RUNTIME)
    monkeypatch.setenv("MINUTES_REGION", "us-east-1")
    return runtime


@pytest.fixture(autouse=True)
def no_real_client(monkeypatch) -> None:
    """Unless a test installs the stub, building a client is the failure."""
    monkeypatch.setattr(lf, "_clients", {})
    monkeypatch.setattr(
        lf.boto3, "client", lambda *a, **k: pytest.fail(f"built a boto3 client: {a} {k}")
    )


def event(
    method: str,
    path: str,
    *,
    body: Any = None,
    headers: dict[str, str] | None = None,
    base64_body: bool = False,
) -> dict[str, Any]:
    """A Function URL payload format 2.0 event, as the service builds them."""
    raw: str | None
    if body is None:
        raw = None
    elif isinstance(body, (dict, list)):
        raw = json.dumps(body)
    else:
        raw = body
    if raw is not None and base64_body:
        raw = base64.b64encode(raw.encode("utf-8")).decode("ascii")
    return {
        "version": "2.0",
        "routeKey": "$default",
        "rawPath": path,
        "rawQueryString": "",
        "headers": {"host": "abc.lambda-url.us-east-1.on.aws", **(headers or {})},
        "requestContext": {"http": {"method": method, "path": path, "protocol": "HTTP/1.1"}},
        "body": raw,
        "isBase64Encoded": base64_body,
    }


def post(payload: Any, key: str | None = KEY, **kwargs: Any) -> dict[str, Any]:
    headers = {"content-type": "application/json"}
    if key is not None:
        headers["x-minutes-key"] = key
    return event("POST", "/api", body=payload, headers=headers, **kwargs)


def body_of(response: dict[str, Any]) -> Any:
    return json.loads(response["body"])


# --- the key -------------------------------------------------------------


def test_a_request_without_the_key_is_refused(configured):
    response = lf.lambda_handler(post({"action": "wake"}, key=None))

    assert response["statusCode"] == 401
    assert body_of(response)["status"] == "error"
    assert configured.calls == []


def test_a_request_with_the_wrong_key_is_refused(configured):
    response = lf.lambda_handler(post({"action": "wake"}, key="wrong"))

    assert response["statusCode"] == 401
    assert configured.calls == []


def test_a_request_with_the_right_key_reaches_the_runtime(configured):
    response = lf.lambda_handler(post({"action": "wake", "case_id": "maya-demo"}))

    assert response["statusCode"] == 200
    assert len(configured.calls) == 1


def test_the_header_name_is_read_case_insensitively(configured):
    headers = {"X-Minutes-Key": KEY, "Content-Type": "application/json"}

    response = lf.lambda_handler(event("POST", "/api", body={"action": "wake"}, headers=headers))

    assert response["statusCode"] == 200


def test_the_key_is_compared_in_constant_time(monkeypatch):
    seen: list[tuple[bytes, bytes]] = []

    def compare(a: bytes, b: bytes) -> bool:
        seen.append((a, b))
        return a == b

    monkeypatch.setattr(lf.hmac, "compare_digest", compare)

    assert lf.key_is_valid({"x-minutes-key": KEY}, KEY) is True
    assert lf.key_is_valid({"x-minutes-key": "wrong"}, KEY) is False
    assert lf.key_is_valid({}, KEY) is False
    # Every check went through the constant-time comparison, wrong or missing
    # included: a plain == that stops at the first differing byte is exactly
    # what would tell a guesser how many bytes were right.
    assert seen == [(KEY.encode(), KEY.encode()), (b"wrong", KEY.encode()), (b"", KEY.encode())]


def test_without_a_configured_key_the_door_is_closed_not_open(configured, monkeypatch):
    monkeypatch.delenv("MINUTES_DEMO_KEY")

    response = lf.lambda_handler(post({"action": "wake"}, key=""))

    assert response["statusCode"] == 503
    assert configured.calls == []


# --- the forwarding ------------------------------------------------------


def test_the_payload_is_forwarded_with_a_stable_padded_session_id(configured):
    payload = {"action": "answer", "case_id": "maya-demo", "answers": {"i-1": "approve"}}

    response = lf.lambda_handler(post(payload))

    call = configured.calls[0]
    assert call["agentRuntimeArn"] == RUNTIME
    assert call["contentType"] == "application/json"
    assert call["accept"] == "application/json"
    assert json.loads(call["payload"]) == payload
    assert call["runtimeSessionId"] == "web-maya-demo-0000000000000000000"
    assert len(call["runtimeSessionId"]) >= 33
    assert response["headers"]["x-minutes-session"] == call["runtimeSessionId"]


def test_the_same_case_lands_on_the_same_session_every_time():
    # An approval answered on Thursday should find the session the wake left
    # on Monday when it is still warm; that is what a stable id buys.
    assert lf.session_id_for("maya-demo") == lf.session_id_for("maya-demo")
    assert lf.session_id_for("maya-demo") != lf.session_id_for("rivera-2027")
    # A visitor token gives the sample its own runtime session -- stable for
    # that visitor, distinct from the shared one and from other visitors -- so
    # a rebuilt runtime is not hidden behind a warm microVM the first visitor
    # happened to land on. No token: unchanged.
    assert lf.session_id_for("maya-demo", "judgeaaaa1111") == lf.session_id_for("maya-demo", "judgeaaaa1111")
    assert lf.session_id_for("maya-demo", "judgeaaaa1111") != lf.session_id_for("maya-demo")
    assert lf.session_id_for("maya-demo", "judgeaaaa1111") != lf.session_id_for("maya-demo", "judgebbbb2222")
    assert lf.session_id_for("maya-demo", "") == lf.session_id_for("maya-demo")
    assert lf.session_id_for("maya-demo", "../x") == lf.session_id_for("maya-demo", "x"), "only safe characters ride in the id"
    assert 33 <= len(lf.session_id_for("maya-demo", "v" * 40)) <= 100


@pytest.mark.parametrize("case_id", ["m", "maya demo", "a/b:c", "x" * 200, 42])
def test_any_case_id_makes_a_legal_session_id(case_id):
    session = lf.session_id_for(case_id)

    assert lf.SESSION_ID_PATTERN.fullmatch(session), session
    assert session.startswith("web-")


def test_a_payload_without_a_case_id_gets_a_fresh_legal_session():
    first, second = lf.session_id_for(None), lf.session_id_for("")

    assert first != second
    assert lf.SESSION_ID_PATTERN.fullmatch(first) and lf.SESSION_ID_PATTERN.fullmatch(second)


def test_the_runtimes_reply_comes_back_verbatim_as_json(configured):
    configured.body = b'{"status": "awaiting_approval", "interrupts": ["i-1"], "letter": "..."}'

    response = lf.lambda_handler(post({"action": "wake", "case_id": "maya-demo"}))

    assert response["statusCode"] == 200
    assert response["headers"]["content-type"] == "application/json"
    assert response["body"] == configured.body.decode("utf-8")
    assert response["isBase64Encoded"] is False


def test_the_runtime_is_reached_in_the_configured_region(configured, monkeypatch):
    regions: list[str | None] = []
    monkeypatch.setattr(lf, "_clients", {})
    configs: list[Any] = []

    def client(service: str, region_name: str | None = None, config: Any = None) -> StubRuntime:
        regions.append(region_name)
        configs.append(config)
        return configured

    monkeypatch.setattr(lf.boto3, "client", client)
    monkeypatch.setenv("MINUTES_REGION", "eu-west-1")

    lf.lambda_handler(post({"action": "wake"}))
    lf.lambda_handler(post({"action": "wake"}))

    assert regions == ["eu-west-1"]  # one client, reused across requests
    assert configs == [lf.CLIENT_CONFIG]


def test_the_runtime_is_never_retried_and_answers_before_the_function_is_cut_off():
    # A wake writes to the case; a retry the SDK made on its own would be a
    # second wake. And the read timeout has to lose the race with the
    # function's own 60 s, or the browser sees the Function URL's 502
    # instead of this proxy's JSON.
    assert lf.CLIENT_CONFIG.retries == {"total_max_attempts": 1}
    assert lf.CLIENT_CONFIG.read_timeout < 60
    assert lf.CLIENT_CONFIG.connect_timeout <= 10


# --- the body ------------------------------------------------------------


def test_a_base64_encoded_body_is_decoded_before_it_is_forwarded(configured):
    payload = {"action": "wake", "case_id": "maya-demo", "today": "2026-12-01"}

    response = lf.lambda_handler(post(payload, base64_body=True))

    assert response["statusCode"] == 200
    assert json.loads(configured.calls[0]["payload"]) == payload


def test_a_body_that_claims_base64_and_is_not_is_a_400_not_a_crash(configured):
    bad = post({"action": "wake"})
    bad["body"] = "{not base64"
    bad["isBase64Encoded"] = True

    response = lf.lambda_handler(bad)

    assert response["statusCode"] == 400
    assert configured.calls == []


@pytest.mark.parametrize("raw", ["not json", "[1, 2]", '"a string"', "", "null"])
def test_a_body_that_is_not_a_json_object_is_refused(configured, raw):
    response = lf.lambda_handler(post(raw))

    assert response["statusCode"] == 400
    assert body_of(response)["status"] == "error"
    assert configured.calls == []


def test_a_body_over_the_cap_never_reaches_the_runtime(configured):
    huge = {"action": "wake", "note": "x" * (lf.MAX_BODY_BYTES + 1)}

    response = lf.lambda_handler(post(huge))

    assert response["statusCode"] == 413
    assert configured.calls == []


def test_a_body_at_the_cap_still_goes_through(configured):
    prefix = '{"action": "wake", "note": "'
    filler = "x" * (lf.MAX_BODY_BYTES - len(prefix) - 2)

    response = lf.lambda_handler(post(prefix + filler + '"}'))

    assert response["statusCode"] == 200


# --- the errors ----------------------------------------------------------


def _client_error(code: str, message: str = "upstream said no", status: int = 400) -> ClientError:
    return ClientError(
        {"Error": {"Code": code, "Message": message}, "ResponseMetadata": {"HTTPStatusCode": status}},
        "InvokeAgentRuntime",
    )


@pytest.mark.parametrize(
    "code, expected",
    [
        ("ThrottlingException", 429),
        ("ValidationException", 400),
        ("AccessDeniedException", 502),
        ("ResourceNotFoundException", 502),
        ("RuntimeClientError", 502),
        ("ServiceUnavailableException", 503),
        ("SomethingNobodyExpected", 502),
    ],
)
def test_a_boto_error_becomes_json_with_a_fitting_status(configured, code, expected):
    configured.raises = _client_error(code)

    response = lf.lambda_handler(post({"action": "wake"}))

    assert response["statusCode"] == expected
    assert response["headers"]["content-type"].startswith("application/json")
    error = body_of(response)
    assert error["status"] == "error"
    assert code in error["error"]
    assert "Traceback" not in response["body"]


def test_a_timeout_is_a_504_and_a_dead_endpoint_a_502(configured):
    configured.raises = ReadTimeoutError(endpoint_url="https://x")
    assert lf.lambda_handler(post({"action": "wake"}))["statusCode"] == 504

    configured.raises = EndpointConnectionError(endpoint_url="https://x")
    assert lf.lambda_handler(post({"action": "wake"}))["statusCode"] == 502


def test_an_unexpected_exception_is_reported_by_kind_without_its_message(configured):
    configured.raises = RuntimeError("secret: the bucket name and a path")

    response = lf.lambda_handler(post({"action": "wake"}))

    assert response["statusCode"] == 502
    assert "RuntimeError" in body_of(response)["error"]
    assert "bucket" not in response["body"]


def test_without_a_runtime_arn_the_proxy_says_so_before_touching_boto(configured, monkeypatch):
    monkeypatch.delenv("MINUTES_RUNTIME_ARN")

    response = lf.lambda_handler(post({"action": "wake"}))

    assert response["statusCode"] == 503
    assert configured.calls == []


def test_only_post_reaches_the_api(configured, site):
    assert lf.lambda_handler(event("GET", "/api"))["statusCode"] == 405
    assert lf.lambda_handler(event("PUT", "/api", body="{}"))["statusCode"] == 405
    assert lf.lambda_handler(event("GET", "/api/anything"))["statusCode"] == 404
    assert configured.calls == []


# --- the screen ----------------------------------------------------------


def test_the_root_serves_the_index(site):
    response = lf.lambda_handler(event("GET", "/"))

    assert response["statusCode"] == 200
    assert response["headers"]["content-type"] == "text/html; charset=utf-8"
    assert response["headers"]["cache-control"] == "no-cache"
    assert response["body"] == "<!doctype html><title>Minutes</title>"
    assert response["isBase64Encoded"] is False


def test_an_unknown_path_serves_the_index_for_the_router(site):
    # A deep link is a hash route the browser resolves; from the server's
    # side it is just "/", and it has to come back 200 so the app can open.
    for path in ["/case/maya-demo", "/letters", "/anything/at/all", "/no-such.html"]:
        response = lf.lambda_handler(event("GET", path))
        assert response["statusCode"] == 200, path
        assert response["body"] == "<!doctype html><title>Minutes</title>"


def test_assets_are_served_with_their_content_type(site):
    js = lf.lambda_handler(event("GET", "/assets/app.js"))
    css = lf.lambda_handler(event("GET", "/assets/styles.css"))

    assert js["statusCode"] == 200
    assert js["headers"]["content-type"] == "text/javascript; charset=utf-8"
    assert js["headers"]["cache-control"] == "no-cache"
    assert js["body"] == "console.log('minutes')"
    assert css["headers"]["content-type"] == "text/css; charset=utf-8"


def test_a_binary_asset_is_base64_encoded_for_the_function_url(site):
    response = lf.lambda_handler(event("GET", "/assets/mark.png"))

    assert response["statusCode"] == 200
    assert response["headers"]["content-type"] == "image/png"
    assert response["isBase64Encoded"] is True
    assert base64.b64decode(response["body"]) == b"\x89PNG\r\n\x1a\n\x00"


def test_a_top_level_bundled_file_is_served_by_name(site):
    response = lf.lambda_handler(event("GET", "/statement.html"))

    assert response["statusCode"] == 200
    assert response["body"] == "<title>statement</title>"


def test_a_missing_asset_is_a_404_not_the_index(site):
    # A script tag handed a page of HTML fails in a way that blames the app.
    response = lf.lambda_handler(event("GET", "/assets/missing.js"))

    assert response["statusCode"] == 404
    assert body_of(response)["status"] == "error"


@pytest.mark.parametrize(
    "path",
    [
        "/assets/../lambda_function.py",
        "/assets/../../etc/passwd",
        "/../index.html",
        "/assets/%2e%2e/lambda_function.py",
        "/assets/..%2flambda_function.py",
        "/assets/./app.js",
        "/assets//app.js",
        "/assets/..\\app.js",
        "/.secret",
        "/assets/notes.py",
    ],
)
def test_nothing_outside_the_bundle_or_off_the_allowlist_is_ever_served(site, path):
    response = lf.lambda_handler(event("GET", path))

    assert response["statusCode"] in {200, 404}
    # Either it is refused outright, or it is the index for the router; it
    # is never a file the path was reaching for.
    body = response["body"]
    assert "def lambda_handler" not in body
    assert "not served" not in body
    assert "nope" not in body
    assert body == "<!doctype html><title>Minutes</title>" or response["statusCode"] == 404
    # And the path a stranger sent is not echoed back to them either.
    assert path not in body


def test_resolve_asset_stays_inside_the_bundle(site):
    assert lf.resolve_asset("/assets/app.js") == site / "assets" / "app.js"
    assert lf.resolve_asset("/assets/../index.html") is None
    assert lf.resolve_asset("/index.html/../index.html") is None
    assert lf.resolve_asset("/assets") is None  # a directory, not a file
    assert lf.resolve_asset("assets/app.js") is None


def test_a_bundle_without_an_index_yet_answers_404_not_500(site):
    (site / "index.html").unlink()

    response = lf.lambda_handler(event("GET", "/"))

    assert response["statusCode"] == 404
    assert "index.html" in body_of(response)["error"]


def test_head_is_answered_like_get_without_a_body(site):
    response = lf.lambda_handler(event("HEAD", "/"))

    assert response["statusCode"] == 200
    assert response["body"] == ""


def test_a_preflight_gets_an_empty_204_and_no_hand_made_cors(site):
    response = lf.lambda_handler(event("OPTIONS", "/api"))

    assert response["statusCode"] == 204
    assert not any(name.lower().startswith("access-control") for name in response["headers"])


def test_the_older_payload_shape_is_understood_too(site):
    response = lf.lambda_handler({"httpMethod": "GET", "path": "/assets/app.js"})

    assert response["statusCode"] == 200
    assert response["body"] == "console.log('minutes')"


# --- the file itself -----------------------------------------------------


def test_the_function_imports_nothing_the_lambda_runtime_does_not_ship():
    source = Path(lf.__file__).read_text(encoding="utf-8")
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    allowed = set(sys.stdlib_module_names) | {"boto3", "botocore"}
    assert imported <= allowed, imported - allowed


def test_no_account_id_or_arn_is_written_into_the_function():
    source = Path(lf.__file__).read_text(encoding="utf-8")

    assert "arn:aws:" not in source
    assert not any(token.isdigit() and len(token) == 12 for token in source.replace(":", " ").split())
