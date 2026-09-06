"""The web deployment, checked without an AWS account anywhere near it.

Every document this script sends is built by a pure function, so the things
that go wrong in a Lambda deploy can be asserted here: a role that could
invoke more than one runtime, a log grant that spills onto other functions'
logs, a zip that forgot the screen, a second run that quietly replaced the
demo key a parent already has in their bookmarks. The AWS calls themselves
are exercised against a stub client that records what it was asked to do.
No test here builds a boto3 client; ``--dry-run`` is tested by making that a
failure.
"""

from __future__ import annotations

import io
import json
import re
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import deploy_web as web  # noqa: E402  (the path above is what makes it importable)

ACCOUNT = "111122223333"
RUNTIME = f"arn:aws:bedrock-agentcore:us-east-1:{ACCOUNT}:runtime/minutes-AbCdEf1234"


class ResourceNotFoundException(Exception):
    pass


class ResourceConflictException(Exception):
    pass


class InvalidParameterValueException(Exception):
    pass


class EntityAlreadyExistsException(Exception):
    pass


class NoSuchEntityException(Exception):
    pass


class StubClient:
    """A boto3 client that records calls and raises or returns whatever a test queued.

    ``raises`` and ``returns`` each map a method name to a list consumed one
    call at a time, so a test can say "not found, then found" — which is
    what create-then-converge looks like from here. A single value in
    ``returns`` is handed back on every call.
    """

    exceptions = SimpleNamespace(
        ResourceNotFoundException=ResourceNotFoundException,
        ResourceConflictException=ResourceConflictException,
        InvalidParameterValueException=InvalidParameterValueException,
        EntityAlreadyExistsException=EntityAlreadyExistsException,
        NoSuchEntityException=NoSuchEntityException,
    )

    def __init__(
        self,
        raises: dict[str, list[Exception | None]] | None = None,
        returns: dict[str, Any] | None = None,
    ) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._raises = {name: list(queue) for name, queue in (raises or {}).items()}
        self._returns = dict(returns or {})

    def __getattr__(self, name: str) -> Any:
        def call(**kwargs: Any) -> Any:
            self.calls.append((name, kwargs))
            queue = self._raises.get(name)
            if queue:
                error = queue.pop(0)
                if error is not None:
                    raise error
            value = self._returns.get(name, {})
            if isinstance(value, list):
                return value.pop(0) if len(value) > 1 else value[0]
            return value

        return call

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.calls]

    def kwargs(self, name: str) -> dict[str, Any]:
        for called, kwargs in self.calls:
            if called == name:
                return kwargs
        raise AssertionError(f"{name} was never called; calls were {self.names}")

    def all_kwargs(self, name: str) -> list[dict[str, Any]]:
        return [kwargs for called, kwargs in self.calls if called == name]


@pytest.fixture
def site(tmp_path: Path) -> Path:
    root = tmp_path / "site"
    (root / "assets").mkdir(parents=True)
    (root / "index.html").write_text("<title>Minutes</title>", encoding="utf-8")
    (root / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")
    (root / "assets" / "mark.png").write_bytes(b"\x89PNG")
    (root / ".DS_Store").write_bytes(b"junk")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "x.pyc").write_bytes(b"junk")
    return root


@pytest.fixture
def plan(site: Path) -> web.Plan:
    return web.build_plan(runtime=RUNTIME, site_dir=site)


@pytest.fixture
def bundle(plan: web.Plan) -> bytes:
    return web.build_bundle(plan.site_dir, plan.lambda_source)


@pytest.fixture
def no_boto3(monkeypatch) -> None:
    """Any boto3 client built from here on is a test failure."""
    monkeypatch.setattr(
        web.boto3, "client", lambda *a, **k: pytest.fail(f"built a boto3 client: {a} {k}")
    )


def _existing(key: str | None = "old-key") -> dict[str, Any]:
    variables = {"MINUTES_RUNTIME_ARN": RUNTIME, "MINUTES_REGION": "us-east-1"}
    if key:
        variables["MINUTES_DEMO_KEY"] = key
    return {
        "Configuration": {
            "FunctionArn": f"arn:aws:lambda:us-east-1:{ACCOUNT}:function:minutes-web",
            "State": "Active",
            "LastUpdateStatus": "Successful",
            "LastModified": "2026-09-06T10:00:00.000+0000",
            "CodeSize": 12345,
            "Runtime": "python3.12",
            "MemorySize": 512,
            "Timeout": 60,
            "Environment": {"Variables": variables},
        }
    }


# --- the documents -------------------------------------------------------


def test_only_lambda_may_assume_the_role_and_only_for_this_function(plan):
    statement = plan.trust_policy["Statement"][0]

    assert statement["Principal"] == {"Service": "lambda.amazonaws.com"}
    assert statement["Action"] == "sts:AssumeRole"
    assert statement["Condition"] == {
        "StringEquals": {"aws:SourceAccount": ACCOUNT},
        "ArnLike": {"aws:SourceArn": f"arn:aws:lambda:us-east-1:{ACCOUNT}:function:minutes-web"},
    }


def test_the_role_may_invoke_this_runtime_and_write_its_own_logs_and_nothing_else(plan):
    log_group = f"arn:aws:logs:us-east-1:{ACCOUNT}:log-group:/aws/lambda/minutes-web"

    assert plan.permission_policy == {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "InvokeTheMinutesRuntime",
                "Effect": "Allow",
                "Action": "bedrock-agentcore:InvokeAgentRuntime",
                "Resource": [RUNTIME, f"{RUNTIME}/*"],
            },
            {
                "Sid": "WriteThisFunctionsLogs",
                "Effect": "Allow",
                "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
                "Resource": [log_group, f"{log_group}:*"],
            },
        ],
    }


def test_no_statement_grants_a_wildcard_resource_or_action(plan):
    for statement in plan.permission_policy["Statement"]:
        actions = statement["Action"] if isinstance(statement["Action"], list) else [statement["Action"]]
        assert "*" not in statement["Resource"]
        assert all(":" in action and not action.endswith("*") for action in actions)


def test_the_function_is_python312_with_the_promised_size_and_patience(plan):
    config = plan.function_config

    assert config["FunctionName"] == "minutes-web"
    assert config["Runtime"] == "python3.12"
    assert config["Handler"] == "lambda_function.lambda_handler"
    assert config["MemorySize"] == 512
    assert config["Timeout"] == 60
    assert config["Role"] == f"arn:aws:iam::{ACCOUNT}:role/minutes-web-lambda"


def test_the_environment_names_the_runtime_its_region_and_the_key(plan):
    env = web.environment(plan, "the-key")

    assert env == {
        "Variables": {
            "MINUTES_RUNTIME_ARN": RUNTIME,
            "MINUTES_REGION": "us-east-1",
            "MINUTES_DEMO_KEY": "the-key",
        }
    }


def test_the_url_is_public_buffered_and_allows_exactly_the_two_headers_the_proxy_reads(plan):
    assert plan.url_config["AuthType"] == "NONE"
    assert plan.url_config["InvokeMode"] == "BUFFERED"
    assert plan.url_config["Cors"]["AllowOrigins"] == ["*"]
    assert set(plan.url_config["Cors"]["AllowMethods"]) == {"GET", "POST", "OPTIONS"}
    assert set(plan.url_config["Cors"]["AllowHeaders"]) == {"content-type", "x-minutes-key"}


def test_the_public_permission_is_scoped_to_the_url_with_auth_none(plan):
    assert plan.url_permission == {
        "FunctionName": "minutes-web",
        "StatementId": "AllowPublicFunctionUrl",
        "Action": "lambda:InvokeFunctionUrl",
        "Principal": "*",
        "FunctionUrlAuthType": "NONE",
    }


def test_the_key_is_unguessable_and_url_safe():
    first, second = web.generate_key(), web.generate_key()

    assert first != second
    assert len(first) >= 32
    assert re.fullmatch(r"[A-Za-z0-9_-]+", first)


# --- what is resolved rather than written down ---------------------------


def test_the_account_region_and_partition_come_out_of_the_runtime_arn(site):
    other = "arn:aws-us-gov:bedrock-agentcore:us-gov-west-1:999988887777:runtime/minutes-Z"

    plan = web.build_plan(runtime=other, site_dir=site)

    assert plan.account == "999988887777"
    assert plan.region == "us-gov-west-1"
    assert plan.role_arn == "arn:aws-us-gov:iam::999988887777:role/minutes-web-lambda"
    assert plan.function_arn == "arn:aws-us-gov:lambda:us-gov-west-1:999988887777:function:minutes-web"
    assert plan.log_group_arn.startswith("arn:aws-us-gov:logs:us-gov-west-1:999988887777:log-group:")


def test_moving_the_function_keeps_the_runtimes_region_in_its_environment(site):
    plan = web.build_plan(runtime=RUNTIME, site_dir=site, region="eu-west-1")

    assert plan.region == "eu-west-1"
    assert plan.runtime_region == "us-east-1"
    assert web.environment(plan, "k")["Variables"]["MINUTES_REGION"] == "us-east-1"
    assert ":eu-west-1:" in plan.function_arn


def test_no_account_id_is_written_into_the_script():
    source = Path(web.__file__).read_text(encoding="utf-8")

    assert set(re.findall(r"\b\d{12}\b", source)) <= {"000000000000"}


def test_the_defaults_are_the_ones_the_docstring_promises():
    args = web.build_parser().parse_args([])

    assert (args.create, args.dry_run, args.delete, args.show) == (False, False, False, False)
    assert args.rotate_key is False
    assert args.name == "minutes-web"
    assert args.arn is None
    assert args.region is None
    assert args.role_name is None
    assert args.site is None


# --- the bundle ----------------------------------------------------------


def test_the_bundle_holds_the_function_at_the_root_and_the_screen_under_site(bundle):
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        names = archive.namelist()
        assert names[0] == "lambda_function.py"
        assert set(names) == {
            "lambda_function.py",
            "site/index.html",
            "site/assets/app.js",
            "site/assets/mark.png",
        }
        assert archive.read("site/index.html") == b"<title>Minutes</title>"
        assert archive.read("site/assets/mark.png") == b"\x89PNG"
        assert b"def lambda_handler" in archive.read("lambda_function.py")


def test_the_bundle_leaves_dotfiles_and_caches_out(bundle):
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        assert not any(".DS_Store" in name or "__pycache__" in name for name in archive.namelist())


def test_the_bundle_is_the_same_bytes_for_the_same_files(plan):
    assert web.build_bundle(plan.site_dir, plan.lambda_source) == web.build_bundle(
        plan.site_dir, plan.lambda_source
    )


def test_the_bundle_reads_the_screen_as_it_is_now_not_as_it_was(plan):
    before = web.build_bundle(plan.site_dir, plan.lambda_source)
    (plan.site_dir / "index.html").write_text("<title>edited</title>", encoding="utf-8")

    after = web.build_bundle(plan.site_dir, plan.lambda_source)

    assert before != after
    with zipfile.ZipFile(io.BytesIO(after)) as archive:
        assert archive.read("site/index.html") == b"<title>edited</title>"


def test_the_bundle_refuses_to_build_without_the_function(tmp_path, site):
    with pytest.raises(FileNotFoundError):
        web.build_bundle(site, tmp_path / "no-such-file.py")


def test_the_real_site_directory_bundles_too():
    files = web.bundle_files(web.SITE_DIR, web.LAMBDA_SOURCE)

    assert files[0][0] == "lambda_function.py"
    assert all(name.startswith("site/") for name, _ in files[1:])


# --- the AWS calls, against a stub ---------------------------------------


def test_dry_run_builds_no_client_and_prints_every_document_and_the_file_list(no_boto3, site, capsys):
    assert web.main(["--dry-run", "--arn", RUNTIME, "--site", str(site)]) == 0

    out = capsys.readouterr().out
    assert "lambda.amazonaws.com" in out
    assert "bedrock-agentcore:InvokeAgentRuntime" in out
    assert "logs:PutLogEvents" in out
    assert '"AuthType": "NONE"' in out
    assert "lambda:InvokeFunctionUrl" in out
    assert "site/index.html" in out
    assert "site/assets/app.js" in out
    assert "lambda_function.py" in out
    assert web.KEY_PLACEHOLDER in out


def test_dry_run_works_on_a_clone_that_has_never_deployed(no_boto3, monkeypatch, capsys):
    monkeypatch.setattr(web, "STATE", Path("no-such-deploy-state.json"))

    assert web.main(["--dry-run"]) == 0

    assert web.EXAMPLE_ARN in capsys.readouterr().out


def test_dry_run_never_prints_a_real_key(no_boto3, site, capsys, monkeypatch):
    monkeypatch.setattr(web, "generate_key", lambda: pytest.fail("generated a key under --dry-run"))

    web.main(["--dry-run", "--arn", RUNTIME, "--site", str(site)])

    assert "KEY:" not in capsys.readouterr().out


def test_without_a_deploy_and_without_an_arn_the_real_modes_stop(no_boto3, monkeypatch):
    monkeypatch.setattr(web, "STATE", Path("no-such-deploy-state.json"))

    with pytest.raises(SystemExit):
        web.main(["--show"])


def test_create_makes_the_role_then_the_function_then_the_url_then_the_permission(plan, bundle):
    iam = StubClient()
    lam = StubClient(
        raises={"get_function": [ResourceNotFoundException()]},
        returns={"create_function_url_config": {"FunctionUrl": "https://abc.lambda-url.us-east-1.on.aws/"}},
    )
    lines: list[str] = []

    url = web.create(
        plan, iam=iam, lam=lam, bundle=bundle, new_key=lambda: "fresh-key",
        sleep=lambda _: None, log=lines.append,
    )

    assert iam.names == ["create_role", "put_role_policy"]
    assert lam.names == [
        "get_function",
        "create_function",
        "get_function_configuration",
        "create_function_url_config",
        "add_permission",
    ]
    assert json.loads(iam.kwargs("create_role")["AssumeRolePolicyDocument"]) == plan.trust_policy
    assert json.loads(iam.kwargs("put_role_policy")["PolicyDocument"]) == plan.permission_policy
    assert iam.kwargs("put_role_policy")["PolicyName"] == "MinutesWebProxy"
    created = lam.kwargs("create_function")
    assert created["Code"] == {"ZipFile": bundle}
    assert created["Runtime"] == "python3.12"
    assert created["Environment"]["Variables"]["MINUTES_DEMO_KEY"] == "fresh-key"
    assert created["Environment"]["Variables"]["MINUTES_RUNTIME_ARN"] == RUNTIME
    assert lam.kwargs("create_function_url_config") == plan.url_config
    assert lam.kwargs("add_permission") == plan.url_permission
    assert url == "https://abc.lambda-url.us-east-1.on.aws/"
    assert "URL:  https://abc.lambda-url.us-east-1.on.aws/" in lines
    assert any(line.startswith("KEY:  fresh-key") for line in lines)


def test_running_create_twice_converges_and_keeps_the_key(plan, bundle):
    iam = StubClient(raises={"create_role": [EntityAlreadyExistsException()]})
    lam = StubClient(
        raises={
            "create_function_url_config": [ResourceConflictException()],
            "add_permission": [ResourceConflictException()],
        },
        returns={
            "get_function": _existing("old-key"),
            "update_function_url_config": {"FunctionUrl": "https://abc.lambda-url.us-east-1.on.aws/"},
        },
    )
    lines: list[str] = []

    web.create(
        plan, iam=iam, lam=lam, bundle=bundle,
        new_key=lambda: pytest.fail("generated a new key on an update"),
        sleep=lambda _: None, log=lines.append,
    )

    assert iam.names == ["create_role", "update_assume_role_policy", "put_role_policy"]
    assert lam.names == [
        "get_function",
        "update_function_configuration",
        "get_function_configuration",
        "update_function_code",
        "get_function_configuration",
        "create_function_url_config",
        "update_function_url_config",
        "add_permission",
    ]
    assert lam.kwargs("update_function_configuration")["Environment"]["Variables"]["MINUTES_DEMO_KEY"] == "old-key"
    assert lam.kwargs("update_function_code")["ZipFile"] == bundle
    assert lam.kwargs("update_function_url_config") == plan.url_config
    assert any(line.startswith("KEY:  old-key") for line in lines)


def test_rotate_key_replaces_the_key_on_an_update(plan, bundle):
    lam = StubClient(returns={"get_function": _existing("old-key"), "update_function_url_config": {"FunctionUrl": "u"}})
    lam._raises["create_function_url_config"] = [ResourceConflictException()]

    web.create(
        plan, iam=StubClient(), lam=lam, bundle=bundle, rotate_key=True,
        new_key=lambda: "rotated-key", sleep=lambda _: None, log=lambda _: None,
    )

    env = lam.kwargs("update_function_configuration")["Environment"]["Variables"]
    assert env["MINUTES_DEMO_KEY"] == "rotated-key"


def test_a_deployed_function_with_no_key_gets_one(plan, bundle):
    lam = StubClient(returns={"get_function": _existing(key=None), "create_function_url_config": {"FunctionUrl": "u"}})

    web.create(
        plan, iam=StubClient(), lam=lam, bundle=bundle,
        new_key=lambda: "new-key", sleep=lambda _: None, log=lambda _: None,
    )

    env = lam.kwargs("update_function_configuration")["Environment"]["Variables"]
    assert env["MINUTES_DEMO_KEY"] == "new-key"


def test_create_waits_for_a_new_role_to_become_assumable(plan, bundle):
    lam = StubClient(
        raises={
            "get_function": [ResourceNotFoundException()],
            "create_function": [InvalidParameterValueException("The role defined for the function cannot be assumed")],
        },
        returns={"create_function_url_config": {"FunctionUrl": "u"}},
    )
    waits: list[float] = []

    web.create(plan, iam=StubClient(), lam=lam, bundle=bundle, sleep=waits.append, log=lambda _: None)

    assert lam.names.count("create_function") == 2
    assert waits == [web.ROLE_PROPAGATION_WAIT]


def test_create_gives_up_rather_than_waiting_forever(plan, bundle):
    lam = StubClient(
        raises={
            "get_function": [ResourceNotFoundException()],
            "create_function": [InvalidParameterValueException("nope")] * web.ROLE_PROPAGATION_TRIES,
        }
    )

    with pytest.raises(InvalidParameterValueException):
        web.create(plan, iam=StubClient(), lam=lam, bundle=bundle, sleep=lambda _: None, log=lambda _: None)

    assert lam.names.count("create_function") == web.ROLE_PROPAGATION_TRIES


def test_create_waits_for_the_function_to_settle_between_configuration_and_code(plan, bundle):
    lam = StubClient(
        returns={
            "get_function": _existing(),
            "get_function_configuration": [
                {"State": "Active", "LastUpdateStatus": "InProgress"},
                {"State": "Active", "LastUpdateStatus": "Successful"},
            ],
            "create_function_url_config": {"FunctionUrl": "u"},
        }
    )
    waits: list[float] = []

    web.create(plan, iam=StubClient(), lam=lam, bundle=bundle, sleep=waits.append, log=lambda _: None)

    # The first poll saw the configuration update still in flight and waited
    # once before the code update went out; the second settled at once.
    assert waits == [web.READY_WAIT]
    assert lam.names.index("update_function_code") > lam.names.index("update_function_configuration")


def test_a_function_that_fails_to_become_active_is_reported_not_polled_forever(plan):
    lam = StubClient(
        returns={"get_function_configuration": {"State": "Failed", "StateReason": "image not found"}}
    )

    with pytest.raises(RuntimeError, match="image not found"):
        web.wait_ready(lam, "minutes-web", sleep=lambda _: None, log=lambda _: None)


def test_the_key_is_printed_exactly_once_and_only_on_the_marked_line(plan, bundle):
    lam = StubClient(
        raises={"get_function": [ResourceNotFoundException()]},
        returns={"create_function_url_config": {"FunctionUrl": "u"}},
    )
    lines: list[str] = []

    web.create(
        plan, iam=StubClient(), lam=lam, bundle=bundle, new_key=lambda: "SECRET-KEY-VALUE",
        sleep=lambda _: None, log=lines.append,
    )

    holding = [line for line in lines if "SECRET-KEY-VALUE" in line]
    assert len(holding) == 1
    assert holding[0].startswith("KEY:  SECRET-KEY-VALUE")


def test_delete_takes_back_the_url_the_permission_the_function_and_the_role_in_that_order(plan):
    iam, lam = StubClient(), StubClient()

    web.delete(plan, iam=iam, lam=lam, log=lambda _: None)

    assert lam.names == ["delete_function_url_config", "remove_permission", "delete_function"]
    assert lam.kwargs("remove_permission") == {
        "FunctionName": "minutes-web",
        "StatementId": "AllowPublicFunctionUrl",
    }
    assert lam.kwargs("delete_function") == {"FunctionName": "minutes-web"}
    assert iam.names == ["delete_role_policy", "delete_role"]
    assert iam.kwargs("delete_role_policy")["PolicyName"] == "MinutesWebProxy"


def test_deleting_what_is_not_there_is_not_an_error(plan):
    lines: list[str] = []

    web.delete(
        plan,
        iam=StubClient(raises={"delete_role_policy": [NoSuchEntityException()]}),
        lam=StubClient(
            raises={
                "delete_function_url_config": [ResourceNotFoundException()],
                "remove_permission": [ResourceNotFoundException()],
                "delete_function": [ResourceNotFoundException()],
            }
        ),
        log=lines.append,
    )

    assert any("no function URL" in line for line in lines)
    assert any("no function named" in line for line in lines)
    assert any("no role named" in line for line in lines)


def test_show_reports_the_url_state_and_bundle_size_but_never_the_key(plan):
    lam = StubClient(
        returns={
            "get_function": _existing("SECRET-KEY-VALUE"),
            "get_function_url_config": {"FunctionUrl": "https://abc.lambda-url.us-east-1.on.aws/"},
        }
    )
    lines: list[str] = []

    web.show(plan, lam=lam, log=lines.append)

    printed = "\n".join(lines)
    assert "https://abc.lambda-url.us-east-1.on.aws/" in printed
    assert "Active / Successful" in printed
    assert "2026-09-06T10:00:00" in printed
    assert "12345 bytes" in printed
    assert "demo key:      set" in printed
    assert "SECRET-KEY-VALUE" not in printed


def test_show_says_so_when_nothing_is_deployed(plan):
    lines: list[str] = []

    web.show(plan, lam=StubClient(raises={"get_function": [ResourceNotFoundException()]}), log=lines.append)

    assert lines == ["nothing deployed: no function named minutes-web in us-east-1"]


def test_the_real_modes_refuse_to_bundle_a_missing_screen(monkeypatch, tmp_path):
    monkeypatch.setattr(web.boto3, "client", lambda *a, **k: StubClient())

    with pytest.raises(SystemExit, match="no screen to bundle"):
        web.main(["--arn", RUNTIME, "--site", str(tmp_path / "missing")])
