"""The weekly schedule, checked without an AWS account anywhere near it.

Every document this script sends is built by a pure function, so the things
that actually go wrong in EventBridge Scheduler can be asserted here: the
universal target ARN that is spelled differently from every other AgentCore
ARN, a session id that has to clear 33 characters after a substitution nobody
controls, and a policy that must name the runtime twice. Scheduler validates
none of that at create time — a wrong Input creates cleanly and then fails on
every invocation forever — so these assertions are the only place it is caught.

The AWS calls themselves are exercised against a stub client that records what
it was asked to do. No test here builds a boto3 client; ``--dry-run`` is tested
by making that a failure.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from zoneinfo import ZoneInfo

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import schedule_weekly as sched  # noqa: E402  (the path above is what makes it importable)

ACCOUNT = "111122223333"
RUNTIME = f"arn:aws:bedrock-agentcore:us-east-1:{ACCOUNT}:runtime/minutes-AbCdEf1234"


class ConflictException(Exception):
    pass


class ValidationException(Exception):
    pass


class ResourceNotFoundException(Exception):
    pass


class EntityAlreadyExistsException(Exception):
    pass


class NoSuchEntityException(Exception):
    pass


class StubClient:
    """A boto3 client that records calls and raises whatever a test queued.

    ``raises`` maps a method name to a list consumed one call at a time, so a
    test can say "fail the first create, then succeed" — which is exactly what
    idempotence and IAM propagation look like from here.
    """

    exceptions = SimpleNamespace(
        ConflictException=ConflictException,
        ValidationException=ValidationException,
        ResourceNotFoundException=ResourceNotFoundException,
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
            return self._returns.get(name, {})

        return call

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.calls]

    def kwargs(self, name: str) -> dict[str, Any]:
        for called, kwargs in self.calls:
            if called == name:
                return kwargs
        raise AssertionError(f"{name} was never called; calls were {self.names}")


@pytest.fixture
def plan() -> sched.Plan:
    return sched.build_plan(runtime=RUNTIME)


@pytest.fixture
def no_boto3(monkeypatch) -> None:
    """Any boto3 client built from here on is a test failure."""
    monkeypatch.setattr(
        sched.boto3, "client", lambda *a, **k: pytest.fail(f"built a boto3 client: {a} {k}")
    )


def _input(plan: sched.Plan) -> dict[str, Any]:
    return json.loads(plan.schedule["Target"]["Input"])


def _payload(plan: sched.Plan) -> dict[str, Any]:
    return json.loads(_input(plan)["Payload"])


# --- the documents -------------------------------------------------------


def test_only_scheduler_may_assume_the_role_and_only_for_this_account(plan):
    statement = plan.trust_policy["Statement"][0]

    assert statement["Principal"] == {"Service": "scheduler.amazonaws.com"}
    assert statement["Action"] == "sts:AssumeRole"
    assert statement["Condition"] == {
        "StringEquals": {
            "aws:SourceAccount": ACCOUNT,
            # A schedule *group*: AWS rejects a schedule ARN or a name prefix here.
            "aws:SourceArn": f"arn:aws:scheduler:us-east-1:{ACCOUNT}:schedule-group/default",
        }
    }


def test_the_role_may_invoke_this_runtime_and_do_nothing_else(plan):
    assert plan.permission_policy == {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "InvokeTheMinutesRuntime",
                "Effect": "Allow",
                "Action": "bedrock-agentcore:InvokeAgentRuntime",
                "Resource": [RUNTIME, f"{RUNTIME}/*"],
            }
        ],
    }


def test_the_endpoint_sub_resource_is_named_too():
    # A policy scoped to the bare runtime ARN passes AgentCore's first check
    # and is denied at the endpoint check. Both forms, or neither.
    resources = sched.permission_policy(RUNTIME)["Statement"][0]["Resource"]

    assert resources == [RUNTIME, RUNTIME + "/*"]


def test_a_dead_letter_queue_adds_exactly_one_grant_scoped_to_that_queue():
    queue = f"arn:aws:sqs:us-east-1:{ACCOUNT}:minutes-schedule-dlq"

    statements = sched.permission_policy(RUNTIME, queue)["Statement"]

    assert len(statements) == 2
    assert statements[1]["Action"] == "sqs:SendMessage"
    assert statements[1]["Resource"] == queue
    assert sched.build_plan(runtime=RUNTIME, dead_letter_arn=queue).schedule["Target"][
        "DeadLetterConfig"
    ] == {"Arn": queue}


def test_the_target_is_the_universal_sdk_target_with_no_hyphen(plan):
    # bedrockagentcore here, bedrock-agentcore in the IAM action and the
    # resource ARNs. A schedule with the hyphen creates cleanly and invokes
    # nothing, forever.
    assert sched.UNIVERSAL_TARGET == "arn:aws:scheduler:::aws-sdk:bedrockagentcore:invokeAgentRuntime"
    assert plan.schedule["Target"]["Arn"] == sched.UNIVERSAL_TARGET
    assert "bedrock-agentcore" not in plan.schedule["Target"]["Arn"]
    assert plan.permission_policy["Statement"][0]["Action"].startswith("bedrock-agentcore:")


def test_the_target_input_names_the_arguments_the_invoke_api_expects(plan):
    body = _input(plan)

    assert set(body) == {
        "AgentRuntimeArn",
        "Qualifier",
        "ContentType",
        "RuntimeSessionId",
        "Payload",
    }
    assert body["AgentRuntimeArn"] == RUNTIME
    assert body["Qualifier"] == "DEFAULT"
    assert body["ContentType"] == "application/json"
    assert isinstance(body["Payload"], str)  # a blob member: JSON in a string, not nested JSON


def test_the_scheduled_payload_names_the_case_and_asks_for_background_mode(plan):
    payload = _payload(sched.build_plan(runtime=RUNTIME, case_id="rivera-2027"))

    assert payload["action"] == "wake"
    assert payload["case_id"] == "rivera-2027"
    # Scheduler holds the invocation open and gives up after ~30s, so a wake
    # that runs the caseworker has to acknowledge first and finish after.
    assert payload["background"] is True
    assert payload["scheduled_time"] == "<aws.scheduler.scheduled-time>"
    assert _payload(plan)["case_id"] == "maya-demo"


def test_the_schedule_says_monday_morning_and_means_it(plan):
    assert plan.schedule["ScheduleExpression"] == "cron(0 8 ? * MON *)"
    assert plan.schedule["ScheduleExpressionTimezone"] == "America/New_York"
    assert plan.schedule["FlexibleTimeWindow"] == {"Mode": "OFF"}
    assert plan.schedule["State"] == "ENABLED"


def test_a_failed_invocation_is_retried_once_not_a_hundred_and_eighty_five(plan):
    # A retry re-runs the wake in a fresh session, so it can repeat its writes.
    assert plan.schedule["Target"]["RetryPolicy"]["MaximumRetryAttempts"] <= 2


# --- the session id ------------------------------------------------------


def test_the_session_id_clears_33_characters_even_if_nothing_is_substituted():
    template = sched.session_id_template("minutes-weekly-wake")

    assert template.endswith(sched.EXECUTION_ID)
    empty = template.replace(sched.EXECUTION_ID, "")
    assert len(empty) >= 33
    assert sched.SESSION_ID_PATTERN.fullmatch(empty)


@pytest.mark.parametrize("substitution", ["", "d32c5kddcf5bb8c3", "ad06616e51cdf74a", "0" * 16])
def test_the_session_id_is_legal_for_every_execution_id_aws_has_shown(substitution):
    resolved = sched.session_id_template("minutes-weekly-wake").replace(
        sched.EXECUTION_ID, substitution
    )

    assert sched.SESSION_ID_PATTERN.fullmatch(resolved)
    assert ":" not in resolved and "/" not in resolved


@pytest.mark.parametrize("name", ["w", "minutes", "a-very-long-schedule-name-for-one-case"])
def test_any_schedule_name_still_makes_a_legal_session_id(name):
    resolved = sched.session_id_template(name).replace(sched.EXECUTION_ID, "")

    assert len(resolved) >= 33
    assert sched.SESSION_ID_PATTERN.fullmatch(resolved)


def test_a_name_that_cannot_make_a_legal_session_id_is_refused():
    with pytest.raises(ValueError):
        sched.session_id_template("x" * 120)


# --- what is resolved rather than written down ---------------------------


def test_the_account_region_and_partition_come_out_of_the_runtime_arn():
    other = "arn:aws-us-gov:bedrock-agentcore:us-gov-west-1:999988887777:runtime/minutes-Z"

    plan = sched.build_plan(runtime=other)

    assert plan.account == "999988887777"
    assert plan.region == "us-gov-west-1"
    assert plan.role_arn == "arn:aws-us-gov:iam::999988887777:role/minutes-weekly-wake-scheduler"
    assert plan.trust_policy["Statement"][0]["Condition"]["StringEquals"]["aws:SourceArn"] == (
        "arn:aws-us-gov:scheduler:us-gov-west-1:999988887777:schedule-group/default"
    )


def test_no_account_id_is_written_into_the_script():
    source = (Path(sched.__file__)).read_text(encoding="utf-8")

    # The only twelve-digit run allowed is the obviously fake example ARN.
    assert set(re.findall(r"\b\d{12}\b", source)) <= {"000000000000"}


def test_the_defaults_are_the_ones_the_docstring_promises():
    args = sched.build_parser().parse_args([])

    assert (args.create, args.dry_run, args.delete, args.show) == (False, False, False, False)
    assert args.schedule == "cron(0 8 ? * MON *)"
    assert args.timezone == "America/New_York"
    assert args.case_id == "maya-demo"
    assert args.name == "minutes-weekly-wake"
    assert args.group == "default"
    assert args.arn is None
    assert args.region is None  # taken from the ARN
    assert args.role_name is None  # <name>-scheduler
    assert sched.build_plan(runtime=RUNTIME).role_name == "minutes-weekly-wake-scheduler"


# --- when it next runs ---------------------------------------------------


def _monday(expression: str = sched.DEFAULT_SCHEDULE, now: str = "2026-09-04T09:00:00") -> Any:
    zone = ZoneInfo("America/New_York")
    return sched.next_run(expression, "America/New_York", datetime.fromisoformat(now).replace(tzinfo=zone))


def test_the_next_run_is_the_next_monday_morning():
    assert _monday().isoformat() == "2026-09-07T08:00:00-04:00"


def test_a_monday_after_the_hour_waits_a_week():
    assert _monday(now="2026-09-07T08:01:00").isoformat() == "2026-09-14T08:00:00-04:00"


def test_a_monday_before_the_hour_runs_today():
    assert _monday(now="2026-09-07T07:59:00").isoformat() == "2026-09-07T08:00:00-04:00"


def test_the_timezone_is_the_parents_not_the_runtimes():
    kolkata = sched.next_run(
        sched.DEFAULT_SCHEDULE,
        "Asia/Kolkata",
        datetime(2026, 9, 4, 9, 0, tzinfo=ZoneInfo("Asia/Kolkata")),
    )

    assert kolkata.isoformat() == "2026-09-07T08:00:00+05:30"


def test_a_rate_expression_admits_it_cannot_say():
    # rate() counts from when the schedule was created, which is not in the
    # expression. Better to say so than to print a confident wrong Monday.
    assert _monday("rate(7 days)") is None


def test_syntax_this_does_not_parse_returns_nothing_rather_than_a_guess():
    assert _monday("cron(15 10 ? * 6L 2026-2027)") is None
    assert _monday("cron(0 8 ? * MON)") is None
    assert _monday("nonsense") is None


def test_a_one_time_schedule_reports_its_one_time():
    assert _monday("at(2026-11-20T13:00:00)").isoformat() == "2026-11-20T13:00:00-05:00"


# --- the AWS calls, against a stub ---------------------------------------


def test_dry_run_builds_no_client_and_prints_every_document(no_boto3, capsys):
    assert sched.main(["--dry-run", "--arn", RUNTIME]) == 0

    out = capsys.readouterr().out
    assert "scheduler.amazonaws.com" in out
    assert "bedrock-agentcore:InvokeAgentRuntime" in out
    assert sched.UNIVERSAL_TARGET in out
    assert "next run 2026-" in out or "next run 2027-" in out


def test_dry_run_works_on_a_clone_that_has_never_deployed(no_boto3, monkeypatch, capsys):
    monkeypatch.setattr(sched, "STATE", Path("no-such-deploy-state.json"))

    assert sched.main(["--dry-run"]) == 0

    assert sched.EXAMPLE_ARN in capsys.readouterr().out


def test_without_a_deploy_and_without_an_arn_the_real_modes_stop(no_boto3, monkeypatch):
    monkeypatch.setattr(sched, "STATE", Path("no-such-deploy-state.json"))

    with pytest.raises(SystemExit):
        sched.main(["--show"])


def test_create_makes_the_role_first_then_the_schedule(plan):
    iam, scheduler = StubClient(), StubClient()

    sched.create(plan, iam=iam, scheduler=scheduler, sleep=lambda _: None, log=lambda _: None)

    assert iam.names == ["create_role", "put_role_policy"]
    assert scheduler.names == ["create_schedule"]
    assert json.loads(iam.kwargs("create_role")["AssumeRolePolicyDocument"]) == plan.trust_policy
    assert json.loads(iam.kwargs("put_role_policy")["PolicyDocument"]) == plan.permission_policy
    assert iam.kwargs("put_role_policy")["PolicyName"] == "InvokeMinutesRuntime"
    assert scheduler.kwargs("create_schedule") == plan.schedule


def test_running_create_twice_converges_instead_of_failing(plan):
    iam = StubClient(raises={"create_role": [EntityAlreadyExistsException()]})
    scheduler = StubClient(raises={"create_schedule": [ConflictException()]})

    sched.create(plan, iam=iam, scheduler=scheduler, sleep=lambda _: None, log=lambda _: None)

    assert iam.names == ["create_role", "update_assume_role_policy", "put_role_policy"]
    assert scheduler.names == ["create_schedule", "update_schedule"]
    # A full replace, so the second run leaves exactly what the first intended.
    assert scheduler.kwargs("update_schedule") == plan.schedule


def test_create_waits_for_a_new_role_to_become_assumable(plan):
    scheduler = StubClient(raises={"create_schedule": [ValidationException("role cannot be assumed")]})
    waits: list[float] = []

    sched.create(plan, iam=StubClient(), scheduler=scheduler, sleep=waits.append, log=lambda _: None)

    assert scheduler.names == ["create_schedule", "create_schedule"]
    assert waits == [sched.ROLE_PROPAGATION_WAIT]


def test_create_gives_up_rather_than_waiting_forever(plan):
    scheduler = StubClient(
        raises={"create_schedule": [ValidationException("nope")] * sched.ROLE_PROPAGATION_TRIES}
    )

    with pytest.raises(ValidationException):
        sched.create(
            plan, iam=StubClient(), scheduler=scheduler, sleep=lambda _: None, log=lambda _: None
        )

    assert len(scheduler.names) == sched.ROLE_PROPAGATION_TRIES


def test_delete_takes_back_the_schedule_and_the_role(plan):
    iam, scheduler = StubClient(), StubClient()

    sched.delete(plan, iam=iam, scheduler=scheduler, log=lambda _: None)

    assert scheduler.kwargs("delete_schedule") == {
        "Name": "minutes-weekly-wake",
        "GroupName": "default",
    }
    assert iam.names == ["delete_role_policy", "delete_role"]


def test_deleting_what_is_not_there_is_not_an_error(plan):
    lines: list[str] = []

    sched.delete(
        plan,
        iam=StubClient(raises={"delete_role_policy": [NoSuchEntityException()]}),
        scheduler=StubClient(raises={"delete_schedule": [ResourceNotFoundException()]}),
        log=lines.append,
    )

    assert any("no schedule named" in line for line in lines)
    assert any("no role named" in line for line in lines)


def test_show_reports_the_schedule_and_when_it_next_runs(plan):
    scheduler = StubClient(returns={"get_schedule": {**plan.schedule, "State": "ENABLED"}})
    lines: list[str] = []

    sched.show(
        plan,
        scheduler=scheduler,
        log=lines.append,
        now=datetime(2026, 9, 4, 9, 0, tzinfo=ZoneInfo("America/New_York")),
    )

    printed = "\n".join(lines)
    assert "minutes-weekly-wake (ENABLED)" in printed
    assert "cron(0 8 ? * MON *) [America/New_York]" in printed
    assert "next run:   2026-09-07T08:00:00-04:00" in printed
    assert sched.UNIVERSAL_TARGET in printed


def test_show_says_so_when_nothing_is_scheduled(plan):
    lines: list[str] = []

    sched.show(
        plan,
        scheduler=StubClient(raises={"get_schedule": [ResourceNotFoundException()]}),
        log=lines.append,
    )

    assert lines == ["nothing scheduled: no schedule named minutes-weekly-wake in group default"]
