"""Make the weekly wake fire on its own, with EventBridge Scheduler.

Minutes is a background agent, and a background agent that only runs when a
human remembers to curl it is not a background agent — it is a CLI with extra
steps. The whole promise is that nobody has to keep the IEP in their head: the
wake happens on Monday morning whether or not anyone thought about it that
week, and the parent hears back only when a decision is waiting. This script is
the piece that makes that true in a real account, and it is deliberately small
— one IAM role, one schedule, no Lambda, and nothing that runs or costs
anything between Mondays.

EventBridge Scheduler calls the runtime directly through its universal target,
``arn:aws:scheduler:::aws-sdk:bedrockagentcore:invokeAgentRuntime``. The service
segment there is ``bedrockagentcore`` with no hyphen, while the IAM action and
every resource ARN use ``bedrock-agentcore`` with one. That is not a typo in
either place, and getting it wrong creates a schedule that fires forever and
invokes nothing.

Least privilege, concretely. The role this creates can do exactly one thing:
``bedrock-agentcore:InvokeAgentRuntime`` on this runtime. It cannot read the
case bucket, cannot call a model, cannot invoke any other runtime in the
account. Only EventBridge Scheduler may assume it, only on behalf of this
account, and only from this schedule group. The runtime ARN is named twice,
once bare and once with ``/*``, because AgentCore authorises an invocation
against the runtime and again against its endpoint sub-resource — a policy
scoped to the bare ARN passes the first check and is denied at the second.

Usage:
    python scripts/schedule_weekly.py --dry-run   # print everything, call nothing
    python scripts/schedule_weekly.py             # create, or converge (--create)
    python scripts/schedule_weekly.py --show      # what is scheduled, and when it next runs
    python scripts/schedule_weekly.py --delete    # take it all back off the account

``--dry-run`` needs no credentials, no deploy and no network: it builds every
document and prints it. The rest need credentials for the account the runtime
is deployed in. No account id, region or partition is written down here — all
three are read out of the runtime ARN, which is read in turn from
``agentcore/.cli/deployed-state.json`` (written by ``agentcore deploy``) unless
``--arn`` says otherwise. There is nothing to keep in step by hand.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, NamedTuple
from zoneinfo import ZoneInfo

import boto3

STATE = Path(__file__).resolve().parents[1] / "agentcore" / ".cli" / "deployed-state.json"

# The universal target: Scheduler makes this SDK call itself, so there is no
# Lambda to maintain, to pay for, or to lose the payload in.
UNIVERSAL_TARGET = "arn:aws:scheduler:::aws-sdk:bedrockagentcore:invokeAgentRuntime"
INVOKE_ACTION = "bedrock-agentcore:InvokeAgentRuntime"

# Scheduler substitutes these into the target at fire time.
EXECUTION_ID = "<aws.scheduler.execution-id>"
SCHEDULED_TIME = "<aws.scheduler.scheduled-time>"

DEFAULT_NAME = "minutes-weekly-wake"
DEFAULT_GROUP = "default"
DEFAULT_SCHEDULE = "cron(0 8 ? * MON *)"  # Monday, 08:00, ahead of the school week
DEFAULT_TIMEZONE = "America/New_York"
DEFAULT_CASE_ID = "maya-demo"  # the sample case; a real deployment names its own
POLICY_NAME = "InvokeMinutesRuntime"

# Stands in for a real runtime under --dry-run on a clone that has never
# deployed, so reading the plan never requires an AWS account first.
EXAMPLE_ARN = "arn:aws:bedrock-agentcore:us-east-1:000000000000:runtime/minutes-EXAMPLE"

# A runtimeSessionId must be at least 33 characters, and has to look like an
# identifier in both directions of the API.
SESSION_ID_MIN = 33
SESSION_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{32,99}$")

# A retry re-runs the wake in a fresh session, so it can re-do the wake's
# writes. One retry covers a blip; the service default of 185 would not be a
# retry policy, it would be a fire drill.
MAX_RETRY_ATTEMPTS = 1
MAX_EVENT_AGE_SECONDS = 3600

# IAM is eventually consistent: Scheduler can reject a role seconds after IAM
# hands it back. This is that wait, and it is not a poll of anything.
ROLE_PROPAGATION_TRIES = 6
ROLE_PROPAGATION_WAIT = 5.0


class Arn(NamedTuple):
    """The ARN fields this script needs so it can hardcode none of them."""

    partition: str
    service: str
    region: str
    account: str
    resource: str


class Plan(NamedTuple):
    """Everything the four modes share: resolved once, then printed or applied."""

    name: str
    group: str
    role_name: str
    role_arn: str
    runtime_arn: str
    region: str
    account: str
    case_id: str
    expression: str
    timezone: str
    dead_letter_arn: str | None
    trust_policy: dict[str, Any]
    permission_policy: dict[str, Any]
    schedule: dict[str, Any]


def runtime_arn(target: str = "default", name: str = "minutes") -> str:
    """The deployed runtime's ARN, from the state `agentcore deploy` leaves behind."""
    if not STATE.exists():
        raise LookupError(f"no deploy state at {STATE}; deploy first, or pass --arn")
    state = json.loads(STATE.read_text(encoding="utf-8"))
    entry = state.get("targets", {}).get(target, {})
    for candidate in _arns(entry):
        if ":runtime/" in candidate and name in candidate:
            return candidate
    raise LookupError(f"no runtime ARN for '{name}' in {STATE}; pass --arn")


def _arns(node: Any) -> Iterator[str]:
    if isinstance(node, str) and node.startswith("arn:aws:bedrock-agentcore:"):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from _arns(value)
    elif isinstance(node, list):
        for value in node:
            yield from _arns(value)


def parse_arn(arn: str) -> Arn:
    """Split an ARN: the account, region and partition all come from it."""
    parts = arn.split(":", 5)
    if len(parts) != 6 or parts[0] != "arn":
        raise ValueError(f"not an ARN: {arn!r}")
    return Arn(parts[1], parts[2], parts[3], parts[4], parts[5])


def session_id_template(name: str) -> str:
    """The runtimeSessionId Scheduler stamps on each run.

    Of the four context attributes Scheduler can substitute, only the execution
    id is usable here: the schedule ARN and the scheduled time both carry ``:``
    and ``/``, which a session id may not. The execution id is also short — the
    ids in AWS's own examples are sixteen characters and no minimum is
    documented — while the runtime demands at least thirty-three. So the
    literal part is padded until it clears thirty-three on its own, and the id
    stays legal even if the substitution came back empty.

    Each run therefore gets a fresh session, which is a choice and not an
    accident: the case lives in S3 under its case id, so a new microVM loses
    nothing, and one week's run cannot leave anything on disk for the next one
    to trip over.
    """
    stem = re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-") or "minutes"
    literal = f"{stem}-".ljust(SESSION_ID_MIN - 1, "0") + "-"
    template = literal + EXECUTION_ID
    for substitution in ("", "d32c5kddcf5bb8c3"):
        resolved = template.replace(EXECUTION_ID, substitution)
        if not SESSION_ID_PATTERN.fullmatch(resolved):
            raise ValueError(f"--name {name!r} makes an illegal session id: {resolved!r}")
    return template


def trust_policy(*, account: str, region: str, partition: str, group: str) -> dict[str, Any]:
    """Who may assume the role: EventBridge Scheduler, for this account only."""
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "AllowSchedulerToAssume",
                "Effect": "Allow",
                "Principal": {"Service": "scheduler.amazonaws.com"},
                "Action": "sts:AssumeRole",
                "Condition": {
                    "StringEquals": {
                        "aws:SourceAccount": account,
                        # A schedule *group* ARN. AWS rejects scoping this
                        # condition to one schedule or to a name prefix.
                        "aws:SourceArn": (
                            f"arn:{partition}:scheduler:{region}:{account}:schedule-group/{group}"
                        ),
                    }
                },
            }
        ],
    }


def permission_policy(runtime: str, dead_letter_arn: str | None = None) -> dict[str, Any]:
    """What the role may do: invoke this runtime, and nothing else anywhere."""
    statements: list[dict[str, Any]] = [
        {
            "Sid": "InvokeTheMinutesRuntime",
            "Effect": "Allow",
            "Action": INVOKE_ACTION,
            # Twice: AgentCore authorises the call against the runtime and then
            # against its endpoint sub-resource, and the docs spell that
            # sub-resource two different ways. '/*' covers both without
            # guessing which spelling is current.
            "Resource": [runtime, f"{runtime}/*"],
        }
    ]
    if dead_letter_arn:
        statements.append(
            {
                "Sid": "ReportFailuresToTheDeadLetterQueue",
                "Effect": "Allow",
                "Action": "sqs:SendMessage",
                "Resource": dead_letter_arn,
            }
        )
    return {"Version": "2012-10-17", "Statement": statements}


def wake_payload(case_id: str) -> dict[str, Any]:
    """The payload the schedule posts: one wake, for one case."""
    return {
        "action": "wake",
        "case_id": case_id,
        # The universal target is synchronous — Scheduler holds the call open
        # and gives up after roughly thirty seconds. A wake that finds a letter
        # runs the caseworker, which takes longer than that, so it acknowledges
        # at once and finishes in the background. Without this the wake would
        # do its work correctly and the invocation would still be recorded as
        # a failure, and then retried.
        "background": True,
        # The entrypoint ignores this. It is here because the scheduled time
        # cannot go in the session id, and it is what ties a CloudWatch line or
        # a dead-letter message back to the run that produced it.
        "scheduled_time": SCHEDULED_TIME,
    }


def target_input(*, runtime: str, case_id: str, name: str) -> str:
    """Target.Input: JSON naming the API's arguments, one of which is more JSON."""
    return json.dumps(
        {
            "AgentRuntimeArn": runtime,
            "Qualifier": "DEFAULT",
            "ContentType": "application/json",
            "RuntimeSessionId": session_id_template(name),
            "Payload": json.dumps(wake_payload(case_id)),
        }
    )


def build_plan(
    *,
    runtime: str,
    name: str = DEFAULT_NAME,
    group: str = DEFAULT_GROUP,
    role_name: str | None = None,
    case_id: str = DEFAULT_CASE_ID,
    expression: str = DEFAULT_SCHEDULE,
    timezone: str = DEFAULT_TIMEZONE,
    dead_letter_arn: str | None = None,
    region: str | None = None,
) -> Plan:
    """Resolve one runtime ARN into every document this script can apply."""
    arn = parse_arn(runtime)
    region = region or arn.region
    role_name = role_name or f"{name}-scheduler"
    role_arn = f"arn:{arn.partition}:iam::{arn.account}:role/{role_name}"

    target: dict[str, Any] = {
        "Arn": UNIVERSAL_TARGET,
        "RoleArn": role_arn,
        "Input": target_input(runtime=runtime, case_id=case_id, name=name),
        "RetryPolicy": {
            "MaximumRetryAttempts": MAX_RETRY_ATTEMPTS,
            "MaximumEventAgeInSeconds": MAX_EVENT_AGE_SECONDS,
        },
    }
    if dead_letter_arn:
        target["DeadLetterConfig"] = {"Arn": dead_letter_arn}

    schedule: dict[str, Any] = {
        "Name": name,
        "GroupName": group,
        "Description": f"Minutes: the weekly wake for case {case_id}.",
        "State": "ENABLED",
        "ScheduleExpression": expression,
        "ScheduleExpressionTimezone": timezone,
        # A parent expecting Monday morning should get Monday morning.
        "FlexibleTimeWindow": {"Mode": "OFF"},
        "Target": target,
    }
    return Plan(
        name=name,
        group=group,
        role_name=role_name,
        role_arn=role_arn,
        runtime_arn=runtime,
        region=region,
        account=arn.account,
        case_id=case_id,
        expression=expression,
        timezone=timezone,
        dead_letter_arn=dead_letter_arn,
        trust_policy=trust_policy(
            account=arn.account, region=region, partition=arn.partition, group=group
        ),
        permission_policy=permission_policy(runtime, dead_letter_arn),
        schedule=schedule,
    )


_MONTHS = {
    name: number
    for number, name in enumerate(
        ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1
    )
}
_WEEKDAYS = {
    name: number for number, name in enumerate(["SUN", "MON", "TUE", "WED", "THU", "FRI", "SAT"], 1)
}


def _value(token: str, names: dict[str, int] | None) -> int:
    if names and token in names:
        return names[token]
    return int(token)


def _field(spec: str, low: int, high: int, names: dict[str, int] | None = None) -> set[int] | None:
    """One cron field as the values it matches; None means every value."""
    spec = spec.upper()
    if spec in {"*", "?"}:
        return None
    values: set[int] = set()
    for part in spec.split(","):
        step = 1
        if "/" in part:
            part, _, raw = part.partition("/")
            step = int(raw)
            if part in {"*", "?", ""}:
                part = f"{low}-{high}"
        if "-" in part:
            first_token, _, last_token = part.partition("-")
            first, last = _value(first_token, names), _value(last_token, names)
        else:
            first = _value(part, names)
            last = high if step > 1 else first
        values.update(range(first, last + 1, step))
    return values


def next_run(expression: str, timezone: str, now: datetime | None = None) -> datetime | None:
    """When this expression fires next, or None when that cannot be worked out.

    Scheduler's API reports what a schedule is, never when it will next run, so
    ``--show`` answers here the question a person actually has. A ``rate()``
    schedule counts from when it was created, which is not in the expression,
    and ``L``/``W``/``#`` in a cron field are real syntax this does not try to
    guess at; both honestly return None rather than a confidently wrong Monday.
    """
    zone = ZoneInfo(timezone)
    now = (now or datetime.now(zone)).astimezone(zone)
    body = expression.strip()

    if body.startswith("at(") and body.endswith(")"):
        try:
            return datetime.fromisoformat(body[3:-1]).replace(tzinfo=zone)
        except ValueError:
            return None
    if not (body.startswith("cron(") and body.endswith(")")):
        return None
    fields = body[5:-1].split()
    if len(fields) != 6:
        return None
    try:
        minutes = _field(fields[0], 0, 59)
        hours = _field(fields[1], 0, 23)
        days = _field(fields[2], 1, 31)
        months = _field(fields[3], 1, 12, _MONTHS)
        weekdays = _field(fields[4], 1, 7, _WEEKDAYS)
        years = _field(fields[5], 1970, 2199)
    except ValueError:
        return None

    hour_list = sorted(hours if hours is not None else range(24))
    minute_list = sorted(minutes if minutes is not None else range(60))
    day = now.date()
    for _ in range(366 * 4):
        # AWS numbers days of the week 1=SUN..7=SAT; Python numbers them 1=MON.
        aws_weekday = (day.isoweekday() % 7) + 1
        if (
            (years is None or day.year in years)
            and (months is None or day.month in months)
            and (days is None or day.day in days)
            and (weekdays is None or aws_weekday in weekdays)
        ):
            for hour in hour_list:
                for minute in minute_list:
                    candidate = datetime(day.year, day.month, day.day, hour, minute, tzinfo=zone)
                    if candidate > now:
                        return candidate
        day += timedelta(days=1)
    return None


def _when(expression: str, timezone: str, now: datetime | None = None) -> str:
    upcoming = next_run(expression, timezone, now)
    return upcoming.isoformat() if upcoming else "not derivable from this expression"


def dry_run(plan: Plan, *, log: Callable[[str], None] = print, now: datetime | None = None) -> None:
    """Print the whole plan and touch nothing. No client is built on this path."""
    log("# --dry-run makes no AWS call and needs no credentials")
    log(f"# runtime  {plan.runtime_arn}")
    log(f"# account  {plan.account}   region {plan.region}   group {plan.group}")
    log(f"# role     {plan.role_arn}")
    log(f"# next run {_when(plan.expression, plan.timezone, now)}")
    log("")
    log(f"# iam create-role --role-name {plan.role_name} --assume-role-policy-document")
    log(json.dumps(plan.trust_policy, indent=2))
    log("")
    log(f"# iam put-role-policy --role-name {plan.role_name} --policy-name {POLICY_NAME}")
    log(json.dumps(plan.permission_policy, indent=2))
    log("")
    log("# scheduler create-schedule")
    log(json.dumps(plan.schedule, indent=2))


def _put_schedule(plan: Plan, scheduler: Any, log: Callable[[str], None]) -> None:
    try:
        scheduler.create_schedule(**plan.schedule)
        log(f"schedule created: {plan.name}")
    except scheduler.exceptions.ConflictException:
        # Idempotence. update_schedule is a full replace and the plan is the
        # whole schedule, so a second run converges instead of failing.
        scheduler.update_schedule(**plan.schedule)
        log(f"schedule updated: {plan.name}")


def create(
    plan: Plan,
    *,
    iam: Any,
    scheduler: Any,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
) -> None:
    """Create the role and the schedule, or converge them. Safe to run twice."""
    trust = json.dumps(plan.trust_policy)
    try:
        iam.create_role(
            RoleName=plan.role_name,
            AssumeRolePolicyDocument=trust,
            Description=f"EventBridge Scheduler invokes Minutes for case {plan.case_id}.",
        )
        log(f"role created: {plan.role_arn}")
    except iam.exceptions.EntityAlreadyExistsException:
        iam.update_assume_role_policy(RoleName=plan.role_name, PolicyDocument=trust)
        log(f"role kept, trust policy updated: {plan.role_arn}")
    iam.put_role_policy(
        RoleName=plan.role_name,
        PolicyName=POLICY_NAME,
        PolicyDocument=json.dumps(plan.permission_policy),
    )
    log(f"policy on the role: {POLICY_NAME}")

    for attempt in range(1, ROLE_PROPAGATION_TRIES + 1):
        try:
            _put_schedule(plan, scheduler, log)
            log(f"the wake now runs on its own: {plan.expression} [{plan.timezone}]")
            log(f"next run: {_when(plan.expression, plan.timezone)}")
            return
        except scheduler.exceptions.ValidationException as exc:
            if attempt == ROLE_PROPAGATION_TRIES:
                raise
            log(f"waiting for the new role to become assumable ({exc})")
            sleep(ROLE_PROPAGATION_WAIT)


def delete(plan: Plan, *, iam: Any, scheduler: Any, log: Callable[[str], None] = print) -> None:
    """Undo everything --create made, leaving nothing orphaned behind it."""
    try:
        scheduler.delete_schedule(Name=plan.name, GroupName=plan.group)
        log(f"schedule deleted: {plan.name}")
    except scheduler.exceptions.ResourceNotFoundException:
        log(f"no schedule named {plan.name} in group {plan.group}")
    try:
        iam.delete_role_policy(RoleName=plan.role_name, PolicyName=POLICY_NAME)
        iam.delete_role(RoleName=plan.role_name)
        log(f"role deleted: {plan.role_name}")
    except iam.exceptions.NoSuchEntityException:
        log(f"no role named {plan.role_name}")


def show(
    plan: Plan,
    *,
    scheduler: Any,
    log: Callable[[str], None] = print,
    now: datetime | None = None,
) -> None:
    """What is actually on the account, and when the parent next hears from it."""
    try:
        current = scheduler.get_schedule(Name=plan.name, GroupName=plan.group)
    except scheduler.exceptions.ResourceNotFoundException:
        log(f"nothing scheduled: no schedule named {plan.name} in group {plan.group}")
        return
    expression = current.get("ScheduleExpression", "")
    timezone = current.get("ScheduleExpressionTimezone") or "UTC"
    target = current.get("Target", {})
    log(f"schedule:   {current.get('Name')} ({current.get('State')})")
    log(f"expression: {expression} [{timezone}]")
    log(f"next run:   {_when(expression, timezone, now)}")
    log(f"target:     {target.get('Arn')}")
    log(f"role:       {target.get('RoleArn')}")
    log(f"input:      {target.get('Input')}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--create", action="store_true", help="create or converge (the default)")
    mode.add_argument("--dry-run", action="store_true", help="print every document, call nothing")
    mode.add_argument("--delete", action="store_true", help="remove the schedule and the role")
    mode.add_argument("--show", action="store_true", help="print the schedule and its next run")
    parser.add_argument("--arn", default=None, help=f"runtime ARN; default: read from {STATE.name}")
    parser.add_argument("--case-id", default=DEFAULT_CASE_ID, help="the case to wake for")
    parser.add_argument("--schedule", default=DEFAULT_SCHEDULE, help="cron() or rate() expression")
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE, help="IANA name")
    parser.add_argument("--region", default=None, help="default: the region in the runtime ARN")
    parser.add_argument("--name", default=DEFAULT_NAME, help="schedule name")
    parser.add_argument("--group", default=DEFAULT_GROUP, help="schedule group")
    parser.add_argument("--role-name", default=None, help="default: <name>-scheduler")
    parser.add_argument(
        "--dead-letter-arn",
        default=None,
        help="standard SQS queue for failed invocations; the role is granted SendMessage on it",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        runtime = args.arn or runtime_arn()
    except LookupError as exc:
        if not args.dry_run:
            raise SystemExit(str(exc)) from exc
        print(f"# {exc}; the plan below uses an example ARN", file=sys.stderr)
        runtime = EXAMPLE_ARN

    plan = build_plan(
        runtime=runtime,
        name=args.name,
        group=args.group,
        role_name=args.role_name,
        case_id=args.case_id,
        expression=args.schedule,
        timezone=args.timezone,
        dead_letter_arn=args.dead_letter_arn,
        region=args.region,
    )

    if args.dry_run:
        dry_run(plan)
        return 0

    scheduler = boto3.client("scheduler", region_name=plan.region)
    if args.show:
        show(plan, scheduler=scheduler)
        return 0
    iam = boto3.client("iam")
    if args.delete:
        delete(plan, iam=iam, scheduler=scheduler)
    else:
        create(plan, iam=iam, scheduler=scheduler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
