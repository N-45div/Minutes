"""Put the parent's screen at one HTTPS URL, with a Lambda Function URL.

The runtime accepts nothing but SigV4-signed calls, so a browser cannot reach
it and must not be given credentials that could. ``web/lambda_function.py`` is
the proxy that stands in between; this script is what puts it on the account.
It is deliberately small — one IAM role, one function, one URL — and nothing
here runs or costs anything while nobody is looking at the screen.

One URL is the whole application. The function serves ``site/`` itself and
forwards ``POST /api`` to the runtime, so there is no bucket to make public,
no CDN to invalidate and no second origin for the browser to negotiate with.
The screen is zipped with the function at deploy time, read fresh from
``site/`` on every run, so redeploying the screen is running this again.

Least privilege, concretely. The role this creates can do exactly two things:
``bedrock-agentcore:InvokeAgentRuntime`` on this runtime, and write to this
function's own log group. It cannot read the case bucket, cannot call a model,
cannot invoke any other runtime, cannot read any other function's logs. Only
Lambda may assume it, only on behalf of this account, and only for this
function. The runtime ARN is named twice, bare and with ``/*``, because
AgentCore authorises an invocation against the runtime and then against its
endpoint sub-resource; the log group is named twice for the same kind of
reason, because creating it and writing streams into it are authorised
against two spellings of its ARN.

The demo key is generated once, when the function is first created, kept on
every later run, and printed to stdout exactly once per run. ``--rotate-key``
replaces it. Nothing else — not ``--show``, not a log line — ever prints it.

Usage:
    python scripts/deploy_web.py --dry-run     # print everything, call nothing
    python scripts/deploy_web.py               # create, or converge (--create)
    python scripts/deploy_web.py --show        # the URL, the state, the bundle size
    python scripts/deploy_web.py --rotate-key  # converge, with a new demo key
    python scripts/deploy_web.py --delete      # take it all back off the account

``--dry-run`` needs no credentials, no deploy and no network: it builds every
document and the zip, and prints them. The rest need credentials for the
account the runtime is deployed in. No account id, region or partition is
written down here — all three are read out of the runtime ARN, which is read
in turn from ``agentcore/.cli/deployed-state.json`` (written by ``agentcore
deploy``) unless ``--arn`` says otherwise.
"""

from __future__ import annotations

import argparse
import io
import json
import secrets
import sys
import time
import zipfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, NamedTuple

import boto3

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "agentcore" / ".cli" / "deployed-state.json"
LAMBDA_SOURCE = ROOT / "web" / "lambda_function.py"
SITE_DIR = ROOT / "site"

INVOKE_ACTION = "bedrock-agentcore:InvokeAgentRuntime"
LOG_ACTIONS = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]

DEFAULT_NAME = "minutes-web"
POLICY_NAME = "MinutesWebProxy"
HANDLER = "lambda_function.lambda_handler"
RUNTIME = "python3.12"
MEMORY_MB = 512
TIMEOUT_SECONDS = 60  # a wake that compiles a letter calls a model; the runtime can take a while
URL_STATEMENT_ID = "AllowPublicFunctionUrl"
KEY_BYTES = 24

# What the function reads; the same names web/lambda_function.py reads them by.
ENV_RUNTIME_ARN = "MINUTES_RUNTIME_ARN"
ENV_REGION = "MINUTES_REGION"
ENV_DEMO_KEY = "MINUTES_DEMO_KEY"
KEY_PLACEHOLDER = "<generated on create, kept on update, replaced by --rotate-key>"

# Stands in for a real runtime under --dry-run on a clone that has never
# deployed, so reading the plan never requires an AWS account first.
EXAMPLE_ARN = "arn:aws:bedrock-agentcore:us-east-1:000000000000:runtime/minutes-EXAMPLE"

# The browser and the API share one origin, so this CORS only matters to a
# page served from somewhere else that wants to call the API. It allows the
# two headers the proxy reads and nothing more.
CORS = {
    "AllowOrigins": ["*"],
    "AllowMethods": ["GET", "POST"],  # preflight is the service's; OPTIONS is not a member (6-char cap)
    "AllowHeaders": ["content-type", "x-minutes-key"],
    "MaxAge": 3600,
}

# The zip's timestamps are fixed so the same files make the same bytes: a
# redeploy that changed nothing uploads nothing new.
ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)

# IAM is eventually consistent: Lambda can reject a role seconds after IAM
# hands it back. This is that wait, and it is not a poll of anything.
ROLE_PROPAGATION_TRIES = 6
ROLE_PROPAGATION_WAIT = 5.0

# A function is not usable the moment its API call returns; it is created,
# then it becomes Active, and a second update while the first is in flight
# is refused. This is the poll for that.
READY_TRIES = 60
READY_WAIT = 2.0


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
    role_name: str
    role_arn: str
    function_arn: str
    log_group_arn: str
    runtime_arn: str
    runtime_region: str
    region: str
    account: str
    site_dir: Path
    lambda_source: Path
    trust_policy: dict[str, Any]
    permission_policy: dict[str, Any]
    function_config: dict[str, Any]
    url_config: dict[str, Any]
    url_permission: dict[str, Any]


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


def generate_key() -> str:
    """A demo key nobody can guess: 24 random bytes, URL-safe, no padding."""
    return secrets.token_urlsafe(KEY_BYTES)


# --- the documents -------------------------------------------------------


def trust_policy(*, account: str, function_arn: str) -> dict[str, Any]:
    """Who may assume the role: Lambda, for this account, for this function."""
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "AllowLambdaToAssume",
                "Effect": "Allow",
                "Principal": {"Service": "lambda.amazonaws.com"},
                "Action": "sts:AssumeRole",
                "Condition": {
                    "StringEquals": {"aws:SourceAccount": account},
                    "ArnLike": {"aws:SourceArn": function_arn},
                },
            }
        ],
    }


def permission_policy(runtime: str, log_group_arn: str) -> dict[str, Any]:
    """What the role may do: invoke this runtime, write its own logs, nothing else."""
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "InvokeTheMinutesRuntime",
                "Effect": "Allow",
                "Action": INVOKE_ACTION,
                # Twice: AgentCore authorises the call against the runtime and
                # then against its endpoint sub-resource. A policy scoped to
                # the bare ARN passes the first check and fails the second.
                "Resource": [runtime, f"{runtime}/*"],
            },
            {
                "Sid": "WriteThisFunctionsLogs",
                "Effect": "Allow",
                "Action": LOG_ACTIONS,
                # Twice again: CreateLogGroup is authorised against the group's
                # bare ARN, CreateLogStream and PutLogEvents against the group
                # with a stream suffix. Neither spelling covers the other.
                "Resource": [log_group_arn, f"{log_group_arn}:*"],
            },
        ],
    }


def environment(plan: Plan, key: str) -> dict[str, Any]:
    """The function's environment: where the runtime is, and the key that opens the door."""
    return {
        "Variables": {
            ENV_RUNTIME_ARN: plan.runtime_arn,
            ENV_REGION: plan.runtime_region,
            ENV_DEMO_KEY: key,
        }
    }


def build_plan(
    *,
    runtime: str,
    name: str = DEFAULT_NAME,
    role_name: str | None = None,
    region: str | None = None,
    site_dir: Path = SITE_DIR,
    lambda_source: Path = LAMBDA_SOURCE,
) -> Plan:
    """Resolve one runtime ARN into every document this script can apply.

    The function lives in the runtime's account, and by default in its region
    too: the proxy's one job is to reach the runtime, and the fewer borders
    between them the better. ``--region`` moves the function; the runtime's
    own region still goes into ``MINUTES_REGION`` so the call finds it.
    """
    arn = parse_arn(runtime)
    region = region or arn.region
    role_name = role_name or f"{name}-lambda"
    role_arn = f"arn:{arn.partition}:iam::{arn.account}:role/{role_name}"
    function_arn = f"arn:{arn.partition}:lambda:{region}:{arn.account}:function:{name}"
    log_group_arn = f"arn:{arn.partition}:logs:{region}:{arn.account}:log-group:/aws/lambda/{name}"

    function_config: dict[str, Any] = {
        "FunctionName": name,
        "Description": "Minutes: the parent's screen, and the proxy to the runtime behind it.",
        "Runtime": RUNTIME,
        "Handler": HANDLER,
        "Role": role_arn,
        "MemorySize": MEMORY_MB,
        "Timeout": TIMEOUT_SECONDS,
    }
    url_config: dict[str, Any] = {
        "FunctionName": name,
        "AuthType": "NONE",
        "InvokeMode": "BUFFERED",
        "Cors": dict(CORS),
    }
    url_permission: dict[str, Any] = {
        "FunctionName": name,
        "StatementId": URL_STATEMENT_ID,
        "Action": "lambda:InvokeFunctionUrl",
        "Principal": "*",
        "FunctionUrlAuthType": "NONE",
    }
    return Plan(
        name=name,
        role_name=role_name,
        role_arn=role_arn,
        function_arn=function_arn,
        log_group_arn=log_group_arn,
        runtime_arn=runtime,
        runtime_region=arn.region,
        region=region,
        account=arn.account,
        site_dir=Path(site_dir),
        lambda_source=Path(lambda_source),
        trust_policy=trust_policy(account=arn.account, function_arn=function_arn),
        permission_policy=permission_policy(runtime, log_group_arn),
        function_config=function_config,
        url_config=url_config,
        url_permission=url_permission,
    )


# --- the bundle ----------------------------------------------------------


def bundle_files(site_dir: Path, lambda_source: Path) -> list[tuple[str, Path]]:
    """What goes in the zip, in zip order: the function at the root, then
    ``site/`` as it is on disk at this moment, since another hand may be
    editing it while this runs. Dotfiles and Python caches stay out."""
    if not lambda_source.is_file():
        raise FileNotFoundError(f"no function source at {lambda_source}")
    files: list[tuple[str, Path]] = [("lambda_function.py", lambda_source)]
    if site_dir.is_dir():
        for path in sorted(site_dir.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(site_dir)
            if any(part.startswith(".") or part == "__pycache__" for part in relative.parts):
                continue
            files.append((f"site/{relative.as_posix()}", path))
    return files


def build_bundle(site_dir: Path, lambda_source: Path) -> bytes:
    """The deployment zip, byte-for-byte the same for the same files."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for arcname, path in bundle_files(site_dir, lambda_source):
            info = zipfile.ZipInfo(arcname, date_time=ZIP_EPOCH)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    return buffer.getvalue()


# --- the four modes ------------------------------------------------------


def dry_run(plan: Plan, *, log: Callable[[str], None] = print) -> None:
    """Print the whole plan and touch nothing. No client is built on this path."""
    files = bundle_files(plan.site_dir, plan.lambda_source)
    bundle = build_bundle(plan.site_dir, plan.lambda_source)
    log("# --dry-run makes no AWS call and needs no credentials")
    log(f"# runtime  {plan.runtime_arn}")
    log(f"# account  {plan.account}   region {plan.region}   function {plan.function_arn}")
    log(f"# role     {plan.role_arn}")
    log("")
    log(f"# iam create-role --role-name {plan.role_name} --assume-role-policy-document")
    log(json.dumps(plan.trust_policy, indent=2))
    log("")
    log(f"# iam put-role-policy --role-name {plan.role_name} --policy-name {POLICY_NAME}")
    log(json.dumps(plan.permission_policy, indent=2))
    log("")
    log("# lambda create-function (or update-function-configuration + update-function-code)")
    log(json.dumps({**plan.function_config, "Environment": environment(plan, KEY_PLACEHOLDER)}, indent=2))
    log("")
    log("# lambda create-function-url-config (or update-function-url-config)")
    log(json.dumps(plan.url_config, indent=2))
    log("")
    log("# lambda add-permission")
    log(json.dumps(plan.url_permission, indent=2))
    log("")
    log(f"# bundle: {len(files)} files, {len(bundle)} bytes zipped")
    for arcname, path in files:
        log(f"#   {arcname}  ({path.stat().st_size} bytes)")
    if not any(arcname == "site/index.html" for arcname, _ in files):
        log(f"# note: {plan.site_dir / 'index.html'} is not there yet; GET / will answer 404 until it is")


def _existing(lam: Any, name: str) -> dict[str, Any] | None:
    try:
        return lam.get_function(FunctionName=name)
    except lam.exceptions.ResourceNotFoundException:
        return None


def _current_key(existing: dict[str, Any] | None) -> str | None:
    if not existing:
        return None
    variables = ((existing.get("Configuration") or {}).get("Environment") or {}).get("Variables") or {}
    return variables.get(ENV_DEMO_KEY) or None


def wait_ready(
    lam: Any,
    name: str,
    *,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
) -> None:
    """Block until the function is Active and its last update is settled.

    Lambda refuses a second change while the first is still being applied,
    and a URL made for a function that is still Pending points at nothing.
    """
    for attempt in range(1, READY_TRIES + 1):
        current = lam.get_function_configuration(FunctionName=name)
        state = current.get("State")
        update = current.get("LastUpdateStatus")
        if state == "Failed" or update == "Failed":
            reason = current.get("StateReason") or current.get("LastUpdateStatusReason") or "no reason"
            raise RuntimeError(f"function {name} failed: {reason}")
        if state != "Pending" and update != "InProgress":
            return
        if attempt == READY_TRIES:
            raise RuntimeError(f"function {name} is still {state}/{update} after {READY_TRIES} checks")
        if attempt == 1:
            log(f"waiting for {name} to become active")
        sleep(READY_WAIT)


def create(
    plan: Plan,
    *,
    iam: Any,
    lam: Any,
    bundle: bytes,
    rotate_key: bool = False,
    new_key: Callable[[], str] = generate_key,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
) -> str:
    """Create the role, the function and the URL, or converge them. Safe to run twice.

    Returns the URL. The demo key is printed once, here, on stdout, in a
    line that says what it is; it is returned to nobody.
    """
    trust = json.dumps(plan.trust_policy)
    try:
        iam.create_role(
            RoleName=plan.role_name,
            AssumeRolePolicyDocument=trust,
            Description=f"Lambda {plan.name} serves the Minutes screen and invokes the runtime.",
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

    existing = _existing(lam, plan.name)
    key = None if rotate_key else _current_key(existing)
    if key:
        log("demo key kept from the deployed function")
    else:
        key = new_key()
        log("demo key generated" + (" (rotated)" if existing and rotate_key else ""))
    env = environment(plan, key)

    if existing is None:
        for attempt in range(1, ROLE_PROPAGATION_TRIES + 1):
            try:
                lam.create_function(
                    **plan.function_config,
                    Environment=env,
                    Code={"ZipFile": bundle},
                    Publish=False,
                )
                log(f"function created: {plan.function_arn} ({len(bundle)} bytes)")
                break
            except lam.exceptions.InvalidParameterValueException as exc:
                if attempt == ROLE_PROPAGATION_TRIES:
                    raise
                log(f"waiting for the new role to become assumable ({exc})")
                sleep(ROLE_PROPAGATION_WAIT)
        wait_ready(lam, plan.name, sleep=sleep, log=log)
    else:
        # Configuration first, then code, with the function settled between
        # them: Lambda refuses the second while the first is in progress.
        lam.update_function_configuration(**plan.function_config, Environment=env)
        log(f"function configuration updated: {plan.function_arn}")
        wait_ready(lam, plan.name, sleep=sleep, log=log)
        lam.update_function_code(FunctionName=plan.name, ZipFile=bundle, Publish=False)
        log(f"function code updated ({len(bundle)} bytes)")
        wait_ready(lam, plan.name, sleep=sleep, log=log)

    try:
        made = lam.create_function_url_config(**plan.url_config)
        log("function URL created")
    except lam.exceptions.ResourceConflictException:
        made = lam.update_function_url_config(**plan.url_config)
        log("function URL kept, configuration updated")
    url = str(made.get("FunctionUrl") or "")

    try:
        lam.add_permission(**plan.url_permission)
        log(f"public invoke permission added: {URL_STATEMENT_ID}")
    except lam.exceptions.ResourceConflictException:
        log(f"public invoke permission already there: {URL_STATEMENT_ID}")

    log("")
    log(f"URL:  {url}")
    log(f"KEY:  {key}    (send as x-minutes-key; printed here and nowhere else)")
    return url


def delete(plan: Plan, *, iam: Any, lam: Any, log: Callable[[str], None] = print) -> None:
    """Undo everything --create made, in the order that leaves nothing orphaned:
    the URL, the permission that made it public, the function, then the role."""
    try:
        lam.delete_function_url_config(FunctionName=plan.name)
        log("function URL deleted")
    except lam.exceptions.ResourceNotFoundException:
        log(f"no function URL on {plan.name}")
    try:
        lam.remove_permission(FunctionName=plan.name, StatementId=URL_STATEMENT_ID)
        log(f"public invoke permission removed: {URL_STATEMENT_ID}")
    except lam.exceptions.ResourceNotFoundException:
        log(f"no permission {URL_STATEMENT_ID} on {plan.name}")
    try:
        lam.delete_function(FunctionName=plan.name)
        log(f"function deleted: {plan.name}")
    except lam.exceptions.ResourceNotFoundException:
        log(f"no function named {plan.name}")
    try:
        iam.delete_role_policy(RoleName=plan.role_name, PolicyName=POLICY_NAME)
        iam.delete_role(RoleName=plan.role_name)
        log(f"role deleted: {plan.role_name}")
    except iam.exceptions.NoSuchEntityException:
        log(f"no role named {plan.role_name}")


def show(plan: Plan, *, lam: Any, log: Callable[[str], None] = print) -> None:
    """What is actually on the account. The key is reported as set or not, never shown."""
    existing = _existing(lam, plan.name)
    if existing is None:
        log(f"nothing deployed: no function named {plan.name} in {plan.region}")
        return
    config = existing.get("Configuration") or {}
    try:
        url = lam.get_function_url_config(FunctionName=plan.name).get("FunctionUrl")
    except lam.exceptions.ResourceNotFoundException:
        url = None
    log(f"url:           {url or 'none (run without --show to create it)'}")
    log(f"function:      {config.get('FunctionArn') or plan.function_arn}")
    log(f"state:         {config.get('State')} / {config.get('LastUpdateStatus')}")
    log(f"last modified: {config.get('LastModified')}")
    log(f"bundle size:   {config.get('CodeSize')} bytes")
    log(f"runtime:       {config.get('Runtime')}  {config.get('MemorySize')} MB  {config.get('Timeout')} s")
    log(f"demo key:      {'set' if _current_key(existing) else 'NOT SET'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--create", action="store_true", help="create or converge (the default)")
    mode.add_argument("--dry-run", action="store_true", help="print every document, call nothing")
    mode.add_argument("--delete", action="store_true", help="remove the URL, function and role")
    mode.add_argument("--show", action="store_true", help="print the URL, state and bundle size")
    parser.add_argument("--arn", default=None, help=f"runtime ARN; default: read from {STATE.name}")
    parser.add_argument(
        "--region", default=None, help="where the function lives; default: the runtime's region"
    )
    parser.add_argument("--name", default=DEFAULT_NAME, help="function name")
    parser.add_argument("--role-name", default=None, help="default: <name>-lambda")
    parser.add_argument(
        "--site", default=None, help=f"directory to bundle as the screen; default: {SITE_DIR}"
    )
    parser.add_argument("--rotate-key", action="store_true", help="replace the demo key on this run")
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
        role_name=args.role_name,
        region=args.region,
        site_dir=Path(args.site) if args.site else SITE_DIR,
    )

    if args.dry_run:
        dry_run(plan)
        return 0

    lam = boto3.client("lambda", region_name=plan.region)
    if args.show:
        show(plan, lam=lam)
        return 0
    iam = boto3.client("iam")
    if args.delete:
        delete(plan, iam=iam, lam=lam)
        return 0
    if not plan.site_dir.is_dir():
        raise SystemExit(f"no screen to bundle at {plan.site_dir}; pass --site")
    bundle = build_bundle(plan.site_dir, plan.lambda_source)
    create(plan, iam=iam, lam=lam, bundle=bundle, rotate_key=args.rotate_key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
