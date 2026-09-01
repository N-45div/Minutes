"""Invoke the deployed Minutes runtime on Amazon Bedrock AgentCore.

Usage:
    python scripts/invoke_runtime.py '{"action": "wake", "today": "2026-12-01"}'
    python scripts/invoke_runtime.py '{"action": "ask", "prompt": "..."}' --session <id>

The runtime ARN is read from agentcore/.cli/deployed-state.json (written by
`agentcore deploy`) unless --arn is given. A session id must be at least 33
characters; a fresh uuid4 string is minted when none is given, and printed so
a follow-up call (an `answer` to a pending approval) can reuse it.
"""

import argparse
import json
import sys
import uuid
from pathlib import Path

import boto3

STATE = Path(__file__).resolve().parents[1] / "agentcore" / ".cli" / "deployed-state.json"


def runtime_arn(target: str = "default", name: str = "minutes") -> str:
    state = json.loads(STATE.read_text(encoding="utf-8"))
    entry = state.get("targets", {}).get(target, {})
    text = json.dumps(entry)
    for candidate in _arns(entry):
        if ":runtime/" in candidate and name in candidate:
            return candidate
    raise SystemExit(f"no runtime ARN for '{name}' in {STATE}; pass --arn. State was: {text[:400]}")


def _arns(node):
    if isinstance(node, str) and node.startswith("arn:aws:bedrock-agentcore:"):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from _arns(value)
    elif isinstance(node, list):
        for value in node:
            yield from _arns(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", help="JSON payload for the entrypoint")
    parser.add_argument("--arn", default=None)
    parser.add_argument("--session", default=None, help="runtimeSessionId (>= 33 chars)")
    parser.add_argument("--region", default="us-east-1")
    args = parser.parse_args()

    session = args.session or str(uuid.uuid4())
    if len(session) < 33:
        raise SystemExit("session id must be at least 33 characters")

    client = boto3.client("bedrock-agentcore", region_name=args.region)
    response = client.invoke_agent_runtime(
        agentRuntimeArn=args.arn or runtime_arn(),
        runtimeSessionId=session,
        contentType="application/json",
        accept="application/json",
        payload=args.payload.encode("utf-8"),
    )
    body = response["response"].read().decode("utf-8")
    print(f"session: {session}", file=sys.stderr)
    print(f"status:  {response.get('statusCode')}", file=sys.stderr)
    try:
        print(json.dumps(json.loads(body), indent=2, ensure_ascii=False))
    except json.JSONDecodeError:
        print(body)


if __name__ == "__main__":
    main()
