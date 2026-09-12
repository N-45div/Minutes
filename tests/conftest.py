"""The suite is hermetic. This file is what makes that true on every machine.

Every test that goes through ``app.invoke`` builds a caseworker, and building a
caseworker constructs a ``BedrockModel``, which constructs a boto3 client. No
request is ever sent -- the model is scripted or the call is monkeypatched --
but boto3 still resolves a credential chain when the client is created, and on
a developer laptop that chain reaches whatever ``~/.aws`` holds. With an
``aws login`` session configured, that is the login provider, which imports a
native module; on a machine where that module is blocked, every one of those
tests fails with a dependency error that has nothing to do with the code.

So the suite pins its own environment: static placeholder credentials that
satisfy the chain without consulting any file, and config paths that do not
exist. The README's claim -- no network, no credentials -- was always the
intent. This makes it a property of the tests rather than of the machine.
"""

import os

import pytest


@pytest.fixture(autouse=True, scope="session")
def _no_real_aws_environment():
    """Placeholder credentials, and no ``~/.aws`` at all."""
    overrides = {
        "AWS_ACCESS_KEY_ID": "testing",
        "AWS_SECRET_ACCESS_KEY": "testing",
        "AWS_SESSION_TOKEN": "testing",
        "AWS_DEFAULT_REGION": "us-east-1",
        "AWS_CONFIG_FILE": os.devnull,
        "AWS_SHARED_CREDENTIALS_FILE": os.devnull,
        # Belt and braces: nothing here may ever be pointed at a real bucket.
        "MINUTES_SESSION_BUCKET": "",
    }
    previous = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
