"""Per-case storage — the place a real family's case lives.

Until this module existed every tool in Minutes read one committed sample
case out of ``fixtures/``. A parent running THEIR case needs three things kept
somewhere the next invocation can find them: the ledger extracted from their
IEP, the correspondence they have pasted in, and the delivery facts read out
of it. That is the whole of what a :class:`CaseStore` holds, and it holds it
as JSON files under ``cases/<case_id>/`` so that the on-disk layout and the S3
layout are the same layout:

    cases/<case_id>/ledger.json          IEPLedger
    cases/<case_id>/correspondence.json  list[Correspondence]
    cases/<case_id>/events.json          list[ServiceEvent]
    cases/<case_id>/meta.json            {created, updated, student_alias, source}

Two stores implement it. :class:`DirCaseStore` is a directory — a laptop, a
test. :class:`S3CaseStore` is the bucket the caseworker sessions already live
in, under a different prefix, because the runtime that serves this product is
a microVM whose disk goes with it and S3 is the one store that survives it.

WHICH CASE IS BEING SERVED is a request-scoped fact, not a global one: one
process can serve a scheduled wake on one family's case while a parent adds a
note to another's. So the current case id is a :class:`contextvars.ContextVar`
set at the top of an invocation and reset at its end, and the data seam in
:mod:`minutes.tools` reads it. Nothing else in the engine changes — every tool
still calls ``load_case_record()`` with no arguments and gets the case the
invocation is about, because a context variable survives the thread hops
Strands makes (its event loop and its tool threads both run under a copied
context).

THE SAMPLE CASE IS READ-ONLY. :data:`DEFAULT_CASE_ID` names the synthetic
demo case, which is served from ``fixtures/`` and never written: a parent who
pastes an IEP without naming a case must not overwrite the one case every
demo and every test relies on, and a store must not be able to hold a file
that shadows it.

Events are stored WITHOUT their attribution and it is re-derived on every
read, exactly as :func:`minutes.correspondence.load_cached_events` does for
the fixture: attribution is a rule over a document's own words, and freezing
it beside the reading would let a tightened rule reach new notes but not the
ones already on file.
"""

from __future__ import annotations

import contextvars
import json
import os
import re
import tempfile
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import BEDROCK_REGION
from .correspondence import DERIVED_FIELDS
from .models import Correspondence, IEPLedger, ServiceEvent

__all__ = [
    "CASE_ID_PATTERN",
    "DEFAULT_CASE_ID",
    "SAMPLE_CASE_IDS",
    "CaseStore",
    "DirCaseStore",
    "S3CaseStore",
    "case_store",
    "current_case_id",
    "is_sample",
    "reset_current_case",
    "set_current_case",
    "validate_case_id",
]

# The sample case. A real deployment names the case in the payload; a payload
# that names nothing gets this, and this is never written.
DEFAULT_CASE_ID = "maya-demo"

# Every name the sample answers to. "maya" is the fixture's own name and the
# default the engine's data seam has always used; "maya-demo" is the id the
# application exposes. Both read the same committed fixtures.
SAMPLE_CASE_IDS = frozenset({DEFAULT_CASE_ID, "maya"})

CASE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{7,63}$")
"""What a case id may look like.

It becomes a directory name, an S3 key segment, a session id and a Cedar
principal, so it is kept to the characters every one of those accepts. The
minimum length keeps ids from colliding with the short names people type
into a demo ("test", "maya").
"""

# Mirrors app.py: the session directory on a laptop, and the bucket the
# caseworker sessions live in on the runtime. Both read the same variables so
# that one deployment setting moves everything a case owns together.
STATE_DIR = Path(os.environ.get("MINUTES_STATE_DIR", Path(tempfile.gettempdir()) / "minutes-state"))
STATE_BUCKET = os.environ.get("MINUTES_SESSION_BUCKET") or None

CASE_DATA_PREFIX = "data/"
"""Where cases live in the bucket. The caseworker sessions are under ``cases/``
(``MINUTES_SESSION_PREFIX``); a case's own files go beside them, not among them,
so a session listing never confuses a ledger for a session."""


# ---------------------------------------------------------------------------
# Which case this invocation is about.
# ---------------------------------------------------------------------------

current_case_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "minutes_current_case_id", default=None
)


def set_current_case(case_id: str) -> contextvars.Token:
    """Name the case every ``load_case_record()`` in this context will read.

    Returns the token :func:`reset_current_case` needs. Set it in a
    ``try``/``finally`` around the invocation: a case id that outlives its
    request is the next request reading the wrong family's file.
    """
    return current_case_id.set(validate_case_id(case_id))


def reset_current_case(token: contextvars.Token) -> None:
    current_case_id.reset(token)


def validate_case_id(case_id: Any) -> str:
    """The id as a string, or a ValueError naming the rule it broke."""
    if not isinstance(case_id, str) or not CASE_ID_PATTERN.fullmatch(case_id):
        raise ValueError(
            f"case_id {case_id!r} is not valid; use 8 to 64 letters, digits, '-' or '_', "
            "starting with a letter or digit"
        )
    return case_id


def is_sample(case_id: str | None) -> bool:
    """Whether an id names the read-only synthetic case."""
    return case_id is None or case_id in SAMPLE_CASE_IDS


# ---------------------------------------------------------------------------
# The store.
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dump_events(events: list[ServiceEvent]) -> list[dict]:
    return [event.model_dump(mode="json", exclude=set(DERIVED_FIELDS)) for event in events]


class CaseStore(ABC):
    """One family's files, read and written as typed models.

    Subclasses implement four raw operations on named files; everything typed
    is built here so that a directory and a bucket cannot disagree about what a
    case looks like. Appends are read-modify-write under a process lock: two
    threads in one process (a background wake and a parent's note) serialize,
    and across processes the last writer wins — survivable, because nothing
    here is a letter and the next read re-derives everything derived.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()

    # -- raw ----------------------------------------------------------------

    @abstractmethod
    def _read(self, case_id: str, name: str) -> str | None:
        """The text of ``cases/<case_id>/<name>``, or None when absent."""

    @abstractmethod
    def _write(self, case_id: str, name: str, text: str) -> None:
        ...

    @abstractmethod
    def exists(self, case_id: str) -> bool:
        """Whether a ledger has been written for this case."""

    def list_ids(self) -> list[str]:  # optional; a store that cannot list says so
        raise NotImplementedError(f"{type(self).__name__} cannot list cases")

    # -- typed ----------------------------------------------------------------

    def _refuse_sample(self, case_id: str) -> None:
        if is_sample(case_id):
            raise ValueError("the sample case is read-only")

    def read_ledger(self, case_id: str) -> IEPLedger | None:
        text = self._read(validate_case_id(case_id), "ledger.json")
        return IEPLedger.model_validate_json(text) if text is not None else None

    def write_ledger(self, case_id: str, ledger: IEPLedger) -> None:
        self._refuse_sample(case_id)
        validate_case_id(case_id)
        with self._lock:
            self._write(case_id, "ledger.json", ledger.model_dump_json(indent=2))

    def read_correspondence(self, case_id: str) -> list[Correspondence]:
        text = self._read(validate_case_id(case_id), "correspondence.json")
        return [Correspondence.model_validate(raw) for raw in json.loads(text)] if text else []

    def append_correspondence(self, case_id: str, items: list[Correspondence]) -> None:
        self._refuse_sample(case_id)
        validate_case_id(case_id)
        with self._lock:
            held = self.read_correspondence(case_id)
            taken = {item.item_id for item in held}
            for item in items:
                if item.item_id in taken:
                    raise ValueError(f"correspondence item {item.item_id!r} is already on this case")
                taken.add(item.item_id)
            payload = [item.model_dump(mode="json") for item in [*held, *items]]
            self._write(case_id, "correspondence.json", json.dumps(payload, indent=2))

    def read_events(self, case_id: str) -> list[ServiceEvent]:
        """The stored events, attribution NOT yet derived — see :mod:`minutes.tools`."""
        text = self._read(validate_case_id(case_id), "events.json")
        return [ServiceEvent.model_validate(raw) for raw in json.loads(text)] if text else []

    def append_events(self, case_id: str, events: list[ServiceEvent]) -> None:
        self._refuse_sample(case_id)
        validate_case_id(case_id)
        with self._lock:
            held = self.read_events(case_id)
            self._write(case_id, "events.json", json.dumps(_dump_events([*held, *events]), indent=2))

    def read_meta(self, case_id: str) -> dict:
        text = self._read(validate_case_id(case_id), "meta.json")
        return json.loads(text) if text else {}

    def write_meta(self, case_id: str, meta: dict) -> None:
        self._refuse_sample(case_id)
        validate_case_id(case_id)
        with self._lock:
            self._write(case_id, "meta.json", json.dumps(meta, indent=2))

    def touch(self, case_id: str, **fields: Any) -> dict:
        """Stamp ``updated`` (and ``created`` the first time), merging ``fields``."""
        with self._lock:
            meta = self.read_meta(case_id)
            meta.setdefault("created", _now())
            meta.setdefault("source", "pasted")
            meta.update(fields)
            meta["updated"] = _now()
            self.write_meta(case_id, meta)
            return meta


class DirCaseStore(CaseStore):
    """Cases as directories under ``root``."""

    def __init__(self, root: Path):
        super().__init__()
        self.root = Path(root)

    def _path(self, case_id: str, name: str) -> Path:
        return self.root / case_id / name

    def _read(self, case_id: str, name: str) -> str | None:
        path = self._path(case_id, name)
        return path.read_text(encoding="utf-8") if path.exists() else None

    def _write(self, case_id: str, name: str, text: str) -> None:
        path = self._path(case_id, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Written beside and renamed over: a reader never sees half a ledger.
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)

    def exists(self, case_id: str) -> bool:
        return self._path(validate_case_id(case_id), "ledger.json").exists()

    def list_ids(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(
            child.name
            for child in self.root.iterdir()
            if child.is_dir() and CASE_ID_PATTERN.fullmatch(child.name) and (child / "ledger.json").exists()
        )


class S3CaseStore(CaseStore):
    """Cases as objects under ``s3://<bucket>/<prefix>cases/<case_id>/``.

    The client is created on first use, never on construction: importing this
    module, building a store, or serving the sample case must not open a
    connection or need credentials, and a test constructs one with a stand-in
    client and never reaches the network.
    """

    def __init__(self, bucket: str, prefix: str = CASE_DATA_PREFIX, region_name: str = BEDROCK_REGION):
        super().__init__()
        self.bucket = bucket
        self.prefix = prefix if not prefix or prefix.endswith("/") else prefix + "/"
        self.region_name = region_name
        self._client: Any = None

    @property
    def client(self) -> Any:
        if self._client is None:
            import boto3

            self._client = boto3.client("s3", region_name=self.region_name)
        return self._client

    def _key(self, case_id: str, name: str) -> str:
        return f"{self.prefix}cases/{case_id}/{name}"

    @staticmethod
    def _is_missing(error: Exception) -> bool:
        code = str(getattr(error, "response", {}).get("Error", {}).get("Code", ""))
        return code in {"NoSuchKey", "NotFound", "404"}

    def _read(self, case_id: str, name: str) -> str | None:
        from botocore.exceptions import ClientError

        try:
            body = self.client.get_object(Bucket=self.bucket, Key=self._key(case_id, name))["Body"]
        except ClientError as error:
            if self._is_missing(error):
                return None
            raise
        raw = body.read()
        return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)

    def _write(self, case_id: str, name: str, text: str) -> None:
        self.client.put_object(
            Bucket=self.bucket,
            Key=self._key(case_id, name),
            Body=text.encode("utf-8"),
            ContentType="application/json",
        )

    def exists(self, case_id: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self.client.head_object(Bucket=self.bucket, Key=self._key(validate_case_id(case_id), "ledger.json"))
        except ClientError as error:
            if self._is_missing(error):
                return False
            raise
        return True

    def list_ids(self) -> list[str]:
        root = f"{self.prefix}cases/"
        ids: list[str] = []
        token: str | None = None
        while True:
            kwargs: dict[str, Any] = {"Bucket": self.bucket, "Prefix": root, "Delimiter": "/"}
            if token:
                kwargs["ContinuationToken"] = token
            page = self.client.list_objects_v2(**kwargs)
            for common in page.get("CommonPrefixes") or []:
                name = common["Prefix"][len(root) :].rstrip("/")
                if CASE_ID_PATTERN.fullmatch(name):
                    ids.append(name)
            if not page.get("IsTruncated"):
                return sorted(ids)
            token = page.get("NextContinuationToken")


# One S3 store per bucket for the life of the process: the client behind it is
# expensive and thread-safe, and the store's lock only serializes appends if
# every writer in the process shares it.
_s3_stores: dict[str, S3CaseStore] = {}


def case_store() -> CaseStore:
    """The store this deployment keeps cases in.

    With ``MINUTES_SESSION_BUCKET`` set, the bucket the caseworker sessions
    already live in, under ``data/``; otherwise a directory beside the
    session files under ``MINUTES_STATE_DIR``. Read at call time, not import
    time, so a test that moves the state directory moves the cases with it.
    """
    if STATE_BUCKET:
        store = _s3_stores.get(STATE_BUCKET)
        if store is None:
            store = _s3_stores[STATE_BUCKET] = S3CaseStore(STATE_BUCKET, prefix=CASE_DATA_PREFIX)
        return store
    return DirCaseStore(Path(STATE_DIR) / "cases")
