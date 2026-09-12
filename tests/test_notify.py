"""The decision notice: when it goes, what it says, and what it can never do.

The weekly wake runs on its own and, most weeks, finds nothing that needs a
person. The weeks it finds a decision are exactly the weeks a parent is not
sitting in front of the app, so the wake has to reach out — or "a background
agent that surfaces only for real decisions" is only true for someone watching.

Two invariants hold everything here. The email links to the decision and
never acts on it: a mail provider fetches every link the moment a message
lands, so an approve-by-link would release a letter no human read. And it
says a letter is waiting without saying what the letter says: the compiled
letter names the child and the missed minutes, and that belongs behind the
app, not in an inbox.

No test here reaches SES or a model. The mailer is a fake that records what
it was asked to do.
"""

from datetime import date
from types import SimpleNamespace

import pytest

import app
from minutes import cases
from minutes.correspondence import load_cached_events, load_correspondence
from minutes.models import DecisionCard, Letter, LetterKind, Urgency
from minutes.notify import SesMailer, notification_for, recipient_is_ready
from minutes.tools import CORRESPONDENCE_FIXTURE, load_case_record

APP = "https://minutes.example.test/"
CASE = "family-notify"


# ---------------------------------------------------------------------------
# Rendering.
# ---------------------------------------------------------------------------


def _card(title: str, why: str = "", urgency: Urgency = Urgency.TIME_SENSITIVE, letter: str | None = None) -> DecisionCard:
    draft = None
    if letter is not None:
        draft = Letter(
            kind=LetterKind.SHORTFALL_NOTICE,
            subject="Service delivery reconciliation",
            body=letter,
            citations=[],
            legal_basis=[],
            disclaimer="",
        )
    return DecisionCard(
        card_id=title.lower().replace(" ", "-")[:24],
        title=title,
        why_now=why,
        facts=[],
        recommended_action="send",
        urgency=urgency,
        draft=draft,
    )


def test_a_quiet_week_sends_nothing():
    """Silence is the correct output most weeks. A notice that fired on nothing
    would train a parent to ignore the ones that matter."""
    assert notification_for(case_id=CASE, student_alias="Maya R.", cards=[], app_url=APP, today=date(2026, 12, 1)) is None


def test_the_notice_names_the_decisions_and_the_child_by_alias():
    n = notification_for(
        case_id=CASE,
        student_alias="Maya R.",
        cards=[_card("Speech therapy is 90 minutes short", "A records request went unanswered."), _card("Annual review is due in 12 days", urgency=Urgency.DEADLINE_IMMINENT)],
        app_url=APP,
        today=date(2026, 12, 1),
    )
    assert n.subject == "Minutes: 2 decisions about Maya R.'s services need you"
    assert "Speech therapy is 90 minutes short" in n.text
    assert "A records request went unanswered." in n.text
    assert "(a deadline is close)" in n.text
    assert "Maya R." in n.html and "Annual review is due in 12 days" in n.html


def test_one_decision_reads_in_the_singular():
    n = notification_for(case_id=CASE, student_alias="Maya R.", cards=[_card("One thing")], app_url=APP, today=date(2026, 12, 1))
    assert n.subject.startswith("Minutes: 1 decision about")
    assert "found 1 decision that need you" in n.text


def test_the_link_opens_the_case_and_decides_nothing():
    """The load-bearing rule. A link a scanner can follow must not be an answer."""
    n = notification_for(case_id=CASE, student_alias="Maya R.", cards=[_card("One thing")], app_url=APP, today=date(2026, 12, 1))

    import re

    link = f"{APP.rstrip('/')}/#/case/{CASE}"
    for body in (n.text, n.html):
        urls = set(re.findall(r"https?://[^\s\"'<>]+", body))
        assert urls == {link}, f"exactly one link, the case route, and nothing else: {urls}"
    assert "?" not in link, "no query string: the link carries no answer, action or token"
    assert "key=" not in n.text and "key=" not in n.html, "the demo key never rides in an email"


def test_the_notice_never_carries_the_letter():
    """What the letter says belongs behind the app, shown to the parent."""
    letter = "Dear Director, the district owes Maya R. 90 minutes of speech therapy for October."
    n = notification_for(case_id=CASE, student_alias="Maya R.", cards=[_card("Shortfall", letter=letter)], app_url=APP, today=date(2026, 12, 1))
    assert "Dear Director" not in n.text and "owes" not in n.text
    assert "Dear Director" not in n.html


def test_the_notice_says_plainly_that_nothing_was_sent():
    n = notification_for(case_id=CASE, student_alias="Maya R.", cards=[_card("One thing")], app_url=APP, today=date(2026, 12, 1))
    assert "Nothing has been sent." in n.text
    assert "never sends anything to a school without you" in n.text


def test_html_escapes_what_the_engine_wrote():
    n = notification_for(case_id=CASE, student_alias="<Maya>", cards=[_card("A & B <c>")], app_url=APP, today=date(2026, 12, 1))
    assert "&lt;Maya&gt;" in n.html and "A &amp; B &lt;c&gt;" in n.html
    assert "<Maya>" not in n.html


@pytest.mark.parametrize(("status", "ready"), [("SUCCESS", True), ("PENDING", False), ("FAILED", False), (None, False)])
def test_only_a_verified_address_is_ready(status, ready):
    assert recipient_is_ready(status) is ready


# ---------------------------------------------------------------------------
# The SES wrapper, against a fake client.
# ---------------------------------------------------------------------------


class _AlreadyExistsException(Exception):
    pass


class _NotFoundException(Exception):
    pass


class _FakeSes:
    def __init__(self, verified: bool = True, known: bool = True):
        self.verified, self.known = verified, known
        self.created: list[str] = []
        self.sent: list[dict] = []

    def create_email_identity(self, EmailIdentity):
        if EmailIdentity in self.created:
            raise _AlreadyExistsException("already there")
        self.created.append(EmailIdentity)
        return {}

    def get_email_identity(self, EmailIdentity):
        if not self.known:
            raise _NotFoundException("no such identity")
        return {"VerifiedForSendingStatus": self.verified}

    def send_email(self, **kwargs):
        self.sent.append(kwargs)
        return {"MessageId": "msg-0001"}


def test_verify_is_idempotent_across_an_existing_identity():
    ses = _FakeSes()
    mailer = SesMailer(ses, "minutes@example.test")
    mailer.verify("parent@example.test")
    mailer.verify("parent@example.test")  # SES says AlreadyExists; swallowed
    assert ses.created == ["parent@example.test"]


def test_status_maps_ses_booleans_to_words():
    assert SesMailer(_FakeSes(verified=True), "m@x").status("p@x") == "SUCCESS"
    assert SesMailer(_FakeSes(verified=False), "m@x").status("p@x") == "PENDING"
    assert SesMailer(_FakeSes(known=False), "m@x").status("p@x") is None


def test_send_posts_the_rendered_notice_from_the_configured_sender():
    ses = _FakeSes()
    n = notification_for(case_id=CASE, student_alias="Maya R.", cards=[_card("One thing")], app_url=APP, today=date(2026, 12, 1))
    message_id = SesMailer(ses, "minutes@example.test").send("parent@example.test", n)

    assert message_id == "msg-0001"
    (call,) = ses.sent
    assert call["FromEmailAddress"] == "minutes@example.test"
    assert call["Destination"] == {"ToAddresses": ["parent@example.test"]}
    assert call["Content"]["Simple"]["Subject"]["Data"] == n.subject
    assert call["Content"]["Simple"]["Body"]["Text"]["Data"] == n.text
    assert call["Content"]["Simple"]["Body"]["Html"]["Data"] == n.html


# ---------------------------------------------------------------------------
# Inside a wake.
# ---------------------------------------------------------------------------


class _FakeMailer:
    """Records what the app asked of it. ``ready`` decides what status() reports."""

    def __init__(self, ready: bool = True, fail_send: bool = False):
        self.ready, self.fail_send = ready, fail_send
        self.verified: list[str] = []
        self.sent: list[tuple[str, str]] = []

    def verify(self, email):
        self.verified.append(email)

    def status(self, email):
        return "SUCCESS" if self.ready else "PENDING"

    def send(self, to, notification):
        if self.fail_send:
            raise RuntimeError("SES is down")
        self.sent.append((to, notification.subject))
        return "msg-0001"


def _invoke(payload: dict) -> dict:
    return app.invoke(payload, SimpleNamespace(session_id=None))


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(app, "STATE_BUCKET", None)
    monkeypatch.setattr(cases, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(cases, "STATE_BUCKET", None)
    monkeypatch.setattr(cases, "_s3_stores", {})
    monkeypatch.setattr(app, "_caseworkers", {})
    monkeypatch.setattr(app, "_runs", {})
    monkeypatch.setattr(app, "_run_threads", {})
    monkeypatch.setattr(app, "_inflight", {})
    monkeypatch.setattr(app.Caseworker, "ask", lambda self, prompt: pytest.fail("a test reached the model"))
    yield
    app._join_background(timeout=15)


@pytest.fixture
def mailer(monkeypatch):
    """Notifications configured, with a fake behind them."""
    fake = _FakeMailer()
    monkeypatch.setattr(app, "NOTIFY_FROM", "minutes@example.test")
    monkeypatch.setattr(app, "APP_URL", APP)
    monkeypatch.setattr(app, "_mailer", lambda: fake)
    return fake


@pytest.fixture
def stored_case():
    """A writable copy of the sample, so a wake on 1 December finds two decisions."""
    store = cases.case_store()
    store.write_ledger(CASE, load_case_record().ledger)
    store.append_correspondence(CASE, load_correspondence(CORRESPONDENCE_FIXTURE))
    store.append_events(CASE, load_cached_events(CORRESPONDENCE_FIXTURE))
    store.touch(CASE, student_alias="Maya R.")
    return CASE


def _wake(case_id: str, today: str) -> dict:
    return _invoke({"action": "wake", "case_id": case_id, "today": today, "ask_parent": False})


def test_a_wake_that_raises_a_decision_emails_the_parent_once(mailer, stored_case):
    _invoke({"action": "set_notify_email", "case_id": stored_case, "email": "Parent@Example.test"})

    out = _wake(stored_case, "2026-12-01")

    assert out["status"] == "done" and len(out["new_cards"]) == 2
    assert out["notification"] == {"sent": True, "to": "parent@example.test", "message_id": "msg-0001", "decisions": 2}
    assert mailer.sent == [("parent@example.test", "Minutes: 2 decisions about Maya R.'s services need you")]

    trail = _invoke({"action": "audit", "case_id": stored_case})["audit"]
    notice = [e for e in trail if e["actor"] == "notify"]
    assert len(notice) == 1 and notice[0]["action"] == "emailed a decision notice"
    assert "cannot approve or decline anything" in notice[0]["detail"]


def test_a_decision_already_shown_is_not_emailed_again(mailer, stored_case):
    """The second wake suppresses the cards the first one raised; so does the notice."""
    _invoke({"action": "set_notify_email", "case_id": stored_case, "email": "parent@example.test"})
    _wake(stored_case, "2026-12-01")
    out = _wake(stored_case, "2026-12-02")

    assert out["new_cards"] == [] and out["suppressed"] == 2
    assert out["notification"] is None
    assert len(mailer.sent) == 1, "one decision, one email"


def test_a_quiet_wake_sends_nothing(mailer, stored_case):
    _invoke({"action": "set_notify_email", "case_id": stored_case, "email": "parent@example.test"})
    out = _wake(stored_case, "2026-09-01")
    assert out["quiet"] is True and out["notification"] is None
    assert mailer.sent == []


def test_no_address_on_file_means_no_email_and_no_error(mailer, stored_case):
    out = _wake(stored_case, "2026-12-01")
    assert len(out["new_cards"]) == 2 and out["notification"] is None
    assert mailer.sent == []


def test_an_unconfirmed_address_holds_the_notice_and_says_so_in_the_trail(monkeypatch, stored_case):
    fake = _FakeMailer(ready=False)
    monkeypatch.setattr(app, "NOTIFY_FROM", "minutes@example.test")
    monkeypatch.setattr(app, "APP_URL", APP)
    monkeypatch.setattr(app, "_mailer", lambda: fake)
    set_out = _invoke({"action": "set_notify_email", "case_id": stored_case, "email": "parent@example.test"})
    assert set_out["ready"] is False and "confirmation email" in set_out["note"]

    out = _wake(stored_case, "2026-12-01")

    assert out["notification"] == {"held": True, "to": "parent@example.test", "verification": "PENDING", "decisions": 2}
    assert fake.sent == []
    trail = _invoke({"action": "audit", "case_id": stored_case})["audit"]
    held = [e for e in trail if e["action"] == "decision notice held"]
    assert len(held) == 1 and "has not confirmed itself with SES yet" in held[0]["detail"]


def test_a_failed_send_is_recorded_and_the_wake_still_returns_the_decision(monkeypatch, stored_case):
    """A week nobody was told about must not read like a quiet week -- and must
    not turn into a wake that did not happen either."""
    fake = _FakeMailer(fail_send=True)
    monkeypatch.setattr(app, "NOTIFY_FROM", "minutes@example.test")
    monkeypatch.setattr(app, "APP_URL", APP)
    monkeypatch.setattr(app, "_mailer", lambda: fake)
    _invoke({"action": "set_notify_email", "case_id": stored_case, "email": "parent@example.test"})

    out = _wake(stored_case, "2026-12-01")

    assert out["status"] == "done" and len(out["new_cards"]) == 2, "the decision is still returned"
    assert out["notification"]["error"].startswith("RuntimeError: SES is down")
    trail = _invoke({"action": "audit", "case_id": stored_case})["audit"]
    assert any(e["action"] == "decision notice failed" for e in trail)


def test_unconfigured_deployments_wake_exactly_as_before(stored_case):
    """No sender, no app URL: the laptop and CI. The wake neither sends nor complains."""
    assert app._mailer() is None
    cases.case_store().touch(stored_case, notify_email="parent@example.test")
    out = _wake(stored_case, "2026-12-01")
    assert len(out["new_cards"]) == 2 and out["notification"] is None


def test_notify_can_be_switched_off_for_one_wake(mailer, stored_case):
    _invoke({"action": "set_notify_email", "case_id": stored_case, "email": "parent@example.test"})
    out = _invoke({"action": "wake", "case_id": stored_case, "today": "2026-12-01", "ask_parent": False, "notify": False})
    assert len(out["new_cards"]) == 2 and out["notification"] is None
    assert mailer.sent == []


# ---------------------------------------------------------------------------
# The two actions.
# ---------------------------------------------------------------------------


def test_setting_an_address_asks_ses_to_confirm_it_and_stores_it_lowercased(mailer, stored_case):
    out = _invoke({"action": "set_notify_email", "case_id": stored_case, "email": "  Parent@Example.TEST "})
    assert out == {**out, "status": "done", "email": "parent@example.test", "verification": "SUCCESS", "ready": True}
    assert mailer.verified == ["parent@example.test"]
    assert cases.case_store().read_meta(stored_case)["notify_email"] == "parent@example.test"

    status = _invoke({"action": "notify_status", "case_id": stored_case})
    assert status == {"status": "done", "configured": True, "email": "parent@example.test", "verification": "SUCCESS", "ready": True}


def test_an_empty_address_clears_it(mailer, stored_case):
    _invoke({"action": "set_notify_email", "case_id": stored_case, "email": "parent@example.test"})
    out = _invoke({"action": "set_notify_email", "case_id": stored_case, "email": ""})
    assert out == {"status": "done", "email": None, "verification": None, "ready": False}
    assert cases.case_store().read_meta(stored_case)["notify_email"] is None
    assert _wake(stored_case, "2026-12-01")["notification"] is None


@pytest.mark.parametrize("bad", ["not-an-address", "a@b", "@example.test", "two words@x.y"])
def test_a_malformed_address_is_refused_before_ses_is_asked(mailer, stored_case, bad):
    out = _invoke({"action": "set_notify_email", "case_id": stored_case, "email": bad})
    assert out["status"] == "error" and "does not look like an address" in out["error"]
    assert mailer.verified == []


def test_the_sample_case_cannot_be_given_an_address(mailer):
    out = _invoke({"action": "set_notify_email", "case_id": "maya-demo", "email": "parent@example.test"})
    assert out["status"] == "error" and "read-only" in out["error"]
    assert _invoke({"action": "notify_status", "case_id": "maya-demo"})["email"] is None


def test_an_unknown_case_cannot_be_given_an_address(mailer):
    out = _invoke({"action": "set_notify_email", "case_id": "family-none", "email": "parent@example.test"})
    assert out["status"] == "error" and "ingest_iep creates a case" in out["error"]


def test_without_configuration_the_action_says_what_is_missing(stored_case):
    out = _invoke({"action": "set_notify_email", "case_id": stored_case, "email": "parent@example.test"})
    assert out["status"] == "error"
    assert "MINUTES_NOTIFY_FROM" in out["error"] and "MINUTES_APP_URL" in out["error"]
    assert _invoke({"action": "notify_status", "case_id": stored_case})["configured"] is False
