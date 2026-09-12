"""Telling the parent a decision is waiting, without becoming the decision.

Minutes wakes itself once a week and, most weeks, finds nothing that needs a
person. The weeks it does — a shortfall crossed the line into a letter, a
statutory clock is about to run out — are exactly the weeks a parent who is
already out of hours will not happen to open the app. So the scheduled wake
that raised the decision has to reach out, or the whole claim of a background
agent that surfaces only for real decisions is only true for someone sitting
and watching.

This module is that reach. Two rules shape all of it.

**It notifies about a decision; it is never the decision.** The email carries a
link to the decision card and nothing that acts. No approve link, no decline
link, no one-click anything. A mail provider's scanner fetches every link in a
message the instant it arrives, so an approve-by-link would release a letter to
a school district that no human ever read — and it would bypass the one rule
the whole product is built on, that a parent sees the compiled letter before it
goes. The strongest thing this email can do is get someone to open the app.

**It says a letter is waiting; it never says what the letter says.** The
compiled letter names the child, the services, the missed minutes. That belongs
behind the app, shown to the parent, not sitting in an inbox and on a mail
provider's servers. The email carries the decision's one-line title and its
urgency, and stops there.

The content is built deterministically from the wake result the engine already
produced — there is no model here, and there is nothing to get wrong that the
cards did not already decide. Sending is a thin SES wrapper kept behind a small
surface so the whole of this can be tested without an account.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .models import DecisionCard, Urgency

__all__ = [
    "Notification",
    "SesMailer",
    "notification_for",
    "recipient_is_ready",
]


# How each urgency reads to a parent skimming an inbox. The enum's own names are
# for the engine; these are for a person deciding whether to stop what they are
# doing.
_URGENCY_WORD: dict[Urgency, str] = {
    Urgency.DEADLINE_IMMINENT: "a deadline is close",
    Urgency.TIME_SENSITIVE: "time-sensitive",
    Urgency.ROUTINE: "when you can",
}


@dataclass(frozen=True)
class Notification:
    """One email, rendered three ways. ``None`` when nothing needs sending.

    ``subject`` and ``text`` are always present; ``html`` is the same content
    for a client that prefers it. A caller that has a :class:`Notification` at
    all has something worth a parent's attention — the decision to send nothing
    is made by :func:`notification_for` returning ``None``, never by handing
    back an empty one.
    """

    subject: str
    text: str
    html: str


def _card_link(app_url: str, case_id: str) -> str:
    """The This Week screen for this case. Where the letter can be read and released.

    Deliberately the route, not a per-card deep link with an action in it: the
    parent lands on the same screen they would have opened themselves, sees the
    letter in full, and decides there. Nothing in this URL decides anything.
    """
    return f"{app_url.rstrip('/')}/#/case/{case_id}"


def notification_for(
    *,
    case_id: str,
    student_alias: str,
    cards: list[DecisionCard],
    app_url: str,
    today: date,
) -> Notification | None:
    """Render the email for a wake, or ``None`` when the wake needs no one.

    ``cards`` are the decisions the wake newly raised — the engine's own
    ``new_cards``, already suppressed against everything a parent has been shown
    before, so an email goes out for a genuinely new decision and not for one
    that has merely persisted. An empty list is the quiet-week answer and
    returns ``None``: silence is the correct output most weeks, and a
    notification that fired on nothing would train a parent to ignore the ones
    that matter.
    """
    if not cards:
        return None

    n = len(cards)
    noun = "decision" if n == 1 else "decisions"
    subject = f"Minutes: {n} {noun} about {student_alias}'s services need you"

    link = _card_link(app_url, case_id)
    lines = [
        f"Minutes checked {student_alias}'s IEP services and found "
        f"{n} {noun} that need you.",
        "",
    ]
    for card in cards:
        word = _URGENCY_WORD.get(card.urgency, "")
        tag = f" ({word})" if word else ""
        lines.append(f"- {card.title}{tag}")
        if card.why_now:
            lines.append(f"  {card.why_now}")
    lines += [
        "",
        f"Open the case to read what Minutes compiled and decide what to do:",
        f"  {link}",
        "",
        "Nothing has been sent. Minutes never sends anything to a school "
        "without you: it compiles a letter and waits for you to read it and "
        "release it. This email is only telling you one is ready.",
    ]
    text = "\n".join(lines)

    return Notification(subject=subject, text=text, html=_html(student_alias, cards, link, n, noun))


def _html(student_alias: str, cards: list[DecisionCard], link: str, n: int, noun: str) -> str:
    """The same words as ``text``, laid out for a client that renders HTML.

    Plain and self-contained: inline styles only, no remote anything, no image
    that a client could use to confirm the address is live. The button is an
    ordinary link to the case; it carries no action, exactly as the text link
    carries none.
    """
    items = "".join(
        f'<li style="margin:0 0 10px"><b>{_esc(card.title)}</b>'
        + (f' <span style="color:#8B2E27">({_esc(_URGENCY_WORD[card.urgency])})</span>' if card.urgency in _URGENCY_WORD else "")
        + (f'<br><span style="color:#4a4a4a">{_esc(card.why_now)}</span>' if card.why_now else "")
        + "</li>"
        for card in cards
    )
    return f"""\
<div style="font-family:Georgia,serif;color:#1B241E;max-width:520px;margin:0 auto;padding:24px">
  <p style="font-size:16px;margin:0 0 16px">Minutes checked <b>{_esc(student_alias)}</b>'s IEP
  services and found {n} {noun} that need you.</p>
  <ul style="font-size:15px;line-height:1.5;padding-left:20px;margin:0 0 20px">{items}</ul>
  <p style="margin:0 0 24px">
    <a href="{_esc(link)}" style="display:inline-block;background:#1B241E;color:#fff;
    text-decoration:none;padding:12px 20px;font-family:system-ui,sans-serif;font-size:15px">
    Open the case</a>
  </p>
  <p style="font-size:13px;color:#4a4a4a;line-height:1.5;margin:0">
    Nothing has been sent. Minutes never sends anything to a school without you:
    it compiles a letter and waits for you to read it and release it. This email
    is only telling you one is ready.</p>
</div>"""


def _esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def recipient_is_ready(status: str | None) -> bool:
    """True only when SES will actually deliver to this address.

    SES reports an identity as ``SUCCESS`` once it is verified and sendable.
    Anything else — ``PENDING`` while a parent has not yet clicked the link SES
    mailed them, ``FAILED``, or no identity at all — means a send would be
    rejected, so the caller holds off and says so rather than trying and
    logging an error a parent never sees.
    """
    return status == "SUCCESS"


class SesMailer:
    """The one place an email actually leaves. Everything else here is content.

    Kept deliberately thin, and behind a client the caller injects, so the whole
    of the interesting logic — what to say, whether to say anything, whether the
    address can even receive it — is decided in plain functions that never touch
    an account. In sandbox, SES delivers only to identities it has verified,
    which is why :meth:`verify` and :meth:`status` exist: a parent's own inbox
    is verified once, by clicking a link SES sends them, and from then on the
    weekly wake can reach it.
    """

    def __init__(self, client, sender: str):
        self._client = client
        self._sender = sender

    def verify(self, email: str) -> None:
        """Ask SES to send its verification email to this address. Idempotent.

        SES raises ``AlreadyExistsException`` if the identity is already on the
        account, which is not an error here — it means a previous call already
        started the parent down this path — so it is swallowed.
        """
        try:
            self._client.create_email_identity(EmailIdentity=email)
        except Exception as exc:  # noqa: BLE001 -- narrowed by name below
            if "AlreadyExists" not in type(exc).__name__:
                raise

    def status(self, email: str) -> str | None:
        """SES's verification status for this address, or ``None`` if unknown.

        SES v2 reports ``VerifiedForSendingStatus`` as a boolean, so it is
        mapped to the ``SUCCESS`` / ``PENDING`` vocabulary
        :func:`recipient_is_ready` reads. An identity SES has never heard of
        returns ``None`` (its own ``NotFoundException``), which reads as "not
        started" rather than as an error.
        """
        try:
            reply = self._client.get_email_identity(EmailIdentity=email)
        except Exception as exc:  # noqa: BLE001
            if "NotFound" in type(exc).__name__:
                return None
            raise
        return "SUCCESS" if reply.get("VerifiedForSendingStatus") else "PENDING"

    def send(self, to: str, notification: Notification) -> str:
        """Send one rendered notification. Returns the SES message id."""
        reply = self._client.send_email(
            FromEmailAddress=self._sender,
            Destination={"ToAddresses": [to]},
            Content={
                "Simple": {
                    "Subject": {"Data": notification.subject},
                    "Body": {
                        "Text": {"Data": notification.text},
                        "Html": {"Data": notification.html},
                    },
                }
            },
        )
        return reply.get("MessageId", "")
