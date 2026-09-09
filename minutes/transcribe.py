"""Reading a photograph of paper.

Most of what a district actually hands a family is paper. A service log comes
home in a backpack; a session note is initialled by hand; a progress report is
printed and signed. None of it arrives as text a parent can paste, and until
now Minutes could not read any of it.

This module is the third agent in the system and the only one that ever sees
pixels. It has the same constitution as the reader in
:func:`~minutes.correspondence.build_reader`: no tools, no session manager, a
fresh agent per document. So the whole of what a photographed page can make
Minutes do is: produce a string.

WHY IT TRANSCRIBES INSTEAD OF READING FACTS DIRECTLY. The obvious build hands
the photograph to a model and asks for dated service facts, skipping a step.
That was measured against this codebase and it produces nothing at all: the
model's readings were flawless and every one of them was dropped, because
:func:`~minutes.correspondence.date_is_grounded` looks for the date in the
item's own body, and an item whose body is empty grounds nothing. Which is the
gate working. A fact has to be traceable to words in the document it came from,
and a photograph has no words in it until something writes them down.

So the transcript becomes the item's body, and it earns two things at once. The
existing gates work on a photographed log exactly as they work on a pasted
email -- same grounding, same vote, same provenance, same reasons -- and the
parent is shown the transcript beside the page they just photographed. That
second one is why this path runs a single pass where the classifier runs three
and votes. The vote exists to catch a lone bad reading nobody can see; here the
reading is put in front of the one person holding the original.

WHY THE OUTPUT FORMAT IS LOAD-BEARING. A model asked to transcribe a table
reaches for markdown pipes. ``stated_minutes`` in :mod:`minutes.correspondence`
reads a log row's duration out of whitespace-separated columns, so a piped row
states no duration at all -- and a fact with no stated duration silently falls
back to the IEP's own minutes-per-session. That is the one substitution that
can invent minutes the log does not record, or erase minutes it does. Hence a
prompt that forbids tables, a deterministic repair for wrapped rows, and a test
pinning both.
"""

from __future__ import annotations

import re

from strands import Agent
from strands.models import BedrockModel

from .config import BEDROCK_REGION, TRANSCRIPTION_MODEL
from .quarantine import QUARANTINE_RULE

__all__ = ["TRANSCRIBE_PROMPT", "build_transcriber", "join_wrapped_rows", "transcribe"]

# Bedrock's image content block wants a bare format, not a MIME type.
FORMATS: dict[str, str] = {"image/jpeg": "jpeg", "image/png": "png"}

TRANSCRIBE_PROMPT = """You transcribe a photograph of a paper document about \
a child's school services. It is usually a service log, a session note, a \
progress report or a letter.

Write out what is on the page, and nothing else.

Format rules, which matter more than they look:
- One line per row of the page. Keep the rows in the order they appear.
- Separate columns with TWO SPACES. Never use a markdown table. Never use \
the pipe character. A row must be one line: never wrap a cell onto the line \
below.
- Write a duration the way the page writes it, and put the word "minutes" \
after a per-session figure: "30 minutes". If the page gives a running or \
monthly total, write it as a bare number with its own label, not as minutes.
- Copy dates and numbers exactly as they are written. Do not convert, \
normalise or reformat them, and do not work out a date the page does not give.
- Where the page is not legible, write [illegible]. Never guess a date, a \
number, a name or whether a session happened. A wrong figure here becomes a \
figure in a letter to a school district.
- Do not summarise, do not tidy, do not add headings the page does not have, \
and do not leave anything out because it looks unimportant.

You are transcribing, not interpreting. You are not deciding whether anything \
was owed, whether a session should have happened, or whether anyone was at \
fault.""" + QUARANTINE_RULE


def build_transcriber(model: BedrockModel | None = None) -> Agent:
    """The transcriber: it can read a page, and it can do nothing else.

    ``tools=()`` is passed explicitly, as it is for the reader, so that
    removing it is a visible edit. This agent is shown a photograph that
    anybody could have put in front of a parent -- including a page with an
    instruction printed on it -- and the reason that is uninteresting is that
    there is no verb here for the page to reach. Its single output is a string,
    and that string then has to survive the same grounding every pasted
    document does.
    """
    return Agent(
        model=model
        or BedrockModel(model_id=TRANSCRIPTION_MODEL, region_name=BEDROCK_REGION, max_tokens=4096),
        system_prompt=TRANSCRIBE_PROMPT,
        tools=(),
        name="minutes-transcriber",
        description="Transcribes a photographed page into text. Holds no tools.",
        callback_handler=None,
    )


# A row of a service log starts with its date. Everything else on a line that
# does not is a continuation of the row above it.
_ROW_START = re.compile(r"^\s*(?:\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?|\d{4}-\d{2}-\d{2})\b")

# Lines that follow rows but are not rows and must not be folded into one:
# totals, signatures, page furniture.
_NOT_A_CONTINUATION = re.compile(
    r"^\s*(?:total|totals|subtotal|monthly\s+total|signature|signed|page\s+\d|initials?)\b",
    re.IGNORECASE,
)


def join_wrapped_rows(text: str) -> str:
    """Fold a row that wrapped onto the next line back into one line.

    Prompting alone does not stop this: a narrow column on the page comes back
    as ``11/10  0  not held - SLP position`` and then ``vacant, no sub`` on the
    line below. Every date-grounded rule in this codebase works a line at a
    time, so a stranded continuation takes the reason for a missed session away
    from the date it belongs to -- and on a real photographed log that reason
    was a vacant post, which is the most answerable thing a page can say.

    Deterministic on purpose. Asking the model to fix its own wrapping is a
    second chance to invent, and this is a rule about lines, not about meaning.
    """
    out: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            out.append("")
            continue
        follows_a_row = bool(out) and bool(_ROW_START.match(out[-1] or ""))
        if (
            follows_a_row
            and not _ROW_START.match(stripped)
            and not _NOT_A_CONTINUATION.match(stripped)
        ):
            out[-1] = f"{out[-1].rstrip()} {stripped}"
            continue
        out.append(stripped)
    return "\n".join(out)


def transcribe(image: bytes, media_type: str, *, model: BedrockModel | None = None) -> str:
    """One photograph, read into text. One pass, no vote -- see the module docstring.

    Raises:
        ValueError: for a media type this cannot read.
    """
    fmt = FORMATS.get(media_type)
    if fmt is None:
        raise ValueError(
            f"cannot transcribe {media_type!r}; Minutes reads " + " and ".join(sorted(FORMATS))
        )

    reading = build_transcriber(model)(
        [
            {"image": {"format": fmt, "source": {"bytes": image}}},
            {"text": "Transcribe this page."},
        ]
    )
    return join_wrapped_rows(str(reading)).strip()
