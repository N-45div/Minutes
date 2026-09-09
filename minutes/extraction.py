"""IEP -> ledger extraction.

The single place the LLM reads the IEP. Runs once per document; the
resulting ledger is cached and everything downstream is deterministic.

Two ways in, one call. A parent who has the IEP as text pastes it; a parent
who has it as the PDF the district emailed sends the file, and Bedrock reads
the document block directly -- rasterising each page, so a SCANNED IEP works
on the same path as a born-digital one with no OCR dependency and no separate
code path to keep honest. That was worth checking rather than assuming: the
obvious build extracts text locally with a PDF library and then has to refuse
the scans, which for a district-issued document is most of them.

What does NOT change is everything after: the model still returns a typed
ledger in which every obligation carries the verbatim sentence it came from,
and every figure downstream is arithmetic over that.
"""

from strands import Agent
from strands.models import BedrockModel

from .config import BEDROCK_REGION, EXTRACTION_MODEL
from .models import IEPLedger
from .quarantine import QUARANTINE_RULE

SYSTEM_PROMPT = """You extract facts from an IEP (Individualized Education \
Program) document into a structured ledger. Rules:
- Extract only what the document states. Never infer services, minutes, or \
dates that are not written.
- Every obligation, deadline, and accommodation must carry the verbatim \
source_quote it came from.
- Use the student's first name and last initial only as student_alias.
- Normalize service schedules exactly as written (e.g. '30 minutes, 2x per \
week' -> minutes_per_session=30, sessions_per_period=2, period=week).
- A 'daily' service uses period=day with sessions_per_period=1.
- The document may be a scan or a photograph of paper. Where a value is not \
legible, omit the whole item rather than guessing it. A ledger missing a \
service can be corrected by the parent who reads it; a ledger carrying an \
invented figure is a letter to a school district asserting something no \
document says.""" + QUARANTINE_RULE


def extract_ledger(
    iep_text: str | None = None,
    *,
    document: bytes | None = None,
    document_format: str = "pdf",
) -> IEPLedger:
    """Read one IEP into a typed ledger. Exactly one of text or document.

    The document arm hands Bedrock the file itself. Pages are rasterised on
    the service side, which is why a scanned IEP needs no OCR here and no
    branch anywhere downstream: both arms return the same ledger, and a
    caller cannot tell from the result which way the document came in.
    """
    if (iep_text is None) == (document is None):
        raise ValueError(
            "extract_ledger takes the IEP as text (iep_text=) or as a file "
            "(document=), not both and not neither"
        )

    agent = Agent(
        model=BedrockModel(model_id=EXTRACTION_MODEL, region_name=BEDROCK_REGION, max_tokens=4096),
        system_prompt=SYSTEM_PROMPT,
        callback_handler=None,
    )

    prompt = (
        [
            {"document": {"format": document_format, "name": "iep", "source": {"bytes": document}}},
            {"text": "Extract the obligations ledger from this IEP document."},
        ]
        if document is not None
        else f"Extract the obligations ledger from this IEP document:\n\n{iep_text}"
    )

    return _read_once_more_if_malformed(agent, prompt)


# A ledger that fails its own schema is not a ledger; it is a reading that did
# not happen. Reading the document again is a better answer to that than the
# alternative, which is telling a parent holding a forty-page PDF to go and
# paste the services pages by hand.
EXTRACTION_ATTEMPTS = 2


def _read_once_more_if_malformed(agent: Agent, prompt) -> IEPLedger:
    """Extract, retrying only the failure a second reading can actually fix.

    Observed on the deployed runtime and not reproducible locally: one run in
    several comes back with a service carrying no start date, which fails
    ``IEPLedger`` validation. It is the same shape of problem the classifier
    already handles by voting across passes -- a malformed reply is a pass that
    said nothing -- except that extraction has only one pass, so the analogue of
    a vote is to read again.

    Deliberately narrow. A page-count refusal, a credential failure and a
    throttle are all deterministic or costly to repeat, so they are re-raised
    on the spot; only a schema failure is retried, and only once.
    """
    last: Exception | None = None
    for _ in range(EXTRACTION_ATTEMPTS):
        try:
            return agent.structured_output(IEPLedger, prompt)
        except Exception as failure:
            if "validation error" not in str(failure).lower():
                raise
            last = failure
    raise last  # type: ignore[misc]
