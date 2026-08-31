"""IEP -> ledger extraction.

The single place the LLM reads the IEP. Runs once per document; the
resulting ledger is cached and everything downstream is deterministic.
"""

from strands import Agent
from strands.models import BedrockModel

from .config import BEDROCK_REGION, EXTRACTION_MODEL
from .models import IEPLedger

SYSTEM_PROMPT = """You extract facts from an IEP (Individualized Education \
Program) document into a structured ledger. Rules:
- Extract only what the document states. Never infer services, minutes, or \
dates that are not written.
- Every obligation, deadline, and accommodation must carry the verbatim \
source_quote it came from.
- Use the student's first name and last initial only as student_alias.
- Normalize service schedules exactly as written (e.g. '30 minutes, 2x per \
week' -> minutes_per_session=30, sessions_per_period=2, period=week).
- A 'daily' service uses period=day with sessions_per_period=1."""


def extract_ledger(iep_text: str) -> IEPLedger:
    agent = Agent(
        model=BedrockModel(model_id=EXTRACTION_MODEL, region_name=BEDROCK_REGION, max_tokens=4096),
        system_prompt=SYSTEM_PROMPT,
        callback_handler=None,
    )
    return agent.structured_output(
        IEPLedger,
        f"Extract the obligations ledger from this IEP document:\n\n{iep_text}",
    )
