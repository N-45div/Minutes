"""Model and runtime configuration.

Model ids are always inference-profile ids (never bare model ids) so
invocations route through cross-region profiles. MINUTES_MODEL overrides
the default for demo-quality runs.
"""

import os

# Dev default: Haiku 4.5. Demo runs set MINUTES_MODEL to Sonnet.
DEFAULT_MODEL = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
CLASSIFIER_MODEL = "us.amazon.nova-lite-v1:0"

EXTRACTION_MODEL = os.environ.get("MINUTES_MODEL", DEFAULT_MODEL)

# Reading a photograph of paper is the one job where the cheap classifier is
# the wrong tool: it renders tables as markdown pipes, and the duration
# grounding in correspondence.py reads columns separated by whitespace, so a
# piped row silently loses its own stated minutes and falls back to the IEP's
# figure -- the one direction that invents or erases a shortfall.
TRANSCRIPTION_MODEL = os.environ.get("MINUTES_TRANSCRIPTION_MODEL", EXTRACTION_MODEL)
BEDROCK_REGION = os.environ.get("MINUTES_REGION", "us-east-1")
