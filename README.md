# Minutes

**Schools send report cards about your child. Nobody sends a statement about the school. Minutes is that statement.**

A child's IEP (Individualized Education Program) is a legal promise — so many minutes of speech therapy per month, occupational therapy sessions per week, reviews by fixed dates. Minutes is a background agent, built with the [Strands Agents SDK](https://strandsagents.com), that enforces that promise:

1. **Active discovery** — it exercises the parent's statutory records rights on a cadence, requesting service-delivery logs, and records the school's *silence* as dated evidence.
2. **Evidence-graded ledger** — every service minute is tracked with provenance: school-confirmed, parent-observed, or documented-silence. Escalation letters cite or stay silent: every claim is footnoted to a ledger line.
3. **The Statement** — once a month, one artifact: owed, delivered, shortfall, evidence, approaching deadlines. The rest of the month, the agent is quiet. It surfaces only when there is a real decision to make.

Minutes is **not** a chatbot, and it does **not** give legal advice. It compiles documentation; the parent decides.

> Built for the AWS [Agents for Humans](https://agentsforhumans.devpost.com/) hackathon. Project started 2026-08-31, inside the submission window.

## Status

Early scaffold — architecture and build in progress.

## Cost discipline

Built to run cheap and demo cheap:

- **Dev loop on Claude Haiku 4.5** (`global.anthropic.claude-haiku-4-5`), **demo/final on Claude Sonnet 4.6** — the model id is a config value (`MINUTES_MODEL`), never hardcoded.
- `global.` inference profiles only (cheaper than geo/regional endpoints).
- Hard `maxTokens` caps and turn limits on every agent invocation; prompt caching on the stable system/ledger prefix.
- Serverless everywhere (AgentCore Runtime, EventBridge cadence) — nothing idles, nothing bills while waiting.
- AWS Budget alarm on the account (alerts at 40% and 80% of monthly cap) so a runaway loop can never burn quietly.

## License

MIT
