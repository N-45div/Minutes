// Minutes: what the caseworker may do, as a policy the framework enforces.
//
// Strands evaluates this file before EVERY tool call (CedarAuthorization in
// minutes/interventions.py). Cedar is deny-by-default: a tool call matches a
// permit below or it does not run. There is no forbid and no wildcard, so a
// tool that is added to the agent without a line here is refused, and a test
// (tests/test_interventions.py) fails the build when the two lists drift.
//
// Principal is the parent the caseworker acts for (Parent::"<case>"). Resource
// is fixed by the SDK to Resource::"agent". Neither policy branches on them:
// there is one parent and one case per agent, and what this file controls is
// the action.

// Reading and drafting. Everything the caseworker may do unattended, because
// none of it leaves the family: it reads the IEP ledger and the evidence, runs
// the deterministic engine, and compiles documents into the conversation, not
// into the post. A draft is not a letter until a parent releases it.
permit (
  principal,
  action in [
    Action::"load_case",
    Action::"reconcile_services",
    Action::"check_deadlines",
    Action::"records_requests_status",
    Action::"draft_records_request",
    Action::"draft_shortfall_letter",
    Action::"build_monthly_statement",
    Action::"pending_decisions",
    Action::"audit_trail"
  ],
  resource
);

// Recording a fact only the family has. This starts a statutory clock, so it
// is not read-only, but it reaches nobody outside the family and its one input
// is a date the parent supplied. Nothing to approve.
permit (
  principal,
  action == Action::"record_request_delivery",
  resource
);

// The two tools whose output is meant to leave the family. Each one compiles
// the exact letter, halts on a Strands interrupt, and only the parent's answer
// to that interrupt lets it continue. They are permitted in exactly two
// states, and `context.session.parent_answer` says which — it is read off the
// agent's own interrupt state by minutes/interventions.py, never off the tool's
// arguments, so nothing the model writes into a call can supply it:
//
//   "asking"   — no answer is on file for this call. The tool can only compile
//                the letter and stop to ask; the interrupt it raises halts it
//                before any line that releases anything.
//   "approved" — the parent's most recent answer on this call approves the
//                letter it was shown. The digest of that letter is in the
//                interrupt's name; the tool recompiles and releases only if the
//                digests still match.
//
// A decline is neither, and is refused before the tool re-enters. So is any
// call for which the value is missing.
permit (
  principal,
  action in [Action::"send_records_request", Action::"send_letter"],
  resource
)
when {
  context.session has parent_answer &&
  (context.session.parent_answer == "asking" ||
   context.session.parent_answer == "approved")
};
