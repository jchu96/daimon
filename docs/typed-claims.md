# Typed final-answer claims

Deployments can ask an agent to append a machine-readable `claims` fence to
its final answer. The prose remains the human-facing answer; the carrier gives
evaluation code exact values, units, bases, dates, and sources without relying
on presentation-sensitive regular expressions.

````text
Revenue was $7.57M for the period.

```claims
[{"metric":"revenue","value":7571234,"unit":"usd","basis":"invoices","as_of":"2026-06-30","source":"invoices.total"}]
```
````

`daimon.core.claims_contract.render_claims_instruction()` generates the agent
instruction from the same strict `Claim` model used by the parser. Deployments
with a measure registry can pass it to the renderer and parser to reject
unknown measure IDs; without one, lowercase registry-style IDs remain valid.

`parse_claims_block()` treats only the final trailing fence as authoritative.
Malformed JSON or an empty/non-list carrier rejects the block. Schema-invalid
rows are reported independently so one bad row does not discard valid siblings.
Unicode dashes and thin/non-breaking spaces are normalized, canonical units use
a closed vocabulary, and formatted numeric strings are converted to exact
decimals. Presentation-only keys are ignored.

Core final and sealed-response extractors remove every carrier before returning
human-facing text, including malformed or non-trailing examples. Discord,
Slack, and headless consumers therefore share the same safe render boundary.
Evaluation retains the raw answer separately for typed grading and audit.
Prose grading remains available for deployments that have not yet adopted
typed claims.
