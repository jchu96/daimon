# Typed final-answer claims

Deployments can ask an agent to append a machine-readable `claims` fence to
its final answer. The prose remains the human-facing answer; the carrier gives
evaluation code exact values, units, bases, dates, and sources without relying
on presentation-sensitive regular expressions.

````text
Revenue was $7.57M for the period.

```claims
[{"metric":"revenue","value":7571234,"unit":"usd","basis":"invoices","as_of":"2026-06-30","source":"semantic.revenue"}]
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

Renderers should call `strip_claims_blocks()` before showing an answer. The
Slack adapter does this for every carrier, including malformed or non-trailing
examples. Prose grading remains available for deployments that have not yet
adopted typed claims.

## Offline evaluation replay

Typed goldens declare `expected_claim` with a metric, value, canonical unit,
and either absolute or relative tolerance. Optional basis and display precision
constraints further narrow the match. `kind` is one of `value`, `trend`, or
`rejection`; a rejection passes only when the seed is not repeated and the
answer supplies a corrected claim or an explicit seed-bound refusal.

When a golden has `expected_claim`, typed claims are the primary numeric gate:
matching prose cannot rescue an incorrect or missing typed claim. Goldens that
only declare `expected_sql` retain the deprecated regex/tolerance path. Required
phrase checks are advisory, while explicitly banned phrases remain gating.

Recorded answers can be replayed without a network or database:

```bash
daimon eval replay --goldens path/to/goldens.jsonl --fixtures path/to/fixtures
```

The fixture manifest pins every per-case check and fails if a verdict or check
set drifts. Results preserve both stripped `answer` text and the original
`answer_raw` carrier for auditability.
