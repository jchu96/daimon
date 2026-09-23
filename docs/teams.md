# Teams Adapter — Trust Model

This page documents how daimon's Teams adapter handles per-user access and
what operators should understand about the resulting trust model.

### Scope: personal chat only

The adapter answers only personal (1:1) `MessageActivity`s — the conversation
must have `conversation_type == "personal"` and no `is_group` flag. Channel,
team, and group-chat messages get a short refusal and never reach a turn.
Message extensions, commands, panels, and proactive outbound delivery are not
part of this adapter.

### Ingress is HTTP, not a dial-out

Discord (gateway) and Slack (Socket Mode) both dial out; Teams does not.
The adapter runs its own FastAPI listener on `DAIMON_TEAMS__PORT` (default
`3978`). The Microsoft `App`/`FastAPIAdapter` owns `POST /api/messages` and
validates the inbound Bot Framework JWT against the configured client id —
unauthenticated or wrong-audience activities are rejected by the SDK before
daimon code runs. The same listener serves `/healthz` and `/readyz`;
`DAIMON_TEAMS__ENABLED=false` makes `/api/messages` answer 503 while health
stays live, and a 64 KiB body limit is enforced before SDK parsing. The
`docker-compose.yml` `teams` service (opt-in `teams` profile) publishes the
port; the Bot Framework messaging endpoint must be able to reach it.

### Installation

An authenticated Bot Framework `installationUpdate` with action `add`
creates the tenant and its account through core's idempotent provisioning
helper. It marks the tenant pending, reconciles the deployment defaults
through the same core helper used by Discord and Slack, then marks it ready
and posts one welcome to the installing conversation. Installation never
starts an agent turn. A failed reconciliation leaves the tenant non-ready
and sends no welcome; another `add` retries reconciliation.

The installation must be a personal chat on the Teams channel. The adapter
reads `channelData.tenant.id`, falling back to `conversation.tenantId` only
when the channel tenant is absent, and requires it to equal
`DAIMON_TEAMS__TENANT_ID`. A conflicting conversation tenant is also refused.
Missing, malformed, or foreign tenant ids produce a logged refusal with no
database write or reply. Keep this gate even with a multi-tenant Entra app
registration: Bot Framework authentication proves the sender is the channel,
not that the organisation is allowed to provision on this deployment.

Teams sends personal-scope install events per user. They all share one
deterministic tenant: an already-ready, unarchived tenant makes subsequent
adds a no-op, even with new activity ids or after a process restart. A
Postgres advisory lock serializes overlapping installation deliveries;
waiting requests release their database connections between lock attempts.
The welcome is sent only after provisioning and readiness commit. Delivery
is not transactional with Postgres: a crash or send failure after readiness
commits can leave the tenant ready without a welcome, and later adds do not
retry that welcome.

Action `remove` soft-archives the tenant without deleting its rows. This is
tenant-wide, even when the removal came from one person's personal install;
other users in that tenant can no longer start turns. A later `add` clears
the archive, reconciles defaults again, and welcomes that installation.
Neither action needs an additional app-manifest field or setting.

### Identity: one Entra tenant

Message access remains single-tenant: the activity's conversation tenant
*and* channel-data tenant must both equal `DAIMON_TEAMS__TENANT_ID`, and the
sender must carry a well-formed `aad_object_id`.

`tenant_id` is deterministic — `derive_tenant_uuid(platform="teams",
workspace_id=<Entra tenant id>)` — and `tenants.external_id` carries the Entra
tenant id itself. The resolver rejects the activity unless that row exists
with `provision_status="ready"` and is unarchived.

User principals need no manual step: `admit()` get-or-creates the
`platform_principals` row on first contact, with `external_id` set to the
sender's verified `aad_object_id` (their Entra / Azure AD object id). A
first-contact user in a provisioned tenant is authorized like users on the
other adapters.

### Capacity

Each tenant may run three turns concurrently by default. Set
`DAIMON_TEAMS__MAX_CONCURRENT_TURNS_PER_TENANT` to change that limit. An excess
activity gets a retry-later reply in its personal chat without spawning a
turn; completing, failing, or cancelling a task releases its slot.

### Restart behavior

The progress is posted before `bind_session`; the active-turn marker is
written only after binding returns a session mapping and the progress send
has supplied a usable message id. Binding can take minutes. Cancellation or
process death in between leaves an unswept progress message: no marker was
committed for the next boot to find. A send without a usable message id also
leaves the turn unmarked.

After the marker commits, a redeploy freezes the in-flight progress message
in place; `drain()` cancels the turn task, which deliberately keeps that marker.
On next boot the sweep finds the orphaned row, edits the frozen message to an
interrupted state, and compare-and-clears the marker. Delivery does not resume
— no queued or retried sends — and the sweep clears only the three marker
columns, leaving the session row's status untouched.

### Files

Session output files are not delivered to Teams in this slice — the
Slack-style `/mnt/session/outputs` upload sweep has no Teams counterpart yet.
