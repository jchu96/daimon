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

An authenticated Bot Framework `installationUpdate` with action `add`,
`add-upgrade`, or the SDK literal `upgrade` creates the installing
organisation's tenant and account through core's
idempotent provisioning helper. Like Discord enrollment, it seeds
`DAIMON_BILLING__SIGNUP_CREDIT` once, inserts the tenant as pending, reconciles
deployment defaults, then marks it ready and sends one short welcome in the
installing conversation. Installation never starts an agent turn. Failed
reconciliation leaves the tenant non-ready; another `add` retries it.

`DAIMON_TEAMS__ALLOWED_TENANT_IDS` controls enrollment. Its default, an empty
JSON array (`[]`), permits any Entra tenant to enroll and receive the configured
signup credit. Set a non-empty JSON array of Entra tenant UUIDs to permit
only those organisations. Microsoft app registration and distribution still
control who can install the app; the enrollment setting does not change them.
`DAIMON_TEAMS__TENANT_ID` identifies the bot's own Entra app registration for
authentication. It is not the enrollment allowlist.

A closed-enrollment install from an unlisted tenant receives "Daimon isn't
enabled for your organisation yet." Set `DAIMON_TEAMS__ENROLLMENT_CONTACT`
to append a contact line. The adapter logs `teams.install.declined` with the
tenant id and writes nothing. Existing provisioned tenants retain message
access if the enrollment allowlist later changes.

Enrollment must be a personal chat on the Teams channel. Removal may also
come from a team channel or group chat. The adapter reads
`channelData.tenant.id`, falling back to `conversation.tenantId` only
when the channel tenant is absent. Missing, malformed, or conflicting tenant
ids are refused before any database write. Bot Framework authentication and
consistent activity identity remain mandatory even with open enrollment.

Teams sends personal-scope install events per user. They share one
deterministic tenant per organisation: an already-ready, unarchived tenant
makes subsequent adds a no-op, even with new activity ids or after a process
restart. A per-tenant process lock serializes overlapping deliveries without
holding database connections during core helper calls. Core database upserts
keep tenant, account, and credit creation idempotent across processes. Concurrent
processes can still reconcile and welcome the same pending tenant; welcome
deduplication across active processes is not guaranteed. Interrupted enrollment
stays pending and the next add retries defaults reconciliation.
The welcome follows provisioning and readiness commits. Delivery is not
transactional with Postgres: a crash or send failure after readiness commits
can leave the tenant ready without a welcome, and later adds do not retry it.

Actions `remove` and `remove-upgrade` in personal scope log
`teams.install.removed` with the user and tenant and write nothing: one user's
uninstall must not disable every other user. Removal from a team channel or
group chat soft-archives the tenant without deleting its rows and logs the same
event. This archive is tenant-wide. A later permitted add clears the archive
and reconciles again; signup credit is not granted again.

The ingress normalizes Microsoft's `add-upgrade` and `remove-upgrade` actions
before SDK parsing and preserves the original action for authenticated handling.
Unknown actions log `teams.install.ignored` once with the action and produce no
writes or replies. Authentication still runs before any installation handling.
No additional app-manifest field is needed.

### Identity: provisioned tenants

Message access requires the activity's conversation tenant *and* channel-data
tenant to identify the same Entra directory, and the sender must carry a
well-formed `aad_object_id`. The bot registration tenant and enrollment
allowlist do not restrict messages from already-provisioned tenants.

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
