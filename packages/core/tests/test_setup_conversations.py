"""Setup identity validation never substitutes a deleted target's namesake."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import httpx
import pytest
from daimon.core.errors import DaimonError
from daimon.core.setup_conversations import get_setup_responder, resolve_setup_agents
from daimon.testing.ma import MARouter, build_fake_anthropic, list_response
from daimon.testing.ma_models import ma_agent


@pytest.mark.parametrize("missing_target", [False, True])
async def test_setup_resolves_builtin_without_parent_default_or_namesake_fallback(
    missing_target: bool,
) -> None:
    tenant_id = uuid.uuid4()
    now = datetime.now(UTC)
    daimon = ma_agent(
        id="ag_builtin",
        name="daimon",
        tenant_id=tenant_id,
        metadata={"daimon_managed": "true"},
        created_at=now,
    )
    specialist = ma_agent(
        id="ag_specialist", name="specialist", tenant_id=tenant_id, created_at=now
    )
    router = MARouter()
    router.add(
        "GET",
        r"/v1/agents",
        lambda req, match: list_response(
            [daimon.model_dump(mode="json"), specialist.model_dump(mode="json")]
        ),
    )
    router.add(
        "GET",
        r"/v1/agents/ag_specialist",
        lambda req, match: httpx.Response(200, json=specialist.model_dump(mode="json")),
    )
    router.add(
        "GET",
        r"/v1/agents/ag_deleted",
        lambda req, match: httpx.Response(
            404, json={"type": "error", "error": {"type": "not_found_error", "message": "missing"}}
        ),
    )
    client = build_fake_anthropic(router.dispatch)
    if missing_target:
        with pytest.raises(DaimonError, match="no longer exists"):
            await resolve_setup_agents(client, tenant_id=tenant_id, target_ma_agent_id="ag_deleted")
    else:
        responder, target = await resolve_setup_agents(
            client, tenant_id=tenant_id, target_ma_agent_id="ag_specialist"
        )
        assert responder.id == "ag_builtin", "Daimon always responds to setup"
        assert target is not None and target.id == "ag_specialist", (
            "selected specialist remains target"
        )


async def test_missing_bound_responder_is_explicit_and_does_not_reconcile() -> None:
    client = build_fake_anthropic(
        lambda req: httpx.Response(
            404, json={"type": "error", "error": {"type": "not_found_error", "message": "missing"}}
        )
    )
    with pytest.raises(DaimonError, match="Daimon responder is missing"):
        await get_setup_responder(client, tenant_id=uuid.uuid4(), ma_agent_id="ag_deleted")
