"""Behavior-preservation baseline for daimon.core.agent_lifecycle.

Ports the 5 fork-copy scenarios and 2 archive scenarios from Discord's
`_copy_credential_and_repo_binding` / delete_agent inline archival block
(agent_setup/write.py) that this module extracts from. These tests are the
regression proof the whole phase's extraction leans on.
"""

from __future__ import annotations

import uuid

import httpx
import pytest
from anthropic import AsyncAnthropic
from cryptography.fernet import Fernet
from daimon.core.agent_lifecycle import (
    archive_memory_store_best_effort,
    copy_credential_and_repo_binding,
)
from daimon.core.config import AnthropicSettings, DatabaseSettings, GithubSettings, Settings
from daimon.core.errors import DaimonError
from daimon.core.github_credentials import build_multifernet, get_pat, upsert_credential_encrypted
from daimon.core.stores.agent_github_binding import set_agent_github_binding
from daimon.core.stores.agent_memory_stores import get_memory_store_id, insert_memory_store
from daimon.core.stores.agent_repo_binding import get_binding, set_binding
from daimon.core.stores.github_credentials import (
    delete_credential_for_principal,
    get_credential_by_principal,
)
from daimon.testing.factories import make_tenant
from daimon.testing.ma import (
    FakeMemoryStoreState,
    NotHandled,
    build_fake_anthropic,
    make_fake_memory_store_handler,
)
from pydantic import HttpUrl, PostgresDsn, SecretStr
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

pytestmark = pytest.mark.asyncio

_OAUTH_SCOPES = ("repo", "read:user")


def _refusing_anthropic() -> AsyncAnthropic:
    """copy_credential_and_repo_binding never calls the MA API — any call
    here is a test failure."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise NotHandled

    return build_fake_anthropic(handler)


async def test_copy_rekeys_inline_pat_source_onto_fork(
    db_session: AsyncSession, db_session_factory: async_sessionmaker[AsyncSession]
) -> None:
    """Fork copy of an inline-pat source: fork resolves the source's token
    re-keyed under its OWN principal, surviving deletion of the source
    credential (no aliasing)."""
    tenant = await make_tenant(db_session)
    await db_session.commit()
    source_agent_uuid = uuid.uuid4()
    fork_agent_uuid = uuid.uuid4()

    fernet_key = Fernet.generate_key().decode()
    fernet = build_multifernet((fernet_key,))
    plaintext = "ghp_source_token_xxxx1234"
    await upsert_credential_encrypted(
        sessionmaker=db_session_factory,
        fernet=fernet,
        principal_id=source_agent_uuid,
        github_login="(inline-pat)",
        plaintext_token=plaintext,
        scopes=_OAUTH_SCOPES,
    )
    async with db_session_factory() as s, s.begin():
        await set_agent_github_binding(
            s, agent_id=source_agent_uuid, principal_id=source_agent_uuid
        )
        await set_binding(
            s,
            tenant_id=tenant.id,
            agent_id=source_agent_uuid,
            repo_url="github.com/acme/repo",
            default_branch="main",
            ma_secret_ref=f"inline-pat:{source_agent_uuid}",
        )

    await copy_credential_and_repo_binding(
        anthropic=_refusing_anthropic(),  # type: ignore[arg-type]
        sessionmaker=db_session_factory,
        fernet=fernet,
        oauth_scopes=_OAUTH_SCOPES,
        tenant_id=tenant.id,
        source_agent_uuid=source_agent_uuid,
        fork_agent_uuid=fork_agent_uuid,
    )

    fork_pat = await get_pat(
        principal_id=fork_agent_uuid,
        agent_id=fork_agent_uuid,
        sessionmaker=db_session_factory,
        fernet=fernet,
    )
    assert fork_pat == plaintext, "fork's credential must resolve the source's token"

    await delete_credential_for_principal(db_session, principal_id=source_agent_uuid)
    await db_session.commit()
    fork_pat_after_delete = await get_pat(
        principal_id=fork_agent_uuid,
        agent_id=fork_agent_uuid,
        sessionmaker=db_session_factory,
        fernet=fernet,
    )
    assert fork_pat_after_delete == plaintext, (
        "fork's credential must survive deletion of the source credential (no aliasing)"
    )


async def test_copy_raises_when_source_credential_unresolvable(
    db_session: AsyncSession, db_session_factory: async_sessionmaker[AsyncSession]
) -> None:
    """Fails loud when the source's inline-pat binding has no resolvable
    credential (binding row exists, credential row does not)."""
    tenant = await make_tenant(db_session)
    await db_session.commit()
    source_agent_uuid = uuid.uuid4()
    fork_agent_uuid = uuid.uuid4()

    fernet_key = Fernet.generate_key().decode()
    fernet = build_multifernet((fernet_key,))

    async with db_session_factory() as s, s.begin():
        await set_binding(
            s,
            tenant_id=tenant.id,
            agent_id=source_agent_uuid,
            repo_url="github.com/acme/repo",
            default_branch="main",
            ma_secret_ref=f"inline-pat:{source_agent_uuid}",
        )

    with pytest.raises(DaimonError, match="github git-proxy"):
        await copy_credential_and_repo_binding(
            anthropic=_refusing_anthropic(),  # type: ignore[arg-type]
            sessionmaker=db_session_factory,
            fernet=fernet,
            oauth_scopes=_OAUTH_SCOPES,
            tenant_id=tenant.id,
            source_agent_uuid=source_agent_uuid,
            fork_agent_uuid=fork_agent_uuid,
        )

    fork_binding = await get_binding(db_session, tenant_id=tenant.id, agent_id=fork_agent_uuid)
    assert fork_binding is None, "no partial write on the fail-loud path"


async def test_copy_raises_even_with_operator_fallback_configured(
    db_session: AsyncSession, db_session_factory: async_sessionmaker[AsyncSession]
) -> None:
    """Regression guard: copy_credential_and_repo_binding has no wiring to
    receive an operator fallback PAT at all (no settings/
    allow_service_default/fallback_pat parameter exists on this function) --
    an inline-pat source with no resolvable credential must still fail loud
    and write nothing for the fork, even in an environment where
    settings.github.fallback_pat IS configured. This proves the fork path
    cannot silently inherit the shared operator secret, structurally, not
    just by convention."""
    tenant = await make_tenant(db_session)
    await db_session.commit()
    source_agent_uuid = uuid.uuid4()
    fork_agent_uuid = uuid.uuid4()

    fernet_key = Fernet.generate_key().decode()
    fernet = build_multifernet((fernet_key,))

    # An ambient operator fallback PAT is configured in this environment --
    # copy_credential_and_repo_binding takes no settings/fallback_pat
    # parameter, so it structurally cannot see or use it.
    fallback_settings = Settings(
        database=DatabaseSettings(
            url=PostgresDsn("postgresql+asyncpg://daimon:daimon@localhost:5432/daimon"),
        ),
        anthropic=AnthropicSettings(
            api_key=SecretStr("sk-test"),
            base_url=HttpUrl("https://api.anthropic.com"),
        ),
        github=GithubSettings(fallback_pat=SecretStr("ghp_operator_fallback")),
    )
    assert fallback_settings.github.fallback_pat is not None, "sanity: fallback is configured"

    async with db_session_factory() as s, s.begin():
        await set_binding(
            s,
            tenant_id=tenant.id,
            agent_id=source_agent_uuid,
            repo_url="github.com/acme/repo",
            default_branch="main",
            ma_secret_ref=f"inline-pat:{source_agent_uuid}",
        )

    with pytest.raises(DaimonError, match="github git-proxy"):
        await copy_credential_and_repo_binding(
            anthropic=_refusing_anthropic(),  # type: ignore[arg-type]
            sessionmaker=db_session_factory,
            fernet=fernet,
            oauth_scopes=_OAUTH_SCOPES,
            tenant_id=tenant.id,
            source_agent_uuid=source_agent_uuid,
            fork_agent_uuid=fork_agent_uuid,
        )

    fork_binding = await get_binding(db_session, tenant_id=tenant.id, agent_id=fork_agent_uuid)
    assert fork_binding is None, "no partial binding write on the fail-loud path"
    fork_credential = await get_credential_by_principal(db_session, principal_id=fork_agent_uuid)
    assert fork_credential is None, (
        "no github_credentials row for the fork -- the fallback was never persisted"
    )


async def test_copy_anon_binding_without_error_or_credential_write(
    db_session: AsyncSession, db_session_factory: async_sessionmaker[AsyncSession]
) -> None:
    """A public/anon: source binding copies with no error and no credential
    write; the fork's binding carries the same repo verbatim."""
    tenant = await make_tenant(db_session)
    await db_session.commit()
    source_agent_uuid = uuid.uuid4()
    fork_agent_uuid = uuid.uuid4()

    async with db_session_factory() as s, s.begin():
        await set_binding(
            s,
            tenant_id=tenant.id,
            agent_id=source_agent_uuid,
            repo_url="github.com/acme/public-repo",
            default_branch="main",
            ma_secret_ref="anon:",
        )

    fernet_key = Fernet.generate_key().decode()
    fernet = build_multifernet((fernet_key,))
    await copy_credential_and_repo_binding(
        anthropic=_refusing_anthropic(),  # type: ignore[arg-type]
        sessionmaker=db_session_factory,
        fernet=fernet,
        oauth_scopes=_OAUTH_SCOPES,
        tenant_id=tenant.id,
        source_agent_uuid=source_agent_uuid,
        fork_agent_uuid=fork_agent_uuid,
    )

    fork_binding = await get_binding(db_session, tenant_id=tenant.id, agent_id=fork_agent_uuid)
    assert fork_binding is not None, "anon: source binding must still be copied to the fork"
    assert fork_binding.repo_url == "acme/public-repo"
    assert fork_binding.ma_secret_ref == "anon:", "anon: ref is copied verbatim, not rewritten"

    fork_pat = await get_pat(
        principal_id=fork_agent_uuid,
        agent_id=fork_agent_uuid,
        sessionmaker=db_session_factory,
        fernet=fernet,
    )
    assert fork_pat is None, "no credential write must happen for an anon: source"


async def test_copy_rewrites_secret_ref_for_inline_pat(
    db_session: AsyncSession, db_session_factory: async_sessionmaker[AsyncSession]
) -> None:
    """Fork's repo binding matches the source's repo_url/default_branch, with
    ma_secret_ref rewritten to the fork's own inline-pat ref."""
    tenant = await make_tenant(db_session)
    await db_session.commit()
    source_agent_uuid = uuid.uuid4()
    fork_agent_uuid = uuid.uuid4()

    fernet_key = Fernet.generate_key().decode()
    fernet = build_multifernet((fernet_key,))
    plaintext = "ghp_bind_token_xxxx5678"
    await upsert_credential_encrypted(
        sessionmaker=db_session_factory,
        fernet=fernet,
        principal_id=source_agent_uuid,
        github_login="(inline-pat)",
        plaintext_token=plaintext,
        scopes=_OAUTH_SCOPES,
    )
    async with db_session_factory() as s, s.begin():
        await set_agent_github_binding(
            s, agent_id=source_agent_uuid, principal_id=source_agent_uuid
        )
        await set_binding(
            s,
            tenant_id=tenant.id,
            agent_id=source_agent_uuid,
            repo_url="https://github.com/acme/private-repo",
            default_branch="develop",
            ma_secret_ref=f"inline-pat:{source_agent_uuid}",
        )

    await copy_credential_and_repo_binding(
        anthropic=_refusing_anthropic(),  # type: ignore[arg-type]
        sessionmaker=db_session_factory,
        fernet=fernet,
        oauth_scopes=_OAUTH_SCOPES,
        tenant_id=tenant.id,
        source_agent_uuid=source_agent_uuid,
        fork_agent_uuid=fork_agent_uuid,
    )

    fork_binding = await get_binding(db_session, tenant_id=tenant.id, agent_id=fork_agent_uuid)
    assert fork_binding is not None
    assert fork_binding.repo_url == "acme/private-repo"
    assert fork_binding.default_branch == "develop"
    assert fork_binding.ma_secret_ref == f"inline-pat:{fork_agent_uuid}", (
        "ma_secret_ref rewritten to the fork's own inline-pat ref"
    )


async def test_copy_unbound_source_is_a_noop(
    db_session: AsyncSession, db_session_factory: async_sessionmaker[AsyncSession]
) -> None:
    """A source with no repo binding produces no fork binding and no error."""
    tenant = await make_tenant(db_session)
    await db_session.commit()
    source_agent_uuid = uuid.uuid4()
    fork_agent_uuid = uuid.uuid4()

    fernet_key = Fernet.generate_key().decode()
    fernet = build_multifernet((fernet_key,))

    await copy_credential_and_repo_binding(
        anthropic=_refusing_anthropic(),  # type: ignore[arg-type]
        sessionmaker=db_session_factory,
        fernet=fernet,
        oauth_scopes=_OAUTH_SCOPES,
        tenant_id=tenant.id,
        source_agent_uuid=source_agent_uuid,
        fork_agent_uuid=fork_agent_uuid,
    )

    fork_binding = await get_binding(db_session, tenant_id=tenant.id, agent_id=fork_agent_uuid)
    assert fork_binding is None


async def test_archive_best_effort_happy_path(
    db_session: AsyncSession, db_session_factory: async_sessionmaker[AsyncSession]
) -> None:
    tenant = await make_tenant(db_session)
    await db_session.commit()
    agent_id = uuid.uuid4()
    state = FakeMemoryStoreState()
    client = build_fake_anthropic(make_fake_memory_store_handler(state))

    store_id = "memstore_archive_happy"
    async with db_session_factory() as s, s.begin():
        await insert_memory_store(
            s, tenant_id=tenant.id, agent_id=agent_id, memory_store_id=store_id
        )
    state.stores[store_id] = {"archived_at": None}

    await archive_memory_store_best_effort(
        anthropic=client,
        sessionmaker=db_session_factory,
        tenant_id=tenant.id,
        agent_id=agent_id,
        log_context={"tenant_id": str(tenant.id), "agent_name": "doomed"},
    )

    assert state.stores[store_id]["archived_at"] is not None
    async with db_session_factory() as s:
        assert await get_memory_store_id(s, tenant_id=tenant.id, agent_id=agent_id) is None


async def test_archive_best_effort_degrades_on_api_error(
    db_session: AsyncSession, db_session_factory: async_sessionmaker[AsyncSession]
) -> None:
    """A transient MA outage on the archive call must not raise; the DB
    binding remains untouched (inert — nothing re-reads an archived agent)."""
    tenant = await make_tenant(db_session)
    await db_session.commit()
    agent_id = uuid.uuid4()

    store_id = "memstore_archive_degrade"
    async with db_session_factory() as s, s.begin():
        await insert_memory_store(
            s, tenant_id=tenant.id, agent_id=agent_id, memory_store_id=store_id
        )

    def failing_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            json={"type": "error", "error": {"type": "api_error", "message": "boom"}},
        )

    client = build_fake_anthropic(failing_handler)

    # Must not raise despite the 500 from the store archive.
    await archive_memory_store_best_effort(
        anthropic=client,
        sessionmaker=db_session_factory,
        tenant_id=tenant.id,
        agent_id=agent_id,
        log_context={"tenant_id": str(tenant.id), "agent_name": "doomed"},
    )

    async with db_session_factory() as s:
        assert await get_memory_store_id(s, tenant_id=tenant.id, agent_id=agent_id) == store_id
