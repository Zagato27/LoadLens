"""Contract tests for the auth repositories.

The in-memory implementation always runs. Set LOADLENS_TEST_PG_DSN to a PostgreSQL DSN (for example
``postgresql://postgres:test@127.0.0.1:55432/postgres``) to run the same tests against
PostgresRepository in a throw-away schema.
"""

import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from loadlens_app.auth.models import AuditEvent, Role
from loadlens_app.auth.repository import DuplicateUserError, InMemoryRepository

PG_DSN = os.environ.get("LOADLENS_TEST_PG_DSN", "")


@pytest.fixture(params=["memory", pytest.param("postgres", marks=pytest.mark.skipif(not PG_DSN, reason="LOADLENS_TEST_PG_DSN is not set"))])
def repo(request):
    if request.param == "memory":
        yield InMemoryRepository()
        return
    import psycopg2
    from loadlens_app.auth.pg_repository import PostgresRepository

    schema = f"auth_test_{uuid.uuid4().hex[:8]}"
    repository = PostgresRepository(lambda: psycopg2.connect(PG_DSN), schema)
    try:
        yield repository
    finally:
        conn = psycopg2.connect(PG_DSN)
        try:
            with conn, conn.cursor() as cur:
                cur.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        finally:
            conn.close()


def _make(repo, name="alice", role=Role.VIEWER, **kwargs):
    return repo.create_user(username=name, role=role, password_hash="hash", **kwargs)


def test_create_and_lookup_is_case_insensitive(repo):
    user = _make(repo, "Alice", Role.ENGINEER, display_name="Alice A", email="a@example.com")
    assert user.id and user.role is Role.ENGINEER and user.provider == "local"
    assert repo.get_by_username("alice").id == user.id
    assert repo.get_by_username("ALICE").display_name == "Alice A"
    assert repo.get_user(user.id).username == "Alice"
    assert repo.get_user(user.id + 999) is None
    assert repo.count_users() == 1


def test_duplicate_username_is_rejected_case_insensitively(repo):
    _make(repo, "alice")
    with pytest.raises(DuplicateUserError):
        _make(repo, "ALICE")
    assert repo.count_users() == 1


def test_external_identity_is_unique_per_provider(repo):
    _make(repo, "ldap-bob", provider="ldap", external_id="bob")
    assert repo.get_by_external("ldap", "bob").username == "ldap-bob"
    assert repo.get_by_external("oidc", "bob") is None
    with pytest.raises(DuplicateUserError):
        _make(repo, "other-name", provider="ldap", external_id="bob")


def test_update_user_changes_only_allowed_fields_and_bumps_epoch(repo):
    user = _make(repo, "alice")
    updated = repo.update_user(user.id, {"display_name": "A", "role": Role.ADMIN, "is_active": False, "password_hash": "x"}, bump_epoch=True)
    assert updated.display_name == "A" and updated.role is Role.ADMIN and updated.is_active is False
    assert updated.session_epoch == user.session_epoch + 1
    assert repo.get_user(user.id).password_hash == "hash"  # not in the whitelist
    assert repo.update_user(user.id + 999, {"display_name": "x"}) is None


def test_count_active_admins_excludes_inactive_and_given_id(repo):
    first = _make(repo, "root1", Role.ADMIN)
    second = _make(repo, "root2", Role.ADMIN)
    _make(repo, "eng", Role.ENGINEER)
    assert repo.count_active_admins() == 2
    assert repo.count_active_admins(exclude_id=first.id) == 1
    repo.update_user(second.id, {"is_active": False})
    assert repo.count_active_admins() == 1
    assert repo.count_active_admins(exclude_id=first.id) == 0


def test_set_password_resets_lockout_and_bumps_epoch(repo):
    user = _make(repo, "alice")
    lock_until = datetime.now(timezone.utc) + timedelta(minutes=5)
    repo.register_failed_login(user.id, max_failures=1, lock_until=lock_until)
    assert repo.get_user(user.id).is_locked()
    repo.set_password(user.id, "new-hash", True)
    fresh = repo.get_user(user.id)
    assert fresh.password_hash == "new-hash" and fresh.must_change_password is True
    assert fresh.locked_until is None and fresh.failed_logins == 0
    assert fresh.session_epoch == user.session_epoch + 1


def test_failed_logins_lock_the_account_and_restart_the_counter(repo):
    user = _make(repo, "alice")
    lock_until = datetime.now(timezone.utc) + timedelta(minutes=5)
    for _ in range(2):
        repo.register_failed_login(user.id, max_failures=3, lock_until=lock_until)
    assert repo.get_user(user.id).failed_logins == 2 and not repo.get_user(user.id).is_locked()
    repo.register_failed_login(user.id, max_failures=3, lock_until=lock_until)
    locked = repo.get_user(user.id)
    assert locked.is_locked() and locked.failed_logins == 0
    repo.unlock_user(user.id)
    assert not repo.get_user(user.id).is_locked()


def test_login_success_clears_counters_and_records_time(repo):
    user = _make(repo, "alice")
    repo.register_failed_login(user.id, max_failures=5, lock_until=datetime.now(timezone.utc))
    repo.record_login_success(user.id)
    fresh = repo.get_user(user.id)
    assert fresh.failed_logins == 0 and fresh.last_login_at is not None


def test_tokens_lifecycle(repo):
    user = _make(repo, "alice", Role.ENGINEER)
    other = _make(repo, "bob")
    expires = datetime.now(timezone.utc) + timedelta(days=1)
    token = repo.create_token(user_id=user.id, name="ci", prefix="ll_abcdefgh", token_hash="h1", role=Role.VIEWER, expires_at=expires)
    repo.create_token(user_id=other.id, name="other", prefix="ll_zzzzzzzz", token_hash="h2", role=Role.VIEWER, expires_at=None)

    found = repo.get_token_by_hash("h1")
    assert found.id == token.id and found.role is Role.VIEWER and found.is_usable()
    assert repo.get_token_by_hash("missing") is None
    assert [t.name for t in repo.list_tokens(user.id)] == ["ci"]
    everyone = repo.list_tokens()
    assert {t.name for t in everyone} == {"ci", "other"} and {t.owner for t in everyone} == {"alice", "bob"}

    assert repo.revoke_token(token.id, user_id=other.id) is False  # someone else's token
    assert repo.revoke_token(token.id, user_id=user.id) is True
    assert repo.revoke_token(token.id, user_id=user.id) is False  # already revoked
    assert repo.get_token_by_hash("h1").is_usable() is False
    repo.touch_token(token.id)
    assert repo.get_token_by_hash("h1").last_used_at is not None


def test_deleting_a_user_removes_their_tokens(repo):
    user = _make(repo, "alice")
    repo.create_token(user_id=user.id, name="ci", prefix="ll_abcdefgh", token_hash="h1", role=Role.VIEWER, expires_at=None)
    assert repo.delete_user(user.id) is True
    assert repo.delete_user(user.id) is False
    assert repo.get_token_by_hash("h1") is None


def test_audit_log_is_newest_first_and_filterable(repo):
    repo.add_audit(AuditEvent(action="login", actor_name="alice", ip="10.0.0.1", details={"provider": "local"}))
    repo.add_audit(AuditEvent(action="user.create", actor_name="root", target="bob", details={"role": "viewer"}))
    repo.add_audit(AuditEvent(action="login.failed", actor_name="mallory", success=False))

    events = repo.list_audit()
    assert [e.action for e in events] == ["login.failed", "user.create", "login"]
    assert events[2].details == {"provider": "local"} and events[2].ip == "10.0.0.1"
    assert events[0].success is False and events[0].created_at is not None
    assert [e.action for e in repo.list_audit(action="login")] == ["login.failed", "login"]
    assert [e.actor_name for e in repo.list_audit(actor="root")] == ["root"]
    assert [e.action for e in repo.list_audit(limit=1, offset=1)] == ["user.create"]
    assert repo.list_audit(action="100%") == []  # LIKE wildcards in the filter are literal
