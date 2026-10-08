import json
import re
import sys
import time as real_time
import types
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from flask import Flask, jsonify

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from loadlens_app import core
from loadlens_app.auth import (
    Actor,
    AuthProvider,
    AuthService,
    ExternalIdentity,
    InMemoryRepository,
    ProviderResult,
    ProviderStatus,
    Role,
    init_auth,
    register_provider,
)
from loadlens_app.auth import cli, passwords, policy, providers, tokens, web
from loadlens_app.auth.config import AuthSettings, load_auth_settings, resolve_secret_key
from loadlens_app.auth.service import ConflictError, ForbiddenError, LoginStatus, NotFoundError, ValidationError
from loadlens_app.auth.throttle import LoginThrottle

PASSWORD = "Correct-Horse-1"
SAFE = {"GET", "HEAD", "OPTIONS"}


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    """Authentication on, cheap password hashes, runtime files in tmp, and no database access."""
    monkeypatch.setattr(core, "CONFIG_RUNTIME_PATH", tmp_path / "settings_runtime.json")
    monkeypatch.setattr(core, "METRICS_RUNTIME_PATH", tmp_path / "metrics_config_runtime.json")

    def no_db():
        raise AssertionError("auth tests must not open database connections")

    for module in list(sys.modules.values()):
        if getattr(module, "__name__", "").startswith("loadlens_app") and hasattr(module, "_ts_conn"):
            monkeypatch.setattr(module, "_ts_conn", no_db)
    monkeypatch.setenv("LOADLENS_AUTH_ENABLED", "1")
    monkeypatch.setattr(passwords, "_method", lambda: "pbkdf2:sha256:1000")
    monkeypatch.setattr(passwords, "_dummy_hash", "")


class Browser:
    """A test client that remembers the CSRF token like the UI does."""

    def __init__(self, app):
        self.client = app.test_client()
        self.csrf = ""

    def request(self, method, url, **kwargs):
        headers = dict(kwargs.pop("headers", {}) or {})
        if self.csrf and method.upper() not in SAFE:
            headers.setdefault("X-CSRF-Token", self.csrf)
        return self.client.open(url, method=method, headers=headers, **kwargs)

    def get(self, url, **kwargs):
        return self.request("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self.request("POST", url, **kwargs)

    def patch(self, url, **kwargs):
        return self.request("PATCH", url, **kwargs)

    def delete(self, url, **kwargs):
        return self.request("DELETE", url, **kwargs)

    def login(self, username, password=PASSWORD, **payload):
        self.csrf = self.get("/auth/me").get_json()["csrf_token"]
        response = self.post("/login", json={"username": username, "password": password, **payload})
        if response.status_code == 200:
            self.csrf = response.get_json()["csrf_token"]
        return response


def add_user(service, name, role, password=PASSWORD, must_change=False, **kwargs):
    return service.create_user(
        username=name, role=role, password=password, must_change_password=must_change, actor=Actor(name="test"), **kwargs
    )


@pytest.fixture
def repo():
    return InMemoryRepository()


@pytest.fixture
def app(repo):
    from app import create_app

    flask_app = create_app(auth_repository=repo)
    flask_app.config["TESTING"] = True
    return flask_app


@pytest.fixture
def service(app):
    return app.extensions[web.EXTENSION_KEY]


@pytest.fixture
def logged_in(app, service):
    """Factory: ``logged_in("viewer")`` returns a signed-in Browser with that role."""
    counter = iter(range(1000))

    def make(role):
        name = f"{role}{next(counter)}"
        add_user(service, name, Role.parse(role))
        browser = Browser(app)
        assert browser.login(name).status_code == 200
        browser.username = name
        return browser

    return make


class Clock:
    def __init__(self):
        self.now = datetime(2030, 1, 1, tzinfo=timezone.utc)
        self.ticks = 1000.0

    def __call__(self):
        return self.now

    def monotonic(self):
        return self.ticks

    def advance(self, **delta):
        self.now += timedelta(**delta)
        self.ticks += timedelta(**delta).total_seconds()


def make_service(repo=None, clock=None, providers_factory=None, **settings):
    repo = repo or InMemoryRepository()
    clock = clock or Clock()
    cfg = replace(AuthSettings(), **settings)
    repo._clock = clock
    chain = providers_factory(repo, cfg, clock) if providers_factory else [providers.LocalProvider(repo, cfg, clock)]
    return AuthService(cfg, repo, chain, clock=clock, monotonic=clock.monotonic), repo, clock


# ---- passwords and tokens ------------------------------------------------------------------


def test_real_password_hash_round_trip(monkeypatch):
    monkeypatch.undo()  # use the production hashing method here
    stored = passwords.hash_password("Some-Long-Passphrase-1")
    assert stored.startswith(("scrypt:", "pbkdf2:"))
    assert passwords.verify_password(stored, "Some-Long-Passphrase-1")
    assert not passwords.verify_password(stored, "some-long-passphrase-1")
    assert not passwords.verify_password("", "x") and not passwords.verify_password("garbage", "x")


@pytest.mark.parametrize(
    "password, username, fragment",
    [
        ("", "alice", "Укажите пароль"),
        ("   " * 6, "alice", "пробел"),
        ("short1!", "alice", "не менее 10"),
        ("x" * 300, "alice", "длиннее"),
        ("my-alice-pass-1", "alice", "имя пользователя"),
        ("password123", "bob", "простой"),
        ("aaaaaaaaaaaa", "bob", "простой"),
    ],
)
def test_password_policy_rejects(password, username, fragment):
    with pytest.raises(passwords.PasswordPolicyError, match=fragment):
        passwords.validate_password(password, username=username, min_length=10)


def test_password_policy_accepts_a_reasonable_password():
    assert passwords.validate_password("Correct-Horse-1", username="alice") == "Correct-Horse-1"


def test_tokens_are_random_prefixed_and_hashed():
    first, prefix, digest = tokens.generate_token()
    second, _, _ = tokens.generate_token()
    assert first != second and first.startswith("ll_") and prefix == first[:11]
    assert digest == tokens.hash_token(first) and first not in digest
    assert tokens.looks_like_token(first)
    assert not tokens.looks_like_token("ll_") and not tokens.looks_like_token("abc") and not tokens.looks_like_token("ll_" + "x" * 300)


# ---- settings ----------------------------------------------------------------------------------


def test_settings_defaults_are_secure():
    cfg = load_auth_settings({}, {})
    assert cfg.enabled and cfg.providers == ("local",) and cfg.api_tokens
    assert cfg.cookie_secure is None and cfg.trusted_proxies == 0 and cfg.password_min_length == 10


def test_settings_environment_overrides_the_config_section():
    assert load_auth_settings({"auth": {"enabled": False}}, {}).enabled is False
    assert load_auth_settings({"auth": {"enabled": False}}, {"LOADLENS_AUTH_ENABLED": "1"}).enabled is True
    assert load_auth_settings({}, {"LOADLENS_AUTH_ENABLED": "off"}).enabled is False
    cfg = load_auth_settings({"auth": {"providers": "ldap, local", "cookie_secure": "yes"}}, {"LOADLENS_TRUSTED_PROXIES": "2"})
    assert cfg.providers == ("ldap", "local") and cfg.cookie_secure is True and cfg.trusted_proxies == 2


def test_settings_ignore_invalid_values():
    cfg = load_auth_settings({"auth": {"session_hours": "abc", "password_min_length": 3, "default_external_role": "root"}}, {})
    assert cfg.session_hours == 12.0 and cfg.password_min_length == 10 and cfg.default_external_role is Role.VIEWER


def test_secret_key_sources(tmp_path):
    key_file = tmp_path / "key"
    configured = replace(AuthSettings(), secret_key="c" * 32)
    assert resolve_secret_key(configured, key_file) == ("c" * 32, "config")

    short = replace(AuthSettings(), secret_key="short")
    key, source = resolve_secret_key(short, key_file)
    assert source == "file" and len(key) >= 32 and key_file.read_text() == key
    assert resolve_secret_key(AuthSettings(), key_file) == (key, "file")  # reused, not regenerated

    key, source = resolve_secret_key(AuthSettings(), tmp_path / "missing-dir" / "key")
    assert source == "ephemeral" and len(key) >= 32


# ---- throttle ---------------------------------------------------------------------------------


def test_login_throttle_blocks_then_expires():
    clock = Clock()
    throttle = LoginThrottle(max_failures=3, window_seconds=60, clock=clock.monotonic)
    for _ in range(2):
        throttle.register_failure("k")
    assert throttle.retry_after("k") == 0
    throttle.register_failure("k")
    assert 0 < throttle.retry_after("k") <= 60
    assert throttle.retry_after("other") == 0
    clock.advance(seconds=61)
    assert throttle.retry_after("k") == 0
    throttle.register_failure("k")
    throttle.reset("k")
    assert throttle.retry_after("k") == 0


# ---- service: login, lockout, sessions -----------------------------------------------------------


def test_login_outcomes_and_audit():
    service, repo, _ = make_service()
    add_user(service, "alice", Role.ENGINEER)
    assert service.login("alice", "wrong-password-1", ip="10.0.0.1").status is LoginStatus.INVALID
    assert service.login("ghost", PASSWORD).status is LoginStatus.INVALID
    assert service.login("", "").status is LoginStatus.INVALID
    outcome = service.login("ALICE", PASSWORD, ip="10.0.0.1")
    assert outcome.status is LoginStatus.OK and outcome.user.username == "alice"
    assert repo.get_by_username("alice").last_login_at is not None
    events = {e.action: e for e in repo.list_audit()}
    assert events["login.failed"].success is False and events["login.failed"].ip == "10.0.0.1"
    assert events["login"].actor_name == "alice"


def test_account_locks_after_repeated_failures_and_unlocks():
    service, repo, clock = make_service(max_failed_logins=3, login_max_attempts=100, lockout_minutes=10)
    user = add_user(service, "alice", Role.VIEWER)
    for _ in range(3):
        service.login("alice", "wrong-password-1")
    assert repo.get_user(user.id).is_locked(clock())
    assert service.login("alice", PASSWORD).status is LoginStatus.INVALID  # the right password is refused while locked
    clock.advance(minutes=11)
    assert service.login("alice", PASSWORD).status is LoginStatus.OK  # the lock expires by itself

    for _ in range(3):
        service.login("alice", "wrong-password-1")
    assert service.login("alice", PASSWORD).status is LoginStatus.INVALID
    service.unlock_user(user.id, actor=Actor(name="admin"))
    assert service.login("alice", PASSWORD).status is LoginStatus.OK


def test_throttle_refuses_attempts_even_with_the_right_password():
    service, _, clock = make_service(login_max_attempts=2, login_window_seconds=60)
    add_user(service, "alice", Role.VIEWER)
    for _ in range(2):
        assert service.login("alice", "wrong-password-1", ip="1.1.1.1").status is LoginStatus.INVALID
    blocked = service.login("alice", PASSWORD, ip="1.1.1.1")
    assert blocked.status is LoginStatus.THROTTLED and blocked.retry_after > 0
    assert service.login("alice", PASSWORD, ip="2.2.2.2").status is LoginStatus.OK  # another address is unaffected
    clock.advance(seconds=61)
    assert service.login("alice", PASSWORD, ip="1.1.1.1").status is LoginStatus.OK


def test_session_user_checks_epoch_and_activity_and_caches_briefly():
    service, repo, clock = make_service(user_cache_seconds=10)
    user = add_user(service, "alice", Role.VIEWER)
    assert service.session_user(user.id, 0).username == "alice"
    assert service.session_user(user.id, 5) is None  # session epoch is stale
    repo.update_user(user.id, {"is_active": False})  # changed behind the service's back
    assert service.session_user(user.id, 0) is not None  # still cached
    clock.advance(seconds=11)
    assert service.session_user(user.id, 0) is None
    service.invalidate_user()


# ---- service: user management invariants ---------------------------------------------------------


def test_create_user_validation():
    service, _, _ = make_service()
    with pytest.raises(ValidationError, match="Логин"):
        add_user(service, "a b", Role.VIEWER)
    with pytest.raises(ValidationError, match="Логин"):
        add_user(service, "ab", Role.VIEWER)
    with pytest.raises(ValidationError, match="не менее"):
        add_user(service, "carol", Role.VIEWER, password="short")
    with pytest.raises(ValidationError, match="роль"):
        add_user(service, "carol", "superuser")
    with pytest.raises(ValidationError, match="Email"):
        add_user(service, "carol", Role.VIEWER, email="not-an-email")
    add_user(service, "carol", Role.VIEWER)
    with pytest.raises(ConflictError):
        add_user(service, "CAROL", Role.VIEWER)


def test_the_last_active_admin_is_protected():
    service, _, _ = make_service()
    only = add_user(service, "root", Role.ADMIN)
    cli_actor = Actor(name="cli")
    with pytest.raises(ConflictError):
        service.update_user(only.id, actor=cli_actor, role="viewer")
    with pytest.raises(ConflictError):
        service.update_user(only.id, actor=cli_actor, is_active=False)
    with pytest.raises(ConflictError):
        service.delete_user(only.id, actor=cli_actor)
    second = add_user(service, "root2", Role.ADMIN)
    service.update_user(only.id, actor=Actor(user=second), role="viewer")  # now another admin remains
    with pytest.raises(ConflictError):
        service.update_user(second.id, actor=cli_actor, role="engineer")


def test_admins_cannot_delete_or_disable_themselves():
    service, _, _ = make_service()
    first = add_user(service, "root", Role.ADMIN)
    add_user(service, "root2", Role.ADMIN)
    with pytest.raises(ForbiddenError):
        service.delete_user(first.id, actor=Actor(user=first))
    with pytest.raises(ForbiddenError):
        service.update_user(first.id, actor=Actor(user=first), is_active=False)


def test_security_changes_end_existing_sessions():
    service, repo, _ = make_service(user_cache_seconds=0)
    add_user(service, "root", Role.ADMIN)
    user = add_user(service, "alice", Role.VIEWER)
    epoch = repo.get_user(user.id).session_epoch
    assert service.session_user(user.id, epoch)

    service.update_user(user.id, actor=Actor(name="t"), display_name="Alice")  # cosmetic: sessions stay
    assert service.session_user(user.id, epoch)
    service.update_user(user.id, actor=Actor(name="t"), role="engineer")
    assert service.session_user(user.id, epoch) is None

    epoch = repo.get_user(user.id).session_epoch
    service.reset_password(user.id, "Brand-New-Pass-5", actor=Actor(name="t"))
    assert service.session_user(user.id, epoch) is None
    assert repo.get_user(user.id).must_change_password is True


def test_change_own_password():
    service, repo, _ = make_service(max_failed_logins=2)
    user = add_user(service, "alice", Role.VIEWER)
    actor = Actor(user=user)
    with pytest.raises(ValidationError, match="Текущий пароль"):
        service.change_own_password(user, "wrong-password-1", "Brand-New-Pass-5", actor=actor)
    with pytest.raises(ValidationError, match="отличаться"):
        service.change_own_password(user, PASSWORD, PASSWORD, actor=actor)
    with pytest.raises(ValidationError, match="не менее"):
        service.change_own_password(user, PASSWORD, "short", actor=actor)
    updated = service.change_own_password(user, PASSWORD, "Brand-New-Pass-5", actor=actor)
    assert updated.must_change_password is False
    assert service.login("alice", "Brand-New-Pass-5").status is LoginStatus.OK
    assert service.login("alice", PASSWORD).status is LoginStatus.INVALID


def test_wrong_current_password_counts_towards_lockout():
    service, repo, clock = make_service(max_failed_logins=2)
    user = add_user(service, "alice", Role.VIEWER)
    for _ in range(2):
        with pytest.raises(ValidationError):
            service.change_own_password(user, "wrong-password-1", "Brand-New-Pass-5", actor=Actor(user=user))
    with pytest.raises(ForbiddenError, match="заблокирована"):
        service.change_own_password(user, PASSWORD, "Brand-New-Pass-5", actor=Actor(user=user))


# ---- service: API tokens -------------------------------------------------------------------------


def test_token_lifecycle_and_limits():
    service, repo, clock = make_service()
    engineer = add_user(service, "eve", Role.ENGINEER)
    admin = add_user(service, "root", Role.ADMIN)
    actor = Actor(user=engineer)

    with pytest.raises(ForbiddenError, match="выше"):
        service.create_token(engineer, "ci", role="admin", actor=actor)
    for days in (0, 366, True, "7"):
        with pytest.raises(ValidationError, match="Срок"):
            service.create_token(engineer, "ci", expires_in_days=days, actor=actor)
    with pytest.raises(ValidationError, match="название"):
        service.create_token(engineer, "  ", actor=actor)

    token, plain = service.create_token(engineer, "ci", role="viewer", expires_in_days=30, actor=actor)
    assert plain.startswith("ll_") and token.token_hash != plain and token.role is Role.VIEWER
    identity = service.authenticate_token(plain)
    assert identity.user.username == "eve" and identity.role is Role.VIEWER

    # A demotion of the owner lowers every token the owner holds.
    wide, wide_plain = service.create_token(engineer, "wide", actor=actor)
    assert service.authenticate_token(wide_plain).role is Role.ENGINEER
    repo.update_user(engineer.id, {"role": Role.VIEWER})
    service.invalidate_user()
    assert service.authenticate_token(wide_plain).role is Role.VIEWER

    assert service.authenticate_token("ll_unknown-token-value") is None
    assert service.authenticate_token("not-a-token") is None
    clock.advance(days=31)
    assert service.authenticate_token(plain) is None  # expired

    with pytest.raises(NotFoundError):
        service.revoke_token(engineer, 9999, actor=actor)
    other = add_user(service, "mallory", Role.VIEWER)
    with pytest.raises(NotFoundError):
        service.revoke_token(other, wide.id, actor=Actor(user=other))  # not the owner
    service.revoke_token(admin, wide.id, actor=Actor(user=admin))  # admins may revoke anybody's
    assert service.authenticate_token(wide_plain) is None


def test_tokens_stop_working_for_disabled_users_and_when_switched_off():
    service, repo, _ = make_service(user_cache_seconds=0)
    user = add_user(service, "eve", Role.ENGINEER)
    _, plain = service.create_token(user, "ci", actor=Actor(user=user))
    assert service.authenticate_token(plain)
    repo.update_user(user.id, {"is_active": False})
    assert service.authenticate_token(plain) is None

    off, _, _ = make_service(api_tokens=False)
    owner = add_user(off, "eve", Role.ENGINEER)
    with pytest.raises(ForbiddenError, match="отключены"):
        off.create_token(owner, "ci", actor=Actor(user=owner))


def test_token_usage_timestamp_is_written_at_most_once_a_minute():
    service, repo, clock = make_service()
    user = add_user(service, "eve", Role.VIEWER)
    token, plain = service.create_token(user, "ci", actor=Actor(user=user))
    service.authenticate_token(plain)
    first = repo.get_token_by_hash(token.token_hash).last_used_at
    clock.advance(seconds=30)
    service.authenticate_token(plain)
    assert repo.get_token_by_hash(token.token_hash).last_used_at == first
    clock.advance(seconds=31)
    service.authenticate_token(plain)
    assert repo.get_token_by_hash(token.token_hash).last_used_at > first


# ---- providers ------------------------------------------------------------------------------------


class FakeDirectory(AuthProvider):
    name = "fake-dir"
    label = "Каталог"

    def __init__(self, accounts):
        self.accounts = accounts
        self.down = False

    def authenticate(self, username, password):
        if self.down:
            return ProviderResult(ProviderStatus.UNAVAILABLE)
        entry = self.accounts.get(username)
        if entry is None or entry["password"] != password:
            return ProviderResult(ProviderStatus.INVALID)
        identity = ExternalIdentity(username=username, external_id=entry["uid"], display_name=entry["name"], role=entry.get("role"))
        return ProviderResult(ProviderStatus.OK, identity=identity)


def _directory_service(accounts):
    directory = FakeDirectory(accounts)
    service, repo, clock = make_service(
        providers_factory=lambda repo, cfg, clock: [directory, providers.LocalProvider(repo, cfg, clock)],
        default_external_role=Role.VIEWER,
    )
    return service, repo, directory


def test_external_identity_is_provisioned_and_kept_in_sync():
    service, repo, directory = _directory_service({"dana": {"password": "dir-pass", "uid": "u-1", "name": "Dana D", "role": Role.ENGINEER}})
    outcome = service.login("dana", "dir-pass")
    assert outcome.status is LoginStatus.OK
    stored = repo.get_by_username("dana")
    assert stored.provider == "fake-dir" and stored.external_id == "u-1" and stored.role is Role.ENGINEER
    assert stored.password_hash is None and stored.display_name == "Dana D"

    directory.accounts["dana"]["role"] = Role.ADMIN
    directory.accounts["dana"]["name"] = "Dana Admin"
    epoch = stored.session_epoch
    assert service.login("dana", "dir-pass").status is LoginStatus.OK
    synced = repo.get_by_username("dana")
    assert synced.role is Role.ADMIN and synced.display_name == "Dana Admin" and synced.session_epoch == epoch + 1
    assert repo.count_users() == 1  # no duplicate on the second login


def test_external_users_default_role_and_deactivation():
    service, repo, _ = _directory_service({"erin": {"password": "dir-pass", "uid": "u-2", "name": "Erin"}})
    assert service.login("erin", "dir-pass").user.role is Role.VIEWER
    repo.update_user(repo.get_by_username("erin").id, {"is_active": False})
    assert service.login("erin", "dir-pass").status is LoginStatus.INVALID


def test_external_identity_cannot_take_over_a_local_account_by_name():
    service, repo, directory = _directory_service({"root": {"password": "dir-pass", "uid": "u-3", "name": "Impostor", "role": Role.ADMIN}})
    local = add_user(service, "root", Role.ADMIN)
    assert service.login("root", "dir-pass").status is LoginStatus.INVALID
    assert repo.get_user(local.id).provider == "local" and repo.count_users() == 1
    assert service.login("root", PASSWORD).status is LoginStatus.OK  # the local provider still answers


def test_local_login_survives_an_unavailable_directory():
    service, _, directory = _directory_service({})
    add_user(service, "root", Role.ADMIN)
    directory.down = True
    assert service.login("root", PASSWORD).status is LoginStatus.OK
    assert service.login("root", "wrong-password-1").status is LoginStatus.UNAVAILABLE


def test_provider_registry():
    class Noop(AuthProvider):
        name = "noop"

        def authenticate(self, username, password):
            return ProviderResult(ProviderStatus.INVALID)

    register_provider("noop-test", lambda ctx: Noop())
    try:
        context = providers.ProviderContext(repository=InMemoryRepository(), settings=AuthSettings(), config={})
        assert [type(p) for p in providers.build_providers(("noop-test", "local"), context)] == [Noop, providers.LocalProvider]
        with pytest.raises(ValueError, match="Неизвестный провайдер"):
            providers.build_providers(("missing",), context)
    finally:
        providers._REGISTRY.pop("noop-test", None)


# ---- access policy --------------------------------------------------------------------------------


def test_every_registered_endpoint_has_an_access_rule(app):
    assert policy.unclassified(app.view_functions) == [], "add the new endpoint(s) to loadlens_app/auth/policy.py"
    assert set(policy.POLICY) - set(app.view_functions) == set(), "policy lists endpoints that do not exist"


def test_unknown_endpoints_default_to_admin_only():
    assert policy.rule_for("brand_new.endpoint").role is Role.ADMIN
    assert policy.rule_for(None).role is Role.ADMIN


def _matrix_app(repo):
    """An app whose endpoints are the real endpoint names with trivial views: tests the gate, not handlers."""
    flask_app = Flask(__name__, template_folder=str(ROOT_DIR / "templates"))
    flask_app.config["TESTING"] = True
    flask_app.context_processor(lambda: {"appearance": {"key": "t", "css": ""}})

    def view(**_):
        return jsonify(job_id="job-1")

    for name in policy.POLICY:
        if not name.startswith("auth.") and name != "static":
            flask_app.add_url_rule(f"/__rule/{name}", endpoint=name, view_func=view, methods=["GET", "POST"])
    flask_app.add_url_rule("/__rule/mystery", endpoint="mystery.endpoint", view_func=view, methods=["GET"])
    init_auth(flask_app, repository=repo, config={}, env={"LOADLENS_AUTH_ENABLED": "1", "LOADLENS_SECRET_KEY": "k" * 32})
    return flask_app


def _persona_browsers(flask_app):
    service = flask_app.extensions[web.EXTENSION_KEY]
    personas = {}
    for role in ("viewer", "engineer", "admin"):
        add_user(service, f"p-{role}", Role.parse(role))
        browser = Browser(flask_app)
        assert browser.login(f"p-{role}").status_code == 200
        personas[role] = browser
    personas["anonymous"] = Browser(flask_app)
    return personas


def test_role_matrix_for_every_endpoint(repo):
    flask_app = _matrix_app(repo)
    personas = _persona_browsers(flask_app)
    failures = []
    for name, rule in policy.POLICY.items():
        if name.startswith("auth.") or name == "static":
            continue
        url = f"/__rule/{name}"
        for persona, browser in personas.items():
            status = browser.get(url).status_code
            if rule.role is None:
                expected = 200
            elif persona == "anonymous":
                expected = 302 if rule.page else 401
            else:
                expected = 200 if Role.parse(persona) >= rule.role else 403
            if status != expected:
                failures.append(f"{persona} {name}: got {status}, expected {expected}")
    assert not failures, "\n".join(failures)


def test_unlisted_endpoint_needs_admin(repo):
    flask_app = _matrix_app(repo)
    personas = _persona_browsers(flask_app)
    assert personas["engineer"].get("/__rule/mystery").status_code == 403
    assert personas["admin"].get("/__rule/mystery").status_code == 200


def test_state_changing_requests_are_audited_without_secrets(repo):
    flask_app = _matrix_app(repo)
    admin = _persona_browsers(flask_app)["admin"]
    body = {"section": "llm", "area": "demo", "data": {"api_key": "SECRET-VALUE", "base_url": "http://x"}}
    assert admin.post("/__rule/config_api.update_config", json=body).status_code == 200
    assert admin.post("/__rule/dashboard.create_report", json={"run_name": "run-7", "service": "svc"}).status_code == 200
    assert admin.get("/__rule/config_api.get_config").status_code == 200  # reads are not audited

    events = {e.action: e for e in repo.list_audit() if e.action.startswith(("config.", "report."))}
    assert set(events) == {"config.update", "report.create"}
    assert events["config.update"].details == {"section": "llm", "area": "demo", "job_id": "job-1"}
    assert events["report.create"].details["run_name"] == "run-7" and events["report.create"].details["job_id"] == "job-1"
    assert events["report.create"].actor_name == "p-admin" and events["report.create"].via == "session"
    assert "SECRET-VALUE" not in json.dumps([e.details for e in repo.list_audit()])


def test_refused_requests_are_not_audited(repo):
    flask_app = _matrix_app(repo)
    viewer = _persona_browsers(flask_app)["viewer"]
    assert viewer.post("/__rule/dashboard.create_report", json={"run_name": "x"}).status_code == 403
    assert not [e for e in repo.list_audit() if e.action == "report.create"]


# ---- HTTP: anonymous access -------------------------------------------------------------------------


def test_anonymous_pages_redirect_to_login_and_apis_answer_401(app):
    anonymous = Browser(app)
    page = anonymous.get("/")
    assert page.status_code == 302 and page.headers["Location"] == "/login?next=%2F"
    assert anonymous.get("/reports?service=a").headers["Location"] == "/login?next=%2Freports%3Fservice%3Da"

    api = anonymous.get("/runs")
    assert api.status_code == 401 and api.get_json()["code"] == "unauthorized"
    assert anonymous.post("/create_report", json={}).status_code == 401  # 401 wins over the missing CSRF token


def test_public_endpoints_stay_reachable(app):
    anonymous = Browser(app)
    assert anonymous.get("/healthz").get_json() == {"status": "ok"}
    assert anonymous.get("/static/css/app.css").status_code == 200
    login = anonymous.get("/login")
    assert login.status_code == 200 and "Войти" in login.get_data(as_text=True)
    assert anonymous.get("/auth/me").get_json()["authenticated"] is False


def test_login_page_explains_how_to_create_the_first_admin(app):
    html = Browser(app).get("/login").get_data(as_text=True)
    assert "Пользователей пока нет" in html and "python -m loadlens_app.auth create-user" in html


def test_security_headers(app):
    response = Browser(app).get("/auth/me")
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "SAMEORIGIN"
    assert response.headers["Cache-Control"] == "no-store"


def test_authenticated_responses_are_not_cacheable_but_public_assets_are(app, logged_in):
    browser = logged_in("viewer")
    assert browser.get("/").headers["Cache-Control"] == "no-store"  # page behind the login
    assert browser.get("/project_areas").headers["Cache-Control"] == "no-store"  # data
    assert Browser(app).get("/login").headers["Cache-Control"] == "no-store"
    assert "no-store" not in Browser(app).get("/static/css/app.css").headers.get("Cache-Control", "")
    assert "Cache-Control" not in Browser(app).get("/healthz").headers


# ---- HTTP: login, logout, sessions --------------------------------------------------------------------


def test_json_login_and_logout(app, service):
    add_user(service, "alice", Role.ENGINEER, display_name="Alice A")
    browser = Browser(app)
    bad = browser.login("alice", "wrong-password-1")
    assert bad.status_code == 401 and bad.get_json()["code"] == "invalid_credentials"
    assert "Неверный логин или пароль" in bad.get_json()["error"]
    assert Browser(app).login("ghost").get_json()["error"] == bad.get_json()["error"]  # no account enumeration

    ok = browser.login("alice")
    assert ok.status_code == 200 and ok.get_json()["redirect"] == "/"
    cookie = next(c for c in ok.headers.getlist("Set-Cookie") if c.startswith("loadlens_session="))
    assert "HttpOnly" in cookie and "SameSite=Lax" in cookie and "Secure" not in cookie

    me = browser.get("/auth/me").get_json()
    assert me["authenticated"] and me["user"]["username"] == "alice" and me["role"] == "engineer" and me["via"] == "session"
    assert "password_hash" not in json.dumps(me)

    out = browser.post("/logout", json={})
    assert out.status_code == 200 and out.get_json()["redirect"] == "/login"
    assert browser.get("/auth/me").get_json()["authenticated"] is False
    assert browser.get("/runs").status_code == 401


def test_html_form_login_follows_next_but_only_within_the_site(app, service):
    add_user(service, "alice", Role.VIEWER)

    def submit(next_url):
        browser = Browser(app)
        page = browser.get("/login").get_data(as_text=True)
        csrf = re.search(r'name="csrf_token" value="([^"]+)"', page).group(1)
        return browser, browser.post("/login", data={"username": "alice", "password": PASSWORD, "next": next_url, "csrf_token": csrf})

    browser, response = submit("/reports?service=a")
    assert response.status_code == 303 and response.headers["Location"] == "/reports?service=a"
    assert browser.get("/auth/me").get_json()["authenticated"] is True
    for evil in ("https://evil.example/", "//evil.example", "/\\evil.example", "javascript:alert(1)"):
        assert submit(evil)[1].headers["Location"] == "/"


def test_failed_form_login_rerenders_the_form_with_a_message(app, service):
    add_user(service, "alice", Role.VIEWER)
    browser = Browser(app)
    page = browser.get("/login").get_data(as_text=True)
    csrf = re.search(r'name="csrf_token" value="([^"]+)"', page).group(1)
    response = browser.post("/login", data={"username": "alice", "password": "nope-nope-nope", "csrf_token": csrf})
    html = response.get_data(as_text=True)
    assert response.status_code == 401 and "Неверный логин или пароль" in html and 'value="alice"' in html


@pytest.mark.parametrize("nxt", ["/", "/reports", "/a/b?c=d#e", "/x%20y"])
def test_safe_next_keeps_local_paths(nxt):
    assert web.safe_next(nxt) == nxt


@pytest.mark.parametrize(
    "nxt", [None, "", "reports", "//evil", "http://evil", "/\\evil", "\\\\evil", "ftp://x", "/ok\\no", "/\t/evil.example", "/\n/evil.example", "/a\x00b"]
)
def test_safe_next_rejects_everything_else(nxt):
    assert web.safe_next(nxt) == "/"


def test_login_requires_the_csrf_token(app, service):
    add_user(service, "alice", Role.VIEWER)
    client = app.test_client()
    assert client.post("/login", json={"username": "alice", "password": PASSWORD}).status_code == 403
    client.get("/auth/me")  # picks up a session
    assert client.post("/login", json={"username": "alice", "password": PASSWORD}, headers={"X-CSRF-Token": "forged"}).status_code == 403


def test_repeated_failures_are_throttled(app, service):
    browser = Browser(app)
    for _ in range(5):
        assert browser.login("ghost", "wrong-password-1").status_code == 401
    blocked = browser.login("ghost", "wrong-password-1")
    assert blocked.status_code == 429 and blocked.get_json()["code"] == "throttled"
    assert int(blocked.headers["Retry-After"]) > 0


def test_a_new_session_replaces_the_old_one_on_login(app, service):
    add_user(service, "alice", Role.VIEWER)
    browser = Browser(app)
    before = browser.get("/auth/me").get_json()["csrf_token"]
    assert browser.login("alice").status_code == 200
    assert browser.csrf != before  # fresh CSRF token: no session fixation


def test_sessions_expire_after_the_absolute_limit(app, service, logged_in, monkeypatch):
    browser = logged_in("viewer")
    assert browser.get("/auth/me").get_json()["authenticated"] is True
    later = real_time.time() + (service.settings.session_max_hours + 1) * 3600
    monkeypatch.setattr(web, "time", types.SimpleNamespace(time=lambda: later))
    assert browser.get("/auth/me").get_json()["authenticated"] is False


def test_logging_in_while_signed_in_redirects_away_from_the_form(app, logged_in):
    browser = logged_in("viewer")
    assert browser.get("/login?next=/reports").headers["Location"] == "/reports"


# ---- HTTP: roles on real endpoints ---------------------------------------------------------------------


def test_viewer_is_limited_to_reading(logged_in):
    viewer = logged_in("viewer")
    assert viewer.get("/project_areas").status_code == 200
    denied = viewer.post("/create_report", json={"run_name": "x"})
    assert denied.status_code == 403 and denied.get_json() == {"error": "Недостаточно прав для этого действия", "code": "forbidden", "required_role": "engineer"}
    assert viewer.get("/config").status_code == 403
    assert viewer.post("/config", json={"section": "llm", "data": {}}).status_code == 403

    page = viewer.get("/new")
    assert page.status_code == 403 and "Нет доступа" in page.get_data(as_text=True)
    assert viewer.get("/settings").status_code == 403


def test_pages_and_navigation_follow_the_role(logged_in):
    def nav_links(browser):
        html = browser.get("/").get_data(as_text=True)
        return set(re.findall(r'<a class="nav-btn[^"]*" href="([^"]+)"', html)), html

    viewer_links, html = nav_links(logged_in("viewer"))
    assert viewer_links == {"/", "/reports", "/compare", "/forecasting"}
    assert 'data-role="viewer"' in html and 'name="loadlens-role" content="viewer"' in html
    assert re.search(r'name="csrf-token" content="[\w-]{20,}"', html)

    engineer = logged_in("engineer")
    assert nav_links(engineer)[0] == {"/", "/new", "/reports", "/compare", "/forecasting"}
    assert engineer.get("/new").status_code == 200 and engineer.get("/settings").status_code == 403

    admin = logged_in("admin")
    assert "/settings" in nav_links(admin)[0] and admin.get("/settings").status_code == 200


def test_user_menu_shows_who_is_signed_in(logged_in):
    browser = logged_in("engineer")
    html = browser.get("/").get_data(as_text=True)
    assert browser.username in html and 'action="/logout"' in html and 'href="/account"' in html


def test_config_api_refuses_to_edit_the_auth_section(logged_in):
    admin = logged_in("admin")
    for section in ("auth", "auth.enabled"):
        response = admin.post("/config", json={"section": section, "data": {"enabled": False}})
        assert response.status_code == 403 and "auth" in response.get_json()["error"]


# ---- HTTP: CSRF ---------------------------------------------------------------------------------------


def test_unsafe_requests_need_a_valid_csrf_token(logged_in):
    admin = logged_in("admin")
    payload = {"username": "newbie", "password": PASSWORD, "role": "viewer"}
    good_token = admin.csrf
    admin.csrf = ""
    missing = admin.post("/auth/users", json=payload)
    assert missing.status_code == 403 and missing.get_json()["code"] == "csrf_failed"
    admin.csrf = "forged-token"
    assert admin.post("/auth/users", json=payload).status_code == 403
    admin.csrf = good_token
    assert admin.post("/auth/users", json=payload).status_code == 201


def test_safe_requests_do_not_need_a_csrf_token(logged_in):
    admin = logged_in("admin")
    admin.csrf = ""
    assert admin.get("/auth/users").status_code == 200


# ---- HTTP: user administration -------------------------------------------------------------------------


def test_admin_manages_users(app, logged_in):
    admin = logged_in("admin")
    created = admin.post("/auth/users", json={"username": "carol", "role": "engineer", "password": "Another-Pass-9", "display_name": "Carol C"})
    assert created.status_code == 201
    carol = created.get_json()["user"]
    assert carol["role"] == "engineer" and carol["must_change_password"] is True and carol["provider"] == "local"
    assert "hash" not in json.dumps(carol) and carol["has_password"] is True

    listing = admin.get("/auth/users").get_json()
    assert {u["username"] for u in listing["users"]} >= {"carol", admin.username}
    assert [r["key"] for r in listing["roles"]] == ["viewer", "engineer", "admin"]
    assert "password_hash" not in json.dumps(listing)

    assert admin.post("/auth/users", json={"username": "CAROL", "role": "viewer", "password": PASSWORD}).status_code == 409
    weak = admin.post("/auth/users", json={"username": "dave", "role": "viewer", "password": "short"})
    assert weak.status_code == 400 and weak.get_json()["code"] == "weak_password"
    assert admin.post("/auth/users", json={"username": "x y", "role": "viewer", "password": PASSWORD}).status_code == 400
    assert admin.post("/auth/users", json={"username": "eve1", "role": "root", "password": PASSWORD}).status_code == 400

    patched = admin.patch(f"/auth/users/{carol['id']}", json={"role": "admin", "display_name": "Carol Admin"})
    assert patched.status_code == 200 and patched.get_json()["user"]["role"] == "admin"
    assert admin.patch("/auth/users/9999", json={"role": "viewer"}).status_code == 404
    assert admin.delete(f"/auth/users/{carol['id']}").status_code == 200
    assert admin.delete(f"/auth/users/{carol['id']}").status_code == 404


def test_admin_cannot_remove_their_own_access(logged_in):
    admin = logged_in("admin")
    me = admin.get("/auth/me").get_json()["user"]["id"]
    assert admin.delete(f"/auth/users/{me}").status_code == 403
    assert admin.patch(f"/auth/users/{me}", json={"is_active": False}).status_code == 403
    assert admin.patch(f"/auth/users/{me}", json={"role": "viewer"}).status_code == 409  # the only admin


def test_non_admins_cannot_manage_users_or_read_the_audit_log(logged_in):
    engineer = logged_in("engineer")
    assert engineer.get("/auth/users").status_code == 403
    assert engineer.post("/auth/users", json={"username": "x1x", "role": "viewer", "password": PASSWORD}).status_code == 403
    assert engineer.get("/auth/audit").status_code == 403


def test_changing_a_role_signs_the_user_out(app, service, logged_in):
    admin = logged_in("admin")
    victim = logged_in("viewer")
    victim_id = victim.get("/auth/me").get_json()["user"]["id"]
    assert admin.patch(f"/auth/users/{victim_id}", json={"role": "engineer"}).status_code == 200
    assert victim.get("/auth/me").get_json()["authenticated"] is False


def test_disabled_users_are_signed_out_and_cannot_log_in(app, service, logged_in):
    admin = logged_in("admin")
    victim = logged_in("viewer")
    victim_id = victim.get("/auth/me").get_json()["user"]["id"]
    assert admin.patch(f"/auth/users/{victim_id}", json={"is_active": False}).status_code == 200
    service.invalidate_user()
    assert victim.get("/auth/me").get_json()["authenticated"] is False
    assert Browser(app).login(victim.username).status_code == 401
    assert admin.patch(f"/auth/users/{victim_id}", json={"is_active": True}).status_code == 200
    assert Browser(app).login(victim.username).status_code == 200


def test_admin_resets_a_password_and_unlocks_an_account(app, service, logged_in):
    admin = logged_in("admin")
    victim = logged_in("viewer")
    victim_id = victim.get("/auth/me").get_json()["user"]["id"]
    for _ in range(service.settings.max_failed_logins):
        service.repo.register_failed_login(victim_id, max_failures=service.settings.max_failed_logins, lock_until=datetime.now(timezone.utc) + timedelta(hours=1))
    service.invalidate_user()
    assert admin.get("/auth/users").get_json()["users"][-1]["locked"] or any(u["locked"] for u in admin.get("/auth/users").get_json()["users"])
    assert admin.post(f"/auth/users/{victim_id}/unlock").get_json()["user"]["locked"] is False

    reset = admin.post(f"/auth/users/{victim_id}/password", json={"new_password": "Reset-By-Admin-3"})
    assert reset.status_code == 200 and reset.get_json()["user"]["must_change_password"] is True
    assert victim.get("/auth/me").get_json()["authenticated"] is False  # old sessions end
    assert Browser(app).login(victim.username, "Reset-By-Admin-3").status_code == 200
    assert admin.post(f"/auth/users/{victim_id}/password", json={"new_password": "weak"}).status_code == 400


# ---- HTTP: forced password change and self-service ---------------------------------------------------------


def test_forced_password_change_blocks_everything_else(app, service):
    add_user(service, "dave", Role.ENGINEER, must_change=True)
    browser = Browser(app)
    login = browser.login("dave")
    assert login.get_json()["redirect"] == "/account?force=1"

    assert browser.get("/").headers["Location"] == "/account?force=1"
    blocked = browser.get("/project_areas")
    assert blocked.status_code == 403 and blocked.get_json()["code"] == "password_change_required"
    account = browser.get("/account?force=1").get_data(as_text=True)
    assert "Нужно сменить пароль" in account and 'action="/logout"' in account  # the way out stays available
    assert not re.findall(r'<a class="nav-btn', account)  # no links to pages that would be refused
    assert browser.get("/auth/me").get_json()["user"]["must_change_password"] is True

    assert browser.post("/auth/password", json={"current_password": "wrong-password-1", "new_password": "Brand-New-Pass-5"}).get_json()["code"] == "wrong_password"
    done = browser.post("/auth/password", json={"current_password": PASSWORD, "new_password": "Brand-New-Pass-5"})
    assert done.status_code == 200
    assert browser.get("/project_areas").status_code == 200  # the same session keeps working


def test_changing_a_password_ends_other_sessions_but_not_this_one(app, service):
    add_user(service, "alice", Role.VIEWER)
    first, second = Browser(app), Browser(app)
    assert first.login("alice").status_code == 200 and second.login("alice").status_code == 200
    assert first.post("/auth/password", json={"current_password": PASSWORD, "new_password": "Brand-New-Pass-5"}).status_code == 200
    service.invalidate_user()
    assert first.get("/auth/me").get_json()["authenticated"] is True
    assert second.get("/auth/me").get_json()["authenticated"] is False
    assert Browser(app).login("alice", "Brand-New-Pass-5").status_code == 200
    assert Browser(app).login("alice", PASSWORD).status_code == 401


def test_account_page_renders_for_everyone_signed_in(logged_in):
    html = logged_in("viewer").get("/account").get_data(as_text=True)
    assert "Смена пароля" in html and "API-токены" in html


# ---- HTTP: API tokens --------------------------------------------------------------------------------------


def test_token_management_over_http(app, logged_in):
    engineer = logged_in("engineer")
    created = engineer.post("/auth/tokens", json={"name": "ci nightly", "role": "viewer", "expires_in_days": 30})
    assert created.status_code == 201
    plain = created.get_json()["token"]
    info = created.get_json()["info"]
    assert plain.startswith("ll_") and info["role"] == "viewer" and info["prefix"] == plain[:11] and info["active"]

    listing = engineer.get("/auth/tokens").get_json()
    assert [t["name"] for t in listing["tokens"]] == ["ci nightly"]
    assert plain not in json.dumps(listing) and "token_hash" not in json.dumps(listing)

    too_high = engineer.post("/auth/tokens", json={"name": "x", "role": "admin"})
    assert too_high.status_code == 403
    assert engineer.post("/auth/tokens", json={"name": "x", "expires_in_days": 0}).status_code == 400

    assert engineer.delete(f"/auth/tokens/{info['id']}").status_code == 200
    assert engineer.delete(f"/auth/tokens/{info['id']}").status_code == 404
    assert Browser(app).get("/project_areas", headers={"Authorization": f"Bearer {plain}"}).status_code == 401


def test_bearer_tokens_authenticate_without_a_cookie_or_csrf(app, logged_in):
    engineer = logged_in("engineer")
    plain = engineer.post("/auth/tokens", json={"name": "ci"}).get_json()["token"]
    header = {"Authorization": f"Bearer {plain}"}

    api_client = Browser(app)
    me = api_client.get("/auth/me", headers=header).get_json()
    assert me["authenticated"] and me["via"] == "token" and me["role"] == "engineer"
    assert api_client.get("/project_areas", headers=header).status_code == 200
    assert api_client.get("/new", headers=header).status_code == 200  # pages work too
    assert api_client.get("/config", headers=header).status_code == 403  # the role still applies


def test_a_leaked_token_cannot_mint_tokens_or_administer_users(app, logged_in):
    admin = logged_in("admin")
    plain = admin.post("/auth/tokens", json={"name": "ci"}).get_json()["token"]
    header = {"Authorization": f"Bearer {plain}"}
    api_client = Browser(app)
    for method, url in (("POST", "/auth/tokens"), ("GET", "/auth/tokens"), ("GET", "/auth/users"), ("POST", "/auth/password"), ("GET", "/auth/audit")):
        response = api_client.request(method, url, json={}, headers=header)
        assert response.status_code == 403 and response.get_json()["code"] == "session_required", (method, url)


def test_a_token_is_capped_by_its_own_role(app, logged_in):
    admin = logged_in("admin")
    plain = admin.post("/auth/tokens", json={"name": "read-only", "role": "viewer"}).get_json()["token"]
    header = {"Authorization": f"Bearer {plain}"}
    api_client = Browser(app)
    assert api_client.get("/project_areas", headers=header).status_code == 200
    assert api_client.get("/config", headers=header).status_code == 403
    assert api_client.post("/create_report", json={}, headers=header).status_code == 403


def test_invalid_bearer_tokens_are_rejected_not_ignored(app, logged_in):
    admin = logged_in("admin")
    # A valid browser session does not rescue a bad Authorization header.
    response = admin.get("/project_areas", headers={"Authorization": "Bearer ll_not-a-real-token-value"})
    assert response.status_code == 401 and response.headers["WWW-Authenticate"].startswith("Bearer")


def test_admin_sees_and_revokes_everybodys_tokens(app, logged_in):
    admin = logged_in("admin")
    engineer = logged_in("engineer")
    token_id = engineer.post("/auth/tokens", json={"name": "eng-ci"}).get_json()["info"]["id"]
    assert engineer.get("/auth/tokens?all=1").get_json()["tokens"][0]["owner"] == engineer.username  # not widened for non-admins
    everyone = admin.get("/auth/tokens?all=1").get_json()["tokens"]
    assert {"eng-ci"} <= {t["name"] for t in everyone} and everyone[0]["owner"] == engineer.username
    assert admin.delete(f"/auth/tokens/{token_id}").status_code == 200


# ---- HTTP: audit log -----------------------------------------------------------------------------------------


def test_audit_log_records_security_events(app, service, logged_in):
    add_user(service, "mallory", Role.VIEWER)
    assert Browser(app).login("mallory", "wrong-password-1").status_code == 401
    admin = logged_in("admin")
    admin.post("/auth/users", json={"username": "newbie", "role": "viewer", "password": PASSWORD})
    admin.post("/auth/tokens", json={"name": "ci"})

    events = admin.get("/auth/audit?limit=50").get_json()["events"]
    failed = next(e for e in events if e["action"] == "login.failed")
    assert failed["actor"] == "mallory" and failed["success"] is False
    created = next(e for e in events if e["action"] == "user.create" and e["target"] == "newbie")
    assert created["actor"] == admin.username and created["via"] == "session"
    assert next(e for e in events if e["action"] == "token.create")["target"] == "ci"
    assert admin.get("/auth/audit?action=login").get_json()["events"] and all(e["action"].startswith("login") for e in admin.get("/auth/audit?action=login").get_json()["events"])
    assert len(admin.get("/auth/audit?limit=1").get_json()["events"]) == 1
    assert admin.get("/auth/audit?limit=abc&offset=-5").status_code == 200  # bad numbers fall back to defaults


# ---- HTTP: switched off, bootstrap ------------------------------------------------------------------------------


def test_everything_is_open_when_authentication_is_disabled(monkeypatch, repo):
    monkeypatch.setenv("LOADLENS_AUTH_ENABLED", "0")
    from app import create_app

    flask_app = create_app(auth_repository=repo)
    client = Browser(flask_app)
    assert client.get("/").status_code == 200 and client.get("/settings").status_code == 200
    assert client.get("/login").headers["Location"] == "/"
    me = client.get("/auth/me").get_json()
    assert me == {"enabled": False, "authenticated": False, "role": "admin", "user": None, "via": "disabled"}
    assert client.post("/config", json={"section": "auth", "data": {}}).status_code == 403  # still guarded
    assert client.post("/auth/users", json={"username": "prep1", "role": "admin", "password": PASSWORD}).status_code == 201
    assert client.post("/auth/password", json={}).get_json()["code"] == "auth_disabled"
    assert 'data-role="admin"' in client.get("/").get_data(as_text=True)


def test_the_first_admin_is_created_from_the_environment(monkeypatch, repo):
    monkeypatch.setenv("LOADLENS_ADMIN_USER", "root")
    monkeypatch.setenv("LOADLENS_ADMIN_PASSWORD", "Initial-Pass-42")
    from app import create_app

    flask_app = create_app(auth_repository=repo)
    assert repo.count_users() == 0  # nothing happens at startup: the database may not be up yet
    browser = Browser(flask_app)
    assert "Пользователей пока нет" not in browser.get("/login").get_data(as_text=True)
    assert repo.count_users() == 1 and repo.get_by_username("root").role is Role.ADMIN
    assert browser.login("root", "Initial-Pass-42").status_code == 200
    Browser(flask_app).get("/login")
    assert repo.count_users() == 1  # idempotent


def test_a_weak_bootstrap_password_is_refused(monkeypatch, repo, caplog):
    monkeypatch.setenv("LOADLENS_ADMIN_USER", "root")
    monkeypatch.setenv("LOADLENS_ADMIN_PASSWORD", "short")
    from app import create_app

    flask_app = create_app(auth_repository=repo)
    with caplog.at_level("ERROR"):
        Browser(flask_app).get("/login")
    assert repo.count_users() == 0 and "rejected" in caplog.text


def test_bootstrap_does_nothing_when_users_exist(monkeypatch, repo):
    monkeypatch.setenv("LOADLENS_ADMIN_USER", "root")
    monkeypatch.setenv("LOADLENS_ADMIN_PASSWORD", "Initial-Pass-42")
    from app import create_app

    flask_app = create_app(auth_repository=repo)
    add_user(flask_app.extensions[web.EXTENSION_KEY], "existing", Role.VIEWER)
    Browser(flask_app).get("/login")
    assert repo.get_by_username("root") is None


def test_database_outage_is_reported_not_crashed(monkeypatch, repo):
    class Broken(InMemoryRepository):
        def get_by_username(self, username):
            raise RuntimeError("database is down")

        def get_user(self, user_id):
            raise RuntimeError("database is down")

        def count_users(self):
            raise RuntimeError("database is down")

    from app import create_app

    flask_app = create_app(auth_repository=Broken())
    browser = Browser(flask_app)
    assert browser.get("/login").status_code == 200  # the form still renders
    outage = browser.login("alice")
    assert outage.status_code == 503 and outage.get_json()["code"] == "auth_unavailable"

    with flask_app.test_client() as client:
        with client.session_transaction() as session:
            session.update(uid=1, epoch=0, iat=int(real_time.time()), csrf="t")
        assert client.get("/runs").status_code == 503  # fails closed


# ---- command line ---------------------------------------------------------------------------------------------


def test_cli_manages_users(monkeypatch, capsys):
    service, repo, _ = make_service()
    monkeypatch.setenv("CLI_PASSWORD", "Cli-Password-77")

    assert cli.main(["create-user", "bob", "--role", "engineer", "--name", "Bob", "--password-env", "CLI_PASSWORD"], service=service) == 0
    assert repo.get_by_username("bob").role is Role.ENGINEER
    assert repo.get_by_username("bob").must_change_password is False

    assert cli.main(["create-user", "bob", "--password-env", "CLI_PASSWORD"], service=service) == 1
    assert "уже существует" in capsys.readouterr().err
    monkeypatch.setenv("CLI_WEAK", "short")
    assert cli.main(["create-user", "weak1", "--password-env", "CLI_WEAK"], service=service) == 1
    assert cli.main(["create-user", "bob2", "--password-env", "UNSET_VARIABLE"], service=service) == 1

    monkeypatch.setenv("CLI_NEW", "Newer-Password-88")
    assert cli.main(["set-password", "bob", "--password-env", "CLI_NEW", "--must-change-password"], service=service) == 0
    assert service.login("bob", "Newer-Password-88").status is LoginStatus.OK
    assert repo.get_by_username("bob").must_change_password is True
    assert cli.main(["set-password", "ghost", "--password-env", "CLI_NEW"], service=service) == 1

    add_user(service, "root", Role.ADMIN)
    assert cli.main(["set-role", "bob", "admin"], service=service) == 0
    assert repo.get_by_username("bob").role is Role.ADMIN
    for _ in range(service.settings.max_failed_logins):
        service.login("bob", "wrong-password-1", ip=str(_))
    assert cli.main(["unlock", "bob"], service=service) == 0
    capsys.readouterr()
    assert cli.main(["list-users"], service=service) == 0
    listing = capsys.readouterr().out
    assert "bob" in listing and "root" in listing and "admin" in listing


def test_cli_records_who_did_it(monkeypatch):
    service, repo, _ = make_service()
    monkeypatch.setenv("CLI_PASSWORD", "Cli-Password-77")
    cli.main(["create-user", "bob", "--password-env", "CLI_PASSWORD"], service=service)
    event = next(e for e in repo.list_audit() if e.action == "user.create")
    assert event.actor_name == "cli" and event.via == "cli" and event.target == "bob"
