"""PostgreSQL implementation of the auth repository.

Tables are created lazily with ``IF NOT EXISTS`` (the same approach as the other stores in this
project) and are mirrored in docker/initdb/02_schema.sql for fresh databases. Every operation opens
its own connection, like the rest of the app: there is no connection pool.
"""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Any, Callable, Optional

import psycopg2
import psycopg2.errors
from psycopg2 import sql
from psycopg2.extras import Json

from .models import ApiToken, AuditEvent, Role, User
from .repository import UPDATABLE_USER_FIELDS, DuplicateUserError, UserRepository

USERS_TABLE = "app_users"
TOKENS_TABLE = "api_tokens"
AUDIT_TABLE = "audit_log"

# Serialises concurrent CREATE TABLE IF NOT EXISTS from several worker processes.
_DDL_LOCK_KEY = 727_001

_USER_COLUMNS = (
    "id, username, display_name, email, role, provider, external_id, password_hash, is_active, "
    "must_change_password, session_epoch, failed_logins, locked_until, last_login_at, created_at, updated_at"
)
_TOKEN_COLUMNS = "id, user_id, name, token_prefix, token_hash, role, expires_at, last_used_at, revoked_at, created_at"
_AUDIT_COLUMNS = "id, created_at, actor_id, actor_name, action, target, success, ip, via, details"


def _user(row: tuple) -> User:
    (uid, username, display_name, email, role, provider, external_id, password_hash, is_active,
     must_change, epoch, failed, locked_until, last_login, created, updated) = row
    return User(
        id=uid,
        username=username,
        role=Role.parse(role),
        display_name=display_name or "",
        email=email or "",
        provider=provider,
        external_id=external_id,
        password_hash=password_hash,
        is_active=bool(is_active),
        must_change_password=bool(must_change),
        session_epoch=int(epoch or 0),
        failed_logins=int(failed or 0),
        locked_until=locked_until,
        last_login_at=last_login,
        created_at=created,
        updated_at=updated,
    )


def _token(row: tuple, owner: str = "") -> ApiToken:
    tid, user_id, name, prefix, token_hash, role, expires, last_used, revoked, created = row[:10]
    return ApiToken(
        id=tid,
        user_id=user_id,
        name=name,
        prefix=prefix,
        role=Role.parse(role),
        token_hash=token_hash,
        created_at=created,
        expires_at=expires,
        last_used_at=last_used,
        revoked_at=revoked,
        owner=owner,
    )


def _audit(row: tuple) -> AuditEvent:
    eid, created, actor_id, actor_name, action, target, success, ip, via, details = row
    return AuditEvent(
        id=eid,
        created_at=created,
        actor_id=actor_id,
        actor_name=actor_name or "",
        action=action,
        target=target or "",
        success=bool(success),
        ip=ip or "",
        via=via or "",
        details=details if isinstance(details, dict) else {},
    )


class PostgresRepository(UserRepository):
    def __init__(self, connect: Callable[[], Any], schema: Callable[[], str] | str = "public") -> None:
        self._connect = connect
        self._schema = schema if callable(schema) else (lambda: schema)
        self._ensured: set[str] = set()
        self._ensure_lock = threading.Lock()

    # ---- plumbing ----------------------------------------------------------

    def _table(self, name: str) -> sql.Identifier:
        return sql.Identifier(self._schema(), name)

    def _ensure_tables(self, conn) -> None:
        schema = self._schema()
        if schema in self._ensured:
            return
        with self._ensure_lock:
            if schema in self._ensured:
                return
            users, tokens, audit = (self._table(n) for n in (USERS_TABLE, TOKENS_TABLE, AUDIT_TABLE))
            with conn.cursor() as cur:
                cur.execute("SELECT pg_advisory_xact_lock(%s)", (_DDL_LOCK_KEY,))
                if schema != "public":
                    cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema)))
                cur.execute(
                    sql.SQL(
                        """
                        CREATE TABLE IF NOT EXISTS {users} (
                            id                   BIGSERIAL PRIMARY KEY,
                            username             TEXT NOT NULL,
                            display_name         TEXT NOT NULL DEFAULT '',
                            email                TEXT NOT NULL DEFAULT '',
                            role                 TEXT NOT NULL DEFAULT 'viewer'
                                                 CHECK (role IN ('viewer', 'engineer', 'admin')),
                            provider             TEXT NOT NULL DEFAULT 'local',
                            external_id          TEXT,
                            password_hash        TEXT,
                            is_active            BOOLEAN NOT NULL DEFAULT TRUE,
                            must_change_password BOOLEAN NOT NULL DEFAULT FALSE,
                            session_epoch        INTEGER NOT NULL DEFAULT 0,
                            failed_logins        INTEGER NOT NULL DEFAULT 0,
                            locked_until         TIMESTAMPTZ,
                            last_login_at        TIMESTAMPTZ,
                            created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
                            updated_at           TIMESTAMPTZ NOT NULL DEFAULT now()
                        )
                        """
                    ).format(users=users)
                )
                cur.execute(
                    sql.SQL("CREATE UNIQUE INDEX IF NOT EXISTS app_users_username_lower_uq ON {} (lower(username))").format(users)
                )
                cur.execute(
                    sql.SQL(
                        "CREATE UNIQUE INDEX IF NOT EXISTS app_users_external_uq ON {} (provider, external_id) "
                        "WHERE external_id IS NOT NULL"
                    ).format(users)
                )
                cur.execute(
                    sql.SQL(
                        """
                        CREATE TABLE IF NOT EXISTS {tokens} (
                            id           BIGSERIAL PRIMARY KEY,
                            user_id      BIGINT NOT NULL REFERENCES {users} (id) ON DELETE CASCADE,
                            name         TEXT NOT NULL,
                            token_prefix TEXT NOT NULL,
                            token_hash   TEXT NOT NULL UNIQUE,
                            role         TEXT NOT NULL DEFAULT 'viewer'
                                         CHECK (role IN ('viewer', 'engineer', 'admin')),
                            expires_at   TIMESTAMPTZ,
                            last_used_at TIMESTAMPTZ,
                            revoked_at   TIMESTAMPTZ,
                            created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
                        )
                        """
                    ).format(tokens=tokens, users=users)
                )
                cur.execute(sql.SQL("CREATE INDEX IF NOT EXISTS api_tokens_user_idx ON {} (user_id)").format(tokens))
                cur.execute(
                    sql.SQL(
                        """
                        CREATE TABLE IF NOT EXISTS {audit} (
                            id         BIGSERIAL PRIMARY KEY,
                            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                            actor_id   BIGINT,
                            actor_name TEXT NOT NULL DEFAULT '',
                            action     TEXT NOT NULL,
                            target     TEXT NOT NULL DEFAULT '',
                            success    BOOLEAN NOT NULL DEFAULT TRUE,
                            ip         TEXT NOT NULL DEFAULT '',
                            via        TEXT NOT NULL DEFAULT 'session',
                            details    JSONB NOT NULL DEFAULT '{{}}'::jsonb
                        )
                        """
                    ).format(audit=audit)
                )
                cur.execute(sql.SQL("CREATE INDEX IF NOT EXISTS audit_log_created_idx ON {} (created_at DESC)").format(audit))
            conn.commit()
            self._ensured.add(schema)

    def _run(self, work: Callable[[Any], Any]) -> Any:
        """Runs ``work(conn)`` in one transaction; recreates the tables once if they vanished."""
        for attempt in (1, 2):
            conn = self._connect()
            try:
                self._ensure_tables(conn)
                result = work(conn)
                conn.commit()
                return result
            except psycopg2.errors.UndefinedTable:
                conn.rollback()
                self._ensured.clear()
                if attempt == 2:
                    raise
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    # ---- users -----------------------------------------------------------

    def count_users(self) -> int:
        def work(conn):
            with conn.cursor() as cur:
                cur.execute(sql.SQL("SELECT count(*) FROM {}").format(self._table(USERS_TABLE)))
                return int(cur.fetchone()[0])

        return self._run(work)

    def count_active_admins(self, exclude_id: Optional[int] = None) -> int:
        def work(conn):
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        "SELECT count(*) FROM {} WHERE role = 'admin' AND is_active AND (%s::bigint IS NULL OR id <> %s)"
                    ).format(self._table(USERS_TABLE)),
                    (exclude_id, exclude_id),
                )
                return int(cur.fetchone()[0])

        return self._run(work)

    def _select_one(self, where: str, params: tuple) -> Optional[User]:
        def work(conn):
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("SELECT {} FROM {} WHERE ").format(sql.SQL(_USER_COLUMNS), self._table(USERS_TABLE))
                    + sql.SQL(where),
                    params,
                )
                row = cur.fetchone()
                return _user(row) if row else None

        return self._run(work)

    def get_user(self, user_id: int) -> Optional[User]:
        return self._select_one("id = %s", (user_id,))

    def get_by_username(self, username: str) -> Optional[User]:
        return self._select_one("lower(username) = lower(%s)", (username,))

    def get_by_external(self, provider: str, external_id: str) -> Optional[User]:
        return self._select_one("provider = %s AND external_id = %s", (provider, external_id))

    def list_users(self) -> list[User]:
        def work(conn):
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("SELECT {} FROM {} ORDER BY lower(username)").format(
                        sql.SQL(_USER_COLUMNS), self._table(USERS_TABLE)
                    )
                )
                return [_user(row) for row in cur.fetchall()]

        return self._run(work)

    def create_user(
        self,
        *,
        username: str,
        role: Role,
        password_hash: Optional[str] = None,
        display_name: str = "",
        email: str = "",
        provider: str = "local",
        external_id: Optional[str] = None,
        must_change_password: bool = False,
        is_active: bool = True,
    ) -> User:
        def work(conn):
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        sql.SQL(
                            "INSERT INTO {} (username, display_name, email, role, provider, external_id, password_hash, "
                            "is_active, must_change_password) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING {}"
                        ).format(self._table(USERS_TABLE), sql.SQL(_USER_COLUMNS)),
                        (username, display_name, email, role.key, provider, external_id, password_hash, is_active,
                         must_change_password),
                    )
                except psycopg2.errors.UniqueViolation as exc:
                    raise DuplicateUserError(username) from exc
                return _user(cur.fetchone())

        return self._run(work)

    def update_user(self, user_id: int, fields: dict, *, bump_epoch: bool = False) -> Optional[User]:
        columns = [name for name in UPDATABLE_USER_FIELDS if name in fields]
        values = [fields[name].key if name == "role" else fields[name] for name in columns]
        assignments = [sql.SQL("{} = %s").format(sql.Identifier(name)) for name in columns]
        if bump_epoch:
            assignments.append(sql.SQL("session_epoch = session_epoch + 1"))
        assignments.append(sql.SQL("updated_at = now()"))

        def work(conn):
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("UPDATE {} SET {} WHERE id = %s RETURNING {}").format(
                        self._table(USERS_TABLE), sql.SQL(", ").join(assignments), sql.SQL(_USER_COLUMNS)
                    ),
                    (*values, user_id),
                )
                row = cur.fetchone()
                return _user(row) if row else None

        return self._run(work)

    def delete_user(self, user_id: int) -> bool:
        def work(conn):
            with conn.cursor() as cur:
                cur.execute(sql.SQL("DELETE FROM {} WHERE id = %s").format(self._table(USERS_TABLE)), (user_id,))
                return cur.rowcount > 0

        return self._run(work)

    def _execute(self, query: str, params: tuple, table: str = USERS_TABLE) -> int:
        def work(conn):
            with conn.cursor() as cur:
                cur.execute(sql.SQL(query).format(self._table(table)), params)
                return cur.rowcount

        return self._run(work)

    def set_password(self, user_id: int, password_hash: str, must_change_password: bool) -> None:
        self._execute(
            "UPDATE {} SET password_hash = %s, must_change_password = %s, session_epoch = session_epoch + 1, "
            "failed_logins = 0, locked_until = NULL, updated_at = now() WHERE id = %s",
            (password_hash, must_change_password, user_id),
        )

    def register_failed_login(self, user_id: int, *, max_failures: int, lock_until: datetime) -> None:
        # Both CASE expressions read the old row, so the counter restarts exactly when the lock is set.
        self._execute(
            "UPDATE {} SET locked_until = CASE WHEN failed_logins + 1 >= %s THEN %s ELSE locked_until END, "
            "failed_logins = CASE WHEN failed_logins + 1 >= %s THEN 0 ELSE failed_logins + 1 END WHERE id = %s",
            (max_failures, lock_until, max_failures, user_id),
        )

    def record_login_success(self, user_id: int) -> None:
        self._execute(
            "UPDATE {} SET failed_logins = 0, locked_until = NULL, last_login_at = now() WHERE id = %s", (user_id,)
        )

    def unlock_user(self, user_id: int) -> None:
        self._execute("UPDATE {} SET failed_logins = 0, locked_until = NULL WHERE id = %s", (user_id,))

    # ---- API tokens -------------------------------------------------------

    def create_token(
        self, *, user_id: int, name: str, prefix: str, token_hash: str, role: Role, expires_at: Optional[datetime]
    ) -> ApiToken:
        def work(conn):
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        "INSERT INTO {} (user_id, name, token_prefix, token_hash, role, expires_at) "
                        "VALUES (%s, %s, %s, %s, %s, %s) RETURNING {}"
                    ).format(self._table(TOKENS_TABLE), sql.SQL(_TOKEN_COLUMNS)),
                    (user_id, name, prefix, token_hash, role.key, expires_at),
                )
                return _token(cur.fetchone())

        return self._run(work)

    def get_token_by_hash(self, token_hash: str) -> Optional[ApiToken]:
        def work(conn):
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("SELECT {} FROM {} WHERE token_hash = %s").format(
                        sql.SQL(_TOKEN_COLUMNS), self._table(TOKENS_TABLE)
                    ),
                    (token_hash,),
                )
                row = cur.fetchone()
                return _token(row) if row else None

        return self._run(work)

    def list_tokens(self, user_id: Optional[int] = None) -> list[ApiToken]:
        def work(conn):
            with conn.cursor() as cur:
                columns = sql.SQL(", ").join(sql.SQL("t.") + sql.SQL(c.strip()) for c in _TOKEN_COLUMNS.split(","))
                cur.execute(
                    sql.SQL(
                        "SELECT {}, u.username FROM {} t JOIN {} u ON u.id = t.user_id "
                        "WHERE (%s::bigint IS NULL OR t.user_id = %s) ORDER BY t.id DESC"
                    ).format(columns, self._table(TOKENS_TABLE), self._table(USERS_TABLE)),
                    (user_id, user_id),
                )
                return [_token(row, owner=row[10] or "") for row in cur.fetchall()]

        return self._run(work)

    def revoke_token(self, token_id: int, user_id: Optional[int] = None) -> bool:
        return (
            self._execute(
                "UPDATE {} SET revoked_at = now() WHERE id = %s AND revoked_at IS NULL "
                "AND (%s::bigint IS NULL OR user_id = %s)",
                (token_id, user_id, user_id),
                TOKENS_TABLE,
            )
            > 0
        )

    def touch_token(self, token_id: int) -> None:
        self._execute("UPDATE {} SET last_used_at = now() WHERE id = %s", (token_id,), TOKENS_TABLE)

    # ---- audit ------------------------------------------------------------

    def add_audit(self, event: AuditEvent) -> None:
        def work(conn):
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        "INSERT INTO {} (actor_id, actor_name, action, target, success, ip, via, details) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
                    ).format(self._table(AUDIT_TABLE)),
                    (event.actor_id, event.actor_name, event.action, event.target, event.success, event.ip,
                     event.via, Json(event.details or {})),
                )

        self._run(work)

    def list_audit(self, *, limit: int = 100, offset: int = 0, action: str = "", actor: str = "") -> list[AuditEvent]:
        like = action.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%" if action else ""

        def work(conn):
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        "SELECT {} FROM {} WHERE (%s = '' OR action LIKE %s) AND (%s = '' OR actor_name = %s) "
                        "ORDER BY id DESC LIMIT %s OFFSET %s"
                    ).format(sql.SQL(_AUDIT_COLUMNS), self._table(AUDIT_TABLE)),
                    (like, like, actor, actor, limit, offset),
                )
                return [_audit(row) for row in cur.fetchall()]

        return self._run(work)
