"""Command line for user administration: ``python -m loadlens_app.auth <command>``.

Use it to create the first administrator or to recover access (lost password, locked account).
It talks to the same database as the web app, taken from ``storage.timescale`` in settings.py.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from typing import Optional, Sequence

from .models import Role
from .service import Actor, AuthError, AuthService

_ACTOR = Actor(name="cli", via="cli")


def _read_password(args: argparse.Namespace, confirm: bool = True) -> str:
    if args.password_env:
        value = os.environ.get(args.password_env)
        if not value:
            raise AuthError(f"Переменная окружения {args.password_env} не задана или пуста")
        return value
    first = getpass.getpass("Пароль: ")
    if confirm and getpass.getpass("Повторите пароль: ") != first:
        raise AuthError("Пароли не совпадают")
    return first


def _find(service: AuthService, username: str):
    user = service.repo.get_by_username(username)
    if user is None:
        raise AuthError(f"Пользователь {username!r} не найден")
    return user


def _create_user(service: AuthService, args: argparse.Namespace) -> None:
    user = service.create_user(
        username=args.username,
        role=args.role,
        password=_read_password(args),
        display_name=args.name or "",
        email=args.email or "",
        must_change_password=args.must_change_password,
        actor=_ACTOR,
    )
    print(f"Создан пользователь {user.username} (роль: {user.role.key})")


def _set_password(service: AuthService, args: argparse.Namespace) -> None:
    user = _find(service, args.username)
    service.reset_password(user.id, _read_password(args), actor=_ACTOR, must_change=args.must_change_password)
    print(f"Пароль пользователя {user.username} изменён, все его сессии завершены")


def _set_role(service: AuthService, args: argparse.Namespace) -> None:
    user = _find(service, args.username)
    updated = service.update_user(user.id, actor=_ACTOR, role=args.role)
    print(f"Роль пользователя {updated.username}: {updated.role.key}")


def _unlock(service: AuthService, args: argparse.Namespace) -> None:
    user = _find(service, args.username)
    service.unlock_user(user.id, actor=_ACTOR)
    print(f"Пользователь {user.username} разблокирован")


def _list_users(service: AuthService, args: argparse.Namespace) -> None:
    users = service.list_users()
    if not users:
        print("Пользователей нет")
        return
    print(f"{'Логин':<24}{'Роль':<10}{'Источник':<10}{'Статус'}")
    for user in users:
        status = "отключён" if not user.is_active else ("заблокирован" if user.is_locked() else "активен")
        print(f"{user.username:<24}{user.role.key:<10}{user.provider:<10}{status}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m loadlens_app.auth", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    roles = [role.key for role in Role]

    def with_password(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--password-env", metavar="VAR", help="взять пароль из переменной окружения вместо запроса")
        sub.add_argument("--must-change-password", action="store_true", help="потребовать смену пароля при входе")

    create = commands.add_parser("create-user", help="создать локального пользователя")
    create.add_argument("username")
    create.add_argument("--role", choices=roles, default=Role.VIEWER.key)
    create.add_argument("--name", help="отображаемое имя")
    create.add_argument("--email")
    with_password(create)
    create.set_defaults(handler=_create_user)

    password = commands.add_parser("set-password", help="задать новый пароль")
    password.add_argument("username")
    with_password(password)
    password.set_defaults(handler=_set_password)

    role = commands.add_parser("set-role", help="изменить роль")
    role.add_argument("username")
    role.add_argument("role", choices=roles)
    role.set_defaults(handler=_set_role)

    unlock = commands.add_parser("unlock", help="снять блокировку после неудачных входов")
    unlock.add_argument("username")
    unlock.set_defaults(handler=_unlock)

    commands.add_parser("list-users", help="показать пользователей").set_defaults(handler=_list_users)
    return parser


def main(argv: Optional[Sequence[str]] = None, service: Optional[AuthService] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if service is None:
            from .web import build_service

            service = build_service()
        args.handler(service, args)
    except AuthError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # database unreachable, bad settings, ...
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
