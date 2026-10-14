"""``jarvis telegram`` — log in and run the Telegram user-account tools."""

from __future__ import annotations

import asyncio
import shutil
import signal
import threading
from pathlib import Path

import click
from rich.console import Console

from openjarvis.core.paths import get_config_dir

_SKILL_SRC = (
    Path(__file__).resolve().parents[1]
    / "skills"
    / "data"
    / "telegram-user"
    / "SKILL.md"
)


def _install_skill(console: Console) -> None:
    """Copy the bundled Telegram skill into the user's skills directory."""
    target = get_config_dir() / "skills" / "telegram-user" / "SKILL.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(_SKILL_SRC, target)
    console.print(f"[green]Installed skill[/green] {target}")


@click.group()
def telegram() -> None:
    """Use your personal Telegram account (messages, profile, calls)."""


@telegram.command()
@click.option("--api-id", help="api_id from https://my.telegram.org")
@click.option("--api-hash", help="api_hash from https://my.telegram.org")
@click.option("--phone", help="Account phone number in international format.")
def login(api_id: str | None, api_hash: str | None, phone: str | None) -> None:
    """Log in as a Telegram user and store the session locally."""
    from openjarvis.connectors import telegram_user as tu
    from openjarvis.core.credentials import get_tool_credential, save_credential

    console = Console()
    telethon = tu.import_telethon()
    api_id = api_id or get_tool_credential(tu.CREDENTIAL_TOOL, tu.API_ID_ENV)
    api_hash = api_hash or get_tool_credential(tu.CREDENTIAL_TOOL, tu.API_HASH_ENV)
    if not api_id or not api_hash:
        console.print(
            "Create an app at [bold]https://my.telegram.org[/bold] → "
            "API development tools."
        )
    api_id = api_id or click.prompt("api_id")
    api_hash = api_hash or click.prompt("api_hash", hide_input=True)
    if not str(api_id).strip().isdigit():
        raise click.BadParameter("api_id must be a number", param_hint="--api-id")
    save_credential(tu.CREDENTIAL_TOOL, tu.API_ID_ENV, str(api_id))
    save_credential(tu.CREDENTIAL_TOOL, tu.API_HASH_ENV, api_hash)

    async def _login() -> object:
        client = telethon.TelegramClient(str(tu.session_path()), int(api_id), api_hash)
        await client.start(
            phone=lambda: phone or click.prompt("Phone number (+country code)"),
            code_callback=lambda: click.prompt("Login code sent by Telegram"),
            password=lambda: click.prompt(
                "Two-step verification password", hide_input=True
            ),
        )
        me = await client.get_me()
        await client.disconnect()
        return me

    me = asyncio.run(_login())
    tu.session_path().with_suffix(".session").chmod(0o600)
    name = " ".join(filter(None, [me.first_name, me.last_name]))  # type: ignore[attr-defined]
    console.print(f"[green]Logged in as[/green] {name} (@{me.username or '-'})")  # type: ignore[attr-defined]
    _install_skill(console)
    console.print(
        "Give an agent the tools with [bold]jarvis ask --agent orchestrator "
        '--tools telegram_list_chats,telegram_send_message,... "..."[/bold] '
        "or list them under [agent].tools in config.toml."
    )


@telegram.command()
def status() -> None:
    """Show whether a Telegram session is configured and authorized."""
    from openjarvis.connectors import telegram_user as tu

    console = Console()
    if not tu.is_logged_in_on_disk():
        console.print("[yellow]Not logged in.[/yellow] Run `jarvis telegram login`.")
        return
    try:
        me = tu.TelegramUserClient.get().run(lambda c: c.get_me(), timeout=30)
    except tu.TelegramUserError as exc:
        console.print(f"[red]{exc}[/red]")
        return
    console.print(f"[green]Authorized[/green] as {tu.entity_summary(me)}")


@telegram.command()
def logout() -> None:
    """Log the session out on Telegram's side and delete it locally."""
    from openjarvis.connectors import telegram_user as tu

    console = Console()
    if tu.is_logged_in_on_disk():
        try:
            tu.TelegramUserClient.get().run(lambda c: c.log_out(), timeout=30)
        except tu.TelegramUserError as exc:
            console.print(f"[yellow]{exc}[/yellow]")
        tu.session_path().with_suffix(".session").unlink(missing_ok=True)
    console.print("Logged out.")


@telegram.command()
@click.option(
    "--allow",
    "allow",
    multiple=True,
    help="User allowed to talk to Jarvis (@username or id). Repeatable; "
    "overrides [tools.telegram_user].allowed_users.",
)
@click.option(
    "--answer-messages/--no-answer-messages",
    default=None,
    help="Reply to allowed users' private messages (default: config).",
)
@click.option(
    "--answer-calls/--no-answer-calls",
    default=None,
    help="Answer allowed users' calls by voice (default: config).",
)
@click.option(
    "--instructions",
    default="The caller is authorized to use you. Help them with whatever they "
    "ask, and keep spoken answers short.",
    show_default=True,
    help="What the assistant should do on incoming calls.",
)
@click.option("--greeting", default="Hi, this is Jarvis. How can I help?")
@click.option("--max-minutes", default=15.0, show_default=True)
@click.option("--verbose", is_flag=True, help="Debug-level detail in the log file.")
def listen(
    allow: tuple[str, ...],
    answer_messages: bool | None,
    answer_calls: bool | None,
    instructions: str,
    greeting: str,
    max_minutes: float,
    verbose: bool,
) -> None:
    """Answer messages and calls from allowed users only; ignore everyone else."""
    import logging

    from openjarvis.connectors.telegram_logging import ensure_telegram_logging

    log_file = ensure_telegram_logging(verbose=verbose)
    log = logging.getLogger("openjarvis.telegram.listen")
    from openjarvis.connectors.telegram_responder import (
        TelegramResponder,
        parse_allowed,
        resolve_allowed_users,
    )
    from openjarvis.connectors.telegram_user import TelegramUserClient
    from openjarvis.connectors.telegram_voice import VoiceCallManager
    from openjarvis.core.config import load_config

    console = Console()
    config = load_config()
    settings = config.tools.telegram_user
    refs = list(allow) or parse_allowed(settings.allowed_users)
    if not refs:
        raise click.UsageError(
            "Nobody is allowed to use Jarvis. Set [tools.telegram_user] "
            "allowed_users in config.toml or pass --allow @username."
        )
    if answer_messages is None:
        answer_messages = settings.answer_messages
    if answer_calls is None:
        answer_calls = settings.answer_calls

    tg = TelegramUserClient.get()
    manager = VoiceCallManager.get()

    async def _start() -> dict[int, str]:
        client = await tg.client()
        allowed = await resolve_allowed_users(client, refs)
        if answer_messages:
            TelegramResponder(allowed, config=config).install(client)
        # Group voice chats only listen to these users, calls or not.
        manager.allowed_speakers = frozenset(allowed)
        manager.persistent = True
        if answer_calls:
            manager.allowed_callers = frozenset(allowed)
            manager.auto_answer = {
                "conversational": True,
                "instructions": instructions,
                "greeting": greeting,
                "max_minutes": max_minutes,
            }
            await manager.install_handlers(await tg.calls())
        return allowed

    allowed = tg.submit(_start()).result(timeout=60)
    names = ", ".join(allowed.values())
    log.info(
        "Listener started for %s (ids %s); messages %s, calls %s; model %s",
        names,
        ", ".join(str(i) for i in allowed),
        "on" if answer_messages else "off",
        "on" if answer_calls else "off",
        config.intelligence.default_model,
    )
    console.print(f"[green]Listening on Telegram[/green] for {names} only.")
    console.print(
        f"  messages: {'answered' if answer_messages else 'ignored'} · "
        f"calls: {'answered' if answer_calls else 'ignored'} · "
        "everyone else: ignored. Ctrl+C to stop."
    )
    console.print(f"  log: {log_file}  (follow with: jarvis telegram logs -f)")
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    while not stop.wait(1.0):
        pass
    log.info("Listener stopping")
    tg.shutdown()


@telegram.command()
@click.option("-n", "--lines", default=60, show_default=True, help="Lines to show.")
@click.option("-f", "--follow", is_flag=True, help="Keep printing new lines.")
def logs(lines: int, follow: bool) -> None:
    """Show the Telegram log (messages, agent runs, tools, approvals, calls)."""
    import time
    from collections import deque

    from openjarvis.connectors.telegram_logging import log_path

    path = log_path()
    if not path.exists():
        Console().print(f"No log yet at {path}.")
        return
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in deque(handle, maxlen=lines):
            click.echo(line.rstrip("\n"))
        while follow:
            line = handle.readline()
            if line:
                click.echo(line.rstrip("\n"))
            else:
                time.sleep(0.5)
