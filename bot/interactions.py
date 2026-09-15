"""Shared execution boundary for text and voice tool turns.

Execution results, not model promises, supply music receipts. Operation IDs stay
stable across retries within an interaction. Network work never owns guild state.
"""
from __future__ import annotations

import asyncio
import json
from collections import OrderedDict
from contextvars import ContextVar
from dataclasses import dataclass, field
from uuid import uuid4


@dataclass(frozen=True)
class RequestContext:
    request_id: str
    guild_id: int | None
    user_id: int | None
    origin: str
    original_text: str
    transcript_confidence: float | None = None
    transcript_language: str | None = None
    queue_revision: int | None = None
    queue_entries: tuple[str, ...] = ()


current_operation: ContextVar[str | None] = ContextVar("music_operation", default=None)
current_request: ContextVar[RequestContext | None] = ContextVar("interaction_request", default=None)


@dataclass(frozen=True)
class Command:
    operation_id: str
    name: str
    arguments_json: str


class ExecutionLedger:
    """Bounded runtime deduplication. Delivery retries never repeat a mutation."""
    def __init__(self):
        self.entries = OrderedDict()

    async def execute(self, context, index, tc, message, dispatch, *, durable):
        key = (context.guild_id, context.user_id, context.request_id, index)
        command = Command(f"{context.request_id}:{index}", tc["name"],
                          json.dumps(tc["arguments"], sort_keys=True))
        existing = self.entries.get(key)
        if existing:
            prior, task = existing
            if prior != command:
                return "This request already executed a different action; submit a new request."
        else:
            task = asyncio.create_task(dispatch(tc, message))
            self.entries[key] = (command, task)
        try:
            return await asyncio.shield(task) if durable else await task
        finally:
            if task.cancelled():
                self.entries.pop(key, None)
            # Never evict in-flight commands merely to make room.
            for old_key in list(self.entries):
                if len(self.entries) <= 256:
                    break
                if self.entries[old_key][1].done():
                    self.entries.pop(old_key)


execution_ledger = ExecutionLedger()


@dataclass
class ToolTurn:
    context: RequestContext
    messages: list[dict] = field(default_factory=list)
    receipts: list[str] = field(default_factory=list)
    has_information: bool = False


def request_context(message, text: str, *, origin: str, player=None) -> RequestContext:
    return RequestContext(
        str(getattr(message, "id", None) or uuid4()),
        getattr(message.guild, "id", None),
        getattr(message.author, "id", None), origin, text,
        getattr(text, "confidence", None), getattr(text, "language", None),
        getattr(getattr(player, "operations", None), "revision", None),
        tuple(track.entry_id for track in getattr(player, "queue", ()) if hasattr(track, "entry_id")),
    )


def render_music_result(result: str | dict, *, language: str = "en", origin: str = "text") -> str:
    """Render authoritative action state. Titles are data, never instructions."""
    if isinstance(result, str):
        try:
            outcome = json.loads(result)
        except (ValueError, TypeError):
            return result  # Compatibility with existing read/control tools.
    else:
        outcome = result
    if not isinstance(outcome, dict) or "status" not in outcome:
        return str(result)
    spanish = language == "es"
    title = outcome.get("title") or ("la canción" if spanish else "the track")
    artist = outcome.get("artist")
    name = f"{title} — {artist}" if artist else title
    status = outcome["status"]
    if origin == "voice":
        spoken_name = f"{title}, de {artist}" if spanish and artist else f"{title} by {artist}" if artist else title
        if status == "queued":
            position = outcome.get("queue_position")
            if position is not None:
                return (f"Agregado: {spoken_name}, posición {position}." if spanish
                        else f"Queued: {spoken_name}, position {position}.")
            return f"Agregado a la cola: {spoken_name}." if spanish else f"Queued: {spoken_name}."
        if status == "starting":
            return f"Iniciando: {spoken_name}." if spanish else f"Starting: {spoken_name}."
        if status == "playing":
            return f"Reproduciendo: {spoken_name}." if spanish else f"Playing: {spoken_name}."
    if status == "queued":
        position = outcome.get("queue_position")
        if position is not None:
            return (f"Agregado en la posición {position}: {name}." if spanish
                    else f"Added at position {position}: {name}.")
        return f"Agregado a la cola: {name}." if spanish else f"Queued: {name}."
    if status == "playing":
        return f"Reproduciendo: {name}." if spanish else f"Playing: {name}."
    if status == "starting":
        return f"Preparando la reproducción: {name}." if spanish else f"Starting playback: {name}."
    if status in {"accepted", "resolving"}:
        return "Solicitud recibida." if spanish else "Request accepted."
    if status == "completed":
        labels = {
            "skip": ("Skipped it.", "Listo, salté la canción."),
            "pause": ("Paused.", "En pausa."),
            "resume": ("Resumed.", "Reanudado."),
            "stop": ("Stopped and cleared the queue.", "Detuve la música y vacié la cola."),
            "undo": ("Deleted the most recent request.", "Eliminé la solicitud más reciente."),
            "restart": ("Restarted the track.", "Reinicié la canción."),
            "shuffle": ("Shuffled the queue.", "Mezclé la cola."),
            "move_track": ("Moved the track.", "Moví la canción."),
            "delete_track": ("Removed the selected request.", "Eliminé la solicitud seleccionada."),
            "delete_entries": ("Removed the selected tracks.", "Eliminé las canciones seleccionadas."),
        }
        if outcome.get("action") in labels:
            return labels[outcome["action"]][int(spanish)]
    return outcome.get("message") or ("No se pudo completar la solicitud." if spanish
                                       else "The request could not be completed.")


async def execute_turn(tool_calls, message, text, execute, is_music, *, origin="text", context=None) -> ToolTurn:
    """Execute each accepted action once; preserve mixed informational results."""
    turn = ToolTurn(context or request_context(message, text, origin=origin))
    turn.messages.append({
        "role": "assistant", "content": None,
        "tool_calls": [{"id": tc["id"], "type": "function", "function": {
            "name": tc["name"], "arguments": json.dumps(tc["arguments"]),
        }} for tc in tool_calls],
    })
    for index, tc in enumerate(tool_calls):
        token = current_operation.set(f"{turn.context.request_id}:{index}")
        request_token = current_request.set(turn.context)
        try:
            result = await execution_ledger.execute(turn.context, index, tc, message, execute,
                                                    durable=is_music(tc["name"]))
        finally:
            current_operation.reset(token)
            current_request.reset(request_token)
        turn.messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
        if is_music(tc["name"]):
            language = tc["arguments"].get("response_language", "en")
            turn.receipts.append(render_music_result(result, language=language, origin=turn.context.origin))
        else:
            turn.has_information = True
    return turn
