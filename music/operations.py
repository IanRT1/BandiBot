"""Ordered per-guild operations with concurrent, bounded preparation.

Preparation never holds a queue lock. Only the short commit follows acceptance
order. Cancellation invalidates every outstanding commit, including threads
which cannot be stopped while a third-party resolver is running.
"""
from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from uuid import uuid4

from music.results import PlayResult
from core.config import MUSIC_OPERATION_TIMEOUT_SECONDS, MUSIC_PREPARE_CONCURRENCY, MUSIC_PENDING_LIMIT


@dataclass
class PendingOperation:
    operation_id: str
    epoch: int
    predecessor: asyncio.Future | None
    finished: asyncio.Future
    label: str = "Pending music request"
    cancelled: asyncio.Event = field(default_factory=asyncio.Event)


class GuildOperations:
    def __init__(self):
        self.epoch = 0
        self.revision = 0
        self.pending: OrderedDict[str, PendingOperation] = OrderedDict()
        self.tasks: dict[str, asyncio.Task] = {}
        self.results: OrderedDict[str, PlayResult] = OrderedDict()
        self.preparation_slots = asyncio.Semaphore(MUSIC_PREPARE_CONCURRENCY)

    def invalidate(self):
        self.epoch += 1
        for operation in self.pending.values():
            operation.cancelled.set()

    def cancel_latest(self) -> bool:
        for operation in reversed(self.pending.values()):
            if not operation.cancelled.is_set():
                operation.cancelled.set()
                return True
        return False

    def cancel(self, operation_id: str) -> bool:
        operation = self.pending.get(operation_id)
        if operation is None or operation.cancelled.is_set():
            return False
        operation.cancelled.set()
        return True

    def valid(self, operation: PendingOperation) -> bool:
        return operation.epoch == self.epoch and not operation.cancelled.is_set()

    async def submit(self, prepare, commit, *, operation_id=None, label="Pending music request", timeout=MUSIC_OPERATION_TIMEOUT_SECONDS):
        operation_id = operation_id or uuid4().hex
        if operation_id in self.results:
            return self.results[operation_id]
        if operation_id in self.tasks:
            return await asyncio.shield(self.tasks[operation_id])
        if len(self.pending) >= MUSIC_PENDING_LIMIT:
            return PlayResult("failed", error_code="busy", message="Too many pending music requests. Try again shortly.")
        loop = asyncio.get_running_loop()
        previous = next(reversed(self.pending.values()), None)
        operation = PendingOperation(operation_id, self.epoch,
                                     previous.finished if previous else None, loop.create_future(), label)
        self.pending[operation_id] = operation
        task = asyncio.create_task(self._run(operation, prepare, commit, timeout))
        self.tasks[operation_id] = task
        return await asyncio.shield(task)

    async def _run(self, operation, prepare, commit, timeout):
        cancelled = asyncio.create_task(operation.cancelled.wait())
        async def bounded_prepare():
            async with self.preparation_slots:
                return await prepare()
        preparation = asyncio.create_task(bounded_prepare())
        result = PlayResult("failed", error_code="cancelled", message="Song request cancelled.")
        try:
            async def finish():
                prepared = await preparation
                if operation.predecessor:
                    await asyncio.shield(operation.predecessor)
                if self.valid(operation):
                    return await commit(prepared, operation)
                return result

            work = asyncio.create_task(finish())
            try:
                done, _ = await asyncio.wait((work, cancelled), timeout=timeout,
                                             return_when=asyncio.FIRST_COMPLETED)
                if cancelled in done or not self.valid(operation):
                    pass
                elif work in done:
                    result = work.result()
                else:
                    result = PlayResult("failed", error_code="timeout", message="Song request timed out.")
            finally:
                if not work.done():
                    work.cancel()
                await asyncio.gather(work, return_exceptions=True)
        except Exception as exc:
            result = getattr(exc, "result", None) or PlayResult("failed", error_code="resolution_failed", message=f"Could not resolve track: {exc}")
        finally:
            cancelled.cancel()
            if not preparation.done():
                preparation.cancel()
            await asyncio.gather(cancelled, preparation, return_exceptions=True)
            self.pending.pop(operation.operation_id, None)
            if not operation.finished.done():
                operation.finished.set_result(None)
            self.tasks.pop(operation.operation_id, None)
        result = replace(result, operation_id=operation.operation_id, revision=self.revision)
        self.results[operation.operation_id] = result
        while len(self.results) > 256:
            self.results.popitem(last=False)
        return result

    async def close(self):
        self.invalidate()
        await asyncio.gather(*list(self.tasks.values()), return_exceptions=True)
