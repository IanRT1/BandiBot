import asyncio

from music.operations import GuildOperations
from music.results import PlayResult


def test_preparation_is_concurrent_but_commits_follow_acceptance_order():
    async def run():
        operations = GuildOperations()
        started = [asyncio.Event(), asyncio.Event()]
        release = [asyncio.Event(), asyncio.Event()]
        committed = []

        async def prepare(index):
            started[index].set()
            await release[index].wait()
            return index

        async def commit(index, operation):
            committed.append(index)
            return PlayResult("queued", title=str(index))

        first = asyncio.create_task(operations.submit(lambda: prepare(0), commit))
        await started[0].wait()
        second = asyncio.create_task(operations.submit(lambda: prepare(1), commit))
        await started[1].wait()
        release[1].set()
        await asyncio.sleep(0)
        assert committed == []
        release[0].set()
        await asyncio.gather(first, second)
        assert committed == [0, 1]
    asyncio.run(run())


def test_stop_releases_pending_work_and_duplicate_requests_commit_once():
    async def run():
        operations = GuildOperations()
        entered = asyncio.Event()
        commits = []

        async def slow():
            entered.set()
            await asyncio.Event().wait()

        async def commit(value, operation):
            commits.append(value)
            return PlayResult("queued", title="song")

        task = asyncio.create_task(operations.submit(slow, commit, operation_id="same"))
        await entered.wait()
        duplicate = asyncio.create_task(operations.submit(slow, commit, operation_id="same"))
        operations.invalidate()
        results = await asyncio.wait_for(asyncio.gather(task, duplicate), 1)
        assert all(result.error_code == "cancelled" for result in results)
        assert commits == []
        assert not operations.pending

        async def ready():
            return "ready"
        await operations.submit(ready, commit, operation_id="next")
        await operations.submit(ready, commit, operation_id="next")
        assert commits == ["ready"]
    asyncio.run(run())


def test_targeted_pending_cancellation_keeps_other_requests():
    async def run():
        operations = GuildOperations()
        entered = asyncio.Event()
        committed = []
        async def slow():
            entered.set()
            await asyncio.Event().wait()
        async def ready():
            return "second"
        async def commit(value, operation):
            committed.append(value)
            return PlayResult("queued")
        first = asyncio.create_task(operations.submit(slow, commit, operation_id="first"))
        await entered.wait()
        second = asyncio.create_task(operations.submit(ready, commit, operation_id="second"))
        assert operations.cancel("first")
        results = await asyncio.wait_for(asyncio.gather(first, second), 1)
        assert results[0].error_code == "cancelled"
        assert results[1].status == "queued"
        assert committed == ["second"]
    asyncio.run(run())
