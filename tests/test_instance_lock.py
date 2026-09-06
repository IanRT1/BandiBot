"""Offline tests for the single-process runtime guard."""

import pytest

from core.instance_lock import InstanceAlreadyRunning, SingleInstanceLock


def test_instance_lock_rejects_a_second_owner(tmp_path):
    path = tmp_path / "bandibot.lock"
    first = SingleInstanceLock(path)
    second = SingleInstanceLock(path)

    first.acquire()
    try:
        with pytest.raises(InstanceAlreadyRunning):
            second.acquire()
    finally:
        first.release()

    second.acquire()
    second.release()


def test_instance_lock_release_is_idempotent(tmp_path):
    lock = SingleInstanceLock(tmp_path / "bandibot.lock")

    lock.release()
    lock.acquire()
    lock.release()
    lock.release()
