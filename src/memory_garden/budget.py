"""Cooperative deadlines shared by tool execution and every network retry."""
from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_deadline: ContextVar[float | None] = ContextVar('memory_garden_deadline', default=None)


class BudgetExceeded(TimeoutError):
    pass


def remaining(default: float) -> float:
    deadline = _deadline.get()
    seconds = default if deadline is None else min(default, deadline - time.monotonic())
    if seconds <= 0:
        raise BudgetExceeded('本次核对已达到时间预算。')
    return seconds


@contextmanager
def budget_scope(seconds: float) -> Iterator[None]:
    deadline = time.monotonic() + seconds
    parent = _deadline.get()
    token = _deadline.set(min(deadline, parent) if parent is not None else deadline)
    try:
        remaining(seconds)
        yield
    finally:
        _deadline.reset(token)
