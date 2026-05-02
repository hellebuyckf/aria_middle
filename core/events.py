import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

ProgressCallback = Callable[[dict[str, Any]], Awaitable[None]]
_callbacks: dict[str, ProgressCallback] = {}


def register(session_id: str, cb: ProgressCallback) -> None:
    _callbacks[session_id] = cb


def unregister(session_id: str) -> None:
    _callbacks.pop(session_id, None)


async def emit(session_id: str, event: dict[str, Any]) -> None:
    cb = _callbacks.get(session_id)
    if cb:
        await cb(event)


def tick(
    session_id: str,
    etape: str,
    pct_start: int,
    pct_end: int,
    duration_s: float,
    message: str = "",
) -> asyncio.Task:
    """Lance une Task qui émet des events de progression réguliers jusqu'à pct_end."""
    steps = max(1, pct_end - pct_start - 1)
    interval = duration_s / (steps + 1)

    async def _run() -> None:
        for pct in range(pct_start + 1, pct_end):
            await asyncio.sleep(interval)
            await emit(
                session_id,
                {"type": "progress", "etape": etape, "pct": pct, "message": message},
            )

    return asyncio.create_task(_run())
