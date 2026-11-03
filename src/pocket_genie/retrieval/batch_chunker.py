from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")

DEFAULT_BATCH_CHUNK_SIZE = 12
DEFAULT_CHUNK_RETRY_ATTEMPTS = 2


def slice_into_batches(items: Sequence[T], chunk_size: int = DEFAULT_BATCH_CHUNK_SIZE) -> list[list[T]]:
    step = max(1, int(chunk_size))
    return [list(items[i : i + step]) for i in range(0, len(items), step)]


def run_chunked_batches(
    items: Sequence[T],
    worker: Callable[[list[T]], list[R]],
    chunk_size: int = DEFAULT_BATCH_CHUNK_SIZE,
    retry_attempts: int = DEFAULT_CHUNK_RETRY_ATTEMPTS,
) -> list[R]:
    """Execute batch workers across bounded chunks with per-chunk retry recovery."""
    results: list[R] = []
    for batch in slice_into_batches(items, chunk_size=chunk_size):
        last_err: Exception | None = None
        for _attempt in range(max(1, retry_attempts + 1)):
            try:
                results.extend(worker(batch))
                last_err = None
                break
            except Exception as exc:
                last_err = exc
        if last_err is not None:
            raise last_err
    return results


async def run_chunked_batches_async(
    items: Sequence[T],
    worker: Callable[[list[T]], Awaitable[list[R]]],
    chunk_size: int = DEFAULT_BATCH_CHUNK_SIZE,
    retry_attempts: int = DEFAULT_CHUNK_RETRY_ATTEMPTS,
) -> list[R]:
    """Execute async batch workers in parallel via asyncio.gather with per-chunk retries."""
    batches = slice_into_batches(items, chunk_size=chunk_size)

    async def _run_single_batch(batch: list[T]) -> list[R]:
        last_err: Exception | None = None
        for _attempt in range(max(1, retry_attempts + 1)):
            try:
                return await worker(batch)
            except Exception as exc:
                last_err = exc
        assert last_err is not None
        raise last_err

    nested = await asyncio.gather(*(_run_single_batch(b) for b in batches))
    out: list[R] = []
    for group in nested:
        out.extend(group)
    return out
