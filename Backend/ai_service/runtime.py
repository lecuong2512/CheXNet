"""Bound CPU work while keeping FastAPI's event loop available for health checks."""
import asyncio

import anyio
from fastapi import HTTPException


class InferenceRunner:
    def __init__(self, capacity=3, queue_timeout=30):
        self.capacity = capacity
        self.queue_timeout = queue_timeout
        self.pending = 0
        self.active = 0
        self._slot = asyncio.Semaphore(1)

    async def run(self, operation):
        if self.pending >= self.capacity:
            raise HTTPException(503, "AI đang bận. Vui lòng thử lại sau.", headers={"Retry-After": "5"})
        self.pending += 1
        acquired = False
        try:
            try:
                await asyncio.wait_for(self._slot.acquire(), self.queue_timeout)
                acquired = True
            except TimeoutError as exc:
                raise HTTPException(503, "Hết thời gian chờ xử lý AI. Vui lòng thử lại.") from exc
            self.active += 1
            try:
                return await anyio.to_thread.run_sync(lambda: asyncio.run(operation()))
            finally:
                self.active -= 1
        finally:
            if acquired:
                self._slot.release()
            self.pending -= 1

    def status(self):
        return {"active": self.active, "waiting": self.pending - self.active, "capacity": self.capacity}
