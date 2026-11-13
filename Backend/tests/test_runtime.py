import asyncio
import threading
import unittest

from fastapi import HTTPException
from Backend.ai_service.runtime import InferenceRunner


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_one_worker_bounded_queue_and_responsive_event_loop(self):
        runner = InferenceRunner(capacity=2, queue_timeout=1)
        release = threading.Event()
        started = threading.Event()

        async def operation():
            started.set()
            release.wait(5)
            return "done"

        first = asyncio.create_task(runner.run(operation))
        try:
            self.assertTrue(await asyncio.to_thread(started.wait, 2))
            self.assertEqual(runner.status()["active"], 1)
            second = asyncio.create_task(runner.run(operation))
            await asyncio.sleep(0.01)
            self.assertEqual(runner.status()["waiting"], 1)
            with self.assertRaises(HTTPException) as error:
                await runner.run(operation)
            self.assertEqual(error.exception.status_code, 503)
            release.set()
            self.assertEqual(await first, "done")
            self.assertEqual(await second, "done")
            self.assertEqual(runner.status(), {"active": 0, "waiting": 0, "capacity": 2})
        finally:
            release.set()
            await first

    async def test_failed_operation_releases_worker(self):
        runner = InferenceRunner()
        async def fail():
            raise ValueError("controlled failure")
        with self.assertRaises(ValueError):
            await runner.run(fail)
        self.assertEqual(runner.pending, 0)
        self.assertEqual(runner.active, 0)


if __name__ == "__main__":
    unittest.main()
