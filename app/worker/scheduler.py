"""PostgreSQL queue claims; no provider call occurs inside a claim transaction."""

from uuid import UUID

from sqlalchemy import func, update

from app.db.models import WorkflowRun
from app.db.repositories import RunRepository


class Scheduler:
    def __init__(self, database):
        self.database = database

    async def claim(self, worker_id: UUID):
        async with self.database.session() as session, session.begin():
            return await RunRepository(session).claim_next(str(worker_id))

    async def heartbeat(self, worker_id: UUID, run_id: UUID):
        async with self.database.session() as session, session.begin():
            result = await session.execute(
                update(WorkflowRun)
                .where(
                    WorkflowRun.run_id == run_id,
                    WorkflowRun.worker_id == str(worker_id),
                    WorkflowRun.status == "running",
                )
                .values(heartbeat_at=func.now(), row_version=WorkflowRun.row_version + 1)
            )
            return result.rowcount == 1

    async def enqueue_due_waits(self):
        raise NotImplementedError("Durable retry/reconciliation timers are pending")
