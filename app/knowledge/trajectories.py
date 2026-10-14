"""Sanitized, versioned examples; unresolved effects cannot become successful exemplars."""

from typing import Any
from uuid import UUID

from app.security import Principal


class TrajectoryService:
    async def summarize(self, run_id: UUID, principal: Principal) -> dict[str, Any]:
        raise NotImplementedError(
            "Summarize verified outcome, versioned tools and recovery evidence"
        )

    async def publish(self, summary: dict[str, Any], principal: Principal) -> None:
        raise NotImplementedError(
            "Review/evaluate before publication; strip resource IDs and approvals"
        )
