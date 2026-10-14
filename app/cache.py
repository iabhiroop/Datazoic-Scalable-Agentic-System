"""Optional exact-query Redis caches; never queue execution or cache write outcomes."""

from typing import Any

from app.security import Principal


class CacheService:
    async def get(
        self, namespace: str, query_hash: str, principal: Principal, revisions: dict[str, int]
    ) -> Any | None:
        raise NotImplementedError(
            "Include authorization fingerprint and registry/ACL/knowledge revisions"
        )

    async def put(
        self,
        namespace: str,
        query_hash: str,
        value: Any,
        principal: Principal,
        revisions: dict[str, int],
        ttl_seconds: int,
    ) -> None:
        raise NotImplementedError("Safe exact caches; Redis loss falls back to database/retrieval")

    async def invalidate(self, principal: Principal, namespace: str) -> None:
        raise NotImplementedError("Revision-based scoped invalidation")
