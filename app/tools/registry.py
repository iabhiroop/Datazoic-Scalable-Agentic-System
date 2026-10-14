"""Tenant-scoped immutable publication with server-owned adapter/validator allowlists."""

import math
from typing import Any
from urllib.parse import unquote

from fastapi import HTTPException
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert

from app.db.models import RegistryTool, SystemVersions
from app.db.session import Database
from app.security import Principal
from app.tools.contracts import ToolAdapter, ToolCard


class ToolRegistry:
    def __init__(
        self,
        adapters: dict[str, ToolAdapter] | set[str],
        *,
        database: Database,
        verification_keys: set[str],
        precondition_keys: set[str] | None = None,
        reconciliation_keys: set[str] | None = None,
    ) -> None:
        # Admin publication needs the approved names, not live provider credentials.
        self.adapters = dict(adapters) if isinstance(adapters, dict) else {}
        self.allowed_adapters = frozenset(adapters)
        self.database = database
        self.verification_keys = frozenset(verification_keys)
        self.precondition_keys = frozenset(precondition_keys or ())
        self.reconciliation_keys = frozenset(reconciliation_keys or ())

    @staticmethod
    def require_admin(principal: Principal) -> None:
        if "tools:manage" not in principal.scopes:
            raise HTTPException(403, "Tool management permission is required")

    async def get(self, tool_id: str, version: str, principal: Principal) -> ToolCard:
        async with self.database.session() as session:
            row = await session.get(RegistryTool, (principal.tenant_id, tool_id, version))
            if row is None or row.status != "published":
                raise HTTPException(404, "Published tool version not found")
            card = ToolCard.model_validate(row.card).model_copy(update={"status": row.status})
        if not set(card.required_scopes).issubset(principal.scopes):
            raise HTTPException(403, "Required tool permissions are missing")
        return await self.validate_import(card)

    @staticmethod
    def validate_schema(schema: dict[str, Any]) -> None:
        """Accept closed local schema references; never fetch imported URLs."""
        try:
            Draft202012Validator.check_schema(schema)
        except SchemaError:
            raise ValueError("Invalid tool JSON Schema") from None

        def walk(value: Any) -> None:
            if isinstance(value, list):
                for item in value:
                    walk(item)
            elif isinstance(value, dict):
                if (
                    "$schema" in value
                    and value["$schema"] != "https://json-schema.org/draft/2020-12/schema"
                ):
                    raise ValueError("Tool schemas must use JSON Schema draft 2020-12")
                if "$id" in value or "$dynamicRef" in value or "$anchor" in value:
                    raise ValueError("Tool schemas support local JSON Pointer references only")
                if "$ref" in value:
                    reference = value["$ref"]
                    if not isinstance(reference, str) or not reference.startswith("#"):
                        raise ValueError("External schema references are not allowed")
                    pointer = unquote(reference[1:])
                    if pointer and not pointer.startswith("/"):
                        raise ValueError("Schema reference must be a local JSON Pointer")
                    target: Any = schema
                    for part in pointer[1:].split("/") if pointer else []:
                        key = part.replace("~1", "/").replace("~0", "~")
                        if isinstance(target, dict) and key in target:
                            target = target[key]
                        else:
                            raise ValueError("Unresolved local schema reference")
                    if not isinstance(target, (dict, bool)):
                        raise ValueError("Schema reference does not target a schema")
                for item in value.values():
                    walk(item)

        walk(schema)

    @classmethod
    def validate_draft(cls, card: ToolCard) -> ToolCard:
        """Validate a review artifact without granting execution eligibility."""
        if not card.tool_id.strip() or not card.schema_version.strip():
            raise ValueError("Tool identity and version must not be empty")
        if card.idempotency.supported and not (
            card.idempotency.header_or_field
            and card.idempotency.retention_seconds
            and card.idempotency.request_consistency_rules
        ):
            raise ValueError("Idempotency metadata is incomplete")
        for schema in (card.input_schema, card.output_schema):
            cls.validate_schema(schema)
        return card.model_copy(deep=True)

    async def validate_import(self, card: ToolCard) -> ToolCard:
        card = self.validate_draft(card)
        if card.endpoint is not None:
            from app.tools.paypal import validate_endpoint_card

            validate_endpoint_card(card)
        if card.adapter_key not in self.allowed_adapters:
            raise ValueError("Tool adapter is not enabled in this environment")
        if card.verification_key not in self.verification_keys:
            raise ValueError("Tool verification policy is not approved")
        if card.precondition_key and card.precondition_key not in self.precondition_keys:
            raise ValueError("Tool precondition policy is not approved")
        if card.operation_type == "write" and not card.precondition_key:
            raise ValueError("Write tools require an approved precondition policy")
        if card.reconciliation_key and card.reconciliation_key not in self.reconciliation_keys:
            raise ValueError("Tool reconciliation policy is not approved")
        return card

    async def save_draft(
        self,
        card: ToolCard,
        principal: Principal,
        embedding: list[float],
        *,
        draft_only: bool = False,
    ) -> None:
        """Store an already computed embedding; no network calls in the transaction."""
        self.require_admin(principal)
        card = self.validate_draft(card) if draft_only else await self.validate_import(card)
        settings = self.database.settings
        if (
            not card.embedding_model
            or not card.embedding_version
            or (
                card.embedding_model != settings.embedding_model
                or card.embedding_version != settings.embedding_version
            )
        ):
            raise ValueError("Tool embedding model/version must match configuration")
        if (
            len(embedding) != settings.embedding_dimension
            or not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                for value in embedding
            )
            or not any(embedding)
        ):
            raise ValueError(
                "Tool embedding must be a finite nonzero vector of configured dimension"
            )
        card = card.model_copy(update={"status": "draft"})
        searchable = "\n".join(
            [card.name, card.description, *card.synonyms, *card.entities, card.action]
        )
        async with self.database.session() as session, session.begin():
            inserted = await session.scalar(
                insert(RegistryTool)
                .values(
                    tenant_id=principal.tenant_id,
                    tool_id=card.tool_id,
                    schema_version=card.schema_version,
                    card=card.model_dump(mode="json"),
                    searchable_text=searchable,
                    embedding=embedding,
                    embedding_version=card.embedding_version,
                    is_current=False,
                    status="draft",
                )
                .on_conflict_do_nothing()
                .returning(RegistryTool.tool_id)
            )
            if inserted is None:
                raise HTTPException(409, "Tool version already exists; import a new version")

    @staticmethod
    async def lock_revision(session, principal: Principal) -> SystemVersions:
        await session.execute(
            insert(SystemVersions).values(tenant_id=principal.tenant_id).on_conflict_do_nothing()
        )
        return await session.scalar(
            select(SystemVersions)
            .where(SystemVersions.tenant_id == principal.tenant_id)
            .with_for_update()
        )

    async def publish(self, card: ToolCard, principal: Principal) -> None:
        self.require_admin(principal)
        card = await self.validate_import(card)
        async with self.database.session() as session, session.begin():
            revision = await self.lock_revision(session, principal)
            row = await session.get(
                RegistryTool,
                (principal.tenant_id, card.tool_id, card.schema_version),
                with_for_update=True,
            )
            if row is None:
                raise HTTPException(404, "Import a draft before publication")
            stored = ToolCard.model_validate(row.card)
            if stored.model_copy(update={"status": "draft"}) != card.model_copy(
                update={"status": "draft"}
            ):
                raise HTTPException(409, "Reviewed card does not match the imported version")
            if row.status == "revoked":
                raise HTTPException(409, "Revoked tool versions cannot be republished")
            if row.status == "published":
                return  # An old pinned version must not become current on replay.
            if (
                row.embedding is None
                or row.embedding_version != self.database.settings.embedding_version
            ):
                raise ValueError("Tool embedding is missing or incompatible")
            await session.execute(
                update(RegistryTool)
                .where(
                    RegistryTool.tenant_id == principal.tenant_id,
                    RegistryTool.tool_id == card.tool_id,
                )
                .values(is_current=False)
            )
            row.status = "published"
            row.is_current = True
            row.card = stored.model_copy(update={"status": "published"}).model_dump(mode="json")
            revision.registry_revision += 1
            await session.flush()

    async def revoke(self, tool_id: str, version: str, principal: Principal) -> None:
        self.require_admin(principal)
        async with self.database.session() as session, session.begin():
            revision = await self.lock_revision(session, principal)
            row = await session.get(
                RegistryTool, (principal.tenant_id, tool_id, version), with_for_update=True
            )
            if row is None:
                raise HTTPException(404, "Tool version not found")
            if row.status == "revoked":
                return
            row.status = "revoked"
            row.is_current = False
            row.card = {**row.card, "status": "revoked"}
            revision.registry_revision += 1
            await session.flush()
