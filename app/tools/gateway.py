"""Verified reads and operation-bound, single-dispatch sandbox writes."""

import hashlib
import json
from decimal import Decimal, InvalidOperation
from typing import Any
from uuid import UUID, uuid4, uuid5

from fastapi import HTTPException
from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import ValidationError

from app.agent.planning import canonical_payload_hash, resolve_arguments
from app.agent.state import AgentState, PlanStep
from app.db.models import OperationState, ToolOperation
from app.db.repositories import OperationRepository, RunRepository
from app.db.session import Database
from app.security import Principal, TokenAuthenticator
from app.tools.contracts import ToolCard, ToolContext, ToolOutcome
from app.tools.registry import ToolRegistry


class ExecutionService:
    def __init__(
        self, database: Database, registry: ToolRegistry, authenticator: TokenAuthenticator
    ) -> None:
        self.database, self.registry, self.authenticator = database, registry, authenticator

    async def execute_step(
        self, step: PlanStep, state: AgentState, principal: Principal
    ) -> ToolOutcome:
        """Refresh scopes, resolve inputs and invoke a verified tool."""
        principal = await self.authenticator.refresh(principal)
        card = await self.registry.get(step.tool_id, step.schema_version, principal)
        async with self.database.session() as session:
            run = await RunRepository(session).get(UUID(state["run_id"]), principal)
            if run.user_id != principal.user_id:
                raise HTTPException(403, "Run execution requires its owner")
            if run.cancel_requested:
                return ToolOutcome(
                    state="failed", data={}, error_code="cancelled", effect_known_absent=True
                )
            conversation_id = run.conversation_id
        arguments = resolve_arguments(step, card, state.get("results", {}))
        adapter = self.registry.adapters.get(card.adapter_key)
        if adapter is None:
            raise ValueError("Executable adapter is unavailable")
        if card.operation_type == "write":
            return await self._execute_write(
                step, card, arguments, state, principal, adapter, conversation_id
            )
        outcome = await adapter.invoke(
            arguments,
            ToolContext(
                principal.tenant_id,
                principal.user_id,
                UUID(state["run_id"]),
                uuid4(),
                None,
                conversation_id,
            ),
        )
        principal = await self.authenticator.refresh(principal)
        await self.registry.get(step.tool_id, step.schema_version, principal)
        if outcome.state == "succeeded":
            await adapter.authorize_result(
                arguments,
                outcome.data,
                ToolContext(
                    principal.tenant_id,
                    principal.user_id,
                    UUID(state["run_id"]),
                    uuid4(),
                    None,
                    conversation_id,
                ),
            )
        return await self.verify(card, outcome, principal)

    @staticmethod
    def write_identity(step: PlanStep, run_id: UUID) -> tuple[str, UUID]:
        canonical = json.dumps(step.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        action_ref = hashlib.sha256(canonical.encode()).hexdigest()
        return action_ref, uuid5(run_id, action_ref)

    async def prepare_write(
        self, step: PlanStep, card: ToolCard, state: AgentState, principal: Principal
    ) -> ToolOperation:
        if card.operation_type != "write":
            raise ValueError("Only writes require operation preparation")
        arguments = resolve_arguments(step, card, state.get("results", {}))
        run_id = UUID(state["run_id"])
        action_ref, operation_id = self.write_identity(step, run_id)
        input_hash = canonical_payload_hash(arguments)
        async with self.database.session() as session:
            existing = await session.get(ToolOperation, operation_id)
            if existing is not None:
                if (
                    existing.tenant_id != principal.tenant_id
                    or existing.action_ref != action_ref
                    or existing.input_hash != input_hash
                ):
                    raise ValueError("Prepared business action changed")
                if existing.status != OperationState.PREPARED:
                    return existing
        await self.check_preconditions(card, arguments, principal)
        operation = ToolOperation(
            tenant_id=principal.tenant_id,
            run_id=run_id,
            operation_id=operation_id,
            action_ref=action_ref,
            tool_id=card.tool_id,
            schema_version=card.schema_version,
            operation_type="write",
            provider=card.provider,
            provider_account_key=card.provider_account_key,
            input_hash=input_hash,
            input_payload=arguments,
            status=OperationState.PREPARED,
            idempotency_key=str(operation_id),
        )
        async with self.database.session() as session, session.begin():
            run = await RunRepository(session).get(run_id, principal)
            if run.cancel_requested or run.user_id != principal.user_id:
                raise ValueError("Run is cancelled or no longer owned by caller")
            return await OperationRepository(session).prepare(principal, operation)

    async def _execute_write(
        self, step, card, arguments, state, principal, adapter, conversation_id
    ) -> ToolOutcome:
        action_ref, operation_id = self.write_identity(step, UUID(state["run_id"]))
        async with self.database.session() as session:
            operation = await OperationRepository(session).get(principal.tenant_id, operation_id)
            if operation.action_ref != action_ref or operation.input_hash != canonical_payload_hash(
                arguments
            ):
                raise ValueError("Prepared business action no longer matches its payload")
            if operation.status == OperationState.SUCCEEDED:
                return ToolOutcome(state="succeeded", data=operation.result_payload or {})
            if operation.status == OperationState.FAILED:
                return ToolOutcome(
                    state="failed",
                    data={},
                    error_code=(operation.error_payload or {}).get("code", "paypal_write_failed"),
                    effect_known_absent=True,
                )
            if operation.status != OperationState.PREPARED:
                return ToolOutcome(state="unknown", data={}, error_code="write_requires_review")
        if state.get("approved_operation_id") != str(operation_id):
            return ToolOutcome(state="failed", data={}, error_code="approval_required")
        await self.check_preconditions(card, arguments, principal)
        principal = await self.authenticator.refresh(principal)
        await self.registry.get(card.tool_id, card.schema_version, principal)
        async with self.database.session() as session, session.begin():
            operation = await OperationRepository(session).get(principal.tenant_id, operation_id)
            if not await OperationRepository(session).mark_dispatching(
                operation, operation.row_version
            ):
                return ToolOutcome(state="unknown", data={}, error_code="write_requires_review")
        context = ToolContext(
            principal.tenant_id,
            principal.user_id,
            UUID(state["run_id"]),
            operation_id,
            str(operation_id),
            conversation_id,
        )
        outcome = await adapter.invoke(arguments, context)
        if outcome.state == "succeeded":
            outcome = await self.verify(card, outcome, principal)
            if outcome.state != "succeeded":
                outcome = ToolOutcome(state="unknown", data={}, error_code="invalid_write_output")
        async with self.database.session() as session, session.begin():
            operation = await OperationRepository(session).get(principal.tenant_id, operation_id)
            await OperationRepository(session).save_outcome(operation, outcome)
        return outcome

    async def reconcile(self, operation_id: UUID, principal: Principal) -> ToolOutcome:
        raise NotImplementedError(
            "Lookup existing operation; respect consistency and idempotency expiry"
        )

    async def check_preconditions(
        self, card: ToolCard, arguments: dict[str, Any], principal: Principal
    ) -> None:
        from app.tools.paypal import adapter_key, validate_endpoint_card

        if card.precondition_key == "sandbox_approved_write":
            validate_endpoint_card(card)
            settings = self.database.settings
            if (
                not settings.paypal_sandbox_writes_enabled
                or settings.paypal_sandbox_tenant_id != principal.tenant_id
                or card.operation_type != "write"
                or "paypal:write" not in principal.scopes
            ):
                raise ValueError("Sandbox write is not enabled for this account")
            Draft202012Validator(card.input_schema, format_checker=FormatChecker()).validate(
                arguments
            )
            return

        create = adapter_key("POST", "/v2/invoicing/invoices")
        send = adapter_key("POST", "/v2/invoicing/invoices/{invoice_id}/send")
        if card.precondition_key != "paypal_reviewed_write" or card.adapter_key not in {
            create,
            send,
        }:
            raise ValueError("Write policy is unavailable")
        if card.adapter_key == create:
            body = arguments.get("body")
            if not isinstance(body, dict) or not all(
                isinstance(body.get(key), list) and body[key]
                for key in ("items", "primary_recipients")
            ):
                raise ValueError("Invoice items and recipients are required")
            if len(json.dumps(arguments, ensure_ascii=False)) > 8192:
                raise ValueError("Invoice body exceeds configured safe size")
            for recipient in body["primary_recipients"]:
                billing = recipient.get("billing_info") if isinstance(recipient, dict) else None
                if not isinstance(billing, dict) or not isinstance(
                    billing.get("email_address"), str
                ):
                    raise TypeError("Recipient email is required")
            for item in body["items"]:
                if not isinstance(item, dict) or not item.get("name"):
                    raise ValueError("Invoice item name is required")
                amount = item.get("unit_amount")
                if not isinstance(amount, dict):
                    raise TypeError("Invoice item amount is required")
                try:
                    value = Decimal(amount["value"])
                except (KeyError, TypeError, InvalidOperation):
                    raise ValueError("Invoice amount must be a decimal string") from None
                if not isinstance(amount["value"], str) or not value.is_finite() or value <= 0:
                    raise ValueError("Invoice amount must be a positive decimal string")
                if not isinstance(amount.get("currency_code"), str):
                    raise TypeError("Invoice currency is required")
        else:
            if not isinstance(arguments.get("invoice_id"), str) or not arguments["invoice_id"]:
                raise ValueError("Invoice ID is required")
            if len(json.dumps(arguments, ensure_ascii=False)) > 8192:
                raise ValueError("Invoice send request exceeds configured safe size")
            from app.tools.paypal import PayPalEndpointAdapter

            adapter = self.registry.adapters.get(
                adapter_key("GET", "/v2/invoicing/invoices/{invoice_id}")
            )
            if not isinstance(adapter, PayPalEndpointAdapter):
                raise ValueError("Invoice status lookup is unavailable")
            outcome = await adapter.invoke(
                {"invoice_id": arguments["invoice_id"]},
                ToolContext(principal.tenant_id, principal.user_id, uuid4(), uuid4(), None),
            )
            if outcome.state != "succeeded" or outcome.data["response"].get("status") != "DRAFT":
                raise ValueError("Invoice must currently be a draft")

    async def verify(
        self, card: ToolCard, outcome: ToolOutcome, principal: Principal
    ) -> ToolOutcome:
        if outcome.state == "succeeded":
            try:
                Draft202012Validator(card.output_schema, format_checker=FormatChecker()).validate(
                    outcome.data
                )
            except ValidationError:
                return ToolOutcome(state="failed", data={}, error_code="invalid_tool_output")
        return outcome

    async def credentials(self, card: ToolCard, principal: Principal) -> Any:
        raise NotImplementedError(
            "Trusted tenant/provider secret resolution, never model credentials"
        )
