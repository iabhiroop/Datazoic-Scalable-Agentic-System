"""Provider adapter stubs; add concrete integrations only after operation guard tests."""

from typing import Any

from app.tools.contracts import ToolContext, ToolOutcome


class ProviderAdapter:
    async def invoke(self, arguments: dict[str, Any], context: ToolContext) -> ToolOutcome:
        raise NotImplementedError(
            "Allowlisted endpoint, schema validation and explicit write retry policy"
        )

    async def reconcile(self, operation: dict[str, Any], context: ToolContext) -> ToolOutcome:
        raise NotImplementedError("Provider-specific status lookup or explicit unsupported outcome")


def normalize_error(error: Exception, operation_type: str) -> ToolOutcome:
    """Timeout/write 5xx may be unknown; HTTP status alone cannot prove absence of effects."""
    raise NotImplementedError("Provider-specific error classification")
