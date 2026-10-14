"""Administrative setup, collection onboarding and reviewed tool publication."""

import argparse
import asyncio
import json
from pathlib import Path
from uuid import uuid4

from app.agent.gateway import LLMError, create_llm_gateway
from app.agent.gemini import GeminiEmbeddingProvider
from app.agent.llm import ModelClient
from app.db.models import RegistryTool
from app.db.session import Database, event_loop_factory
from app.knowledge.context import ContextBuilder
from app.knowledge.ingestion import IngestionService
from app.knowledge.retrieval import RetrievalService
from app.observability import Telemetry
from app.security import TokenAuthenticator
from app.settings import Settings
from app.storage import PayloadStore, StorageError
from app.tools.contracts import ToolCard, ToolContext
from app.tools.paypal import (
    REVIEWED_INVOICE_ENDPOINTS,
    adapter_key,
    load_endpoint_catalog,
    save_endpoint_catalog,
)
from app.tools.rag_pipeline import RagPipeline
from app.tools.registry import ToolRegistry


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Agent administration")
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("import-tool", "import-document"):
        command = commands.add_parser(name)
        command.add_argument("path", type=Path)
        if name == "import-tool":
            command.add_argument("--token-file", type=Path, default=Path("runtime/local-auth.json"))
            command.add_argument("--draft-only", action="store_true")
        if name == "import-document":
            command.add_argument("--corpus", required=True)
            command.add_argument("--source-key")
            command.add_argument("--scope", action="append")
            command.add_argument("--token-file", type=Path, default=Path("runtime/local-auth.json"))
    for name in ("publish-tool", "revoke-tool"):
        command = commands.add_parser(name)
        command.add_argument("tool_id")
        command.add_argument("--version", required=True)
        command.add_argument("--token-file", type=Path, default=Path("runtime/local-auth.json"))
    commands.add_parser("curate-trajectories")
    commands.add_parser("setup-checkpoints")
    commands.add_parser("setup-storage")
    query = commands.add_parser(
        "query-knowledge", help="Run standalone authorized document retrieval/RAG"
    )
    query.add_argument("query")
    query.add_argument("--corpus", action="append", default=[])
    query.add_argument("--mode", choices=("context", "answer"), default="context")
    query.add_argument("--token-file", type=Path, default=Path("runtime/local-auth.json"))
    commands.add_parser("cleanup-payloads")
    convert = commands.add_parser(
        "convert-collection", aliases=["convert-paypal"], help="Generate inert Postman tool drafts"
    )
    convert.add_argument("path", type=Path)
    convert.add_argument("--output", type=Path, default=Path("runtime/tool-drafts"))
    onboard = commands.add_parser(
        "onboard-collection", help="Import every collection tool resumably"
    )
    onboard.add_argument("path", type=Path)
    onboard.add_argument("--output", type=Path, default=Path("runtime/tool-drafts"))
    onboard.add_argument("--token-file", type=Path, default=Path("runtime/local-auth.json"))
    onboard.add_argument("--publish-reads", action="store_true")
    onboard.add_argument(
        "--enable-supported",
        action="store_true",
        help="Review taxonomy and publish supported cards after deterministic checks",
    )
    onboard.add_argument(
        "--reviewer",
        choices=("gemini", "manual"),
        default="gemini",
        help="Use Gemini metadata review or the local human-reviewed taxonomy rules",
    )
    onboard.add_argument("--specs", type=Path, default=Path("runtime/api-specs"))
    onboard.add_argument("--fetch-specs", action="store_true")
    onboard.add_argument("--distance-threshold", type=float)
    onboard.add_argument(
        "--evaluation-file",
        type=Path,
        default=Path("resources/taxonomy/retrieval-evaluation.json"),
    )
    commands.add_parser(
        "check-llm", help="Make one small call to verify the configured LLM provider"
    )
    commands.add_parser(
        "check-langfuse", help="Authenticate and send one synthetic trace; no LLM call"
    )
    return root


def main() -> None:
    root = parser()
    arguments = root.parse_args()
    if arguments.command in ("convert-collection", "convert-paypal"):
        from app.tools.paypal_collection import convert

        cards, inventory = convert(arguments.path)
        arguments.output.mkdir(parents=True, exist_ok=True)
        for card in cards:
            target = arguments.output / f"{card.tool_id}.json"
            target.write_text(card.model_dump_json(indent=2) + "\n", encoding="utf-8")
        (arguments.output / "inventory.json").write_text(
            json.dumps(inventory, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Generated {len(cards)} draft tools in {arguments.output}")
        return
    if arguments.command == "onboard-collection":
        try:
            with asyncio.Runner(loop_factory=event_loop_factory) as runner:
                rows = runner.run(collection_admin(arguments))
        except Exception:  # noqa: BLE001 -- keep collection and infrastructure data private
            root.exit(status=2, message="Collection onboarding failed; inspect configuration.\n")
        if any(row["outcome"] == "failed" for row in rows):
            root.exit(status=2, message="Some tools failed; rerun to resume from saved imports.\n")
        return
    if arguments.command in ("import-tool", "publish-tool", "revoke-tool"):
        try:
            with asyncio.Runner(loop_factory=event_loop_factory) as runner:
                runner.run(tool_admin(arguments))
        except Exception:  # noqa: BLE001 -- never echo imported cards or infrastructure secrets
            root.exit(
                status=2,
                message="Tool administration failed; check permissions and card/version.\n",
            )
        return
    if arguments.command == "query-knowledge":
        try:
            with asyncio.Runner(loop_factory=event_loop_factory) as runner:
                runner.run(query_knowledge(arguments))
        except Exception:  # noqa: BLE001 -- do not print unsafe infrastructure details
            root.exit(status=2, message="Knowledge query failed; check configuration and access.\n")
        return
    if arguments.command in ("setup-storage", "import-document"):
        try:
            with asyncio.Runner(loop_factory=event_loop_factory) as runner:
                runner.run(document_admin(arguments))
        except (ValueError, OSError, StorageError, LLMError):
            root.exit(
                status=2,
                message="Document/storage command failed; check configuration and input.\n",
            )
        except Exception:  # noqa: BLE001 -- never echo provider/database errors with credentials
            root.exit(status=2, message="Document/storage operation failed.\n")
        return
    if arguments.command == "check-langfuse":
        try:
            with asyncio.Runner(loop_factory=event_loop_factory) as runner:
                runner.run(check_langfuse())
        except ValueError as error:
            root.exit(status=2, message=f"Langfuse check failed: {error}\n")
        return
    if arguments.command == "check-llm":
        try:
            with asyncio.Runner(loop_factory=event_loop_factory) as runner:
                runner.run(check_llm())
        except (LLMError, ValueError) as error:
            root.exit(status=2, message=f"LLM check failed: {error}\n")
        return
    if arguments.command == "setup-checkpoints":
        with asyncio.Runner(loop_factory=event_loop_factory) as runner:
            runner.run(setup_checkpoints())
        return
    root.exit(status=2, message=f"{arguments.command}: not implemented; no changes made.\n")


async def setup_checkpoints() -> None:
    database = Database(Settings())
    try:
        await database.open(require_checkpoints=False)
        await database.setup_checkpoints()
    finally:
        await database.close()


def admin_registry(database: Database, settings: Settings) -> ToolRegistry:
    return ToolRegistry(
        {"system_search", "rag_pipeline"}
        | {
            adapter_key(method, path)
            for method, path in REVIEWED_INVOICE_ENDPOINTS
            if method == "GET" or settings.paypal_sandbox_writes_enabled
        }
        | {
            key
            for key, endpoint in load_endpoint_catalog(settings.endpoint_catalog_path).items()
            if endpoint.method == "GET" or settings.paypal_sandbox_writes_enabled
        },
        # Server-owned catalog was generated by contract-backed onboarding.
        database=database,
        verification_keys={"system_search_result", "rag_result", "paypal_response"},
        precondition_keys={"paypal_reviewed_write", "sandbox_approved_write"}
        if settings.paypal_sandbox_writes_enabled
        else set(),
    )


async def collection_admin(arguments) -> list[dict]:
    from app.tools.onboarding import onboard_collection
    from app.tools.openapi import download_specifications, enrich
    from app.tools.paypal_collection import convert

    settings = Settings()
    local = json.loads(arguments.token_file.read_text(encoding="utf-8"))
    principal = TokenAuthenticator(settings).authenticate(local["bearer_token"])
    ToolRegistry.require_admin(principal)
    cards, inventory = convert(arguments.path)
    if arguments.fetch_specs:
        await download_specifications(arguments.specs)
    if (arguments.specs / "source.json").exists():
        cards = enrich(cards, inventory, arguments.specs)
    elif arguments.enable_supported or arguments.publish_reads:
        raise ValueError("Official contracts are required to enable collection tools")
    if arguments.enable_supported or arguments.publish_reads:
        save_endpoint_catalog(
            settings.endpoint_catalog_path,
            [
                card
                for card in cards
                if card.endpoint is not None
                and (arguments.enable_supported or card.operation_type == "read")
            ],
        )
    database = Database(settings)
    provider = None
    gateway = None
    try:
        await database.open()
        provider = GeminiEmbeddingProvider(settings)
        registry = admin_registry(database, settings)
        service = IngestionService(
            database, None, RetrievalService(provider), settings, registry=registry
        )
        model_client = None
        if (
            arguments.enable_supported or arguments.publish_reads
        ) and arguments.reviewer == "gemini":
            # This explicitly named onboarding reviewer is separate from the chat provider.
            gateway = create_llm_gateway(settings.model_copy(update={"llm_provider": "gemini"}))
            model_client = ModelClient(gateway)
        rows = await onboard_collection(
            cards,
            inventory,
            arguments.output,
            service,
            principal,
            publish_reads=arguments.publish_reads,
            enable_supported=arguments.enable_supported,
            model_client=model_client,
            distance_threshold=arguments.distance_threshold
            if arguments.distance_threshold is not None
            else settings.taxonomy_distance_threshold,
            evaluation_file=arguments.evaluation_file,
            reviewer=arguments.reviewer,
        )
        print(f"Collection onboarding report: {arguments.output / 'onboarding-report.json'}")
        return rows
    finally:
        if provider:
            await provider.aclose()
        if gateway:
            await gateway.aclose()
        await database.close()


async def tool_admin(arguments) -> None:
    settings = Settings()
    local = json.loads(arguments.token_file.read_text(encoding="utf-8"))
    principal = TokenAuthenticator(settings).authenticate(local["bearer_token"])
    ToolRegistry.require_admin(principal)
    database = Database(settings)
    provider = None
    registry = admin_registry(database, settings)
    try:
        await database.open()
        if arguments.command == "import-tool":
            provider = GeminiEmbeddingProvider(settings)
            service = IngestionService(
                database, None, RetrievalService(provider), settings, registry=registry
            )
            card = await service.import_tool(
                arguments.path, principal, draft_only=arguments.draft_only
            )
            print("Tool draft imported:", card.tool_id, card.schema_version)
        elif arguments.command == "publish-tool":
            async with database.session() as session:
                record = await session.get(
                    RegistryTool, (principal.tenant_id, arguments.tool_id, arguments.version)
                )
                if record is None:
                    raise ValueError("Tool draft not found")
                card = ToolCard.model_validate(record.card)
            await registry.publish(card, principal)
            print("Tool version published:", card.tool_id, card.schema_version)
        else:
            await registry.revoke(arguments.tool_id, arguments.version, principal)
            print("Tool version revoked:", arguments.tool_id, arguments.version)
    finally:
        if provider:
            await provider.aclose()
        await database.close()


async def document_admin(arguments) -> None:
    settings = Settings()
    database = Database(settings)
    store = provider = None
    try:
        if arguments.command == "import-document":
            local = json.loads(arguments.token_file.read_text(encoding="utf-8"))
            principal = TokenAuthenticator(settings).authenticate(local["bearer_token"])
            if "knowledge:manage" not in principal.scopes:
                raise ValueError("Document management permission is required")
        store = PayloadStore(database, settings)
        if arguments.command == "setup-storage":
            await store.setup()
            print("Object storage bucket is ready.")
            return
        await database.open()
        provider = GeminiEmbeddingProvider(settings)
        ingestion = IngestionService(database, store, RetrievalService(provider), settings)
        identity = await ingestion.import_document(
            arguments.path,
            arguments.corpus,
            principal,
            source_key=arguments.source_key,
            required_scopes=arguments.scope,
        )
        print("Document imported and activated:", identity)
    finally:
        if provider:
            await provider.aclose()
        if store:
            await store.aclose()
        await database.close()


async def query_knowledge(arguments) -> None:
    settings = Settings()
    auth = TokenAuthenticator(settings)
    local = json.loads(arguments.token_file.read_text(encoding="utf-8"))
    principal = auth.authenticate(local["bearer_token"])
    database = Database(settings)
    provider = gateway = None
    telemetry = Telemetry.from_settings(settings)
    try:
        await database.open()
        provider = GeminiEmbeddingProvider(settings)
        retrieval = RetrievalService(provider, database=database, settings=settings)
        if arguments.mode == "answer":
            gateway = create_llm_gateway(settings, telemetry=telemetry)
        tool = RagPipeline(
            retrieval,
            ContextBuilder(settings),
            auth,
            ModelClient(gateway) if gateway else None,
            telemetry=telemetry,
        )
        # Diagnostic invocation IDs; this command does not enqueue an agent workflow.
        context = ToolContext(principal.tenant_id, principal.user_id, uuid4(), uuid4(), None)
        result = await tool.invoke(
            {"query": arguments.query, "mode": arguments.mode, "corpus_ids": arguments.corpus},
            context,
        )
        if result.state != "succeeded":
            print(f"Knowledge tool failed: {result.error_code}")
            raise SystemExit(2)
        print(json.dumps(result.data, indent=2, ensure_ascii=False))
    finally:
        if gateway:
            await gateway.aclose()
        if provider:
            await provider.aclose()
        await database.close()
        await telemetry.aclose()


async def check_llm() -> None:
    gateway = create_llm_gateway(Settings())
    try:
        response = await gateway.generate("Reply with only OK.")
        print(f"LLM connection successful: {response.provider}, {response.model}")
    finally:
        await gateway.aclose()


async def check_langfuse() -> None:
    telemetry = Telemetry.from_settings(Settings())
    try:
        if not await telemetry.auth_check():
            raise ValueError("Credentials, endpoint or connectivity could not be verified")
        with telemetry.observe(
            "connection_check", input={"message": "Synthetic Langfuse check"}
        ) as observation:
            observation.update(output={"status": "ok"})
            traceparent = observation.traceparent
        await telemetry.flush()
        if telemetry.degraded or traceparent is None:
            raise ValueError("Diagnostic trace could not be queued")
        print("Langfuse authentication successful; synthetic trace submitted")
        print("Trace ID:", traceparent.split("-")[1])
    finally:
        await telemetry.aclose()


if __name__ == "__main__":
    main()
