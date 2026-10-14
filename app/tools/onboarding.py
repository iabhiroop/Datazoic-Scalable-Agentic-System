"""Resumable tool onboarding through the standard import and embedding service."""

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from sqlalchemy import select

from app.agent.gateway import LLMError
from app.db.models import RegistryTool, TaxonomyCategory
from app.knowledge.ingestion import IngestionService
from app.knowledge.taxonomy import (
    TaxonomyCluster,
    TaxonomyLabel,
    TaxonomyReview,
    TaxonomyToolDecision,
    apply_taxonomy_review,
    build_clusters,
    choose_stable_category_ids,
    evaluate_thresholds,
    validate_review,
)
from app.security import Principal
from app.tools.contracts import ToolCard


def contract(card: ToolCard) -> dict:
    return card.model_dump(
        mode="json",
        exclude={
            "status",
            "schema_version",
            "embedding_model",
            "embedding_version",
            "category_ids",
        },
    )


def choose_version(card: ToolCard, records: list, settings) -> tuple[ToolCard, bool]:
    """Reuse compatible versions; taxonomy-only changes do not require new embeddings."""
    for record in sorted(records, key=lambda row: bool(row.is_current), reverse=True):
        stored = ToolCard.model_validate(record.card)
        if (
            record.status != "revoked"
            and record.embedding is not None
            and stored.embedding_model == settings.embedding_model
            and stored.embedding_version == settings.embedding_version
            and record.embedding_version == settings.embedding_version
            and contract(stored) == contract(card)
        ):
            return stored.model_copy(update={"status": record.status}), True
    versions = [
        int(record.schema_version) for record in records if record.schema_version.isdecimal()
    ]
    return card.model_copy(update={"schema_version": str(max(versions, default=0) + 1)}), False


def write_report(path: Path, rows: list[dict], metadata: dict | None = None) -> None:
    report = {"counts": dict(Counter(row["outcome"] for row in rows)), "tools": rows}
    if metadata:
        report["taxonomy"] = metadata
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _review_input(clusters: list[TaxonomyCluster], cards: list[ToolCard]) -> dict:
    by_id = {card.tool_id: card for card in cards}
    return {
        "clusters": [
            {
                "cluster_key": cluster.cluster_key,
                "parent_cluster_key": cluster.parent_cluster_key,
                "tool_ids": list(cluster.tool_ids),
                "representative_tool_ids": list(cluster.representative_tool_ids),
            }
            for cluster in clusters
        ],
        "tool_metadata": [
            {
                "tool_id": tool_id,
                "name": by_id[tool_id].name,
                "description": by_id[tool_id].description,
                "synonyms": by_id[tool_id].synonyms,
                "entities": by_id[tool_id].entities,
                "action": by_id[tool_id].action,
            }
            for tool_id in sorted(by_id)
        ],
    }


LOCAL_REVIEW_VERSION = "local-manual-review-v7"


async def _evaluation_vectors(
    evaluation_file: Path,
    cache_path: Path,
    tool_ids: set[str],
    service: IngestionService,
    *,
    allow_embedding: bool = True,
) -> list[tuple[list[float], set[str]]]:
    cases = json.loads(evaluation_file.read_text(encoding="utf-8"))
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "model": service.settings.embedding_model,
                "version": service.settings.embedding_version,
                "dimension": service.settings.embedding_dimension,
                "cases": [
                    {
                        "query": case["query"],
                        "tool_ids": sorted(set(case.get("expected_tool_ids", [])) & tool_ids),
                    }
                    for case in cases
                ],
            },
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    cached = {}
    if cache_path.exists():
        try:
            entry = json.loads(cache_path.read_text(encoding="utf-8"))
            if entry.get("fingerprint") == fingerprint:
                cached = entry.get("vectors", {})
        except (OSError, ValueError, TypeError):
            cached = {}
    result = []
    for case in cases:
        relevant = set(case.get("expected_tool_ids", [])) & tool_ids
        if not relevant:
            continue
        key = hashlib.sha256(case["query"].encode()).hexdigest()
        vector = cached.get(key)
        if not isinstance(vector, list) or len(vector) != service.settings.embedding_dimension:
            if not allow_embedding:
                raise ValueError("Cached retrieval evaluation vector is unavailable")
            vector = await service.retrieval.query_vector(case["query"], 1)
            cached[key] = vector
            temporary = cache_path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps({"fingerprint": fingerprint, "vectors": cached}) + "\n",
                encoding="utf-8",
            )
            temporary.replace(cache_path)
        result.append((vector, relevant))
    return result


async def _cached_or_new_review(
    path: Path,
    clusters: list[TaxonomyCluster],
    cards: list[ToolCard],
    model_client,
    reviewer: str,
    route_paths: dict[str, str],
) -> tuple[TaxonomyReview, str]:
    payload = _review_input(clusters, cards)
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "reviewer": reviewer,
                "review_version": LOCAL_REVIEW_VERSION,
                "route_paths": route_paths,
                **payload,
            },
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    if path.exists():
        try:
            cache = json.loads(path.read_text(encoding="utf-8"))
            if cache.get("fingerprint") == fingerprint:
                review = TaxonomyReview.model_validate(cache["review"])
                validate_review(clusters, {card.tool_id for card in cards}, review)
                return review, f"cached_{reviewer}"
        except (OSError, ValueError, KeyError, TypeError):
            pass
    if reviewer == "manual":
        card_by_id = {card.tool_id: card for card in cards}
        labels = []
        family_names = {
            "billing": "Billing and subscriptions",
            "catalogs": "Product catalog",
            "checkout": "Orders and checkout",
            "customer": "Customer disputes",
            "notifications": "Webhooks and notifications",
            "payments": "Payments and payouts",
            "reporting": "Financial reports",
            "shipping": "Shipping and tracking",
            "invoicing": "Invoices and templates",
            "vault": "Payment tokens",
            "pricing": "Pricing",
            "identity": "Identity and accounts",
            "oauth2": "Authentication",
        }

        def route_parts(card: ToolCard) -> list[str]:
            path = route_paths.get(card.tool_id) or (card.endpoint.path if card.endpoint else "")
            parts = path.strip("/").split("/") if path else []
            parts = parts[1:] if parts and re.fullmatch(r"v\d+", parts[0]) else parts
            return [part for part in parts if not part.startswith("{")]

        def route_family(card: ToolCard) -> str:
            parts = route_parts(card)
            return parts[0] if parts else "other"

        def family_label(cluster: TaxonomyCluster) -> tuple[str, float]:
            counts: Counter[str] = Counter()
            for tool_id in cluster.tool_ids:
                counts[route_family(card_by_id[tool_id])] += 1
            if not counts:
                return "Business operations", 0.0
            family, count = counts.most_common(1)[0]
            return family_names.get(family, "Business operations"), count / len(cluster.tool_ids)

        def route_subject(card: ToolCard) -> str:
            parts = route_parts(card)
            pair = tuple(parts[:2])
            subjects = {
                ("billing", "plans"): "Plans and pricing",
                ("billing", "subscriptions"): "Subscriptions",
                ("catalogs", "products"): "Products",
                ("checkout", "orders"): "Orders",
                ("checkout", "payment-resources"): "Payment resources",
                ("customer", "disputes"): "Disputes",
                ("customer", "managed-accounts"): "Managed accounts",
                ("notifications", "webhooks"): "Webhooks",
                ("notifications", "webhooks-events"): "Webhook events",
                ("notifications", "webhooks-event-types"): "Webhook event types",
                ("payments", "payouts"): "Payouts",
                ("payments", "payouts-item"): "Payout items",
                ("payments", "authorizations"): "Payment authorizations",
                ("payments", "captures"): "Payment captures",
                ("payments", "refunds"): "Payment refunds",
                ("reporting", "balances"): "Financial reports",
                ("reporting", "transactions"): "Financial reports",
                ("shipping", "trackers"): "Shipping and tracking",
                ("shipping", "trackers-batch"): "Shipping and tracking",
                ("invoicing", "invoices"): "Invoices",
                ("invoicing", "templates"): "Invoice templates",
                ("vault", "payment-tokens"): "Payment tokens",
                ("vault", "setup-tokens"): "Payment setup tokens",
                ("pricing", "quote-exchange-rates"): "Exchange rates",
                ("identity", "generate-token"): "Identity tokens",
                ("oauth2", "token"): "OAuth tokens",
            }
            return subjects.get(
                pair,
                family_names.get(parts[0], "Business operations")
                if parts
                else "Business operations",
            )

        for cluster in clusters:
            family, family_share = family_label(cluster)
            if cluster.parent_cluster_key is None:
                root_cards = [card_by_id[tool_id] for tool_id in cluster.tool_ids]
                root_names = " ".join(card.name.casefold() for card in root_cards)
                root_families = {route_family(card) for card in root_cards}
                root_subjects = Counter(route_subject(card) for card in root_cards)
                subject, count = root_subjects.most_common(1)[0]
                if root_names.count("details") >= len(root_cards) / 2:
                    if root_families == {"payments"}:
                        label = "Payment details"
                    elif root_families <= {"checkout", "payments"}:
                        label = "Order and payment details"
                    elif root_families <= {"billing", "invoicing"}:
                        label = "Billing and invoice details"
                    else:
                        label = "Record details"
                elif count >= len(root_cards) * 0.6:
                    label = subject
                elif family_share >= 0.6:
                    label = family if family != "Customer disputes" else subject
                elif "payments" in root_families:
                    label = "Payout operations" if "payout" in root_names else "Payment operations"
                else:
                    label = "Business operations"
                description = f"Tools for {label.casefold()} tasks."
            else:
                cards_in_cluster = [card_by_id[tool_id] for tool_id in cluster.tool_ids]
                names = " ".join(card.name.casefold() for card in cards_in_cluster)
                families = {route_family(card) for card in cards_in_cluster}
                if names.count("details") >= len(cards_in_cluster) / 2:
                    if families == {"payments"}:
                        label = "Payment details"
                    elif families <= {"checkout", "payments"}:
                        label = "Order and payment details"
                    elif families <= {"billing", "invoicing"}:
                        label = "Billing and invoice details"
                    else:
                        label = "Record details"
                elif names.count("fully update") >= len(cards_in_cluster) / 2:
                    label = "Full record updates"
                elif names.count("token") >= len(cards_in_cluster) / 2:
                    label = "Token operations"
                else:
                    subjects = Counter(route_subject(card) for card in cards_in_cluster)
                    subject, count = subjects.most_common(1)[0]
                    if count >= len(cards_in_cluster) * 0.6:
                        label = subject
                    elif len(families) == 1 and "payments" in families:
                        label = "Payout operations" if "payout" in names else "Payment operations"
                    elif families <= {"billing", "invoicing"}:
                        label = "Plans and invoice operations"
                    else:
                        label = subject
                description = f"Tools for {label.casefold()} across the connected service."
            labels.append(
                TaxonomyLabel(
                    cluster_key=cluster.cluster_key,
                    label=label[:80],
                    description=description[:240],
                    parent_cluster_key=cluster.parent_cluster_key,
                )
            )
        decisions = []
        for card in cards:
            endpoint_path = route_paths.get(card.tool_id, "").rstrip("/").casefold()
            ambiguous_identity_action = endpoint_path.endswith(("/generate-token", "/terminate"))
            approved = bool(
                card.name.strip()
                and card.description.strip()
                and card.action.strip()
                and not ambiguous_identity_action
            )
            decisions.append(
                TaxonomyToolDecision(
                    tool_id=card.tool_id,
                    approved=approved,
                    reason_code="approved" if approved else "unclear_metadata",
                )
            )
        review = TaxonomyReview(labels=labels, tools=decisions)
    elif reviewer == "gemini":
        review = await model_client.review_taxonomy(
            clusters,
            payload["tool_metadata"],
            progress_path=path,
            fingerprint=fingerprint,
        )
    else:
        raise ValueError("Unknown taxonomy reviewer")
    validate_review(clusters, {card.tool_id for card in cards}, review)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"fingerprint": fingerprint, "review": review.model_dump(mode="json")}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return review, reviewer


async def _review_and_publish(
    cards: list[ToolCard],
    inventory: dict,
    rows: list[dict],
    output: Path,
    service: IngestionService,
    principal: Principal,
    model_client,
    distance_threshold: float,
    evaluation_file: Path,
    enable_supported: bool,
    reviewer: str,
) -> dict:
    staged_ids = {card.tool_id for card in cards}
    async with service.database.session() as session:
        records = list(
            await session.scalars(
                select(RegistryTool).where(
                    RegistryTool.tenant_id == principal.tenant_id,
                    RegistryTool.tool_id.in_(staged_ids),
                )
            )
        )
        category_rows = list(
            await session.scalars(
                select(TaxonomyCategory).where(TaxonomyCategory.tenant_id == principal.tenant_id)
            )
        )
    selected = {}
    records_by_key = {(record.tool_id, record.schema_version): record for record in records}
    for card in cards:
        record = records_by_key.get((card.tool_id, card.schema_version))
        if (
            record is not None
            and record.embedding is not None
            and record.embedding_version == service.settings.embedding_version
        ):
            selected[card.tool_id] = card
    if not selected:
        return {"status": "failed", "reason_code": "no_compatible_tool_embeddings"}
    vectors = {
        tool_id: list(records_by_key[(tool_id, card.schema_version)].embedding)
        for tool_id, card in selected.items()
    }

    evaluation = {"evaluated_queries": 0, "selected_threshold": distance_threshold, "scores": []}
    try:
        query_vectors = await _evaluation_vectors(
            evaluation_file,
            output / "taxonomy-evaluation-vectors.json",
            set(vectors),
            service,
            allow_embedding=reviewer != "manual",
        )
        if query_vectors:
            candidates = sorted(
                {
                    round(max(0.01, min(2.0, distance_threshold * ratio)), 4)
                    for ratio in (0.5, 0.75, 1.0, 1.25, 1.5)
                }
            )
            scores = evaluate_thresholds(vectors, query_vectors, candidates)
            chosen = max(
                scores,
                key=lambda item: (
                    item["mean_category_f1"],
                    -abs(item["distance_threshold"] - distance_threshold),
                ),
            )
            distance_threshold = chosen["distance_threshold"]
            evaluation = {
                "evaluated_queries": len(query_vectors),
                "selected_threshold": distance_threshold,
                "scores": scores,
            }
    except (OSError, ValueError, KeyError, TypeError):
        evaluation["status"] = "configured_threshold_used"

    clusters = build_clusters(
        vectors,
        distance_threshold=distance_threshold,
        subcategory_threshold=distance_threshold * 0.5,
    )
    cached_path = output / "taxonomy-review.json"
    route_paths = {entry["tool_id"]: entry["path"] for entry in inventory["endpoints"]}
    review, review_source = await _cached_or_new_review(
        cached_path, clusters, list(selected.values()), model_client, reviewer, route_paths
    )
    validate_review(clusters, set(selected), review)

    by_key = {cluster.cluster_key: cluster for cluster in clusters}
    current_assignments = {tool_id: card.category_ids for tool_id, card in selected.items()}
    existing_categories = {row.category_id: (row.depth, row.parent_id) for row in category_rows}
    category_ids = choose_stable_category_ids(clusters, current_assignments, existing_categories)
    labels = {label.cluster_key: label for label in review.labels}
    assigned_categories: dict[str, list[str]] = {}
    for cluster in clusters:
        if cluster.parent_cluster_key is None:
            for tool_id in cluster.tool_ids:
                assigned_categories.setdefault(tool_id, [category_ids[cluster.cluster_key]])
    for cluster in clusters:
        if cluster.parent_cluster_key:
            for tool_id in cluster.tool_ids:
                assigned_categories[tool_id] = [
                    category_ids[cluster.parent_cluster_key],
                    category_ids[cluster.cluster_key],
                ]
    for decision in review.tools:
        assigned_categories[decision.tool_id].extend(
            category_ids[key]
            for key in decision.secondary_cluster_keys
            if category_ids[key] not in assigned_categories[decision.tool_id]
        )

    supported = {
        entry["tool_id"]
        for entry in inventory["endpoints"]
        if entry.get("execution_supported")
        and entry["tool_id"] in selected
        and (enable_supported or selected[entry["tool_id"]].operation_type == "read")
    }
    outcomes = await apply_taxonomy_review(
        service.database,
        service.registry,
        principal,
        selected,
        vectors,
        clusters,
        review,
        supported_tool_ids=supported,
        deterministic_errors={},
    )
    review_decisions = {decision.tool_id: decision for decision in review.tools}
    row_by_id = {row["tool_id"]: row for row in rows}
    inventory_by_id = {entry["tool_id"]: entry for entry in inventory["endpoints"]}
    for tool_id, outcome in outcomes.items():
        row = row_by_id[tool_id]
        row["status"] = outcome["status"]
        row["review_reason"] = (
            inventory_by_id[tool_id].get("review_reason") or outcome["reason_code"]
            if outcome["reason_code"] == "execution_contract_unsupported"
            else outcome["reason_code"]
        )
        row["review_decision"] = "approved" if review_decisions[tool_id].approved else "rejected"
        if outcome.get("schema_version"):
            row["schema_version"] = outcome["schema_version"]
        row["assigned_category_ids"] = assigned_categories[tool_id]
        row["assigned_categories"] = [
            labels[cluster_key].label
            for cluster_key in by_key
            if tool_id in by_key[cluster_key].tool_ids
        ]
        if outcome["status"] == "published":
            row["outcome"] = (
                "published"
                if outcome["reason_code"] in {"auto_review_approved", "taxonomy_version_created"}
                else "retained"
            )
        elif row["outcome"] not in {"failed", "imported"}:
            row["outcome"] = "draft"
    return {
        "status": "completed",
        "algorithm": "average_linkage_cosine",
        "review_source": review_source,
        "distance_threshold": distance_threshold,
        "evaluation": evaluation,
        "clusters": [
            {
                "cluster_key": cluster.cluster_key,
                "parent_cluster_key": cluster.parent_cluster_key,
                "tool_ids": list(cluster.tool_ids),
                "representative_tool_ids": list(cluster.representative_tool_ids),
                "label": labels[cluster.cluster_key].label,
            }
            for cluster in clusters
        ],
    }


async def onboard_collection(
    cards: list[ToolCard],
    inventory: dict,
    output: Path,
    service: IngestionService,
    principal: Principal,
    *,
    publish_reads: bool = False,
    enable_supported: bool = False,
    model_client=None,
    distance_threshold: float = 0.42,
    evaluation_file: Path = Path("resources/taxonomy/retrieval-evaluation.json"),
    reviewer: str = "gemini",
) -> list[dict]:
    output.mkdir(parents=True, exist_ok=True)
    (output / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    async with service.database.session() as session:
        records = list(
            await session.scalars(
                select(RegistryTool).where(
                    RegistryTool.tenant_id == principal.tenant_id,
                    RegistryTool.tool_id.in_([card.tool_id for card in cards]),
                )
            )
        )
    by_tool: dict[str, list] = {}
    for record in records:
        by_tool.setdefault(record.tool_id, []).append(record)
    metadata = {entry["tool_id"]: entry for entry in inventory["endpoints"]}
    rows, staged_cards = [], []
    for position, generated in enumerate(cards, 1):
        card, reused = choose_version(
            generated, by_tool.get(generated.tool_id, []), service.settings
        )
        target = output / f"{card.tool_id}.json"
        target.write_text(card.model_dump_json(indent=2) + "\n", encoding="utf-8")
        row = {
            "tool_id": card.tool_id,
            "schema_version": card.schema_version,
            "name": card.name,
            "method": metadata[card.tool_id]["method"],
            "path": metadata[card.tool_id]["path"],
            "needs_manual_mapping": metadata[card.tool_id]["needs_manual_mapping"],
            "execution_supported": metadata[card.tool_id].get("execution_supported", False)
            and (enable_supported or not publish_reads or card.operation_type == "read"),
            "outcome": "reused" if reused else "imported",
            "status": card.status if reused else "draft",
        }
        if not row["execution_supported"]:
            row["review_reason"] = (
                metadata[card.tool_id].get("review_reason") or "execution_contract_unsupported"
            )
        try:
            if not reused:
                card = await service.import_tool(target, principal, draft_only=True)
            staged_cards.append(card)
        except Exception as error:  # noqa: BLE001 -- keep private provider and database data out of reports
            row.update(outcome="failed", status="not_imported")
            row["error_code"] = error.code if isinstance(error, LLMError) else "onboarding_failed"
        rows.append(row)
        write_report(output / "onboarding-report.json", rows)
        print(
            f"[{position}/{len(cards)}] {row['outcome']}: {card.tool_id} v{card.schema_version} ({row['status']})",
            flush=True,
        )

    taxonomy_report = None
    if enable_supported or publish_reads:
        if model_client is None and reviewer == "gemini":
            for row in rows:
                if row["execution_supported"] and row["status"] != "published":
                    row.update(outcome="failed", review_reason="gemini_review_unavailable")
            taxonomy_report = {"status": "failed", "reason_code": "gemini_review_unavailable"}
        else:
            try:
                taxonomy_report = await _review_and_publish(
                    staged_cards,
                    inventory,
                    rows,
                    output,
                    service,
                    principal,
                    model_client,
                    distance_threshold,
                    evaluation_file,
                    enable_supported,
                    reviewer,
                )
            except Exception as error:  # noqa: BLE001 -- failures leave staged tools unpublished
                reason = error.code if isinstance(error, LLMError) else "taxonomy_review_failed"
                for row in rows:
                    if row["execution_supported"] and row["status"] != "published":
                        row.update(outcome="failed", review_reason=reason)
                taxonomy_report = {
                    "status": "failed",
                    "reason_code": reason,
                    "error_type": type(error).__name__,
                }
    write_report(output / "onboarding-report.json", rows, taxonomy_report)
    return rows
