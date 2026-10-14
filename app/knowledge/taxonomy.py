"""Local embedding clustering and guarded publication of reviewed tool categories."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert

from app.db.models import RegistryTool, TaxonomyCategory
from app.security import Principal


class TaxonomyContract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TaxonomyLabel(TaxonomyContract):
    cluster_key: str = Field(min_length=1, max_length=80)
    label: str = Field(min_length=2, max_length=80)
    description: str = Field(min_length=8, max_length=240)
    parent_cluster_key: str | None = None


class TaxonomyToolDecision(TaxonomyContract):
    tool_id: str = Field(min_length=1, max_length=120)
    approved: bool
    reason_code: Literal[
        "approved", "unclear_metadata", "misgrouped", "duplicate_capability", "other"
    ]
    secondary_cluster_keys: list[str] = Field(default_factory=list)


class TaxonomyReview(TaxonomyContract):
    labels: list[TaxonomyLabel]
    tools: list[TaxonomyToolDecision]


class TaxonomyLabelsResponse(TaxonomyContract):
    labels: list[TaxonomyLabel]


class TaxonomyDecisionsResponse(TaxonomyContract):
    tools: list[TaxonomyToolDecision]


@dataclass(frozen=True)
class ReviewedCategory:
    category_id: str
    label: str
    description: str
    parent_id: str | None
    depth: int


@dataclass(frozen=True)
class TaxonomyCluster:
    cluster_key: str
    tool_ids: tuple[str, ...]
    representative_tool_ids: tuple[str, ...]
    parent_cluster_key: str | None = None


def normalize(vector: list[float]) -> tuple[float, ...]:
    if not vector or not all(math.isfinite(value) for value in vector):
        raise ValueError("Taxonomy vectors must be finite and nonempty")
    norm = math.sqrt(sum(value * value for value in vector))
    if not norm:
        raise ValueError("Taxonomy vectors must be nonzero")
    return tuple(value / norm for value in vector)


def cosine_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right):
        raise ValueError("Taxonomy vectors must have matching dimensions")
    return max(0.0, min(2.0, 1.0 - sum(a * b for a, b in zip(left, right, strict=True))))


def average_linkage(
    vectors: dict[str, list[float] | tuple[float, ...]], threshold: float
) -> list[tuple[str, ...]]:
    """Deterministic agglomerative clustering using mean pairwise cosine distance."""
    if not 0 <= threshold <= 2:
        raise ValueError("Taxonomy distance threshold must be between 0 and 2")
    if not vectors:
        return []
    normalized = {key: normalize(list(vector)) for key, vector in vectors.items()}
    dimensions = {len(vector) for vector in normalized.values()}
    if len(dimensions) != 1:
        raise ValueError("Taxonomy vectors must have matching dimensions")
    clusters = [(key,) for key in sorted(normalized)]
    distances = {
        (left, right): cosine_distance(normalized[left[0]], normalized[right[0]])
        for index, left in enumerate(clusters)
        for right in clusters[index + 1 :]
    }
    while len(clusters) > 1:
        candidates = [
            (distances[(left, right)], left, right)
            for index, left in enumerate(clusters)
            for right in clusters[index + 1 :]
        ]
        distance, left, right = min(candidates, key=lambda row: (row[0], row[1], row[2]))
        if distance > threshold:
            break
        merged = tuple(sorted(left + right))
        remaining = [cluster for cluster in clusters if cluster not in {left, right}]
        for other in remaining:
            left_distance = distances[tuple(sorted((left, other)))]
            right_distance = distances[tuple(sorted((right, other)))]
            distances[tuple(sorted((merged, other)))] = (
                len(left) * left_distance + len(right) * right_distance
            ) / (len(left) + len(right))
        clusters = sorted(remaining + [merged])
    return sorted(clusters)


def _centroid_distance(
    tool_ids: tuple[str, ...], vectors: dict[str, list[float] | tuple[float, ...]]
) -> dict[str, float]:
    normalized = {key: normalize(list(vectors[key])) for key in tool_ids}
    # A medoid remains defined even when a group's normalized vectors cancel out.
    medoid = min(
        tool_ids,
        key=lambda candidate: (
            sum(
                cosine_distance(normalized[candidate], other)
                for key, other in normalized.items()
                if key != candidate
            )
            / max(1, len(normalized) - 1),
            candidate,
        ),
    )
    return {key: cosine_distance(vector, normalized[medoid]) for key, vector in normalized.items()}


def _stable_key(prefix: str, anchor: str) -> str:
    return f"{prefix}-{hashlib.sha256(anchor.encode()).hexdigest()[:12]}"


def build_clusters(
    vectors: dict[str, list[float] | tuple[float, ...]],
    *,
    distance_threshold: float,
    subcategory_threshold: float,
) -> list[TaxonomyCluster]:
    """Build broad clusters and only add children when vectors form distinct groups."""
    roots = average_linkage(vectors, distance_threshold)
    proposals: list[TaxonomyCluster] = []
    used_keys: set[str] = set()
    for root in roots:
        representatives = tuple(
            key
            for key, _ in sorted(
                _centroid_distance(root, vectors).items(), key=lambda item: (item[1], item[0])
            )[:3]
        )
        root_key = _stable_key("cap", representatives[0])
        if root_key in used_keys:
            root_key = _stable_key("cap", ",".join(root))
        used_keys.add(root_key)
        proposals.append(TaxonomyCluster(root_key, root, representatives))
        if len(root) < 3 or subcategory_threshold >= distance_threshold:
            continue
        children = average_linkage({key: vectors[key] for key in root}, subcategory_threshold)
        substantial = [child for child in children if len(child) >= 2]
        if len(substantial) < 2:
            continue
        for child in substantial:
            child_representatives = tuple(
                key
                for key, _ in sorted(
                    _centroid_distance(child, vectors).items(), key=lambda item: (item[1], item[0])
                )[:3]
            )
            child_key = _stable_key("sub", f"{root_key}:{child_representatives[0]}")
            if child_key in used_keys:
                child_key = _stable_key("sub", f"{root_key}:{','.join(child)}")
            used_keys.add(child_key)
            proposals.append(TaxonomyCluster(child_key, child, child_representatives, root_key))
    return proposals


def validate_review(
    clusters: list[TaxonomyCluster], tool_ids: set[str], review: TaxonomyReview
) -> None:
    """Require exact IDs and membership; model output cannot mutate assignments."""
    expected_cluster_keys = {cluster.cluster_key for cluster in clusters}
    labels = {label.cluster_key: label for label in review.labels}
    if set(labels) != expected_cluster_keys or len(review.labels) != len(labels):
        raise ValueError("Gemini taxonomy review returned an invalid cluster set")
    decisions = {decision.tool_id: decision for decision in review.tools}
    if set(decisions) != tool_ids or len(review.tools) != len(decisions):
        raise ValueError("Gemini taxonomy review returned an invalid tool set")
    parents = {cluster.cluster_key: cluster.parent_cluster_key for cluster in clusters}
    for key, label in labels.items():
        if label.parent_cluster_key != parents[key]:
            raise ValueError("Gemini taxonomy review changed the proposed hierarchy")
        if "paypal" in (label.label + " " + label.description).casefold():
            raise ValueError("Gemini taxonomy review returned a provider-specific label")
    roots = [cluster for cluster in clusters if cluster.parent_cluster_key is None]
    seen: set[str] = set()
    by_key = {cluster.cluster_key: cluster for cluster in clusters}
    for cluster in roots:
        if not set(cluster.tool_ids).issubset(tool_ids) or seen.intersection(cluster.tool_ids):
            raise ValueError("Taxonomy roots must partition the selected tools")
        seen.update(cluster.tool_ids)
    if seen != tool_ids:
        raise ValueError("Taxonomy clusters must include every selected tool")
    for parent_key in {
        cluster.parent_cluster_key for cluster in clusters if cluster.parent_cluster_key
    }:
        parent = by_key.get(parent_key)
        children = [cluster for cluster in clusters if cluster.parent_cluster_key == parent_key]
        child_members: set[str] = set()
        for child in children:
            if parent is None or not set(child.tool_ids).issubset(parent.tool_ids):
                raise ValueError("Taxonomy child must be a subset of its parent")
            if child_members.intersection(child.tool_ids):
                raise ValueError("Taxonomy child clusters must not overlap")
            child_members.update(child.tool_ids)
    roots_by_tool = {
        tool_id: cluster.cluster_key for cluster in roots for tool_id in cluster.tool_ids
    }
    root_keys = {cluster.cluster_key for cluster in roots}
    for tool_id, decision in decisions.items():
        if decision.approved != (decision.reason_code == "approved"):
            raise ValueError("Gemini taxonomy approval and reason code disagree")
        if len(decision.secondary_cluster_keys) != len(set(decision.secondary_cluster_keys)):
            raise ValueError("Gemini taxonomy review returned duplicate secondary categories")
        if any(
            key not in root_keys or key == roots_by_tool[tool_id]
            for key in decision.secondary_cluster_keys
        ):
            raise ValueError("Gemini taxonomy review returned an invalid secondary category")


def choose_stable_category_ids(
    clusters: list[TaxonomyCluster],
    current_assignments: dict[str, list[str]],
    existing_categories: dict[str, tuple[int, str | None]],
) -> dict[str, str]:
    """Reuse existing category IDs by greatest tool overlap; hash anchors for new IDs."""
    candidates = []
    for cluster in clusters:
        if cluster.parent_cluster_key:
            continue
        members = set(cluster.tool_ids)
        counts: dict[str, int] = {}
        for tool_id in members:
            for category_id in current_assignments.get(tool_id, []):
                if existing_categories.get(category_id, (0, None))[0] != 1:
                    continue
                counts[category_id] = counts.get(category_id, 0) + 1
        for category_id, overlap in counts.items():
            candidates.append((-overlap, category_id, cluster.cluster_key))
    assigned: dict[str, str] = {}
    used_ids: set[str] = set()
    for _, category_id, cluster_key in sorted(candidates):
        if cluster_key not in assigned and category_id not in used_ids:
            assigned[cluster_key] = category_id
            used_ids.add(category_id)
    for cluster in clusters:
        if cluster.parent_cluster_key:
            continue
        if cluster.cluster_key not in assigned:
            assigned[cluster.cluster_key] = _stable_key("cap", cluster.representative_tool_ids[0])
    for cluster in clusters:
        if cluster.parent_cluster_key:
            parent_id = assigned.get(cluster.parent_cluster_key)
            if parent_id is None:
                raise ValueError("Taxonomy child references an unknown parent")
            current_counts: dict[str, int] = {}
            for tool_id in cluster.tool_ids:
                for category_id in current_assignments.get(tool_id, []):
                    depth, parent = existing_categories.get(category_id, (0, None))
                    if depth == 2 and parent == parent_id:
                        current_counts[category_id] = current_counts.get(category_id, 0) + 1
            child = next(
                (
                    category_id
                    for _, category_id in sorted(
                        (-count, category_id) for category_id, count in current_counts.items()
                    )
                    if category_id not in used_ids
                ),
                None,
            )
            assigned[cluster.cluster_key] = child or _stable_key(
                "sub", f"{parent_id}:{cluster.representative_tool_ids[0]}"
            )
            used_ids.add(assigned[cluster.cluster_key])
    return assigned


async def apply_taxonomy_review(
    database,
    registry,
    principal: Principal,
    cards: dict[str, object],
    vectors: dict[str, list[float]],
    clusters: list[TaxonomyCluster],
    review: TaxonomyReview,
    *,
    supported_tool_ids: set[str],
    deterministic_errors: dict[str, str],
) -> dict[str, dict[str, str]]:
    """Persist labels and publish approved, supported cards in one transaction."""
    from app.tools.contracts import ToolCard

    registry.require_admin(principal)
    validate_review(clusters, set(cards), review)
    label_by_key = {label.cluster_key: label for label in review.labels}
    decision_by_id = {decision.tool_id: decision for decision in review.tools}
    async with database.session() as session:
        category_rows = list(
            await session.scalars(
                select(TaxonomyCategory).where(TaxonomyCategory.tenant_id == principal.tenant_id)
            )
        )
    old_categories = {row.category_id: (row.depth, row.parent_id) for row in category_rows}
    old_assignments = {tool_id: card.category_ids for tool_id, card in cards.items()}
    id_by_key = choose_stable_category_ids(clusters, old_assignments, old_categories)
    category_by_id: dict[str, ReviewedCategory] = {}
    child_membership: dict[str, str] = {}
    for cluster in clusters:
        label = label_by_key[cluster.cluster_key]
        parent_id = id_by_key[cluster.parent_cluster_key] if cluster.parent_cluster_key else None
        category_id = id_by_key[cluster.cluster_key]
        category_by_id[category_id] = ReviewedCategory(
            category_id, label.label, label.description, parent_id, 2 if parent_id else 1
        )
        if parent_id:
            for tool_id in cluster.tool_ids:
                child_membership[tool_id] = category_id
    assignments: dict[str, list[str]] = {}
    for cluster in clusters:
        if cluster.parent_cluster_key is None:
            for tool_id in cluster.tool_ids:
                assignments.setdefault(tool_id, [id_by_key[cluster.cluster_key]])
    for tool_id, child_id in child_membership.items():
        parent_id = category_by_id[child_id].parent_id
        assignments[tool_id] = [parent_id, child_id]

    reviewed_cards: dict[str, ToolCard] = {}
    for tool_id, candidate in cards.items():
        card = candidate.model_copy(update={"category_ids": assignments[tool_id]})
        decision = decision_by_id[tool_id]
        if (
            tool_id in supported_tool_ids
            and decision.approved
            and tool_id not in deterministic_errors
        ):
            try:
                await registry.validate_import(card)
            except (ValueError, TypeError):
                deterministic_errors[tool_id] = "execution_policy_validation_failed"
        reviewed_cards[tool_id] = card

    outcomes = {}
    async with database.session() as session, session.begin():
        revisions = await registry.lock_revision(session, principal)
        rows = list(
            await session.scalars(
                select(RegistryTool)
                .where(
                    RegistryTool.tenant_id == principal.tenant_id,
                    RegistryTool.tool_id.in_(set(cards)),
                )
                .with_for_update()
            )
        )
        rows_by_key = {(row.tool_id, row.schema_version): row for row in rows}
        rows_by_tool: dict[str, list] = {}
        for row in rows:
            rows_by_tool.setdefault(row.tool_id, []).append(row)
        ordered_categories = sorted(
            category_by_id.values(), key=lambda item: (item.depth, item.category_id)
        )
        taxonomy_changed = False
        for category in ordered_categories:
            previous = next(
                (row for row in category_rows if row.category_id == category.category_id), None
            )
            changed = previous is None or (
                previous.label,
                previous.description,
                previous.parent_id,
                previous.depth,
            ) != (
                category.label,
                category.description,
                category.parent_id,
                category.depth,
            )
            taxonomy_changed |= changed
            values = {
                "tenant_id": principal.tenant_id,
                "category_id": category.category_id,
                "label": category.label,
                "description": category.description,
                "revision": 1,
                "parent_id": category.parent_id,
                "depth": category.depth,
                "parent_depth": 1,
            }
            statement = insert(TaxonomyCategory).values(**values)
            if changed:
                statement = statement.on_conflict_do_update(
                    index_elements=[TaxonomyCategory.tenant_id, TaxonomyCategory.category_id],
                    set_={
                        "label": statement.excluded.label,
                        "description": statement.excluded.description,
                        "revision": TaxonomyCategory.revision + 1,
                        "parent_id": statement.excluded.parent_id,
                        "depth": statement.excluded.depth,
                        "parent_depth": 1,
                    },
                )
            else:
                statement = statement.on_conflict_do_nothing(
                    index_elements=[TaxonomyCategory.tenant_id, TaxonomyCategory.category_id]
                )
            await session.execute(statement)
        if taxonomy_changed:
            revisions.taxonomy_revision += 1

        for tool_id, card in reviewed_cards.items():
            selected = rows_by_key.get((tool_id, card.schema_version))
            decision = decision_by_id[tool_id]
            reason = deterministic_errors.get(tool_id)
            if not decision.approved and reason is None:
                reason = decision.reason_code
            if tool_id not in supported_tool_ids and reason is None:
                reason = "execution_contract_unsupported"
            if selected is None:
                outcomes[tool_id] = {"status": "draft", "reason_code": "staged_version_missing"}
                continue
            publish = reason is None and tool_id in supported_tool_ids
            if selected.status == "revoked":
                outcomes[tool_id] = {"status": "revoked", "reason_code": "version_revoked"}
                continue
            if selected.status == "published":
                if selected.is_current and publish:
                    stored = ToolCard.model_validate(selected.card)
                    if stored.category_ids == card.category_ids:
                        outcomes[tool_id] = {
                            "status": "published",
                            "reason_code": "already_current",
                        }
                        continue
                    versions = [
                        int(row.schema_version)
                        for row in rows_by_tool.get(tool_id, [])
                        if row.schema_version.isdecimal()
                    ]
                    versioned = card.model_copy(
                        update={
                            "schema_version": str(max(versions, default=0) + 1),
                            "status": "published",
                        }
                    )
                    if (
                        selected.embedding is None
                        or selected.embedding_version != database.settings.embedding_version
                    ):
                        outcomes[tool_id] = {
                            "status": "published",
                            "reason_code": "embedding_incompatible",
                        }
                        continue
                    await session.execute(
                        update(RegistryTool)
                        .where(
                            RegistryTool.tenant_id == principal.tenant_id,
                            RegistryTool.tool_id == tool_id,
                            RegistryTool.is_current.is_(True),
                        )
                        .values(is_current=False)
                    )
                    await session.execute(
                        insert(RegistryTool).values(
                            tenant_id=principal.tenant_id,
                            tool_id=tool_id,
                            schema_version=versioned.schema_version,
                            card=versioned.model_dump(mode="json"),
                            searchable_text=selected.searchable_text,
                            embedding=selected.embedding,
                            embedding_version=selected.embedding_version,
                            is_current=True,
                            status="published",
                        )
                    )
                    revisions.registry_revision += 1
                    outcomes[tool_id] = {
                        "status": "published",
                        "reason_code": "taxonomy_version_created",
                        "schema_version": versioned.schema_version,
                    }
                else:
                    outcomes[tool_id] = {
                        "status": "published" if selected.is_current else "draft",
                        "reason_code": reason or "existing_version_not_current",
                    }
                continue
            if selected.status != "draft":
                outcomes[tool_id] = {
                    "status": selected.status,
                    "reason_code": "version_not_publishable",
                }
                continue
            if (
                selected.embedding is None
                or selected.embedding_version != database.settings.embedding_version
            ):
                publish = False
                reason = "embedding_incompatible"
            draft_card = card.model_copy(update={"status": "draft"})
            selected.card = draft_card.model_dump(mode="json")
            if publish:
                await session.execute(
                    update(RegistryTool)
                    .where(
                        RegistryTool.tenant_id == principal.tenant_id,
                        RegistryTool.tool_id == tool_id,
                    )
                    .values(is_current=False)
                )
                selected.status = "published"
                selected.is_current = True
                selected.card = draft_card.model_copy(update={"status": "published"}).model_dump(
                    mode="json"
                )
                revisions.registry_revision += 1
                outcomes[tool_id] = {"status": "published", "reason_code": "auto_review_approved"}
            else:
                outcomes[tool_id] = {
                    "status": "draft",
                    "reason_code": reason or "gemini_review_rejected",
                }
        await session.flush()
    return outcomes


def evaluate_thresholds(
    vectors: dict[str, list[float] | tuple[float, ...]],
    queries: list[tuple[tuple[float, ...], set[str]]],
    thresholds: list[float],
) -> list[dict[str, float]]:
    """Measure how much of each query's relevant set remains in its nearest category."""
    results = []
    for threshold in thresholds:
        clusters = average_linkage(vectors, threshold)
        score = 0.0
        for query_vector, relevant in queries:
            if not relevant:
                continue
            query = normalize(list(query_vector))
            chosen = min(
                clusters,
                key=lambda cluster: (
                    min(cosine_distance(query, normalize(list(vectors[item]))) for item in cluster),
                    cluster,
                ),
            )
            overlap = len(set(chosen).intersection(relevant))
            precision = overlap / len(chosen)
            recall = overlap / len(relevant)
            score += 2 * precision * recall / (precision + recall) if overlap else 0.0
        results.append(
            {"distance_threshold": threshold, "mean_category_f1": score / max(1, len(queries))}
        )
    return results
