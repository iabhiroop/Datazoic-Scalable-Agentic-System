import json

import pytest

from app.agent.gateway import LLMGateway, LLMResponse
from app.agent.llm import ModelClient
from app.knowledge.taxonomy import (
    TaxonomyCluster,
    TaxonomyLabel,
    TaxonomyReview,
    TaxonomyToolDecision,
    average_linkage,
    build_clusters,
    choose_stable_category_ids,
    evaluate_thresholds,
    validate_review,
)


def test_average_linkage_uses_cosine_distance_and_deterministic_membership():
    vectors = {
        "a1": [1.0, 0.0],
        "a2": [0.999, 0.01],
        "b1": [0.0, 1.0],
        "b2": [-0.01, 0.999],
    }
    first = average_linkage(vectors, 0.001)
    second = average_linkage(vectors, 0.001)
    assert first == second == [("a1", "a2"), ("b1", "b2")]


def test_subcategories_require_two_separated_capability_groups():
    vectors = {
        "a1": [1.0, 0.0],
        "a2": [0.999, 0.01],
        "b1": [0.0, 1.0],
        "b2": [0.01, 0.999],
    }
    clusters = build_clusters(vectors, distance_threshold=1.2, subcategory_threshold=0.001)
    roots = [cluster for cluster in clusters if cluster.parent_cluster_key is None]
    children = [cluster for cluster in clusters if cluster.parent_cluster_key is not None]
    assert len(roots) == 1
    assert {cluster.tool_ids for cluster in children} == {("a1", "a2"), ("b1", "b2")}
    assert all(cluster.parent_cluster_key == roots[0].cluster_key for cluster in children)


def test_taxonomy_review_must_preserve_membership_and_hierarchy():
    root = TaxonomyCluster("root", ("a", "b", "c", "d"), ("a", "c"))
    left = TaxonomyCluster("left", ("a", "b"), ("a",), "root")
    right = TaxonomyCluster("right", ("c", "d"), ("c",), "root")
    review = TaxonomyReview(
        labels=[
            TaxonomyLabel(cluster_key="root", label="Payments", description="Payment operations"),
            TaxonomyLabel(
                cluster_key="left",
                label="Capture",
                description="Capture operations",
                parent_cluster_key="root",
            ),
            TaxonomyLabel(
                cluster_key="right",
                label="Refunds",
                description="Refund operations",
                parent_cluster_key="root",
            ),
        ],
        tools=[
            TaxonomyToolDecision(tool_id=tool_id, approved=True, reason_code="approved")
            for tool_id in "abcd"
        ],
    )
    validate_review([root, left, right], set("abcd"), review)
    bad = review.model_copy(update={"tools": review.tools[:-1]})
    with pytest.raises(ValueError, match="tool set"):
        validate_review([root, left, right], set("abcd"), bad)
    moved = review.model_copy(
        update={
            "labels": [
                review.labels[0],
                review.labels[1].model_copy(update={"parent_cluster_key": None}),
                review.labels[2],
            ]
        }
    )
    with pytest.raises(ValueError, match="hierarchy"):
        validate_review([root, left, right], set("abcd"), moved)


def test_stable_category_ids_reuse_prior_membership():
    root = TaxonomyCluster("root", ("a", "b"), ("a",))
    child = TaxonomyCluster("child", ("a", "b"), ("a",), "root")
    assigned = choose_stable_category_ids(
        [root, child],
        {"a": ["operations", "payments"], "b": ["operations", "payments"]},
        {"operations": (1, None), "payments": (2, "operations")},
    )
    assert assigned == {"root": "operations", "child": "payments"}


def test_threshold_evaluation_prefers_relevant_category_cohesion():
    vectors = {"a1": [1.0, 0.0], "a2": [0.999, 0.01], "b": [0.0, 1.0]}
    scores = evaluate_thresholds(
        vectors,
        [((1.0, 0.05), {"a1", "a2"})],
        [0.001, 2.0],
    )
    assert scores[0]["mean_category_f1"] > scores[1]["mean_category_f1"]


@pytest.mark.asyncio
async def test_gemini_review_prompt_contains_only_allowed_tool_metadata():
    class FakeProvider:
        model = "fake"

        def __init__(self):
            self.requests = []

        async def generate(self, request):
            self.requests.append(request)
            prompt = json.loads(request.prompt)
            if "clusters" in prompt:
                result = {
                    "labels": [
                        {
                            "cluster_key": cluster["cluster_key"],
                            "label": "Invoices",
                            "description": "Manage invoice records",
                            "parent_cluster_key": cluster["parent_cluster_key"],
                        }
                        for cluster in prompt["clusters"]
                    ]
                }
            else:
                result = {
                    "tools": [
                        {
                            "tool_id": tool["tool_id"],
                            "approved": True,
                            "reason_code": "approved",
                            "secondary_cluster_keys": [],
                        }
                        for tool in prompt["tools"]
                    ]
                }
            return LLMResponse(
                text=json.dumps(result),
                provider="gemini",
                model="fake",
            )

        async def aclose(self):
            return None

    provider = FakeProvider()
    cluster = TaxonomyCluster("root", ("tool-a",), ("tool-a",))
    model = ModelClient(LLMGateway(provider))
    review = await model.review_taxonomy(
        [cluster],
        [
            {
                "tool_id": "tool-a",
                "name": "List invoices",
                "description": "Find invoice records",
                "synonyms": ["search invoices"],
                "entities": ["invoice"],
                "action": "list",
                "input_schema": {"secret": "must-not-send"},
                "endpoint": {"path": "/private/path"},
            }
        ],
    )
    assert review.tools[0].approved
    assert len(provider.requests) == 2
    review_prompts = [json.loads(request.prompt) for request in provider.requests]
    sent_metadata = [
        item
        for prompt in review_prompts
        for key in ("representative_metadata", "tools")
        for item in prompt.get(key, [])
    ]
    assert all("input_schema" not in item and "endpoint" not in item for item in sent_metadata)
    assert all("must-not-send" not in request.prompt for request in provider.requests)
    assert all("/private/path" not in request.prompt for request in provider.requests)
