import json

from app.tools.paypal_collection import convert


def test_collection_deduplicates_variants_and_ignores_saved_responses(tmp_path):
    source = tmp_path / "collection.json"
    source.write_text(
        json.dumps(
            {
                "info": {"name": "fixture"},
                "item": [
                    {
                        "name": "PayPal Invoices",
                        "auth": {"type": "oauth2"},
                        "item": [
                            {
                                "name": "List",
                                "request": {
                                    "method": "GET",
                                    "url": {
                                        "raw": "{{base_url}}/v2/invoicing/invoices?page=1",
                                        "query": [{"key": "page", "value": "1"}],
                                    },
                                },
                                "response": [{"body": "SECRET_CUSTOMER_DATA"}],
                            },
                            {
                                "name": "List variant",
                                "request": {
                                    "method": "GET",
                                    "url": {
                                        "raw": "{{base_url}}/v2/invoicing/invoices?page=2",
                                        "query": [{"key": "page", "value": "2"}],
                                    },
                                },
                                "event": [{"listen": "test", "script": {"exec": ["unsafe code"]}}],
                            },
                            {
                                "name": "Get",
                                "request": {
                                    "method": "GET",
                                    "url": {
                                        "raw": "https://api-m.sandbox.paypal.com/v2/invoicing/invoices/{{invoice_id}}"
                                    },
                                },
                            },
                        ],
                    }
                ],
                "auth": {"type": "bearer", "bearer": [{"key": "token", "type": "string"}]},
                "variable": [{"key": "client_secret", "value": "SECRET_VALUE"}],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    cards, inventory = convert(source)
    assert len(cards) == 2
    assert inventory["counts"]["duplicate_variants"] == 1
    assert inventory["counts"]["scripted_requests"] == 1
    assert inventory["auth_types"]["oauth2"] == 3
    assert inventory["collection_auth_type"] == "bearer"
    assert inventory["collection_variable_names"] == ["client_secret"]
    assert "SECRET" not in json.dumps([c.model_dump() for c in cards] + [inventory])
    assert all(card.status == "draft" for card in cards)
    assert cards[1].input_schema["required"] == ["invoice_id"]
    assert all("paypal" not in card.name.lower() for card in cards)
    assert all("paypal" not in card.description.lower() for card in cards)
    assert all("paypal" not in " ".join(card.category_ids).lower() for card in cards)
    assert cards[0].category_ids == ["invoicing", "invoices"]
    assert [card.name for card in cards] == ["List invoices", "Retrieve invoice"]


def test_collection_rejects_untrusted_host_and_auth_endpoint(tmp_path):
    source = tmp_path / "collection.json"
    source.write_text(
        json.dumps(
            {
                "item": [
                    {
                        "request": {
                            "method": "GET",
                            "url": "https://example.org/v2/invoicing/invoices",
                        }
                    },
                    {"request": {"method": "POST", "url": "{{base_url}}/v1/oauth2/token"}},
                ]
            }
        ),
        encoding="utf-8",
    )
    cards, inventory = convert(source)
    assert cards == []
    assert inventory["counts"]["untrusted_host"] == 1
    assert inventory["counts"]["internal_auth"] == 1
