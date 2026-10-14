"""Invoice-list test program formats only the requested fields."""

from scripts.test_paypal_invoice_list import invoice_rows


def test_invoice_rows_extracts_ids_status_and_amount_and_caps_at_five():
    items = [
        {
            "id": f"INV-{index}",
            "status": "DRAFT",
            "amount": {"value": f"{index}.00", "currency_code": "USD"},
        }
        for index in range(7)
    ]
    result = {"results": {"step_1": {"response": {"items": items}}}}

    rows = invoice_rows(result)

    assert len(rows) == 5
    assert rows[0] == {"invoice_id": "INV-0", "status": "DRAFT", "amount": "0.00 USD"}


def test_invoice_rows_handles_empty_list_and_missing_amount():
    assert invoice_rows({"results": {"list": {"response": {"items": []}}}}) == []
    assert invoice_rows(
        {"results": {"list": {"response": {"items": [{"id": "INV-1", "status": "PAID"}]}}}}
    ) == [{"invoice_id": "INV-1", "status": "PAID", "amount": "Not returned"}]
