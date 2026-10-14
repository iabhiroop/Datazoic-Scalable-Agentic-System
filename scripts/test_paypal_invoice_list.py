"""Submit a read-only PayPal invoice-list request through the local agent API."""

import argparse
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ui import AgentAPI, UIError, local_api_url, local_token

PROMPT = "List up to five invoices in my PayPal sandbox account. Show the invoice ID, status, and amount."
FINAL_STATUSES = {"completed", "failed", "needs_review", "cancelled"}


def _invoice_items(value: Any):
    """Find invoice-list items inside the tool results without depending on plan step IDs."""
    if isinstance(value, dict):
        items = value.get("items")
        if isinstance(items, list) and any(
            isinstance(item, dict) and (item.get("id") or item.get("invoice_id")) for item in items
        ):
            yield from items
        for child in value.values():
            yield from _invoice_items(child)
    elif isinstance(value, list):
        for child in value:
            yield from _invoice_items(child)


def _text(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("value") or value.get("status") or "Not returned")
    return str(value) if value not in (None, "") else "Not returned"


def _amount(invoice: dict[str, Any]) -> str:
    candidates = [
        invoice.get("amount"),
        invoice.get("amount_due"),
        invoice.get("total_amount"),
        (invoice.get("detail") or {}).get("amount")
        if isinstance(invoice.get("detail"), dict)
        else None,
    ]
    for amount in candidates:
        if isinstance(amount, dict) and amount.get("value") is not None:
            currency = amount.get("currency_code")
            return f"{amount['value']} {currency}" if currency else str(amount["value"])
        if isinstance(amount, (str, int, float)):
            return str(amount)
    return "Not returned"


def invoice_rows(result: dict[str, Any]) -> list[dict[str, str]]:
    rows = []
    seen = set()
    for invoice in _invoice_items(result.get("results", {})):
        if not isinstance(invoice, dict):
            continue
        invoice_id = invoice.get("id") or invoice.get("invoice_id")
        if not isinstance(invoice_id, str) or invoice_id in seen:
            continue
        seen.add(invoice_id)
        rows.append(
            {
                "invoice_id": invoice_id,
                "status": _text(invoice.get("status")),
                "amount": _amount(invoice),
            }
        )
        if len(rows) == 5:
            break
    return rows


def run(timeout_seconds: int, poll_seconds: float) -> int:
    api = AgentAPI(local_api_url(), local_token())
    try:
        accepted = api.submit(PROMPT, None)
        run_id = accepted["run_id"]
        print(f"Submitted read-only invoice query (run {run_id}).")
        print(f"LLM trace: {ROOT / 'runtime' / 'llm-traces' / f'{run_id}.jsonl'}")
        deadline = time.monotonic() + timeout_seconds
        while True:
            status = api.request("GET", f"/runs/{run_id}")
            state = status.get("status") or "unknown"
            if state in FINAL_STATUSES:
                break
            if state.startswith("waiting_"):
                print(f"Run needs attention ({state}): {status.get('wait_reason') or ''}")
                print(f"Open the chat UI to continue run {run_id}.")
                return 2
            if time.monotonic() >= deadline:
                print(f"Timed out waiting; run {run_id} is still {state}.")
                return 2
            time.sleep(poll_seconds)

        result = status.get("result") or {}
        if state != "completed":
            print(f"Agent run ended with status: {state}")
            if result.get("summary"):
                print(result["summary"])
            return 1

        if result.get("summary"):
            print(result["summary"])
        rows = invoice_rows(result)
        if not rows:
            results = result.get("results") or {}
            successful_read = isinstance(results, dict) and any(
                isinstance(value, dict)
                and value.get("http_status") == 200
                and isinstance(value.get("response"), dict)
                for value in results.values()
            )
            if successful_read:
                print("The invoice-list tool returned HTTP 200 with no invoice rows.")
                print("Inspect the run in the chat UI for the agent response and LLM activity.")
                return 0
            print("The run completed, but no successful invoice-list result was present.")
            return 1

        print("\nInvoice ID | Status | Amount")
        print("-----------|--------|-------")
        for invoice in rows:
            print(f"{invoice['invoice_id']} | {invoice['status']} | {invoice['amount']}")
        return 0
    except UIError as error:
        print(str(error), file=sys.stderr)
        return 1
    finally:
        api.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=240, help="Maximum wait in seconds")
    parser.add_argument("--poll", type=float, default=2, help="Status polling interval in seconds")
    args = parser.parse_args()
    if args.timeout < 1 or args.poll <= 0:
        parser.error("--timeout must be positive and --poll must be greater than zero")
    return run(args.timeout, args.poll)


if __name__ == "__main__":
    raise SystemExit(main())
