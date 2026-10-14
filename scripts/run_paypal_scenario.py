"""Run a manual PayPal business scenario through the local agent API.

The agent pauses for approval of each requested write after server-side authorization and
precondition checks; this script never calls provider APIs directly. Each run prints its redacted
LLM exchange trace path under runtime/llm-traces/.
"""

import argparse
import calendar
import re
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ui import AgentAPI, UIError, local_api_url, local_token

FINAL_STATUSES = {"completed", "failed", "needs_review", "cancelled"}
EMAIL = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


@dataclass(frozen=True)
class Scenario:
    title: str
    prompt: str
    checks: tuple[str, ...]


def previous_month_window(today: date) -> tuple[date, date]:
    """Return the previous full calendar month, including year-boundary cases."""
    if today.month == 1:
        year, month = today.year - 1, 12
    else:
        year, month = today.year, today.month - 1
    return date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1])


def build_scenarios(today: date, recipient_email: str) -> dict[str, Scenario]:
    if not EMAIL.fullmatch(recipient_email):
        raise ValueError("Recipient must be a valid email address")
    start, end = previous_month_window(today)
    bounds = f"{start.isoformat()} through {end.isoformat()}"
    return {
        "invoice": Scenario(
            title="Create and send a synthetic $50 invoice",
            prompt=(
                f"Send an invoice for $50 to {recipient_email} for one synthetic test item. "
                "Use USD and complete the request if the sandbox write policy permits it."
            ),
            checks=(
                "Verify the synthetic recipient, amount, currency, and item in the created invoice.",
                "Verify the final invoice ID and sent status.",
                "If the send outcome is uncertain, confirm the run stops for review.",
            ),
        ),
        "sales": Scenario(
            title=f"Sales volume for {start.strftime('%B %Y')}",
            prompt=(
                "What was my total sales volume last month? Show the date range, currency, "
                "and calculation. For this test, last month is " + bounds + ". "
                "Report gross captured sales before refunds and net sales after refunds "
                "as separate totals. Keep each currency separate and say when there is no data."
            ),
            checks=(
                f"Verify the report covers exactly {bounds}.",
                "Verify gross totals include successful captures and exclude failed/pending records.",
                "Verify net totals subtract refunds without mixing currencies.",
                "If no transactions exist in the period, expect an explicit no-data result.",
            ),
        ),
        "dispute": Scenario(
            title="Find an open dispute for synthetic user_123",
            prompt=(
                "Is there a dispute open from user_123? Show the dispute ID, status, amount, "
                "and transaction date. Treat user_123 as a synthetic application alias; "
                "verify it against the configured sandbox buyer/transaction mapping."
            ),
            checks=(
                "Only read/list dispute tools should be selected.",
                "Verify the dispute against the synthetic buyer's transaction mapping.",
                "Do not treat user_123 as a PayPal account ID or claim a match without evidence.",
            ),
        ),
    }


def run(scenario_id: str, recipient_email: str, timeout_seconds: int, poll_seconds: float) -> int:
    scenarios = build_scenarios(date.today(), recipient_email)
    scenario = scenarios[scenario_id]
    api_url = local_api_url()
    api = AgentAPI(api_url, local_token())
    try:
        accepted = api.submit(scenario.prompt, None)
        run_id = accepted["run_id"]
        print(f"Scenario: {scenario.title}")
        print(f"Run: {run_id}")
        print(f"LLM trace: {ROOT / 'runtime' / 'llm-traces' / f'{run_id}.jsonl'}")
        print(f"Chat UI: {api_url.replace(':8000', ':8501')}")
        print("Manual checks:")
        for check in scenario.checks:
            print(f"- {check}")

        deadline = time.monotonic() + timeout_seconds
        previous_status = None
        while True:
            status = api.request("GET", f"/runs/{run_id}")
            state = status.get("status") or "unknown"
            if state != previous_status:
                print(f"Run status: {state}")
                if state == "waiting_for_input":
                    kind = status.get("wait_kind") or "input"
                    print(f"Action required in the chat UI ({kind}); the runner will keep polling.")
                previous_status = state
            if state in FINAL_STATUSES:
                result = status.get("result") or {}
                if result.get("summary"):
                    print("\nAgent response:")
                    print(result["summary"])
                if state != "completed":
                    detail = (
                        " · ".join(
                            str(value)
                            for value in (
                                status.get("error_code") or result.get("error_code"),
                                result.get("failure_reason"),
                                status.get("error_type"),
                            )
                            if value
                        )
                        or "No failure code recorded"
                    )
                    print(f"Failure detail: {detail}")
                print(f"LLM calls recorded: {len(status.get('llm_activity') or [])}")
                return 0 if state == "completed" else 1
            if time.monotonic() >= deadline:
                print(f"Timed out; run {run_id} remains {state}.")
                return 2
            time.sleep(poll_seconds)
    except UIError as error:
        print(str(error), file=sys.stderr)
        return 1
    finally:
        api.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", choices=("invoice", "sales", "dispute"))
    parser.add_argument(
        "--recipient-email",
        default="agent.test.recipient@example.com",
        help="Synthetic sandbox recipient for the invoice scenario",
    )
    parser.add_argument("--timeout", type=int, default=600, help="Maximum wait in seconds")
    parser.add_argument("--poll", type=float, default=2, help="Status polling interval in seconds")
    args = parser.parse_args()
    if args.timeout < 1 or args.poll <= 0:
        parser.error("--timeout must be positive and --poll must be greater than zero")
    try:
        return run(args.scenario, args.recipient_email, args.timeout, args.poll)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
