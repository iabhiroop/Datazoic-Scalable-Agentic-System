from datetime import date

import pytest

from scripts.run_paypal_scenario import build_scenarios, previous_month_window


@pytest.mark.parametrize(
    ("today", "expected"),
    [
        (date(2026, 10, 14), (date(2026, 9, 1), date(2026, 9, 30))),
        (date(2025, 3, 1), (date(2025, 2, 1), date(2025, 2, 28))),
        (date(2024, 3, 1), (date(2024, 2, 1), date(2024, 2, 29))),
        (date(2026, 1, 1), (date(2025, 12, 1), date(2025, 12, 31))),
    ],
)
def test_previous_month_window(today, expected):
    assert previous_month_window(today) == expected


def test_scenario_prompts_cover_automatic_invoice_gross_net_and_buyer_mapping():
    scenarios = build_scenarios(date(2026, 10, 14), "agent.test.recipient@example.com")

    assert set(scenarios) == {"invoice", "sales", "dispute"}
    assert "agent.test.recipient@example.com" in scenarios["invoice"].prompt
    assert "complete the request" in scenarios["invoice"].prompt
    assert "approve" not in scenarios["invoice"].prompt.lower()
    assert "2026-09-01 through 2026-09-30" in scenarios["sales"].prompt
    assert "gross captured sales before refunds" in scenarios["sales"].prompt
    assert "net sales after refunds" in scenarios["sales"].prompt
    assert "synthetic application alias" in scenarios["dispute"].prompt


def test_scenarios_reject_malformed_recipient_email():
    with pytest.raises(ValueError, match="valid email"):
        build_scenarios(date(2026, 10, 14), "not-an-email")
