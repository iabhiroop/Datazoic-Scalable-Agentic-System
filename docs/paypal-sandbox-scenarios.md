# Sandbox business-scenario tests

These are manual end-to-end tests for tool selection through the chat UI. Use the
local sandbox tenant and review each proposed write in the UI. Every write must
receive its own approval. Do not paste credentials or real customer details into
the prompts.

## 1. Create and send a synthetic invoice

**Prompt**

> Send an invoice for $50 to agent.test.recipient@example.com for a synthetic test item.

**Expected flow**

1. The agent selects the draft-invoice tool and requests approval before creating.
2. Review the recipient, one item, amount `50.00`, and currency `USD`; approve.
3. The agent selects the send-invoice tool for the returned invoice ID and requests a
   separate approval; review and approve.
4. The run completes with the invoice ID and sent status. Confirm it with the
   read-only invoice detail/list tool.

**Pass criteria**

- Both writes have distinct operation records and distinct approvals.
- The send uses the invoice ID returned by the create call.
- The final invoice is visible in the sandbox and is not reported as sent after an
  uncertain outcome.

The recipient address is intentionally synthetic. If PayPal rejects it because it
does not belong to a sandbox buyer, create a buyer account in the sandbox and
substitute that account's synthetic email; do not substitute a real address.

## 2. Query sales volume for the previous calendar month

**Prompt**

> What was my total sales volume last month? Show the date range, currency, and how you calculated the total.

For a test run on October 14, 2026, “last month” is **September 1–30, 2026**. The
scenario runner calculates the previous full calendar month when launched. The
phrase “sales volume” is intentionally tested with both interpretations until
the product definition is chosen:

- **Gross:** sum successful captured/sale transactions before refunds.
- **Net:** gross sales less refunds attributable to those sales in that period.

First run the read-only transaction report and record its date bounds, rows,
currencies, and gross total. If refunds are present, run the net interpretation
against those same transactions. Keep currencies separate; do not add unlike
currencies or silently convert them.

**Pass criteria**

- The selected report tool is read-only and covers exactly September 1–30, 2026.
- The result states whether it is gross or net, reports currency, and agrees with
  the returned transaction/refund records.
- If there is no data, the agent says that clearly instead of claiming zero sales.

**Data limitation**

The normal order/capture APIs create transactions at the time of capture. They do
not provide a supported way in this test flow to backdate a capture into September.
If the sandbox has no September transactions, this historical-period test cannot
be seeded through ordinary API writes. Use an existing sandbox transaction dataset
or obtain a PayPal-supported historical test fixture; a current-date capture only
tests the current-period equivalent.

## 3. Find an open dispute for a synthetic buyer

**Prompt**

> Is there an open dispute from user_123? Show the dispute ID, status, amount, and transaction date.

Map `user_123` to a synthetic sandbox buyer before running this test. The PayPal
API does not know the application's `user_123` label unless the returned dispute
or its underlying transaction can be tied to that buyer.

**Setup required to seed a dispute**

1. Create or select a sandbox buyer and record the mapping to `user_123` locally.
2. Have PayPal enable the app's sandbox `DISPUTE_CREATE` capability and configure
   the required buyer-side authorization.
3. Create a sandbox transaction from that buyer to the bound business account and
   retain the transaction ID.
4. Use PayPal's sandbox-only dispute-creation flow with the buyer authorization
   assertion. The current application does not expose this fixture-creation path;
   do not try to create a dispute through the merchant's case-management tools.
5. Query the dispute using the published read-only dispute list/detail tools and
   verify the transaction-to-buyer mapping.

**Pass criteria**

- The agent chooses dispute search/read tools only.
- It filters or verifies the dispute against the mapped buyer transaction, and
  does not treat the literal `user_123` string as a PayPal account identifier.
- It reports no open dispute when the seeded case is closed or belongs to another
  buyer.

## Current readiness

The collection has published read tools for invoice listing/details, transaction
reports, and dispute listing/details. Invoice creation and sending are available
through the approved-write workflow. The September historical dataset and the
buyer-side dispute-creation setup are external prerequisites; neither is faked by
these tests.

## Run a scenario

Start the local services with `./start.ps1`, then run one scenario from the project
PowerShell terminal:

```powershell
.\.venv\Scripts\python.exe scripts\run_paypal_scenario.py invoice
.\.venv\Scripts\python.exe scripts\run_paypal_scenario.py sales
.\.venv\Scripts\python.exe scripts\run_paypal_scenario.py dispute
```

The runner submits the prompt to the local agent API and polls the run. For invoice
creation and sending, approve each action separately by typing `approve` in the
chat or using its button. It never calls
PayPal directly or approves an action itself. The default synthetic invoice email
can be replaced with `--recipient-email` if you have created a dedicated sandbox
buyer account. Add `--timeout` to extend the polling window.
