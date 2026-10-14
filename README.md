# Simplified Agentic System

Incremental implementation of [the supplied LLD](docs/LLD.md), using the agreed
compact modular structure.

## Implemented foundation

- All 13 application tables are mapped with lifecycle constraints, tenant foreign
  keys, idempotency/action/input uniqueness, text-search indexes and vector columns.
- A frozen Alembic revision creates the schema with an explicitly chosen embedding
  dimension. Application migrations exclude LangGraph-owned tables.
- Per-process database/checkpoint pools and an HTTP client open and close through
  FastAPI lifespan. Startup validates schema revisions and embedding dimensions.
- Server-configured bearer-token hashes bind trusted tenants, users and scopes.
- Run submission commits the run, conversation message and event together.
  Duplicate keys replay the existing run; conflicting content returns 409.
- Current status queries enforce ownership and tenant scope. Cancellation is
  idempotent and preserves unresolved submitted-write evidence.
- API errors have a correlation ID and omit raw validation inputs. Liveness and
  resource readiness are separate.
- A provider-neutral LLM gateway currently uses local Ollama for chat and retains
  Google AI Studio as an optional generation provider.
  Classification, plan proposals, grounded summaries and knowledge answers validate
  structured output locally. The gateway is initialized once per API process.
- Langfuse captures redacted prompts/responses, model parameters, token usage,
  validation failures and request/run correlation. Telemetry is optional and its
  failures do not change application outcomes.

- Registry publication/revocation and exact-version lookup enforce tenant scope,
  configured adapters/validators and immutable published versions.
- Sequential read plans validate dependencies and input contracts. System Search
  works standalone with scoped capabilities, fresh status and redacted event history.
- Async Gemini embeddings, scoped hybrid retrieval and standalone RAG are implemented.
  RAG rejects changed/revoked evidence and requires valid sources for generated answers.

The graph and worker support sequential reads and guarded PayPal sandbox
Invoicing writes, including durable per-write approval, single dispatch,
saved-result replay and manual review for uncertain outcomes. Models propose
plans from registered tool metadata; no business sequence is selected by
application code. Generic provider write policies, automated reconciliation and
durable retry timers remain pending. Local Ollama generation, structured output,
and synthetic tool-call continuation have been verified.

## PayPal collection drafts

The collection converter produces one inert draft card per distinct HTTP method
and endpoint, plus a metadata-only inventory. It never executes Postman scripts
or copies saved responses, credentials or example customer data into cards:

```powershell
.venv\Scripts\python.exe -m app.cli convert-collection "Postman API collection\PayPal APIs.postman_collection.json"
```

Review the ignored `runtime/tool-drafts/inventory.json` and individual card
files. The current collection yields 113 drafts from 116 request entries, with
two duplicate variants and one OAuth token request excluded. Four Invoicing
endpoints are allowlisted: list, retrieve, create draft and send.
The generated names, descriptions, actions, and categories use endpoint terms
rather than a repeated provider label; stable IDs and adapter/security metadata
still identify the sandbox integration.

Set `PAYPAL_SANDBOX_CLIENT_ID`, `PAYPAL_SANDBOX_CLIENT_SECRET`, and
`PAYPAL_SANDBOX_TENANT_ID` in `.env`. Grant the bound local identity `paypal:read`
for reads and `paypal:write` for writes. No PayPal credentials are required to
generate drafts. Import and publish reviewed cards with `agent-admin import-tool`
and `agent-admin publish-tool`; write import and execution additionally require
`PAYPAL_SANDBOX_WRITES_ENABLED=true`. Keep this flag off until the sandbox
create/send smoke test passes with your credentials. In the primary appv2 chat,
enabled sandbox writes execute directly and their dispatch/result events are
stored in PostgreSQL. The legacy API workflow retains its operation approval
and uncertain-write review behavior; its worker is no longer started by
`start.ps1`.

To test the read-only list flow end to end through the agent API, first start the
local services and publish the list-invoices tool with `paypal:read` access. Then
run:

```powershell
.\.venv\Scripts\python.exe scripts\test_paypal_invoice_list.py
```

The program submits “List up to five invoices in my PayPal sandbox account. Show
the invoice ID, status, and amount.”, polls the run, and prints up to five rows.
It uses the ignored `runtime/local-auth.json` bearer token and the configured
local API; PayPal credentials stay in the worker's `.env` configuration.

## One-command local startup

From PowerShell in the project root, run:

```powershell
.\start.ps1
```

The script checks `.env` and `runtime/local-auth.json`, installs the UI
dependencies when needed, starts PostgreSQL and document storage, applies
migrations, prepares the document bucket, and launches appv2 on port 8501. The
UI and data services stay attached to this terminal; closing it stops services
started by that invocation. If 8501 is busy, startup selects the next free port
from 8502 through 8510. Use `-SetupOnly` to start the data services and apply migrations
without launching the UI, or `-SkipInstall` when dependencies are already
installed.
If PowerShell blocks local scripts, run `powershell -ExecutionPolicy Bypass -File .\start.ps1`
for this invocation rather than changing the machine policy.

## Local chat interface

Install the optional Python UI dependency from the project root if needed:

```powershell
uv pip install --python .venv\Scripts\python.exe -e ".[ui]"
```

`start.ps1` launches appv2 automatically. To run it manually after starting
PostgreSQL and applying migrations:

```powershell
.venv\Scripts\python.exe -m streamlit run appv2\ui.py --server.address 127.0.0.1 --server.port 8501
```

Open `http://127.0.0.1:8501`. Conversations, turn messages, and redacted
activity events are stored in PostgreSQL and scoped to the local development
identity. The interface uses the existing published tool registry and does not
include tool onboarding or administration screens.

## Local setup

Use Python 3.11 or later, from the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Set unique local `OBJECT_STORAGE_ACCESS_KEY` and `OBJECT_STORAGE_SECRET_KEY` values
in `.env` before starting Docker services. Compose passes these to MinIO without
embedding credentials in its image. Then run `docker compose up -d --build`.
The first MinIO startup compiles its pinned official source release using
`Dockerfile.minio`; subsequent startups reuse the built image. The source is
[the official MinIO release](https://github.com/minio/minio/releases/tag/RELEASE.2025-10-15T17-29-55Z).

Set `EMBEDDING_DIMENSION` in `.env` to the selected embedding model's dimension.
This is required for the first migration and must stay consistent across all
deployments using that database. Changing dimensions later requires a migration
and re-embedding. Do not treat the test dimension of 3 as a production default.

Configuration loads the root `.env` through `load_dotenv(override=False)` in
`app/settings.py`, then Pydantic validates types and masks secrets. Existing process
environment variables take precedence; deployment can inject them directly.
Run commands from the project root. Restart processes after editing `.env`.
Tests disable dotenv loading to avoid using local credentials.

The primary appv2 chat classifies requests, then retrieves and binds only
published tools authorized for the tenant and relevant to each action. Python
validates tool arguments and refreshes authorization before dispatch. RAG and
System Search retain their tenant-scoped implementations. New appv2 turns use
PostgreSQL conversation/run/event records without graph checkpoints or worker
recovery; the original API and graph remain available for legacy runs.

Chat retrieves up to five eligible tools per classified action. It never sends
the full catalog to the model. Enabled sandbox writes execute directly and are
recorded in the Postgres event history.

Legacy v1/v2/v3 runs write an incremental, redacted JSONL exchange trace to
`runtime/llm-traces/<run-id>.jsonl`. `LLM_TRACE_DIR` changes the local output
directory.

To preview or clear the local exchange history and persisted `debug_*` events,
run `python scripts/clear_llm_history.py`. The command previews counts by default;
pass `--apply` and type `DELETE` to remove them. Add `--files-only` to clear JSONL
traces without connecting to PostgreSQL. It leaves workflow history, checkpoints,
operational events, and remote Langfuse traces intact.

Then initialize the application and checkpoint schemas separately:

```powershell
python -m alembic upgrade head
python -m app.cli setup-checkpoints
python -m app.main
```

The last command uses a Selector event loop on Windows, as required by Psycopg.
The installed `agent-api` command is equivalent. Uvicorn's ordinary Linux entry
point also works. Startup does not run migrations or silently create checkpoint
tables. Without a database URL, the API can expose its documentation/liveness
but readiness and database operations remain unavailable.

## Initial authentication

Configure `AUTH_BINDINGS_JSON` in `.env` as a JSON array of server-owned records:

```json
[
  {
    "token_sha256": "<64-character lowercase SHA-256 hash of a random bearer token>",
    "tenant_id": "<tenant UUID>",
    "user_id": "developer",
    "scopes": ["runs:read", "runs:write"]
  }
]
```

Send the original token as `Authorization: Bearer <token>`. Hashes are not
accepted as bearer tokens. Requests cannot override tenant, user or scopes.
`runs:manage` grants access to other owners' runs within the same tenant only.
Submission to an existing conversation still requires its owner.

This is a small initial auth backend, not an external identity-provider
integration. Bindings load once per process; configuration changes/revocation
require restarting the API and worker. Do not commit real tokens or bindings.

The configured local development identity is `developer`. Its generated token
and tenant information are stored in ignored `runtime/local-auth.json`; `.env`
stores only the token hash and binding. Read the token locally when constructing
an authorization header, and keep it out of source code and console output.
It has run submission/read, System Search, knowledge read and tool-management
permissions. Restart running API/worker processes after changing bindings.

`POST /runs` also requires an `Idempotency-Key` header. Browse `/docs` for request
contracts. `GET /runs/{id}` exposes waiting metadata and an as-of timestamp.
`POST /runs/{id}/input` accepts checkpoint-matched clarification payloads and
deduplicates accepted inputs. Operation-bound approval inputs remain blocked.

## Read-only worker

```powershell
python -m app.worker.main
```

The worker uses a dedicated PostgreSQL advisory lock and processes one run at a
time. `--once` processes at most one queued run. Queue claims commit before model
or tool calls; graph checkpoints use synchronous durability. Permissions and
result access are refreshed around nodes, and only registered read adapters run.
Read calls may repeat after a lost checkpoint; business writes remain disabled.

After confirming a previous worker process has stopped, explicitly recover its
unfinished runs using its UUID from the startup log:

```powershell
python -m app.worker.main --recover-worker <previous-worker-UUID>
```

A stale heartbeat alone does not authorize recovery. Existing checkpoints choose
initial execution, continuation, clarification resume or projection repair; a
completed checkpoint is not re-executed. Runs with existing write operations or
unsupported executable versions require review. The database-backed path is
tested with mocked model decisions; live model-backed execution remains pending.

## Document ingestion

Initialize the bucket, then import a reviewed Markdown/plain-text document:

```powershell
python -m app.cli setup-storage
python -m app.cli import-document tests/fixtures/policy.md --corpus demo --source-key example-store-policy
```

The importer reads its bearer token from ignored `runtime/local-auth.json` by
default (`--token-file` can select another local token file). The identity requires
`knowledge:manage`. Source files must be UTF-8 `.md` or `.txt`; extraction retains
headings and chunks carry line locators. New documents require `knowledge:read`
for retrieval by default; repeated `--scope` options can set document permissions.
Replacement imports preserve existing permissions when `--scope` is omitted.

Original bytes are archived before a short transaction activates a complete
document version and its chunks. Embedding/upload failures preserve the old
active version. Re-importing an unchanged active version returns its existing ID.
Failed activation may leave archived payloads for the pending orphan-cleanup job.
Automatic cleanup and workflow payload offload remain pending.

## Standalone knowledge retrieval

Model calls use configurable bounded retries for temporary provider failures,
including HTTP 503. Defaults are `LLM_MAX_ATTEMPTS=3`,
`LLM_RETRY_INITIAL_SECONDS=10`, and `LLM_RETRY_MAX_SECONDS=30` (waits: 10, 20).
`LLM_TIMEOUT_SECONDS` limits each attempt. Authentication/contract failures do not
retry; cancellation interrupts waits. SDK automatic retries are disabled so
attempt budgets do not multiply. These retries do not apply to business writes.

```powershell
python -m app.cli query-knowledge "What is the return window?" --corpus demo --mode context
python -m app.cli query-knowledge "What is the return window?" --corpus demo --mode answer
```

These commands use the local token file and require `knowledge:read`. They run
the standalone knowledge tool without enqueueing an agent workflow. Retrieval
filters tenant, document scopes, corpus, active/effective version and embedding
version before lexical/vector ranking. Answer mode cites retrieved chunk IDs or
returns insufficient evidence. Context mode returns source passages without a
generation call; its evidence flag means passages were found, not that an answer
was verified. Live context retrieval is verified; answer generation remains pending.

The application does not cap model input or output tokens. It keeps retrieval
candidate counts and document passage counts bounded so prompts remain relevant.
Provider-reported token usage is saved per completed model call, shown in the chat
activity panel, and sent to Langfuse when tracing is enabled.

## Tool onboarding

```powershell
python -m app.cli onboard-collection "Postman API collection/PayPal APIs.postman_collection.json" --fetch-specs --enable-supported
python -m app.cli import-tool resources/tool_cards/system_search.json
python -m app.cli publish-tool system_search --version 1
python -m app.cli revoke-tool system_search --version 1
```

Collection onboarding requires `tools:manage` and reads the local bearer-token
file. It streams and converts the collection, matches official API contracts,
and imports cards through the standard embedding service. With
`--enable-supported`, it evaluates nearby cosine-distance thresholds against
`resources/taxonomy/retrieval-evaluation.json`, clusters descriptions locally,
then asks Gemini to label the shallow taxonomy and review sanitized tool
descriptions, synonyms, entities, actions, and assignments. It publishes only
cards approved by Gemini and the deterministic adapter, schema, and policy
checks. Unsupported cards and failed reviews remain drafts. Gemini receives no
schemas, endpoint paths, Postman examples, credentials, or customer data.
The resumable report and cached review are written under `runtime/tool-drafts`;
compatible embeddings and reviewed versions are reused on reruns. Every
published write still requires per-operation user approval.

The individual `import-tool`, `publish-tool`, and `revoke-tool` commands remain
available for built-in cards and manual administration. Publication keeps old
versions available to pinned plans.

For local review without Gemini, pass `--reviewer manual`. This uses the
configured clustering threshold and locally reviewed route-family labels,
checks descriptive metadata, then runs the same deterministic publication
gates. It does not call Gemini for review or retrieval-query embeddings. Cards
with ambiguous identity-token actions stay drafts. The default remains Gemini.

## Verification

```powershell
python -m pytest -q
python -m ruff check app migrations tests
python -m compileall -q app migrations tests
```

Database integration tests are opt-in: set `TEST_DATABASE_URL` to an isolated,
migrated PostgreSQL/pgvector database whose checkpoint schema has been initialized.
The test schema must use `TEST_EMBEDDING_DIMENSION` (default 3); it is not the main
development database. Integration tests insert data and exercise real constraints.

Validated in an isolated PostgreSQL 16/pgvector container: concurrent duplicates,
conflicting submissions, ownership/tenant isolation, scoped last-request lookup,
cancellation, append-only events, taxonomy depth, vector dimensions, pool cleanup,
migration downgrade/re-upgrade and an Alembic drift check. Remaining acceptance
scenarios are recorded in [tests/integration/README.md](tests/integration/README.md).

## Next slice

Verify live model-backed execution when Gemini generation is available, then add
durable retry timers and crash/restart fault tests.

Dependencies and container tags remain provisional; the Google SDK is pinned to
the tested version. Business-provider integration has not been checked or locked.
The local Compose API profile consumes
`EMBEDDING_DIMENSION` and `AUTH_BINDINGS_JSON` from the environment/`.env`.
The local terminal commands above are the verified development path; the full
Compose app profile has not been validated end to end. Object buckets are not
created automatically; Langfuse uses the credentials and endpoint supplied in `.env`.

See [the design checklist](docs/design.md) for component ownership and remaining work.

## LLM provider

Chat generation uses the locally running Ollama model selected in `.env`:

```dotenv
LLM_PROVIDER=ollama
OLLAMA_MODEL_NAME=llama3.2
OLLAMA_BASE_URL=http://127.0.0.1:11434
```

The Ollama adapter uses the local `/api/chat` endpoint for JSON-schema responses
and tool calls. The application validates every proposed tool call before dispatch.
Ollama does not provide the gateway's optional Google Search feature. The configured
Gemini API key remains necessary for document and tool embeddings; it is not sent to
Ollama. The explicitly named Gemini onboarding reviewer also remains separate from
the chat provider.

To select Gemini for chat again, set `LLM_PROVIDER=gemini` and provide
`GEMINI_MODEL_NAME` and `GEMINI_API_KEY`. No generic `MODEL_NAME` or
`MODEL_API_KEY` setting is used. Restart the API and worker after changing providers.
Existing paused runs may contain provider-specific continuation; finish them with
their original provider when possible.

To make one small model call after setup:

```powershell
python -m app.cli check-llm
```

The gateway remains provider neutral through the `LLMProvider` interface in
`app/agent/gateway.py`. The local adapter follows Ollama's
[chat API](https://docs.ollama.com/api/chat) and
[tool-calling format](https://docs.ollama.com/capabilities/tool-calling).

## Langfuse tracing

The integration uses the existing `.env` credentials:

```dotenv
LANGFUSE_PUBLIC_KEY=<project public key>
LANGFUSE_SECRET_KEY=<project secret key>
LANGFUSE_BASE_URL=<your project's regional or self-hosted URL>
LANGFUSE_TRACING_ENABLED=true
LANGFUSE_CAPTURE_CONTENT=true
LANGFUSE_ENVIRONMENT=development
LANGFUSE_TIMEOUT_SECONDS=5
LANGFUSE_MAX_CONTENT_CHARS=32000
```

Prompt/response capture is enabled as requested. Exported content masks configured
secrets, sensitive structured fields, bearer tokens, common API-key/JWT patterns,
URL credentials and email addresses. Redaction happens on a copy before SDK calls,
with additional SDK/export-stage masking. Text is capped after redaction. These
rules are not a complete classifier for every domain's restricted data; extend
the rules when adding sensitive provider payloads. Actual LLM inputs and database
recovery records remain unchanged. Model thought steps are never exported.

Set `LANGFUSE_CAPTURE_CONTENT=false` for metadata-only traces or
`LANGFUSE_TRACING_ENABLED=false` to disable tracing. Missing credentials disable
tracing; partial configuration or telemetry errors are logged without secret
details. Langfuse is not a readiness requirement. A readiness response reports
its local `configured`/`disabled`/`degraded` state, which is not a live reachability
probe. No remote auth check is performed on application startup or every request.

Each API process creates one Langfuse client and closes it after application
resources. Standalone LLM gateways own/close their telemetry automatically.
LLM generation observations sit beneath named classification/planning/response
spans. They track usage, model configuration and safe error codes. Request IDs
use isolated context variables; run submission stores a W3C traceparent when
available so the worker can reconnect to the trace. Worker, graph and read-tool
observations are implemented for the current read-only workflow.
Infrastructure-wide auto-instrumentation is not enabled; only application-owned
observations are exported through this Langfuse client.

Verify credentials and submit a synthetic trace, without calling an LLM:

```powershell
python -m app.cli check-langfuse
```

Authentication and stored synthetic observation were verified against the
configured project using the current v2 observations API. Langfuse's
[instrumentation](https://langfuse.com/docs/observability/sdk/instrumentation),
[masking](https://langfuse.com/docs/observability/features/masking), and
[public API](https://langfuse.com/docs/api-and-data-platform/features/public-api)
documentation describe the supported SDK interfaces.
