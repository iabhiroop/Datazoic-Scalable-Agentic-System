# appv2: simple local agent chat

`appv2` is the primary Streamlit chat. It classifies each request, retrieves a
small shortlist of published tools the local identity can use, and binds only
that shortlist to the configured Gemini or Ollama model. Python validates each
call and refreshes authorization before dispatch. Tool calls run one at a time.

Conversation turns, messages, and redacted tool/model activity are stored in the
existing PostgreSQL conversation, workflow-run, and workflow-event tables. Prior
conversation messages are included as context on follow-up turns. The app does
not use a LangGraph checkpoint or worker for new turns. RAG and System Search
use the existing tenant-scoped implementations. Enabled PayPal sandbox writes
execute directly and are recorded in the run event history.

From PowerShell in the project root, run `.\start.ps1`. The terminal remains
attached to PostgreSQL, document storage, and the chat UI; closing it stops
services started by that invocation. Use `.\start.ps1 -SetupOnly` to start the
data services in the background and apply database migrations without the UI.
If port 8501 is occupied, startup chooses a free port through 8510 and prints
the chat URL.

The app uses `.env` and `runtime/local-auth.json`. It does not alter or onboard
catalog records. PayPal sandbox tools read their credentials only from `.env`.
