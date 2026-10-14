Designing a Scalable Agentic System
As mentioned, the core challenge here is that LLM performance isn't static. As you keep attaching more and more tools, the accuracy degrades.When building an agentic system that needs to interact with a massive catalog of tools—such as 50+ PayPal APIs scaling up to hundreds or thousands—the biggest hurdle is LLM confusion and performance degradation. Exposing every tool definition at once overloads the model's context and interferes with tool selection.
Technology Choices and Trade-offs
Explain why each major component fits this design:
LangGraph: Makes transitions, pauses, and recovery explicit; application scheduling and execution guards still need implementation.
PostgreSQL with pgvector: Consolidates state and retrieval initially; growing search workloads may compete with transactional work.
Single worker: Simplifies the first implementation; slow calls constrain throughput.
Redis: Reduces repeated retrieval work; introduces cache invalidation requirements.
Langfuse: Provides model traces, token usage, latency, and outcome diagnostics; it does not replace durable operation records.
Here is how the design tackles this challenge across the five core areas:
1. Agent Structure
The architecture uses a single, explicit workflow graph via LangGraph. Since we are using a chat interface, it is best to minimize the use of multiple agents and maintain the thread with one using a different prompt to improve response time.
Why this works: It keeps ownership, recovery paths, and execution logic transparent and easy to reason about.
Tracking: Every user interaction can spawn multiple workflows, cleanly isolated using unique application run_ids paired with LangGraph thread_ids.
<AI insert>

The graph uses node-specific prompts for classification, planning/tool selection, and response generation. Reduced coordination overhead is an intended benefit; improved response time must be established through measurement. A conversation can contain several workflows. Clarification or approval resumes the relevant existing workflow using its saved thread identity, while a separate request receives a new run/thread pair.

</AI insert>
2. Tool Selection & Routing
To keep the LLM focused and accurate, the system avoids dumping the full catalog into the prompt.
Tool Onboarding & Normalization:
Standardized Metadata: OpenAPI definitions, Postman collections, or manual definitions are ingested and transformed into structured "tool cards." This is done through an onboarding script if we have an expected structure or can be batch processed with an LLM to create the metadata.
Rich Schemas: Each card explicitly defines stable tool IDs, purposes, input/output schemas, provider references, category tags, permissions, preconditions, success criteria, timeouts, and retry/idempotency rules.
Vector Embeddings & Taxonomy: Capabilities and descriptions are embedded into vector spaces and clustered into a shallow taxonomy to aid initial catalog organization and human review. The human review here is replaced with Gemini’s review but I recommend a onetime HITL for production systems while onboarding.
This creates the categories we need to reduce the tool scope.
<AI insert>

Gemini-assisted review is an onboarding variation from the supplied HLD/LLD, which specifies reviewed publication. Generated descriptions and category suggestions must pass deterministic contract checks. Production publication should include trusted review of permissions, allowlisted adapter mappings, financial preconditions, and provider idempotency rules. Review is also needed when those contracts change. Category predictions are soft relevance signals; authorization filters remain mandatory. If the shortlist misses a required capability, discovery expands within the same authorized catalog.

</AI insert>
Runtime Retrieval & Bounded Routing:
Filtered Discovery: At runtime, the system first screens the catalog against current tenant, permission, environment, and availability rules.
Hybrid Search: It then runs lexical and vector searches over the eligible subset, combining their rankings using reciprocal-rank fusion.
Token Budget Control: By pulling just 3 to 5 candidate tools per subgoal, the system feeds the planner a tiny, highly relevant subset under a strict token budget.
Execution Trace Ingestion (RAG Enrichment): 
Successful tool execution: Successful tool execution traces—including resolved parameters, step-by-step reasoning, and observed outcomes—are indexed into the RAG knowledge base. During runtime planning, the system retrieves semantically relevant historical execution traces alongside document context, add successful run history directly into the model's prompt to improve plan accuracy and reduce hallucinated arguments.
<AI insert>

For this report, execution-history enrichment means sanitized summaries of the goal, versioned tool sequence, parameter structure, verified outcomes, and recovery events. It excludes private model reasoning. Only reviewed or deterministically evaluated, authorized, version-compatible examples are eligible for reuse. Historical customer IDs, amounts, credentials, and approval evidence cannot be copied into a new action. Current parameters are resolved from the current request and live observations. Any improvement in planning accuracy remains an evaluation hypothesis.

</AI insert>

3. State Management
Robust state tracking ensures the system never loses its place, even if components restart mid-workflow.
Separation of Concerns: LangGraph checkpoints handle graph execution progress, while PostgreSQL records handle application scheduling, ownership, user inputs, and external-operation evidence.
Durable Waits: Pauses for user clarification, approvals, or retries use stable wait tokens and persisted input IDs, ensuring that crashes or network drops don't cause duplicate actions or lost data.
<AI insert>

Checkpointing supports recovery rather than guaranteeing that state can never be lost. Input IDs prevent duplicate application of follow-up input; operation guards and provider evidence address external-action replay. A provider write and a local checkpoint are separate commits, so there is no universal exactly-once guarantee. Recovery inspects both checkpoint progress and operation records before continuing.

</AI insert>
4. Scalability
The system is built to scale both its tool catalog and its processing load gracefully.
Catalog Growth: Handled by indexing the tool registry and maintaining a tight cap on candidate context size.
Throughput & Infrastructure: While the initial setup relies on a single sequential worker and PostgreSQL (consolidating transactional state, full-text search, and vector embeddings), horizontal scaling is accounted for by designing worker exclusivity, ownership checks, and safe recovery boundaries from the ground up.
5. Error Handling & Safety
When dealing with financial APIs or external writes, failure handling must be deterministic and safe.
No Model-Generated Code: Python code—not the LLM—validates all plans, checks dependencies, rejects invalid schemas, and verifies permissions.
Guarded Execution: Preconditions, permissions, and approvals are checked right before dispatch and re-verified if a task was paused waiting for input.
Operation Tracking: Every write has a stable operation identity tracking payload hashes and outcomes. If a request fails halfway, the system relies on provider reconciliation or saved-result replay rather than blindly executing financial actions twice.
This concise summary bridges the gap between high-level architectural goals and practical engineering safeguards for massive tool catalogs.
<AI insert>

## 6. Intent Classification and Stepwise Execution

The classifier extracts the goal, ordered subgoals, entities, constraints, and missing information. Ambiguous recipients, resource IDs, amounts, or currencies trigger clarification before a financial write.

The current repository LLD introduces two execution modes. `plan_only` returns proposed steps without dispatching tools. `execute` presents the current subgoal's authorized shortlist as function declarations and allows one proposed call at a time. Python resolves that call within the shortlist, validates its input contract, refreshes authorization, and executes it through the gateway. The graph uses the resulting observation to continue.

Dependent actions use validated outputs from earlier calls. For example, an invoice creation result supplies the invoice ID for the subsequent send action. Model output cannot introduce an arbitrary endpoint, executable code, or a tool outside the eligible shortlist. The original structured-plan design also validates dependency order, output bindings, and plan limits before execution.

## 7. Separate RAG Pipeline and System Search Tools

**RAG Pipeline:** A registered read-only tool retrieves authorized documentation and business-policy passages. It supports context retrieval or a grounded answer, includes source/version references, and reports insufficient evidence when the knowledge base cannot support an answer. Critical financial eligibility rules remain enforced by approved code and current provider data.

**System Search:** A separate registered read-only tool supports capability discovery, request status, and sanitized execution history. It accepts typed filters rather than model-generated SQL or unrestricted log queries. "My last request" resolves within the authenticated user's conversation and ownership scope, excluding the current search request. Critical status queries read authoritative database records rather than stale cached summaries.

Document knowledge, reusable execution summaries, and live financial state serve different purposes. Current payment, invoice, and dispute status comes from provider APIs.

## 8. Action-Bound Approval and Outcome Verification

When policy requires approval, it binds to the specific operation, normalized payload hash, target, amount/currency, and principal. The application validates that evidence again before dispatch. A changed financial action requires renewed approval; an uncertain submitted write cannot be approved into a blind retry.

Adapters normalize observations into `succeeded`, `pending`, `failed`, or `unknown`. An accepted request remains pending until its required postconditions are verified. Graph completion alone does not prove business completion.

Partial effects are preserved. If invoice creation succeeds but sending fails, the response identifies the created invoice and the unsuccessful send step. Recovery must not create a second invoice simply because the overall request remains incomplete.

## 9. Bounded Recovery and Request Lifecycle

Transient reads may retry with bounded exponential backoff and jitter. Rate limits use provider guidance and persisted retry times. Replanning has a configured budget and starts from verified state. Permission or business-rule failures stop the prohibited action rather than trying another tool to bypass the restriction.

Pending or unknown writes require provider reconciliation under endpoint-specific rules. If lookup or idempotency support cannot establish a safe outcome, the run moves to `needs_review`. The supplied LLD proposes initial limits of three total safe attempts and two replans; these are tuning values, not measured optimal settings.

FastAPI exposes submission, status, follow-up input, and cancellation. A client submission key returns the existing run for the same request and rejects conflicting reuse. Stable wait tokens bind clarification or approval to the expected pause. Scheduled recovery stores due times in PostgreSQL so a worker restart does not erase the schedule.

Cancellation stops future business dispatches. It does not reverse an in-flight action or erase its evidence; an unresolved submitted write still needs its outcome established and reported.

## 10. Versioning and Worker Ownership

Published tool contracts are immutable, versioned, and revocable. Active runs pin compatible graph, prompt, tool, policy, and embedding versions. Revocation blocks new dispatch even for a previously selected tool. Incompatible deployment changes must retain the old handler or explicitly migrate affected runs. The current LLD retains graph v1 for older runs while new requests use v2.

The initial deployment has one active sequential worker. A stale heartbeat flags possible failure but does not prove that the previous worker or its remote call stopped. Replacement requires confirmed ownership handover and inspection of saved evidence. Multiple workers are a future extension requiring exclusive claims and ownership enforcement; horizontal scaling is not an existing throughput guarantee.

## 11. Safe Caching, Object Storage, and Security

Redis caches tool metadata, safe knowledge context, and short-lived session/progress summaries using tenant, authorization-scope, and relevant revision boundaries. PostgreSQL remains authoritative. Cache loss falls back to normal retrieval, and cache hits cannot grant permission. Semantically similar financial results are never treated as evidence that a new write completed.

Original documents and large payloads live in object storage with authorized database references, checksums, and retention rules. Prompts and checkpoints contain small typed results or references. Essential provider IDs and statuses remain recoverable even if storing a large response fails.

Imported tool descriptions, documents, and outputs are untrusted data. Their instructions cannot change execution policy. Credentials, tenant identity, adapter mappings, and endpoint routing come from trusted server configuration. Authorization is enforced at submission, retrieval, validation, and dispatch, including after waits. Logs and traces redact secrets and restricted fields.

## 12. Observability and Evaluation

Correlate request, conversation, run, operation, and trace IDs across the API, graph, gateway, and provider calls. Langfuse and infrastructure telemetry record candidate retrieval, selected tools, validation failures, attempts, verification, recovery decisions, token usage, latency, and cost. Telemetry export failure must not invalidate durable execution evidence.

Evaluate catalog sizes of approximately 50, 100, 500, and 1,000 tools against an all-eligible-tools baseline using the same model and authorization rules. Include similar tools, cross-category requests, compound goals, missing arguments, and restricted capabilities. Record any baseline context-limit failure explicitly.

Measure candidate recall, correct tool selection, argument validity, verified task completion, prompt tokens, latency, and cost. Fault-injection checks cover duplicate submission, provider success before acknowledgment, saved results before checkpoint failure, restart during waits, revoked permissions, cancellation during writes, and cache/telemetry outages. Duplicate and unauthorized writes are critical failure indicators. Accuracy and response-time improvements remain unproven until these measurements are available.

## 13. Illustrative End-to-End Flows

**Invoice:** For "Send an invoice for $50 to customer ABC," resolve the customer, currency, and required invoice details; retrieve permitted tools; validate the proposed actions; obtain required approval; create or retrieve the invoice; send it; and verify its resulting status. Preserve and explain partial completion.

**Sales report:** For "What was my total sales volume last month?", establish timezone, calendar boundaries, sales definition, and currency treatment. Retrieve current provider records, account for pagination, and calculate totals in deterministic code. Keep currencies separate unless an explicit conversion policy is available.

**Request status:** For "What happened to my last request?", use System Search to select the prior owned request and return its timestamp, status, verified effects, and any waiting or review reason.

These examples describe intended behavior, not measured sandbox results.

## 14. Implementation Status and Limitations

This report combines architectural goals with implementation safeguards. Proposed features must be distinguished from demonstrated behavior. The repository README reports scoped retrieval, standalone RAG/System Search, sequential graph execution, and selected PayPal sandbox invoice writes with durable approval and saved-result replay. It identifies generic provider write policies, automated reconciliation, and durable retry timers as pending. These status statements reflect documentation, not independent verification performed for this report.

Remaining limitations include single-worker throughput, retrieval potentially missing a required tool, provider-specific reconciliation constraints, and the separate commit boundaries between provider actions, application records, and graph checkpoints. Supporting thousands of registered tools does not by itself demonstrate high execution throughput or reliable task completion.

Before submission, add measured retrieval and end-to-end results, model/embedding versions, catalog size, test conditions, and exercised recovery scenarios. Remove the AI insertion markers after reviewing the additions and align the original claims with the qualifications above.

</AI insert>
