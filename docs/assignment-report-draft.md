# Design of a Scalable Agentic System

**Datazoic Programming Task — Report Draft**  
**Date:** 14 October 2026  
**Basis:** Supplied problem statement, Simplified Agentic System HLD v0.1, and accompanying LLD.  
**Status:** Design report; performance and scalability results remain to be measured.

## 1. Executive summary

This report proposes an agentic system that converts natural-language requests into validated API actions while supporting a growing catalog of tools. The assignment identifies the central challenge: exposing many tools to an LLM can degrade tool selection and parameter generation. The proposed architecture addresses this through selective discovery, bounded model context, and deterministic execution checks.

The system stores the full tool catalog in a versioned registry, retrieves a small set of relevant and authorized candidates for each subgoal, and asks the LLM to produce a structured plan. Python code validates the plan and executes registered adapters. PostgreSQL stores workflow progress and external-operation evidence so interrupted requests can be recovered without blindly repeating financial actions.

The initial deployment uses FastAPI and one sequential LangGraph worker. PostgreSQL with full-text search and pgvector supports retrieval and persistence; Redis provides optional caching; object storage holds documents and large payloads; Langfuse supports model observability. RAG Pipeline and System Search are separate registered tools. The design provides a path toward larger catalogs, while acknowledging that catalog scalability and execution throughput require different solutions.

## 2. Problem statement and objectives

The assignment asks for a system that can use more than 50 PayPal API operations and eventually hundreds or thousands of tools across providers. Users should interact through ordinary chat, without knowing endpoint names or API schemas. Example requests include sending an invoice, calculating last month's sales volume, and checking disputes for a customer.

Passing every tool definition to the model creates a large decision space and consumes context that could otherwise support planning. Similar operations may differ in required identifiers, permissions, and business effects. A plausible tool call is therefore insufficient: the system must also establish that the selected operation is appropriate, its arguments are valid, and its outcome satisfies the user's request.

The design has five objectives:

1. Discover relevant tools without exposing the entire catalog to the planner.
2. Translate user intent into structured, validated plans with explicit dependencies.
3. Preserve state across clarification, approval, retries, and worker restarts.
4. Handle uncertain external writes without creating duplicate financial effects.
5. Make selection, execution, recovery, latency, and cost observable and evaluable.

The problem statement's guidance is treated as assignment context. The HLD and LLD supply proposed design decisions; they do not constitute evidence that all capabilities have been implemented or tested.

## 3. Scope and assumptions

The first version targets invoice, payment, refund, dispute, and reporting operations, with one Python codebase and two processes: an API service and a background worker. Execution is sequential. Separate agents for each category, Kafka, Temporal, and distributed compensation orchestration are outside this initial design.

Each API operation must have reviewed metadata and an allowlisted adapter before it can execute. Provider credentials and user permissions come from trusted application configuration. Endpoint-specific idempotency and reconciliation behavior must be established during adapter onboarding.

The exact API inventory, approval rules, tenant model, reporting definitions, and retention policies remain configuration decisions. Candidate counts, context budgets, and retry limits are starting hypotheses rather than measured optimal settings.

## 4. Proposed architecture

FastAPI authenticates callers and exposes request submission, progress, follow-up input, and cancellation endpoints. A submitted request is committed to PostgreSQL before the API returns its run identifier. The worker claims queued work and advances the LangGraph workflow.

The main processing path is:

**Request → intent classification → authorized tool retrieval → bounded context → structured plan → deterministic validation → guarded execution → outcome verification → response.**

Failures route to bounded recovery, durable waiting, or human review. Resuming a workflow uses its saved checkpoint and pending state rather than automatically starting the request again.

The components have distinct responsibilities:

- **LangGraph:** explicit workflow transitions, checkpointed progress, and interruption/resumption.
- **PostgreSQL:** registry, runs, inputs, operation records, audit events, and knowledge metadata.
- **Full-text search and pgvector:** lexical and semantic retrieval of tools and supporting context.
- **Redis:** optional caches with access-scope and version boundaries.
- **Object storage:** original documents and large payloads referenced from database records.
- **Langfuse and infrastructure telemetry:** diagnostic traces, model usage, latency, and failure evidence.

PostgreSQL and provider observations remain the recovery authorities. Cache entries and observability traces do not determine whether a financial action succeeded.

## 5. Tool onboarding and selection

OpenAPI definitions, Postman collections, or manual definitions are normalized into tool cards. Each card contains a stable tool ID, purpose, input/output schemas, provider and adapter references, category tags, permissions, preconditions, success criteria, timeout, and retry/idempotency rules.

Embeddings represent business capabilities and descriptions. Clustering suggests a shallow taxonomy for human review. Categories help discovery; they cannot grant permissions or create executable endpoints. Cross-domain tools can carry multiple tags.

At runtime, the system first applies tenant, permission, environment, and availability filters. It then performs lexical and vector searches within the eligible catalog and combines their rankings using reciprocal-rank fusion. Category classification provides a soft relevance preference so an incorrect category prediction does not permanently exclude a necessary tool.

The proposed starting point is approximately three to five candidate tools per subgoal. If coverage is insufficient, discovery expands within the authorized set. The planner receives selected schemas, relevant conversation facts, a few knowledge passages, and compatible execution summaries under a configured token budget.

This reduces the catalog-dependent portion of planner context. It does not guarantee constant total context: complex requests may contain several subgoals, and retrieval cost still grows with the indexed catalog. Retrieval recall must therefore be measured alongside final tool-selection accuracy.

## 6. Planning, validation, and execution

The classifier extracts the goal, subgoals, entities, constraints, relevant categories, and missing information. Ambiguous recipients, amounts, currencies, or resource identifiers trigger clarification before a write.

The planner returns registered tool IDs, pinned schema versions, arguments, dependencies, and completion conditions. Arguments are literal values or typed references to previous outputs. Python resolves those references and rejects unknown tools, invalid schemas, missing dependencies, cycles, excessive plan lengths, and unauthorized actions. Model-generated code is not used to bind arguments.

Before each dispatch, the system refreshes authorization and checks cancellation, resource ownership, business preconditions, and required approval. These checks run again after a wait because permissions or resource state may have changed.

Every intended write has a stable operation identity that survives retries and replanning. The operation record stores its normalized payload hash, tool version, provider key, attempts, and outcome. A saved verified success is replayed locally. A pending or unknown operation is reconciled before another write is considered.

## 7. State management and failure handling

A conversation can contain several workflows. Each workflow has its own application `run_id` and LangGraph `thread_id`. Checkpoints store graph progress; application records store scheduling, ownership, user input, and external-operation evidence. These persistence boundaries are separate and must be repaired after partial failures.

Durable waits cover clarification, approval, retries, and reconciliation. A stable wait token binds follow-up input to the expected interruption. Persisted input IDs prevent the same response from being applied twice after a crash. Scheduled recovery uses stored due times rather than relying on a long-running in-process sleep.

Recovery depends on the observed failure:

- Transient reads may retry with bounded backoff.
- Rate limits schedule a retry using provider guidance.
- Invalid parameters require correction or clarification.
- Permission and business-rule rejections stop the prohibited action.
- An unsuccessful strategy may replan from verified state when no write remains unresolved.
- An ambiguous write requires reconciliation or a review state.

For example, a refund may succeed remotely before the worker saves the response. Recovery must establish that operation's outcome through provider lookup or supported idempotency behavior. It must not invent a new refund operation. The design does not claim universal exactly-once execution because the provider action and local database commit are not one atomic transaction.

## 8. RAG Pipeline and System Search

The **RAG Pipeline Tool** retrieves authorized, active document passages with source and version references. It supports returning context or a grounded answer. Insufficient evidence produces an explicit evidence gap. Retrieved policy text can explain a decision, while approved business checks and current provider data govern financial eligibility.

The **System Search Tool** supports capability discovery, request status, and sanitized execution history. It queries scoped registry and application records through typed filters. It does not accept model-generated SQL or unrestricted log access. “My last request” resolves within the user's conversation and ownership scope, excluding the current search request.

These tools answer different questions: RAG searches external knowledge; System Search inspects application capabilities and execution evidence. Current financial state is obtained from live provider APIs.

## 9. Illustrative request flows

### Sending an invoice

For “Send an invoice for $50 to customer ABC,” the system resolves the recipient, currency, and required invoice details. It retrieves appropriate invoice creation, lookup, and send capabilities. A validated plan creates the draft when necessary, binds its returned identifier to the send step, obtains any policy-required approval, and verifies the provider's resulting status. A create success followed by a send failure is reported as a partial outcome with the existing invoice reference preserved.

### Reporting sales volume

For “What was my total sales volume last month?”, the system resolves timezone, calendar boundaries, sales definition, and currency treatment. It retrieves reporting or transaction-read capabilities, handles pagination, and calculates the agreed total using deterministic code. The response states the reporting period and basis of calculation. Multi-currency values are kept separate unless an explicit conversion policy is available.

### Checking a dispute

For “Is there a dispute open from user_123?”, the system resolves the user reference to the relevant provider identity, retrieves permitted dispute reads, and verifies the returned statuses. An ambiguous customer mapping requires clarification; the model cannot guess the account identity.

These flows illustrate proposed behavior. They are not records of completed sandbox executions.

## 10. Technology choices and trade-offs

LangGraph fits the proposed explicit node flow and resumable state. The design favors one controlled graph over separate domain agents to keep ownership, recovery, and execution policy easier to reason about. This choice still requires application code for scheduling and operation guards. LangChain-style integrations or LlamaIndex-style retrieval components could be used where useful without replacing those application responsibilities. Multi-agent frameworks would add coordination complexity that the first version does not require.

PostgreSQL consolidates transactional state, lexical search, and initial vector retrieval. This reduces infrastructure breadth, but search and workflow workloads can compete for database resources. Exact vector search is a reasonable initial design for a small tool catalog; larger datasets require benchmarking before introducing approximate indexes or a separate retrieval service.

Langfuse is the selected model-observability platform, satisfying the assignment's request to consider observability. It records planning and execution diagnostics alongside infrastructure telemetry. This report does not claim a measured advantage over LangSmith.

A dedicated workflow platform such as Temporal could become appropriate as scheduling and distributed recovery requirements grow. The first version accepts the operational responsibility of an application-managed worker. Redis improves optional read paths but adds invalidation complexity, so correctness must survive its loss.

## 11. Scalability and evaluation plan

Tool-catalog growth is addressed by indexing the registry and bounding candidate context. Execution throughput remains constrained by one sequential worker, especially during slow provider calls. Adding workers requires exclusive claims, ownership enforcement, and safe recovery before horizontal scaling is enabled.

Evaluation should compare the proposed retrieval approach against an all-eligible-tools baseline using the same model and task set. Catalog sizes should include approximately 50, 100, 500, and 1,000 tools, with distractors, similar operations, compound requests, missing fields, and restricted capabilities. Both approaches must use identical authorization constraints. If the baseline exceeds model context limits, record that as a feasibility limit.

Measure candidate recall, correct tool selection, valid arguments, verified task completion, clarification behavior, prompt tokens, cost, latency, retries, and replans. Fault-injection tests should cover provider success before local acknowledgment, result persistence before checkpointing, restart during a wait, duplicate input, revoked permissions, cancellation during a write, and optional-service outages.

Duplicate and unauthorized writes are critical failure indicators. Numerical accuracy or latency targets should be set after baseline measurements; no benchmark results are claimed in this draft.

## 12. Delivery plan and current evidence

Implementation should proceed from reviewed tool contracts and scoped retrieval to standalone RAG/System Search, structured planning, durable workflow execution, guarded writes, and measured optimization.

The repository README currently describes implemented registry/retrieval foundations, standalone RAG and System Search, sequential graph execution, and selected PayPal sandbox invoice writes with approval and saved-result replay. It also identifies generic provider write policies, automated reconciliation, and durable retry timers as pending. These are README-reported capabilities, not independently verified results from preparing this report.

Before submission, add measured evidence for tool selection and representative end-to-end requests. Record the model and embedding versions, catalog size, test dataset, environment, test outcomes, observed limitations, and which failure-recovery features were actually exercised. This will distinguish a justified architecture proposal from demonstrated implementation behavior.

## 13. Source documents

- [Assignment problem statement](</C:/Users/ilasy/Downloads/Datazoic Programming Task.txt>)
- [Supplied HLD](</C:/Users/ilasy/Downloads/Simplified_Agentic_System_HLD.md>)
- [Supplied LLD](</C:/Users/ilasy/Downloads/Simplified_Agentic_System_LLD.md>)
- [Repository README](../README.md), used only for the separately qualified implementation-progress note.

The supplied design documents contain platform reference links. This draft summarizes their proposed architecture and does not independently revalidate current platform APIs or provider endpoint contracts.
