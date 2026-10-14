`0001_initial.py` is a frozen migration for all 13 application tables, constraints
and indexes. It also enforces append-only workflow events through a trigger.
It does not import the current ORM mappings or create LangGraph tables.

Set `EMBEDDING_DIMENSION` before running migrations, or supply Alembic's
`-x embedding_dimension=<positive integer>` option. Each vector column is created
as `vector(D)`. Runtime startup checks the same dimension against PostgreSQL.

Initialize LangGraph checkpoint tables separately with `agent-admin setup-checkpoints`.
Do not cascade-delete pending operation evidence. The downgrade removes application
tables; it leaves the shared pgvector extension and checkpoint schema intact.
