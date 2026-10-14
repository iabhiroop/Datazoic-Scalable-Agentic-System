"""Database-enforced boundaries independent of API checks."""

import asyncio
from uuid import uuid4

import psycopg
import pytest

from app.db.session import Database, event_loop_factory


def test_cross_tenant_conversation_reference_rejected(connection):
    tenant, other, conversation = uuid4(), uuid4(), uuid4()
    connection.execute(
        "INSERT INTO conversations (conversation_id,tenant_id,user_id,title) VALUES (%s,%s,'owner','test')",
        (conversation, tenant),
    )
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        connection.execute(
            """INSERT INTO workflow_runs
            (run_id,tenant_id,user_id,conversation_id,thread_id,submission_key,request_hash,
             request_payload,graph_version,prompt_version,status)
            VALUES (%s,%s,'owner',%s,%s,'test','hash','{}','v1','v1','queued')""",
            (uuid4(), other, conversation, str(uuid4())),
        )


def test_vector_dimension_rejected_by_database(connection):
    with pytest.raises(psycopg.errors.DataException):
        connection.execute(
            """INSERT INTO tool_registry
            (tenant_id,tool_id,schema_version,card,searchable_text,embedding)
            VALUES (%s,'bad','1','{}','test','[1,2]')""",
            (uuid4(),),
        )


def test_taxonomy_cannot_have_third_level(connection):
    tenant = uuid4()
    connection.execute(
        """INSERT INTO taxonomy_categories
        (tenant_id,category_id,label,description,revision) VALUES (%s,'root','root','root',1)""",
        (tenant,),
    )
    connection.execute(
        """INSERT INTO taxonomy_categories
        (tenant_id,category_id,label,description,revision,parent_id,depth)
        VALUES (%s,'child','child','child',1,'root',2)""",
        (tenant,),
    )
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        connection.execute(
            """INSERT INTO taxonomy_categories
            (tenant_id,category_id,label,description,revision,parent_id,depth)
            VALUES (%s,'grandchild','grandchild','grandchild',1,'child',2)""",
            (tenant,),
        )


def test_append_only_event_trigger(client, connection):
    response = client.post(
        "/runs",
        headers={"Authorization": "Bearer owner", "Idempotency-Key": "audit"},
        json={"message": "Hello"},
    )
    run_id = response.json()["run_id"]
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        connection.execute(
            "UPDATE workflow_events SET evidence = '{}' WHERE run_id = %s", (run_id,)
        )
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        connection.execute("DELETE FROM workflow_events WHERE run_id = %s", (run_id,))


def test_startup_rejects_dimension_mismatch_and_closes_pools(config):
    config.embedding_dimension += 1

    async def check():
        database = Database(config)
        with pytest.raises(ValueError, match="dimension"):
            await database.open()
        assert database.engine is None
        assert database.checkpoint_pool is None

    with asyncio.Runner(loop_factory=event_loop_factory) as runner:
        runner.run(check())
