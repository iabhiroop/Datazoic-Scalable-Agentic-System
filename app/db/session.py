"""Process-owned application/checkpoint pools; schema setup is an explicit CLI operation."""

import asyncio
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import Error as PsycopgError
from psycopg_pool import AsyncConnectionPool, PoolTimeout
from sqlalchemy import MetaData, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.settings import Settings

SCHEMA_REVISION = "0001_initial"


def event_loop_factory():
    """Psycopg async connections require SelectorEventLoop on Windows."""
    return asyncio.SelectorEventLoop() if sys.platform == "win32" else asyncio.new_event_loop()


class Base(DeclarativeBase):
    metadata = MetaData(
        naming_convention={
            "ix": "ix_%(table_name)s_%(column_0_name)s",
            "uq": "uq_%(table_name)s_%(column_0_N_name)s",
            "fk": "fk_%(table_name)s_%(column_0_N_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s",
        }
    )


class Database:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.engine = None
        self.sessions = None
        self.checkpoint_pool = None
        self.saver = None

    async def open(self, *, require_checkpoints: bool = True) -> None:
        if not self.settings.database_url:
            raise ValueError("DATABASE_URL is required")
        self.engine = create_async_engine(
            self.settings.database_url.get_secret_value(),
            pool_pre_ping=True,
            connect_args={"connect_timeout": 5},
        )
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        try:
            if require_checkpoints:
                if not self.settings.checkpoint_database_url:
                    raise ValueError("CHECKPOINT_DATABASE_URL is required")
                self.checkpoint_pool = AsyncConnectionPool(
                    self.settings.checkpoint_database_url.get_secret_value(),
                    open=False,
                    min_size=1,
                    max_size=5,
                    timeout=5,
                    kwargs={"autocommit": True, "prepare_threshold": 0},
                )
                await self.checkpoint_pool.open(wait=True, timeout=5)
                self.saver = AsyncPostgresSaver(self.checkpoint_pool)
            await self.validate_versions(require_checkpoints=require_checkpoints)
        except Exception:
            await self.close()
            raise

    async def close(self) -> None:
        if self.checkpoint_pool is not None:
            await self.checkpoint_pool.close()
        if self.engine is not None:
            await self.engine.dispose()
        self.engine = self.sessions = self.checkpoint_pool = self.saver = None

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        if self.sessions is None:
            raise RuntimeError("Database resources are not initialized")
        async with self.sessions() as session:
            yield session

    async def checkpointer(self) -> AsyncPostgresSaver:
        if self.saver is None:
            raise RuntimeError("Checkpoint pool is not initialized")
        return self.saver

    async def setup_checkpoints(self) -> None:
        saver = await self.checkpointer()
        await saver.setup()

    async def validate_versions(self, *, require_checkpoints: bool = True) -> None:
        if self.settings.embedding_dimension is None:
            raise ValueError("EMBEDDING_DIMENSION must match the migrated database")
        async with self.session() as session:
            revision = await session.scalar(text("SELECT version_num FROM alembic_version"))
            if revision != SCHEMA_REVISION:
                raise ValueError("Application migrations do not match this executable")
            dimensions = (
                await session.execute(
                    text("""
                SELECT c.relname, a.atttypmod
                FROM pg_attribute a JOIN pg_class c ON c.oid = a.attrelid
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = current_schema() AND a.attname = 'embedding'
                  AND c.relname IN ('tool_registry','knowledge_chunks','trajectory_summaries')
            """)
                )
            ).all()
            if len(dimensions) != 3 or any(
                dimension != self.settings.embedding_dimension for _, dimension in dimensions
            ):
                raise ValueError("Embedding dimension differs from the migrated schema")
        # No setup on API startup; administrators initialize checkpoint tables explicitly.
        if not require_checkpoints:
            return
        if self.checkpoint_pool is None:
            raise RuntimeError("Checkpoint pool is not initialized")
        async with self.checkpoint_pool.connection() as connection, connection.cursor() as cursor:
            await cursor.execute("""
                    SELECT to_regclass('checkpoints'), to_regclass('checkpoint_blobs'),
                           to_regclass('checkpoint_writes'), to_regclass('checkpoint_migrations')
                """)
            tables = await cursor.fetchone()
            if not tables or any(table is None for table in tables):
                raise ValueError("Run agent-admin setup-checkpoints before accepting work")
            await cursor.execute("SELECT max(v) FROM checkpoint_migrations")
            migration = await cursor.fetchone()
            if migration[0] != len(AsyncPostgresSaver.MIGRATIONS) - 1:
                raise ValueError("Checkpoint schema version does not match the installed saver")

    async def ready(self) -> bool:
        try:
            await self.validate_versions()
            return True
        except (ValueError, RuntimeError, SQLAlchemyError, PsycopgError, PoolTimeout):
            return False
