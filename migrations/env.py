"""Application migrations only. LangGraph checkpoint setup is separate."""

from alembic import context
from pgvector.sqlalchemy import Vector
from sqlalchemy import create_engine

from app.db import models  # noqa: F401 -- register mappings
from app.db.session import Base
from app.settings import Settings

config = context.config
settings = Settings()
dimension = context.get_x_argument(as_dictionary=True).get("embedding_dimension")
if dimension is not None:
    dimension = int(dimension)
else:
    dimension = settings.embedding_dimension
if dimension is None or dimension <= 0:
    raise ValueError("Set EMBEDDING_DIMENSION or pass -x embedding_dimension=<positive integer>")
config.attributes["embedding_dimension"] = dimension


def compare_type(context, inspected_column, metadata_column, inspected_type, metadata_type):
    if isinstance(inspected_type, Vector) and isinstance(metadata_type, Vector):
        return inspected_type.dim != dimension
    return None


def include_object(object, name, type_, reflected, compare_to):
    return not (
        type_ == "table"
        and name
        in {
            "checkpoints",
            "checkpoint_blobs",
            "checkpoint_writes",
            "checkpoint_migrations",
        }
    )


def configure(connection=None):
    context.configure(
        connection=connection,
        dialect_name="postgresql",
        target_metadata=Base.metadata,
        literal_binds=connection is None,
        compare_type=compare_type,
        include_object=include_object,
    )
    with context.begin_transaction():
        context.run_migrations()


def online():
    if not settings.database_url:
        raise ValueError("DATABASE_URL is required for online migrations")
    engine = create_engine(
        settings.database_url.get_secret_value(), connect_args={"connect_timeout": 5}
    )
    try:
        with engine.connect() as connection:
            configure(connection)
    finally:
        engine.dispose()


if context.is_offline_mode():
    configure()
else:
    online()
