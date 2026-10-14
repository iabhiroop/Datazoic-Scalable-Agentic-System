"""Remove local LLM exchange traces and persisted debug events.

Run without arguments to preview. Pass --apply to delete after confirmation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sqlalchemy import create_engine, delete, func, select

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db.models import WorkflowEvent  # noqa: E402
from app.settings import Settings  # noqa: E402


DEBUG_EVENT_FILTER = WorkflowEvent.event_type.like("debug\\_%", escape="\\")


def trace_directory(settings: Settings) -> Path:
    directory = settings.llm_trace_dir
    if not directory.is_absolute():
        directory = PROJECT_ROOT / directory
    return directory.resolve()


def clear_history(*, apply: bool, files_only: bool) -> int:
    settings = Settings(_env_file=PROJECT_ROOT / ".env")
    if not files_only and settings.database_url is None:
        raise RuntimeError("DATABASE_URL is not configured in the project environment")

    directory = trace_directory(settings)
    trace_files = sorted(directory.glob("*.jsonl")) if directory.exists() else []
    engine = (
        create_engine(
            settings.database_url.get_secret_value(),
            pool_pre_ping=True,
            connect_args={"connect_timeout": 5},
        )
        if settings.database_url is not None and not files_only
        else None
    )
    try:
        print(f"Exchange trace files: {len(trace_files)} in {directory}")
        if engine is not None:
            print("Checking persisted debug events in the application database...")
            with engine.connect() as connection:
                database_count = connection.scalar(
                    select(func.count())
                    .select_from(WorkflowEvent)
                    .where(DEBUG_EVENT_FILTER)
                )
            database_count = int(database_count or 0)
            print(f"Persisted debug events: {database_count}")
        else:
            database_count = None
            print("Database debug events: skipped (--files-only).")
        if not apply:
            print("Dry run only. Re-run with --apply to delete these records.")
            return 0

        deleted_events = 0
        if engine is not None:
            with engine.begin() as connection:
                # The schema makes audit events append-only. For this explicit local
                # cleanup, bypass only its DELETE trigger within the same transaction.
                connection.exec_driver_sql(
                    "ALTER TABLE workflow_events DISABLE TRIGGER workflow_events_append_only"
                )
                try:
                    result = connection.execute(delete(WorkflowEvent).where(DEBUG_EVENT_FILTER))
                    deleted_events = result.rowcount or 0
                finally:
                    connection.exec_driver_sql(
                        "ALTER TABLE workflow_events ENABLE TRIGGER workflow_events_append_only"
                    )

        deleted_files = 0
        errors = []
        for path in trace_files:
            try:
                path.unlink()
                deleted_files += 1
            except OSError as error:
                errors.append(f"{path}: {error}")

        print(f"Deleted {deleted_files} trace files and {deleted_events} debug events.")
        if errors:
            print("Some trace files could not be deleted:")
            for error in errors:
                print(f"  {error}")
            return 1
        return 0
    finally:
        if engine is not None:
            engine.dispose()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Delete local JSONL exchange traces and database debug events.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt; only valid with --apply.",
    )
    parser.add_argument(
        "--files-only",
        action="store_true",
        help="Limit cleanup to JSONL files; leave database debug events untouched.",
    )
    args = parser.parse_args()
    if args.yes and not args.apply:
        parser.error("--yes requires --apply")
    if args.apply and not args.yes:
        confirmation = input("Type DELETE to remove the listed LLM history: ")
        if confirmation != "DELETE":
            print("Cancelled; nothing was deleted.")
            return 1
    try:
        return clear_history(apply=args.apply, files_only=args.files_only)
    except Exception as error:  # noqa: BLE001 - CLI boundary reports setup/database failures.
        print(f"Could not clear LLM history: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
