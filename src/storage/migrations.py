"""Small checksum-bound SQL migration runner."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from psycopg import Connection


MIGRATION_NAME = re.compile(r"^[0-9]{4}_[a-z0-9_]+[.]sql$")
MIGRATION_LOCK_NAME = "longitudinal_memory_schema_migrations"


class MigrationError(RuntimeError):
    """Reject missing, malformed, or changed migrations."""


@dataclass(frozen=True)
class Migration:
    version: str
    checksum: str
    sql: str


def load_migrations(directory: str | Path) -> tuple[Migration, ...]:
    path = Path(directory)
    try:
        entries = sorted(item for item in path.iterdir() if item.is_file())
    except OSError as error:
        raise MigrationError(f"could not read migration directory: {error}") from error
    if not entries:
        raise MigrationError("migration directory is empty")
    migrations: list[Migration] = []
    for item in entries:
        if not MIGRATION_NAME.fullmatch(item.name):
            raise MigrationError(f"invalid migration filename: {item.name}")
        try:
            payload = item.read_bytes()
            sql = payload.decode("utf-8")
        except (OSError, UnicodeDecodeError) as error:
            raise MigrationError(f"could not read migration {item.name}: {error}") from error
        if not sql.strip():
            raise MigrationError(f"migration {item.name} is empty")
        migrations.append(
            Migration(
                version=item.name,
                checksum=hashlib.sha256(payload).hexdigest(),
                sql=sql,
            )
        )
    versions = [item.version for item in migrations]
    if len(versions) != len(set(versions)):
        raise MigrationError("migration versions must be unique")
    return tuple(migrations)


def apply_migrations(
    connection: "Connection[object]",
    directory: str | Path = "migrations",
) -> tuple[str, ...]:
    """Apply new migrations once under a transaction-scoped advisory lock."""

    migrations = load_migrations(directory)
    applied: list[str] = []
    with connection.transaction():
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version text PRIMARY KEY,
                checksum text NOT NULL CHECK (checksum ~ '^[0-9a-f]{64}$'),
                applied_at timestamptz NOT NULL DEFAULT transaction_timestamp()
            )
            """
        )
        connection.execute(
            "SELECT pg_advisory_xact_lock(hashtext(%s))",
            (MIGRATION_LOCK_NAME,),
        )
        for migration in migrations:
            row = connection.execute(
                "SELECT checksum FROM schema_migrations WHERE version = %s",
                (migration.version,),
            ).fetchone()
            if row is not None:
                if row[0] != migration.checksum:
                    raise MigrationError(
                        f"migration checksum changed: {migration.version}"
                    )
                continue
            connection.execute(migration.sql)
            connection.execute(
                "INSERT INTO schema_migrations (version, checksum) VALUES (%s, %s)",
                (migration.version, migration.checksum),
            )
            applied.append(migration.version)
    return tuple(applied)
