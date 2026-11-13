"""
Database schema migration manager for Advance_RAG_Chatbot.
Replaces raw startup DDL with version-tracked idempotent migrations.
"""

import logging
from pathlib import Path
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent.parent / "migrations" / "sql"


@asynccontextmanager
async def _get_conn(pool_or_conn):
    if hasattr(pool_or_conn, "connection"):
        async with pool_or_conn.connection() as conn:
            yield conn
    else:
        yield pool_or_conn


async def run_migrations(pool) -> list[str]:
    """
    Check and execute pending SQL migrations in alphabetical order.
    Tracks applied versions in `schema_migrations` table.
    """
    applied_versions: list[str] = []

    async with _get_conn(pool) as conn:
        # Create migration tracking table if not exists
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version TEXT PRIMARY KEY,
                applied_at TIMESTAMPTZ DEFAULT now()
            );
        """)

        # Fetch already applied versions
        cursor = await conn.execute("SELECT version FROM schema_migrations;")
        rows = await cursor.fetchall()
        existing = {row[0] for row in rows}

        if not MIGRATIONS_DIR.exists():
            logger.warning("Migrations directory not found at %s", MIGRATIONS_DIR)
            return applied_versions

        migration_files = sorted(MIGRATIONS_DIR.glob("*.sql"))
        for sql_file in migration_files:
            version = sql_file.stem
            if version in existing:
                continue

            logger.info("Applying database migration: %s", version)
            sql_content = sql_file.read_text(encoding="utf-8")
            await conn.execute(sql_content)
            await conn.execute(
                "INSERT INTO schema_migrations (version) VALUES (%s);",
                (version,)
            )
            applied_versions.append(version)

    return applied_versions
