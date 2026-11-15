"""
Memory / Checkpointing module for LangGraph.

SRP: Responsible only for instantiating the correct graph checkpointer.
OCP/DIP: Returns a BaseCheckpointSaver. If Postgres is configured, 
uses Postgres. Otherwise falls back to MemorySaver.
"""
from typing import Optional
from langchain_core.runnables import RunnableConfig

_SHARED_POOL = None


def get_postgres_pool(postgres_url: Optional[str] = None):
    """
    Return the shared AsyncConnectionPool singleton if postgres_url is configured.
    Uses open=False so it can be instantiated in sync context and opened in async lifespan.
    """
    global _SHARED_POOL
    if not postgres_url:
        return None
    if _SHARED_POOL is None:
        try:
            from psycopg_pool import AsyncConnectionPool

            _SHARED_POOL = AsyncConnectionPool(
                conninfo=postgres_url,
                min_size=1,
                max_size=20,
                open=False,
                kwargs={"autocommit": True},
            )
        except ImportError:
            return None
    return _SHARED_POOL


def resolve_checkpointer_backend(postgres_url: Optional[str] = None) -> tuple[str, Optional[str]]:
    """
    Expose checkpointer backend choice without opening pools.
    Returns (backend_type, reason_or_error).
    backend_type is 'postgres' or 'memory'.
    """
    if not postgres_url:
        return "memory", "POSTGRES_URL not set"
    try:
        import psycopg_pool
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        return "postgres", None
    except ImportError as exc:
        return "memory", f"missing package: {exc}"


def get_checkpointer(postgres_url: Optional[str] = None):
    """
    Factory for the checkpointer.
    
    If postgres_url is provided and drivers are installed, returns AsyncPostgresSaver.
    Otherwise, returns MemorySaver (RAM).
    """
    backend, reason = resolve_checkpointer_backend(postgres_url)
    if backend == "postgres":
        pool = get_postgres_pool(postgres_url)
        if pool is not None:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            return AsyncPostgresSaver(pool)
    
    from langgraph.checkpoint.memory import MemorySaver
    return MemorySaver()
