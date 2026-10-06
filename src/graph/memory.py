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


def get_checkpointer(postgres_url: Optional[str] = None):
    """
    Factory for the checkpointer.
    
    If postgres_url is provided, it tries to initialize a PostgresSaver.
    Otherwise, it returns a MemorySaver (RAM).
    """
    if postgres_url:
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

            pool = get_postgres_pool(postgres_url)
            if pool is not None:
                return AsyncPostgresSaver(pool)
        except ImportError:
            print("WARNING: langgraph-checkpoint-postgres or psycopg not installed. Falling back to MemorySaver.")
    
    from langgraph.checkpoint.memory import MemorySaver
    return MemorySaver()
