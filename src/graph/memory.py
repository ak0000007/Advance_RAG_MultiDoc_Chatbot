"""
Memory / Checkpointing module for LangGraph.

SRP: Responsible only for instantiating the correct graph checkpointer.
OCP/DIP: Returns a BaseCheckpointSaver. If Postgres is configured, 
uses Postgres. Otherwise falls back to MemorySaver.
"""
from typing import Optional
from langchain_core.runnables import RunnableConfig

def get_checkpointer(postgres_url: Optional[str] = None):
    """
    Factory for the checkpointer.
    
    If postgres_url is provided, it tries to initialize a PostgresSaver.
    Otherwise, it returns a MemorySaver (RAM).
    """
    if postgres_url:
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            from psycopg_pool import AsyncConnectionPool
            
            # Use AsyncConnectionPool for async IO and thread safety.
            # Setup (DB creation) is deferred to the caller (e.g., FastAPI lifespan)
            # because it is an async method and this factory remains sync to support @lru_cache.
            pool = AsyncConnectionPool(
                conninfo=postgres_url,
                max_size=20,
                kwargs={"autocommit": True}
            )
            saver = AsyncPostgresSaver(pool)
            return saver
        except ImportError:
            print("WARNING: langgraph-checkpoint-postgres or psycopg not installed. Falling back to MemorySaver.")
    
    from langgraph.checkpoint.memory import MemorySaver
    return MemorySaver()
