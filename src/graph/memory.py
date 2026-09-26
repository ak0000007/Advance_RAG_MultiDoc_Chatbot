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
            from langgraph.checkpoint.postgres import PostgresSaver
            # Note: For production with async, use AsyncPostgresSaver and connection pooling.
            # We use PostgresSaver with a sync connection for simplicity in the factory,
            # or require the caller to handle context managers if using async.
            import psycopg
            conn = psycopg.connect(postgres_url, autocommit=True)
            saver = PostgresSaver(conn)
            saver.setup()
            return saver
        except ImportError:
            print("WARNING: langgraph-checkpoint-postgres or psycopg not installed. Falling back to MemorySaver.")
    
    from langgraph.checkpoint.memory import MemorySaver
    return MemorySaver()
