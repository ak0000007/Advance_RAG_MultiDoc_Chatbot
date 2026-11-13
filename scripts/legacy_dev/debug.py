import sys
import json
from src.api.dependencies import get_compiled_graph
graph = get_compiled_graph()
try:
    print("Invoking graph...")
    result = graph.invoke({"question": "hello", "history": []}, {"configurable": {"thread_id": "test_thread"}})
    print("Done:", result)
except Exception as e:
    import traceback
    traceback.print_exc()
