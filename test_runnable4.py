from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

class TestWrapper(Runnable):
    def __init__(self, model):
        self._model = model
    def __getattr__(self, name):
        return getattr(self._model, name)
    def invoke(self, *args, **kwargs):
        print("WRAPPER INVOKE CALLED")
        return self._model.invoke(*args, **kwargs)

model = ChatOpenAI(api_key="sk-fake")
wrapper = TestWrapper(model)

bound = wrapper.bind_tools([{"type": "function", "function": {"name": "test", "description": "test"}}])
try:
    bound.invoke("hello")
except Exception as e:
    pass
