from langchain_core.runnables import Runnable
from langchain_core.language_models.chat_models import BaseChatModel
from src.llm.provider import _RateLimitAwareModel

# See if changing it to inherit from Runnable fixes the pipe
class TestWrapper(Runnable):
    def __init__(self, model):
        self._model = model
    def invoke(self, *args, **kwargs):
        pass

t = TestWrapper(None)

from langchain_core.prompts import PromptTemplate
p = PromptTemplate.from_template("Hello {name}")

chain = p | t
print("Success!")
