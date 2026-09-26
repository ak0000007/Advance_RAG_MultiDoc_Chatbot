with open("src/llm/provider.py", "r") as f:
    content = f.read()

fallback_model_class = """
class _FallbackAwareModel(Runnable):
    \"\"\"
    A proxy that ensures chain modifications like with_structured_output
    are applied to BOTH the primary and fallback models, before 
    re-applying the fallback routing logic.
    \"\"\"
    def __init__(self, primary, fallback):
        self.primary = primary
        self.fallback = fallback
        self.runnable = primary.with_fallbacks(
            [fallback], 
            exceptions_to_handle=(ModelRateLimitError,)
        )

    def invoke(self, *args, **kwargs):
        return self.runnable.invoke(*args, **kwargs)

    async def ainvoke(self, *args, **kwargs):
        return await self.runnable.ainvoke(*args, **kwargs)
        
    def stream(self, *args, **kwargs):
        return self.runnable.stream(*args, **kwargs)

    def bind_tools(self, *args, **kwargs):
        return self.primary.bind_tools(*args, **kwargs).with_fallbacks(
            [self.fallback.bind_tools(*args, **kwargs)],
            exceptions_to_handle=(ModelRateLimitError,)
        )

    def with_structured_output(self, *args, **kwargs):
        return self.primary.with_structured_output(*args, **kwargs).with_fallbacks(
            [self.fallback.with_structured_output(*args, **kwargs)],
            exceptions_to_handle=(ModelRateLimitError,)
        )
        
    def __getattr__(self, name):
        return getattr(self.runnable, name)
"""

content = content.replace(
    "class _RateLimitAwareModel(Runnable):",
    fallback_model_class + "\nclass _RateLimitAwareModel(Runnable):"
)

# Update create_llm to use _FallbackAwareModel
content = content.replace(
    "    return primary_llm.with_fallbacks(\n        [fallback_llm],\n        exceptions_to_handle=(\n            ModelRateLimitError,\n        ),\n    )",
    "    return _FallbackAwareModel(primary=primary_llm, fallback=fallback_llm)"
)

with open("src/llm/provider.py", "w") as f:
    f.write(content)
