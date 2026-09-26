import re

with open("src/graph/nodes.py", "r") as f:
    content = f.read()

# Patch Retrieval Grader
content = content.replace(
    "    grader_chain = (\n        grader_prompt\n        | llm\n        | parser\n    )",
    "    structured_llm = llm.with_structured_output(RetrievalGrade)\n    grader_chain = (\n        grader_prompt\n        | structured_llm\n    )"
)

# Patch Answer Grader
content = content.replace(
    "    grader_chain = (\n        grader_prompt\n        | llm\n        | parser\n    )",
    "    structured_llm = llm.with_structured_output(AnswerGrade)\n    grader_chain = (\n        grader_prompt\n        | structured_llm\n    )"
)

# We still need to pass format_instructions so the prompt template doesn't crash, 
# even though the LLM will ignore it because we are using structured output.
with open("src/graph/nodes.py", "w") as f:
    f.write(content)

print("Nodes patched.")
