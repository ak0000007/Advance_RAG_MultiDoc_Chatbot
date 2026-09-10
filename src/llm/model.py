"""
Qwen3 LLM loader.

Provides a LangChain-compatible ChatModel using
Qwen3-4B-Instruct-2507.

Features:

- Proper Qwen chat template
- Non-thinking mode
- LangChain AIMessage output
- LangChain bind_tools() support
- Qwen tool-call parsing
- 4-bit quantization
"""

import json
import re

from typing import (
    Any,
    List,
    Optional,
)

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from pydantic import ConfigDict

from langchain_core.language_models.chat_models import (
    BaseChatModel,
)

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from langchain_core.outputs import (
    ChatGeneration,
    ChatResult,
)

from langchain_core.utils.function_calling import (
    convert_to_openai_tool,
)


MODEL_ID = (
    "Qwen/Qwen3-4B-Instruct-2507"
)


class Qwen3ChatModel(BaseChatModel):
    """
    LangChain ChatModel wrapper around Qwen3.

    Supports:

    - normal chat generation
    - structured parser-based output
    - LangChain tool binding
    - Qwen3 tool-call parsing
    """

    tokenizer: Any
    model: Any

    max_new_tokens: int = 300

    # Qwen recommended settings for non-thinking mode
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20

    # Tools bound through bind_tools()
    bound_tools: List[dict] = []

    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )

    @property
    def _llm_type(self) -> str:
        return "qwen3-chat"

    @property
    def _identifying_params(self) -> dict:

        return {
            "model_id": MODEL_ID,
            "max_new_tokens": (
                self.max_new_tokens
            ),
            "temperature": (
                self.temperature
            ),
            "top_p": self.top_p,
            "top_k": self.top_k,
        }

    # =====================================================
    # LangChain Tool Binding
    # =====================================================

    def bind_tools(
        self,
        tools,
        *,
        tool_choice=None,
        **kwargs: Any,
    ):
        """
        Bind LangChain tools to the Qwen model.

        This makes tool schemas available to the model.

        The model itself does NOT execute tools.

        It can only request a tool call.

        Tool execution is handled later by LangGraph
        ToolNode.
        """

        formatted_tools = [
            convert_to_openai_tool(tool)
            for tool in tools
        ]

        # -------------------------------------------------
        # Create a copy with bound tools.
        #
        # We intentionally do not mutate the original
        # model because the same model may be used by
        # other parts of the RAG system.
        # -------------------------------------------------

        return self.model_copy(
            update={
                "bound_tools": formatted_tools
            }
        )

    # =====================================================
    # Message Conversion
    # =====================================================

    def _convert_messages(
        self,
        messages: List[BaseMessage],
    ):

        qwen_messages = []

        for message in messages:

            # -------------------------------------------------
            # System message
            # -------------------------------------------------

            if isinstance(
                message,
                SystemMessage,
            ):

                qwen_messages.append(
                    {
                        "role": "system",
                        "content": str(
                            message.content
                        ),
                    }
                )

            # -------------------------------------------------
            # Human message
            # -------------------------------------------------

            elif isinstance(
                message,
                HumanMessage,
            ):

                qwen_messages.append(
                    {
                        "role": "user",
                        "content": str(
                            message.content
                        ),
                    }
                )

            # -------------------------------------------------
            # AI message
            # -------------------------------------------------

            elif isinstance(
                message,
                AIMessage,
            ):

                assistant_message = {
                    "role": "assistant",
                    "content": (
                        str(message.content)
                        if message.content
                        else ""
                    ),
                }

                # ---------------------------------------------
                # Previous tool calls
                # ---------------------------------------------

                if message.tool_calls:

                    assistant_message[
                        "tool_calls"
                    ] = []

                    for tool_call in (
                        message.tool_calls
                    ):

                        assistant_message[
                            "tool_calls"
                        ].append(
                            {
                                "id": tool_call.get(
                                    "id",
                                    "call_"
                                    + tool_call.get(
                                        "name",
                                        "tool",
                                    ),
                                ),
                                "type": "function",
                                "function": {
                                    "name": tool_call[
                                        "name"
                                    ],
                                    "arguments": json.dumps(
                                        tool_call.get(
                                            "args",
                                            {},
                                        )
                                    ),
                                },
                            }
                        )

                qwen_messages.append(
                    assistant_message
                )

            # -------------------------------------------------
            # Tool message
            # -------------------------------------------------

            elif isinstance(
                message,
                ToolMessage,
            ):

                # Qwen3 treats tool responses as special
                # user-side tool response messages.
                qwen_messages.append(
                    {
                        "role": "tool",
                        "content": str(
                            message.content
                        ),
                        "tool_call_id": (
                            message.tool_call_id
                        ),
                        "name": (
                            message.name
                        ),
                    }
                )

            # -------------------------------------------------
            # Fallback
            # -------------------------------------------------

            else:

                qwen_messages.append(
                    {
                        "role": "user",
                        "content": str(
                            message.content
                        ),
                    }
                )

        return qwen_messages

    # =====================================================
    # Tool Call Parser
    # =====================================================

    def _parse_tool_calls(
        self,
        response_text: str,
    ):
        """
        Parse Qwen3 tool-call blocks.

        Expected form:

        <tool_call>
        {
            "name": "search_documents",
            "arguments": {
                "query": "..."
            }
        }
        </tool_call>
        """

        pattern = re.compile(
            r"<tool_call>\s*(.*?)\s*</tool_call>",
            re.DOTALL,
        )

        matches = pattern.findall(
            response_text
        )

        tool_calls = []

        for index, match in enumerate(
            matches
        ):

            try:

                payload = json.loads(
                    match
                )

                name = payload.get(
                    "name"
                )

                arguments = payload.get(
                    "arguments",
                    {},
                )

                # Some Qwen/tool templates can emit
                # arguments as a JSON string.
                if isinstance(
                    arguments,
                    str,
                ):

                    arguments = json.loads(
                        arguments
                    )

                tool_calls.append(
                    {
                        "name": name,
                        "args": arguments,
                        "id": (
                            f"call_{index}"
                        ),
                        "type": "tool_call",
                    }
                )

            except (
                json.JSONDecodeError,
                TypeError,
                ValueError,
            ):

                # Invalid tool calls are ignored.
                # The model output will remain available
                # as normal content.
                continue

        return tool_calls

    # =====================================================
    # Generation
    # =====================================================

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs: Any,
    ) -> ChatResult:

        # -------------------------------------------------
        # 1. Convert LangChain messages to Qwen format
        # -------------------------------------------------

        qwen_messages = (
            self._convert_messages(
                messages
            )
        )

        # -------------------------------------------------
        # 2. Apply Qwen chat template
        #
        # When tools are bound, pass their schemas to
        # the tokenizer so Qwen knows what functions exist.
        # -------------------------------------------------

        template_kwargs = {}

        if self.bound_tools:

            template_kwargs[
                "tools"
            ] = self.bound_tools

        inputs = (
            self.tokenizer.apply_chat_template(
                qwen_messages,
                tokenize=True,
                add_generation_prompt=True,

                # Qwen3-Instruct-2507 is non-thinking.
                return_dict=True,
                return_tensors="pt",

                **template_kwargs,
            )
        )

        # -------------------------------------------------
        # 3. Move model inputs to model device
        # -------------------------------------------------

        inputs = {
            key: value.to(
                self.model.device
            )
            for key, value in inputs.items()
            if hasattr(
                value,
                "to",
            )
        }

        # Number of tokens belonging to prompt
        input_token_count = (
            inputs["input_ids"]
            .shape[-1]
        )

        # -------------------------------------------------
        # 4. Generate response
        # -------------------------------------------------

        with torch.no_grad():

            generated_ids = (
                self.model.generate(
                    **inputs,

                    max_new_tokens=(
                        self.max_new_tokens
                    ),

                    do_sample=True,

                    temperature=(
                        self.temperature
                    ),

                    top_p=self.top_p,

                    top_k=self.top_k,

                    pad_token_id=(
                        self.tokenizer.eos_token_id
                    ),
                )
            )

        # -------------------------------------------------
        # 5. Remove original prompt
        # -------------------------------------------------

        generated_ids = (
            generated_ids[
                :,
                input_token_count:
            ]
        )

        # -------------------------------------------------
        # 6. Decode response
        # -------------------------------------------------

        response_text = (
            self.tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True,
            )
            .strip()
        )

        # -------------------------------------------------
        # 7. Parse tool calls
        # -------------------------------------------------

        tool_calls = []

        if self.bound_tools:

            tool_calls = (
                self._parse_tool_calls(
                    response_text
                )
            )

        # -------------------------------------------------
        # 8. Remove tool-call markup from visible content
        # -------------------------------------------------

        clean_content = re.sub(
            r"<tool_call>\s*.*?\s*</tool_call>",
            "",
            response_text,
            flags=re.DOTALL,
        ).strip()

        # -------------------------------------------------
        # 9. Create LangChain AIMessage
        # -------------------------------------------------

        ai_message = AIMessage(
            content=clean_content,
            tool_calls=tool_calls,
        )

        # -------------------------------------------------
        # 10. Return LangChain ChatResult
        # -------------------------------------------------

        generation = ChatGeneration(
            message=ai_message
        )

        return ChatResult(
            generations=[
                generation
            ]
        )


# =========================================================
# Model Loader
# =========================================================


def load_llm():
    """
    Load Qwen3-4B-Instruct-2507 using 4-bit quantization.

    Returns:
        Qwen3ChatModel
    """

    # -----------------------------------------------------
    # 1. 4-bit quantization
    # -----------------------------------------------------

    quantization_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=(
                torch.float16
            ),
            bnb_4bit_use_double_quant=True,
        )
    )

    # -----------------------------------------------------
    # 2. Tokenizer
    # -----------------------------------------------------

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_ID
        )
    )

    # -----------------------------------------------------
    # 3. Model
    # -----------------------------------------------------

    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=(
                quantization_config
            ),
            device_map="auto",
        )
    )

    # -----------------------------------------------------
    # 4. LangChain ChatModel
    # -----------------------------------------------------

    llm = Qwen3ChatModel(
        tokenizer=tokenizer,
        model=model,
    )

    return llm