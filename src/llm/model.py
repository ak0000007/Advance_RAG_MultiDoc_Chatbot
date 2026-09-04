"""
Qwen3 LLM loader.

Provides a LangChain-compatible chat model using
Qwen3-4B-Instruct-2507.

Features:
- Proper Qwen chat template
- Non-thinking mode for normal RAG
- Sampling-based generation
- LangChain AIMessage output
"""

from typing import Any, List, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatResult,
)


MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"


class Qwen3ChatModel(BaseChatModel):
    """
    LangChain ChatModel wrapper around Qwen3.
    """

    tokenizer: Any
    model: Any

    max_new_tokens: int = 300

    # Qwen recommended settings for non-thinking mode
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20

    @property
    def _llm_type(self) -> str:
        return "qwen3-chat"

    @property
    def _identifying_params(self) -> dict:
        return {
            "model_id": MODEL_ID,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs: Any,
    ) -> ChatResult:

        # ---------------------------------------------------------
        # 1. Convert LangChain messages to Qwen message format
        # ---------------------------------------------------------

        qwen_messages = []

        for message in messages:

            if isinstance(message, SystemMessage):
                role = "system"

            elif isinstance(message, HumanMessage):
                role = "user"

            elif isinstance(message, AIMessage):
                role = "assistant"

            else:
                role = "user"

            qwen_messages.append(
                {
                    "role": role,
                    "content": message.content,
                }
            )

        # ---------------------------------------------------------
        # 2. Apply Qwen chat template
        #
        # return_dict=True is IMPORTANT.
        #
        # It gives us:
        #   input_ids
        #   attention_mask
        #
        # which can safely be passed to model.generate().
        # ---------------------------------------------------------

        inputs = self.tokenizer.apply_chat_template(
            qwen_messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
        )

        # ---------------------------------------------------------
        # 3. Move model inputs to the model device
        # ---------------------------------------------------------

        inputs = {
            key: value.to(self.model.device)
            for key, value in inputs.items()
            if hasattr(value, "to")
        }

        # Number of tokens belonging to the prompt.
        input_token_count = inputs["input_ids"].shape[-1]

        # ---------------------------------------------------------
        # 4. Generate response
        # ---------------------------------------------------------

        with torch.no_grad():

            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,

                # IMPORTANT:
                # Do not use greedy decoding with Qwen3.
                do_sample=True,

                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,

                pad_token_id=self.tokenizer.eos_token_id,
            )

        # ---------------------------------------------------------
        # 5. Remove the original prompt tokens
        #
        # We only want the newly generated answer.
        # ---------------------------------------------------------

        generated_ids = generated_ids[
            :,
            input_token_count:
        ]

        # ---------------------------------------------------------
        # 6. Decode generated tokens
        # ---------------------------------------------------------

        response_text = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
        ).strip()

        # ---------------------------------------------------------
        # 7. Return proper LangChain AIMessage
        # ---------------------------------------------------------

        generation = ChatGeneration(
            message=AIMessage(
                content=response_text
            )
        )

        return ChatResult(
            generations=[generation]
        )


def load_llm():
    """
    Load Qwen3-4B-Instruct-2507 using 4-bit quantization.
    """

    # ---------------------------------------------------------
    # 4-bit quantization
    # ---------------------------------------------------------

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # ---------------------------------------------------------
    # Tokenizer
    # ---------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto",
    )

    # ---------------------------------------------------------
    # Create LangChain ChatModel
    # ---------------------------------------------------------

    llm = Qwen3ChatModel(
        tokenizer=tokenizer,
        model=model,
    )

    return llm