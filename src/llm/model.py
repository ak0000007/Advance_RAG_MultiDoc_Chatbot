"""
Qwen3 LLM loader.

Provides a LangChain-compatible chat model using
Qwen3-4B-Instruct-2507.

Important:
- Uses Qwen's chat template correctly.
- Disables thinking mode for normal RAG answers.
- Uses sampling instead of greedy decoding.
- Returns clean AIMessage responses.
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
from langchain_core.outputs import ChatGeneration, ChatResult


MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"


class Qwen3ChatModel(BaseChatModel):
    """
    LangChain ChatModel wrapper around Qwen3.

    This explicitly applies Qwen's chat template so that
    system/user/assistant roles are preserved correctly.
    """

    tokenizer: Any
    model: Any

    max_new_tokens: int = 300
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

        # Convert LangChain messages into Qwen chat format.
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
        # IMPORTANT:
        # Qwen3 chat template is applied explicitly.
        #
        # enable_thinking=False prevents Qwen from entering
        # reasoning/thinking mode for our normal RAG responses.
        # ---------------------------------------------------------

        inputs = self.tokenizer.apply_chat_template(
            qwen_messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
        )

        # Move tensors to the same device as the model.
        inputs = inputs.to(self.model.device)

        input_token_count = inputs.shape[-1]

        # ---------------------------------------------------------
        # Generate the answer.
        #
        # We intentionally use sampling rather than greedy decoding.
        # Qwen3 documentation recommends sampling because greedy
        # decoding can cause repetition/endless-generation problems.
        # ---------------------------------------------------------

        with torch.no_grad():

            generated_ids = self.model.generate(
                inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Only keep newly generated tokens.
        generated_ids = generated_ids[
            :, input_token_count:
        ]

        # Convert tokens back to text.
        response_text = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
        ).strip()

        generation = ChatGeneration(
            message=AIMessage(content=response_text)
        )

        return ChatResult(
            generations=[generation]
        )


def load_llm():
    """
    Load Qwen3-4B-Instruct-2507 in 4-bit quantization.

    Returns:
        Qwen3ChatModel
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
    # LangChain ChatModel
    # ---------------------------------------------------------

    llm = Qwen3ChatModel(
        tokenizer=tokenizer,
        model=model,
    )

    return llm