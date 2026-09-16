from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


DEFAULT_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"


class Qwen3ChatModel(BaseChatModel):
    """
    LangChain ChatModel wrapper around Qwen3.

    This class keeps the existing local Qwen implementation
    behind LangChain's standard BaseChatModel interface.
    """

    model_id: str
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    max_new_tokens: int = 300

    _tokenizer: Any = None
    _model: Any = None

    @property
    def _llm_type(self) -> str:
        return "qwen3-local"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        max_new_tokens: int = 300,
        **kwargs: Any,
    ):
        super().__init__(
            model_id=model_id,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )

        self._load_model()

    def _load_model(self) -> None:
        """Load tokenizer and quantized Qwen model."""

        print(f"Loading local Qwen model: {self.model_id}")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id
        )

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        print("Qwen model loaded successfully.")

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:

        # Convert LangChain messages into the format expected
        # by the Qwen chat template.
        chat_messages = []

        for message in messages:

            if message.type == "system":
                role = "system"

            elif message.type == "human":
                role = "user"

            elif message.type == "ai":
                role = "assistant"

            elif message.type == "tool":
                role = "tool"

            else:
                role = message.type

            chat_messages.append(
                {
                    "role": role,
                    "content": message.content,
                }
            )

        inputs = self._tokenizer.apply_chat_template(
            chat_messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self._model.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                do_sample=True,
            )

        input_length = inputs["input_ids"].shape[1]

        generated_tokens = outputs[0][input_length:]

        response_text = self._tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        message = AIMessage(content=response_text)

        generation = ChatGeneration(message=message)

        return ChatResult(
            generations=[generation]
        )


def load_llm(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    temperature: float = 0.7,
    max_new_tokens: int = 300,
) -> BaseChatModel:
    """
    Load the local Qwen3 model.

    This function is intentionally kept as the local-provider
    entry point for the LLM factory.
    """

    return Qwen3ChatModel(
        model_id=model_id,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )