import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from langchain_core.language_models.llms import LLM
from typing import Any, List, Optional
from pydantic import Field

MODEL_ID = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"


class CustomDeepSeekLLM(LLM):
    model: Any = Field(default=None)
    tokenizer: Any = Field(default=None)
    max_new_tokens: int = 400

    class Config:
        arbitrary_types_allowed = True

    model_config = {
        "arbitrary_types_allowed": True
    }

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        messages = [
            {"role": "user", "content": prompt}
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )

        generated_tokens = outputs[
            0,
            inputs["input_ids"].shape[-1]:
        ]

        return self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )

    @property
    def _llm_type(self) -> str:
        return "custom_deepseek"


def load_llm():

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto",
    )

    return CustomDeepSeekLLM(model=model, tokenizer=tokenizer)