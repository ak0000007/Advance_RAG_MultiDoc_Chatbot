import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)

from langchain_huggingface import HuggingFacePipeline


MODEL_ID = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"


def load_llm():
    """
    Load DeepSeek-R1-0528-Qwen3-8B and expose it
    as a LangChain Runnable.
    """

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=False,
    )

    llm = HuggingFacePipeline(
        pipeline=text_pipeline
    )

    return llm