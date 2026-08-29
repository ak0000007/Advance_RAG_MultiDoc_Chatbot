import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)

from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace


MODEL_ID = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"


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

    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=400,
        do_sample=False,
        return_full_text=False,
    )

    llm = HuggingFacePipeline(
        pipeline=text_pipeline
    )

    return ChatHuggingFace(llm=llm)