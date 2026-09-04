import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)

from langchain_huggingface import HuggingFacePipeline


MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"


def load_llm():

    # --------------------------------------------------
    # 1. Quantization configuration
    # --------------------------------------------------

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # --------------------------------------------------
    # 2. Tokenizer
    # --------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID
    )

    # --------------------------------------------------
    # 3. Model
    # --------------------------------------------------

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        device_map="auto",
    )

    # --------------------------------------------------
    # 4. Hugging Face generation pipeline
    # --------------------------------------------------

    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=300,
        do_sample=False,
        return_full_text=False
    )

    # --------------------------------------------------
    # 5. Convert HF pipeline into LangChain LLM
    # --------------------------------------------------

    llm = HuggingFacePipeline(
        pipeline=text_pipeline
    )

    return llm