import torch

from langchain_huggingface import (
    HuggingFaceEmbeddings
)


class BGEEmbeddings:

    def __init__(
        self,
        model_name="BAAI/bge-m3",
        device=None
    ):

        if device is None:

            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = device
        self.model_name = model_name

        model_kwargs = {
            "device": device,
        }
        # Avoid hanging on DNS/network check if offline or in sandbox
        try:
            from huggingface_hub import try_to_load_from_cache
            # if cached locally, enforce local_files_only
            model_kwargs["local_files_only"] = True
        except Exception:
            pass

        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs={
                    "normalize_embeddings": True
                }
            )
        except Exception:
            # Fall back without local_files_only if not cached yet
            model_kwargs.pop("local_files_only", None)
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs={
                    "normalize_embeddings": True
                }
            )

    def get_embeddings(self):

        return self.embeddings