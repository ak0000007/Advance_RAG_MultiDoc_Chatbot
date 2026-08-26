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

        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,

            model_kwargs={
                "device": device
            },

            encode_kwargs={
                "normalize_embeddings": True
            }
        )

    def get_embeddings(self):

        return self.embeddings