from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download
import torch

class Embedder:

    def __init__(
        self,
        model_name=(
            "BAAI/bge-m3"
        )
    ):
        model_path = snapshot_download(repo_id=model_name)

        self.model = SentenceTransformer(
            model_path,
            device=None
        )

        if device is None:

            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = device

        self.model = SentenceTransformer(
            model_name,
            device=self.device
        )

    def embed_documents(
        self,
        texts
    ):

        return self.model.encode(

            texts,

            batch_size=8,

            convert_to_numpy=True,

            normalize_embeddings=True,

            show_progress_bar=True
        )

    def embed_query(
        self,
        query
    ):

        return self.model.encode(

            query,

            convert_to_numpy=True,

            normalize_embeddings=True
        )