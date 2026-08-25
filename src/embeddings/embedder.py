from sentence_transformers import SentenceTransformer


class Embedder:

    def __init__(
        self,
        model_name=(
            "sentence-transformers/"
            "BAAI/bge-m3"
        )
    ):

        self.model = SentenceTransformer(
            model_name,
            device="cuda"
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