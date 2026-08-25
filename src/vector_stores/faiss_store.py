import faiss
import numpy as np


class FAISSVectorStore:

    def __init__(
        self,
        dimension
    ):

        self.dimension = dimension

        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.chunks = []

    def add(
        self,
        embeddings,
        chunks
    ):

        embeddings = np.asarray(
            embeddings,
            dtype="float32"
        )

        self.index.add(
            embeddings
        )

        self.chunks.extend(
            chunks
        )

    def search(
        self,
        query_embedding,
        top_k=5
    ):

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32"
        )

        scores, indices = (
            self.index.search(
                query_embedding,
                top_k
            )
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            if index == -1:
                continue

            results.append({

                "score":
                    float(score),

                "chunk":
                    self.chunks[index]
            })

        return results
    def save(self, directory):

        # Create directory if it doesn't exist
        os.makedirs(
            directory,
            exist_ok=True
        )

        # -----------------------------
        # Save FAISS vector index
        # -----------------------------

        index_path = os.path.join(
            directory,
            "faiss.index"
        )

        faiss.write_index(
            self.index,
            index_path
        )

        # -----------------------------
        # Save chunks
        # -----------------------------

        chunks_path = os.path.join(
            directory,
            "chunks.pkl"
        )

        with open(
            chunks_path,
            "wb"
        ) as f:

            pickle.dump(
                self.chunks,
                f
            )

    @classmethod
    def load(cls, directory):

        # -----------------------------
        # Load FAISS index
        # -----------------------------

        index_path = os.path.join(
            directory,
            "faiss.index"
        )

        index = faiss.read_index(
            index_path
        )

        # -----------------------------
        # Load chunks
        # -----------------------------

        chunks_path = os.path.join(
            directory,
            "chunks.pkl"
        )

        with open(
            chunks_path,
            "rb"
        ) as f:

            chunks = pickle.load(
                f
            )

        # -----------------------------
        # Reconstruct vector store
        # -----------------------------

        store = cls(
            dimension=index.d
        )

        store.index = index
        store.chunks = chunks

        return store