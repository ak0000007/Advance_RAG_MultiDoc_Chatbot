from langchain_community.vectorstores import FAISS


class VectorStore:

    def __init__(
        self,
        embedding_model
    ):

        self.embedding_model = (
            embedding_model
        )

        self.vector_store = None

    def create_from_documents(
        self,
        documents
    ):

        self.vector_store = (
            FAISS.from_documents(
                documents,
                self.embedding_model
            )
        )

        return self.vector_store

    def save(self, directory):

        if self.vector_store is None:

            raise ValueError(
                "Vector store has not been created."
            )

        self.vector_store.save_local(
            directory
        )

    def load(self, directory):

        self.vector_store = (
            FAISS.load_local(
                directory,
                self.embedding_model,
                allow_dangerous_deserialization=True
            )
        )

        return self.vector_store

    def as_retriever(
        self,
        top_k=5
    ):

        if self.vector_store is None:

            raise ValueError(
                "Vector store has not been loaded "
                "or created."
            )

        return self.vector_store.as_retriever(
            search_kwargs={
                "k": top_k
            }
        )