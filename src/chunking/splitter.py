from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class DocumentChunker:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split_documents(
        self,
        documents: list[Document],
    ) -> list[Document]:
        if not documents:
            return []
        return self.splitter.split_documents(documents)

    def split_text(self, text: str) -> list[str]:
        if not text:
            return []
        return self.splitter.split_text(text)