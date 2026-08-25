import uuid

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from src.models.chunk import Chunk


class TextChunker:

    def __init__(
        self,
        chunk_size=1000,
        chunk_overlap=150
    ):

        self.splitter = (
            RecursiveCharacterTextSplitter(

                chunk_size=chunk_size,

                chunk_overlap=chunk_overlap,

                separators=[
                    "\n\n",
                    "\n",
                    ". ",
                    " ",
                    ""
                ]
            )
        )

    def chunk_document(
        self,
        document
    ):

        texts = self.splitter.split_text(
            document.content
        )

        chunks = []

        for index, text in enumerate(texts):

            metadata = dict(
                document.metadata
            )

            metadata.update({

                "document_name":
                    document.name,

                "document_type":
                    document.document_type,

                "chunk_index":
                    index
            })

            chunk = Chunk(

                id=str(uuid.uuid4()),

                document_id=document.id,

                content=text,

                metadata=metadata
            )

            chunks.append(chunk)

        return chunks
    