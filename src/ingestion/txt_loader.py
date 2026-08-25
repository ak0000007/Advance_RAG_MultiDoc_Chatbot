import uuid
from pathlib import Path

from src.models.document import Document
from src.ingestion.base import BaseLoader


class TXTLoader(BaseLoader):

    def load(self, file_path: str):

        path = Path(file_path)

        with open(path, "r", encoding="utf-8") as file:

            content = file.read()

        return Document(
            id=str(uuid.uuid4()),
            name=path.name,
            content=content,
            document_type="txt",
            source=str(path),
            metadata={
                "extension": path.suffix.lower(),
                "file_path": str(path)
            }
        )