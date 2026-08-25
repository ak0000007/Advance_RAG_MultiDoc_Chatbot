import uuid
from pathlib import Path

from pypdf import PdfReader

from src.models.document import Document
from src.ingestion.base import BaseLoader


class PDFLoader(BaseLoader):

    def load(self, file_path: str):

        reader = PdfReader(file_path)

        pages = []

        for page in reader.pages:

            text = page.extract_text()

            if text:
                pages.append(text)

        content = "\n\n".join(pages)

        path = Path(file_path)

        return Document(
            id=str(uuid.uuid4()),
            name=path.name,
            content=content,
            document_type="pdf",
            source=str(path),
            metadata={
                "extension": path.suffix.lower(),
                "file_path": str(path),
                "page_count": len(reader.pages)
            }
        )