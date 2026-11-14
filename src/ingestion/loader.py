import logging
from pathlib import Path
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class MultiDocumentLoader:

    def __init__(self, data_directory="data", allowed_directories: list[str | Path] | None = None):
        self.data_directory = Path(data_directory)
        self.allowed_directories = [
            Path(d).resolve() for d in (allowed_directories or [])
        ]
        # Always allow the configured data_directory
        self.allowed_directories.append(self.data_directory.resolve())

    def _validate_path(self, file_path: str | Path) -> Path:
        resolved = Path(file_path).resolve()
        
        # Verify resolved path is strictly within one of the allowed directories
        is_safe = any(
            allowed_dir == resolved or allowed_dir in resolved.parents
            for allowed_dir in self.allowed_directories
        )
        if not is_safe:
            raise PermissionError(
                f"Path traversal detected: '{file_path}' resolves outside allowed directories: "
                f"{[str(d) for d in self.allowed_directories]}"
            )
        if not resolved.exists() or not resolved.is_file():
            raise FileNotFoundError(f"File not found: {resolved}")
        return resolved

    def load_file(self, file_path):
        resolved_path = self._validate_path(file_path)
        extension = resolved_path.suffix.lower()

        if extension == ".txt":
            content = resolved_path.read_text(encoding="utf-8")
            documents = [Document(page_content=content, metadata={"source": str(resolved_path)})]
        elif extension == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(str(resolved_path))
            documents = loader.load()
        elif extension == ".docx":
            from langchain_community.document_loaders import UnstructuredWordDocumentLoader
            loader = UnstructuredWordDocumentLoader(str(resolved_path))
            documents = loader.load()
        elif extension == ".csv":
            from langchain_community.document_loaders import CSVLoader
            loader = CSVLoader(str(resolved_path))
            documents = loader.load()
        elif extension in [".xlsx", ".xls"]:
            from langchain_community.document_loaders import UnstructuredExcelLoader
            loader = UnstructuredExcelLoader(str(resolved_path))
            documents = loader.load()
        else:
            raise ValueError(
                f"Unsupported file type: {extension}"
            )

        for document in documents:
            document.metadata["file_name"] = (
                resolved_path.name
            )
            document.metadata["file_type"] = (
                extension
            )

        return documents

    def load_directory(self):

        all_documents = []

        for file_path in self.data_directory.rglob("*"):

            if not file_path.is_file():
                continue

            try:

                documents = self.load_file(
                    file_path
                )

                all_documents.extend(
                    documents
                )

                logger.info("Loaded: %s", file_path)

            except ValueError:

                logger.warning("Skipped unsupported file: %s", file_path)

        return all_documents