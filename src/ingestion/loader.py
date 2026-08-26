from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
)


class MultiDocumentLoader:

    def __init__(self, data_directory="data"):

        self.data_directory = Path(
            data_directory
        )

    def load_file(self, file_path):

        file_path = Path(file_path)

        extension = file_path.suffix.lower()

        if extension == ".pdf":

            loader = PyPDFLoader(
                str(file_path)
            )

        elif extension == ".docx":

            loader = UnstructuredWordDocumentLoader(
                str(file_path)
            )

        elif extension == ".csv":

            loader = CSVLoader(
                str(file_path)
            )

        elif extension in [".xlsx", ".xls"]:

            loader = UnstructuredExcelLoader(
                str(file_path)
            )

        elif extension == ".txt":

            loader = TextLoader(
                str(file_path),
                encoding="utf-8"
            )

        else:

            raise ValueError(
                f"Unsupported file type: {extension}"
            )

        documents = loader.load()

        for document in documents:

            document.metadata["file_name"] = (
                file_path.name
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

                print(
                    f"Loaded: {file_path}"
                )

            except ValueError:

                print(
                    f"Skipped unsupported file: "
                    f"{file_path}"
                )

        return all_documents