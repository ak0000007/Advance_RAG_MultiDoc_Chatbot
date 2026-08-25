from pathlib import Path

from src.ingestion.factory import get_loader


SUPPORTED_EXTENSIONS = {

    ".pdf",
    ".docx",
    ".txt",
    ".csv",
    ".xlsx"
}


def load_documents(directory: str):

    directory = Path(directory)

    documents = []

    for file_path in directory.rglob("*"):

        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        print(
            f"Loading: {file_path}"
        )

        loader = get_loader(
            str(file_path)
        )

        documents.append(
            loader.load(
                str(file_path)
            )
        )

    return documents