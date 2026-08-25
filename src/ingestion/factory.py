from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.docx_loader import DOCXLoader
from src.ingestion.txt_loader import TXTLoader
from src.ingestion.csv_loader import CSVLoader
from src.ingestion.excel_loader import ExcelLoader


LOADER_MAP = {

    ".pdf": PDFLoader,

    ".docx": DOCXLoader,

    ".txt": TXTLoader,

    ".csv": CSVLoader,

    ".xlsx": ExcelLoader,
}


def get_loader(file_path: str):

    extension = "." + file_path.split(".")[-1].lower()

    loader_class = LOADER_MAP.get(
        extension
    )

    if loader_class is None:

        raise ValueError(
            f"Unsupported file type: {extension}"
        )

    return loader_class()