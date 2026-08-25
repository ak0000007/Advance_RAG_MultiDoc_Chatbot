import uuid
from pathlib import Path

import pandas as pd

from src.models.document import Document
from src.ingestion.base import BaseLoader


class CSVLoader(BaseLoader):

    def load(self, file_path: str):

        path = Path(file_path)

        df = pd.read_csv(path)

        rows = []

        for _, row in df.iterrows():

            row_text = "\n".join(
                f"{column}: {value}"
                for column, value in row.items()
            )

            rows.append(row_text)

        content = "\n\n".join(rows)

        return Document(
            id=str(uuid.uuid4()),
            name=path.name,
            content=content,
            document_type="csv",
            source=str(path),
            metadata={
                "extension": path.suffix.lower(),
                "file_path": str(path),
                "rows": len(df),
                "columns": list(df.columns)
            }
        )