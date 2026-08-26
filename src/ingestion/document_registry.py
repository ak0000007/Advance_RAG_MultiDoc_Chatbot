import hashlib
import json
from pathlib import Path
from datetime import datetime


class DocumentRegistry:

    def __init__(
        self,
        registry_path="vector_store/document_registry.json"
    ):

        self.registry_path = Path(
            registry_path
        )

        self.registry_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self.registry = self._load()

    def _load(self):

        if not self.registry_path.exists():

            return {}

        with open(
            self.registry_path,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    def _save(self):

        with open(
            self.registry_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                self.registry,
                file,
                indent=4
            )

    def calculate_hash(
        self,
        file_path
    ):

        sha256 = hashlib.sha256()

        with open(
            file_path,
            "rb"
        ) as file:

            for chunk in iter(
                lambda: file.read(8192),
                b""
            ):

                sha256.update(chunk)

        return sha256.hexdigest()

    def get_status(
        self,
        file_path
    ):

        file_path = Path(file_path)

        file_hash = self.calculate_hash(
            file_path
        )

        file_key = str(
            file_path
        )

        if file_key not in self.registry:

            return "new", file_hash

        stored_hash = self.registry[
            file_key
        ]["file_hash"]

        if stored_hash == file_hash:

            return "unchanged", file_hash

        return "modified", file_hash

    def mark_indexed(
        self,
        file_path,
        file_hash
    ):

        file_key = str(
            file_path
        )

        self.registry[file_key] = {
            "file_hash": file_hash,
            "indexed_at": datetime.utcnow().isoformat()
        }

        self._save()