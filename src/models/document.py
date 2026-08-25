from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class Document:

    id: str

    name: str

    content: str

    document_type: str

    source: Optional[str] = None

    metadata: Dict = field(default_factory=dict)