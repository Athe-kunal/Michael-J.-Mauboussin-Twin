import pydantic
from typing import Optional


class ExtractData(pydantic.BaseModel):
    url: str
    title: str
    author: Optional[list[str]] = ["Michael Mauboussin"]
    date: Optional[str] = None
    pdf_path: str
