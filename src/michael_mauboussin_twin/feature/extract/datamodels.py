from typing import Optional

import pydantic


class ExtractData(pydantic.BaseModel):
    url: str
    title: str
    author: Optional[list[str]] = ["Michael Mauboussin"]
    date: Optional[str] = None
    pdf_path: str
