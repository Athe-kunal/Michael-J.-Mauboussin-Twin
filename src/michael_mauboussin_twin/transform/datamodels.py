import pydantic
from PIL import Image
from typing import Any
import torch
import uuid
from qdrant_client.http import models


class Metadata(pydantic.BaseModel):
    title: str
    author: list[str]
    date: str
    url: str
    base64_image: str


class DocumentToVectorDB(pydantic.BaseModel):
    id: pydantic.UUID4 = pydantic.Field(default_factory=uuid.uuid4)
    doc: str | Image.Image
    metadata: Metadata

    def to_point(self, vector: torch.Tensor) -> models.PointStruct:
        return models.PointStruct(
            id=str(self.id),
            vector=vector.tolist(),
            payload=self.metadata.model_dump(),
        )

    class Config:
        arbitrary_types_allowed = True


class QueryResult(pydantic.BaseModel):
    query: str
    metadata: Metadata

    class Config:
        arbitrary_types_allowed = True
