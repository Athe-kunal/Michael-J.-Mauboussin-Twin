import pydantic
from PIL import Image
from typing import Any
import torch
from qdrant_client.http import models


class DocumentToVectorDB(pydantic.BaseModel):
    id: pydantic.UUID4 = pydantic.Field(default_factory=pydantic.UUID4)
    doc: str | Image.Image
    metadata: dict[str, Any]

    def to_point(self, vector: torch.Tensor) -> models.PointStruct:
        return models.PointStruct(
            id=self.id,
            vector=vector,
            payload=self.metadata,
        )
