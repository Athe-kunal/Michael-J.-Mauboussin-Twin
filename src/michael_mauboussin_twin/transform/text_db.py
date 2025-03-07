import torch
import sentence_transformers

from michael_mauboussin_twin.transform import base, settings, datamodels

CHUNK_SIZE = 100
CHUNK_OVERLAP = 25


class TextVectorDB(base.VectorDB):
    def __init__(
        self,
        model: sentence_transformers.SentenceTransformer,
        db_settings: settings.DBSettings,
        qdrant_settings: settings.QdrantSettings,
    ) -> None:
        super().__init__(db_settings, qdrant_settings)
        self.model = model

    def encode_docs(
        self, docs: list[datamodels.DocumentToVectorDB]
    ) -> list[torch.Tensor]:
        raise NotImplementedError
