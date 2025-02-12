from michael_mauboussin_twin.transform import base, settings, datamodels
import torch
from colpali_engine import models as colpali_model


class VisionVectorStore(base.VectorStore):
    def __init__(
        self,
        model: colpali_model.ColQwen2,
        processor: colpali_model.ColQwen2Processor,
        db_settings: settings.DBSettings,
        qdrant_settings: settings.QdrantSettings,
    ) -> None:
        super().__init__(model, db_settings, qdrant_settings, processor)

    def encode_docs(
        self, docs: list[datamodels.DocumentToVectorDB]
    ) -> list[torch.Tensor]:
        images = [doc.doc for doc in docs]
        batch_images = self.processor(images).to(self.model.device)
        with torch.no_grad():
            image_embeddings = self.model(**batch_images)
        return image_embeddings
