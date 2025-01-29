from michael_mauboussin_twin.transform import base, settings, datamodels
import torch


class VisionVectorStore(base.VectorStore):
    def __init__(
        self,
        db_settings: settings.DBSettings,
        qdrant_settings: settings.QdrantSettings,
    ) -> None:
        super().__init__(db_settings, qdrant_settings)

    def encode_docs(
        self, docs: list[datamodels.DocumentToVectorDB]
    ) -> list[torch.Tensor]:
        images = [doc.doc for doc in docs]
        batch_images = self.processor(images).to(self.model.device)
        with torch.no_grad():
            image_embeddings = self.model(**batch_images)
        return image_embeddings
