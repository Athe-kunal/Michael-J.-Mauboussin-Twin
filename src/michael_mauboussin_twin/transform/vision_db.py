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
        batch_images = self.processor.process_images(images).to(self.model.device)
        with torch.no_grad():
            image_embeddings = self.model(**batch_images)
        return image_embeddings

    def query_db(self, query: str, k: int = 10) -> list[datamodels.QueryResult]:
        with torch.no_grad():
            processed_query = self.processor.process_queries([query]).to(
                self.model.device
            )
            query_emb = self.model(**processed_query)
        multivector_query = query_emb[0].cpu().float().numpy().tolist()

        results = self.qdrant_client.query(
            collection_name=self.qdrant_settings.collection_name,
            query_vector=query_emb,
            limit=k,
        )
        return results
