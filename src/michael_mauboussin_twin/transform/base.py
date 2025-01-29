from typing import Generic, TypeVar
import abc
import torch
from colpali_engine import models as colpali_model
import tqdm
import stamina
import sentence_transformers
import loguru
from qdrant_client.http import models
import qdrant_client

from michael_mauboussin_twin.transform import settings, datamodels

T = TypeVar("T", bound="VectorStore")

logger = loguru.logger


@stamina.retry(on=Exception, attempts=3)
def upsert_to_qdrant(
    qdrant_client_: qdrant_client.QdrantClient,
    collection_name: str,
    points: list[models.PointStruct],
    start: int,
    end: int,
):
    try:
        qdrant_client_.upsert(
            collection_name=collection_name,
            points=points,
            wait=False,
        )
        logger.info(f"Upserted from {start} to {end} points to Qdrant")
    except Exception as e:
        logger.error(f"Error during upsert: {e}")
        return False
    return True


class VectorStore(abc.ABC, Generic[T]):

    def __init__(
        self,
        model: colpali_model.ColQwen2 | sentence_transformers.SentenceTransformer,
        db_settings: settings.DBSettings,
        qdrant_settings: settings.QdrantSettings,
        processor: colpali_model.ColQwen2Processor | None = None,
        batch_size: int = 10,
    ) -> None:
        self.model = model
        self.processor = processor
        self.db_settings = db_settings
        self.qdrant_settings = qdrant_settings
        self.qdrant_client = qdrant_client.QdrantClient(
            host=self.db_settings.QDRANT_DATABASE_HOST,
        )
        self.qdrant_client.recreate_collection(
            collection_name=self.qdrant_settings.collection_name,
            on_disk_payload=self.qdrant_settings.on_disk_payload,
            optimizers_config=self.qdrant_settings.optimizers_config,
            vectors_config=self.qdrant_settings.vector_params,
            quantization_config=self.qdrant_settings.scalar_params,
        )
        self.batch_size = batch_size

    @abc.abstractmethod
    def encode_docs(
        self, docs: list[datamodels.DocumentToVectorDB]
    ) -> list[torch.Tensor]:
        pass

    def batch_encode_and_upsert_docs(
        self,
        docs: list[datamodels.DocumentToVectorDB],
    ) -> list[models.PointStruct]:
        point_struct_models: list[models.PointStruct] = []
        with tqdm.tqdm(total=len(docs), desc="Indexing Progress") as pbar:
            for i in range(0, len(docs), self.batch_size):
                batch = docs[i : i + self.batch_size]
                vector_emb = self.encode_docs(batch)
                assert vector_emb.shape[0] == len(
                    batch
                ), f"Number of vectors {vector_emb.shape[0]} does not match number of documents {len(batch)}"
                current_batch: list[models.PointStruct] = []
                for vemb, b in zip(vector_emb, batch, strict=True):
                    current_batch.append(b.to_point(vemb))
                upsert_to_qdrant(
                    self.qdrant_client,
                    self.qdrant_settings.collection_name,
                    current_batch,
                    i,
                    i + self.batch_size,
                )
                point_struct_models.extend(current_batch)
                pbar.update(self.batch_size)
        return point_struct_models

    @classmethod
    def from_pretrained(
        cls: type[T],
        model_config: settings.VisionEmbeddingModel | settings.TextEmbeddingModel,
        config: settings.DBSettings,
    ) -> T:
        if isinstance(model_config, settings.VisionEmbeddingModel):
            model = colpali_model.ColQwen2.from_pretrained(
                model_config.name,
                torch_dtype=torch.bfloat16,
                device_map=config.RAG_MODEL_DEVICE,
            ).eval()
            processor = colpali_model.ColQwen2Processor.from_pretrained(
                model_config.name
            )

            return cls(model=model, processor=processor)

        elif isinstance(model_config, settings.TextEmbeddingModel):
            model = sentence_transformers.SentenceTransformer(
                model_config.name, trust_remote_code=True
            ).to(config.RAG_MODEL_DEVICE)

            return cls(model=model)
