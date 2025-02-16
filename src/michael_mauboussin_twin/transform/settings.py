import pydantic_settings
import pydantic
from typing import NamedTuple
from qdrant_client.http import models
import os


class VisionEmbeddingModel(NamedTuple):
    name: str = "vidore/colqwen2-v1.0"


class TextEmbeddingModel(NamedTuple):
    name: str = "billatsectorflow/stella_en_400M_v5"
    query_prompt_name: str | None = "s2p_query"


def get_default_multi_vector_config(
    vector_size: int,
    indexing_threshold: int = 100,
) -> tuple[models.VectorParams, models.ScalarQuantization, models.OptimizersConfigDiff]:

    optimizers_config = models.OptimizersConfigDiff(
        indexing_threshold=indexing_threshold
    )

    vectors_config = models.VectorParams(
        size=vector_size,
        distance=models.Distance.COSINE,
        multivector_config=models.MultiVectorConfig(
            comparator=models.MultiVectorComparator.MAX_SIM
        ),
    )
    quantization_config = models.ScalarQuantization(
        scalar=models.ScalarQuantizationConfig(
            type=models.ScalarType.INT8,
            quantile=0.99,
            always_ram=True,
        ),
    )
    return vectors_config, quantization_config, optimizers_config


def get_default_single_vector_config(
    vector_size: int,
    indexing_threshold: int = 100,
) -> tuple[models.VectorParams, models.ScalarQuantization, models.OptimizersConfigDiff]:

    optimizers_config = models.OptimizersConfigDiff(
        indexing_threshold=indexing_threshold
    )

    vectors_config = models.VectorParams(
        size=vector_size,
        distance=models.Distance.COSINE,
    )
    quantization_config = models.ScalarQuantization(
        scalar=models.ScalarQuantizationConfig(
            type=models.ScalarType.INT8,
            quantile=0.99,
            always_ram=True,
        ),
    )
    return vectors_config, quantization_config, optimizers_config


class QdrantSettings(pydantic.BaseModel):
    collection_name: str = "mauboussinTwin"
    on_disk_payload: bool = True
    optimizers_config: models.OptimizersConfigDiff
    vector_params: models.VectorParams
    scalar_params: models.ScalarQuantization


class DBSettings(pydantic_settings.BaseSettings):
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    VISION_EMBEDDING_MODEL_PARAMS: VisionEmbeddingModel | None = None
    TEXT_EMBEDDING_MODEL_PARAMS: TextEmbeddingModel | None = None

    RAG_MODEL_DEVICE: str = f"cuda:{os.environ['CUDA_VISIBLE_DEVICES'] or '0'}"

    USE_QDRANT_CLOUD: bool = False
    QDRANT_DATABASE_PATH: str = os.getcwd() + "/mj-db"
    QDRANT_DATABASE_HOST: str = os.environ.get("QDRANT_DATABASE_HOST", "localhost")
    QDRANT_DATABASE_PORT: int = int(os.environ.get("QDRANT_DATABASE_PORT", "6333"))
    QDRANT_CLOUD_URL: str = os.environ.get("QDRANT_CLOUD_URL", "http://localhost:6333")
    QDRANT_APIKEY: str | None = os.environ.get("QDRANT_APIKEY")

    @pydantic.model_validator(mode="after")
    def validate_embedding_models(cls, values):
        vision_model = values.VISION_EMBEDDING_MODEL_PARAMS
        text_model = values.TEXT_EMBEDDING_MODEL_PARAMS

        if vision_model is None and text_model is None:
            raise ValueError("Either vision or text embedding model must be specified")

        if vision_model is not None and text_model is not None:
            raise ValueError(
                "Only one embedding model (vision or text) can be specified at a time"
            )

        return values
