from michael_mauboussin_twin.transform import (
    vision_db,
    settings,
    datamodels as vision_datamodels,
)
import pathlib
import asyncio
from qdrant_client.http import models

vectors_config, quantization_config, optimizers_config = (
    settings.get_default_multi_vector_config(vector_size=1024)
)
vision_db_model = vision_db.VisionVectorStore.from_pretrained(
    settings.VisionEmbeddingModel(),
    settings.DBSettings(VISION_EMBEDDING_MODEL_PARAMS=settings.VisionEmbeddingModel()),
    qdrant_settings=settings.QdrantSettings(
        vector_params=vectors_config,
        scalar_params=quantization_config,
        optimizers_config=optimizers_config,
    ),
)


async def get_docs(
    vision_db_model: vision_db.VisionVectorStore,
) -> list[vision_datamodels.DocumentToVectorDB]:
    docs = await vision_db_model.read_from_pdfs(
        pathlib.Path(
            "michael_mauboussin_twin/feature/extract/data/extraction_metadata.json"
        )
    )
    return docs


async def add_docs(
    vision_db_model: vision_db.VisionVectorStore,
    docs: list[vision_datamodels.DocumentToVectorDB],
    batch_size: int = 5,
) -> None:
    await vision_db_model.batch_encode_and_upsert_docs(docs, batch_size)


async def main() -> None:
    docs = await get_docs(vision_db_model)
    await add_docs(vision_db_model, docs)


if __name__ == "__main__":
    asyncio.run(main())
