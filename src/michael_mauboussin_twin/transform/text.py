from michael_mauboussin_twin.transform import base, settings


class TextVectorDB(base.VectorDB):
    def __init__(
        self, db_settings: settings.DBSettings, qdrant_settings: settings.QdrantSettings
    ) -> None:
        super().__init__(db_settings, qdrant_settings)
