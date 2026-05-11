from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration ARIA chargée depuis .env ou les variables d'environnement."""

    MEDIAPIPE_MODEL_PATH: str = Field(
        default="pose_landmarker_full.task",
        description="Chemin vers le fichier modèle MediaPipe PoseLandmarker (.task).",
    )
    VLLM_BASE_URL: str = Field(
        default="http://localhost:8001/v1",
        description="URL de l'API vLLM (aria_llm sur le PC RTX).",
    )
    URL_BACK_LLM: str = Field(
        default="http://localhost:8001",
        description="URL racine du back LLM pour le health check.",
    )
    CHROMADB_PATH: str = Field(
        default="./data/corpus",
        description="Répertoire de la base vectorielle ChromaDB.",
    )
    SESSIONS_DIR: str = Field(
        default="./data/sessions",
        description="Répertoire de stockage des sessions.",
    )
    SESSION_RETENTION_DAYS: int = Field(
        default=30,
        description="Durée de rétention des vidéos en jours (RGPD).",
    )
    LOG_LEVEL: str = Field(default="INFO")
    BLUR_FACES: bool = Field(
        default=True,
        description="Active le floutage des visages (RGPD). Désactiver uniquement en dev local.",
    )

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
