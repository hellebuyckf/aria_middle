import chromadb
from chromadb.api import ClientAPI
from loguru import logger
from sentence_transformers import SentenceTransformer

from core.config import settings
from core.state import PubMedReference

MODEL_NAME = "intfloat/multilingual-e5-base"
COLLECTION_NAME = "aria_pubmed"
# Préfixe obligatoire pour les requêtes avec les modèles e5
E5_QUERY_PREFIX = "query: "
QUERY_SUFFIX = "running gait biomechanics rehabilitation"


class ChromaDBService:
    def __init__(self) -> None:
        self._client: ClientAPI = chromadb.PersistentClient(path=settings.CHROMADB_PATH)
        self._model = SentenceTransformer(MODEL_NAME)

    async def retrieve(
        self, pathologie: str, n_results: int = 5
    ) -> list[PubMedReference]:
        try:
            collection = self._client.get_collection(name=COLLECTION_NAME)
        except Exception:
            logger.warning("ChromaDB : collection '{}' introuvable", COLLECTION_NAME)
            return []

        if collection.count() == 0:
            logger.warning("ChromaDB : collection '{}' vide", COLLECTION_NAME)
            return []

        query = f"{E5_QUERY_PREFIX}{pathologie} {QUERY_SUFFIX}"
        embedding = self._model.encode([query], normalize_embeddings=True).tolist()

        results = collection.query(
            query_embeddings=embedding,
            n_results=min(n_results, collection.count()),
            include=["documents", "metadatas"],
        )

        refs: list[PubMedReference] = []
        documents = results.get("documents") or [[]]
        metadatas = results.get("metadatas") or [[]]

        for doc, meta in zip(documents[0], metadatas[0]):
            if not meta:
                continue
            refs.append(
                PubMedReference(
                    pmid=str(meta.get("pmid", "")),
                    titre=str(meta.get("titre", "")),
                    extrait=doc or "",
                )
            )

        return refs
