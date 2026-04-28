"""Indexe les abstracts PubMed dans ChromaDB avec des embeddings multilinguaux.

Usage:
    uv run python scripts/aria_index_corpus.py

Modèle : intfloat/multilingual-e5-base
    - Les documents sont préfixés "passage: " à l'encodage (convention e5).
    - Les requêtes devront être préfixées "query: " côté retrieval.

Collection ChromaDB : "aria_pubmed"
Upsert idempotent : relancer le script ne réindexe pas les docs existants.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

# Accès aux modules du projet depuis scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from sentence_transformers import SentenceTransformer

from core.config import settings

CORPUS_DIR = Path(__file__).parent.parent / "data" / "corpus"
COLLECTION_NAME = "aria_pubmed"
MODEL_NAME = "intfloat/multilingual-e5-base"
BATCH_SIZE = 64
# Préfixe obligatoire pour l'encodage des passages avec les modèles e5
E5_PASSAGE_PREFIX = "passage: "


def load_corpus() -> list[dict]:
    records: list[dict] = []
    for json_file in sorted(CORPUS_DIR.glob("*.json")):
        data = json.loads(json_file.read_text())
        records.extend(data)
        print(f"  {json_file.name} : {len(data)} documents chargés")
    return records


def batch(lst: list, size: int):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def index_corpus() -> None:
    print(f"Chargement du corpus depuis {CORPUS_DIR}…")
    records = load_corpus()

    if not records:
        print("Aucun document trouvé. Lance d'abord scripts/fetch_pubmed.py.")
        sys.exit(1)

    print(f"\n{len(records)} documents au total.\n")

    print(f"Chargement du modèle {MODEL_NAME}…")
    model = SentenceTransformer(MODEL_NAME)

    chroma_path = Path(settings.CHROMADB_PATH)
    chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    existing_ids: set[str] = set(collection.get(include=[])["ids"])
    print(f"Documents déjà indexés : {len(existing_ids)}\n")

    def chroma_id(r: dict) -> str:
        return f"{r['pathologie']}_{r['pmid']}"

    new_records = [r for r in records if chroma_id(r) not in existing_ids]
    print(f"Nouveaux documents à indexer : {len(new_records)}")

    if not new_records:
        print("Collection déjà à jour — aucun upsert nécessaire.")
    else:
        for i, chunk in enumerate(batch(new_records, BATCH_SIZE)):
            texts = [E5_PASSAGE_PREFIX + r["abstract"] for r in chunk]
            embeddings = model.encode(texts, normalize_embeddings=True).tolist()

            collection.upsert(
                ids=[chroma_id(r) for r in chunk],
                documents=[r["abstract"] for r in chunk],
                embeddings=embeddings,
                metadatas=[
                    {"pmid": r["pmid"], "titre": r["titre"], "pathologie": r["pathologie"]}
                    for r in chunk
                ],
            )
            done = min((i + 1) * BATCH_SIZE, len(new_records))
            print(f"  {done}/{len(new_records)} indexés…")

    # Rapport final par pathologie
    print("\n--- Documents indexés par pathologie ---")
    raw_meta = collection.get(include=["metadatas"])["metadatas"] or []
    counts: dict[str, int] = defaultdict(int)
    for meta in raw_meta:
        slug_val = meta.get("pathologie", "") if meta else ""
        counts[str(slug_val)] += 1
    for slug, count in sorted(counts.items()):
        print(f"  {slug:<40} {count:>4}")
    print(f"  {'TOTAL':<40} {sum(counts.values()):>4}")


if __name__ == "__main__":
    index_corpus()
