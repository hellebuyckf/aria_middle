from core.state import ARIAState
from services.rag.chromadb_service import ChromaDBService


async def rag_agent(state: ARIAState) -> dict:
    diagnostic = state["diagnostic"]
    if diagnostic is None:
        return {
            "rag_refs": [],
            "statut": "erreur",
            "erreur": "diagnostic manquant avant rag_agent",
        }

    refs = await ChromaDBService().retrieve(diagnostic.pathologie, n_results=5)
    return {
        "rag_refs": refs,
        "statut": "rag",
        "erreur": None,
    }
