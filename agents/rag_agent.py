import core.events as events
from core.state import ARIAState
from services.rag.chromadb_service import ChromaDBService


async def rag_agent(state: ARIAState) -> dict:
    session_id = state["session_id"]
    await events.emit(
        session_id,
        {
            "type": "progress",
            "etape": "rag",
            "pct": 53,
            "message": "Recherche bibliographique PubMed...",
        },
    )

    diagnostic = state["diagnostic"]
    if diagnostic is None:
        return {
            "rag_refs": [],
            "statut": "erreur",
            "erreur": "diagnostic manquant avant rag_agent",
        }

    refs = await ChromaDBService().retrieve(diagnostic.pathologie, n_results=5)
    await events.emit(
        session_id,
        {
            "type": "progress",
            "etape": "rag",
            "pct": 60,
            "message": f"{len(refs)} références trouvées",
        },
    )
    return {
        "rag_refs": refs,
        "statut": "rag",
        "erreur": None,
    }
