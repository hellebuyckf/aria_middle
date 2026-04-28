from typing import Literal, TypedDict

from models.diagnostic import DiagnosticLLM
from models.metrics import BiomechanicalMetrics


class PubMedReference(TypedDict):
    """Référence bibliographique issue de ChromaDB / PubMed."""

    pmid: str
    titre: str
    extrait: str


class ARIAState(TypedDict):
    """État partagé entre tous les nœuds du graphe LangGraph ARIA."""

    session_id: str
    patient_id: str
    video_path: str
    video_path_posterior: str | None
    pathologie_declaree: str | None
    metrics: BiomechanicalMetrics | None
    diagnostic: DiagnosticLLM | None
    rag_refs: list[PubMedReference]
    prompt: str | None
    report: str | None
    statut: Literal["idle", "video", "diagnostic", "rag", "llm", "rapport", "erreur"]
    erreur: str | None
