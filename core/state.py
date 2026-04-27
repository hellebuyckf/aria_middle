from typing import Literal, TypedDict

from models.metrics import BiomechanicalMetrics
from models.report import ARIAReport, PubMedReference


class ARIAState(TypedDict):
    """État partagé entre tous les nœuds du graphe LangGraph ARIA."""

    session_id: str
    patient_id: str
    video_path: str
    metrics: BiomechanicalMetrics | None
    rag_refs: list[PubMedReference]
    prompt: str | None
    report: ARIAReport | None
    statut: Literal["idle", "video", "rag", "llm", "rapport", "erreur"]
    erreur: str | None
