from typing import Literal, TypedDict

from models.diagnostic import DiagnosticLLM
from models.metrics import BiomechanicalMetrics
from models.report import ARIAReport


class PubMedReference(TypedDict):
    """Référence bibliographique issue de ChromaDB / PubMed."""

    pmid: str
    titre: str
    extrait: str


class ProfilChaussure(TypedDict, total=False):
    marque: str
    modele: str
    drop_mm: int
    stabilite: str
    amorti: str
    poids_type: str
    dynamisme: str


class ARIAState(TypedDict):
    """État partagé entre tous les nœuds du graphe LangGraph ARIA."""

    session_id: str
    patient_id: str
    video_path: str
    video_path_posterior: str | None
    pathologie_declaree: str | None
    # Profil patient
    age: int | None
    taille_cm: int | None
    poids_kg: float | None
    km_semaine: int | None
    niveau_pratique: str | None
    profil_chaussure: ProfilChaussure | None
    strava_charge: dict | None
    garmin_charge: dict | None
    # Pipeline
    key_frames: list[str]  # base64 PNG annotées, max 4, générées par video_agent
    metrics: BiomechanicalMetrics | None
    diagnostic: DiagnosticLLM | None
    rag_refs: list[PubMedReference]
    prompt: str | None
    report: ARIAReport | None
    statut: Literal[
        "idle", "video", "diagnostic", "rag", "pret", "llm", "rapport", "erreur"
    ]
    erreur: str | None
