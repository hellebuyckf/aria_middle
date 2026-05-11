from pydantic import BaseModel, Field


_AVERTISSEMENT_DEFAULT = (
    "Ce rapport est généré automatiquement par ARIA. "
    "Il ne constitue pas un diagnostic médical. "
    "Veuillez faire confirmer les conclusions par un professionnel de santé qualifié."
)


class ARIAReport(BaseModel):
    session_id: str
    patient_id: str
    pathologie: str
    confiance: str
    justification_diagnostic: str
    metriques_anormales: list[str] = Field(default_factory=list)
    recommandations: list[str] = Field(default_factory=list)
    references_pubmed: list[str] = Field(default_factory=list)
    avertissement: str = _AVERTISSEMENT_DEFAULT
    date_generation: str
