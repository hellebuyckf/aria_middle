from pydantic import BaseModel


class ARIAReport(BaseModel):
    session_id: str
    patient_id: str
    pathologie: str
    confiance: str
    justification_diagnostic: str
    metriques_anormales: list[str]
    recommandations: list[str]
    references_pubmed: list[str]
    avertissement: str
    date_generation: str
