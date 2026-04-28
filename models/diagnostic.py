from typing import Literal

from pydantic import BaseModel, Field


class DiagnosticLLM(BaseModel):
    """Diagnostic clinique produit par le modèle ARIA-ft."""

    pathologie: str = Field(
        description="Nom de la pathologie ou du pattern biomécanique identifié "
        "(ex. : 'attaque talon excessive', 'syndrome fémoro-patellaire probable')."
    )

    confiance: Literal["élevée", "modérée", "faible"] = Field(
        description="Niveau de confiance du modèle dans le diagnostic. "
        "'élevée' : signal clair et métriques cohérentes. "
        "'modérée' : indices présents mais incomplets. "
        "'faible' : hypothèse à confirmer par examen clinique."
    )

    justification: str = Field(
        description="Explication courte appuyant le diagnostic, "
        "en référence aux métriques biomécaniques observées "
        "(cadence, angle attaque pied, flexion genou, etc.)."
    )
