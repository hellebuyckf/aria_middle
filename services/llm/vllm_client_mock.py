import re

from models.diagnostic import DiagnosticLLM


def _extract_value(prompt: str, key: str) -> float | None:
    # Cherche "key": <nombre> ou "key" : <nombre>
    match = re.search(rf'"{key}"\s*:\s*([0-9]+(?:\.[0-9]+)?)', prompt)
    return float(match.group(1)) if match else None


async def generate_diagnostic(prompt: str) -> DiagnosticLLM:
    inclinaison = _extract_value(prompt, "inclinaison_tronc")
    valgus = _extract_value(prompt, "valgus_genou")

    if inclinaison is not None and inclinaison > 20:
        return DiagnosticLLM(
            pathologie="Lombalgie",
            confiance="élevée",
            justification=(
                f"Inclinaison du tronc excessive ({inclinaison}°) détectée sur la vue "
                "sagittale, cohérente avec une surcharge lombaire."
            ),
        )

    if valgus is not None and valgus > 7:
        return DiagnosticLLM(
            pathologie="SBIT",
            confiance="modérée",
            justification=(
                f"Valgus genou ({valgus}°) supérieur au seuil de 7°, "
                "évocateur d'un syndrome de la bandelette ilio-tibiale."
            ),
        )

    return DiagnosticLLM(
        pathologie="Tendinite rotulienne",
        confiance="faible",
        justification=(
            "Aucun signal biomécanique prédominant détecté. "
            "Hypothèse à confirmer par examen clinique."
        ),
    )
