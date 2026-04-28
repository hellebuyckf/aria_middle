import re
from datetime import datetime, timezone

from models.diagnostic import DiagnosticLLM
from models.report import ARIAReport


def _extract_value(prompt: str, key: str) -> float | None:
    # Cherche "key": <nombre> ou "key" : <nombre>
    match = re.search(rf'"{key}"\s*:\s*([0-9]+(?:\.[0-9]+)?)', prompt)
    return float(match.group(1)) if match else None


def _extract_str(prompt: str, key: str) -> str | None:
    match = re.search(rf'"{key}"\s*:\s*"([^"]+)"', prompt)
    return match.group(1) if match else None


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


async def generate_report(
    prompt: str,
    session_id: str,
    patient_id: str,
    response_format: dict | None = None,
) -> str:
    pathologie = _extract_str(prompt, "pathologie") or "Tendinite rotulienne"
    confiance = _extract_str(prompt, "confiance") or "faible"
    justification = _extract_str(prompt, "justification") or "Données insuffisantes."

    inclinaison = _extract_value(prompt, "inclinaison_tronc")
    valgus = _extract_value(prompt, "valgus_genou")
    cadence = _extract_value(prompt, "cadence")

    metriques_anormales: list[str] = []
    if inclinaison is not None and inclinaison > 10:
        metriques_anormales.append(
            f"inclinaison_tronc = {inclinaison}° (norme : 5–10°)"
        )
    if valgus is not None and valgus > 8:
        metriques_anormales.append(f"valgus_genou = {valgus}° (norme : < 8°)")
    if cadence is not None and cadence < 170:
        metriques_anormales.append(f"cadence = {cadence} spm (norme : 170–180 spm)")

    recommandations = [
        "Renforcement des abducteurs de hanche (3×15 reps, 3×/semaine)",
        "Travail de cadence avec métronome à 175 spm",
        "Exercices proprioceptifs en appui unipodal",
        "Étirements du fascia lata et du tenseur du fascia lata",
        "Réduction de l'allure de 10 % pendant 2 semaines",
    ]

    refs: list[str] = []
    for line in prompt.splitlines():
        if line.strip().startswith("- Titre :"):
            refs.append(line.strip().removeprefix("- Titre :").strip())

    report = ARIAReport(
        session_id=session_id,
        patient_id=patient_id,
        pathologie=pathologie,
        confiance=confiance,
        justification_diagnostic=justification,
        metriques_anormales=metriques_anormales,
        recommandations=recommandations,
        references_pubmed=refs,
        avertissement=(
            "Ce rapport est généré par un système d'IA à titre d'aide à la décision. "
            "Il ne remplace pas l'évaluation clinique d'un professionnel de santé qualifié. "
            "Le praticien est invité à confirmer le diagnostic avant toute prise en charge."
        ),
        date_generation=datetime.now(timezone.utc).isoformat(),
    )
    return report.model_dump_json()
