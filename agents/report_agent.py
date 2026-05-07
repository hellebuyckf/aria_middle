import json
from datetime import date

from loguru import logger

import core.events as events
from core.state import ARIAState
from models.metrics import BiomechanicalMetrics
from models.report import ARIAReport
from services.llm import vllm_client as llm


def _build_prompt(state: ARIAState) -> str:
    diagnostic = state["diagnostic"]
    metrics: BiomechanicalMetrics | None = state["metrics"]
    refs = state["rag_refs"]

    lines: list[str] = [
        "Tu es ARIA-ft, assistant clinique spécialisé en biomécanique de la course.",
        "Génère un rapport de rééducation structuré au format JSON.",
        "",
        "## Diagnostic",
        f'- "pathologie": "{diagnostic.pathologie if diagnostic else "inconnue"}"',
        f'- "confiance": "{diagnostic.confiance if diagnostic else "faible"}"',
        f'- "justification": "{diagnostic.justification if diagnostic else ""}"',
        "",
        "## Métriques biomécaniques (valeurs non nulles)",
    ]

    if metrics:
        for field_name, value in metrics.model_dump().items():
            if value is not None and field_name != "vue_posterieure_disponible":
                field_info = BiomechanicalMetrics.model_fields[field_name]
                lines.append(
                    f'- "{field_name}": {value}  # {field_info.description or ""}'
                )

    lines += ["", "## Références PubMed (top-5)"]
    for ref in refs[:5]:
        lines.append(f"- Titre : {ref['titre']}")
        lines.append(f"  Extrait : {ref['extrait']}")

    lines += [
        "",
        "## Consigne",
        "Retourne UNIQUEMENT un objet JSON valide avec exactement ces champs :",
        '- "pathologie": string — pathologie identifiée',
        '- "confiance": string — niveau de confiance (élevé / moyen / faible)',
        '- "justification_diagnostic": string — justification clinique détaillée',
        '- "metriques_anormales": array of strings — liste des métriques hors norme',
        '- "recommandations": array of strings — protocole de rééducation personnalisé',
        '- "references_pubmed": array of strings — titres des références utilisées',
        '- "avertissement": string — invitation au praticien à confirmer le diagnostic',
    ]

    return "\n".join(lines)


async def report_agent(state: ARIAState) -> dict:
    session_id = state["session_id"]
    try:
        await events.emit(
            session_id,
            {
                "type": "progress",
                "etape": "rapport",
                "pct": 62,
                "message": "Construction du prompt clinique...",
            },
        )
        prompt = _build_prompt(state)

        await events.emit(
            session_id,
            {
                "type": "progress",
                "etape": "rapport",
                "pct": 65,
                "message": "Génération du rapport par ARIA-ft...",
            },
        )
        ticker = events.tick(
            session_id, "rapport", 65, 88, 18.0, "Génération du rapport..."
        )
        try:
            raw_json: str = await llm.generate_report(
                prompt=prompt,
                session_id=session_id,
                patient_id=state["patient_id"],
                response_format=ARIAReport.model_json_schema(),
            )
        finally:
            ticker.cancel()

        data = json.loads(raw_json)
        data["session_id"] = session_id
        data["patient_id"] = state["patient_id"]
        data["date_generation"] = date.today().isoformat()
        report = ARIAReport.model_validate(data)
        await events.emit(
            session_id,
            {
                "type": "progress",
                "etape": "rapport",
                "pct": 90,
                "message": "Rapport généré",
            },
        )
        return {"report": report, "statut": "rapport", "erreur": None}
    except Exception as exc:
        logger.exception("[{}] report_agent échec : {}", session_id, exc)
        return {"report": None, "statut": "erreur", "erreur": str(exc)}
