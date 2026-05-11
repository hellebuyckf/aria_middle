import json
from datetime import date

from loguru import logger

import core.events as events
from core.pathologies import metriques_pour
from core.state import ARIAState
from models.metrics import BiomechanicalMetrics
from models.report import ARIAReport
from services.llm import vllm_client as llm

_EXCLUDED_FIELDS = {"vue_posterieure_disponible"}

# Seuils (lo, hi) — None = borne ouverte. Miroir de frame_annotator._THRESHOLDS.
_THRESHOLDS: dict[str, tuple[float | None, float | None]] = {
    "cadence": (170.0, 180.0),
    "angle_attaque_pied": (None, 10.0),
    "flexion_genou_impact": (15.0, 25.0),
    "inclinaison_tronc": (5.0, 10.0),
    "oscillation_verticale": (None, 8.0),
    "ratio_contact_suspension": (0.35, 0.65),
    "pelvic_drop": (None, 5.0),
    "valgus_genou": (None, 8.0),
    "asymetrie_charge": (None, 10.0),
    "oscillation_laterale_hanche": (None, 3.0),
    "pronation_pied": (None, 8.0),
}


# Schema JSON attendu du LLM — sans pathologie/confiance/metriques_anormales
# qui sont injectés déterministiquement après le parsing.
_REPORT_LLM_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "justification_diagnostic": {"type": "string"},
        "recommandations": {"type": "array", "items": {"type": "string"}},
        "references_pubmed": {"type": "array", "items": {"type": "string"}},
        "avertissement": {"type": "string"},
    },
    "required": [
        "justification_diagnostic",
        "recommandations",
        "references_pubmed",
        "avertissement",
    ],
}


def _is_abnormal(field: str, value: float) -> bool:
    lo, hi = _THRESHOLDS.get(field, (None, None))
    return (lo is not None and value < lo) or (hi is not None and value > hi)


def _compute_abnormal_metrics(
    metrics: BiomechanicalMetrics,
    pathologie: str,  # noqa: ARG001
) -> list[str]:
    """Retourne tous les champs hors norme (sagittaux + postérieurs).

    Non filtrés par pathologie : aligne le comportement avec le PDF et permet
    à l'UI d'afficher correctement les points rouges sur toutes les métriques.
    """
    data = metrics.model_dump()
    return [
        field
        for field, (lo, hi) in _THRESHOLDS.items()
        if isinstance(data.get(field), (int, float))
        and _is_abnormal(field, float(data[field]))
    ]


def _relevant_metrics(
    metrics: BiomechanicalMetrics, pathologie: str
) -> dict[str, object]:
    """Retourne uniquement les métriques pertinentes pour la pathologie diagnostiquée."""
    allowed = metriques_pour(pathologie)
    data = metrics.model_dump()
    if allowed is None:
        return {
            k: v for k, v in data.items() if k not in _EXCLUDED_FIELDS and v is not None
        }
    return {k: data[k] for k in allowed if data.get(k) is not None}


def _build_prompt(state: ARIAState) -> str:
    diagnostic = state["diagnostic"]
    metrics: BiomechanicalMetrics | None = state["metrics"]
    refs = state["rag_refs"]
    pathologie = diagnostic.pathologie if diagnostic else ""
    profil_chaussure = state.get("profil_chaussure")

    lines: list[str] = [
        "Tu es ARIA-ft, assistant clinique spécialisé en biomécanique de la course.",
        "Génère un rapport de rééducation structuré au format JSON.",
        "",
        "## Diagnostic",
        f'- "pathologie": "{pathologie or "inconnue"}"',
        f'- "confiance": "{diagnostic.confiance if diagnostic else "faible"}"',
        f'- "justification": "{diagnostic.justification if diagnostic else ""}"',
        "",
        "## Métriques biomécaniques pertinentes pour cette pathologie",
    ]

    if metrics:
        for field_name, value in _relevant_metrics(metrics, pathologie).items():
            field_info = BiomechanicalMetrics.model_fields[field_name]
            lines.append(f'- "{field_name}": {value}  # {field_info.description or ""}')

    if profil_chaussure:
        lines += ["", "## Chaussures actuelles du coureur"]
        if profil_chaussure.get("marque") or profil_chaussure.get("modele"):
            lines.append(
                f"- Modèle : {profil_chaussure.get('marque', '')} {profil_chaussure.get('modele', '')}".strip()
            )
        if "drop_mm" in profil_chaussure:
            lines.append(f"- Drop actuel : {profil_chaussure['drop_mm']} mm")
        if profil_chaussure.get("stabilite"):
            lines.append(f"- Stabilité : {profil_chaussure['stabilite']}")
        if profil_chaussure.get("amorti"):
            lines.append(f"- Amorti : {profil_chaussure['amorti']}")
        if profil_chaussure.get("poids_type"):
            lines.append(f"- Poids : {profil_chaussure['poids_type']}")
        if profil_chaussure.get("dynamisme"):
            lines.append(f"- Dynamisme : {profil_chaussure['dynamisme']}")

    lines += ["", "## Références PubMed (top-5)"]
    for ref in refs[:5]:
        lines.append(f"- Titre : {ref['titre']}")
        lines.append(f"  Extrait : {ref['extrait']}")

    chaussure_consigne = (
        " Inclure obligatoirement une recommandation sur les chaussures : "
        "évaluer si le drop actuel est adapté à la pathologie et aux métriques, "
        "et préciser le drop cible recommandé (en mm) avec justification biomécanique."
        if profil_chaussure and "drop_mm" in profil_chaussure
        else " Si pertinent, inclure une recommandation sur le type de chaussures adapté."
    )

    lines += [
        "",
        "## Consigne",
        "Le diagnostic et la confiance sont FIXÉS ci-dessus — ne les modifie pas.",
        "Retourne UNIQUEMENT un objet JSON valide avec exactement ces champs :",
        '- "justification_diagnostic": string — justification clinique détaillée',
        f'- "recommandations": array of strings — protocole de rééducation personnalisé.{chaussure_consigne}',
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
                response_format=_REPORT_LLM_SCHEMA,
            )
        finally:
            ticker.cancel()

        data = json.loads(raw_json)
        required = {"justification_diagnostic", "recommandations", "references_pubmed"}
        missing = required - data.keys()
        if missing:
            raise ValueError(f"Champs manquants dans la réponse LLM : {missing}")
        data["session_id"] = session_id
        data["patient_id"] = state["patient_id"]
        data["date_generation"] = date.today().isoformat()
        # Pathologie et confiance : toujours celles du diagnosis_agent
        diagnostic = state.get("diagnostic")
        data["pathologie"] = diagnostic.pathologie if diagnostic else "inconnue"
        data["confiance"] = diagnostic.confiance if diagnostic else "faible"
        # Métriques anormales : calculées déterministiquement depuis les seuils
        metrics = state.get("metrics")
        data["metriques_anormales"] = (
            _compute_abnormal_metrics(metrics, data["pathologie"]) if metrics else []
        )
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
