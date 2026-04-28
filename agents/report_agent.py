from core.state import ARIAState
from models.metrics import BiomechanicalMetrics
from models.report import ARIAReport
from services.llm import vllm_client_mock as llm


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
        "Retourne un JSON valide correspondant au schéma ARIAReport avec :",
        "- les métriques hors norme identifiées (metriques_anormales)",
        "- un protocole de rééducation personnalisé (recommandations)",
        "- les titres des références utilisées (references_pubmed)",
        "- un avertissement invitant le praticien à confirmer le diagnostic (avertissement)",
    ]

    return "\n".join(lines)


async def report_agent(state: ARIAState) -> dict:
    try:
        prompt = _build_prompt(state)
        raw_json: str = await llm.generate_report(
            prompt=prompt,
            session_id=state["session_id"],
            patient_id=state["patient_id"],
            response_format=ARIAReport.model_json_schema(),
        )
        report = ARIAReport.model_validate_json(raw_json)
        return {"report": report, "statut": "rapport", "erreur": None}
    except Exception as exc:
        return {"report": None, "statut": "erreur", "erreur": str(exc)}
