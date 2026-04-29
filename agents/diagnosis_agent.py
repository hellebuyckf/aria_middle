import asyncio

from core.state import ARIAState
from models.diagnostic import DiagnosticLLM
from services.llm.prompt_builder import build_diagnostic_prompt

# TODO: basculer sur services.llm.vllm_client en prod
from services.llm.vllm_client_mock import generate_diagnostic


async def diagnosis_agent(state: ARIAState) -> dict:
    metrics = state["metrics"]
    if metrics is None:
        return {
            "diagnostic": None,
            "statut": "erreur",
            "erreur": "diagnosis_agent: metrics est None, impossible de construire le prompt",
        }

    prompt = build_diagnostic_prompt(
        metrics,
        state["pathologie_declaree"],
        age=state.get("age"),
        taille_cm=state.get("taille_cm"),
        poids_kg=state.get("poids_kg"),
        niveau_pratique=state.get("niveau_pratique"),
        km_semaine=state.get("km_semaine"),
        profil_chaussure=state.get("profil_chaussure"),
        strava_charge=state.get("strava_charge"),
        garmin_charge=state.get("garmin_charge"),
    )

    try:
        result: DiagnosticLLM = await generate_diagnostic(prompt)
    except asyncio.TimeoutError:
        return {
            "diagnostic": None,
            "statut": "erreur",
            "erreur": "diagnosis_agent: timeout lors de l'appel au LLM",
        }
    except Exception as exc:
        return {
            "diagnostic": None,
            "statut": "erreur",
            "erreur": f"diagnosis_agent: {exc}",
        }

    return {
        "diagnostic": result,
        "statut": "diagnostic",
        "erreur": None,
    }
